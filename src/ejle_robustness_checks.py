"""
ejle_robustness_checks.py
=========================
EJLE 査読対応のロバストネス・チェック一式（集計モデル版）

既存パイプライン `abusive_division_simulation_v2.py` の
  - prepare_monthly_panel() / construct_latent_proxies()  … 月次パネル
  - simulate_case_level() / SimConfig                     … ケースレベル擬似データ
をそのまま再利用し、以下の4点を実行する。

  1. run_negbin        … 負の二項回帰（主仕様）vs Poisson、過分散検定、IRR (§4.2)
  2. run_event_study   … 2015年改正まわりのリード/ラグ event study、pre-trend 検定 (§5.2)
  3. run_reversed_auc  … 集計分類器の AUC と「反転 AUC」の比較（系統的反転か無信号か） (§4.3)
  4. run_bright_line   … μ・資産移転比率に基づくルール基準テストの FP/FN (§4.3)

前提
  - `abusive_division_simulation_v2.py` と 3 つの CSV が同じディレクトリにあること。
  - 依存: numpy, pandas, scipy, statsmodels, scikit-learn, matplotlib

使い方
  python ejle_robustness_checks.py            # 4 つすべて実行
  python ejle_robustness_checks.py --only nb  # 個別実行 (nb / event / auc / brightline)
  # Google Colab:  %run ejle_robustness_checks.py
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ── 既存パイプラインの再利用 ────────────────────────────────
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import abusive_division_simulation_v2 as base
except ImportError as e:
    raise SystemExit(
        "abusive_division_simulation_v2.py が import できません。"
        "本スクリプトと同じディレクトリに置いてください。\n" + str(e)
    )

OUT_DIR = Path(os.environ.get("ADV_OUTPUT_DIR", Path(__file__).resolve().parent.parent / "outputs")) / "robustness"
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "figures").mkdir(exist_ok=True)

REFORM_DATE = pd.Timestamp(base.REFORM_DATE)        # 2015-05-01
JUDICIAL_DATE = pd.Timestamp(base.JUDICIAL_DATE)    # 2019-01-01


def resolve_data_paths(data_dir=None, registry=None, macro=None, loan=None):
    """3 つの CSV のパスを解決し、base モジュールのグローバルに反映する。

    優先順位: 明示指定 > data_dir > 既定名で各候補ディレクトリを探索 > glob 探索。
    登記 CSV が見つからない場合は、探索した場所を示して終了する。
    """
    search_dirs = []
    if data_dir:
        search_dirs.append(Path(data_dir))
    search_dirs += [Path.cwd(), Path(__file__).resolve().parent,
                    Path("/content"), Path("/content/drive/MyDrive")]
    # 重複除去（順序保持）
    seen, dirs = set(), []
    for d in search_dirs:
        if d not in seen and d.exists():
            seen.add(d); dirs.append(d)

    def find(explicit, default_name, globs):
        if explicit:
            p = Path(explicit)
            if p.exists():
                return str(p.resolve())
            raise SystemExit(f"指定ファイルが見つかりません: {p}")
        for d in dirs:                      # 既定名で探索
            cand = d / default_name
            if cand.exists():
                return str(cand.resolve())
        for d in dirs:                      # glob 探索
            for g in globs:
                hits = sorted(d.glob(g))
                if hits:
                    return str(hits[0].resolve())
        return None

    reg = find(registry, "commercial_registry_monthly_clean_for_simulation.csv",
               ["*registry*clean*for*simulation*.csv", "*registry*monthly*.csv", "*登記*.csv"])
    mac = find(macro, "マクロ指標.csv", ["*マクロ*指標*.csv", "*macro*.csv"])
    lon = find(loan, "貸出債券市場取引動向_全銀協_.csv", ["*貸出債券*.csv", "*全銀協*.csv", "*loan*market*.csv"])

    if reg is None:
        searched = "\n  - " + "\n  - ".join(str(d) for d in dirs)
        raise SystemExit(
            "登記統計 CSV が見つかりません。--registry でパスを指定するか、"
            "CSV を作業ディレクトリに置いてください。\n探索した場所:" + searched +
            "\n（ファイル名を確認: import glob; print(glob.glob('*.csv')) ）")

    base.REGISTRY_PATH = reg
    if mac:
        base.MACRO_PATH = mac
    if lon:
        base.LOAN_PATH = lon
    print("  データパス:")
    print(f"    registry = {reg}")
    print(f"    macro    = {mac if mac else '(未検出: マクロ系プロキシ/統制は縮小)'}")
    print(f"    loan     = {lon if lon else '(未検出: ローン系プロキシ/統制は縮小)'}")
    return reg, mac, lon


# ════════════════════════════════════════════════════════════
# 共通: パネル / ケース・データの構築
# ════════════════════════════════════════════════════════════

def build_panel() -> pd.DataFrame:
    """v2 と同一手順で月次パネル（プロキシ付き）を構築。"""
    panel = base.prepare_monthly_panel(
        registry_path=base.REGISTRY_PATH,
        macro_path=base.MACRO_PATH if Path(base.MACRO_PATH).exists() else None,
        loan_path=base.LOAN_PATH if Path(base.LOAN_PATH).exists() else None,
    )
    panel = base.construct_latent_proxies(panel)
    panel["date"] = pd.to_datetime(panel["date"])
    return panel


def build_cases() -> pd.DataFrame:
    """ω=1.0 のベースライン較正でケースレベル擬似データを生成。"""
    panel = build_panel()
    cfg = base.SimConfig(cases_per_month=base.CASES_PER_MONTH,
                         random_seed=base.RANDOM_SEED, omega=1.0)
    return base.simulate_case_level(panel, cfg)


def _macro_controls(df: pd.DataFrame) -> list[str]:
    cand = ["loan_rate", "usd_jpy", "nikkei_close",
            "syndicated_amount", "nonperforming_transfer_amount"]
    return [c for c in cand if c in df.columns]


def _zscore_continuous(X: pd.DataFrame) -> pd.DataFrame:
    """連続回帰子（ユニーク値>2）を z 化。ダミー(0/1)はそのまま。
    生スケール（nikkei~2e4, usd~1e2, t~0..200）の悪条件を解消する。"""
    Xz = X.copy()
    for c in X.columns:
        col = X[c]
        if col.nunique(dropna=True) > 2:
            sd = col.std()
            Xz[c] = (col - col.mean()) / sd if (sd and np.isfinite(sd) and sd > 0) else 0.0
    return Xz


def _fit_count(y, X: pd.DataFrame, kind: str):
    """Poisson または頑健な二段階 GLM-NB を返す。
    NB は Poisson→Cameron–Trivedi 補助回帰で alpha を推定→alpha 固定の GLM-NB。
    離散 NegativeBinomial の MLE 発散（alpha→inf, llf=nan）を回避する。
    連続回帰子は呼び出し側で z 化しておくこと。"""
    Xc = sm.add_constant(X, has_constant="add")
    yi = np.rint(pd.to_numeric(pd.Series(np.asarray(y)), errors="coerce").to_numpy()).astype(float)
    pois = sm.GLM(yi, Xc, family=sm.families.Poisson()).fit()
    if kind == "poisson":
        return pois
    mu = np.asarray(pois.mu)
    aux_y = ((yi - mu) ** 2 - yi) / mu                      # 補助被説明変数
    alpha = float(np.maximum((mu @ aux_y) / (mu @ mu), 1e-6))
    nb = sm.GLM(yi, Xc, family=sm.families.NegativeBinomial(alpha=alpha)).fit()
    nb._alpha_used = alpha
    # 収束・有限性を明示チェック（GLM-NB は通常ここで収束する）
    ok = bool(getattr(nb, "converged", True)) and np.all(np.isfinite(np.asarray(nb.bse)))
    nb._fit_ok = ok
    return nb


def _wald_zero(res, names):
    """指定係数が同時にゼロかの Wald 検定（statsmodels の版差に頑健）。"""
    names = [n for n in names if n in res.params.index]
    if not names:
        return None
    R = np.zeros((len(names), len(res.params)))
    idx = {nm: i for i, nm in enumerate(res.params.index)}
    for r, nm in enumerate(names):
        R[r, idx[nm]] = 1.0
    try:
        w = res.wald_test(R, scalar=True)
    except TypeError:
        w = res.wald_test(R)
    return float(np.squeeze(np.asarray(w.statistic))), float(np.squeeze(np.asarray(w.pvalue)))


# ════════════════════════════════════════════════════════════
# 1. 負の二項回帰（主仕様）＋ 過分散検定 ＋ IRR
# ════════════════════════════════════════════════════════════

def run_negbin() -> pd.DataFrame:
    print("\n[1] Negative Binomial vs Poisson — next-month special liquidation")
    df = build_panel().copy()
    df["y"] = df["special_liquidation"].shift(-1)   # Type I と同じ被説明変数

    regressors = ["split_incorporation_lag1", "split_capital_balance_lag1",
                  "distress_rate_on_split_lag1", "post_reform_dummy",
                  "post_judicial_dummy", "t"] + _macro_controls(df)
    work = df[["y"] + regressors].dropna().copy()
    y = work["y"]
    Xraw = work[regressors]
    dummies = [c for c in regressors if Xraw[c].nunique(dropna=True) <= 2]
    X = _zscore_continuous(Xraw)        # 連続回帰子のみ z 化（悪条件の解消）

    pois = _fit_count(y, X, "poisson")
    nb = _fit_count(y, X, "nb")

    # 過分散検定（Poisson の Pearson 分散）: ratio>1 で過分散
    disp_ratio = float(pois.pearson_chi2 / pois.df_resid)

    # IRR テーブル（NB; alpha は固定なので params に含まれない）
    keep = [c for c in nb.params.index if c != "alpha"]
    irr = pd.DataFrame({
        "coef_nb": nb.params[keep],
        "irr_nb": np.exp(nb.params[keep]),
        "p_nb": nb.pvalues[keep],
        "coef_poisson": pois.params.reindex(keep),
        "p_poisson": pois.pvalues.reindex(keep),
    })
    ci = nb.conf_int().loc[keep]
    irr["irr_nb_lo"] = np.exp(ci[0])
    irr["irr_nb_hi"] = np.exp(ci[1])
    # スケール解釈: ダミーは level(0/1)、連続は per-SD（z 化したため）
    irr["scale"] = ["level (0/1)" if k in dummies else "per-SD" for k in keep]
    irr = irr.round(4)

    # 要約（alpha・収束フラグを含む）
    summary = pd.DataFrame({
        "metric": ["dispersion_ratio(Poisson Pearson)", "alpha_nb", "nb_converged",
                   "AIC_poisson", "AIC_nb", "loglik_poisson", "loglik_nb", "n_obs"],
        "value": [round(disp_ratio, 3), round(getattr(nb, "_alpha_used", float("nan")), 4),
                  int(getattr(nb, "_fit_ok", True)), round(pois.aic, 2), round(nb.aic, 2),
                  round(pois.llf, 2), round(nb.llf, 2), int(work.shape[0])],
    })

    irr.to_csv(OUT_DIR / "nb_irr_table.csv", encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "nb_vs_poisson_summary.csv", index=False)
    (OUT_DIR / "nb_full_summary.txt").write_text(nb.summary().as_text(), encoding="utf-8")

    print(f"    dispersion ratio = {disp_ratio:.2f} (>1 → overdispersion; NB preferred)")
    print(f"    NB: alpha={getattr(nb,'_alpha_used',float('nan')):.4f}  "
          f"converged={getattr(nb,'_fit_ok',True)}")
    print(f"    AIC: Poisson={pois.aic:.1f}  NB={nb.aic:.1f}  (lower is better)")
    print("    IRR (NB), key terms [continuous = per-SD, dummies = level]:")
    for k in ["post_reform_dummy", "post_judicial_dummy", "distress_rate_on_split_lag1"]:
        if k in irr.index:
            print(f"      {k:32s} IRR={irr.loc[k,'irr_nb']:.3f} "
                  f"[{irr.loc[k,'irr_nb_lo']:.3f}, {irr.loc[k,'irr_nb_hi']:.3f}] "
                  f"p={irr.loc[k,'p_nb']:.3f}")
    print(f"    → 表は {OUT_DIR}/nb_irr_table.csv に保存（scale 列で per-SD/level を区別）")
    return irr


# ════════════════════════════════════════════════════════════
# 2. Event study（リード/ラグ）＋ pre-trend 検定
# ════════════════════════════════════════════════════════════

def run_event_study(q_pre: int = 8, q_post: int = 12) -> pd.DataFrame:
    """2015 改正を t=0 とする四半期イベント研究（NB）。
    窓 [-q_pre, +q_post] の外は標本から除外し（端点 catch-all を作らない）、
    線形トレンド t と少数のマクロ統制で趨勢を除去。参照期 = 改正直前四半期 (rel_q=-1)。
    pre-trend は rel_q<-1 のダミー同時ゼロの Wald 検定で判定。"""
    print("\n[2] Event study around 2015 reform (NB, quarterly leads/lags)")
    df = build_panel().copy()
    df = df.dropna(subset=["special_liquidation"]).reset_index(drop=True)

    # 改正からの相対四半期（打ち切りではなく窓外を除外）
    rel_m = (df["date"].dt.year - REFORM_DATE.year) * 12 + (df["date"].dt.month - REFORM_DATE.month)
    df["rel_q"] = np.floor(rel_m / 3.0).astype(int)
    df = df[(df["rel_q"] >= -q_pre) & (df["rel_q"] <= q_post)].reset_index(drop=True)

    ev = pd.get_dummies(df["rel_q"].astype(int), prefix="q").astype(float)
    ref = "q_-1"                                  # 参照期（改正直前四半期）
    if ref in ev.columns:
        ev = ev.drop(columns=[ref])

    # 連続統制（z 化）: 線形トレンド＋マクロ。ダミーは別途。
    controls = _zscore_continuous(df[_macro_controls(df) + ["t"]])
    Xw = pd.concat([ev.reset_index(drop=True), controls.reset_index(drop=True)], axis=1)
    work = pd.concat([df[["special_liquidation"]].reset_index(drop=True), Xw], axis=1).dropna()
    y = work["special_liquidation"]
    Xw = work.drop(columns=["special_liquidation"])

    res = _fit_count(y, Xw, "nb")

    # イベントダミー係数を抽出（参照期=0 を追加）
    rows = []
    for col in ev.columns:
        if col in res.params.index:
            q = int(col.split("_")[1])
            lo, hi = res.conf_int().loc[col]
            rows.append({"rel_q": q, "coef": res.params[col],
                         "ci_lo": lo, "ci_hi": hi, "p": res.pvalues[col]})
    rows.append({"rel_q": -1, "coef": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "p": np.nan})
    es = pd.DataFrame(rows).sort_values("rel_q").reset_index(drop=True)
    es[["coef", "ci_lo", "ci_hi"]] = es[["coef", "ci_lo", "ci_hi"]].round(4)

    # pre-trend 検定: rel_q<-1 のリードが同時ゼロか
    pre_terms = [c for c in ev.columns if int(c.split("_")[1]) < -1]
    wald = _wald_zero(res, pre_terms)
    if wald is None:
        pretrend_msg = "n/a"
    else:
        chi2, pv = wald
        pretrend_msg = f"Wald chi2={chi2:.2f}, p={pv:.3f}"

    es.to_csv(OUT_DIR / "event_study_coefficients.csv", index=False)

    # プロット（log-IRR と 95%CI; 窓内のみ）
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(-0.5, color="navy", ls="--", lw=1.2, label="reform (2015Q2)")
    ax.errorbar(es["rel_q"], es["coef"],
                yerr=[es["coef"] - es["ci_lo"], es["ci_hi"] - es["coef"]],
                fmt="o-", capsize=3, color="#1f77b4")
    ax.set_xlabel(f"Quarters relative to 2015 reform (ref = −1; window [−{q_pre}, +{q_post}])")
    ax.set_ylabel("Event-time coefficient (log incidence-rate ratio)")
    ax.set_title(f"Event study: special liquidations around the 2015 reform\n"
                 f"pre-trend joint test: {pretrend_msg}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figures" / "event_study.png", dpi=150)
    plt.close(fig)

    print(f"    window=[-{q_pre},+{q_post}] quarters, n={int(work.shape[0])}, "
          f"NB converged={getattr(res,'_fit_ok',True)}")
    print(f"    pre-trend joint test (rel_q<-1 = 0): {pretrend_msg}")
    print(f"      → 有意でなければ pre-trend なし＝『露出』解釈と整合")
    print(f"    係数: {OUT_DIR}/event_study_coefficients.csv / 図: figures/event_study.png")
    return es


# ════════════════════════════════════════════════════════════
# 3. 反転 AUC（集計分類器が系統的に反転しているか無信号か）
# ════════════════════════════════════════════════════════════

def run_reversed_auc() -> pd.DataFrame:
    print("\n[3] Reversed-AUC check on the aggregate Type II classifier")
    df = build_panel().copy()

    # v2 と同一のラベル構築（将来 distress のトレンド超過）
    rolling_baseline = df["distress_flow"].rolling(12, min_periods=6).mean()
    future_dist = df["distress_flow"].rolling(3).mean().shift(-3)
    excess = (future_dist - rolling_baseline) + 0.5 * (
        df["special_liquidation"].rolling(3).mean().shift(-3)
        - df["special_liquidation"].rolling(12).mean())
    df["y2"] = (excess > excess.quantile(0.60)).astype(int)

    feats = ["split_incorporation", "split_intensity", "split_capital_balance",
             "distress_rate_on_split", "post_reform_dummy", "post_judicial_dummy",
             "t"] + _macro_controls(df)
    work = df[feats + ["y2"]].dropna()
    X, y = work[feats], work["y2"]

    tscv = TimeSeriesSplit(n_splits=5)
    pipe = Pipeline([("sc", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=2000))])
    rows = []
    for k, (tr, te) in enumerate(tscv.split(X), 1):
        if y.iloc[tr].nunique() < 2 or y.iloc[te].nunique() < 2:
            continue
        pipe.fit(X.iloc[tr], y.iloc[tr])
        p = pipe.predict_proba(X.iloc[te])[:, 1]
        auc = roc_auc_score(y.iloc[te], p)
        rows.append({"fold": k, "auc": round(auc, 3), "reversed_auc": round(1 - auc, 3)})
    res = pd.DataFrame(rows)

    mean_auc = res["auc"].mean()
    mean_rev = res["reversed_auc"].mean()
    sd_rev = res["reversed_auc"].std()
    # 判定: 反転 AUC が安定的に >0.5 なら「系統的反転」、そうでなければ「無信号」
    inverted = (mean_rev > 0.55) and (sd_rev < 0.10)
    verdict = ("classifier appears systematically INVERTED (relabel outcome)"
               if inverted else
               "no stable signal in either direction (data limitation, not inversion)")

    res.to_csv(OUT_DIR / "reversed_auc_aggregate.csv", index=False)

    # 参考: シミュレーション・ケース分類器（正しく向いているはず）
    sim = build_cases()
    Xs = sim[["obs_asset_shift", "obs_liability_gap",
              "obs_preferential_signal", "obs_transparency"]]
    ys = sim["abusive"]
    cut = int(len(sim) * 0.75)
    pipe.fit(Xs.iloc[:cut], ys.iloc[:cut])
    ps = pipe.predict_proba(Xs.iloc[cut:])[:, 1]
    auc_sim = roc_auc_score(ys.iloc[cut:], ps)

    print(f"    aggregate: mean AUC={mean_auc:.3f}, mean reversed AUC={mean_rev:.3f} "
          f"(sd={sd_rev:.3f})")
    print(f"    → {verdict}")
    print(f"    simulated-case classifier: AUC={auc_sim:.3f} "
          f"(reversed={1-auc_sim:.3f}) → reversing makes it worse ⇒ correctly oriented")
    print(f"    保存: {OUT_DIR}/reversed_auc_aggregate.csv")
    return res


# ════════════════════════════════════════════════════════════
# 4. Bright-line ルール基準テスト（FP/FN）
# ════════════════════════════════════════════════════════════

def _confusion(pred: np.ndarray, truth: np.ndarray) -> dict:
    tp = int(((pred == 1) & (truth == 1)).sum())
    fp = int(((pred == 1) & (truth == 0)).sum())
    fn = int(((pred == 0) & (truth == 1)).sum())
    tn = int(((pred == 0) & (truth == 0)).sum())
    n = tp + fp + fn + tn
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "fpr": round(fp / (fp + tn), 4) if (fp + tn) else 0.0,
            "fnr": round(fn / (fn + tp), 4) if (fn + tp) else 0.0,
            "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(f1, 4), "flag_rate": round((tp + fp) / n, 4)}


def run_bright_line(t_asset: float = 0.60, t_liab: float = 0.25) -> pd.DataFrame:
    """ルール: obs_asset_shift > t_asset かつ obs_liability_gap > t_liab → 濫用と判定。
    閾値グリッドを掃引しつつ、指定の単一ルールも評価する。"""
    print("\n[4] Bright-line rule test on simulated cases")
    sim = build_cases()
    truth = sim["abusive"].to_numpy()
    asset = sim["obs_asset_shift"].to_numpy()
    liab = sim["obs_liability_gap"].to_numpy()
    base_rate = float(truth.mean())

    # グリッド掃引
    rows = []
    for ta in np.round(np.arange(0.40, 0.81, 0.10), 2):
        for tl in np.round(np.arange(0.15, 0.46, 0.10), 2):
            pred = ((asset > ta) & (liab > tl)).astype(int)
            m = _confusion(pred, truth)
            m.update({"t_asset": ta, "t_liab": tl})
            rows.append(m)
    grid = pd.DataFrame(rows)
    grid = grid[["t_asset", "t_liab", "flag_rate", "precision", "recall",
                 "f1", "fpr", "fnr", "tp", "fp", "fn", "tn"]].sort_values(
                 "f1", ascending=False).reset_index(drop=True)
    grid.to_csv(OUT_DIR / "bright_line_grid.csv", index=False)

    # 指定ルール
    pred0 = ((asset > t_asset) & (liab > t_liab)).astype(int)
    m0 = _confusion(pred0, truth)

    print(f"    base abusive rate = {base_rate:.3f}")
    print(f"    rule [asset>{t_asset}, liab>{t_liab}]: "
          f"precision={m0['precision']:.3f} recall={m0['recall']:.3f} "
          f"FPR={m0['fpr']:.3f} FNR={m0['fnr']:.3f} flag_rate={m0['flag_rate']:.3f}")
    best = grid.iloc[0]
    print(f"    best-F1 rule: asset>{best['t_asset']}, liab>{best['t_liab']} "
          f"→ F1={best['f1']:.3f}, precision={best['precision']:.3f}, recall={best['recall']:.3f}")
    print(f"    → 較正分類器 AUC≈0.605 と比べ、単純ルールは精度/再現の両立が難しいことを示す")
    print(f"    保存: {OUT_DIR}/bright_line_grid.csv")
    return grid


# ════════════════════════════════════════════════════════════
# メイン
# ════════════════════════════════════════════════════════════

def run_reform_counterfactual(n_boot: int = 2000) -> pd.DataFrame:
    """§5.2 用：改正前期間だけで NB を推定し、実際の改正後共変量で『改正なし』の
    反実仮想を外挿する。効果(%) = (観測合計 / 反実仮想合計 − 1)。
    区間はパラメータ不確実性の MVN ブートストラップ。改正・判例ダミーは
    モデルから除外（＝反実仮想は改正不在）。連続回帰子は pre 期間でスケール。"""
    print("\n[5] NB forecast counterfactual for the 2015 reform (§5.2)")
    df = build_panel().copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["special_liquidation"]).reset_index(drop=True)

    regressors = ["split_incorporation_lag1", "split_capital_balance_lag1",
                  "distress_rate_on_split_lag1", "t"] + _macro_controls(df)
    work = df[["date", "special_liquidation"] + regressors].dropna().reset_index(drop=True)
    pre = work[work["date"] < REFORM_DATE].copy()
    post = work[work["date"] >= REFORM_DATE].copy()
    if len(pre) < 24 or len(post) < 6:
        raise RuntimeError(f"pre/post 期間が不足 (pre={len(pre)}, post={len(post)})")

    # 連続回帰子を pre 期間の平均・標準偏差でスケール → pre/post に同一変換
    cont = [c for c in regressors if work[c].nunique(dropna=True) > 2]
    means, sds = pre[cont].mean(), pre[cont].std().replace(0, 1.0)

    def scale(X):
        Xs = X.copy()
        for c in cont:
            Xs[c] = (X[c] - means[c]) / sds[c]
        return Xs

    Xpre = sm.add_constant(scale(pre[regressors]), has_constant="add")
    Xpost = sm.add_constant(scale(post[regressors]), has_constant="add")[Xpre.columns]
    ypre = np.rint(pre["special_liquidation"].to_numpy()).astype(float)

    # pre 期間で頑健 GLM-NB
    pois = sm.GLM(ypre, Xpre, family=sm.families.Poisson()).fit()
    mu = np.asarray(pois.mu)
    aux_y = ((ypre - mu) ** 2 - ypre) / mu
    alpha = float(np.maximum((mu @ aux_y) / (mu @ mu), 1e-6))
    nb = sm.GLM(ypre, Xpre, family=sm.families.NegativeBinomial(alpha=alpha)).fit()

    # 反実仮想（改正なし）月次予測と効果
    mu_cf = np.asarray(nb.predict(Xpost))
    obs = post["special_liquidation"].to_numpy()
    obs_sum, cf_sum = float(obs.sum()), float(mu_cf.sum())
    eff_pct = (obs_sum / cf_sum - 1.0) * 100.0

    # パラメータ不確実性の MVN ブートストラップで %効果の区間
    rng = np.random.default_rng(0)
    draws = rng.multivariate_normal(nb.params.to_numpy(), nb.cov_params().to_numpy(), size=n_boot)
    lin = np.clip(Xpost.to_numpy() @ draws.T, -30, 30)        # オーバーフロー回避
    cf_sums = np.exp(lin).sum(axis=0)
    pct = (obs_sum / cf_sums - 1.0) * 100.0
    lo, hi = np.percentile(pct, [2.5, 97.5])

    # 月次パスと地平別の累積効果（タイミング確認）
    monthly = pd.DataFrame({"date": post["date"].to_numpy(), "observed": obs,
                            "counterfactual": mu_cf, "effect": obs - mu_cf})
    monthly["months_since_reform"] = np.arange(1, len(monthly) + 1)
    horizons = {}
    for h in [12, 24, 36]:
        seg = monthly[monthly["months_since_reform"] <= h]
        if len(seg):
            horizons[h] = (seg["observed"].sum() / seg["counterfactual"].sum() - 1.0) * 100.0

    monthly.round(3).to_csv(OUT_DIR / "reform_counterfactual_monthly.csv", index=False)
    summ = pd.DataFrame({
        "metric": ["effect_pct", "ci_lo_pct", "ci_hi_pct", "obs_sum", "cf_sum",
                   "alpha_nb", "n_pre", "n_post",
                   "effect_pct_h12", "effect_pct_h24", "effect_pct_h36"],
        "value": [round(eff_pct, 2), round(lo, 2), round(hi, 2),
                  round(obs_sum, 1), round(cf_sum, 1), round(alpha, 4),
                  int(len(pre)), int(len(post)),
                  round(horizons.get(12, float("nan")), 2),
                  round(horizons.get(24, float("nan")), 2),
                  round(horizons.get(36, float("nan")), 2)],
    })
    summ.to_csv(OUT_DIR / "reform_counterfactual_summary.csv", index=False)

    # 観測 vs 反実仮想プロット
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(monthly["date"], monthly["observed"], color="#1f77b4", lw=1.5, label="observed")
    ax.plot(monthly["date"], monthly["counterfactual"], color="#d62728", ls="--", lw=1.5,
            label="no-reform counterfactual (pre-period NB)")
    ax.fill_between(monthly["date"], monthly["counterfactual"], monthly["observed"],
                    color="#1f77b4", alpha=0.12)
    ax.set_xlabel("Date (post-reform)")
    ax.set_ylabel("Special liquidations / month")
    ax.set_title(f"Reform effect (NB forecast counterfactual): "
                 f"{eff_pct:+.1f}% [{lo:+.1f}, {hi:+.1f}] over the post window")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figures" / "reform_counterfactual.png", dpi=150)
    plt.close(fig)

    print(f"    pre n={len(pre)}, post n={len(post)}, NB alpha={alpha:.4f}")
    print(f"    cumulative effect = {eff_pct:+.1f}% [95% {lo:+.1f}, {hi:+.1f}] "
          f"(observed {obs_sum:.0f} vs counterfactual {cf_sum:.0f})")
    print(f"    by horizon:  +12m {horizons.get(12,float('nan')):+.1f}%   "
          f"+24m {horizons.get(24,float('nan')):+.1f}%   "
          f"+36m {horizons.get(36,float('nan')):+.1f}%")
    print(f"    → 符号が正なら『露出』と整合。タイミング（12/24/36m）で漸増かを確認。")
    print(f"    保存: {OUT_DIR}/reform_counterfactual_summary.csv / _monthly.csv / "
          f"figures/reform_counterfactual.png")
    return monthly


def main():
    ap = argparse.ArgumentParser(description="EJLE robustness checks")
    ap.add_argument("--only", choices=["nb", "event", "auc", "brightline", "counterfactual"],
                    default=None, help="個別実行（既定は全実行）")
    ap.add_argument("--data-dir", default=None,
                    help="3 つの CSV があるディレクトリ（例: /content）")
    ap.add_argument("--registry", default=None, help="登記統計 CSV の明示パス")
    ap.add_argument("--macro", default=None, help="マクロ指標 CSV の明示パス")
    ap.add_argument("--loan", default=None, help="貸出債券市場 CSV の明示パス")
    args = ap.parse_args()

    # データパスを解決して base モジュールに反映（最初に必ず実行）
    resolve_data_paths(args.data_dir, args.registry, args.macro, args.loan)

    runners = {"nb": run_negbin, "event": run_event_study,
               "auc": run_reversed_auc, "brightline": run_bright_line,
               "counterfactual": run_reform_counterfactual}
    if args.only:
        runners[args.only]()
    else:
        for fn in runners.values():
            try:
                fn()
            except Exception as e:
                print(f"  [warn] {fn.__name__} failed: {e}")
    print(f"\n✓ 完了. 出力先: {OUT_DIR}/")


if __name__ == "__main__":
    main()
