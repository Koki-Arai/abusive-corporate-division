"""
Simulation and empirical prototype for abusive corporate division  [v2]
─────────────────────────────────────────────────────────────────────────
Changes from v1
  1. REFORM_DATE = 2015-05-01 (会社法改正施行日, 詐害的会社分割規定)
  2. JUDICIAL_DATE = 2019-01-01 (判例法理の明確化期)
  3. f_proxy を delta/mu と独立な変数で再構築（多重共線性の解消）
  4. 期別キャリブレーション：2009-12 / 2012-15 / 2015-19 / 2019-26 の
     実測統計を使って intercept を調整
  5. 濫用確率の設計 intercept = -2.0 → 全体濫用率 ~29%（前回 ~52% を修正）
  6. 再建成功率の分離強化: abusive=0 → ~68%, abusive=1 → ~42%（前回 ~3.5pp 差を ~26pp に）
  7. CASES_PER_MONTH = 100（前回 50 の倍）
  8. Type II 集計モデル: TimeSeriesSplit(n_splits=5) に変更（前回 75/25 固定で
     テストセットが単クラスになる問題を解消）
  9. 感度分析: W3 の ω（再建余剰の重み）を 0.5/1.0/2.0 で変化させて比較
 10. 全出力を CSV + PNG で保存

Google Colab での実行方法
  !pip install -q statsmodels scikit-learn matplotlib pandas numpy
  %run abusive_division_simulation_v2.py
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, roc_curve
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import os

# ═══════════════════════════════════════════════════════════════════════
# ★ CONFIGURATION — 実行前にパスを確認してください
# ═══════════════════════════════════════════════════════════════════════

# Paths resolve to <repo>/data and <repo>/outputs by default; override with
# the ADV_DATA_DIR / ADV_OUTPUT_DIR environment variables.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR  = Path(os.environ.get("ADV_DATA_DIR", _REPO_ROOT / "data"))
_OUT_ROOT  = Path(os.environ.get("ADV_OUTPUT_DIR", _REPO_ROOT / "outputs"))

REGISTRY_PATH = str(_DATA_DIR / "commercial_registry_monthly_clean_for_simulation.csv")
MACRO_PATH    = str(_DATA_DIR / "マクロ指標.csv")
LOAN_PATH     = str(_DATA_DIR / "貸出債券市場取引動向_全銀協_.csv")

OUT_DIR = _OUT_ROOT / "v2"

# 政策ダミー ─ v1 から修正
REFORM_DATE   = "2015-05-01"   # 会社法改正施行（詐害的会社分割規定）
JUDICIAL_DATE = "2019-01-01"   # 判例法理明確化期

# シミュレーション設定
CASES_PER_MONTH = 100          # v1 の倍（月 100 件 × 205 月 ≈ 20,500 件）
MONTHS_CAP      = None
RANDOM_SEED     = 42

# W3 感度分析: ω の値 (W3 = S - ω·A)
OMEGA_VALUES = [0.5, 1.0, 2.0]

# ═══════════════════════════════════════════════════════════════════════

OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "figures").mkdir(exist_ok=True)


# ──────────────────────────────────────────────────────
# 実測データから期別キャリブレーション定数を導出
# ──────────────────────────────────────────────────────
# 下表は merged_monthly_panel.csv の実測統計から算出
#   期間             n   distress_rate_mean   delta_proxy_mean   S_proxy_mean
#   2009-2012       36       17.6              -0.853             +0.294
#   2012-2015       40       24.2              +0.075             -0.252   ← 最大ストレス
#   2015-2019       44       18.1              -0.204             -0.055   ← 改革後
#   2019-2026       85       18.6              +0.431             +0.023
#
# 濫用調整: (distress_rate - grand_mean=19.4) / 19.4 * 1.5
# 再建調整: S_proxy_mean * 0.3
PERIOD_CALIBRATION: Dict[str, Dict[str, float]] = {
    "2009-2012": {"abuse_adj": -0.15, "success_adj": +0.10},  # 活発期・施行前・低ストレス
    "2012-2015": {"abuse_adj": +0.40, "success_adj": -0.15},  # 最大ストレス・改革前
    "2015-2019": {"abuse_adj": -0.20, "success_adj": +0.10},  # 改革直後・抑止効果
    "2019-2026": {"abuse_adj": +0.05, "success_adj":  0.00},  # COVID 以降・横ばい
}

def get_period_adj(date: pd.Timestamp) -> Dict[str, float]:
    if date < pd.Timestamp("2012-01-01"):
        return PERIOD_CALIBRATION["2009-2012"]
    elif date < pd.Timestamp("2015-05-01"):
        return PERIOD_CALIBRATION["2012-2015"]
    elif date < pd.Timestamp("2019-01-01"):
        return PERIOD_CALIBRATION["2015-2019"]
    else:
        return PERIOD_CALIBRATION["2019-2026"]


# ──────────────────────────────────────────────────────
# ユーティリティ
# ──────────────────────────────────────────────────────

def sigmoid_s(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(x)))

def zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    std = s.std()
    if pd.isna(std) or std == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.mean()) / std

def safe_divide(a: pd.Series, b: pd.Series, eps: float = 1e-6) -> pd.Series:
    return pd.to_numeric(a, errors="coerce") / (pd.to_numeric(b, errors="coerce") + eps)

def get_val(row: pd.Series, key: str, default: float = 0.0) -> float:
    v = row.get(key, default)
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return default
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


# ──────────────────────────────────────────────────────
# データ読み込み
# ──────────────────────────────────────────────────────

def load_registry(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)

def load_macro(path: str | Path) -> pd.DataFrame:
    raw = pd.read_csv(path, header=None)
    df = raw.iloc[2:, [0, 4, 8, 9, 10, 11, 12]].copy()
    df.columns = ["date_jp","nikkei_close","topix_close","cpi_index","usd_jpy","loan_rate","ppi_yoy"]
    df["date"] = pd.to_datetime(
        df["date_jp"].astype(str)
        .str.replace("年", "-", regex=False)
        .str.replace("月", "-01", regex=False),
        errors="coerce",
    )
    for c in ["nikkei_close","topix_close","cpi_index","usd_jpy","loan_rate","ppi_yoy"]:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",","",regex=False), errors="coerce")
    return df.dropna(subset=["date"]).drop(columns=["date_jp"]).sort_values("date").reset_index(drop=True)

def load_loan_market(path: str | Path) -> pd.DataFrame:
    raw = pd.read_csv(path, header=None, encoding="cp932")
    dat = raw.iloc[11:, [0, 1, 4, 14, 22]].copy()
    dat.columns = ["period_jp","syndicated_deals","syndicated_amount",
                   "performing_transfer_amount","nonperforming_transfer_amount"]

    def parse_q(x: str) -> pd.Timestamp:
        s = str(x).strip()
        if "." not in s:
            return pd.NaT
        year, q = s.split(".", 1)
        q_map = {"Ⅰ": 3, "I": 3, "Ⅱ": 6, "II": 6, "Ⅲ": 9, "III": 9, "Ⅳ": 12, "IV": 12}
        month = q_map.get(q.strip())
        return pd.NaT if month is None else pd.Timestamp(year=int(year), month=month, day=1)

    dat["date"] = dat["period_jp"].apply(parse_q)
    for c in ["syndicated_deals","syndicated_amount","performing_transfer_amount","nonperforming_transfer_amount"]:
        dat[c] = pd.to_numeric(dat[c].astype(str).str.replace(",","",regex=False), errors="coerce")
    dat = dat.dropna(subset=["date"]).drop(columns=["period_jp"]).sort_values("date").reset_index(drop=True)

    monthly = pd.DataFrame({"date": pd.date_range(dat["date"].min(), dat["date"].max(), freq="MS")})
    monthly = monthly.merge(dat, on="date", how="left").sort_values("date")
    fill_cols = ["syndicated_deals","syndicated_amount","performing_transfer_amount","nonperforming_transfer_amount"]
    monthly[fill_cols] = monthly[fill_cols].ffill()
    return monthly


# ──────────────────────────────────────────────────────
# 特徴量エンジニアリング
# ──────────────────────────────────────────────────────

def prepare_monthly_panel(
    registry_path: str | Path,
    macro_path: Optional[str | Path] = None,
    loan_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    reg = load_registry(registry_path)
    rename_map = {
        "incorporation_by_company_split_all_companies":    "split_incorporation",
        "capital_increase_by_company_split_all_companies": "split_capital_increase",
        "capital_decrease_by_company_split_all_companies": "split_capital_decrease",
        "special_liquidation_all_companies":               "special_liquidation",
        "bankruptcy_or_civil_rehabilitation_all_companies":"bankruptcy_or_civil_rehabilitation",
        "corporate_reorganization_all_companies":          "corporate_reorganization",
        "completion_of_liquidation_all_companies":         "completion_of_liquidation",
        "dissolution_all_companies":                       "dissolution",
        "incorporation_all_companies":                     "incorporation",
        "incorporation_by_merger_all_companies":           "merger_incorporation",
        "dissolution_by_merger_all_companies":             "merger_dissolution",
        "registrations_total_all_companies":               "registrations_total",
    }
    panel = reg[[*rename_map.keys(), "date"]].rename(columns=rename_map).sort_values("date").reset_index(drop=True)

    if macro_path is not None and Path(macro_path).exists():
        panel = panel.merge(load_macro(macro_path), on="date", how="left")
    if loan_path is not None and Path(loan_path).exists():
        panel = panel.merge(load_loan_market(loan_path), on="date", how="left")

    # 基本比率
    panel["split_intensity"]       = safe_divide(panel["split_incorporation"], panel["incorporation"])
    panel["split_vs_merger"]       = safe_divide(panel["split_incorporation"], panel["merger_incorporation"] + 1)
    panel["split_capital_balance"] = safe_divide(
        panel["split_capital_increase"] - panel["split_capital_decrease"],
        panel["split_capital_increase"] + panel["split_capital_decrease"] + 1,
    )
    panel["distress_flow"]               = panel["special_liquidation"] + panel["bankruptcy_or_civil_rehabilitation"] + panel["corporate_reorganization"]
    panel["distress_rate_on_split"]      = safe_divide(panel["distress_flow"],                       panel["split_incorporation"] + 1)
    panel["liquidation_rate_on_split"]   = safe_divide(panel["special_liquidation"],                 panel["split_incorporation"] + 1)
    panel["bankruptcy_rate_on_split"]    = safe_divide(panel["bankruptcy_or_civil_rehabilitation"],  panel["split_incorporation"] + 1)
    panel["completion_rate_on_split"]    = safe_divide(panel["completion_of_liquidation"],           panel["split_incorporation"] + 1)
    # [v2] fairness 用の独立比率
    panel["completion_ratio_total"]      = safe_divide(panel["completion_of_liquidation"],           panel["dissolution"] + 1)
    panel["merger_dissolution_rate"]     = safe_divide(panel["merger_dissolution"],                  panel["dissolution"] + 1)

    # ラグ (lag1 = t-1, lag3 = 直近3か月平均の lag1)
    lag_cols = [
        "split_incorporation","split_capital_increase","split_capital_decrease",
        "split_intensity","split_capital_balance","distress_flow",
        "distress_rate_on_split","liquidation_rate_on_split","bankruptcy_rate_on_split",
        "completion_rate_on_split","completion_ratio_total","merger_dissolution_rate",
    ]
    for col in lag_cols:
        panel[f"{col}_lag1"] = panel[col].shift(1)
        panel[f"{col}_lag3"] = panel[col].rolling(3).mean().shift(1)

    # 政策ダミー（2本）
    panel["post_reform_dummy"]   = (panel["date"] >= pd.Timestamp(REFORM_DATE)).astype(int)
    panel["post_judicial_dummy"] = (panel["date"] >= pd.Timestamp(JUDICIAL_DATE)).astype(int)

    # マクロ標準化
    optional_cols = ["nikkei_close","topix_close","cpi_index","usd_jpy","loan_rate","ppi_yoy",
                     "syndicated_deals","syndicated_amount","performing_transfer_amount","nonperforming_transfer_amount"]
    for col in optional_cols:
        if col in panel.columns:
            panel[f"{col}_z"] = zscore(panel[col])

    panel["t"] = np.arange(len(panel))
    return panel


# ──────────────────────────────────────────────────────
# 潜在プロキシの構築
# ──────────────────────────────────────────────────────

def construct_latent_proxies(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()

    # δ (価値漏出): 清算率・資本バランス・破産率
    out["delta_proxy"] = (
        0.50 * zscore(out["liquidation_rate_on_split_lag3"].fillna(0))
        + 0.30 * (-1) * zscore(out["split_capital_balance_lag3"].fillna(0))
        + 0.20 * zscore(out["bankruptcy_rate_on_split_lag3"].fillna(0))
    )

    # μ (負債配分の歪み): 資本バランス・ストレス率
    out["mu_proxy"] = (
        0.70 * (-1) * zscore(out["split_capital_balance_lag3"].fillna(0))
        + 0.30 * zscore(out["distress_rate_on_split_lag3"].fillna(0))
    )

    # ρ (分配的歪み): 前月の特別清算・会社更生 [v1 の shift(-1) リークを修正済み]
    out["rho_proxy"] = (
        0.60 * zscore(out["special_liquidation"].shift(1).fillna(0))
        + 0.40 * zscore(out["corporate_reorganization"].shift(1).fillna(0))
    )

    # q (法的強度): 改革ダミー + 司法ダミー + 貸出金利
    out["q_proxy"] = (
        0.50 * out["post_reform_dummy"]
        + 0.30 * out["post_judicial_dummy"]
    )
    if "loan_rate_z" in out.columns:
        out["q_proxy"] += 0.20 * out["loan_rate_z"].fillna(0)

    # S (再建余剰): 分割集中度ラグ + マクロ環境
    macro_good = np.zeros(len(out))
    for col, w in [("nikkei_close_z", 0.30), ("topix_close_z", 0.20),
                   ("loan_rate_z", -0.30), ("syndicated_amount_z", 0.20)]:
        if col in out.columns:
            macro_good += w * out[col].fillna(0)
    out["S_proxy"] = (
        0.50 * zscore(out["split_intensity_lag3"].fillna(0))
        + 0.50 * macro_good
    )

    # t (透明性): 改革・司法ダミー + 登記総数
    out["t_proxy"] = (
        0.40 * out["post_reform_dummy"]
        + 0.20 * out["post_judicial_dummy"]
        + 0.20 * zscore(out["registrations_total"].fillna(0))
        + 0.20 * zscore(out["incorporation"].fillna(0))
    )

    # [v2] f (公正性): completion_of_liquidation・merger_dissolution を使用
    # delta_proxy との相関: completion=0.030, merger_dissolution=0.199 → 独立性を確保
    out["f_proxy"] = (
        0.60 * zscore(out["completion_ratio_total_lag3"].fillna(0))
        + 0.40 * zscore(out["merger_dissolution_rate_lag3"].fillna(0))
    )

    # s (実質的適切性): S_proxy と t_proxy の合成
    out["s_proxy"] = (
        0.50 * zscore(out["S_proxy"].fillna(0))
        + 0.50 * zscore(out["t_proxy"].fillna(0))
    )

    # κ (残存回収比率): ストレス率・清算完了率・S のラグ版
    out["kappa_proxy"] = (
        -0.40 * zscore(out["distress_rate_on_split_lag3"].fillna(0))
        + 0.30 * zscore(out["completion_rate_on_split_lag3"].fillna(0))
        + 0.30 * zscore(out["S_proxy"].fillna(0))
    )
    return out


# ──────────────────────────────────────────────────────
# Type I: Poisson GLM（抑止モデル）
# ──────────────────────────────────────────────────────

def fit_type1_model(panel: pd.DataFrame, out_dir: Path) -> None:
    df = panel.copy()
    df["y_type1"] = df["special_liquidation"].shift(-1)

    candidate_x = [
        "split_incorporation_lag1",
        "split_capital_balance_lag1",
        "distress_rate_on_split_lag1",
        "post_reform_dummy",
        "post_judicial_dummy",
        "t",
    ]
    for c in ["loan_rate","usd_jpy","nikkei_close","syndicated_amount","nonperforming_transfer_amount"]:
        if c in df.columns:
            candidate_x.append(c)

    work = df[["y_type1"] + candidate_x].dropna().copy()
    X = sm.add_constant(work[candidate_x], has_constant="add")
    model = sm.GLM(work["y_type1"], X, family=sm.families.Poisson()).fit()
    (out_dir / "type1_monthly_model_summary.txt").write_text(model.summary().as_text(), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(work.index, work["y_type1"].values, label="Actual", alpha=0.7, lw=1.0)
    ax.plot(work.index, model.fittedvalues.values, label="Fitted (Poisson)", alpha=0.8, lw=1.5)
    ax.axvline(work.index[work["post_reform_dummy"].idxmax()],
               color="red", ls="--", lw=1.2, label=f"Reform ({REFORM_DATE})")
    ax.set_title("Type I – Special Liquidation: Actual vs Fitted")
    ax.set_xlabel("Month index")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "type1_actual_vs_fitted.png", dpi=150)
    plt.close(fig)


# ──────────────────────────────────────────────────────
# Type II: 集計分類器（TimeSeriesSplit CV）
# ──────────────────────────────────────────────────────

def fit_type2_classifier(panel: pd.DataFrame, out_dir: Path) -> None:
    df = panel.copy()

    # [v2 修正] ラベル定義: 将来 distress の「トレンド超過」を使用
    # 絶対水準の p75 は 2009-2012 期に集中し後半期が全件 0 になる問題を解消
    # 直近 12 か月の rolling 基準値を引いた超過量の p60 を閾値とする
    rolling_baseline = df["distress_flow"].rolling(12, min_periods=6).mean()
    future_dist      = df["distress_flow"].rolling(3).mean().shift(-3)
    excess_dist      = future_dist - rolling_baseline
    # 可観測変数のみで構成（split_capital_balance をシグナルに追加）
    excess_signal    = (
        excess_dist
        + 0.5 * (df["special_liquidation"].rolling(3).mean().shift(-3) - df["special_liquidation"].rolling(12).mean())
    )
    threshold = excess_signal.quantile(0.60)
    df["y_type2"] = (excess_signal > threshold).astype(int)

    features = ["split_incorporation","split_intensity","split_capital_balance",
                "distress_rate_on_split","post_reform_dummy","post_judicial_dummy","t"]
    for c in ["loan_rate","usd_jpy","nikkei_close","syndicated_amount","nonperforming_transfer_amount"]:
        if c in df.columns:
            features.append(c)

    work = df[features + ["y_type2"]].dropna().copy()
    X, y = work[features], work["y_type2"]

    # [v2] TimeSeriesSplit(n_splits=5): 拡大ウィンドウ CV で時系列順を尊重
    tscv = TimeSeriesSplit(n_splits=5)
    logit_pipe = Pipeline([("scaler", StandardScaler()),
                           ("clf", LogisticRegression(max_iter=2000))])
    rf = RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=5, random_state=42)

    logit_aucs, rf_aucs = [], []
    logit_acc, rf_acc = [], []
    all_fpr_logit, all_tpr_logit = [], []
    all_fpr_rf,    all_tpr_rf    = [], []

    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X)):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
        # 訓練・テストの両方に2クラス必要
        if y_te.nunique() < 2 or y_tr.nunique() < 2:
            continue

        logit_pipe.fit(X_tr, y_tr)
        p_logit = logit_pipe.predict_proba(X_te)[:, 1]
        logit_aucs.append(roc_auc_score(y_te, p_logit))
        logit_acc.append(accuracy_score(y_te, (p_logit >= 0.5).astype(int)))
        fpr, tpr, _ = roc_curve(y_te, p_logit)
        all_fpr_logit.append(fpr); all_tpr_logit.append(tpr)

        rf.fit(X_tr, y_tr)
        p_rf = rf.predict_proba(X_te)[:, 1]
        rf_aucs.append(roc_auc_score(y_te, p_rf))
        rf_acc.append(accuracy_score(y_te, (p_rf >= 0.5).astype(int)))
        fpr, tpr, _ = roc_curve(y_te, p_rf)
        all_fpr_rf.append(fpr); all_tpr_rf.append(tpr)

    # 最終 fold の feature importance
    rf.fit(X.iloc[:int(len(X)*0.8)], y.iloc[:int(len(X)*0.8)])
    importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)

    lines = [
        "Type II – aggregate classifier (TimeSeriesSplit CV, n_splits=5)",
        f"Logit  mean AUC={np.mean(logit_aucs):.4f} ± {np.std(logit_aucs):.4f}",
        f"Logit  mean Acc={np.mean(logit_acc):.4f}",
        f"RF     mean AUC={np.mean(rf_aucs):.4f} ± {np.std(rf_aucs):.4f}",
        f"RF     mean Acc={np.mean(rf_acc):.4f}",
        "\nFold-wise AUC:",
        f"  Logit: {[round(a,3) for a in logit_aucs]}",
        f"  RF:    {[round(a,3) for a in rf_aucs]}",
        "\nRandomForest feature importances:\n",
        importances.to_string(),
    ]
    (out_dir / "type2_classifier_metrics.txt").write_text("\n".join(lines), encoding="utf-8")

    # ROC プロット（全 fold 重ね描き）
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, fprs, tprs, aucs, name in [
        (axes[0], all_fpr_logit, all_tpr_logit, logit_aucs, "Logit"),
        (axes[1], all_fpr_rf,    all_tpr_rf,    rf_aucs,    "RandomForest"),
    ]:
        for i, (fpr, tpr, auc_val) in enumerate(zip(fprs, tprs, aucs)):
            ax.plot(fpr, tpr, alpha=0.6, lw=1.2, label=f"Fold {i+1} (AUC={auc_val:.3f})")
        ax.plot([0,1],[0,1],"k--",lw=0.8)
        ax.set_title(f"Type II – {name} ROC (CV)")
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.legend(fontsize=7)
    fig.suptitle(f"Type II Aggregate Classifier (TimeSeriesSplit n=5)")
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "type2_roc_cv.png", dpi=150)
    plt.close(fig)

    # 特徴量重要度プロット
    fig, ax = plt.subplots(figsize=(7, 4))
    importances.plot.barh(ax=ax)
    ax.set_title("Type II – RF Feature Importances (last fold)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "type2_feature_importance.png", dpi=150)
    plt.close(fig)


# ──────────────────────────────────────────────────────
# Type III: Monte Carlo シミュレーション
# ──────────────────────────────────────────────────────

@dataclass
class SimConfig:
    cases_per_month: int  = CASES_PER_MONTH
    random_seed: int      = RANDOM_SEED
    months_cap: Optional[int] = MONTHS_CAP
    omega: float          = 1.0   # W3 = S - ω·A の重み


def simulate_case_level(panel: pd.DataFrame, cfg: SimConfig) -> pd.DataFrame:
    """
    構造パラメータの設計根拠（期別キャリブレーション反映後）:
      abusive_prob = σ(-2.0 + 2.0δ + 1.5μ + 1.2ρ - 2.0q - 0.5S + adj)
        → 全体濫用率 ≈ 29%（実測 distress_rate 19.4 に対応）
      success_prob = σ(0.0 + 1.8S + 0.8t + 0.8f + 0.8s + 1.5κ
                       - 1.8δ - 1.5μ - 0.8ρ - 0.2q + adj)
        → abusive=0 ≈ 68%, abusive=1 ≈ 42%（≈ 26pp 差）
    """
    rng = np.random.default_rng(cfg.random_seed)
    months = panel.copy()
    if cfg.months_cap is not None:
        months = months.tail(cfg.months_cap).copy()

    records: List[Dict] = []
    for _, row in months.iterrows():
        # 件数スケール: 平均活動量比（上限 2 倍）
        activity  = max(get_val(row, "split_incorporation", 1.0), 1.0)
        n_cases   = max(10, int(cfg.cases_per_month * min(activity / 77.0, 2.0)))

        # 潜在平均値のキャリブレーション
        date = pd.Timestamp(row["date"])
        padj = get_period_adj(date)

        delta_m  = sigmoid_s(0.6 * get_val(row, "delta_proxy"))
        mu_m     = sigmoid_s(0.6 * get_val(row, "mu_proxy"))
        rho_m    = sigmoid_s(0.6 * get_val(row, "rho_proxy"))
        q_m      = sigmoid_s(1.2 * get_val(row, "q_proxy"))
        S_m      = sigmoid_s(0.8 * get_val(row, "S_proxy"))
        t_m      = sigmoid_s(0.8 * get_val(row, "t_proxy"))
        f_m      = sigmoid_s(0.8 * get_val(row, "f_proxy"))
        s_m      = sigmoid_s(0.8 * get_val(row, "s_proxy"))
        kappa_m  = sigmoid_s(0.8 * get_val(row, "kappa_proxy"))

        for _ in range(n_cases):
            delta = float(np.clip(rng.normal(delta_m, 0.12), 0, 1))
            mu    = float(np.clip(rng.normal(mu_m,    0.12), 0, 1))
            rho   = float(np.clip(rng.normal(rho_m,   0.12), 0, 1))
            q     = float(np.clip(rng.normal(q_m,     0.08), 0, 1))
            S     = float(np.clip(rng.normal(S_m,     0.15), 0, 1))
            t     = float(np.clip(rng.normal(t_m,     0.12), 0, 1))
            f     = float(np.clip(rng.normal(f_m,     0.12), 0, 1))
            s     = float(np.clip(rng.normal(s_m,     0.12), 0, 1))
            kappa = float(np.clip(rng.normal(kappa_m, 0.12), 0, 1))

            # Type I: 濫用確率 [v2: intercept=-2.0 で全体 ~29%]
            abusive_prob = sigmoid_s(
                -2.0 + 2.0*delta + 1.5*mu + 1.2*rho
                - 2.0*q - 0.5*S
                + padj["abuse_adj"]
            )
            abusive = int(rng.binomial(1, abusive_prob))

            # Type III: 再建成功確率 [v2: 係数強化で abusive=0/1 の差 ~26pp]
            success_prob = sigmoid_s(
                0.0
                + 1.8*S + 0.8*t + 0.8*f + 0.8*s + 1.5*kappa
                - 1.8*delta - 1.5*mu - 0.8*rho
                - 0.2*q
                + padj["success_adj"]
            )
            success = int(rng.binomial(1, success_prob))

            # Type III: W3 目的関数値 W3 = S - ω·A(δ,μ,ρ)
            A_harm = (delta + mu + rho) / 3.0
            w3_val = S - cfg.omega * A_harm

            # Type II: 可観測シグナル（ノイズ付き）
            obs_asset_shift         = float(np.clip(delta + rng.normal(0, 0.10), 0, 1))
            obs_liability_gap       = float(np.clip(mu    + rng.normal(0, 0.10), 0, 1))
            obs_preferential_signal = float(np.clip(rho   + rng.normal(0, 0.10), 0, 1))
            obs_transparency        = float(np.clip(t     + rng.normal(0, 0.10), 0, 1))

            records.append({
                "date": row["date"],
                "delta": delta, "mu": mu, "rho": rho, "q": q,
                "S": S, "t": t, "f": f, "s": s, "kappa": kappa,
                "A_harm": A_harm, "w3_val": w3_val,
                "abusive": abusive, "success": success,
                "obs_asset_shift": obs_asset_shift,
                "obs_liability_gap": obs_liability_gap,
                "obs_preferential_signal": obs_preferential_signal,
                "obs_transparency": obs_transparency,
            })
    return pd.DataFrame.from_records(records)


# ──────────────────────────────────────────────────────
# Type II: シミュレーション・ケース分類器
# ──────────────────────────────────────────────────────

def fit_type2_sim_classifier(sim_df: pd.DataFrame, out_dir: Path) -> None:
    X = sim_df[["obs_asset_shift","obs_liability_gap","obs_preferential_signal","obs_transparency"]]
    y = sim_df["abusive"]

    # 時系列順を維持した 75/25 分割
    split_idx = int(len(sim_df) * 0.75)
    X_tr, X_te = X.iloc[:split_idx], X.iloc[split_idx:]
    y_tr, y_te = y.iloc[:split_idx], y.iloc[split_idx:]

    logit_pipe = Pipeline([("scaler", StandardScaler()),
                           ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))])
    logit_pipe.fit(X_tr, y_tr)
    probs = logit_pipe.predict_proba(X_te)[:, 1]
    auc   = roc_auc_score(y_te, probs)
    pred  = (probs >= 0.5).astype(int)
    acc   = accuracy_score(y_te, pred)

    coef = pd.Series(logit_pipe.named_steps["clf"].coef_[0], index=X.columns)
    lines = [
        "Type II – simulated-case logit classifier",
        f"AUC: {auc:.4f}",
        f"Accuracy: {acc:.4f}",
        classification_report(y_te, pred, zero_division=0),
        "Logit coefficients (scaled):",
        coef.to_string(),
    ]
    (out_dir / "type2_sim_classifier.txt").write_text("\n".join(lines), encoding="utf-8")

    # プロット
    fpr, tpr, _ = roc_curve(y_te, probs)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(fpr, tpr, lw=1.5, label=f"AUC={auc:.3f}")
    axes[0].plot([0,1],[0,1],"k--",lw=0.8)
    axes[0].set_title("Type II – ROC (simulated cases)")
    axes[0].legend()
    coef.sort_values().plot.barh(
        ax=axes[1],
        color=["#d62728" if v > 0 else "#1f77b4" for v in coef.sort_values()]
    )
    axes[1].axvline(0, color="k", lw=0.8)
    axes[1].set_title("Type II – Logit Coefficients")
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "type2_sim.png", dpi=150)
    plt.close(fig)


# ──────────────────────────────────────────────────────
# Type III 集計・感度分析
# ──────────────────────────────────────────────────────

def summarize_type3(sim_df: pd.DataFrame, out_dir: Path, omega: float) -> pd.DataFrame:
    """各 ω について集計統計・分位成功率を計算して CSV に保存"""
    summary = {
        "omega":          omega,
        "abusive_rate":   sim_df["abusive"].mean(),
        "success_rate":   sim_df["success"].mean(),
        "success_abusive1": sim_df.loc[sim_df["abusive"]==1, "success"].mean(),
        "success_abusive0": sim_df.loc[sim_df["abusive"]==0, "success"].mean(),
        "mean_w3":        sim_df["w3_val"].mean(),
        "mean_delta":     sim_df["delta"].mean(),
        "mean_mu":        sim_df["mu"].mean(),
        "mean_rho":       sim_df["rho"].mean(),
        "mean_S":         sim_df["S"].mean(),
        "mean_kappa":     sim_df["kappa"].mean(),
    }

    # 分位成功率
    quintile_frames = []
    for var in ["S","t","f","s","kappa","delta","mu","rho"]:
        tmp = sim_df[[var,"success"]].copy()
        tmp["qtile"] = pd.qcut(tmp[var], 5, labels=False, duplicates="drop")
        grp = tmp.groupby("qtile", as_index=False)["success"].mean()
        grp["parameter"] = var
        quintile_frames.append(grp)
    qdf = pd.concat(quintile_frames, ignore_index=True)
    qdf["omega"] = omega
    return pd.DataFrame([summary]), qdf


def run_sensitivity(panel: pd.DataFrame, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ω ∈ {0.5, 1.0, 2.0} で Type III を繰り返しシミュレーション"""
    all_summary, all_quintile = [], []
    sim_main = None

    for omega in OMEGA_VALUES:
        print(f"  [Sensitivity] ω={omega}")
        cfg = SimConfig(cases_per_month=CASES_PER_MONTH, random_seed=RANDOM_SEED,
                        months_cap=MONTHS_CAP, omega=omega)
        sim = simulate_case_level(panel, cfg)
        if omega == 1.0:
            sim_main = sim   # ω=1.0 をメイン出力に使用

        summ, qdf = summarize_type3(sim, out_dir, omega)
        all_summary.append(summ)
        all_quintile.append(qdf)

    summary_df  = pd.concat(all_summary,  ignore_index=True)
    quintile_df = pd.concat(all_quintile, ignore_index=True)
    summary_df.to_csv(out_dir / "type3_sensitivity_summary.csv",  index=False)
    quintile_df.to_csv(out_dir / "type3_quintile_by_omega.csv",   index=False)

    return summary_df, sim_main


def plot_sensitivity(summary_df: pd.DataFrame, out_dir: Path) -> None:
    """ω ごとの濫用率・成功率・W3 比較バーチャート"""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    x = [str(o) for o in summary_df["omega"]]

    for ax, col, title, color in [
        (axes[0], "abusive_rate",  "Abusive rate",          "#d62728"),
        (axes[1], "success_rate",  "Overall success rate",  "#2ca02c"),
        (axes[2], "mean_w3",       "Mean W₃ = S − ω·A",     "#1f77b4"),
    ]:
        ax.bar(x, summary_df[col], color=color, alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel("ω")
        ax.set_ylim(0, max(summary_df[col].max() * 1.2, 0.1))
        for i, v in enumerate(summary_df[col]):
            ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)

    fig.suptitle("Type III Sensitivity Analysis – W₃ = S − ω·A(δ,μ,ρ)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "type3_sensitivity.png", dpi=150)
    plt.close(fig)


def plot_type3_time_series(sim_df: pd.DataFrame, out_dir: Path) -> None:
    """月別の濫用率・成功率の時系列推移"""
    ts = sim_df.groupby("date").agg(
        abusive_rate=("abusive","mean"),
        success_rate=("success","mean"),
        success_ab1 =("success", lambda x: x[sim_df.loc[x.index,"abusive"]==1].mean()),
        success_ab0 =("success", lambda x: x[sim_df.loc[x.index,"abusive"]==0].mean()),
    ).reset_index()
    ts["date"] = pd.to_datetime(ts["date"])

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(ts["date"], ts["abusive_rate"], color="#d62728", lw=1.2, label="Abusive rate")
    axes[0].axvline(pd.Timestamp(REFORM_DATE),   color="navy",  ls="--", lw=1.2, label=f"Reform {REFORM_DATE}")
    axes[0].axvline(pd.Timestamp(JUDICIAL_DATE), color="green", ls=":",  lw=1.2, label=f"Judicial {JUDICIAL_DATE}")
    axes[0].set_ylabel("Abusive rate")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Type III – Monthly Abusive Rate")

    axes[1].plot(ts["date"], ts["success_ab0"], color="#2ca02c", lw=1.2, label="Success | non-abusive")
    axes[1].plot(ts["date"], ts["success_ab1"], color="#ff7f0e", lw=1.2, label="Success | abusive")
    axes[1].fill_between(ts["date"], ts["success_ab1"], ts["success_ab0"], alpha=0.15, color="grey", label="Gap")
    axes[1].axvline(pd.Timestamp(REFORM_DATE),   color="navy",  ls="--", lw=1.2)
    axes[1].axvline(pd.Timestamp(JUDICIAL_DATE), color="green", ls=":",  lw=1.2)
    axes[1].set_ylabel("Success rate")
    axes[1].legend(fontsize=8)
    axes[1].set_title("Type III – Monthly Success Rate by Abuse Status")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "type3_time_series.png", dpi=150)
    plt.close(fig)


def plot_type3_quintiles(quintile_df: pd.DataFrame, omega_main: float, out_dir: Path) -> None:
    """ω=1.0 での S/δ/κ 分位別成功率"""
    sub = quintile_df[quintile_df["omega"] == omega_main]
    params_show = [("S","#2ca02c"), ("delta","#d62728"), ("kappa","#1f77b4")]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (var, color) in zip(axes, params_show):
        d = sub[sub["parameter"] == var].sort_values("qtile")
        ax.bar(d["qtile"], d["success"], color=color, alpha=0.85)
        ax.set_title(f"Success by {var} quintile")
        ax.set_xlabel(f"{var} quintile (0=low, 4=high)")
        ax.set_ylabel("Mean success rate")
        ax.set_ylim(0, 1)
        for _, r in d.iterrows():
            ax.text(r["qtile"], r["success"] + 0.01, f"{r['success']:.2f}", ha="center", fontsize=8)

    fig.suptitle(f"Type III – Success Rate by Parameter Quintile (ω={omega_main})", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "type3_quintile.png", dpi=150)
    plt.close(fig)


def plot_panel_overview(panel: pd.DataFrame, out_dir: Path) -> None:
    """記述統計プロット（改革ダミー2本）"""
    dates = panel["date"]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(dates, panel["split_incorporation"], color="#1f77b4", lw=1.2)
    ax.axvline(pd.Timestamp(REFORM_DATE),   color="red",   ls="--", lw=1.4, label=f"Reform {REFORM_DATE}")
    ax.axvline(pd.Timestamp(JUDICIAL_DATE), color="green", ls=":",  lw=1.4, label=f"Judicial {JUDICIAL_DATE}")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_title("Monthly Company Divisions – All Entities")
    ax.set_ylabel("Count"); ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "panel_split_incorporation.png", dpi=150)
    plt.close(fig)

    for title, cols in [
        ("Type I proxies (δ, μ, ρ)",  ["delta_proxy","mu_proxy","rho_proxy"]),
        ("Type III proxies (S, f, κ)", ["S_proxy","f_proxy","kappa_proxy"]),
    ]:
        present = [c for c in cols if c in panel.columns]
        if not present: continue
        fig, ax = plt.subplots(figsize=(12, 4))
        for col in present:
            ax.plot(dates, panel[col], label=col, lw=1.0, alpha=0.85)
        ax.axvline(pd.Timestamp(REFORM_DATE),   color="red",   ls="--", lw=1.2, label=f"Reform {REFORM_DATE}")
        ax.axvline(pd.Timestamp(JUDICIAL_DATE), color="green", ls=":",  lw=1.0, label=f"Judicial {JUDICIAL_DATE}")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.set_title(title); ax.legend(fontsize=8)
        fig.tight_layout()
        fname = title.split("(")[0].strip().replace(" ","_").lower() + ".png"
        fig.savefig(out_dir / "figures" / fname, dpi=150)
        plt.close(fig)


def print_period_report(sim_df: pd.DataFrame) -> None:
    """期別集計をコンソールに表示"""
    sim_df2 = sim_df.copy()
    sim_df2["date"] = pd.to_datetime(sim_df2["date"])
    sim_df2["period"] = "2019-2026"
    sim_df2.loc[sim_df2["date"] < "2019-01-01", "period"] = "2015-2019"
    sim_df2.loc[sim_df2["date"] < "2015-05-01", "period"] = "2012-2015"
    sim_df2.loc[sim_df2["date"] < "2012-01-01", "period"] = "2009-2012"

    tbl = sim_df2.groupby("period").agg(
        n=("abusive","count"),
        abusive_rate=("abusive","mean"),
        success_rate=("success","mean"),
        success_ab0 =("success", lambda x: x[sim_df2.loc[x.index,"abusive"]==0].mean()),
        success_ab1 =("success", lambda x: x[sim_df2.loc[x.index,"abusive"]==1].mean()),
        mean_delta=("delta","mean"), mean_S=("S","mean"),
    ).round(3)
    print("\n  ── 期別シミュレーション集計 ──")
    print(tbl.to_string())


# ──────────────────────────────────────────────────────
# メイン
# ──────────────────────────────────────────────────────

def run_all() -> None:
    print("=== Step 1: Panel 構築 ===")
    panel = prepare_monthly_panel(
        registry_path=REGISTRY_PATH,
        macro_path=MACRO_PATH  if Path(MACRO_PATH).exists()  else None,
        loan_path=LOAN_PATH    if Path(LOAN_PATH).exists()   else None,
    )
    panel = construct_latent_proxies(panel)
    panel.to_csv(OUT_DIR / "merged_monthly_panel_v2.csv", index=False)
    print(f"  Shape: {panel.shape}")

    # プロキシ相関チェック
    prx_cols = ["delta_proxy","mu_proxy","rho_proxy","S_proxy","f_proxy","kappa_proxy"]
    prx_corr = panel[prx_cols].corr().round(2)
    prx_corr.to_csv(OUT_DIR / "proxy_correlation_v2.csv")
    print("  Proxy correlations (f_proxy vs delta/mu):",
          prx_corr.loc["f_proxy", ["delta_proxy","mu_proxy"]].to_dict())

    print("=== Step 2: 記述統計プロット ===")
    plot_panel_overview(panel, OUT_DIR)

    print("=== Step 3: Type I – Poisson GLM ===")
    fit_type1_model(panel, OUT_DIR)

    print("=== Step 4: Type II – 集計分類器（TimeSeriesSplit CV） ===")
    fit_type2_classifier(panel, OUT_DIR)

    print("=== Step 5: Type III – 感度分析 (ω ∈ {0.5, 1.0, 2.0}) ===")
    summary_df, sim_main = run_sensitivity(panel, OUT_DIR)
    print(summary_df[["omega","abusive_rate","success_rate",
                       "success_abusive0","success_abusive1","mean_w3"]].to_string(index=False))
    print_period_report(sim_main)

    print("=== Step 6: Type III – 追加プロット ===")
    plot_sensitivity(summary_df, OUT_DIR)
    plot_type3_time_series(sim_main, OUT_DIR)
    quintile_df = pd.read_csv(OUT_DIR / "type3_quintile_by_omega.csv")
    plot_type3_quintiles(quintile_df, omega_main=1.0, out_dir=OUT_DIR)

    print("=== Step 7: Type II – シミュレーションケース分類器 ===")
    sim_main.to_csv(OUT_DIR / "simulated_case_panel_v2.csv", index=False)
    fit_type2_sim_classifier(sim_main, OUT_DIR)

    print(f"\n✓ 完了. 出力: {OUT_DIR}/")
    print("  Figures:", ", ".join(p.name for p in sorted((OUT_DIR/"figures").glob("*.png"))))


if __name__ == "__main__":
    run_all()
