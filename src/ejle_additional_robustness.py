"""
Additional robustness for the abusive-corporate-division paper (referee response).

Part 1  Outcome robustness   : re-estimate the §4.2 NB and the §5.2 forecast
                               counterfactual with the dependent variable =
                               (a) special liquidation, (b) bankruptcy/civil
                               rehabilitation, (c) combined; with and without
                               the (overlapping) distress regressor.
Part 2  Period-constant      : perturb the hand-set period calibration
        sensitivity            constants (abuse_adj / success_adj), plus a
                               zero-constant placebo, re-simulate, and check how
                               much of the period pattern (2012-15 highest) and
                               the overall abuse rate / gradients depend on them.
Part 3  Trend-form robustness: re-estimate the §4.2 NB with the time trend as
                               linear / quadratic / natural-cubic-spline, and
                               report post-2015 / post-2019 coefficients with a
                               joint Wald test and a multiple-testing note.

Run:  python ejle_additional_robustness.py --part all
"""
import sys, copy, argparse, os
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.api as sm

_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))
import abusive_division_simulation_v2 as base
import ejle_robustness_checks as rc

_REPO = _SRC.parent
DATA_DIR = Path(os.environ.get("ADV_DATA_DIR", _REPO / "data"))
OUT_DIR  = Path(os.environ.get("ADV_OUTPUT_DIR", _REPO / "outputs")) / "additional_robustness"
OUT_DIR.mkdir(parents=True, exist_ok=True)
base.REGISTRY_PATH = str(DATA_DIR / "commercial_registry_monthly_clean_for_simulation.csv")
base.MACRO_PATH    = str(DATA_DIR / "マクロ指標.csv")
base.LOAN_PATH     = str(DATA_DIR / "貸出債券市場取引動向_全銀協_.csv")

REFORM = pd.Timestamp("2015-05-01")
KEY = ["post_reform_dummy", "post_judicial_dummy"]


def get_panel():
    p = rc.build_panel().copy()
    p["date"] = pd.to_datetime(p["date"])
    if "t" not in p.columns:
        p["t"] = np.arange(len(p), dtype=float)
    return p


def base_regressors(panel, with_distress=True):
    regs = ["split_incorporation_lag1", "split_capital_balance_lag1"]
    if with_distress:
        regs += ["distress_rate_on_split_lag1"]
    regs += ["post_reform_dummy", "post_judicial_dummy", "t"]
    regs += rc._macro_controls(panel)
    return [c for c in regs if c in panel.columns]


def irr(nb, name):
    if name not in nb.params.index:
        return None
    b, se, p = nb.params[name], nb.bse[name], nb.pvalues[name]
    return np.exp(b), np.exp(b - 1.96 * se), np.exp(b + 1.96 * se), p


# ───────────────────────────── Part 1 ─────────────────────────────
def counterfactual(panel, ycol, regressors, n_boot=1000):
    df = panel.copy()
    df["y"] = df[ycol]
    work = df[["date", "y"] + regressors].dropna().reset_index(drop=True)
    pre = work[work["date"] < REFORM]
    post = work[work["date"] >= REFORM]
    cont = [c for c in regressors if work[c].nunique(dropna=True) > 2]
    mu, sd = pre[cont].mean(), pre[cont].std().replace(0, 1.0)

    def scale(d):
        d = d.copy()
        for c in cont:
            d[c] = (d[c] - mu[c]) / sd[c]
        return d

    Xpre = sm.add_constant(scale(pre[regressors]), has_constant="add")
    Xpost = sm.add_constant(scale(post[regressors]), has_constant="add")[Xpre.columns]
    ypre = np.rint(pre["y"]).astype(float)
    pois = sm.GLM(ypre, Xpre, family=sm.families.Poisson()).fit()
    m = np.asarray(pois.mu)
    aux = ((ypre - m) ** 2 - ypre) / m
    alpha = float(np.maximum((m @ aux) / (m @ m), 1e-6))
    nb = sm.GLM(ypre, Xpre, family=sm.families.NegativeBinomial(alpha=alpha)).fit()
    cf = np.asarray(nb.predict(Xpost)); obs = np.asarray(post["y"], float)
    eff = (obs.sum() - cf.sum()) / cf.sum() * 100.0
    rng = np.random.default_rng(0)
    draws = rng.multivariate_normal(nb.params.values, nb.cov_params().values, n_boot)
    effs = [(obs.sum() - np.exp(Xpost.values @ d).sum()) / np.exp(Xpost.values @ d).sum() * 100.0
            for d in draws]
    lo, hi = np.percentile(effs, [2.5, 97.5])
    return dict(effect=eff, lo=lo, hi=hi, obs=obs.sum(), cf=cf.sum(),
                n_pre=len(pre), n_post=len(post), alpha=alpha)


def part1(panel):
    print("\n================ PART 1: OUTCOME ROBUSTNESS ================")
    panel = panel.copy()
    panel["combined_distress"] = (panel["special_liquidation"]
                                  + panel["bankruptcy_or_civil_rehabilitation"])
    outcomes = [("special_liquidation", "special liquidation"),
                ("bankruptcy_or_civil_rehabilitation", "bankruptcy / civil rehab"),
                ("combined_distress", "combined (SL + bankruptcy)")]
    rows = []
    for ycol, label in outcomes:
        for wd in ([True, False] if ycol != "special_liquidation" else [True]):
            regs = base_regressors(panel, with_distress=wd)
            df = panel.copy(); df["y"] = df[ycol].shift(-1)
            work = df[["y"] + regs].dropna().copy()
            X = rc._zscore_continuous(work[regs])
            nb = rc._fit_count(work["y"], X, "nb")
            r = {"outcome": label, "distress_reg": wd, "n": int(len(work)),
                 "alpha": round(getattr(nb, "_alpha_used", float("nan")), 4)}
            for k in KEY + (["distress_rate_on_split_lag1"] if wd else []):
                v = irr(nb, k)
                if v:
                    r[f"{k}_IRR"] = round(v[0], 3)
                    r[f"{k}_CI"] = f"[{v[1]:.2f},{v[2]:.2f}]"
                    r[f"{k}_p"] = round(v[3], 3)
            rows.append(r)
            print(f"  [{label} | distress_reg={wd} | n={r['n']}] "
                  f"reform IRR {r.get('post_reform_dummy_IRR')} {r.get('post_reform_dummy_CI')} "
                  f"p={r.get('post_reform_dummy_p')} | "
                  f"judicial IRR {r.get('post_judicial_dummy_IRR')} {r.get('post_judicial_dummy_CI')} "
                  f"p={r.get('post_judicial_dummy_p')}")
    pd.DataFrame(rows).to_csv(str(OUT_DIR / "out_part1_irr.csv"), index=False)

    print("\n  -- forecast counterfactual (cumulative % vs no-reform), distress reg ON --")
    crows = []
    for ycol, label in outcomes:
        regs = base_regressors(panel, with_distress=True)
        c = counterfactual(panel, ycol, regs, n_boot=1500)
        c2 = {"outcome": label, **{k: (round(v, 2) if isinstance(v, float) else v)
                                   for k, v in c.items()}}
        crows.append(c2)
        print(f"  [{label}] cum effect {c['effect']:+.1f}%  95%CI [{c['lo']:+.1f},{c['hi']:+.1f}]  "
              f"(obs {c['obs']:.0f} vs cf {c['cf']:.0f}; n_pre {c['n_pre']}, n_post {c['n_post']})")
    pd.DataFrame(crows).to_csv(str(OUT_DIR / "out_part1_counterfactual.csv"), index=False)


# ───────────────────────────── Part 2 ─────────────────────────────
PERIODS = ["2009-2012", "2012-2015", "2015-2019", "2019-2026"]
ORIG = copy.deepcopy(base.PERIOD_CALIBRATION)


def set_constants(vals):
    for k in PERIODS:
        base.PERIOD_CALIBRATION[k]["abuse_adj"] = vals[k]["abuse_adj"]
        base.PERIOD_CALIBRATION[k]["success_adj"] = vals[k]["success_adj"]


def reset_constants():
    set_constants(ORIG)


def period_label(d):
    d = pd.Timestamp(d)
    if d < pd.Timestamp("2012-01-01"): return "2009-2012"
    if d < pd.Timestamp("2015-05-01"): return "2012-2015"
    if d < pd.Timestamp("2019-01-01"): return "2015-2019"
    return "2019-2026"


def summarize_sim(sim):
    sim = sim.copy()
    sim["per"] = sim["date"].map(period_label)
    per = sim.groupby("per")["abusive"].mean().reindex(PERIODS)
    def grad(v):
        t = sim[[v, "success"]].copy()
        t["q"] = pd.qcut(t[v].rank(method="first"), 5, labels=False)
        g = t.groupby("q")["success"].mean()
        return g.iloc[-1] - g.iloc[0]
    return dict(abuse=sim["abusive"].mean(), per=per,
                S_grad=grad("S"), k_grad=grad("kappa"), d_grad=grad("delta"),
                argmax=per.idxmax())


def part2(panel, N=70, sa=0.15, ss=0.10):
    print("\n================ PART 2: PERIOD-CONSTANT SENSITIVITY ================")
    cfg = base.SimConfig(random_seed=42)
    reset_constants()
    base_sum = summarize_sim(base.simulate_case_level(panel, cfg))
    print("  baseline period abuse rates:",
          {k: round(v, 3) for k, v in base_sum["per"].items()},
          "| overall", round(base_sum["abuse"], 3), "| argmax", base_sum["argmax"])

    # placebo: all constants = 0
    set_constants({k: {"abuse_adj": 0.0, "success_adj": 0.0} for k in PERIODS})
    plac = summarize_sim(base.simulate_case_level(panel, cfg))
    reset_constants()
    print("  PLACEBO (all constants = 0) period abuse rates:",
          {k: round(v, 3) for k, v in plac["per"].items()},
          "| overall", round(plac["abuse"], 3), "| argmax", plac["argmax"])

    # perturbations
    rng = np.random.default_rng(7)
    rows = []
    for i in range(N):
        vals = {k: {"abuse_adj": ORIG[k]["abuse_adj"] + rng.normal(0, sa),
                    "success_adj": ORIG[k]["success_adj"] + rng.normal(0, ss)}
                for k in PERIODS}
        set_constants(vals)
        sm_ = summarize_sim(base.simulate_case_level(panel, cfg))
        rows.append(dict(abuse=sm_["abuse"], S_grad=sm_["S_grad"], k_grad=sm_["k_grad"],
                         d_grad=sm_["d_grad"], argmax=sm_["argmax"],
                         **{f"per_{k}": sm_["per"][k] for k in PERIODS}))
    reset_constants()
    R = pd.DataFrame(rows)
    R.to_csv(str(OUT_DIR / "out_part2_period.csv"), index=False)
    print(f"\n  perturbations N={N}  (abuse_adj ~N(0,{sa}), success_adj ~N(0,{ss}))")
    print(f"  overall abuse : min={R.abuse.min():.3f} med={R.abuse.median():.3f} max={R.abuse.max():.3f}")
    print(f"  2012-15 is highest-abuse period in {(R.argmax=='2012-2015').mean():.3f} of draws")
    print(f"  S_grad>0 {(R.S_grad>0).mean():.3f} | k_grad>0 {(R.k_grad>0).mean():.3f} | d_grad<0 {(R.d_grad<0).mean():.3f}")
    for k in PERIODS:
        c = R[f"per_{k}"]
        print(f"  abuse[{k}] : min={c.min():.3f} med={c.median():.3f} max={c.max():.3f}")


# ───────────────────────────── Part 3 ─────────────────────────────
def part3(panel):
    print("\n================ PART 3: TREND-FORM ROBUSTNESS (special liquidation) ================")
    df = panel.copy(); df["y"] = df["special_liquidation"].shift(-1)
    base_regs = ["split_incorporation_lag1", "split_capital_balance_lag1",
                 "distress_rate_on_split_lag1", "post_reform_dummy",
                 "post_judicial_dummy"] + rc._macro_controls(panel)
    base_regs = [c for c in base_regs if c in panel.columns]

    forms = {}
    forms["linear"] = (["t"], None)
    df["t2"] = df["t"] ** 2
    forms["quadratic"] = (["t", "t2"], None)
    try:
        import patsy
        forms["natural_spline(df=4)"] = ("SPLINE", 4)
    except Exception as e:
        df["t3"] = df["t"] ** 3
        forms["cubic_poly"] = (["t", "t2", "t3"], None)
        print("  (patsy unavailable; using cubic polynomial instead of spline)")

    rows = []
    for name, (tcols, sdf) in forms.items():
        d = df.copy()
        if tcols == "SPLINE":
            import patsy
            B = patsy.dmatrix(f"cr(t, df={sdf}) - 1", {"t": d["t"]}, return_type="dataframe")
            B.columns = [f"spl{i}" for i in range(B.shape[1])]
            d = pd.concat([d, B], axis=1)
            trend_cols = list(B.columns)
        else:
            trend_cols = tcols
        regs = base_regs + trend_cols
        work = d[["y"] + regs].dropna().copy()
        X = rc._zscore_continuous(work[regs])
        nb = rc._fit_count(work["y"], X, "nb")
        rr = irr(nb, "post_reform_dummy"); jj = irr(nb, "post_judicial_dummy")
        wj = rc._wald_zero(nb, KEY)
        row = dict(trend=name, n=int(len(work)),
                   reform_IRR=round(rr[0], 3), reform_CI=f"[{rr[1]:.2f},{rr[2]:.2f}]", reform_p=round(rr[3], 3),
                   judicial_IRR=round(jj[0], 3), judicial_CI=f"[{jj[1]:.2f},{jj[2]:.2f}]", judicial_p=round(jj[3], 3),
                   judicial_p_bonf=round(min(1.0, jj[3] * 2), 3),
                   joint_Wald_p=round(wj[1], 3) if wj else None)
        rows.append(row)
        print(f"  [{name:22s}] reform IRR {row['reform_IRR']} {row['reform_CI']} p={row['reform_p']} | "
              f"judicial IRR {row['judicial_IRR']} {row['judicial_CI']} p={row['judicial_p']} "
              f"(Bonf {row['judicial_p_bonf']}) | jointWald p={row['joint_Wald_p']}")
    pd.DataFrame(rows).to_csv(str(OUT_DIR / "out_part3_trend.csv"), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all")
    ap.add_argument("--n2", type=int, default=70)
    a = ap.parse_args()
    panel = get_panel()
    print(f"panel rows = {len(panel)}  cols incl t={'t' in panel.columns}")
    if a.part in ("1", "all"): part1(panel)
    if a.part in ("3", "all"): part3(panel)
    if a.part in ("2", "all"): part2(panel, N=a.n2)


if __name__ == "__main__":
    main()
