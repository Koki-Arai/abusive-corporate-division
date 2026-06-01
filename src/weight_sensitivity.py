"""
Weight-sensitivity analysis for Appendix C latent-construct weights.

Rebuilds the latent proxies (delta, mu, rho, S, t, f, s, kappa) under
randomly perturbed construct weights, re-runs the case-level simulation
(holding the simulation RNG fixed so that only the weights vary), and
records the key qualitative outputs:
  - overall simulated abuse rate
  - success rates (overall / non-abusive / abusive) and the gap
  - quintile gradients of success across S, kappa (increasing) and delta (decreasing)
  - monotonicity / ordering flags

Purpose: rebut the "tautology / arbitrary weights" critique by showing
the qualitative conclusions survive sizable weight perturbations.
"""
import sys, os, numpy as np, pandas as pd
from pathlib import Path
_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))

import abusive_division_simulation_v2 as v2

_REPO = _SRC.parent
DATA_DIR = Path(os.environ.get("ADV_DATA_DIR", _REPO / "data"))
OUT_DIR  = Path(os.environ.get("ADV_OUTPUT_DIR", _REPO / "outputs")) / "weight_sensitivity"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REG  = str(DATA_DIR / "commercial_registry_monthly_clean_for_simulation.csv")
MAC  = str(DATA_DIR / "マクロ指標.csv")
LOAN = str(DATA_DIR / "貸出債券市場取引動向_全銀協_.csv")

zscore = v2.zscore

# ---- baseline construct weights (Appendix C) ----
BASE = {
    "delta": [0.50, 0.30, 0.20],          # liq(+), capbal(-), bank(+)
    "mu":    [0.70, 0.30],                 # capbal(-), distress(+)
    "rho":   [0.60, 0.40],                 # specliq(+), reorg(+)
    "S":     [0.50, 0.50],                 # intensity(+), macro_good(+)
    "t":     [0.40, 0.20, 0.20, 0.20],     # reform, judicial, reg_total, incorp
    "f":     [0.60, 0.40],                 # completion(+), merger_diss(+)
    "s":     [0.50, 0.50],                 # S_z, t_z
    "kappa": [-0.40, 0.30, 0.30],          # distress(-), completion_rate(+), S(+)
}

def renorm_nonneg(w):
    w = np.abs(np.asarray(w, float)); return list(w / w.sum())

def renorm_signed(w):
    w = np.asarray(w, float); s = np.sign(w); a = np.abs(w)
    a = a / a.sum(); return list(s * a)

def perturb(base, sigma, rng):
    """Multiplicative lognormal perturbation, sign-preserving, renormalized."""
    out = {}
    for k, w in base.items():
        w = np.asarray(w, float)
        fac = np.exp(rng.normal(0.0, sigma, size=len(w)))
        pw = w * fac
        out[k] = renorm_signed(pw) if k == "kappa" else renorm_nonneg(pw)
    return out

def build_proxies(panel, W):
    out = panel.copy()
    d = W["delta"]
    out["delta_proxy"] = (d[0]*zscore(out["liquidation_rate_on_split_lag3"].fillna(0))
        + d[1]*(-1)*zscore(out["split_capital_balance_lag3"].fillna(0))
        + d[2]*zscore(out["bankruptcy_rate_on_split_lag3"].fillna(0)))
    m = W["mu"]
    out["mu_proxy"] = (m[0]*(-1)*zscore(out["split_capital_balance_lag3"].fillna(0))
        + m[1]*zscore(out["distress_rate_on_split_lag3"].fillna(0)))
    r = W["rho"]
    out["rho_proxy"] = (r[0]*zscore(out["special_liquidation"].shift(1).fillna(0))
        + r[1]*zscore(out["corporate_reorganization"].shift(1).fillna(0)))
    out["q_proxy"] = 0.50*out["post_reform_dummy"] + 0.30*out["post_judicial_dummy"]
    if "loan_rate_z" in out.columns:
        out["q_proxy"] += 0.20*out["loan_rate_z"].fillna(0)
    macro_good = np.zeros(len(out))
    for col, w in [("nikkei_close_z",0.30),("topix_close_z",0.20),
                   ("loan_rate_z",-0.30),("syndicated_amount_z",0.20)]:
        if col in out.columns:
            macro_good += w*out[col].fillna(0)
    sW = W["S"]
    out["S_proxy"] = sW[0]*zscore(out["split_intensity_lag3"].fillna(0)) + sW[1]*macro_good
    t = W["t"]
    out["t_proxy"] = (t[0]*out["post_reform_dummy"] + t[1]*out["post_judicial_dummy"]
        + t[2]*zscore(out["registrations_total"].fillna(0))
        + t[3]*zscore(out["incorporation"].fillna(0)))
    f = W["f"]
    out["f_proxy"] = (f[0]*zscore(out["completion_ratio_total_lag3"].fillna(0))
        + f[1]*zscore(out["merger_dissolution_rate_lag3"].fillna(0)))
    sc = W["s"]
    out["s_proxy"] = sc[0]*zscore(out["S_proxy"].fillna(0)) + sc[1]*zscore(out["t_proxy"].fillna(0))
    k = W["kappa"]
    out["kappa_proxy"] = (k[0]*zscore(out["distress_rate_on_split_lag3"].fillna(0))
        + k[1]*zscore(out["completion_rate_on_split_lag3"].fillna(0))
        + k[2]*zscore(out["S_proxy"].fillna(0)))
    return out

def summarize(sim):
    ab = sim["abusive"].mean()
    so = sim["success"].mean()
    s0 = sim.loc[sim["abusive"]==0,"success"].mean()
    s1 = sim.loc[sim["abusive"]==1,"success"].mean()
    def grad(var):
        t = sim[[var,"success"]].copy()
        t["q"] = pd.qcut(t[var].rank(method="first"), 5, labels=False)
        g = t.groupby("q")["success"].mean()
        mono_inc = bool((g.diff().dropna() >= -1e-9).all())
        mono_dec = bool((g.diff().dropna() <= 1e-9).all())
        return g.iloc[0], g.iloc[-1], g.iloc[-1]-g.iloc[0], mono_inc, mono_dec
    S_lo,S_hi,S_d,S_mi,_ = grad("S")
    k_lo,k_hi,k_d,k_mi,_ = grad("kappa")
    d_lo,d_hi,d_d,_,d_md = grad("delta")
    return dict(abuse=ab, succ=so, succ_nonab=s0, succ_ab=s1, gap=s0-s1,
                S_lo=S_lo,S_hi=S_hi,S_grad=S_d,S_mono_inc=S_mi,
                k_lo=k_lo,k_hi=k_hi,k_grad=k_d,k_mono_inc=k_mi,
                d_lo=d_lo,d_hi=d_hi,d_grad=d_d,d_mono_dec=d_md)

def main():
    print("Loading panel ...")
    panel = v2.prepare_monthly_panel(REG, MAC, LOAN)
    print(f"panel rows = {len(panel)}")
    cfg = v2.SimConfig(random_seed=42)  # fixed sim seed -> only weights vary

    # --- baseline ---
    base_panel = build_proxies(panel, BASE)
    base_sim = v2.simulate_case_level(base_panel, cfg)
    base = summarize(base_sim)
    print("\n=== BASELINE (Appendix C weights) ===")
    for k in ["abuse","succ","succ_nonab","succ_ab","gap",
              "S_lo","S_hi","S_grad","k_lo","k_hi","k_grad","d_lo","d_hi","d_grad"]:
        print(f"  {k:12s} {base[k]:.4f}")

    # cross-check: corr of baseline proxies with construct columns (sanity)
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    SIGMA = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25
    print(f"\nRunning {N} weight perturbations (sigma={SIGMA}) ...")
    rng = np.random.default_rng(2025)
    rows = []
    for i in range(N):
        W = perturb(BASE, SIGMA, rng)
        p = build_proxies(panel, W)
        sim = v2.simulate_case_level(p, cfg)
        rows.append(summarize(sim))
        if (i+1) % 25 == 0:
            print(f"  ... {i+1}/{N}")
    R = pd.DataFrame(rows)
    R.to_csv(str(OUT_DIR / "weight_sensitivity_runs.csv"), index=False)

    def band(col):
        s = R[col]
        return f"min={s.min():.3f}  p5={s.quantile(.05):.3f}  med={s.median():.3f}  p95={s.quantile(.95):.3f}  max={s.max():.3f}"
    print("\n=== PERTURBATION DISTRIBUTION ===")
    for c in ["abuse","succ","succ_nonab","succ_ab","gap","S_grad","k_grad","d_grad"]:
        print(f"  {c:10s} {band(c)}")
    print("\n=== QUALITATIVE STABILITY (share of runs) ===")
    print(f"  abuse in [0.20,0.32]          : {(R['abuse'].between(0.20,0.32)).mean():.3f}")
    print(f"  non-abusive succeeds more     : {(R['gap']>0).mean():.3f}")
    print(f"  S gradient positive           : {(R['S_grad']>0).mean():.3f}")
    print(f"  S gradient monotone-increasing: {R['S_mono_inc'].mean():.3f}")
    print(f"  kappa gradient positive       : {(R['k_grad']>0).mean():.3f}")
    print(f"  kappa gradient monotone-inc   : {R['k_mono_inc'].mean():.3f}")
    print(f"  delta gradient negative       : {(R['d_grad']<0).mean():.3f}")
    print(f"  delta gradient monotone-dec   : {R['d_mono_dec'].mean():.3f}")
    print(f"  ALL FOUR sign conditions hold : {((R['gap']>0)&(R['S_grad']>0)&(R['k_grad']>0)&(R['d_grad']<0)).mean():.3f}")
    # equal-weights alternative
    EQ = {k:[1.0/len(v)]*len(v) if k!='kappa' else [-1/3.,1/3.,1/3.] for k,v in BASE.items()}
    eq_sim = v2.simulate_case_level(build_proxies(panel, EQ), cfg)
    eq = summarize(eq_sim)
    print("\n=== EQUAL-WEIGHTS ALTERNATIVE ===")
    print(f"  abuse={eq['abuse']:.3f} gap={eq['gap']:.3f} S_grad={eq['S_grad']:.3f} "
          f"k_grad={eq['k_grad']:.3f} d_grad={eq['d_grad']:.3f}")

if __name__ == "__main__":
    main()
