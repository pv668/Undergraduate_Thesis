# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 13:03:25 2026

@author: prakh
"""

# -*- coding: utf-8 -*-
"""
Two-fluid TOV solver in f(R) = R + alpha R^2 gravity
Dimensionless formulation following Kenji Nishiwaki notes:
- Two-fluid f(R) equations: Eqs. (1.50) - (1.54)
- Fermionic DM EOS: Eqs. (3.5) - (3.7)

State vector in f(R):
    y = [m, pB, pD, R, Rp]

State vector in GR limit alpha=0:
    y = [m, pB, pD]

Conventions:
- c = G = 1 in the TOV system
- reference length rb = r_S,Sun = 2 G M_sun / c^2 ~ 2.954 km
- baryon EOS = BSR2 7-layer piecewise polytrope
- dark matter EOS = Kenji Sec. 3.1 fermionic model, numerically tabulated and interpolated
"""

import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

# ============================================================
# Plot settings
# ============================================================
plt.rcParams.update({
    "figure.dpi": 110,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "lines.linewidth": 1.8,
    "axes.spines.top": True,
    "axes.spines.right": True
})

# ============================================================
# Physical constants and reference scales
# ============================================================
GSI = 6.67430e-11
cSI = 2.99792458e8
Msun = 1.989e30

rb = 2.0 * GSI * Msun / cSI**2
rbkm = rb / 1.0e3

# ============================================================
# User settings
# ============================================================
ALPHA_FIXED_KM2 = 1.0
DELTA_CENTRAL = 1.0
PCMINKM2 = 3.0e-6
PCMAXKM2 = 8.0e-3
NPC = 40
RSTART = 1.0e-4
REND = 80.0
PTOL = 1.0e-12
MAXSTEP = 0.02
RTOL = 1.0e-7
ATOL = 1.0e-9
CMAPNAME = "plasma"

# ============================================================
# Unit conversions
# ============================================================
def alphatodimless(alpha_km2):
    return alpha_km2 / rbkm**2

def ptodimless(p_km2):
    return p_km2 * rbkm**2

def pdimlesstoPa(p_tilde):
    return p_tilde * cSI**4 / (GSI * rb**2)

def epsdimlesstoPa(eps_tilde):
    return eps_tilde * cSI**4 / (GSI * rb**2)

def mtosolar(m_tilde):
    return m_tilde / 0.5

def rtokm(r_tilde):
    return r_tilde * rbkm

def riccitokm2(R_tilde):
    return R_tilde / rbkm**2

def ricciprimetokm3(Rp_tilde):
    return Rp_tilde / rbkm**3

# ============================================================
# BSR2 EOS (baryons)
# ============================================================
BSR2LOG10K0 = 12.4812
BSR2GAMMAS = np.array([1.6379, 1.3113, 0.8349, 1.3136, 3.2464, 2.8221, 2.3788], dtype=float)
BSR2LOG10RHO = np.array([6.9304, 11.3669, 12.7363, 14.0413, 14.8162, 14.9832], dtype=float)
BSR2RHOBCGS = 10.0**BSR2LOG10RHO
RHOREFCGS = 2.5e14
CCGS = cSI * 1.0e2

KCGS = np.zeros(7, dtype=float)
KCGS[0] = 10.0**BSR2LOG10K0
for i in range(1, 7):
    KCGS[i] = KCGS[i-1] * BSR2RHOBCGS[i-1]**(BSR2GAMMAS[i-1] - BSR2GAMMAS[i])

ACGS = np.zeros(7, dtype=float)
EPSBCGS = np.zeros(7, dtype=float)
for i in range(1, 7):
    rhoi = BSR2RHOBCGS[i-1]
    epsi = (1.0 + ACGS[i-1]) * rhoi + KCGS[i-1] * rhoi**BSR2GAMMAS[i-1] / (BSR2GAMMAS[i-1] - 1.0) / CCGS**2
    EPSBCGS[i] = epsi
    ACGS[i] = epsi / rhoi - 1.0 - KCGS[i] * rhoi**(BSR2GAMMAS[i] - 1.0) / (BSR2GAMMAS[i] - 1.0) / CCGS**2

PBCGS = np.zeros(7, dtype=float)
for i in range(1, 7):
    PBCGS[i] = KCGS[i-1] * BSR2RHOBCGS[i-1]**BSR2GAMMAS[i-1]

PTOTILDE = 0.1 * GSI * rb**2 / cSI**4
RHOTOTILDE = 1.0e3 * GSI * rb**2 / cSI**2
PBTILDE = PBCGS * PTOTILDE
RHOREFTILDE = RHOREFCGS * RHOTOTILDE
KPREFTILDE = KCGS * (RHOREFCGS**BSR2GAMMAS) * PTOTILDE

def bsr2_layer_from_p(ptilde):
    ptilde = max(float(ptilde), 0.0)
    return int(np.clip(np.searchsorted(PBTILDE[1:], ptilde, side="right"), 0, 6))

def baryon_eps_of_p(ptilde):
    p = max(float(ptilde), 1.0e-30)
    i = bsr2_layer_from_p(p)
    Gi = BSR2GAMMAS[i]
    ai = ACGS[i]
    Kpr = KPREFTILDE[i]
    rho_ratio = (p / Kpr)**(1.0 / Gi)
    rho_tilde = RHOREFTILDE * rho_ratio
    eps_tilde = (1.0 + ai) * rho_tilde + p / (Gi - 1.0)
    return eps_tilde

def baryon_eps_array(parr):
    return np.array([baryon_eps_of_p(p) for p in np.asarray(parr, dtype=float)], dtype=float)

# ============================================================
# Dark-matter EOS from Kenji notes Sec. 3.1
# Eqs. (3.5) - (3.7)
# ============================================================
DM_MREF_GEV = 1.0
DM_MASS_GEV = 1.0
DM_C_GEV_INV = 1.0
DM_NHAT_MIN = 1.0e-12
DM_NHAT_MAX = 1.0e2
DM_NTAB = 1000

GEV_TO_J = 1.602176634e-10
HBARC_GEV_FM = 0.1973269804
HBARC_GEV_M = HBARC_GEV_FM * 1.0e-15
GEV4_TO_PA = GEV_TO_J / (HBARC_GEV_M**3)
GEV4_TO_TOV_DIMLESS = GEV4_TO_PA * GSI * rb**2 / cSI**4

DM_mhat = DM_MASS_GEV / DM_MREF_GEV
DM_chat = DM_C_GEV_INV * DM_MREF_GEV

def dm_kf_hat(nhat):
    return (3.0 * np.pi**2 * nhat)**(1.0 / 3.0)

def dm_eps_hat_of_nhat(nhat):
    if nhat <= 0.0:
        return 0.0
    kF = dm_kf_hat(nhat)
    def integrand(k):
        return k**2 * np.sqrt(k**2 + DM_mhat**2)
    val, _ = quad(integrand, 0.0, kF, epsrel=1e-9, limit=200)
    return val / (2.0 * np.pi**2) + 0.5 * DM_chat**2 * nhat**2

def dm_p_hat_of_nhat(nhat):
    if nhat <= 0.0:
        return 0.0
    kF = dm_kf_hat(nhat)
    def integrand(k):
        return k**4 / np.sqrt(k**2 + DM_mhat**2)
    val, _ = quad(integrand, 0.0, kF, epsrel=1e-9, limit=200)
    return val / (6.0 * np.pi**2) + 0.5 * DM_chat**2 * nhat**2

def build_dm_eos_table(nhat_min=DM_NHAT_MIN, nhat_max=DM_NHAT_MAX, npts=DM_NTAB):
    nhat_grid = np.logspace(np.log10(nhat_min), np.log10(nhat_max), npts)
    p_hat = np.empty_like(nhat_grid)
    eps_hat = np.empty_like(nhat_grid)
    for i, nh in enumerate(nhat_grid):
        p_hat[i] = dm_p_hat_of_nhat(nh)
        eps_hat[i] = dm_eps_hat_of_nhat(nh)
    p_tilde = p_hat * GEV4_TO_TOV_DIMLESS
    eps_tilde = eps_hat * GEV4_TO_TOV_DIMLESS
    mask = np.isfinite(p_tilde) & np.isfinite(eps_tilde) & (p_tilde >= 0.0) & (eps_tilde >= 0.0)
    p_tilde = p_tilde[mask]
    eps_tilde = eps_tilde[mask]
    order = np.argsort(p_tilde)
    p_tilde = p_tilde[order]
    eps_tilde = eps_tilde[order]
    keep = np.ones(len(p_tilde), dtype=bool)
    keep[1:] = np.diff(p_tilde) > 0.0
    p_tilde = p_tilde[keep]
    eps_tilde = eps_tilde[keep]
    return nhat_grid, p_tilde, eps_tilde

DM_nhat_grid, DM_P_TILDE_TAB, DM_EPS_TILDE_TAB = build_dm_eos_table()
DM_PMIN = float(DM_P_TILDE_TAB[0])
DM_PMAX = float(DM_P_TILDE_TAB[-1])

DM_eps_of_p_interp = interp1d(
    DM_P_TILDE_TAB,
    DM_EPS_TILDE_TAB,
    kind="linear",
    bounds_error=False,
    fill_value=(DM_EPS_TILDE_TAB[0], DM_EPS_TILDE_TAB[-1])
)

def dm_eps_of_p(ptilde):
    if ptilde <= PTOL:
        return 0.0
    p = max(float(ptilde), DM_PMIN)
    return float(DM_eps_of_p_interp(p))

def dm_eps_array(parr):
    parr = np.asarray(parr, dtype=float)
    out = np.zeros_like(parr)
    mask = parr > PTOL
    if np.any(mask):
        pp = np.maximum(parr[mask], DM_PMIN)
        out[mask] = DM_eps_of_p_interp(pp)
    return out

# ============================================================
# Two-fluid f(R) RHS
# ============================================================
def tov_twofluid_fr_rhs(r, y, alpha):
    m, pB, pD, R, Rp = y

    pB = max(pB, 0.0)
    pD = max(pD, 0.0)
    epsB = baryon_eps_of_p(pB) if pB > PTOL else 0.0
    epsD = dm_eps_of_p(pD) if pD > PTOL else 0.0

    p = pB + pD
    eps = epsB + epsD

    if p <= PTOL and eps <= PTOL:
        return np.array([0.0, 0.0, 0.0, Rp, 0.0])

    r2m = r - 2.0 * m
    if r <= 0.0 or r2m <= 0.0:
        return np.full(5, np.nan)

    xi = 1.0 - 2.0 * m / r
    den = 1.0 + alpha * (2.0 * R + r * Rp)
    if abs(den) < 1.0e-14:
        return np.full(5, np.nan)

    common_num = (
        4.0 * np.pi * r**3 * p
        + m
        + alpha * R * r * (r**2 * R / 4.0 + 1.0)
        - alpha * (2.0 * r * Rp + R) * (r - 2.0 * m)
    )
    common_den = r * (r - 2.0 * m) * den
    grad_common = common_num / common_den

    dpBdr = -(pB + epsB) * grad_common if pB > PTOL else 0.0
    dpDdr = -(pD + epsD) * grad_common if pD > PTOL else 0.0
    dpdr = dpBdr + dpDdr

    fac = 1.0 + 2.0 * alpha * R
    C1 = Rp / (r - 2.0 * m) - fac / (2.0 * alpha * r * (r - 2.0 * m))
    C2 = (
        -Rp * (r - 3.0 * m) / (r * (r - 2.0 * m))
        + fac / (2.0 * alpha) * (
            -3.0 * dpdr / (r * max(p + eps, 1.0e-30))
            + (-3.0 * m - 0.5 * r**3 * R) / (r**2 * (r - 2.0 * m))
        )
    )

    brR = r**2 * R / 4.0 + 1.0
    cRp = 2.0 * r * Rp + R
    Anum = 4.0 * np.pi * r**2 * eps - m / r - alpha * R * brR + alpha * xi * cRp
    Bcoef = alpha * xi * r**2
    denom_m = den - Bcoef * C1
    if abs(denom_m) < 1.0e-14:
        return np.full(5, np.nan)

    dmdr = (Anum + Bcoef * C2 + den * m / r) / denom_m
    Rpp = C1 * dmdr + C2

    out = np.array([dmdr, dpBdr, dpDdr, Rp, Rpp], dtype=float)
    return out if np.all(np.isfinite(out)) else np.full(5, np.nan)

# ============================================================
# Two-fluid GR RHS (alpha = 0)
# ============================================================
def tov_twofluid_gr_rhs(r, y):
    m, pB, pD = y

    pB = max(pB, 0.0)
    pD = max(pD, 0.0)
    epsB = baryon_eps_of_p(pB) if pB > PTOL else 0.0
    epsD = dm_eps_of_p(pD) if pD > PTOL else 0.0

    p = pB + pD
    eps = epsB + epsD

    if p <= PTOL and eps <= PTOL:
        return np.array([0.0, 0.0, 0.0])

    r2m = r - 2.0 * m
    if r <= 0.0 or r2m <= 0.0:
        return np.full(3, np.nan)

    dmdr = 4.0 * np.pi * r**2 * eps
    grad_common = (m + 4.0 * np.pi * r**3 * p) / (r * (r - 2.0 * m))
    dpBdr = -(pB + epsB) * grad_common if pB > PTOL else 0.0
    dpDdr = -(pD + epsD) * grad_common if pD > PTOL else 0.0

    out = np.array([dmdr, dpBdr, dpDdr], dtype=float)
    return out if np.all(np.isfinite(out)) else np.full(3, np.nan)

# ============================================================
# Events
# ============================================================
def event_pB_zero_fr(r, y, alpha):
    return y[1] - PTOL
event_pB_zero_fr.terminal = False
event_pB_zero_fr.direction = -1

def event_pD_zero_fr(r, y, alpha):
    return y[2] - PTOL
event_pD_zero_fr.terminal = False
event_pD_zero_fr.direction = -1

def event_pall_zero_fr(r, y, alpha):
    return max(y[1], 0.0) + max(y[2], 0.0) - PTOL
event_pall_zero_fr.terminal = True
event_pall_zero_fr.direction = -1

def event_horizon_fr(r, y, alpha):
    return 1.0 - 2.0 * y[0] / r
event_horizon_fr.terminal = True
event_horizon_fr.direction = -1

def event_pB_zero_gr(r, y):
    return y[1] - PTOL
event_pB_zero_gr.terminal = False
event_pB_zero_gr.direction = -1

def event_pD_zero_gr(r, y):
    return y[2] - PTOL
event_pD_zero_gr.terminal = False
event_pD_zero_gr.direction = -1

def event_pall_zero_gr(r, y):
    return max(y[1], 0.0) + max(y[2], 0.0) - PTOL
event_pall_zero_gr.terminal = True
event_pall_zero_gr.direction = -1

def event_horizon_gr(r, y):
    return 1.0 - 2.0 * y[0] / r
event_horizon_gr.terminal = True
event_horizon_gr.direction = -1

# ============================================================
# Solver for one star
# ============================================================
# ============================================================
# Solver for one star
# ============================================================
def solve_star_twofluid(pBc_tilde, pDc_tilde, alpha_tilde,
                        rstart=RSTART, rend=REND, rtol=RTOL, atol=ATOL):

    if alpha_tilde == 0.0:
        y0 = [0.0, pBc_tilde, pDc_tilde]
        sol = solve_ivp(
            tov_twofluid_gr_rhs,
            (rstart, rend),
            y0,
            events=[event_pB_zero_gr, event_pD_zero_gr, event_pall_zero_gr, event_horizon_gr],
            method="RK45",
            rtol=rtol,
            atol=atol,
            max_step=MAXSTEP
        )

        r = sol.t
        m = sol.y[0]
        pB = np.maximum(sol.y[1], 0.0)
        pD = np.maximum(sol.y[2], 0.0)
        epsB = baryon_eps_array(pB)
        epsD = dm_eps_array(pD)
        p = pB + pD
        eps = epsB + epsD
        R = -8.0 * np.pi * (eps - 3.0 * p)
        Rp = np.gradient(R, r, edge_order=2) if len(r) >= 3 else np.zeros_like(r)

    else:
        epsBc = baryon_eps_of_p(pBc_tilde)
        epsDc = dm_eps_of_p(pDc_tilde)
        R0 = -8.0 * np.pi * (epsBc + epsDc - 3.0 * pBc_tilde - 3.0 * pDc_tilde)
        Rp0 = 0.0
        y0 = [0.0, pBc_tilde, pDc_tilde, R0, Rp0]
        sol = solve_ivp(
            tov_twofluid_fr_rhs,
            (rstart, rend),
            y0,
            args=(alpha_tilde,),
            events=[event_pB_zero_fr, event_pD_zero_fr, event_pall_zero_fr, event_horizon_fr],
            method="Radau",
            rtol=rtol,
            atol=atol,
            max_step=MAXSTEP
        )

        r = sol.t
        m = sol.y[0]
        pB = np.maximum(sol.y[1], 0.0)
        pD = np.maximum(sol.y[2], 0.0)
        R = sol.y[3]
        Rp = sol.y[4]

    if sol.status < 0 or len(r) == 0:
        raise RuntimeError(f"ODE integration failed: {sol.message}")

    def first_event(idx):
        tev = sol.t_events[idx]
        return tev[0] if len(tev) > 0 else sol.t[-1]

    rB = first_event(0)
    rD = first_event(1)
    rstar = max(rB, rD)

    jB = np.searchsorted(r, rB, side="left")
    jB = min(max(jB, 0), len(r) - 1)

    jD = np.searchsorted(r, rD, side="left")
    jD = min(max(jD, 0), len(r) - 1)

    jstar = np.searchsorted(r, rstar, side="left")
    jstar = min(max(jstar, 0), len(r) - 1)

    mstar = m[jstar]
    Rstar = R[jstar]
    Rpstar = Rp[jstar]
    exterior_slice = slice(jstar, len(r))
    R_ext = R[exterior_slice]

    bad_exterior = (
        (mstar <= 0.0) or
        (not np.isfinite(mstar)) or
        np.any(~np.isfinite(R_ext))
    )

    if bad_exterior:
        raise RuntimeError("Unphysical exterior branch selected")

    return {
        "sol": sol,
        "r": r,
        "m": m,
        "pB": pB,
        "pD": pD,
        "R": R,
        "Rp": Rp,
        "epsB": baryon_eps_array(pB),
        "epsD": dm_eps_array(pD),
        "RB_km": rtokm(rB),
        "RD_km": rtokm(rD),
        "Rvis_km": rtokm(rB),
        "Rtot_km": rtokm(rstar),
        "Mtot_solar": mtosolar(mstar),
        "Mstar_tilde": mstar,
        "Rstar_tilde": rstar,
        "RB_index": jB,
        "RD_index": jD,
        "Rstar_index": jstar,
        "Rstar_tildeRicci": Rstar,
        "Rpstar_tildeRicci": Rpstar,
    }

# ============================================================
# Sweep sequence
# ============================================================
def run_sequence(alpha_tilde, delta_central=DELTA_CENTRAL):
    pBc_km2_vals = np.logspace(np.log10(PCMINKM2), np.log10(PCMAXKM2), NPC)
    results = []
    for pBc_km2 in pBc_km2_vals:
        pDc_km2 = delta_central * pBc_km2
        pBc_tilde = ptodimless(pBc_km2)
        pDc_tilde = ptodimless(pDc_km2)
        try:
            star = solve_star_twofluid(pBc_tilde, pDc_tilde, alpha_tilde)
            if np.isfinite(star["Mtot_solar"]) and star["Mtot_solar"] > 0.0:
                star["pBc_km2"] = pBc_km2
                star["pDc_km2"] = pDc_km2
                star["pBc_tilde"] = pBc_tilde
                star["pDc_tilde"] = pDc_tilde
                results.append(star)
        except Exception as e:
            print(f"FAILED for pBc = {pBc_km2:.3e} km^-2 : {repr(e)}")
    return results

# ============================================================
# Plot helpers
# ============================================================
def make_cmap_norm(results):
    logpc = np.log10([r["pBc_km2"] for r in results])
    norm = mcolors.Normalize(vmin=np.min(logpc), vmax=np.max(logpc))
    smap = cm.ScalarMappable(norm=norm, cmap=CMAPNAME)
    smap.set_array([])
    return smap, norm, logpc

def style_ax(ax, title, xlabel, ylabel):
    ax.set_title(title, pad=6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, ls="--", lw=0.7, alpha=0.3)
    ax.tick_params(which="both", direction="in", top=True, right=True)

# ============================================================
# Plot 1: sample profiles
# ============================================================
def plot_sample_profiles(results, alpha_km2):
    if len(results) == 0:
        return
    imax = int(np.argmax([r["Mtot_solar"] for r in results]))
    s = results[imax]
    rkm = rtokm(s["r"])
    msol = mtosolar(s["m"])
    pBPa = pdimlesstoPa(s["pB"]) / 1.0e34
    pDPa = pdimlesstoPa(s["pD"]) / 1.0e34
    Rkm2 = riccitokm2(s["R"])

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.8))
    axs[0].plot(rkm, msol, color="#1f77b4")
    axs[0].axvline(s["RB_km"], color="tab:green", ls="--", lw=1.0, label=r"$R_B$")
    axs[0].axvline(s["RD_km"], color="tab:red", ls="--", lw=1.0, label=r"$R_D$")
    style_ax(axs[0], r"Mass profile", r"$r$ [km]", r"$m(r)$ [$M_\odot$]")
    axs[0].legend()

    axs[1].plot(rkm, np.maximum(pBPa, 1e-20), label=r"$p_B$", color="tab:green")
    axs[1].plot(rkm, np.maximum(pDPa, 1e-20), label=r"$p_D$", color="tab:red")
    axs[1].set_yscale("log")
    style_ax(axs[1], r"Pressure profiles", r"$r$ [km]", r"$p(r)$ [$10^{34}$ Pa]")
    axs[1].legend()

    axs[2].plot(rkm, Rkm2, color="tab:purple")
    axs[2].axhline(0.0, color="gray", ls="--", lw=0.8)
    style_ax(axs[2], r"Ricci profile", r"$r$ [km]", r"$\tilde R/r_b^2$ [km$^{-2}$]")

    fig.suptitle(
        rf"Two-fluid $f(R)=R+\alpha R^2$ ; $\alpha={alpha_km2:.3f}$ km$^2$ ; "
        rf"$\delta=p_{{Dc}}/p_{{Bc}}={DELTA_CENTRAL:.3f}$",
        y=1.02
    )
    fig.tight_layout()
    plt.show()

# ============================================================
# Plot 2: M-R sequence
# ============================================================
def plot_mass_radius(results, alpha_km2):
    if len(results) == 0:
        return
    smap, norm, logpc = make_cmap_norm(results)
    cmap = plt.get_cmap(CMAPNAME)
    M = np.array([r["Mtot_solar"] for r in results])
    Rvis = np.array([r["Rvis_km"] for r in results])
    Rtot = np.array([r["Rtot_km"] for r in results])

    fig, axs = plt.subplots(1, 2, figsize=(12.5, 5.0))
    for i in range(len(results) - 1):
        col = cmap(norm(0.5 * (logpc[i] + logpc[i+1])))
        axs[0].plot(Rvis[i:i+2], M[i:i+2], color=col, lw=2.0)
    sc0 = axs[0].scatter(Rvis, M, c=logpc, cmap=CMAPNAME, norm=norm, s=28, zorder=5)
    style_ax(axs[0], r"$M_{\rm tot}$ vs $R_B$", r"$R_B$ [km]", r"$M_{\rm tot}$ [$M_\odot$]")

    for i in range(len(results) - 1):
        col = cmap(norm(0.5 * (logpc[i] + logpc[i+1])))
        axs[1].plot(Rtot[i:i+2], M[i:i+2], color=col, lw=2.0)
    axs[1].scatter(Rtot, M, c=logpc, cmap=CMAPNAME, norm=norm, s=28, zorder=5)
    style_ax(axs[1], r"$M_{\rm tot}$ vs $R_{\rm tot}$", r"$R_{\rm tot}$ [km]", r"$M_{\rm tot}$ [$M_\odot$]")

    fig.suptitle(
        rf"Two-fluid sequence in $f(R)=R+\alpha R^2$ ; $\alpha={alpha_km2:.3f}$ km$^2$ ; "
        rf"$\delta=p_{{Dc}}/p_{{Bc}}={DELTA_CENTRAL:.3f}$",
        y=1.02
    )
    fig.tight_layout(rect=[0.0, 0.0, 0.92, 1.0])
    cax = fig.add_axes([0.94, 0.15, 0.018, 0.70])
    cb = fig.colorbar(sc0, cax=cax)
    cb.set_label(r"$\log_{10}(p_{Bc}\,[{\rm km}^{-2}])$")
    plt.show()

# ============================================================
# Multi-alpha comparison plots
# ============================================================
def run_alpha_grid(alpha_km2_list, delta_central=DELTA_CENTRAL):
    all_results = {}
    for alpha_km2 in alpha_km2_list:
        print("-" * 70)
        print(f"Running sequence for alpha = {alpha_km2:.3f} km^2")
        alpha_tilde = alphatodimless(alpha_km2)
        results = run_sequence(alpha_tilde, delta_central=delta_central)
        all_results[alpha_km2] = results

        if len(results) == 0:
            print("No valid models found.")
        else:
            M = np.array([r["Mtot_solar"] for r in results])
            Rvis = np.array([r["Rvis_km"] for r in results])
            imax = int(np.argmax(M))
            print(f"Models solved = {len(results)} / {NPC}")
            print(f"Mmax = {M[imax]:.6f} Msun at R_B = {Rvis[imax]:.6f} km")

    print("-" * 70)
    return all_results


def plot_mass_radius_multi_alpha(all_results, use_total_radius=False):
    fig, ax = plt.subplots(figsize=(7.4, 5.6))

    alpha_vals = list(all_results.keys())
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(alpha_vals)))

    radius_key = "Rtot_km" if use_total_radius else "Rvis_km"
    xlabel = r"$R_{\rm tot}$ [km]" if use_total_radius else r"$R_B$ [km]"

    for alpha_km2, col in zip(alpha_vals, colors):
        results = all_results[alpha_km2]
        if len(results) == 0:
            continue

        M = np.array([r["Mtot_solar"] for r in results], dtype=float)
        R = np.array([r[radius_key] for r in results], dtype=float)

        mask = np.isfinite(M) & np.isfinite(R) & (M > 0.0) & (R > 0.0)
        M = M[mask]
        R = R[mask]

        if len(M) == 0:
            continue

        label = rf"$\alpha = {alpha_km2:g}\,{{\rm km}}^2$"
        ax.plot(R, M, color=col, lw=2.1, label=label)

        imax = int(np.argmax(M))
        ax.scatter(R[imax], M[imax], color=col, s=36, edgecolor="k", zorder=5)

    title = rf"Mass-radius relation for different $\alpha$ ; $\delta = p_{{Dc}}/p_{{Bc}} = {DELTA_CENTRAL:.3f}$"
    style_ax(ax, title, xlabel, r"$M_{\rm tot}$ [$M_\odot$]")
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.show()

def plot_enclosed_mass_multi_alpha(all_results, use_total_radius=False):
    fig, ax = plt.subplots(figsize=(7.4, 5.6))

    alpha_vals = list(all_results.keys())
    colors = plt.cm.plasma(np.linspace(0.08, 0.92, len(alpha_vals)))

    for alpha_km2, col in zip(alpha_vals, colors):
        results = all_results[alpha_km2]
        if len(results) == 0:
            continue

        Mseq = np.array([r["Mtot_solar"] for r in results], dtype=float)
        if len(Mseq) == 0:
            continue

        imax = int(np.argmax(Mseq))
        s = results[imax]

        if use_total_radius:
            j = s["Rstar_index"] + 1
            r_surface = s["Rtot_km"]
        else:
            j = s["RB_index"] + 1
            r_surface = s["RB_km"]

        rkm = rtokm(s["r"][:j])
        msol = mtosolar(s["m"][:j])

        mask = np.isfinite(rkm) & np.isfinite(msol)
        rkm = rkm[mask]
        msol = msol[mask]

        if len(rkm) == 0:
            continue

        label = rf"$\alpha = {alpha_km2:g}\,{{\rm km}}^2$"
        ax.plot(rkm, msol, color=col, lw=2.1, label=label)
        ax.scatter(rkm[-1], msol[-1], color=col, s=35, edgecolor="k", zorder=5)
        ax.axvline(r_surface, color=col, ls="--", lw=0.9, alpha=0.7)

    surface_text = r"$R_{\rm tot}$" if use_total_radius else r"$R_B$"
    title = rf"Enclosed mass profiles up to {surface_text} for different $\alpha$ ; $\delta = p_{{Dc}}/p_{{Bc}} = {DELTA_CENTRAL:.3f}$"
    style_ax(ax, title, r"$r$ [km]", r"$m(r)$ [$M_\odot$]")
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.show()
# ============================================================
# Summary
# ============================================================
def print_summary(results, alpha_km2):
    print("=" * 70)
    print("Two-fluid f(R) sequence")
    print(f"alpha = {alpha_km2:.6f} km^2")
    print(f"alpha_tilde = {alphatodimless(alpha_km2):.6e}")
    print(f"delta = pDc/pBc = {DELTA_CENTRAL:.6f}")
    print(f"DM model: m = {DM_MASS_GEV:.3f} GeV, c = {DM_C_GEV_INV:.3f} GeV^-1, Mref = {DM_MREF_GEV:.3f} GeV")
    print(f"DM pressure table range = [{DM_PMIN:.3e}, {DM_PMAX:.3e}] (TOV dimensionless)")
    print(f"Models solved = {len(results)} / {NPC}")
    if len(results) == 0:
        print("No valid solutions.")
        print("=" * 70)
        return
    M = np.array([r["Mtot_solar"] for r in results])
    Rvis = np.array([r["Rvis_km"] for r in results])
    Rtot = np.array([r["Rtot_km"] for r in results])
    imax = int(np.argmax(M))
    print(f"Mmax = {M[imax]:.6f} Msun")
    print(f"R_B at Mmax = {Rvis[imax]:.6f} km")
    print(f"R_tot at Mmax = {Rtot[imax]:.6f} km")
    print(f"pc_B at Mmax = {results[imax]['pBc_km2']:.6e} km^-2")
    print(f"pc_D at Mmax = {results[imax]['pDc_km2']:.6e} km^-2")
    print("=" * 70)

# ============================================================
# Main
# ============================================================
# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    alpha_list_km2 = [0.0, 0.5, 1.0, 1.5, 2.0]

    print("=" * 70)
    print("Running two-fluid TOV solver in f(R)=R+alpha R^2")
    print(f"rb = {rbkm:.6f} km")
    print(f"alpha list = {alpha_list_km2} km^2")
    print(f"delta = pDc/pBc = {DELTA_CENTRAL:.6f}")
    print(f"pc_B sweep = [{PCMINKM2:.3e}, {PCMAXKM2:.3e}] km^-2 with {NPC} points")
    print(f"DM model: m = {DM_MASS_GEV:.3f} GeV, c = {DM_C_GEV_INV:.3f} GeV^-1, Mref = {DM_MREF_GEV:.3f} GeV")
    print(f"DM pressure table range = [{DM_PMIN:.3e}, {DM_PMAX:.3e}] (TOV dimensionless)")
    print("=" * 70)

    all_results = run_alpha_grid(alpha_list_km2, delta_central=DELTA_CENTRAL)

    # Mass-radius comparison using visible radius R_B
    plot_mass_radius_multi_alpha(all_results, use_total_radius=False)

    # If you also want the total radius version, uncomment this:
    # plot_mass_radius_multi_alpha(all_results, use_total_radius=True)

    # Enclosed mass profile comparison for the maximum-mass star from each alpha-sequence
    plot_enclosed_mass_multi_alpha(all_results)
