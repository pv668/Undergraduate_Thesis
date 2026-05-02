
"""
Created on Fri Apr 10 14:28:08 2026

@author: prakh
"""

"""
TOV Equations Solver for f(R) = R + alpha*R^2 Gravity   [VERSION 4]
=====================================================================
Reference: Nishiwaki (2026) – "A brief note on the TOV equations in f(R) = R + alphaR^2"
 
WHAT THIS VERSION DOES
───────────────────────
Instead of varying alpha, this version fixes alpha at a single constant value
and sweeps the central pressure p_c (the initial boundary condition, Eq. 1.25).
Each value of p_c produces one neutron-star solution. Collecting all solutions
traces out the Mass-Radius (M-R) curve for that gravity model.
 
Five figures are produced (all appear in Spyder's Plots pane):
 
  Figure 1  –  m(r) profiles [M_sun vs km]  for all p_c values
  Figure 2  –  p(r) and ε(r) profiles [10^34 Pa vs km]
  Figure 3  –  Ricci scalar R(r) profile [km⁻²] and dR/dr(r) [km⁻³]
  Figure 4  –  M-R diagram: every (R_star, M_star) point connected into a curve
  Figure 5  –  Summary 2×2: m(r) | p(r) | M-R | R_Ricci(r)
 
All profile plots use a sequential colourmap (plasma) so each p_c value gets
its own colour; a shared colourbar shows the mapping.  This style is distinctly
different from the reference images while remaining publication-quality.
 
═══════════════════════════════════════════════════════════════════════════════
HOW TO CHANGE SETTINGS  ←  read this
═══════════════════════════════════════════════════════════════════════════════
 
▸ Fixed alpha [km²]:
    Change ALPHA_FIXED_KM2.  Set to 0.0 for standard GR.
    α̃ = ALPHA_FIXED_KM2 / r_b_km²   (r_b_km ≈ 2.954 km)
 
▸ Central pressure sweep [km⁻²]:
    P_C_MIN_KM2, P_C_MAX_KM2 — log-spaced endpoints of the p_c scan.
    N_PC          — number of p_c values (= number of stars solved).
    Nuclear saturation ≈ 1e-5 km⁻².  Reasonable range: 1e-6 to 1e-3 km⁻².
 
▸ EOS:
    GAMMA_DEFAULT — polytropic index Γ.  2.0=soft, 2.5=medium, 3.0=stiff.
    K_TILDE       — polytropic constant (auto-calibrated; override if needed).
 
    To replace with a multi-layer EOS:  replace the body of eos_eps() only.
    All ODE and plotting code is EOS-agnostic.  See the HOW TO REPLACE section.
 
▸ ODE tolerances:
    In solve_star(): rtol=1e-6, atol=1e-8.  Tighten for higher accuracy.
 
▸ Colourmap:
    CMAP_NAME — any matplotlib colourmap string, e.g. 'viridis', 'plasma',
    'inferno', 'cividis', 'coolwarm'.
 
═══════════════════════════════════════════════════════════════════════════════
HOW TO REPLACE THE EOS  (ready for multi-layer extension)
═══════════════════════════════════════════════════════════════════════════════
 
The EOS is entirely contained in two small functions:
 
    eos_eps(p_tilde)         → scalar   (called inside ODE)
    eos_eps_array(p_arr)     → ndarray  (called for plotting)
 
Both must map dimensionless pressure p̃ → dimensionless energy density ε̃.
 
  Option A – change Γ and/or K̃:
      Edit GAMMA_DEFAULT and K_TILDE in CONFIG.
 
  Option B – piecewise polytrope (two-layer approximation):
      def eos_eps(p_tilde, *args, **kwargs):
          p_t = p_to_dimless(3e-6)   # transition pressure in km⁻²
          K1, G1 = 8000.0, 1.35      # crust layer
          K2, G2 = 344.27, 2.5       # core layer
          if p_tilde <= p_t:
              return (max(p_tilde, 1e-30) / K1) ** (1.0 / G1)
          return (max(p_tilde, 1e-30) / K2) ** (1.0 / G2)
 
      def eos_eps_array(p_arr, *args, **kwargs):
          return np.array([eos_eps(p) for p in p_arr])
 
  Option C – tabulated EOS (CompOSE, LORENE, etc.):
      import numpy as np
      from scipy.interpolate import interp1d
      _data     = np.loadtxt('eos_table.dat')   # cols: rho [kg/m³], p [Pa]
      _scale    = G_SI / c_SI**4 * r_b**2
      _p_d      = _data[:, 1] * _scale
      _eps_d    = _data[:, 0] * c_SI**2 * _scale
      _eos_tab  = interp1d(_p_d, _eps_d, kind='cubic',
                           bounds_error=False,
                           fill_value=(_eps_d[0], _eps_d[-1]))
 
      def eos_eps(p_tilde, *args, **kwargs):
          return float(_eos_tab(max(p_tilde, _p_d[0])))
 
      def eos_eps_array(p_arr, *args, **kwargs):
          return _eos_tab(np.maximum(p_arr, _p_d[0]))
 
  In every case: the ODE solver, boundary conditions, unit conversions,
  and all plotting functions require zero changes.
"""
 
# ══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ── Spyder inline Plots pane ──────────────────────────────────────────────────
# In Spyder: run as-is. The IPython kernel sets the backend to 'inline'
# automatically, so every plt.show() renders in the Plots pane.
# From a plain terminal: uncomment the two lines below.
#   import matplotlib
#   matplotlib.use('Qt5Agg')   # or 'TkAgg'
# ══════════════════════════════════════════════════════════════════════════════
import numpy as np
from scipy.integrate import solve_ivp
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.cm as cm
 
plt.rcParams.update({
    'figure.dpi'        : 110,
    'font.size'         : 10,
    'axes.titlesize'    : 11,
    'axes.labelsize'    : 10,
    'legend.fontsize'   : 8.5,
    'xtick.labelsize'   : 9,
    'ytick.labelsize'   : 9,
    'lines.linewidth'   : 1.8,
    'axes.spines.top'   : True,
    'axes.spines.right' : True,
})
 
# ══════════════════════════════════════════════════════════════════════════════
# PHYSICAL CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
G_SI  = 6.67430e-11        # m³ kg⁻¹ s⁻²
c_SI  = 2.99792458e8       # m s⁻¹
M_sun = 1.989e30           # kg
 
# Reference length = Schwarzschild radius of the Sun ≈ 2.954 km
# By construction: 1 M_sun ↔ m̃ = 0.5   (see Nishiwaki notes §1.3)
r_b    = 2.0 * G_SI * M_sun / c_SI**2   # metres
r_b_km = r_b / 1e3                       # km ≈ 2.954
G_c2   = G_SI / c_SI**2                  # m kg⁻¹
 
# ══════════════════════════════════════════════════════════════════════════════
# CONFIG  ←  EDIT HERE
# ══════════════════════════════════════════════════════════════════════════════
 
# ── Fixed gravity parameter ───────────────────────────────────────────────────
ALPHA_FIXED_KM2 = 2.0      # alpha in km²  (set 0.0 for standard GR)
#                            α̃ = ALPHA_FIXED_KM2 / r_b_km²
 
# ── Central pressure sweep (Eq. 1.25 boundary condition) ─────────────────────
P_C_MIN_KM2 = 3.0e-6      # minimum p_c  [km⁻²]  (low-mass, large-radius stars)
P_C_MAX_KM2 = 8.0e-3      # maximum p_c  [km⁻²]  (maximum-mass region)
N_PC        = 50          # number of p_c values to sweep
 
# ── EOS parameters ───────────────────────────────────────────────────────────
GAMMA_DEFAULT = 2.5         # polytropic index
# K_TILDE auto-calibrated below.  To override: uncomment and set manually.
# K_TILDE = 344.27
 
# ── Smoothing for M-R curve ───────────────────────────────────────────────────
SMOOTH_WIN = 3              # running-average window; 1 = no smoothing
 
# ── Colourmap ─────────────────────────────────────────────────────────────────
CMAP_NAME = 'coolwarm'        # try: 'viridis', 'inferno', 'cividis', 'coolwarm'
 
# ══════════════════════════════════════════════════════════════════════════════
# UNIT HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def alpha_to_dimless(alpha_km2):
    """Physical α [km²] → dimensionless α̃."""
    return alpha_km2 / r_b_km**2
 
def p_to_dimless(p_km2):
    """Pressure [km⁻²] → dimensionless p̃ = p_km2 * r_b_km²."""
    return p_km2 * r_b_km**2
 
def p_dimless_to_Pa(p_tilde):
    """Dimensionless p̃ → SI pressure [Pa]."""
    return p_tilde / (G_SI / c_SI**4 * r_b**2)
 
def eps_dimless_to_Pa(eps_tilde):
    """Dimensionless ε̃ → SI energy density [Pa = J/m³]."""
    return eps_tilde / (G_SI / c_SI**4 * r_b**2)
 
def m_to_solar(m_tilde):
    """Dimensionless m̃ → solar masses.  (1 M_sun ↔ m̃ = 0.5)"""
    return m_tilde / 0.5
 
def r_to_km(r_tilde):
    """Dimensionless r̃ → km."""
    return r_tilde * r_b_km
 
def R_ricci_to_km2(R_tilde):
    """Dimensionless R̃ → km⁻²."""
    return R_tilde / r_b_km**2
 
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# EOS — 7-layer piecewise polytrope  "BSR2"
#   from Sec. 2.4 of Kenji Nishiwaki's extended notes (Apr 2026).
#
# ── What replaces Γ and K̃? ────────────────────────────────────────────────
#   In the old single-polytrope EOS there was one global Γ and one K̃.
#   BSR2 uses SEVEN pairs (Γ_i, K_i) for i = 0…6, one per density interval.
#   Layer 0–3 describe the crust, layers 4–6 the core.
#   K_i are NOT free parameters — they are fixed by pressure continuity
#   across boundaries (Eq. 2.17 in the notes), so the only primary inputs are:
#       K_0  (one free constant, log₁₀(K₀) = 12.4812)
#       Γ_i  (seven adiabatic indices, given by the BSR2 fit)
#       ρ_i  (six boundary mass densities, given by the BSR2 fit)
#   The coefficients a_i (Eq. 2.19) enforce energy-density continuity.
#   GAMMA_DEFAULT and K_TILDE are kept as np.nan so nothing else in the
#   existing code breaks (they only appear in labels; see suptitle patches).
#
# ── Unit conventions ─────────────────────────────────────────────────────
#   Input data: ρ  in  g cm⁻³,  p  in  dyn cm⁻²  (CGS).
#   Solver:     dimensionless  (p̃, ε̃)  via  r_b  (Schwarzschild radius of Sun)
#                 p̃ = p × (G/c⁴) × r_b²
#                 ε̃ = ε × (G/c⁴) × r_b²   (c = 1 natural units ⟹ same factor)
#   ρ̃ = ρ × (G/c²) × r_b²                (mass density)
# ══════════════════════════════════════════════════════════════════════════════

# ---------- primary BSR2 fit parameters (Nishiwaki notes, Sec. 2.4.1) --------
#  log₁₀(K₀) = 12.4812   → K₀ in  dyn cm⁻²  /  (g cm⁻³)^Γ₀
#  Γ₀…Γ₆  (seven adiabatic indices)
#  ρ₁…ρ₆  (six boundary densities in g cm⁻³)
_BSR2_LOG10_K0   = 12.4812
_BSR2_GAMMAS     = np.array([1.6379, 1.3113, 0.8349,
                              1.3136, 3.2464, 2.8221, 2.3788], dtype=float)
_BSR2_LOG10_RHO  = np.array([6.9304, 11.3669, 12.7363,
                              14.0413, 14.8162, 14.9832], dtype=float)
_BSR2_RHO_B_CGS  = 10.0 ** _BSR2_LOG10_RHO   # ρ₁…ρ₆  [g cm⁻³]

# ---------- reference density (notes use ρ_ref = 2.5×10¹⁴ g cm⁻³) ----------
_RHO_REF_CGS = 2.5e14   # g cm⁻³   (nucleon saturation density from notes)

# ---------- CGS physical constants needed for unit conversion ----------------
_C_CGS = c_SI * 1e2          # speed of light  [cm s⁻¹]
# note: G_SI and c_SI are already defined above in the file

# ---------- recursive K_i from Eq. (2.17): K_{i} = K_{i-1} ρ_i^{Γ_{i-1}−Γ_i}
_K_CGS = np.zeros(7, dtype=float)
_K_CGS[0] = 10.0 ** _BSR2_LOG10_K0
for _i in range(1, 7):
    _K_CGS[_i] = _K_CGS[_i-1] * _BSR2_RHO_B_CGS[_i-1] ** (_BSR2_GAMMAS[_i-1] - _BSR2_GAMMAS[_i])

# ---------- recursive ε(ρ_i) and a_i from Eqs. (2.18), (2.19) ---------------
#  ε is stored in g cm⁻³ (same units as ρ), which is valid under c = 1.
#  The factor 1/c² converts the pressure term p [dyn cm⁻²] → [g cm⁻³].
_A_CGS    = np.zeros(7, dtype=float)    # a_0 = 0 (Eq. 2.16)
_EPS_B_CGS = np.zeros(7, dtype=float)  # ε(ρ_i) at lower boundary of layer i
for _i in range(1, 7):
    _rho_i = _BSR2_RHO_B_CGS[_i-1]
    # ε(ρ_i) evaluated with layer (i-1) EOS — Eq. (2.18)
    _eps_i = ((1.0 + _A_CGS[_i-1]) * _rho_i
              + _K_CGS[_i-1] * _rho_i**_BSR2_GAMMAS[_i-1]
              / ((_BSR2_GAMMAS[_i-1] - 1.0) * _C_CGS**2))
    _EPS_B_CGS[_i] = _eps_i
    # a_i from Eq. (2.19)
    _A_CGS[_i] = (_eps_i / _rho_i - 1.0
                  - _K_CGS[_i] * _rho_i**(_BSR2_GAMMAS[_i] - 1.0)
                  / ((_BSR2_GAMMAS[_i] - 1.0) * _C_CGS**2))

# ---------- pressure boundaries p̃_i for layer selection (Eq. 2.35) ----------
#  p_i [dyn cm⁻²] = K_{i-1} * ρ_i^{Γ_{i-1}},  p_0 = 0
_P_B_CGS   = np.zeros(7, dtype=float)
for _i in range(1, 7):
    _P_B_CGS[_i] = _K_CGS[_i-1] * _BSR2_RHO_B_CGS[_i-1]**_BSR2_GAMMAS[_i-1]

# ---------- conversion factors CGS → solver dimensionless --------------------
#  p_tilde = p_cgs * _P_TO_TILDE
#  eps_tilde = eps_cgs * _P_TO_TILDE   (same factor since c=1 ⟹ [ε]=[p])
_P_TO_TILDE     = 0.1 * G_SI / c_SI**4 * r_b**2   # dyn cm⁻² → dimensionless
_RHO_TO_TILDE   = 1e3  * G_SI / c_SI**2 * r_b**2  # g cm⁻³   → dimensionless

_P_B_TILDE      = _P_B_CGS * _P_TO_TILDE          # pressure boundaries in solver units
_RHO_REF_TILDE  = _RHO_REF_CGS * _RHO_TO_TILDE
_KPREF_TILDE    = _K_CGS * (_RHO_REF_CGS ** _BSR2_GAMMAS) * _P_TO_TILDE  # K̃_i (ρ̃_ref)^{Γ_i}

# Keep old names as NaN so existing label strings that reference them don't crash
GAMMA_DEFAULT = float('nan')
K_TILDE       = float('nan')
EOS_NAME      = 'BSR2 (7-layer piecewise polytrope)'

# ---------- helper: which layer does a given p̃ fall into? -------------------
def _bsr2_layer(p_tilde):
    """Return layer index i ∈ {0,…,6} such that p̃_i ≤ p̃ < p̃_{i+1}."""
    return int(np.clip(np.searchsorted(_P_B_TILDE[1:], float(p_tilde), side='right'), 0, 6))

# ── Replace ONLY these two functions to swap in a different EOS ──────────────
def eos_eps(p_tilde, gamma=None, k_tilde=None):
    """
    7-layer BSR2 dimensionless ε̃(p̃)  —  Eq. (2.34) of Kenji's notes.

    Each layer i uses:
        ε̃ = (1 + a_i) * ρ̃_ref * (p̃ / K̃_i(ρ̃_ref)^{Γ_i})^{1/Γ_i}
           + p̃ / (Γ_i − 1)
    The first term is the rest-mass + internal-energy contribution (proportional
    to the dimensionless mass density).  The second is the thermal/pressure term.
    Layer boundaries are determined by pressure via Eq. (2.35).
    """
    p = max(float(p_tilde), 1.0e-30)
    i = _bsr2_layer(p)
    Gi   = _BSR2_GAMMAS[i]
    ai   = _A_CGS[i]
    Kpr  = _KPREF_TILDE[i]          # K̃_i (ρ̃_ref)^{Γ_i}  [dimensionless]
    rho_ratio = (p / Kpr) ** (1.0 / Gi)   # (ρ̃/ρ̃_ref)
    return (1.0 + ai) * _RHO_REF_TILDE * rho_ratio  +  p / (Gi - 1.0)

def eos_eps_array(p_arr, gamma=None, k_tilde=None):
    """Vectorised BSR2 eos_eps over a numpy array (used for plotting)."""
    return np.array([eos_eps(p) for p in np.asarray(p_arr, dtype=float)], dtype=float)

# ── helper for the standalone BSR2 EOS plot (added at end of script) ─────────
def bsr2_p_rho_cgs():
    """
    Return arrays (rho_cgs, p_cgs) sampling the full BSR2 EOS in physical
    CGS units [g cm⁻³, dyn cm⁻²], suitable for a log-log EOS plot.
    """
    rho_min = 1e4       # well below ρ₁  (outer crust baseline)
    rho_max = 5e15      # above ρ₆  (deep core)
    rho = np.logspace(np.log10(rho_min), np.log10(rho_max), 2000)
    p   = np.empty_like(rho)
    for j, rh in enumerate(rho):
        # find which layer this ρ belongs to
        idx = int(np.searchsorted(_BSR2_RHO_B_CGS, rh, side='right'))
        idx = min(idx, 6)
        p[j] = _K_CGS[idx] * rh ** _BSR2_GAMMAS[idx]
    return rho, p

# ══════════════════════════════════════════════════════════════════════════════
 
 
# ══════════════════════════════════════════════════════════════════════════════
# ODE RIGHT-HAND SIDES  (Eqs. 1.31, 1.32, 1.33)
# State: y = [m̃, p̃, R̃, R̃']
# ══════════════════════════════════════════════════════════════════════════════
def tov_fR_rhs(r, y, alpha, gamma, k_tilde):
    m, p, R, Rp = y
    if p <= 1.0e-12:
        return [0.0, 0.0, Rp, 0.0]
    eps = eos_eps(p, gamma, k_tilde)
    xi  = 1.0 - 2.0 * m / r
    r2m = r - 2.0 * m
    if xi <= 0.0 or abs(r2m) < 1.0e-12:
        return [np.nan, np.nan, np.nan, np.nan]
    brR = r**2 * R / 4.0 + 1.0
    cRp = 2.0 * r * Rp + R
    den = 1.0 + alpha * (2.0 * R + r * Rp)
    if abs(den) < 1.0e-14:
        return [np.nan, np.nan, np.nan, np.nan]
    # Eq. 1.32 — dp/dr
    num_p = 4.0*np.pi*r**3*p + m + alpha*R*r*brR - alpha*cRp*r2m
    dp_dr = -(p + eps) * num_p / (r * r2m * den)
    # Eq. 1.33 — R'' = C1*(dm/dr) + C2
    fac   = (1.0 + 2.0*alpha*R) / (2.0*alpha)
    C1    = Rp/r2m - fac/(r*r2m)
    C2    = (fac*(-3.0*dp_dr/(r*(p+eps)) + (-3.0*m - r**3*R/2.0)/(r**2*r2m))
             + Rp*(r - 3.0*m)/(r*r2m))
    # Eq. 1.31 — dm/dr  (coupled system, solved algebraically)
    A_num   = 4.0*np.pi*r**2*eps - m/r - alpha*R*brR + alpha*xi*cRp
    B_coef  = alpha * xi * r**2
    denom_m = den - B_coef * C1
    if abs(denom_m) < 1.0e-14:
        return [np.nan, np.nan, np.nan, np.nan]
    dm_dr = (A_num + B_coef*C2 + (m/r)*den) / denom_m
    Rpp   = C1 * dm_dr + C2
    if not np.all(np.isfinite([dm_dr, dp_dr, Rp, Rpp])):
        return [np.nan, np.nan, np.nan, np.nan]
    return [dm_dr, dp_dr, Rp, Rpp]
 
def tov_gr_rhs(r, y, gamma, k_tilde):
    """Standard GR limit (α=0), Eqs. 1.15 & 1.18."""
    m, p = y
    if p <= 1.0e-12: return [0.0, 0.0]
    eps = eos_eps(p, gamma, k_tilde)
    if 1.0 - 2.0*m/r <= 0.0: return [np.nan, np.nan]
    rhs = [4.0*np.pi*r**2*eps,
           -(p+eps)*(4.0*np.pi*r**3*p + m)/(r*(r - 2.0*m))]
    if not np.all(np.isfinite(rhs)):
        return [np.nan, np.nan]
    return rhs
 
# Surface detection events
def _pzero_fR(r, y, alpha, gamma, k_tilde): return y[1] - 1.0e-10
_pzero_fR.terminal = True; _pzero_fR.direction = -1
 
def _pzero_gr(r, y, gamma, k_tilde): return y[1] - 1.0e-10
_pzero_gr.terminal = True; _pzero_gr.direction = -1

# Abort if a trapped surface / horizon is reached instead of silently freezing
def _horizon_fR(r, y, alpha, gamma, k_tilde): return 1.0 - 2.0*y[0]/r
_horizon_fR.terminal = True; _horizon_fR.direction = -1

def _horizon_gr(r, y, gamma, k_tilde): return 1.0 - 2.0*y[0]/r
_horizon_gr.terminal = True; _horizon_gr.direction = -1
 
# ══════════════════════════════════════════════════════════════════════════════
# SINGLE-STAR SOLVER  [FIXED: Radau solver + max_step]
# ══════════════════════════════════════════════════════════════════════════════
def solve_star(p_c_tilde, alpha_tilde,
               gamma=None, k_tilde=None,
               r_start=1.0e-4, r_end=150.0,
               rtol=1.0e-7, atol=1.0e-9):
    """
    Integrate the TOV system for a single neutron star.
 
    Parameters
    ──────────
    p_c_tilde  : float – dimensionless central pressure [boundary condition]
    alpha_tilde: float – dimensionless α̃ = alpha_km2 / r_b_km²
    gamma      : float – EOS polytropic index (default: GAMMA_DEFAULT)
    k_tilde    : float – EOS constant K̃ (default: K_TILDE)
    r_start    : float – inner boundary ε̃ (Eq. 1.25; default 1e-4)
    r_end      : float – outer integration limit (must exceed star radius)
    rtol, atol : float – ODE solver tolerances
 
    Returns
    ───────
    sol   : ODE result object (sol.t=r̃, sol.y[0]=m̃, sol.y[1]=p̃,
                               sol.y[2]=R̃, sol.y[3]=R̃')
    M_sun : float – total gravitational mass [M_sun]
    R_km  : float – stellar radius [km]
 
    FIX 1 — method='Radau' (was 'RK45'):
        The coupled (m̃, p̃, R̃, R̃') system is stiff, especially for large α̃
        or near the stellar surface where p → 0. Radau is an implicit
        Runge-Kutta solver designed for stiff problems. It takes fewer
        rejected steps and is far more stable than the explicit RK45.
 
    FIX 2 — max_step=0.02 (was unset):
        Without a cap, the adaptive solver can jump over the short-scale
        oscillations in the Ricci scalar R̃(r̃), producing aliased or
        artificially smooth R̃(r̃) profiles. 0.02 in r̃ ≈ 0.06 km with
        r_b ≈ 2.954 km, which safely resolves sub-km features.
    """
    if gamma is None:
        gamma = GAMMA_DEFAULT
    if k_tilde is None:
        k_tilde = K_TILDE
 
    if alpha_tilde == 0.0:
        # ── Standard GR fallback (Eqs. 1.15 & 1.18) ──────────────────────
        sol = solve_ivp(
            tov_gr_rhs,
            [r_start, r_end],
            [0.0, p_c_tilde],
            args=(gamma, k_tilde),
            events=(_pzero_gr, _horizon_gr),
            method='RK45',
            rtol=rtol,
            atol=atol,
            max_step=0.02,
        )
        dummy = np.zeros_like(sol.t)
        sol.y = np.vstack([sol.y, dummy, dummy])
 
    else:
        # ── Boundary conditions, Eq. 1.25 / 1.34 ─────────────────────────
        eps_c = eos_eps(p_c_tilde, gamma, k_tilde)
        T_c   = eps_c - 3.0 * p_c_tilde          # Eq. 1.7:  T = ε − 3p
        R0    = -8.0 * np.pi * T_c               # Eq. 1.23: R(ε) = −κT
        Rp0   = 0.0                               # Eq. 1.24: dR/dr|₀ = 0
        y0    = [0.0, p_c_tilde, R0, Rp0]
 
        sol = solve_ivp(
            tov_fR_rhs,
            [r_start, r_end],
            y0,
            args=(alpha_tilde, gamma, k_tilde),
            events=(_pzero_fR, _horizon_fR),
            method='Radau',
            rtol=rtol,
            atol=atol,
            max_step=0.02,
        )
 
    return sol, m_to_solar(sol.y[0, -1]), r_to_km(sol.t[-1])
 
# ══════════════════════════════════════════════════════════════════════════════
# SWEEP OVER p_c  — the main computation
# ══════════════════════════════════════════════════════════════════════════════
def run_sweep(alpha_tilde, gamma=None, k_tilde=None):
    """
    Solve one star per p_c value.
 
    Returns a list of dicts, one per successful solution:
        {'p_c_km2', 'p_c_tilde',
         'r_km', 'm_sol', 'p_Pa', 'eps_Pa', 'R_ricci_km2', 'Rp_ricci_km3',
         'M_total', 'R_total'}
    """
    if gamma   is None: gamma   = GAMMA_DEFAULT
    if k_tilde is None: k_tilde = K_TILDE
 
    pc_km2_vals = np.logspace(np.log10(P_C_MIN_KM2),
                              np.log10(P_C_MAX_KM2), N_PC)
    results = []
    for pc_km2 in pc_km2_vals:
        pc_tilde = p_to_dimless(pc_km2)
        try:
            sol, M_tot, R_tot = solve_star(pc_tilde, alpha_tilde,
                                           gamma=gamma, k_tilde=k_tilde)
            # ✅ FIXED — only reject obviously broken solutions
            if M_tot <= 0.0 or R_tot <= 0.0 or not np.isfinite(M_tot) or not np.isfinite(R_tot):
                continue
 
            r_km_arr      = r_to_km(sol.t)
            m_sol_arr     = m_to_solar(sol.y[0])
            p_Pa_arr      = p_dimless_to_Pa(sol.y[1]) / 1.0e34     # 10^34 Pa
            eps_tilde_arr = eos_eps_array(sol.y[1], gamma, k_tilde)
            eps_Pa_arr    = eps_dimless_to_Pa(eps_tilde_arr) / 1.0e34
 
            if alpha_tilde == 0.0:
                R_tilde = -8.0*np.pi*(eps_tilde_arr - 3.0*sol.y[1])
                Rp_tilde= np.gradient(R_tilde, sol.t)
            else:
                R_tilde  = sol.y[2]
                Rp_tilde = sol.y[3]
 
            R_ricci_km2_arr  = R_ricci_to_km2(R_tilde)
            Rp_ricci_km3_arr = R_ricci_to_km2(Rp_tilde) / r_b_km   # km⁻³
 
            results.append({
                'p_c_km2'        : pc_km2,
                'p_c_tilde'      : pc_tilde,
                'r_km'           : r_km_arr,
                'm_sol'          : m_sol_arr,
                'p_Pa'           : p_Pa_arr,
                'eps_Pa'         : eps_Pa_arr,
                'R_ricci_km2'    : R_ricci_km2_arr,
                'Rp_ricci_km3'   : Rp_ricci_km3_arr,
                'M_total'        : M_tot,
                'R_total'        : R_tot,
            })
        except Exception:
            pass
 
    print(f"  Solved {len(results)}/{N_PC} stars  "
          f"(α = {alpha_tilde/alpha_to_dimless(1.0):.2f} km²  if 1 km²={alpha_to_dimless(1):.4f})")
    return results
 
# ══════════════════════════════════════════════════════════════════════════════
# COLOUR HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _make_cmap_norm(results):
    """Return (ScalarMappable, Normalise) keyed on log10(p_c_km2)."""
    log_pc_vals = [np.log10(r['p_c_km2']) for r in results]
    norm = mcolors.Normalize(vmin=min(log_pc_vals), vmax=max(log_pc_vals))
    smap = cm.ScalarMappable(cmap=CMAP_NAME, norm=norm)
    smap.set_array([])
    return smap, norm, log_pc_vals
 
def _add_colorbar(fig, smap, label=r'$\log_{10}(p_c\ [\mathrm{km}^{-2}])$',
                  ax=None):
    """Attach a shared colourbar to the figure."""
    if ax is None:
        cbar = fig.colorbar(smap, ax=fig.axes, shrink=0.6, pad=0.02, aspect=30)
    else:
        cbar = fig.colorbar(smap, ax=ax, shrink=0.85, pad=0.03)
    cbar.set_label(label, fontsize=9)
    return cbar
 
def _style(ax, title='', xlabel='', ylabel='', grid_alpha=0.25):
    ax.set_title(title,   fontsize=10, fontweight='bold', pad=6)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.tick_params(which='both', direction='in', top=True, right=True)
    ax.grid(True, ls=':', lw=0.7, alpha=grid_alpha)
 
# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1  —  Mass profile  m(r)
# ══════════════════════════════════════════════════════════════════════════════
def fig1_mass_profiles(results, alpha_km2):
    print("[Figure 1]  m(r) profiles ...")
    smap, norm, log_pcs = _make_cmap_norm(results)
    cmap = plt.get_cmap(CMAP_NAME)
 
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for res, lpc in zip(results, log_pcs):
        col = cmap(norm(lpc))
        ax.plot(res['r_km'], res['m_sol'], color=col, lw=1.5)
        # mark stellar surface with a small circle
        ax.plot(res['R_total'], res['M_total'], 'o', color=col,
                ms=4, zorder=5)
 
    _style(ax,
           title=rf'Enclosed Mass Profile   ($\alpha = {alpha_km2:.1f}\ \mathrm{{km}}^2$,'
                 rf'  $\Gamma={GAMMA_DEFAULT}$)',
           xlabel=r'$r\ [\mathrm{km}]$',
           ylabel=r'$m(r)\ [M_\odot]$')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
 
    cb = fig.colorbar(smap, ax=ax, pad=0.02)
    cb.set_label(r'$\log_{10}(p_c\ [\mathrm{km}^{-2}])$', fontsize=9)
 
    # Arrow annotation showing direction of increasing p_c
    ax.annotate('', xy=(0.72, 0.60), xytext=(0.60, 0.45),
                xycoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color='#444444', lw=1.5))
    ax.text(0.61, 0.42, r'increasing $p_c$', transform=ax.transAxes,
            fontsize=8, color='#444444', ha='left', va='top')
 
    fig.tight_layout()
    plt.show()
 
# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2  —  Pressure and energy density profiles
# ══════════════════════════════════════════════════════════════════════════════
def fig2_pressure_profiles(results, alpha_km2):
    print("[Figure 2]  p(r) and ε(r) profiles ...")
    smap, norm, log_pcs = _make_cmap_norm(results)
    cmap = plt.get_cmap(CMAP_NAME)
 
    fig, (ax_p, ax_e) = plt.subplots(1, 2, figsize=(13, 5.2))
 
    for res, lpc in zip(results, log_pcs):
        col = cmap(norm(lpc))
        ax_p.plot(res['r_km'], res['p_Pa'],   color=col, lw=1.5)
        ax_e.plot(res['r_km'], res['eps_Pa'], color=col, lw=1.5)
 
    _style(ax_p,
           title=rf'Pressure Profile   ($\alpha={alpha_km2:.1f}\ \mathrm{{km}}^2$)',
           xlabel=r'$r\ [\mathrm{km}]$',
           ylabel=r'$p(r)\ [10^{34}\ \mathrm{Pa}]$')
    ax_p.set_yscale('log')
    ax_p.set_xlim(left=0)
 
    _style(ax_e,
           title=rf'Energy Density Profile   ($\alpha={alpha_km2:.1f}\ \mathrm{{km}}^2$)',
           xlabel=r'$r\ [\mathrm{km}]$',
           ylabel=r'$\varepsilon(r)\ [10^{34}\ \mathrm{Pa}]$')
    ax_e.set_yscale('log')
    ax_e.set_xlim(left=0)
 
    cb = fig.colorbar(smap, ax=[ax_p, ax_e], shrink=0.85, pad=0.02)
    cb.set_label(r'$\log_{10}(p_c\ [\mathrm{km}^{-2}])$', fontsize=9)
 
    fig.suptitle(rf'$f(R)=R+\alpha R^2$  |  $\Gamma={GAMMA_DEFAULT}$,  '
                 rf'$\tilde{{K}}={K_TILDE:.1f}$  |  $r_b={r_b_km:.3f}$ km',
                 fontsize=9, y=1.00)
    fig.tight_layout()
    plt.show()
 
# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3  —  Ricci scalar R(r)  and  dR/dr(r)
# ══════════════════════════════════════════════════════════════════════════════
def fig3_ricci_profiles(results, alpha_km2):
    print("[Figure 3]  Ricci scalar profiles ...")
    smap, norm, log_pcs = _make_cmap_norm(results)
    cmap = plt.get_cmap(CMAP_NAME)
 
    fig, (ax_R, ax_Rp) = plt.subplots(1, 2, figsize=(13, 5.2))
 
    for res, lpc in zip(results, log_pcs):
        col = cmap(norm(lpc))
        ax_R.plot(res['r_km'],  res['R_ricci_km2'],  color=col, lw=1.5)
        ax_Rp.plot(res['r_km'], res['Rp_ricci_km3'], color=col, lw=1.5)
 
    _style(ax_R,
           title=rf'Ricci Scalar   ($\alpha={alpha_km2:.1f}\ \mathrm{{km}}^2$)',
           xlabel=r'$r\ [\mathrm{km}]$',
           ylabel=r'$R(r)\ [\mathrm{km}^{-2}]$')
    ax_R.axhline(0, color='gray', lw=0.7, ls='--', alpha=0.5)
    ax_R.set_xlim(left=0)
 
    _style(ax_Rp,
           title=rf'Ricci Scalar Derivative   ($\alpha={alpha_km2:.1f}\ \mathrm{{km}}^2$)',
           xlabel=r'$r\ [\mathrm{km}]$',
           ylabel=r'$\mathrm{d}R/\mathrm{d}r\ [\mathrm{km}^{-3}]$')
    ax_Rp.axhline(0, color='gray', lw=0.7, ls='--', alpha=0.5)
    ax_Rp.set_xlim(left=0)
 
    cb = fig.colorbar(smap, ax=[ax_R, ax_Rp], shrink=0.85, pad=0.02)
    cb.set_label(r'$\log_{10}(p_c\ [\mathrm{km}^{-2}])$', fontsize=9)
 
    fig.suptitle(rf'$f(R)=R+\alpha R^2$  |  $\Gamma={GAMMA_DEFAULT}$  |  '
                 rf'$r_b={r_b_km:.3f}$ km',
                 fontsize=9, y=1.00)
    fig.tight_layout()
    plt.show()
 
# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4  —  Mass-Radius diagram
# ══════════════════════════════════════════════════════════════════════════════
def fig4_mass_radius(results, alpha_km2):
    print("[Figure 4]  M-R diagram ...")
    smap, norm, log_pcs = _make_cmap_norm(results)
    cmap = plt.get_cmap(CMAP_NAME)
 
    Ms = np.array([r['M_total'] for r in results])
    Rs = np.array([r['R_total'] for r in results])
    log_pcs_arr = np.array(log_pcs)
 
    # ── CRITICAL FIX: keep results in NATURAL p_c order (increasing p_c).
    # Sorting by R scrambles the M-R curve because R is NOT monotonic in p_c:
    # R first increases slightly, peaks, then decreases as p_c → ∞.
    # Sorting by R merges the stable and unstable branches randomly, producing
    # the zigzag/oscillating pattern seen in earlier versions.
    # The correct M-R curve is traced by connecting (R, M) in p_c order.
    Ms_s      = Ms           # already in p_c order from run_sweep()
    Rs_s      = Rs
    lpc_s     = log_pcs_arr
 
    fig, ax = plt.subplots(figsize=(7.5, 6))
 
    # Draw the curve with colour-coded individual points
    for i in range(len(results) - 1):
        mid_lpc = 0.5 * (lpc_s[i] + lpc_s[i+1])
        col = cmap(norm(mid_lpc))
        ax.plot(Rs_s[i:i+2], Ms_s[i:i+2], color=col, lw=2.2)
 
    sc = ax.scatter(Rs_s, Ms_s, c=lpc_s, cmap=CMAP_NAME,
                    norm=norm, s=28, zorder=5, edgecolors='none')
 
    # Mark maximum mass
    i_max = np.argmax(Ms_s)
    ax.plot(Rs_s[i_max], Ms_s[i_max], '*', ms=13,
            color='white', markeredgecolor='#333333', markeredgewidth=0.8,
            zorder=10, label=rf'$M_\mathrm{{max}}={Ms_s[i_max]:.2f}\ M_\odot$'
                              rf'  at  $R={Rs_s[i_max]:.1f}$ km')
 
    # Observational constraint: PSR J0740+6620
    ax.axhline(2.08, color='#555555', lw=1.0, ls='--', alpha=0.7)
    ax.axhspan(2.01, 2.15, color='#888888', alpha=0.12)
    ax.text(ax.get_xlim()[0] if ax.get_xlim()[0] > 0 else 5.5,
            2.10, 'PSR J0740+6620', color='#555555', fontsize=8, va='bottom')
 
    _style(ax,
           title=rf'Mass-Radius Diagram   ($\alpha={alpha_km2:.1f}\ \mathrm{{km}}^2$,'
                 rf'  $\Gamma={GAMMA_DEFAULT}$)',
           xlabel=r'$R_*\ [\mathrm{km}]$',
           ylabel=r'$M_*\ [M_\odot]$')
    ax.legend(loc='lower left', framealpha=0.85, fontsize=9)
 
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label(r'$\log_{10}(p_c\ [\mathrm{km}^{-2}])$', fontsize=9)
 
    fig.tight_layout()
    plt.show()
 
# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5  —  Summary 2×2 panel
# ══════════════════════════════════════════════════════════════════════════════
def fig5_summary(results, alpha_km2):
    print("[Figure 5]  Summary panel ...")
    smap, norm, log_pcs = _make_cmap_norm(results)
    cmap = plt.get_cmap(CMAP_NAME)
 
    Ms = np.array([r['M_total'] for r in results])
    Rs = np.array([r['R_total'] for r in results])
    log_pcs_arr = np.array(log_pcs)
    # Keep in natural p_c order (see fig4 comment for explanation)
    Ms_s = Ms; Rs_s = Rs; lpc_s = log_pcs_arr
 
    fig = plt.figure(figsize=(13, 9))
    gs  = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.10)
    ax_m  = fig.add_subplot(gs[0, 0])   # m(r)
    ax_p  = fig.add_subplot(gs[0, 1])   # p(r)
    ax_mr = fig.add_subplot(gs[1, 0])   # M-R
    ax_R  = fig.add_subplot(gs[1, 1])   # R_Ricci(r)
 
    # ── profile panels ────────────────────────────────────────────────────────
    for res, lpc in zip(results, log_pcs):
        col = cmap(norm(lpc))
        ax_m.plot(res['r_km'], res['m_sol'],        color=col, lw=1.4)
        ax_p.plot(res['r_km'], res['p_Pa'],         color=col, lw=1.4)
        ax_R.plot(res['r_km'], res['R_ricci_km2'],  color=col, lw=1.4)
 
    _style(ax_m, title=r'$m(r)$',
           xlabel=r'$r$ [km]', ylabel=r'$m(r)\ [M_\odot]$')
    ax_m.set_xlim(left=0); ax_m.set_ylim(bottom=0)
 
    _style(ax_p, title=r'$p(r)$',
           xlabel=r'$r$ [km]', ylabel=r'$p\ [10^{34}\ \mathrm{Pa}]$')
    ax_p.set_yscale('log'); ax_p.set_xlim(left=0)
 
    _style(ax_R, title=r'$R(r)$',
           xlabel=r'$r$ [km]', ylabel=r'$R\ [\mathrm{km}^{-2}]$')
    ax_R.axhline(0, color='gray', lw=0.6, ls='--', alpha=0.5)
    ax_R.set_xlim(left=0)
 
    # ── M-R panel ─────────────────────────────────────────────────────────────
    for i in range(len(results) - 1):
        mid_lpc = 0.5*(lpc_s[i]+lpc_s[i+1])
        ax_mr.plot(Rs_s[i:i+2], Ms_s[i:i+2], color=cmap(norm(mid_lpc)), lw=2.2)
    ax_mr.scatter(Rs_s, Ms_s, c=lpc_s, cmap=CMAP_NAME, norm=norm,
                  s=22, zorder=5, edgecolors='none')
    i_max = np.argmax(Ms_s)
    ax_mr.plot(Rs_s[i_max], Ms_s[i_max], '*', ms=12, color='white',
               markeredgecolor='#333333', markeredgewidth=0.8, zorder=10)
    ax_mr.axhline(2.08, color='#555555', lw=0.9, ls='--', alpha=0.6)
    ax_mr.axhspan(2.01, 2.15, color='#888888', alpha=0.10)
    _style(ax_mr, title='M–R Diagram',
           xlabel=r'$R_*$ [km]', ylabel=r'$M_*\ [M_\odot]$')
 
    # ── shared colourbar ──────────────────────────────────────────────────────
    fig.subplots_adjust(right=0.87)
    cax  = fig.add_axes([0.89, 0.12, 0.022, 0.76])
    cb   = fig.colorbar(smap, cax=cax)
    cb.set_label(r'$\log_{10}(p_c\ [\mathrm{km}^{-2}])$', fontsize=9)
 
    fig.suptitle(
        rf'$f(R)=R+\alpha R^2$  |  $\alpha={alpha_km2:.1f}\ \mathrm{{km}}^2$  |  '
        rf'$\Gamma={GAMMA_DEFAULT}$,  $\tilde{{K}}={K_TILDE:.1f}$  |  '
        rf'$r_b={r_b_km:.3f}$ km  |  $N_{{p_c}}={N_PC}$',
        fontsize=9.5, fontweight='bold', y=1.005)
    plt.show()
 
# ══════════════════════════════════════════════════════════════════════════════
# CONSOLE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
def print_summary(results, alpha_km2):
    Ms  = [r['M_total'] for r in results]
    Rs  = [r['R_total'] for r in results]
    pcs = [r['p_c_km2'] for r in results]
 
    print("\n" + "═"*62)
    print(f"  Run Summary:  α = {alpha_km2:.2f} km²  "
          f"(α̃ = {alpha_to_dimless(alpha_km2):.4f})")
    print(f"  EOS: Γ = {GAMMA_DEFAULT},  K̃ = {K_TILDE:.2f}")
    print(f"  Stars solved: {len(results)}  out of  {N_PC}")
    print(f"  p_c range: {min(pcs):.2e} – {max(pcs):.2e}  km⁻²")
    i_max = int(np.argmax(Ms))
    print(f"  Maximum mass:  M_max = {Ms[i_max]:.3f} M_sun")
    print(f"  Radius at M_max:  R   = {Rs[i_max]:.2f} km")
    print(f"  Compactness 2GM/c²R  = "
          f"{Ms[i_max]*2*r_b_km/Rs[i_max]:.4f}")
    print("═"*62 + "\n")
 
# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    alpha_km2   = ALPHA_FIXED_KM2
    alpha_tilde = alpha_to_dimless(alpha_km2)
 
    print("=" * 62)
    print("  TOV Solver  f(R) = R + α R²   [VERSION 4]")
    print(f"  Fixed α   = {alpha_km2:.2f} km²  →  α̃ = {alpha_tilde:.5f}")
    print(f"  EOS       : Γ = {GAMMA_DEFAULT},  K̃ = {K_TILDE:.2f}")
    print(f"  p_c sweep : {P_C_MIN_KM2:.1e} – {P_C_MAX_KM2:.1e} km⁻²  "
          f"({N_PC} points, log-spaced)")
    print(f"  Colourmap : {CMAP_NAME}")
    print("=" * 62)
 
    print("\nSolving star sequence ...")
    results = run_sweep(alpha_tilde)
 
    if len(results) == 0:
        print("ERROR: no valid solutions found. "
              "Try adjusting P_C_MIN_KM2 / P_C_MAX_KM2.")
    else:
        print_summary(results, alpha_km2)
        fig1_mass_profiles(results,   alpha_km2)
        fig2_pressure_profiles(results, alpha_km2)
        fig3_ricci_profiles(results,  alpha_km2)
        fig4_mass_radius(results,     alpha_km2)
        fig5_summary(results,         alpha_km2)
        print("Done.")
        
# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — BSR2 EOS: Pressure vs Mass Density  (paste at the END of __main__)
# ══════════════════════════════════════════════════════════════════════════════
# This block uses the BSR2 CGS arrays already built earlier in the script:
#   _K_CGS, _BSR2_GAMMAS, _BSR2_RHO_B_CGS
# It does NOT depend on any TOV solution — it just evaluates p(ρ) directly.

print("[Figure 6] BSR2 EOS: p(ρ) ...")

# ── build ρ and p arrays in CGS ───────────────────────────────────────────────
_rho_eos = np.logspace(4, np.log10(5e15), 3000)   # g cm⁻³,  from outer crust to deep core
_p_eos   = np.empty_like(_rho_eos)

for _j, _rh in enumerate(_rho_eos):
    _idx      = int(np.searchsorted(_BSR2_RHO_B_CGS, _rh, side='right'))
    _idx      = min(_idx, 6)
    _p_eos[_j] = _K_CGS[_idx] * _rh ** _BSR2_GAMMAS[_idx]

# ── plot ──────────────────────────────────────────────────────────────────────
_fig_eos, _ax_eos = plt.subplots(figsize=(8, 5.5))

_ax_eos.loglog(_rho_eos, _p_eos, color='#01696f', lw=2.2, label='BSR2 EOS')

# shade each layer a different colour and label with its Γ_i
_layer_colors = ['#b5d4d1','#c8dde0','#d9e8e3','#cde0d8',
                 '#f5d4c0','#f5c4a8','#f5b490']
_rho_bounds_plot = np.concatenate(([1e4], _BSR2_RHO_B_CGS, [5e15]))
_layer_labels = [
    'Crust\nlayer 0', 'Crust\nlayer 1', 'Crust\nlayer 2', 'Crust\nlayer 3',
    'Core\nlayer 4',  'Core\nlayer 5',  'Core\nlayer 6'
]

for _i in range(7):
    _rl = _rho_bounds_plot[_i]
    _rr = _rho_bounds_plot[_i + 1]
    _ax_eos.axvspan(_rl, _rr, alpha=0.25, color=_layer_colors[_i], lw=0)
    # vertical dashed boundary line
    if _i > 0:
        _ax_eos.axvline(_rl, color='gray', lw=0.9, ls='--', alpha=0.6)
    # Γ_i annotation at geometric midpoint of layer
    _rm  = np.sqrt(_rl * _rr)
    _pm  = _K_CGS[_i] * _rm ** _BSR2_GAMMAS[_i]
    _ax_eos.text(_rm, _pm * 8,
             f'$\\Gamma_{{{_i}}}$={_BSR2_GAMMAS[_i]:.4f}\n{_layer_labels[_i]}',
             fontsize=7, ha='center', va='bottom', color='#333333',
             bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.75))

# reference densities
_ax_eos.axvline(2.5e14, color='#964219', lw=1.2, ls=':', alpha=0.85,
                label=r'$\rho_0 = 2.5\times10^{14}$ g cm$^{-3}$ (nuclear sat.)')

_ax_eos.set_xlabel(r'Mass density  $\rho$  [g cm$^{-3}$]', fontsize=11)
_ax_eos.set_ylabel(r'Pressure  $p$  [dyn cm$^{-2}$]', fontsize=11)
_ax_eos.set_ylabel(r'Pressure  $p$  [dyn cm$^{-2}$]',      fontsize=11)
_ax_eos.set_title('BSR2 — 7-layer Piecewise Polytropic EOS', fontsize=12, fontweight='bold')
_ax_eos.tick_params(which='both', direction='in', top=True, right=True)
_ax_eos.grid(True, which='both', ls=':', lw=0.6, alpha=0.3)
_ax_eos.legend(fontsize=9, loc='upper left')
_ax_eos.set_xlim(_rho_bounds_plot[0], _rho_bounds_plot[-1])

_fig_eos.tight_layout()
plt.show()
