import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# -----------------------------
# Physical constants (CGS)
# -----------------------------
c = 2.99792458e10          # cm/s
G = 6.673e-8               # cm^3 g^-1 s^-2
hbar = 1.0546e-27          # erg s
m_n = 1.675e-24            # g
M_sun = 1.989e33           # g
km = 1.0e5                 # cm

# -----------------------------
# Pure neutron Fermi-gas EOS fit
# ebar = A_NR * pbar^(3/5) + A_R * pbar
# -----------------------------
A_NR = 2.4216
A_R = 2.8663

# Energy density scale from Eq. (44)
eps0 = (m_n**4 * c**5) / (3.0 * np.pi**2 * hbar**3)

def epsilon_of_p(p):
    """Energy density epsilon(p) in erg/cm^3."""
    if p <= 0.0:
        return 0.0
    pbar = p / eps0
    ebar = A_NR * pbar**(3.0 / 5.0) + A_R * pbar
    return eps0 * ebar

def tov_rhs(r, y):
    """
    TOV equations:
    y[0] = pressure p(r) in erg/cm^3
    y[1] = enclosed mass m(r) in g
    """
    p, m = y

    if p <= 0.0:
        return [0.0, 0.0]

    eps = epsilon_of_p(p)

    if r <= 0.0:
        return [0.0, 0.0]

    rho = eps / c**2
    dmdr = 4.0 * np.pi * r**2 * rho

    if m <= 0.0:
        dpdr = -(4.0 / 3.0) * np.pi * G * eps * rho * r / c**2
        return [dpdr, dmdr]

    denom = 1.0 - 2.0 * G * m / (r * c**2)
    if denom <= 0.0:
        return [0.0, 0.0]

    dpdr = -(G * eps * m / (c**2 * r**2))
    dpdr *= (1.0 + p / eps)
    dpdr *= (1.0 + 4.0 * np.pi * r**3 * p / (m * c**2))
    dpdr /= denom

    return [dpdr, dmdr]

def surface_event(r, y):
    return y[0] - 1.0e-12 * eps0

surface_event.terminal = True
surface_event.direction = -1

def solve_star(p0_bar, r0=1.0e-4 * km, rmax=30.0 * km):
    """
    Solve one star for a given dimensionless central pressure p0_bar.
    Returns (R_km, M_solar) or None.
    """
    p_c = p0_bar * eps0
    eps_c = epsilon_of_p(p_c)
    rho_c = eps_c / c**2

    m0 = (4.0 / 3.0) * np.pi * r0**3 * rho_c
    y0 = [p_c, m0]

    sol = solve_ivp(
        tov_rhs,
        (r0, rmax),
        y0,
        method="DOP853",
        events=surface_event,
        rtol=1e-10,
        atol=1e-12,
        max_step=2.0e3
    )

    if not sol.success:
        return None

    if len(sol.t_events[0]) > 0:
        R = sol.t_events[0][0]
        M = sol.y_events[0][0][1]
    else:
        R = sol.t[-1]
        M = sol.y[1, -1]

    R_km = R / km
    M_solar = M / M_sun

    if not (0.0 < R_km < 40.0 and 0.0 < M_solar < 3.0):
        return None

    return R_km, M_solar

# --------------------------------------------------
# Pressure sampling:
# keep order in p0, do NOT sort by radius later
# denser near the peak, lighter at extreme high p0
# --------------------------------------------------
p0_bar_values = np.concatenate([
    np.geomspace(1e-4, 5e-3, 50),
    np.geomspace(5e-3, 5e-2, 60),
    np.geomspace(5e-2, 5e-1, 70),
    np.geomspace(5e-1, 5e0, 60),
    np.geomspace(5e0, 3e1, 25)
])

R_vals = []
M_vals = []
P_vals = []

for p0_bar in p0_bar_values:
    result = solve_star(p0_bar)
    if result is None:
        continue
    R_km, M_sol = result
    R_vals.append(R_km)
    M_vals.append(M_sol)
    P_vals.append(p0_bar)

R_vals = np.array(R_vals)
M_vals = np.array(M_vals)
P_vals = np.array(P_vals)

# Maximum mass point
imax = np.argmax(M_vals)
R_max = R_vals[imax]
M_max = M_vals[imax]

# -----------------------------
# Plot in original p0 order
# -----------------------------
plt.figure(figsize=(8.5, 6))
plt.plot(R_vals, M_vals, color='navy', lw=1.6)
plt.scatter(R_max, M_max, color='crimson', zorder=5)

plt.annotate(
    f"Max mass ≈ {M_max:.3f} M$_\\odot$\nR ≈ {R_max:.2f} km",
    xy=(R_max, M_max),
    xytext=(R_max + 1.5, M_max - 0.08),
    arrowprops=dict(arrowstyle="->", lw=1.0),
    fontsize=10
)

plt.xlabel("Radius R (km)", fontsize=12)
plt.ylabel("Mass M (M$_\\odot$)", fontsize=12)
plt.title("Mass-Radius Relation for Pure Neutron Stars (Fermi Gas EOS)", fontsize=13)
plt.xlim(4.5, 25)
plt.ylim(0.0, 1.0)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()