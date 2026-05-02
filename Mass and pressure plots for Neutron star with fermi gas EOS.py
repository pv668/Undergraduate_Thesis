import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ---------------------------------
# Constants (CGS)
# ---------------------------------
C = 2.99792458e10          # cm/s
G = 6.673e-8               # cm^3 g^-1 s^-2
HBAR = 1.0546e-27          # erg s
M_N = 1.675e-24            # g
M_SUN = 1.989e33           # g
KM = 1.0e5                 # cm

# ---------------------------------
# Fitted Fermi-gas EOS parameters
# ebar = A_NR * pbar^(3/5) + A_R * pbar
# ---------------------------------
A_NR = 2.4216
A_R = 2.8663

# Energy-density scale from Eq. (44)
EPS0 = (M_N**4 * C**5) / (3.0 * np.pi**2 * HBAR**3)

# Choose central dimensionless pressure
pbar_c = 1.0e-2
p_c = pbar_c * EPS0

def energy_density_from_pressure(p):
    """Return epsilon(p) in erg/cm^3."""
    if p <= 0.0:
        return 0.0
    pbar = p / EPS0
    ebar = A_NR * pbar**(3.0 / 5.0) + A_R * pbar
    return EPS0 * ebar

def rhs_newton(r, y):
    """
    Newtonian structure equations.
    y = [p, m], with p in erg/cm^3 and m in g.
    """
    p, m = y
    if p <= 0.0:
        return [0.0, 0.0]

    eps = energy_density_from_pressure(p)
    rho = eps / C**2

    dmdr = 4.0 * np.pi * r**2 * rho

    if r <= 0.0:
        dpdr = 0.0
    else:
        dpdr = -G * rho * m / r**2

    return [dpdr, dmdr]

def rhs_tov(r, y):
    """
    TOV equations.
    y = [p, m], with p in erg/cm^3 and m in g.
    """
    p, m = y
    if p <= 0.0:
        return [0.0, 0.0]

    eps = energy_density_from_pressure(p)
    rho = eps / C**2
    dmdr = 4.0 * np.pi * r**2 * rho

    if r <= 0.0:
        return [0.0, dmdr]

    if m <= 0.0:
        dpdr = -(4.0 / 3.0) * np.pi * G * eps * rho * r / C**2
        return [dpdr, dmdr]

    schwarzschild = 1.0 - 2.0 * G * m / (r * C**2)
    if schwarzschild <= 0.0:
        return [0.0, 0.0]

    dpdr = -(G * eps * m / (C**2 * r**2))
    dpdr *= (1.0 + p / eps)
    dpdr *= (1.0 + 4.0 * np.pi * r**3 * p / (m * C**2))
    dpdr /= schwarzschild

    return [dpdr, dmdr]

def hit_surface(r, y):
    """Stop when pressure becomes negligibly small."""
    return y[0] - 1.0e-10 * p_c

hit_surface.terminal = True
hit_surface.direction = -1

# ---------------------------------
# Initial conditions
# ---------------------------------
r0 = 1.0e-4 * KM
eps_c = energy_density_from_pressure(p_c)
rho_c = eps_c / C**2
m0 = (4.0 / 3.0) * np.pi * r0**3 * rho_c
y0 = [p_c, m0]

rmax = 16.0 * KM

# ---------------------------------
# Integrate Newtonian and TOV
# ---------------------------------
sol_newton = solve_ivp(
    rhs_newton,
    (r0, rmax),
    y0,
    method="DOP853",
    events=hit_surface,
    rtol=1e-10,
    atol=1e-12,
    max_step=2.0e3
)

sol_tov = solve_ivp(
    rhs_tov,
    (r0, rmax),
    y0,
    method="DOP853",
    events=hit_surface,
    rtol=1e-10,
    atol=1e-12,
    max_step=2.0e3
)

# ---------------------------------
# Extract profiles
# ---------------------------------
r_newt = sol_newton.t / KM
pbar_newt = sol_newton.y[0] / EPS0
m_newt = sol_newton.y[1] / M_SUN

r_tov = sol_tov.t / KM
pbar_tov = sol_tov.y[0] / EPS0
m_tov = sol_tov.y[1] / M_SUN

# Final radius and mass
R_newt = r_newt[-1]
M_newt = m_newt[-1]
R_tov = r_tov[-1]
M_tov = m_tov[-1]

print("Pure neutron star with fitted Fermi-gas EOS")
print(f"epsilon_0 = {EPS0:.4e} erg/cm^3")
print(f"Central pressure p_c = {p_c:.4e} erg/cm^3")
print(f"Central energy density eps_c = {eps_c:.4e} erg/cm^3")
print(f"Newtonian: R = {R_newt:.2f} km, M = {M_newt:.3f} M_sun")
print(f"TOV:       R = {R_tov:.2f} km, M = {M_tov:.3f} M_sun")

# ---------------------------------
# Plot
# ---------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Colors changed from the example:
# Newtonian -> dark orange
# TOV -> teal
axes[0].plot(r_newt, pbar_newt, lw=1.8, color="#c96a23", label="Newtonian")
axes[0].plot(r_tov, pbar_tov, lw=2.6, color="#0f8b8d", label="TOV (GR)")
axes[0].set_xlabel("r (km)", fontsize=12)
axes[0].set_ylabel(r"$\bar{p}(r)$", fontsize=12)
axes[0].set_title("Pressure Profile", fontsize=13)
axes[0].set_xlim(0, 13.5)
axes[0].set_ylim(0, 0.011)
axes[0].grid(True, alpha=0.25)
axes[0].legend(fontsize=10)

axes[1].plot(r_newt, m_newt, lw=1.8, color="#c96a23", label="Newtonian")
axes[1].plot(r_tov, m_tov, lw=2.6, color="#0f8b8d", label="TOV (GR)")
axes[1].set_xlabel("r (km)", fontsize=12)
axes[1].set_ylabel(r"$M(r)$ ($M_{\odot}$)", fontsize=12)
axes[1].set_title("Mass Profile", fontsize=13)
axes[1].set_xlim(0, 13.5)
axes[1].set_ylim(0, 1.1)
axes[1].grid(True, alpha=0.25)
axes[1].legend(fontsize=10)

plt.tight_layout()
plt.show()