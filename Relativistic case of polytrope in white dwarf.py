import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Parameters taken from the paper for relativistic electron degenerate gas (γ=4/3)
gamma = 4/3
alpha = 1.0     # According to the paper's Table I
beta = 52.46    # According to Eq. 29 or Table I
p0 = 1e-16      # Starting central dimensionless pressure

def rhs(r, y):
    p, M = y
    if p <= 0 or r <= 0:
        return [0.0, 0.0]
    density = p ** (1/gamma)
    dpdr = -alpha * density * M / r**2         # <-- alpha factor included here
    dMdr = beta * r**2 * density
    return [dpdr, dMdr]

def stop_event(r, y):
    return y[0]   # Stop when pressure crosses zero

stop_event.terminal = True
stop_event.direction = -1

# Integrate
sol = solve_ivp(rhs, [1e-8, 20000], [p0, 0.0],
                events=stop_event, max_step=10, rtol=1e-8, atol=1e-12)

r = sol.t
pressure = sol.y[0]
mass = sol.y[1]

# Plot pressure profile
plt.figure(figsize=(7, 4))
plt.plot(r, pressure)
plt.xlabel('Radius (km)')
plt.ylabel('Dimensionless Pressure $\\bar{p}(r)$')
plt.title('Pressure Profile (relativistic WD, γ=4/3)')
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot mass profile
plt.figure(figsize=(7, 4))
plt.plot(r, mass)
plt.xlabel('Radius (km)')
plt.ylabel('Dimensionless Mass $\\bar{M}(r)$')
plt.title('Mass Profile (relativistic WD, γ=4/3)')
plt.grid(True)
plt.tight_layout()
plt.show()
