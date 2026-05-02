import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Parameters for non-relativistic electron gas (γ = 5/3)
gamma = 5/3
alpha = 0.05
beta = 0.005924
p0 = 1e-16       # Central dimensionless pressure

def rhs(r, y):
    p, M = y
    if p <= 0 or r <= 0:
        return [0.0, 0.0]
    density = p**(1/gamma)
    dpdr = -alpha * density * M / r**2
    dMdr = beta * r**2 * density
    return [dpdr, dMdr]

def stop_event(r, y):
    return y[0]
stop_event.terminal = True
stop_event.direction = -1

sol = solve_ivp(rhs, [1e-5, 16000], [p0, 0.0],
                events=stop_event, max_step=10, rtol=1e-8, atol=1e-12)

r = sol.t
pressure = sol.y[0]
mass = sol.y[1]

# Plot Pressure profile
plt.figure(figsize=(7, 4))
plt.plot(r, pressure)
plt.xlabel('Radius (km)')
plt.ylabel('Dimensionless Pressure $\overline{p}(r)$')
plt.title('Pressure Profile, White Dwarf (Non-Relativistic $\gamma=5/3$)')
plt.ylim(0, 1.2e-16)
plt.xlim(0, 14000)
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot Mass profile
plt.figure(figsize=(7, 4))
plt.plot(r, mass)
plt.xlabel('Radius (km)')
plt.ylabel('Dimensionless Mass $\overline{M}(r)$')
plt.title('Mass Profile, White Dwarf (Non-Relativistic $\gamma=5/3$)')
plt.xlim(0, 14000)
plt.grid(True)
plt.tight_layout()
plt.show()
