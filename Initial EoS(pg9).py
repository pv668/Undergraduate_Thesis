import numpy as np
import matplotlib.pyplot as plt

# Constants
m_e = 9.10938356e-28  # electron mass [grams]
h = 6.62607015e-27    # Planck constant [erg·s]
pi = np.pi

# Number density (cm^-3)
n = np.linspace(1e29, 1e32, 100)  # Adjusted to view clearer regime

# Fermi energy (non-relativistic)
E_F = (h**2 / (2 * m_e)) * (3 * pi**2 * n)**(2/3)  # erg

# Energy density
epsilon = n * E_F  # erg/cm^3

# Pressure from non-relativistic Fermi gas EoS
p_eos = (2/5) * n * E_F

# Polytropic comparison
p_poly = epsilon ** (5/3)

plt.figure(figsize=(8,5))
plt.plot(epsilon, p_eos, label='Degenerate Fermi Gas (EoS)')
plt.plot(epsilon, p_poly, label=r'$p = \epsilon^{5/3}$', linestyle='dashed')
plt.xlabel('Energy Density $\\epsilon$ (erg/cm$^3$)')
plt.ylabel('Pressure $p$ (erg/cm$^3$)')
plt.title('EoS vs Polytropic')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.grid(True, which='both', linestyle=':')
plt.tight_layout()
plt.show()
