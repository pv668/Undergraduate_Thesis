import numpy as np
import matplotlib.pyplot as plt

# ----------------- Toy polytropic EoS -----------------
def make_polytropic_eos(K, n):
    def eps_of_p(p):
        p = np.maximum(p, 0.0)
        return (p / K)**(n/(n+1.0))
    return eps_of_p

# Tune stiffness so OM and DM curves resemble Fig. 2[attached_file:1]
K_OM, n_OM = 1.0, 1.3
K_DM, n_DM = 0.7, 1.3

EoS_OM = make_polytropic_eos(K_OM, n_OM)
EoS_DM = make_polytropic_eos(K_DM, n_DM)

# ----------------- Two-fluid TOV (dimensionless) -----------------
def tov_rhs_twofluid(r, y):
    p_om, m_om, p_dm, m_dm = y
    m_tot = m_om + m_dm

    if r <= 0.0:
        return np.zeros_like(y)
    if p_om <= 0.0 and p_dm <= 0.0:
        return np.zeros_like(y)

    eps_om = EoS_OM(p_om) if p_om > 0.0 else 0.0
    eps_dm = EoS_DM(p_dm) if p_dm > 0.0 else 0.0

    if r - 2.0*m_tot <= 0.0:
        return np.zeros_like(y)

    dnudr = (m_tot + 4.0*np.pi*r**3*(p_om+p_dm)) / (r*(r-2.0*m_tot))

    dp_om = -(p_om + eps_om) * dnudr if p_om > 0.0 else 0.0
    dm_om = 4.0*np.pi*r**2*eps_om
    dp_dm = -(p_dm + eps_dm) * dnudr if p_dm > 0.0 else 0.0
    dm_dm = 4.0*np.pi*r**2*eps_dm
    return np.array([dp_om, dm_om, dp_dm, dm_dm])

def rk4_step(f, r, y, h):
    k1 = f(r, y)
    k2 = f(r + 0.5*h, y + 0.5*h*k1)
    k3 = f(r + 0.5*h, y + 0.5*h*k2)
    k4 = f(r + h,     y + h*k3)
    return y + h*(k1 + 2*k2 + 2*k3 + k4)/6.0

def integrate_twofluid(p_om_c, ratio_r,
                       r0=1e-3, h=5e-3, fac_term=1e-6):
    p_dm_c = ratio_r * p_om_c
    eps_om_c = EoS_OM(p_om_c)
    eps_dm_c = EoS_DM(p_dm_c)

    m_om0 = 4.0*np.pi/3.0 * r0**3 * eps_om_c
    m_dm0 = 4.0*np.pi/3.0 * r0**3 * eps_dm_c

    r = r0
    y = np.array([p_om_c, m_om0, p_dm_c, m_dm0])

    rs = [r]
    ylist = [y.copy()]

    while (y[0] > fac_term*p_om_c or y[2] > fac_term*p_dm_c) and r < 10.0:
        y = rk4_step(tov_rhs_twofluid, r, y, h)
        r += h
        rs.append(r)
        ylist.append(y.copy())

    rs = np.array(rs)
    yarr = np.array(ylist)

    p_om_arr = yarr[:,0]
    p_dm_arr = yarr[:,2]

    idx_om = np.argmax(p_om_arr <= fac_term*p_om_c)
    idx_dm = np.argmax(p_dm_arr <= fac_term*p_dm_c)
    if idx_om == 0: idx_om = len(rs)-1
    if idx_dm == 0: idx_dm = len(rs)-1

    R_om = rs[idx_om]
    R_dm = rs[idx_dm]
    M_om = yarr[idx_om,1]
    M_dm = yarr[idx_dm,3]

    return rs, yarr, R_om, R_dm, M_om, M_dm

# ----------------- choose central pressure & ratio -----------------
p_om_c = 1.0      # central OM pressure
ratio_r = 1     # central DM / OM pressure

rs, yarr, R_om, R_dm, M_om, M_dm = integrate_twofluid(p_om_c, ratio_r)
p_om_arr = yarr[:,0]; p_dm_arr = yarr[:,2]
m_om_arr = yarr[:,1]; m_dm_arr = yarr[:,3]

# ----------------- rescale to look like Fig. 2 -----------------
# set OM radius ~ 7.5 km and DM ~ 5 km
scale_r = 7.5 / R_om
R_km = rs * scale_r
R_om_km = R_om * scale_r
R_dm_km = R_dm * scale_r

# normalize pressures so p_om(0) ~ 1000
scale_p = 1000.0 / p_om_arr[0]
p_om_plot = p_om_arr * scale_p
p_dm_plot = p_dm_arr * scale_p

# normalize masses so M_om(max) ~ 0.8 M_sun
scale_m = 0.8 / m_om_arr.max()
m_om_plot = m_om_arr * scale_m
m_dm_plot = m_dm_arr * scale_m

# ----------------- plot -----------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7,5), sharex=True)

# pressure panel
ax1.plot(R_km, p_om_plot, 'k-',  label="p(r) (OM)")
ax1.plot(R_km, p_dm_plot, 'k--', label="p(r) (DM)")
ax1.axvline(R_dm_km, color='green', label="DM-radius")
ax1.axvline(R_om_km, color='red',   label="OM-radius")
ax1.set_ylabel("p in MeV fm$^{-3}$ (scaled)")
ax1.legend(loc="upper right")
ax1.grid(True)

# mass panel
ax2.plot(R_km, m_om_plot, 'k-',  label="m(r) (OM)")
ax2.plot(R_km, m_dm_plot, 'k--', label="m(r) (DM)")
ax2.axvline(R_dm_km, color='green', label="DM-radius")
ax2.axvline(R_om_km, color='red',   label="OM-radius")
ax2.set_xlabel("r in km (scaled)")
ax2.set_ylabel("m in $M_\\odot$ (scaled)")
ax2.legend(loc="lower right")
ax2.grid(True)

plt.tight_layout()
plt.show()
