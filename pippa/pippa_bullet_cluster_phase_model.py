
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Pippa-inspired Bullet Cluster toy simulation
# ============================================================
#
# Основная идея:
#
# 1. Газ после столкновения теряет фазовую когерентность
#    -> локально D_eff -> 1
#
# 2. Галактики сохраняют глобальный цикл квадрантов
#    -> D_eff ~ 4/pi
#
# 3. Эффективная гравитационная масса зависит
#    не только от rho, но и от phase coherence.
#
# ============================================================

N = 700

x = np.linspace(-8, 8, N)
y = np.linspace(-5, 5, N)

X, Y = np.meshgrid(x, y)

# ------------------------------------------------------------
# Geometry
# ------------------------------------------------------------

# Gas peaks (lag behind after collision)
gas1 = (-1.3, 0.0)
gas2 = ( 1.1, 0.3)

# Galaxy peaks (pass through)
gal1 = (-2.8, 0.0)
gal2 = ( 3.0, 0.5)

# ------------------------------------------------------------
# Gaussian helper
# ------------------------------------------------------------

def gaussian(x0, y0, sigma, mass):
    r2 = (X - x0)**2 + (Y - y0)**2
    return mass * np.exp(-r2/(2*sigma**2))

# ------------------------------------------------------------
# Matter components
# ------------------------------------------------------------

rho_gas = (
    gaussian(*gas1, sigma=1.2, mass=1.0)
    + gaussian(*gas2, sigma=1.0, mass=0.7)
)

rho_gal = (
    gaussian(*gal1, sigma=0.45, mass=0.35)
    + gaussian(*gal2, sigma=0.45, mass=0.28)
)

rho_baryon = rho_gas + rho_gal

# ------------------------------------------------------------
# Pippa phase coherence
# ------------------------------------------------------------
#
# Gas:
#   strong decoherence after collision
#   D_local -> 1
#
# Galaxies:
#   preserve global phase cycle
#   D_global = 4/pi
#
# ------------------------------------------------------------

D_global = 4/np.pi
D_local = 1.0

# coherence parameter:
# 0 -> decohered
# 1 -> coherent

coh_gas = 0.12
coh_gal = 1.0

# Effective information capacity
#
# We model the Pippa enhancement as:
#
# enhancement ~ D_global / D_eff
#
# If D_eff -> 1:
#   geometry becomes locally Euclidean
#   effective fractal amplification disappears
#
# If D_eff -> 4/pi:
#   full Pippa enhancement survives
#

D_eff_gas = D_local + coh_gas * (D_global - D_local)
D_eff_gal = D_local + coh_gal * (D_global - D_local)

enh_gas = D_eff_gas / D_local
enh_gal = D_eff_gal / D_local

# ------------------------------------------------------------
# Effective gravitational lensing field
# ------------------------------------------------------------

phi_eff = (
    enh_gas * rho_gas
    + 3.2 * enh_gal * rho_gal
)

# Normalize
rho_baryon /= rho_baryon.max()
phi_eff /= phi_eff.max()

# ------------------------------------------------------------
# Diagnostics
# ------------------------------------------------------------

baryon_idx = np.unravel_index(np.argmax(rho_baryon), rho_baryon.shape)
grav_idx = np.unravel_index(np.argmax(phi_eff), phi_eff.shape)

baryon_peak = (X[baryon_idx], Y[baryon_idx])
grav_peak = (X[grav_idx], Y[grav_idx])

dx = grav_peak[0] - baryon_peak[0]
dy = grav_peak[1] - baryon_peak[1]

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(11, 7))

ax.imshow(
    rho_baryon,
    extent=[x.min(), x.max(), y.min(), y.max()],
    origin="lower",
    alpha=0.75,
)

levels = np.linspace(0.2, 0.95, 8)

ax.contour(
    X,
    Y,
    phi_eff,
    levels=levels,
)

ax.scatter(
    [gas1[0], gas2[0]],
    [gas1[1], gas2[1]],
    marker="x",
    s=120,
)

ax.scatter(
    [gal1[0], gal2[0]],
    [gal1[1], gal2[1]],
    marker="+",
    s=180,
)

ax.set_title("Pippa-inspired Bullet Cluster")
ax.set_xlabel("x")
ax.set_ylabel("y")

summary = f'''
PIPPA BULLET CLUSTER TOY MODEL
================================

Global fractal dimension:
D_global = 4/pi = {D_global:.6f}

Gas sector:
coherence = {coh_gas:.3f}
D_eff_gas = {D_eff_gas:.6f}
enhancement = {enh_gas:.6f}

Galaxy sector:
coherence = {coh_gal:.3f}
D_eff_gal = {D_eff_gal:.6f}
enhancement = {enh_gal:.6f}

Peak baryonic density:
x = {baryon_peak[0]:.3f}
y = {baryon_peak[1]:.3f}

Peak effective gravity:
x = {grav_peak[0]:.3f}
y = {grav_peak[1]:.3f}

Offset:
dx = {dx:.3f}
dy = {dy:.3f}
'''

print(summary)
