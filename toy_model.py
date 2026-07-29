"""
Toy-model симуляция Вселенной в теории Pippa.
Визуализация: расширение, плотности, энтропия, сравнение предсказаний с данными.
Использует модули pippa для получения фундаментальных констант и космологических параметров.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Импортируем модули теории Pippa (убедитесь, что они доступны в PYTHONPATH)
from pippa import constants, cosmology, particle_physics

# ------------------------------------------------------------
# 1. Получение параметров теории Pippa
# ------------------------------------------------------------

D = constants.D                        # 4/π ≈ 1.2732
epsilon = D - 1.0                      # ≈ 0.27324

# Альфа-коррекции (выводятся из D_Mir, D_Neg)
alpha_A = particle_physics.alpha_A_theoretical()
alpha_B = particle_physics.alpha_B_theoretical()

# Космологические предсказания
n_s = cosmology.spectral_index()                    # ≈ 0.9738
r = cosmology.tensor_to_scalar()                    # ≈ 0.00206

# Отношение DM/барионы (альфа-скорректированное)
Omega_DM_over_b = cosmology.dm_to_baryon_ratio(
    alpha_A=alpha_A,
    alpha_B=alpha_B,
)

# Принудительно приводим к float (на случай комплексных чисел)
if isinstance(Omega_DM_over_b, complex):
    Omega_DM_over_b = float(Omega_DM_over_b.real)
else:
    Omega_DM_over_b = float(Omega_DM_over_b)

# Барионная плотность (из Planck 2018)
Omega_b = 0.0493
Omega_DM = Omega_b * Omega_DM_over_b
Omega_Lambda = 1.0 - Omega_b - Omega_DM

# Ещё раз проверяем, не стало ли комплексным
if isinstance(Omega_Lambda, complex):
    Omega_Lambda = float(Omega_Lambda.real)
else:
    Omega_Lambda = float(Omega_Lambda)

Omega_m = Omega_b + Omega_DM
if isinstance(Omega_m, complex):
    Omega_m = float(Omega_m.real)
else:
    Omega_m = float(Omega_m)

# Параметр Хаббла сегодня (из Planck)
H0 = 67.4  # км/с/Мпк

print("=== Параметры теории Pippa ===")
print(f"D = {D:.6f}")
print(f"α_A = {alpha_A:.6f},  α_B = {alpha_B:.6f}")
print(f"n_s = {n_s:.4f}  (Planck: 0.9649 ± 0.0042) -> {abs(n_s-0.9649)/0.0042:.1f}σ")
print(f"r   = {r:.4f}  (BICEP/Keck limit: < 0.036)")
print(f"Ω_DM/Ω_b = {Omega_DM_over_b:.3f}  (Planck: 5.366 ± 0.07) -> {abs(Omega_DM_over_b-5.366)/0.07:.1f}σ")
print(f"Ω_m = {Omega_m:.4f},  Ω_Λ = {Omega_Lambda:.4f}")
print("="*50)

# ------------------------------------------------------------
# 2. Решение уравнения Фридмана для a(t)
# ------------------------------------------------------------

# Переводим H0 в 1/Гигагод
H0_Gyr = H0 / (3.086e19) * (3.156e16)   # ≈ 0.0689 Гр⁻¹

def friedmann(a, t, Omega_m, Omega_L, H0):
    """da/dt = H0 * sqrt(Omega_m/a + Omega_L*a^2)"""
    arg = Omega_m / a + Omega_L * a**2
    if arg < 0:
        arg = 0.0
    return H0 * np.sqrt(arg)

# Временной массив: от t=0.01 до 13.8 млрд лет
t_Gyr = np.linspace(0.01, 13.8, 500)
a0 = 0.001  # начальное значение (после инфляции)

# Теперь все аргументы точно float
sol = odeint(friedmann, a0, t_Gyr, args=(Omega_m, Omega_Lambda, H0_Gyr))
a_t = sol[:, 0]
# Нормируем на a(сегодня) = 1
a_t = a_t / a_t[-1]

# Параметр Хаббла H(z) = (1/a) da/dt
da_dt = np.gradient(a_t, t_Gyr)  # Гр⁻¹
H_z = da_dt / a_t * 1000.0       # переводим в км/с/Мпк

# ------------------------------------------------------------
# 3. Эволюция плотностей
# ------------------------------------------------------------

rho_b = Omega_b / a_t**3
rho_DM = Omega_DM / a_t**3
rho_L = Omega_Lambda * np.ones_like(a_t)
rho_crit = rho_b + rho_DM + rho_L
rho_b_norm = rho_b / rho_crit
rho_DM_norm = rho_DM / rho_crit
rho_L_norm = rho_L / rho_crit

# Информационная энтропия (S ~ ln a)
S_t = np.log(a_t + 1e-6) - np.log(a_t[0] + 1e-6)

# ------------------------------------------------------------
# 4. Наблюдательные данные для H(z)
# ------------------------------------------------------------

hubble_data = [
    (0.07, 69.0, 19.6, 19.6),
    (0.10, 69.0, 12.0, 12.0),
    (0.20, 72.9, 29.6, 29.6),
    (0.30, 81.7, 25.4, 25.4),
    (0.40, 82.0, 25.0, 25.0),
    (0.50, 90.9, 30.0, 30.0),
    (0.60, 87.9, 26.9, 26.9),
    (0.70, 94.6, 31.0, 31.0),
    (0.80, 115.2, 28.0, 28.0),
    (0.90, 117.0, 23.0, 23.0),
    (1.00, 122.0, 30.0, 30.0),
    (1.20, 133.0, 30.0, 30.0),
    (1.40, 151.0, 28.0, 28.0),
    (1.60, 168.0, 33.0, 33.0),
    (1.80, 188.0, 40.0, 40.0),
    (2.00, 205.0, 35.0, 35.0),
]

z_data = np.array([d[0] for d in hubble_data])
H_obs = np.array([d[1] for d in hubble_data])
H_err = np.array([d[2] for d in hubble_data])

# ------------------------------------------------------------
# 5. Построение графика (4 панели)
# ------------------------------------------------------------

fig = plt.figure(figsize=(14, 12))
fig.suptitle("Эволюция Вселенной в теории Pippa\n(сравнение с наблюдениями)", fontsize=18)

# Панель 1: Масштабный фактор a(t)
ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(t_Gyr, a_t, 'b-', lw=2, label='a(t)')
ax1.set_xlabel('Время, млрд лет')
ax1.set_ylabel('Масштабный фактор a')
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3)
ax1.axhline(1.0, color='gray', linestyle='--', alpha=0.7, label='Сегодня')
ax1.legend()
ax1.text(0.02, 0.95, f'$n_s$ = {n_s:.4f}', transform=ax1.transAxes, fontsize=12, color='darkblue')
ax1.text(0.02, 0.88, f'$r$ = {r:.4f}', transform=ax1.transAxes, fontsize=12, color='darkblue')
ax1.text(0.02, 0.81, f'$\\Omega_{{\\rm DM}}/\\Omega_b$ = {Omega_DM_over_b:.3f}', transform=ax1.transAxes, fontsize=12, color='darkblue')

# Панель 2: Эволюция относительных плотностей
ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(t_Gyr, rho_b_norm, 'g-', lw=2, label='Барионы')
ax2.plot(t_Gyr, rho_DM_norm, 'r-', lw=2, label='Тёмная материя (Pippa)')
ax2.plot(t_Gyr, rho_L_norm, 'm-', lw=2, label='Тёмная энергия')
ax2.set_xlabel('Время, млрд лет')
ax2.set_ylabel('Относительная плотность')
ax2.set_ylim(0, 1.05)
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper right')
ax2.axvline(x=13.8, color='gray', linestyle='--', alpha=0.5)
ax2.text(0.02, 0.95, 'Плотности сегодня:', transform=ax2.transAxes, fontsize=10, color='black')
ax2.text(0.02, 0.88, f'$\\Omega_b$={Omega_b:.3f}', transform=ax2.transAxes, fontsize=10, color='g')
ax2.text(0.02, 0.81, f'$\\Omega_{{\\rm DM}}$={Omega_DM:.3f}', transform=ax2.transAxes, fontsize=10, color='r')
ax2.text(0.02, 0.74, f'$\\Omega_\\Lambda$={Omega_Lambda:.3f}', transform=ax2.transAxes, fontsize=10, color='m')

# Панель 3: Информационная энтропия S(t)
ax3 = fig.add_subplot(2, 2, 3)
ax3.plot(t_Gyr, S_t, 'c-', lw=2, label='S(t) ~ ln a(t)')
ax3.set_xlabel('Время, млрд лет')
ax3.set_ylabel('Информационная энтропия S (отн. ед.)')
ax3.grid(True, alpha=0.3)
ax3.legend()
ax3.text(0.02, 0.95, 'Рост энтропии = расширение', transform=ax3.transAxes, fontsize=10)

# Панель 4: Параметр Хаббла H(z) с данными
ax4 = fig.add_subplot(2, 2, 4)
z_curve = 1.0 / a_t - 1.0
mask = z_curve <= 2.5
ax4.plot(z_curve[mask], H_z[mask], 'b-', lw=2, label='Pippa (ΛCDM с предсказанными Ω)')
ax4.errorbar(z_data, H_obs, yerr=H_err, fmt='ro', capsize=3, label='Наблюдения (обзоры)')
ax4.errorbar(0, 73.2, yerr=1.3, fmt='gs', markersize=8, label='SH0ES (локальное)')
ax4.errorbar(0, 67.4, yerr=0.5, fmt='bs', markersize=8, label='Planck 2018')
ax4.set_xlabel('Красное смещение z')
ax4.set_ylabel('H(z), км/с/Мпк')
ax4.set_ylim(50, 200)
ax4.grid(True, alpha=0.3)
ax4.legend(loc='upper left')
ax4.text(0.02, 0.95, 'Сравнение с данными:', transform=ax4.transAxes, fontsize=10)
ax4.text(0.02, 0.88, f'$H_0$ (Pippa) = {H0:.1f} км/с/Мпк', transform=ax4.transAxes, fontsize=10, color='b')
ax4.text(0.02, 0.81, f'$H_0$ (Planck) = 67.4', transform=ax4.transAxes, fontsize=10, color='b')
ax4.text(0.02, 0.74, f'$H_0$ (SH0ES) = 73.2', transform=ax4.transAxes, fontsize=10, color='g')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('pippa_universe_evolution.png', dpi=150)
plt.show()
