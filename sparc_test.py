from pathlib import Path
import io
import math
import zipfile
import argparse
from dataclasses import dataclass

import requests
requests.packages.urllib3.disable_warnings()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import multiprocessing

# ---------------------------------------------------------------------
# Pippa constants
# ---------------------------------------------------------------------
D = 4.0 / math.pi

# ---------------------------------------------------------------------
# Download SPARC data
# ---------------------------------------------------------------------
SPARC_URL = "https://zenodo.org/records/16284118/files/Rotmod_LTG.zip?download=1"

def load_galaxy_from_zip(galaxy_name):
    print(f"  Downloading {galaxy_name}...")
    response = requests.get(SPARC_URL, timeout=120)
    response.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(response.content))
    target_file = f"{galaxy_name}_rotmod.dat"
    if target_file not in z.namelist():
        raise RuntimeError(f"{target_file} not found in archive")
    with z.open(target_file) as f:
        lines = f.read().decode("utf-8").splitlines()
    rows = []
    for line in lines:
        if line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 6:
            continue
        try:
            rows.append({
                "r": float(parts[0]),
                "vobs": float(parts[1]),
                "err": float(parts[2]),
                "vgas": float(parts[3]),
                "vdisk": float(parts[4]),
                "vbul": float(parts[5]),
            })
        except:
            pass
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------
# LCDM: NFW profile
# ---------------------------------------------------------------------
G = 4.30091e-6  # kpc (km/s)^2 / Msun

def nfw_velocity(r, rho0, rs):
    x = r / rs
    mass = 4.0 * math.pi * rho0 * rs**3 * (np.log(1.0 + x) - x / (1.0 + x))
    return np.sqrt(G * mass / r)

# ---------------------------------------------------------------------
# Pippa halo profile
# ---------------------------------------------------------------------
def pippa_density(r, rho0, r0, A_M, gamma, intersector_strength=1.0):
    try:
        r = np.asarray(r, dtype=np.float64)
        if intersector_strength < 0.0 or not np.isfinite(intersector_strength):
            raise ValueError("intersector_strength must be finite and non-negative")
        if intersector_strength == 0.0:
            return np.zeros_like(r)
        r = np.clip(r, 1e-3, None)
        r0_safe = max(abs(r0), 1e-6)
        aeff = (2.0 - D) + gamma * (1.0 - np.exp(-r / r0_safe))
        aeff = np.clip(aeff, -5.0, 5.0)
        m_op = intersector_strength * (
            1.0 + A_M * (1.0 - np.exp(-r / r0_safe))
        )
        m_op = np.clip(m_op, 0.0, 1e6)
        base = r0_safe / r
        base = np.clip(base, 1e-30, 1e30)
        val = rho0 * np.power(base, aeff) * m_op
        return np.nan_to_num(val, nan=0.0, posinf=1e30, neginf=0.0)
    except Exception:
        return np.zeros_like(r)

def enclosed_mass_pippa(r, rho0, r0, A_M, gamma, intersector_strength=1.0):
    if r <= 0:
        return 0.0
    r_safe = max(r, 1e-3)
    if r_safe <= 1e-3:
        return 0.0
    try:
        rs = np.linspace(1e-3, r_safe, 200)
        rho = pippa_density(
            rs,
            rho0,
            r0,
            A_M,
            gamma,
            intersector_strength=intersector_strength,
        )
        integrand = rho * rs**2
        dr = rs[1:] - rs[:-1]
        avg = 0.5 * (integrand[1:] + integrand[:-1])
        integral = np.sum(avg * dr)
        mass = 4.0 * math.pi * integral
        return mass if np.isfinite(mass) else 1e30
    except Exception:
        return 1e30

def pippa_velocity(r, rho0, r0, A_M, gamma, intersector_strength=1.0):
    vel = []
    for ri in np.atleast_1d(r):
        m = enclosed_mass_pippa(
            ri,
            rho0,
            r0,
            A_M,
            gamma,
            intersector_strength=intersector_strength,
        )
        if m <= 0:
            vel.append(0.0)
        else:
            v2 = G * m / ri
            vel.append(0.0 if v2 <= 0 or not np.isfinite(v2) else np.sqrt(v2))
    return np.array(vel)

# ---------------------------------------------------------------------
# Total rotation curves
# ---------------------------------------------------------------------
def total_velocity_nfw(r, rho0, rs, gas, disk, bulge):
    vdm = nfw_velocity(r, rho0, rs)
    return np.sqrt(vdm**2 + gas**2 + disk**2 + bulge**2)

def total_velocity_pippa(r, rho0, r0, A_M, gamma, gas, disk, bulge):
    try:
        vdm = pippa_velocity(r, rho0, r0, A_M, gamma)
        vtot = np.sqrt(vdm**2 + gas**2 + disk**2 + bulge**2)
        return vtot if np.all(np.isfinite(vtot)) else np.full_like(r, 1e9)
    except Exception:
        return np.full_like(r, 1e9)

# ---------------------------------------------------------------------
# Fit tasks (глобальные, чтобы multiprocessing мог их сериализовать)
# ---------------------------------------------------------------------
def _fit_nfw_task(r, vobs, gas, disk, bulge):
    def model(r, rho0, rs):
        return total_velocity_nfw(r, rho0, rs, gas, disk, bulge)
    popt, _ = curve_fit(model, r, vobs,
                        p0=[1e7, 5.0],
                        bounds=([1e4, 0.1], [1e10, 100.0]),
                        maxfev=30000)
    pred = model(r, *popt)
    chi2 = np.mean((pred - vobs) ** 2)
    return ("NFW", popt, chi2, pred)

def _fit_pippa_task(r, vobs, gas, disk, bulge):
    def model(r, rho0, r0, A_M, gamma):
        return total_velocity_pippa(r, rho0, r0, A_M, gamma,
                                    gas, disk, bulge)
    popt, _ = curve_fit(model, r, vobs,
                        p0=[1e7, 5.0, 0.5, 0.2],
                        bounds=([1e4, 0.1, -2.0, -1.0],
                                [1e10, 100.0, 5.0, 2.0]),
                        maxfev=5000)
    pred = model(r, *popt)
    chi2 = np.mean((pred - vobs) ** 2)
    return ("Pippa", popt, chi2, pred)

def _run_fit_process(queue, func, args):
    """Глобальная функция для запуска в Process."""
    try:
        result = func(*args)
        queue.put(result)
    except Exception as e:
        queue.put(e)

def _fit_with_timeout(func, args, timeout=30):
    """Запускает fit-функцию с таймаутом, возвращает результат или None."""
    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=_run_fit_process, args=(queue, func, args))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        print("    ⚠ Таймаут, пропускаем галактику.")
        return None
    else:
        out = queue.get()
        if isinstance(out, Exception):
            print(f"    ⚠ Ошибка в процессе: {out}")
            return None
        return out

# ---------------------------------------------------------------------
# Catalog analysis
# ---------------------------------------------------------------------
def list_galaxies():
    response = requests.get(SPARC_URL, timeout=120)
    response.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(response.content))
    names = []
    for f in z.namelist():
        if f.endswith("_rotmod.dat"):
            names.append(f.replace("_rotmod.dat", ""))
    return sorted(names)

@dataclass
class FitResult:
    model: str
    params: tuple
    chi2: float

def run_catalog_analysis(limit=None):
    galaxies = list_galaxies()
    if limit:
        galaxies = galaxies[:limit]
    results = []
    total = len(galaxies)
    for i, galaxy in enumerate(galaxies):
        print(f"[{i+1}/{total}] {galaxy}")
        try:
            gal = load_galaxy_from_zip(galaxy)
            gal = gal[(gal["r"] > 0) &
                      gal["vobs"].notna() &
                      gal["vgas"].notna() &
                      gal["vdisk"].notna() &
                      gal["vbul"].notna()]
            if len(gal) < 10:
                continue

            r = gal["r"].values
            vobs = gal["vobs"].values
            gas = gal["vgas"].values
            disk = gal["vdisk"].values
            bulge = gal["vbul"].values

            nfw_out = _fit_with_timeout(_fit_nfw_task, (r, vobs, gas, disk, bulge), timeout=1)
            pippa_out = _fit_with_timeout(_fit_pippa_task, (r, vobs, gas, disk, bulge), timeout=1)

            if nfw_out is None or pippa_out is None:
                print(f"    ⚠ Пропущена из-за ошибки или таймаута.")
                continue

            _, nfw_params, nfw_chi2, pred_nfw = nfw_out
            _, pippa_params, pippa_chi2, pred_pippa = pippa_out

            winner = "Pippa" if pippa_chi2 < nfw_chi2 else "NFW"
            results.append({
                "galaxy": galaxy,
                "chi2_nfw": nfw_chi2,
                "chi2_pippa": pippa_chi2,
                "winner": winner,
            })
        except Exception as e:
            print(f"    ⚠ Полный сбой: {e}")
            continue

    return pd.DataFrame(results)

def main():
    df = run_catalog_analysis()
    print("\n" + "=" * 80)
    print(df.head())
    print("=" * 80)
    pippa_wins = np.sum(df["winner"] == "Pippa")
    nfw_wins = np.sum(df["winner"] == "NFW")
    print(f"Pippa wins : {pippa_wins}")
    print(f"NFW wins   : {nfw_wins}")
    print("\nAverage χ²")
    print(f"NFW   : {df['chi2_nfw'].mean():.4f}")
    print(f"Pippa : {df['chi2_pippa'].mean():.4f}")
    df.to_csv("sparc_results.csv", index=False)
    print("\nSaved: sparc_results.csv")
    plt.figure(figsize=(8,6))
    plt.hist(df["chi2_nfw"], bins=30, alpha=0.6, label="NFW")
    plt.hist(df["chi2_pippa"], bins=30, alpha=0.6, label="Pippa")
    plt.xlabel("Mean squared residual")
    plt.ylabel("Count")
    plt.title("SPARC fit quality")
    plt.legend()
    plt.grid(True)
    plt.savefig("sparc_histograms.png", dpi=180, bbox_inches="tight")
    print("Saved: sparc_histograms.png")

if __name__ == "__main__":
    main()
