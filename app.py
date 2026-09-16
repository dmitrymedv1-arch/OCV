"""
OCV-based Transport Number Estimation for Proton-Conducting Electrolytes
Streamlit application for estimating ionic transport numbers from OCV measurements.

Версия с усиленной защитой от протечек session_state и встроенной диагностикой.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import pandas as pd
import io
import time
import warnings
from typing import Dict, Any, Optional, Tuple, List

warnings.filterwarnings('ignore')

# ============================================
# КОНСТАНТЫ
# ============================================

R_GAS = 8.314          # J/(mol·K)
F_FARADAY = 96485.0    # C/mol

# Таблица pH2O от температуры барботёра (0–100 °C)
T_BUBBLER = np.arange(0, 101, 1, dtype=float)
PH2O_TABLE = np.array([
    0.006032963, 0.006485665, 0.006968172, 0.007482161, 0.008029509,
    0.008611892, 0.009231384, 0.009889958, 0.010589687, 0.011331853,
    0.012120405, 0.012957316, 0.013843573, 0.014783124, 0.015778929,
    0.016832963, 0.0179472,   0.019126573, 0.020374044, 0.0216906,
    0.023082161, 0.02455169,  0.02610116,  0.027736491, 0.02945966,
    0.031275598, 0.033189243, 0.035203553, 0.037323464, 0.039553911,
    0.041899827, 0.044365162, 0.046955835, 0.049676783, 0.052532939,
    0.055531211, 0.058675549, 0.061973847, 0.065431039, 0.069054034,
    0.072848754, 0.076822107, 0.080981002, 0.085332346, 0.089884037,
    0.094643967, 0.099620035, 0.104811251, 0.110249198, 0.115914138,
    0.121825808, 0.128003948, 0.134448557, 0.141159635, 0.148156921,
    0.155460153, 0.163059462, 0.170974587, 0.179215396, 0.187791759,
    0.196713546, 0.206000493, 0.215652603, 0.225689613, 0.236121392,
    0.24694794,  0.258208734, 0.269893906, 0.282023193, 0.294616334,
    0.307683198, 0.321233654, 0.335277572, 0.34984456,  0.364944486,
    0.380587219, 0.396792499, 0.413570195, 0.430940044, 0.448921786,
    0.467535159, 0.486790032, 0.506706144, 0.527293363, 0.548581298,
    0.570589687, 0.593328399, 0.616817172, 0.641085616, 0.666133728,
    0.692000987, 0.718707131, 0.746252159, 0.774675549, 0.803997039,
    0.834236368, 0.865413274, 0.897557365, 0.93067851,  0.964806316,
    0.999950654
])


# ============================================
# КОНФИГУРАЦИЯ ГРАФИКОВ
# ============================================

plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.labelweight': 'bold',
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.0,
    'xtick.color': 'black',
    'ytick.color': 'black',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'legend.frameon': True,
    'legend.framealpha': 0.9,
    'legend.edgecolor': 'black',
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

st.set_page_config(
    page_title="OCV Transport Number Analysis",
    page_icon="⚡",
    layout="wide"
)


# ============================================
# ИНИЦИАЛИЗАЦИЯ SESSION STATE
# ============================================

def initialize_session_state():
    defaults = {
        'mode': 'A',
        'experimental_data': None,
        'data_loaded': False,
        'analysis_done': False,

        'air_pH2O_mode': 'bubbler',
        'air_bubbler_T': 25.0,
        'air_pH2O_direct': 0.0313,

        'fuel_pH2O_mode': 'bubbler',
        'fuel_bubbler_T': 25.0,
        'fuel_pH2O_direct': 0.0313,
        'fuel_H2_fraction': 1.0,

        'fixed_T_BC': 700.0,

        'results': None,

        'grid_step': 0.01,
        'tolerance_mode': 'auto',
        'tolerance_value_mV': 5.0,
        'min_solutions': 10,
        'max_tolerance_mV': 50.0,

        'style': {
            'scenario1_color': '#1f77b4',
            'scenario2_color': '#2ca02c',
            'scenario3_color': '#d62728',
            'te_color': '#9467bd',
            'common_tH_color': '#ff7f0e',
            'common_tO_color': '#17becf',
            'common_ti_color': '#8c564b',
            'marker_size': 40,
            'line_width': 2,
            'point_alpha': 0.85,
            'band_alpha': 0.18
        }
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_session_state()


# ============================================
# ФИЗИЧЕСКИЕ ФУНКЦИИ
# ============================================

def logK(T_C: float) -> float:
    """log10 K для реакции H2O = H2 + 0.5 O2, T в °C."""
    T_C = float(T_C)
    T_K = T_C + 273.15
    return (-1783.32 + 0.40561 * T_K) / (1 + 0.135996 * T_K)


def K_eq(T_C: float) -> float:
    """Константа равновесия K реакции H2O = H2 + 0.5 O2."""
    return 10.0 ** logK(T_C)


def pH2O_from_bubbler(T_bubbler_C: float) -> float:
    """Парциальное давление H2O по температуре барботёра (интерполяция таблицы)."""
    T_bubbler_C = float(T_bubbler_C)
    if T_bubbler_C <= T_BUBBLER[0]:
        return float(PH2O_TABLE[0])
    if T_bubbler_C >= T_BUBBLER[-1]:
        return float(PH2O_TABLE[-1])
    return float(np.interp(T_bubbler_C, T_BUBBLER, PH2O_TABLE))


def _clamp_pH2O(x: float) -> float:
    """Принудительное приведение pH2O к допустимому диапазону."""
    x = float(x)
    if not np.isfinite(x):
        return np.nan
    return min(max(x, 1e-9), 0.999)


def compute_pressures(T_C: float,
                     air_pH2O: float,
                     fuel_pH2O: float,
                     fuel_H2_fraction: float = 1.0
                     ) -> Dict[str, float]:
    """
    Расчёт парциальных давлений на обеих сторонах.
    Все входы принудительно приводятся к float и clamp.
    """
    T_C = float(T_C)
    air_pH2O = _clamp_pH2O(air_pH2O)
    fuel_pH2O = _clamp_pH2O(fuel_pH2O)
    fuel_H2_fraction = float(fuel_H2_fraction)
    fuel_H2_fraction = min(max(fuel_H2_fraction, 1e-6), 1.0)

    K = K_eq(T_C)

    # Воздух (катод)
    pO2_air = 0.21 * (1.0 - air_pH2O)
    if pO2_air > 0:
        pH2_air = K * air_pH2O / np.sqrt(pO2_air)
    else:
        pH2_air = 0.0

    # Топливо (анод)
    if fuel_H2_fraction >= 1.0 - 1e-9:
        # чистый H2 + H2O (без инертного газа)
        if fuel_pH2O < 1.0:
            pO2_fuel = (K * fuel_pH2O / (1.0 - fuel_pH2O)) ** 2
        else:
            pO2_fuel = 0.0
        pH2_fuel = 1.0 - fuel_pH2O
    else:
        # H2 + inert + H2O
        pH2_fuel = fuel_H2_fraction * (1.0 - fuel_pH2O)
        denom = pH2_fuel * (1.0 - fuel_pH2O)
        if denom > 0:
            pO2_fuel = (K * fuel_pH2O / denom) ** 2
        else:
            pO2_fuel = 0.0

    return {
        'pO2_air': pO2_air,
        'pH2_air': pH2_air,
        'pO2_fuel': pO2_fuel,
        'pH2_fuel': pH2_fuel,
        'air_pH2O': air_pH2O,
        'fuel_pH2O': fuel_pH2O,
        'fuel_H2_fraction': fuel_H2_fraction,
        'K': K
    }


def EO_value(T_C: float, p: Dict[str, float]) -> float:
    """Термодинамическая ЭДС кислородионной ячейки, В."""
    T_K = float(T_C) + 273.15
    if p['pO2_air'] <= 0 or p['pO2_fuel'] <= 0:
        return np.nan
    return R_GAS * T_K / (4.0 * F_FARADAY) * np.log(p['pO2_air'] / p['pO2_fuel'])


def EH_value(T_C: float, p: Dict[str, float]) -> float:
    """Термодинамическая ЭДС протонной ячейки (классическая форма), В."""
    T_K = float(T_C) + 273.15
    if p['pH2_air'] <= 0 or p['pH2_fuel'] <= 0:
        return np.nan
    return R_GAS * T_K / (2.0 * F_FARADAY) * np.log(p['pH2_fuel'] / p['pH2_air'])


def EH2O_value(T_C: float, p: Dict[str, float]) -> float:
    """ЭДС в форме через влажности, В."""
    T_K = float(T_C) + 273.15
    if p['air_pH2O'] <= 0 or p['fuel_pH2O'] <= 0:
        return np.nan
    return R_GAS * T_K / (2.0 * F_FARADAY) * np.log(p['fuel_pH2O'] / p['air_pH2O'])


# ============================================
# СЦЕНАРИИ РАСЧЁТА ЧИСЕЛ ПЕРЕНОСА
# ============================================

def scenario_1(Emeas: float, EH: float) -> Optional[Dict[str, float]]:
    """Сценарий ❶: H+ + e-. tH = Emeas/EH."""
    if EH is None or not np.isfinite(EH) or abs(EH) < 1e-9:
        return None
    tH = Emeas / EH
    if tH < 0.0 or tH > 1.0:
        return None
    return {'tH': tH, 'tO': 0.0, 'tion': tH, 'te': 1.0 - tH}


def scenario_2(Emeas: float, EO: float, EH: float) -> Optional[Dict[str, float]]:
    """Сценарий ❷: O2- + H+. tH = (Emeas - EO)/(EH - EO)."""
    if EH is None or EO is None:
        return None
    if not np.isfinite(EH) or not np.isfinite(EO):
        return None
    if abs(EH - EO) < 1e-9:
        return None
    tH = (Emeas - EO) / (EH - EO)
    tO = 1.0 - tH
    if tH < 0.0 or tH > 1.0 or tO < 0.0 or tO > 1.0:
        return None
    return {'tH': tH, 'tO': tO, 'tion': 1.0, 'te': 0.0}


def scenario_3_grid(Emeas: float, EO: float, EH2O: float,
                   step: float = 0.01) -> List[Dict[str, float]]:
    """Сценарий ❸: численный перебор (ti, tH)."""
    if not np.isfinite(EO) or not np.isfinite(EH2O):
        return []

    solutions = []
    n_steps = int(round(1.0 / step))
    for i in range(n_steps + 1):
        ti = i * step
        for j in range(int(round(ti / step)) + 1):
            tH = j * step
            tO = ti - tH
            if tO < -1e-12:
                continue
            if tH < 0.0 or tO < 0.0 or ti > 1.0 + 1e-12:
                continue
            E_model = ti * EO + tH * EH2O
            err = E_model - Emeas
            solutions.append({
                'ti': ti, 'tH': tH, 'tO': max(tO, 0.0),
                'te': 1.0 - ti, 'E_model': E_model, 'err': err
            })
    return solutions


def filter_solutions_by_tolerance(solutions: List[Dict[str, float]],
                                  tolerance: float
                                  ) -> List[Dict[str, float]]:
    return [s for s in solutions if abs(s['err']) <= tolerance]


def auto_select_tolerance(Emeas: float, EO: float, EH2O: float,
                         step: float, min_solutions: int,
                         max_tol: float) -> float:
    """Автоподбор δ: минимальное δ, дающее >= min_solutions решений."""
    sols = scenario_3_grid(Emeas, EO, EH2O, step=step)
    if not sols:
        return max_tol
    errs = np.array(sorted(abs(s['err']) for s in sols))
    if min_solutions <= 0:
        min_solutions = 1
    if min_solutions > len(errs):
        return max_tol
    tol = errs[min_solutions - 1]
    if tol < 1e-9:
        tol = max(errs[0], 1e-6)
    return min(tol, max_tol)


# ============================================
# ГЛАВНЫЙ РАСЧЁТ
# ============================================

def run_analysis(data: np.ndarray,
                mode: str,
                fixed_T_BC: float,
                air_pH2O: float,
                fuel_pH2O: float,
                fuel_H2_fraction: float,
                grid_step: float,
                tolerance_mode: str,
                tolerance_value_mV: float,
                min_solutions: int,
                max_tolerance_mV: float
                ) -> Dict[str, Any]:
    """
    Прогон всех трёх сценариев для каждой точки данных.
    Все параметры передаются явно, без обращения к st.session_state.
    """
    xs = np.asarray(data[:, 0], dtype=float)
    ocvs = np.asarray(data[:, 1], dtype=float)

    scenario1, scenario2, scenario3 = [], [], []
    s3_grids = []

    # Отладочный вывод всех входных параметров
    print("=" * 70)
    print(f"[RUN_ANALYSIS] mode={mode}")
    print(f"[RUN_ANALYSIS] fixed_T_BC={fixed_T_BC}")
    print(f"[RUN_ANALYSIS] air_pH2O={air_pH2O}")
    print(f"[RUN_ANALYSIS] fuel_pH2O={fuel_pH2O}")
    print(f"[RUN_ANALYSIS] fuel_H2_fraction={fuel_H2_fraction}")
    print(f"[RUN_ANALYSIS] grid_step={grid_step}, "
          f"tol_mode={tolerance_mode}, tol_mV={tolerance_value_mV}, "
          f"min_sol={min_solutions}, max_tol_mV={max_tolerance_mV}")
    print(f"[RUN_ANALYSIS] N points = {len(xs)}")
    print("=" * 70)

    for idx_pt, (x, ocv) in enumerate(zip(xs, ocvs)):
        x = float(x)
        ocv = float(ocv)

        # Определяем T_C, p_air, p_fuel для этой точки
        if mode == 'A':
            T_C = x
            p_air = air_pH2O
            p_fuel = fuel_pH2O
        elif mode == 'B':
            T_C = float(fixed_T_BC)
            p_air = x
            p_fuel = fuel_pH2O
        elif mode == 'C':
            T_C = float(fixed_T_BC)
            p_air = air_pH2O
            p_fuel = x
        else:
            T_C = x
            p_air = air_pH2O
            p_fuel = fuel_pH2O

        p = compute_pressures(T_C, p_air, p_fuel, fuel_H2_fraction)
        EO = EO_value(T_C, p)
        EH = EH_value(T_C, p)
        EH2O = EH2O_value(T_C, p)

        # Отладочный вывод промежуточных величин
        print(f"[PT {idx_pt}] x={x:.6g}, ocv={ocv:.6f}")
        print(f"  T_C={T_C:.2f}, p_air={p_air:.6f}, p_fuel={p_fuel:.6f}, "
              f"H2_frac={fuel_H2_fraction:.4f}")
        print(f"  K(T)={p['K']:.6e}")
        print(f"  p'O2={p['pO2_air']:.6e}, p'H2={p['pH2_air']:.6e}")
        print(f"  p''O2={p['pO2_fuel']:.6e}, p''H2={p['pH2_fuel']:.6e}")
        print(f"  EO={EO:.6f}, EH={EH:.6f}, EH2O={EH2O:.6f}")

        s1 = scenario_1(ocv, EH)
        s2 = scenario_2(ocv, EO, EH)
        scenario1.append(s1)
        scenario2.append(s2)
        print(f"  s1={s1}")
        print(f"  s2={s2}")

        # Сценарий 3
        grid = scenario_3_grid(ocv, EO, EH2O, step=grid_step)

        if tolerance_mode == 'auto':
            tol_V = auto_select_tolerance(ocv, EO, EH2O,
                                          step=grid_step,
                                          min_solutions=min_solutions,
                                          max_tol=max_tolerance_mV * 1e-3)
        else:
            tol_V = tolerance_value_mV * 1e-3

        sols = filter_solutions_by_tolerance(grid, tol_V)
        print(f"  s3: tol={tol_V*1e3:.3f} mV, N_sol={len(sols)}")

        s3_grids.append({
            'T_C': T_C,
            'K': p['K'],
            'pO2_air': p['pO2_air'],
            'pH2_air': p['pH2_air'],
            'pO2_fuel': p['pO2_fuel'],
            'pH2_fuel': p['pH2_fuel'],
            'air_pH2O_used': p['air_pH2O'],
            'fuel_pH2O_used': p['fuel_pH2O'],
            'fuel_H2_fraction_used': p['fuel_H2_fraction'],
            'EO': EO, 'EH': EH, 'EH2O': EH2O,
            'OCV': ocv,
            'tolerance_V': tol_V,
            'solutions': sols
        })

        if sols:
            ti_arr = np.array([s['ti'] for s in sols])
            tH_arr = np.array([s['tH'] for s in sols])
            tO_arr = np.array([s['tO'] for s in sols])
            te_arr = np.array([s['te'] for s in sols])
            s3_summary = {
                'ti_min': ti_arr.min(), 'ti_max': ti_arr.max(),
                'tH_min': tH_arr.min(), 'tH_max': tH_arr.max(),
                'tO_min': tO_arr.min(), 'tO_max': tO_arr.max(),
                'te_min': te_arr.min(), 'te_max': te_arr.max(),
                'n_sol': len(sols),
                'tolerance_V': tol_V
            }
        else:
            s3_summary = {
                'ti_min': np.nan, 'ti_max': np.nan,
                'tH_min': np.nan, 'tH_max': np.nan,
                'tO_min': np.nan, 'tO_max': np.nan,
                'te_min': np.nan, 'te_max': np.nan,
                'n_sol': 0,
                'tolerance_V': tol_V
            }
        scenario3.append(s3_summary)

    return {
        'mode': mode,
        'xs': xs,
        'ocvs': ocvs,
        'scenario1': scenario1,
        'scenario2': scenario2,
        'scenario3': scenario3,
        's3_grids': s3_grids,
        'fixed_T_BC': float(fixed_T_BC),
        'air_pH2O': float(air_pH2O) if np.isfinite(air_pH2O) else np.nan,
        'fuel_pH2O': float(fuel_pH2O) if np.isfinite(fuel_pH2O) else np.nan,
        'fuel_H2_fraction': float(fuel_H2_fraction),
        'grid_step': float(grid_step),
        'tolerance_mode': tolerance_mode,
        'tolerance_value_mV': float(tolerance_value_mV),
        'min_solutions': int(min_solutions),
        'max_tolerance_mV': float(max_tolerance_mV)
    }


# ============================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ ГРАФИКОВ
# ============================================

def _x_label(mode: str) -> str:
    if mode == 'A':
        return 'Temperature (°C)'
    if mode == 'B':
        return "p'H₂O (air side)"
    return "p''H₂O (fuel side)"


def _extract_series(results: Dict[str, Any], key: str) -> np.ndarray:
    xs = results['xs']
    out = np.full_like(xs, np.nan, dtype=float)

    if key in ('tH', 'tO', 'tion', 'te'):
        for i, s in enumerate(results['scenario1']):
            if s is not None:
                out[i] = s[key]
        return out

    if key.startswith('s2_'):
        name = key[3:]
        for i, s in enumerate(results['scenario2']):
            if s is not None:
                out[i] = s[name]
        return out

    if key.startswith('s3_'):
        field = key[3:]
        for i, s in enumerate(results['scenario3']):
            out[i] = s.get(field, np.nan)
        return out

    return out


def _plot_common_band(ax, xs, arrays, color, alpha):
    mins, maxs = [], []
    for arr in arrays:
        v = arr[~np.isnan(arr)]
        if v.size:
            mins.append(arr)
            maxs.append(arr)
    if not mins:
        return
    stack_min = np.vstack(mins)
    stack_max = np.vstack(maxs)
    lo = np.nanmax(stack_min, axis=0)
    hi = np.nanmin(stack_max, axis=0)
    mask = (~np.isnan(lo)) & (~np.isnan(hi)) & (hi >= lo)
    if not mask.any():
        return
    idx = np.where(mask)[0]
    if idx.size == 0:
        return
    splits = np.where(np.diff(idx) > 1)[0]
    segments = np.split(idx, splits + 1)
    for seg in segments:
        if seg.size < 2:
            continue
        x_seg = xs[seg]
        lo_seg = lo[seg]
        hi_seg = hi[seg]
        ax.fill_between(x_seg, lo_seg, hi_seg, color=color,
                        alpha=alpha, zorder=1)


# ============================================
# ГРАФИКИ
# ============================================

def create_overview_plot(results: Dict[str, Any], style: Dict[str, Any]) -> plt.Figure:
    xs = results['xs']
    xlabel = _x_label(results['mode'])

    x_pad = 0.02 * (xs.max() - xs.min()) if xs.max() > xs.min() else 1.0
    xlim = (xs.min() - x_pad, xs.max() + x_pad)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)
    titles = ['tH (proton)', 'tO (oxide-ion)', 'tion = tH + tO']

    s1_tH = _extract_series(results, 'tH')
    s2_tH = _extract_series(results, 's2_tH')
    s3_tH_min = _extract_series(results, 's3_tH_min')
    s3_tH_max = _extract_series(results, 's3_tH_max')

    ax = axes[0]
    if (~np.isnan(s1_tH)).any():
        ax.plot(xs, s1_tH, 'o-', color=style['scenario1_color'],
                markersize=5, linewidth=style['line_width'] - 0.5,
                alpha=style['point_alpha'], label='Scenario ❶ (H⁺+e⁻)')
    if (~np.isnan(s2_tH)).any():
        ax.plot(xs, s2_tH, 's-', color=style['scenario2_color'],
                markersize=5, linewidth=style['line_width'] - 0.5,
                alpha=style['point_alpha'], label='Scenario ❷ (O²⁻+H⁺)')

    valid = ~np.isnan(s3_tH_min) & ~np.isnan(s3_tH_max)
    if valid.any():
        ax.fill_between(xs, s3_tH_min, s3_tH_max,
                        color=style['scenario3_color'],
                        alpha=style['band_alpha'], label='Scenario ❸ range')

    _plot_common_band(ax, xs, [s1_tH, s2_tH, s3_tH_min, s3_tH_max],
                      color=style['common_tH_color'], alpha=style['band_alpha'])

    ax.set_title(titles[0]); ax.set_ylabel('tH')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(xlim)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=8)

    s2_tO = _extract_series(results, 's2_tO')
    s3_tO_min = _extract_series(results, 's3_tO_min')
    s3_tO_max = _extract_series(results, 's3_tO_max')

    ax = axes[1]
    if (~np.isnan(s2_tO)).any():
        ax.plot(xs, s2_tO, 's-', color=style['scenario2_color'],
                markersize=5, linewidth=style['line_width'] - 0.5,
                alpha=style['point_alpha'], label='Scenario ❷ (O²⁻+H⁺)')
    if valid.any():
        ax.fill_between(xs, s3_tO_min, s3_tO_max,
                        color=style['scenario3_color'],
                        alpha=style['band_alpha'], label='Scenario ❸ range')
    _plot_common_band(ax, xs, [s2_tO, s3_tO_min, s3_tO_max],
                      color=style['common_tO_color'], alpha=style['band_alpha'])
    ax.set_title(titles[1]); ax.set_ylabel('tO')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(xlim)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=8)

    s1_tion = s1_tH.copy()
    s2_tion = np.ones_like(xs)
    for i, s in enumerate(results['scenario2']):
        if s is None:
            s2_tion[i] = np.nan
    s3_ti_min = _extract_series(results, 's3_ti_min')
    s3_ti_max = _extract_series(results, 's3_ti_max')

    ax = axes[2]
    if (~np.isnan(s1_tion)).any():
        ax.plot(xs, s1_tion, 'o-', color=style['scenario1_color'],
                markersize=5, linewidth=style['line_width'] - 0.5,
                alpha=style['point_alpha'], label='Scenario ❶ (H⁺+e⁻)')
    if (~np.isnan(s2_tion)).any():
        ax.plot(xs, s2_tion, 's-', color=style['scenario2_color'],
                markersize=5, linewidth=style['line_width'] - 0.5,
                alpha=style['point_alpha'], label='Scenario ❷ (O²⁻+H⁺)')
    if valid.any():
        ax.fill_between(xs, s3_ti_min, s3_ti_max,
                        color=style['scenario3_color'],
                        alpha=style['band_alpha'], label='Scenario ❸ range')
    _plot_common_band(ax, xs, [s1_tion, s2_tion, s3_ti_min, s3_ti_max],
                      color=style['common_ti_color'], alpha=style['band_alpha'])
    ax.set_title(titles[2]); ax.set_ylabel('tion')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(xlim)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=8)

    for a in axes:
        a.set_xlabel(xlabel)

    plt.tight_layout()
    fig.set_dpi(600)
    return fig


def create_scenario_plot(results: Dict[str, Any], scenario_idx: int,
                        style: Dict[str, Any]) -> plt.Figure:
    xs = results['xs']
    xlabel = _x_label(results['mode'])

    x_pad = 0.02 * (xs.max() - xs.min()) if xs.max() > xs.min() else 1.0
    xlim = (xs.min() - x_pad, xs.max() + x_pad)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    has_data = False

    if scenario_idx == 1:
        tH = _extract_series(results, 'tH')
        tion = tH.copy()
        te = np.where(~np.isnan(tH), 1.0 - tH, np.nan)
        if (~np.isnan(tH)).any():
            ax.plot(xs, tH, 'o-', color=style['scenario1_color'],
                    markersize=6, linewidth=style['line_width'], label='tH')
            ax.plot(xs, tion, '^--', color=style['scenario1_color'],
                    markersize=5, linewidth=style['line_width'] - 0.5,
                    alpha=0.7, label='tion = tH')
            ax.plot(xs, te, 'v:', color=style['te_color'],
                    markersize=5, linewidth=style['line_width'] - 0.5,
                    label='te = 1 - tH')
            has_data = True
        ax.set_title('Scenario ❶: H⁺ + e⁻')

    elif scenario_idx == 2:
        tH = _extract_series(results, 's2_tH')
        tO = _extract_series(results, 's2_tO')
        if (~np.isnan(tH)).any():
            ax.plot(xs, tH, 's-', color=style['scenario2_color'],
                    markersize=6, linewidth=style['line_width'], label='tH')
            ax.plot(xs, tO, 'D--', color=style['scenario2_color'],
                    markersize=5, linewidth=style['line_width'] - 0.5,
                    alpha=0.7, label='tO')
            has_data = True
        ax.set_title('Scenario ❷: O²⁻ + H⁺ (tion = 1, te = 0)')

    else:
        ti_min = _extract_series(results, 's3_ti_min')
        ti_max = _extract_series(results, 's3_ti_max')
        tH_min = _extract_series(results, 's3_tH_min')
        tH_max = _extract_series(results, 's3_tH_max')
        tO_min = _extract_series(results, 's3_tO_min')
        tO_max = _extract_series(results, 's3_tO_max')

        valid = ~np.isnan(ti_min)
        if valid.any():
            ax.fill_between(xs, ti_min, ti_max, color=style['scenario3_color'],
                            alpha=style['band_alpha'], label='tion range')
            ax.fill_between(xs, tH_min, tH_max, color=style['common_tH_color'],
                            alpha=style['band_alpha'], label='tH range')
            ax.fill_between(xs, tO_min, tO_max, color=style['common_tO_color'],
                            alpha=style['band_alpha'], label='tO range')
            ax.plot(xs, (tH_min + tH_max) / 2, 'o-',
                    color=style['scenario3_color'], markersize=5,
                    linewidth=style['line_width'], label='tH (mid)')
            ax.plot(xs, (tO_min + tO_max) / 2, 's-',
                    color=style['common_tO_color'], markersize=5,
                    linewidth=style['line_width'] - 0.5, label='tO (mid)')
            has_data = True
        ax.set_title('Scenario ❸: O²⁻ + H⁺ + e⁻ (range)')

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Transport number')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(xlim)

    if not has_data:
        ax.text(0.5, 0.5, 'Нет физически допустимых точек\nдля этого сценария',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=11, fontweight='bold', color='gray',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    ax.grid(True, alpha=0.3, linestyle='--')
    if has_data:
        ax.legend(loc='best', fontsize=9)

    plt.tight_layout()
    fig.set_dpi(600)
    return fig


def create_te_plot(results: Dict[str, Any], style: Dict[str, Any]) -> plt.Figure:
    xs = results['xs']
    xlabel = _x_label(results['mode'])

    x_pad = 0.02 * (xs.max() - xs.min()) if xs.max() > xs.min() else 1.0
    xlim = (xs.min() - x_pad, xs.max() + x_pad)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    has_data = False

    tH1 = _extract_series(results, 'tH')
    te1 = np.where(~np.isnan(tH1), 1.0 - tH1, np.nan)
    if (~np.isnan(te1)).any():
        ax.plot(xs, te1, 'o-', color=style['scenario1_color'],
                markersize=6, linewidth=style['line_width'],
                label='te, Scenario ❶')
        has_data = True

    tH2 = _extract_series(results, 's2_tH')
    te2 = np.where(~np.isnan(tH2), 0.0, np.nan)
    if (~np.isnan(te2)).any():
        ax.plot(xs, te2, 's-', color=style['scenario2_color'],
                markersize=6, linewidth=style['line_width'],
                label='te, Scenario ❷ (=0)')
        has_data = True

    te_min = _extract_series(results, 's3_te_min')
    te_max = _extract_series(results, 's3_te_max')
    if (~np.isnan(te_min)).any():
        ax.fill_between(xs, te_min, te_max, color=style['scenario3_color'],
                        alpha=style['band_alpha'], label='te, Scenario ❸ range')
        has_data = True

    ax.set_xlabel(xlabel)
    ax.set_ylabel('te')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(xlim)
    ax.set_title('Electronic transport number')

    if not has_data:
        ax.text(0.5, 0.5, 'Нет физически допустимых точек',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=11, fontweight='bold', color='gray',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    ax.grid(True, alpha=0.3, linestyle='--')
    if has_data:
        ax.legend(loc='best', fontsize=9)

    plt.tight_layout()
    fig.set_dpi(600)
    return fig


def create_decision_map_scatter(grid_solutions: List[Dict[str, float]],
                               EO: float, EH2O: float, Emeas: float,
                               style: Dict[str, Any]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 5))

    if grid_solutions:
        ti = np.array([s['ti'] for s in grid_solutions])
        tH = np.array([s['tH'] for s in grid_solutions])
        te = np.array([s['te'] for s in grid_solutions])
        sc = ax.scatter(ti, tH, c=te, cmap='viridis', s=30,
                        edgecolor='black', linewidth=0.3, vmin=0, vmax=1)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label('te = 1 - ti', fontweight='bold')

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5,
            label='tO = 0 boundary')
    ax.fill_between([0, 1], [0, 1], [1, 1], color='gray',
                    alpha=0.08, label='tO < 0 (forbidden)')

    ax.set_xlabel('tion = ti', fontweight='bold')
    ax.set_ylabel('tH', fontweight='bold')
    ax.set_title(f'Decision map ❸ (points)\nEmeas = {Emeas*1000:.2f} mV',
                fontweight='bold')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower right', fontsize=8)

    plt.tight_layout()
    fig.set_dpi(600)
    return fig


def create_decision_map_contour(EO: float, EH2O: float, Emeas: float,
                               style: Dict[str, Any]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 5))

    ti_vals = np.linspace(0, 1, 200)
    tH_vals = np.linspace(0, 1, 200)
    TI, TH = np.meshgrid(ti_vals, tH_vals)
    TO = TI - TH
    mask = TO >= -1e-9
    E_model = TI * EO + TH * EH2O
    E_model_masked = np.where(mask, E_model, np.nan)

    levels = np.linspace(np.nanmin(E_model_masked),
                        np.nanmax(E_model_masked), 15)
    cs = ax.contourf(TI, TH, E_model_masked, levels=levels,
                    cmap='coolwarm', alpha=0.7)
    cbar = fig.colorbar(cs, ax=ax)
    cbar.set_label('E_model (V)', fontweight='bold')

    cs2 = ax.contour(TI, TH, E_model_masked, levels=[Emeas],
                    colors='black', linewidths=2.5)
    ax.clabel(cs2, fmt='E = %.3f V', fontsize=9)

    ax.fill_between([0, 1], [0, 1], [1, 1], color='gray', alpha=0.25,
                    hatch='//', label='tO < 0 (forbidden)')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.7)

    ax.set_xlabel('tion = ti', fontweight='bold')
    ax.set_ylabel('tH', fontweight='bold')
    ax.set_title(f'Decision map ❸ (contour)\nEmeas = {Emeas*1000:.2f} mV',
                fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.legend(loc='lower right', fontsize=8)

    plt.tight_layout()
    fig.set_dpi(600)
    return fig


# ============================================
# ПАРСИНГ ДАННЫХ
# ============================================

@st.cache_data(ttl=3600, show_spinner=False)
def parse_data_cached(text: str) -> np.ndarray:
    lines = text.strip().split('\n')
    rows = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        for sep in ['\t', ' ', ',', ';']:
            if sep in line:
                parts = [p.strip() for p in line.split(sep) if p.strip()]
                if len(parts) >= 2:
                    try:
                        rows.append([float(parts[0]), float(parts[1])])
                        break
                    except ValueError:
                        continue
    if not rows:
        raise ValueError("Не удалось распарсить данные. Проверьте формат.")
    return np.array(rows)


# ============================================
# ИНТЕРФЕЙС
# ============================================

def main():
    st.title("⚡ OCV Transport Number Analysis")
    st.markdown("Оценка чисел переноса протонпроводящих электролитов по данным OCV")

    with st.sidebar:
        st.header("1. Режим эксперимента")
        mode = st.radio(
            "Режим измерений:",
            options=['A', 'B', 'C'],
            format_func=lambda x: {
                'A': 'A. OCV(T) — фиксированные атмосферы',
                'B': "B. OCV(p'H₂O) — варьируется воздух",
                'C': "C. OCV(p''H₂O) — варьируется топливо"
            }[x],
            index=['A', 'B', 'C'].index(st.session_state.mode),
            key='mode_radio'
        )
        st.session_state.mode = mode

        fixed_T_BC = None
        if mode in ('B', 'C'):
            fixed_T_BC = st.number_input(
                "Температура измерения T (°C):",
                value=float(st.session_state.fixed_T_BC),
                step=10.0, format="%.1f",
                key='fixed_T_BC_input'
            )
            st.session_state.fixed_T_BC = fixed_T_BC

        st.divider()
        st.header("2. Газовая атмосфера")

        air_varies = (mode == 'B')
        fuel_varies = (mode == 'C')

        # ---------- Воздух ----------
        st.subheader("Воздух (катод)")

        if air_varies:
            st.info("p'H₂O задаётся в загружаемых данных (ось X). "
                    "Поля ввода отключены для режима B.")
            air_pH2O = np.nan
        else:
            air_mode = st.radio(
                "Задание p'H₂O:",
                options=['bubbler', 'direct'],
                format_func=lambda x: 'Через T барботёра' if x == 'bubbler' else 'Напрямую',
                index=0 if st.session_state.air_pH2O_mode == 'bubbler' else 1,
                key='air_mode_radio'
            )
            st.session_state.air_pH2O_mode = air_mode

            if air_mode == 'bubbler':
                air_T = st.number_input(
                    "T барботёра воздуха (°C):",
                    value=float(st.session_state.air_bubbler_T),
                    min_value=0.0, max_value=100.0, step=1.0,
                    key='air_bubbler_T_input'
                )
                st.session_state.air_bubbler_T = air_T
                air_pH2O = pH2O_from_bubbler(air_T)
                st.caption(f"→ p'H₂O = {air_pH2O:.5f}")
            else:
                air_pH2O = st.number_input(
                    "p'H₂O (воздух):",
                    value=float(st.session_state.air_pH2O_direct),
                    min_value=1e-6, max_value=0.99, step=0.001,
                    format="%.5f", key='air_pH2O_direct_input'
                )
                st.session_state.air_pH2O_direct = air_pH2O

        # ---------- Топливо ----------
        st.subheader("Топливо (анод)")

        if fuel_varies:
            st.info("p''H₂O задаётся в загружаемых данных (ось X). "
                    "Поля ввода отключены для режима C.")
            fuel_pH2O = np.nan
            fuel_H2_fraction = 1.0   # в режиме C топливо чистое H2, разбавление не задаётся
        else:
            fuel_mode = st.radio(
                "Задание p''H₂O:",
                options=['bubbler', 'direct'],
                format_func=lambda x: 'Через T барботёра' if x == 'bubbler' else 'Напрямую',
                index=0 if st.session_state.fuel_pH2O_mode == 'bubbler' else 1,
                key='fuel_mode_radio'
            )
            st.session_state.fuel_pH2O_mode = fuel_mode

            if fuel_mode == 'bubbler':
                fuel_T = st.number_input(
                    "T барботёра топлива (°C):",
                    value=float(st.session_state.fuel_bubbler_T),
                    min_value=0.0, max_value=100.0, step=1.0,
                    key='fuel_bubbler_T_input'
                )
                st.session_state.fuel_bubbler_T = fuel_T
                fuel_pH2O = pH2O_from_bubbler(fuel_T)
                st.caption(f"→ p''H₂O = {fuel_pH2O:.5f}")
            else:
                fuel_pH2O = st.number_input(
                    "p''H₂O (топливо):",
                    value=float(st.session_state.fuel_pH2O_direct),
                    min_value=1e-6, max_value=0.99, step=0.001,
                    format="%.5f", key='fuel_pH2O_direct_input'
                )
                st.session_state.fuel_pH2O_direct = fuel_pH2O

            fuel_H2_fraction = st.slider(
                "Исходная мольная доля H₂ в сухом топливе (остальное — inert):",
                min_value=0.01, max_value=1.0,
                value=float(st.session_state.fuel_H2_fraction),
                step=0.01,
                key='fuel_H2_fraction_slider',
                help="1.0 = чистый H₂. Если < 1 — разбавление инертным газом."
            )
            st.session_state.fuel_H2_fraction = fuel_H2_fraction

        st.divider()
        st.header("3. Параметры перебора (сценарий ❸)")
        grid_step = st.select_slider(
            "Шаг сетки по ti и tH:",
            options=[0.005, 0.01, 0.02, 0.05],
            value=st.session_state.grid_step,
            key='grid_step_slider'
        )
        st.session_state.grid_step = grid_step

        tolerance_mode = st.radio(
            "Допуск δ:",
            options=['auto', 'manual'],
            format_func=lambda x: 'Автоматически' if x == 'auto' else 'Вручную',
            index=0 if st.session_state.tolerance_mode == 'auto' else 1,
            key='tol_mode_radio'
        )
        st.session_state.tolerance_mode = tolerance_mode

        if tolerance_mode == 'auto':
            min_solutions = st.number_input(
                "Минимум решений:",
                min_value=1, max_value=500,
                value=int(st.session_state.min_solutions),
                step=1,
                key='min_sol_input'
            )
            st.session_state.min_solutions = min_solutions

            max_tolerance_mV = st.number_input(
                "Верхняя граница δ (мВ):",
                min_value=1.0, max_value=500.0,
                value=float(st.session_state.max_tolerance_mV),
                step=1.0,
                key='max_tol_input'
            )
            st.session_state.max_tolerance_mV = max_tolerance_mV

            tolerance_value_mV = st.session_state.tolerance_value_mV
        else:
            tolerance_value_mV = st.number_input(
                "δ (мВ):",
                min_value=0.1, max_value=500.0,
                value=float(st.session_state.tolerance_value_mV),
                step=0.5,
                key='tol_value_input'
            )
            st.session_state.tolerance_value_mV = tolerance_value_mV
            min_solutions = st.session_state.min_solutions
            max_tolerance_mV = st.session_state.max_tolerance_mV

        st.divider()
        st.header("4. Данные OCV")
        data_option = st.radio(
            "Способ ввода:",
            ['Пример', 'Вручную', 'Файл'],
            index=0, key='data_option_radio'
        )

        if mode == 'A':
            x_help = "T (°C)  OCV (В)"
            x_example = """500\t0.842
550\t0.851
600\t0.858
650\t0.862
700\t0.864
750\t0.863
800\t0.860
850\t0.855
900\t0.848"""
        elif mode == 'B':
            x_help = "p'H₂O  OCV (В)"
            x_example = """0.01\t0.812
0.02\t0.826
0.03\t0.835
0.05\t0.847
0.08\t0.858
0.12\t0.868
0.17\t0.876"""
        else:
            x_help = "p''H₂O  OCV (В)"
            x_example = """0.01\t0.902
0.02\t0.878
0.03\t0.862
0.05\t0.842
0.08\t0.820
0.12\t0.799
0.17\t0.780"""

        if data_option == 'Пример':
            if st.button("Загрузить пример", key='load_example'):
                try:
                    raw = parse_data_cached(x_example)
                    st.session_state.experimental_data = raw
                    st.session_state.data_loaded = True
                    st.session_state.analysis_done = False
                    st.success(f"Загружено {len(raw)} точек")
                except Exception as e:
                    st.error(f"Ошибка: {e}")
        elif data_option == 'Вручную':
            data_text = st.text_area(
                f"Введите данные ({x_help}):",
                value=x_example, height=200,
                key='manual_data_input'
            )
            if st.button("Загрузить данные", key='load_manual'):
                try:
                    raw = parse_data_cached(data_text)
                    st.session_state.experimental_data = raw
                    st.session_state.data_loaded = True
                    st.session_state.analysis_done = False
                    st.success(f"Загружено {len(raw)} точек")
                except Exception as e:
                    st.error(f"Ошибка: {e}")
        else:
            uploaded = st.file_uploader("Загрузите файл (txt/csv/dat)",
                                       type=['txt', 'csv', 'dat'],
                                       key='file_uploader')
            if uploaded is not None:
                try:
                    raw = parse_data_cached(uploaded.getvalue().decode())
                    st.session_state.experimental_data = raw
                    st.session_state.data_loaded = True
                    st.session_state.analysis_done = False
                    st.success(f"Загружено {len(raw)} точек")
                except Exception as e:
                    st.error(f"Ошибка: {e}")

        # ---- Итоговое подтверждение входных параметров перед запуском ----
        st.divider()
        st.subheader("📌 Итоговые параметры расчёта")
        summary = {
            'Режим': mode,
            'T (фиксир.) для B/C': fixed_T_BC if fixed_T_BC is not None else '—',
            'p\'H₂O воздуха': air_pH2O if air_pH2O is not None and np.isfinite(air_pH2O) else 'из данных',
            'p\'\'H₂O топлива': fuel_pH2O if fuel_pH2O is not None and np.isfinite(fuel_pH2O) else 'из данных',
            'Доля H₂ в топливе': fuel_H2_fraction,
            'Шаг сетки': grid_step,
            'δ режим': tolerance_mode,
            'δ (мВ)': tolerance_value_mV if tolerance_mode == 'manual' else 'авто',
            'min решений': min_solutions,
            'max δ (мВ)': max_tolerance_mV,
        }
        st.table(pd.DataFrame([summary]).T.rename(columns={0: 'значение'}))

        if st.button("🚀 Запустить анализ", type="primary", use_container_width=True):
            if st.session_state.experimental_data is None:
                st.error("Сначала загрузите данные!")
            else:
                with st.spinner("Выполняется расчёт..."):
                    t0 = time.time()
                    results = run_analysis(
                        data=st.session_state.experimental_data,
                        mode=mode,
                        fixed_T_BC=(fixed_T_BC if fixed_T_BC is not None else np.nan),
                        air_pH2O=air_pH2O,
                        fuel_pH2O=fuel_pH2O,
                        fuel_H2_fraction=fuel_H2_fraction,
                        grid_step=grid_step,
                        tolerance_mode=tolerance_mode,
                        tolerance_value_mV=tolerance_value_mV,
                        min_solutions=min_solutions,
                        max_tolerance_mV=max_tolerance_mV
                    )
                    st.session_state.results = results
                    st.session_state.analysis_done = True
                    st.success(f"Готово за {time.time()-t0:.2f} с")

    # ============================================
    # ОСНОВНОЙ КОНТЕНТ
    # ============================================
    if st.session_state.experimental_data is not None:
        st.header("📋 Загруженные данные")
        xlabel = _x_label(st.session_state.mode)
        df = pd.DataFrame(st.session_state.experimental_data,
                         columns=[xlabel, 'OCV (V)'])
        st.dataframe(df, use_container_width=True)

        fig_raw, ax_raw = plt.subplots(figsize=(7, 3.5))
        ax_raw.plot(df.iloc[:, 0], df.iloc[:, 1], 'o-',
                   color='#1f77b4', markersize=6, linewidth=1.5)
        ax_raw.set_xlabel(xlabel, fontweight='bold')
        ax_raw.set_ylabel('OCV (V)', fontweight='bold')
        ax_raw.set_title('Экспериментальные данные OCV', fontweight='bold')
        ax_raw.grid(True, alpha=0.3, linestyle='--')
        fig_raw.set_dpi(600)
        st.pyplot(fig_raw)

    if st.session_state.analysis_done and st.session_state.results is not None:
        res = st.session_state.results
        style = st.session_state.style

        st.header("📊 Результаты анализа")

        tab_main, tab_s1, tab_s2, tab_s3, tab_te, tab_maps, tab_diag, tab_tables = st.tabs([
            "🔷 Сводный график",
            "❶ Сценарий H⁺+e⁻",
            "❷ Сценарий O²⁻+H⁺",
            "❸ Сценарий смешанный",
            "🔌 te (электронный)",
            "🗺️ Карты решений",
            "🔬 Диагностика",
            "📑 Таблицы"
        ])

        with tab_main:
            fig_main = create_overview_plot(res, style)
            st.pyplot(fig_main)
            st.info(
                "**Сводный график.** Три сценария на одном полотне. "
                "Полупрозрачная закраска — области пересечения диапазонов "
                "между сценариями (tH, tO, tion)."
            )

        with tab_s1:
            fig1 = create_scenario_plot(res, 1, style)
            st.pyplot(fig1)

        with tab_s2:
            fig2 = create_scenario_plot(res, 2, style)
            st.pyplot(fig2)

        with tab_s3:
            fig3 = create_scenario_plot(res, 3, style)
            st.pyplot(fig3)

            st.subheader("Допуск δ и число решений по точкам")
            rows = []
            for i, s in enumerate(res['s3_grids']):
                rows.append({
                    'x': res['xs'][i],
                    'OCV (V)': s['OCV'],
                    'EO (V)': s['EO'],
                    'EH (V)': s['EH'],
                    'EH2O (V)': s['EH2O'],
                    'δ (мВ)': s['tolerance_V'] * 1e3,
                    'N решений': len(s['solutions'])
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

        with tab_te:
            fig_te = create_te_plot(res, style)
            st.pyplot(fig_te)

        with tab_maps:
            st.subheader("Карты решений для сценария ❸")
            st.markdown("Выберите точку данных:")

            labels = [
                f"точка {i+1}: x={res['xs'][i]:.4g}, OCV={res['ocvs'][i]:.4f} V"
                for i in range(len(res['xs']))
            ]
            idx = st.selectbox("Точка:", options=list(range(len(labels))),
                              format_func=lambda i: labels[i],
                              key='map_point_idx')

            grid_pt = res['s3_grids'][idx]
            sols = grid_pt['solutions']

            col1, col2 = st.columns(2)

            with col1:
                fig_m1 = create_decision_map_scatter(
                    sols, grid_pt['EO'], grid_pt['EH2O'], grid_pt['OCV'], style
                )
                st.pyplot(fig_m1)
                st.caption("Вариант 1: точки (ti, tH), цвет — te = 1 - ti")

            with col2:
                fig_m2 = create_decision_map_contour(
                    grid_pt['EO'], grid_pt['EH2O'], grid_pt['OCV'], style
                )
                st.pyplot(fig_m2)
                st.caption("Вариант 2: линии уровня E_model(ti, tH), "
                          "жирная — уровень E = Emeas")

            st.markdown(
                f"**δ = {grid_pt['tolerance_V']*1e3:.3f} мВ**, "
                f"найдено решений: **{len(sols)}**"
            )

        with tab_diag:
            st.subheader("🔬 Диагностика входных параметров и промежуточных величин")
            st.markdown(
                "Эта таблица показывает, **что реально уходит в расчёт** для каждой "
                "точки. Сравни колонки `p'H2`, `p''H2`, `EO`, `EH` с ручным расчётом "
                "— они должны совпадать."
            )

            dbg_rows = []
            for i, s in enumerate(res['s3_grids']):
                dbg_rows.append({
                    'точка': i + 1,
                    'x (вход)': res['xs'][i],
                    'OCV (V)': s['OCV'],
                    'T_C (°C)': s['T_C'],
                    "p'H₂O (возд)": s['air_pH2O_used'],
                    "p''H₂O (топл)": s['fuel_pH2O_used'],
                    'H₂ frac': s['fuel_H2_fraction_used'],
                    'K(T)': s['K'],
                    "p'O₂": s['pO2_air'],
                    "p'H₂": s['pH2_air'],
                    "p''O₂": s['pO2_fuel'],
                    "p''H₂": s['pH2_fuel'],
                    'EO (V)': s['EO'],
                    'EH (V)': s['EH'],
                    'EH2O (V)': s['EH2O'],
                    'Emeas − EO': s['OCV'] - s['EO'],
                    'EH − EO': s['EH'] - s['EO'],
                })
            df_dbg = pd.DataFrame(dbg_rows)
            st.dataframe(df_dbg, use_container_width=True)

            st.markdown("---")
            st.subheader("Резюме параметров запуска")
            meta = {
                'mode': res['mode'],
                'fixed_T_BC': res['fixed_T_BC'],
                'air_pH2O (вход)': res['air_pH2O'],
                'fuel_pH2O (вход)': res['fuel_pH2O'],
                'fuel_H2_fraction': res['fuel_H2_fraction'],
                'grid_step': res['grid_step'],
                'tolerance_mode': res['tolerance_mode'],
                'tolerance_value_mV': res['tolerance_value_mV'],
                'min_solutions': res['min_solutions'],
                'max_tolerance_mV': res['max_tolerance_mV'],
            }
            st.table(pd.DataFrame([meta]).T.rename(columns={0: 'значение'}))

        with tab_tables:
            st.subheader("Числа переноса по точкам")
            rows = []
            for i in range(len(res['xs'])):
                s1 = res['scenario1'][i]
                s2 = res['scenario2'][i]
                s3 = res['scenario3'][i]
                rows.append({
                    'x': res['xs'][i],
                    'OCV (V)': res['ocvs'][i],
                    '❶ tH': s1['tH'] if s1 else np.nan,
                    '❶ te': s1['te'] if s1 else np.nan,
                    '❷ tH': s2['tH'] if s2 else np.nan,
                    '❷ tO': s2['tO'] if s2 else np.nan,
                    '❸ tH_min': s3['tH_min'],
                    '❸ tH_max': s3['tH_max'],
                    '❸ tO_min': s3['tO_min'],
                    '❸ tO_max': s3['tO_max'],
                    '❸ tion_min': s3['ti_min'],
                    '❸ tion_max': s3['ti_max'],
                    '❸ te_min': s3['te_min'],
                    '❸ te_max': s3['te_max'],
                    '❸ N': s3['n_sol'],
                    '❸ δ (мВ)': s3['tolerance_V'] * 1e3
                })
            df_res = pd.DataFrame(rows)
            st.dataframe(df_res, use_container_width=True)

            csv = df_res.to_csv(index=False)
            st.download_button("📥 Скачать CSV", csv,
                              file_name="transport_numbers.csv",
                              mime="text/csv")

        st.divider()
        st.subheader("📥 Экспорт графиков (PNG, 600 DPI)")

        col1, col2, col3 = st.columns(3)
        with col1:
            buf = io.BytesIO()
            create_overview_plot(res, style).savefig(buf, format='png', dpi=600)
            st.download_button("Сводный график", buf.getvalue(),
                              file_name="overview.png", mime="image/png")
        with col2:
            buf = io.BytesIO()
            create_scenario_plot(res, 1, style).savefig(buf, format='png', dpi=600)
            st.download_button("❶ Сценарий 1", buf.getvalue(),
                              file_name="scenario1.png", mime="image/png")
        with col3:
            buf = io.BytesIO()
            create_te_plot(res, style).savefig(buf, format='png', dpi=600)
            st.download_button("te", buf.getvalue(),
                              file_name="te.png", mime="image/png")

    else:
        st.info("👈 Настройте параметры и загрузите данные в боковой панели, "
                "затем нажмите «Запустить анализ».")

    st.divider()
    st.markdown(
        "<div style='text-align:center'>"
        "<p>OCV Transport Number Analysis | 600 DPI export</p>"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
