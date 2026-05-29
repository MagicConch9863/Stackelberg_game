from dataclasses import dataclass, field
import numpy as np
import pandas as pd


@dataclass
class StackelbergConfig:
    # -----------------------------
    # Time
    # -----------------------------
    horizon: int = 9
    dt: float = 0.25
    seed: int = 42

    # -----------------------------
    # Selected buses and roles
    # -----------------------------
    selected_buses: tuple = (
        "Bus R1", "Bus R3", "Bus R6", "Bus R9", "Bus R12", "Bus R15"
    )

    n_passive_load: int = 1
    n_pv_only: int = 1
    n_pv_battery: int = 4

    active_buses: tuple = field(init=False)
    load_buses: tuple = field(init=False)
    pv_buses: tuple = field(init=False)
    battery_buses: tuple = field(init=False)

    # -----------------------------
    # SimBench data
    # -----------------------------
    simbench_code: str = "1-LV-urban6--0-sw"
    load_peak_range_kw: tuple = (3.5, 5.0)
    pv_peak_range_kw: tuple = (3.0, 4.5)

    # -----------------------------
    # Compared methods
    # duality is not executed unless added here
    # -----------------------------
    compare_methods: tuple = ("sos1", "bigm", "big_hype_fd")

    # -----------------------------
    # Gurobi settings
    # -----------------------------
    time_limit_seconds: int = 3600
    mip_gap: float = 0.001
    mip_gap_abs: float = 1e-4
    solver_seed: int = 1

    # -----------------------------
    # Big-M settings
    # -----------------------------
    big_m_dual: float = 500.0

    # -----------------------------
    # BIG-Hype-like finite-difference settings
    # -----------------------------
    big_hype_outer_iter: int = 300
    big_hype_step: float = 0.02
    big_hype_fd_eps: float = 1e-4
    big_hype_init_jitter: float = 0.005
    big_hype_seed: int = 1

    # -----------------------------
    # Price settings
    # -----------------------------
    p_min: float = 0.04
    p_max: float = 0.42
    max_markup: float = 0.06
    ramp: float = 0.10
    sell_discount: float = 0.55

    # -----------------------------
    # Device limits
    # -----------------------------
    buy_max: float = 25.0
    sell_max: float = 25.0

    bat_pmax: float = 4.0
    soc_min: float = 1.0
    soc_max: float = 10.0
    soc_init: float = 5.0
    eta_ch: float = 0.95
    eta_dis: float = 0.95

    # -----------------------------
    # Follower costs
    # -----------------------------
    alpha_dr: float = 4.0
    beta_bat: float = 0.10
    c_curt: float = 0.04

    # -----------------------------
    # Leader objective weights
    # -----------------------------
    rho_peak: float = 15.0
    rho_flat: float = 4.0
    rho_price: float = 10.0
    rho_curt: float = 4.0

    # -----------------------------
    # LinDistFlow limits
    # -----------------------------
    v_min: float = 0.955
    v_max: float = 1.045
    line_margin: float = 0.92

    # -----------------------------
    # Output
    # -----------------------------
    output_dir: str = "outputs"

    def __post_init__(self):
        total = self.n_passive_load + self.n_pv_only + self.n_pv_battery
        if total != len(self.selected_buses):
            raise ValueError(
                "n_passive_load + n_pv_only + n_pv_battery must equal len(selected_buses)."
            )

        rng = np.random.default_rng(self.seed)
        buses = list(self.selected_buses)
        rng.shuffle(buses)

        passive = buses[:self.n_passive_load]
        pv_only = buses[self.n_passive_load:self.n_passive_load + self.n_pv_only]
        pv_bat = buses[self.n_passive_load + self.n_pv_only:]

        self.active_buses = tuple(buses)

        # In this model, all selected buses have load.
        self.load_buses = tuple(buses)

        # PV exists at PV-only and PV+battery buses.
        self.pv_buses = tuple(pv_only + pv_bat)

        # Battery exists only at PV+battery buses.
        self.battery_buses = tuple(pv_bat)

        print("\n=== Random role assignment ===")
        print(f"passive load : {passive}")
        print(f"PV only      : {pv_only}")
        print(f"PV + battery : {pv_bat}")


def load_price_data(path="price_data.csv", horizon=96):
    df = pd.read_csv(path)
    col = "price_eur_per_mwh" if "price_eur_per_mwh" in df.columns else df.columns[-1]

    price = df[col].to_numpy(dtype=float)[:horizon] / 1000.0

    if len(price) < horizon:
        price = np.resize(price, horizon)

    return np.clip(price, 0.0, None)


def _find_profile(profiles, element, variable):
    candidates = [
        (element, variable),
        f"{element}.{variable}",
        f"{element}_{variable}",
        element,
    ]

    for key in candidates:
        if key in profiles:
            obj = profiles[key]
            if isinstance(obj, pd.DataFrame):
                return obj
            if isinstance(obj, dict) and variable in obj:
                return obj[variable]

    for key, value in profiles.items():
        key_str = str(key).lower()
        if element in key_str and variable in key_str and isinstance(value, pd.DataFrame):
            return value

    return None


def _resize_rows(arr, horizon):
    if arr.shape[0] >= horizon:
        return arr[:horizon, :]

    repeat = int(np.ceil(horizon / arr.shape[0]))
    return np.tile(arr, (repeat, 1))[:horizon, :]


def _take_columns(df, n, horizon):
    arr = df.to_numpy(dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.maximum(arr, 0.0)
    arr = _resize_rows(arr, horizon)

    energy = arr.sum(axis=0)
    order = np.argsort(-energy)

    if len(order) < n:
        order = np.resize(order, n)
    else:
        order = order[:n]

    return arr[:, order].T


def _scale_to_peak(profile_kw, target_peak_kw):
    profile_kw = np.asarray(profile_kw, dtype=float)
    peak = float(np.max(profile_kw))

    if peak <= 1e-9:
        return np.zeros_like(profile_kw)

    return profile_kw / peak * target_peak_kw


def _fallback_pv_shape(cfg):
    t = np.arange(cfg.horizon)
    shape = np.maximum(0.0, np.sin(np.pi * (t / cfg.horizon - 0.23) / 0.52))

    pv = np.zeros((len(cfg.active_buses), cfg.horizon))
    bus_to_idx = {b: k for k, b in enumerate(cfg.active_buses)}

    rng = np.random.default_rng(cfg.seed + 100)

    for b in cfg.pv_buses:
        idx = bus_to_idx[b]
        peak = rng.uniform(*cfg.pv_peak_range_kw)
        pv[idx, :] = peak * shape

    return pv


def make_profiles(cfg: StackelbergConfig):
    try:
        import simbench as sb
    except ImportError as exc:
        raise RuntimeError("Please install simbench: pip install simbench") from exc

    rng = np.random.default_rng(cfg.seed)
    n_bus = len(cfg.active_buses)
    bus_to_idx = {b: k for k, b in enumerate(cfg.active_buses)}

    load = np.zeros((n_bus, cfg.horizon))
    pv = np.zeros_like(load)

    net = sb.get_simbench_net(cfg.simbench_code)
    profiles = sb.get_absolute_values(net, profiles_instead_of_study_cases=True)

    load_df = _find_profile(profiles, "load", "p_mw")
    if load_df is None:
        raise RuntimeError("Could not find load p_mw profile in SimBench data.")

    raw_load_kw = 1000.0 * _take_columns(load_df, n_bus, cfg.horizon)

    for j, b in enumerate(cfg.active_buses):
        idx = bus_to_idx[b]
        target_peak = rng.uniform(*cfg.load_peak_range_kw)
        load[idx, :] = _scale_to_peak(raw_load_kw[j, :], target_peak)

    sgen_df = _find_profile(profiles, "sgen", "p_mw")

    if sgen_df is not None and not sgen_df.empty and len(cfg.pv_buses) > 0:
        raw_pv_kw = 1000.0 * _take_columns(sgen_df, len(cfg.pv_buses), cfg.horizon)

        for j, b in enumerate(cfg.pv_buses):
            idx = bus_to_idx[b]
            target_peak = rng.uniform(*cfg.pv_peak_range_kw)
            pv[idx, :] = _scale_to_peak(raw_pv_kw[j, :], target_peak)
    else:
        pv = _fallback_pv_shape(cfg)

    print("\n=== SimBench profiles loaded and scaled ===")
    for k, b in enumerate(cfg.active_buses):
        role = "passive load"
        if b in cfg.pv_buses:
            role = "PV only"
        if b in cfg.battery_buses:
            role = "PV + battery"

        print(
            f"{b:8s} | role = {role:13s} | "
            f"load peak = {load[k].max():.3f} kW, "
            f"load energy = {load[k].sum() * cfg.dt:.3f} kWh, "
            f"PV peak = {pv[k].max():.3f} kW"
        )

    return load, pv