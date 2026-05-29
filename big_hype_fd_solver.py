import time
import numpy as np


def project_prices(cfg, market, p_buy, p_sell):
    p_buy = np.clip(p_buy, market, market + cfg.max_markup)
    p_buy = np.clip(p_buy, cfg.p_min, cfg.p_max)

    p_sell = np.clip(p_sell, 0.0, cfg.sell_discount * p_buy)

    for t in range(1, len(p_buy)):
        p_buy[t] = min(p_buy[t], p_buy[t - 1] + cfg.ramp)
        p_buy[t] = max(p_buy[t], p_buy[t - 1] - cfg.ramp)
        p_buy[t] = np.clip(p_buy[t], market[t], market[t] + cfg.max_markup)
        p_buy[t] = np.clip(p_buy[t], cfg.p_min, cfg.p_max)

        p_sell[t] = min(p_sell[t], p_sell[t - 1] + cfg.ramp)
        p_sell[t] = max(p_sell[t], p_sell[t - 1] - cfg.ramp)
        p_sell[t] = np.clip(p_sell[t], 0.0, cfg.sell_discount * p_buy[t])

    return p_buy, p_sell


def solve_followers_given_price(cfg, data, p_buy, p_sell):
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as exc:
        raise RuntimeError("Please install gurobipy and configure a valid Gurobi license.") from exc

    T = cfg.horizon
    buses = list(data["buses"])
    load = data["load"]
    pv = data["pv"]

    battery_buses = [b for b in buses if b in cfg.battery_buses]
    pv_buses = [b for b in buses if np.max(pv[b]) > 1e-9]

    m = gp.Model("followers_given_price")
    m.Params.OutputFlag = 0
    m.Params.NonConvex = 2

    buy = m.addVars(buses, T, lb=0.0, ub=cfg.buy_max, name="P_buy")
    sell = m.addVars(buses, T, lb=0.0, ub=cfg.sell_max, name="P_sell")
    dl = m.addVars(buses, T, lb=-GRB.INFINITY, name="dL")

    curt = m.addVars(pv_buses, T, lb=0.0, name="P_curt")

    ch = m.addVars(battery_buses, T, lb=0.0, ub=cfg.bat_pmax, name="P_ch")
    dis = m.addVars(battery_buses, T, lb=0.0, ub=cfg.bat_pmax, name="P_dis")
    soc = m.addVars(
        battery_buses,
        T + 1,
        lb=cfg.soc_min,
        ub=cfg.soc_max,
        name="SOC",
    )

    obj = 0.0

    for b in buses:
        has_pv = b in pv_buses
        has_bat = b in battery_buses

        m.addConstr(
            gp.quicksum(dl[b, t] for t in range(T)) == 0.0,
            name=f"dr_energy_{b}",
        )

        for t in range(T):
            dr_lim = 0.15 * load[b][t]
            m.addConstr(dl[b, t] >= -dr_lim, name=f"dr_lb_{b}_{t}")
            m.addConstr(dl[b, t] <= dr_lim, name=f"dr_ub_{b}_{t}")

            curt_t = curt[b, t] if has_pv else 0.0
            ch_t = ch[b, t] if has_bat else 0.0
            dis_t = dis[b, t] if has_bat else 0.0

            if has_pv:
                m.addConstr(curt[b, t] <= pv[b][t], name=f"curt_ub_{b}_{t}")

            m.addConstr(
                buy[b, t] - sell[b, t] + dis_t - ch_t - dl[b, t] - curt_t
                == load[b][t] - pv[b][t],
                name=f"local_balance_{b}_{t}",
            )

            obj += cfg.dt * (
                p_buy[t] * buy[b, t]
                - p_sell[t] * sell[b, t]
                + 0.5 * cfg.alpha_dr * dl[b, t] * dl[b, t]
            )

            if has_pv:
                obj += cfg.dt * cfg.c_curt * curt[b, t]

            if has_bat:
                obj += cfg.dt * 0.5 * cfg.beta_bat * (
                    ch[b, t] * ch[b, t] + dis[b, t] * dis[b, t]
                )

        if has_bat:
            m.addConstr(soc[b, 0] == cfg.soc_init, name=f"soc_init_{b}")
            m.addConstr(soc[b, T] == cfg.soc_init, name=f"soc_terminal_{b}")

            for t in range(T):
                m.addConstr(
                    soc[b, t + 1]
                    == soc[b, t]
                    + cfg.eta_ch * cfg.dt * ch[b, t]
                    - cfg.dt / cfg.eta_dis * dis[b, t],
                    name=f"soc_dyn_{b}_{t}",
                )

    m.setObjective(obj, GRB.MINIMIZE)
    m.optimize()

    if m.SolCount == 0:
        return None

    schedules = {}

    for b in buses:
        schedules[b] = {
            "buy": np.array([buy[b, t].X for t in range(T)]),
            "sell": np.array([sell[b, t].X for t in range(T)]),
            "dload": np.array([dl[b, t].X for t in range(T)]),
            "curt": np.array([curt[b, t].X for t in range(T)]) if b in pv_buses else np.zeros(T),
            "ch": np.array([ch[b, t].X for t in range(T)]) if b in battery_buses else np.zeros(T),
            "dis": np.array([dis[b, t].X for t in range(T)]) if b in battery_buses else np.zeros(T),
            "soc": np.array([soc[b, t].X for t in range(T + 1)]) if b in battery_buses else np.zeros(T + 1),
        }

    agg_load = np.array([
        sum(schedules[b]["buy"][t] - schedules[b]["sell"][t] for b in buses)
        for t in range(T)
    ])

    return {
        "schedules": schedules,
        "agg_load": agg_load,
    }


def leader_objective(cfg, data, market, p_buy, p_sell, follower):
    buses = list(data["buses"])
    baseline = np.sum([data["load"][b] - data["pv"][b] for b in buses], axis=0)
    lref = float(np.mean(baseline))
    agg = follower["agg_load"]

    retail_profit = 0.0
    curt_total = 0.0

    for b in buses:
        sch = follower["schedules"][b]
        retail_profit += np.sum(cfg.dt * (p_buy * sch["buy"] - p_sell * sch["sell"]))
        curt_total += np.sum(sch["curt"])

    wholesale_cost = np.sum(cfg.dt * market * agg)
    peak_pen = cfg.rho_peak * np.max(np.abs(agg))
    flat_pen = cfg.rho_flat * np.sum(cfg.dt * (agg - lref) ** 2)

    price_reg = cfg.rho_price * np.sum(
        cfg.dt * (
            (p_buy - market) ** 2
            + (p_sell - cfg.sell_discount * market) ** 2
        )
    )

    curt_pen = cfg.rho_curt * curt_total

    return wholesale_cost - retail_profit + peak_pen + flat_pen + price_reg + curt_pen


def solve_big_hype_fd(cfg, data, market):
    """
    BIG-Hype-like finite-difference solver.

    This is not the full BIG Hype sensitivity-learning algorithm.
    It is a simplified local hypergradient benchmark:

        1. choose leader prices,
        2. solve follower best response,
        3. approximate leader hypergradient by finite difference,
        4. update projected prices.
    """
    start_time = time.time()
    T = cfg.horizon

    rng = np.random.default_rng(int(getattr(cfg, "big_hype_seed", 0)))
    jitter = float(getattr(cfg, "big_hype_init_jitter", 0.0))

    p_buy = market + 0.03 + rng.uniform(-jitter, jitter, size=T)
    p_sell = cfg.sell_discount * market + rng.uniform(-0.5 * jitter, 0.5 * jitter, size=T)

    p_buy, p_sell = project_prices(cfg, market, p_buy, p_sell)

    best = None
    best_obj = np.inf

    for it in range(cfg.big_hype_outer_iter):
        follower = solve_followers_given_price(cfg, data, p_buy, p_sell)
        if follower is None:
            break

        base_obj = leader_objective(cfg, data, market, p_buy, p_sell, follower)

        if base_obj < best_obj:
            best_obj = base_obj
            best = {
                "method": "big_hype_fd",
                "status": 1,
                "runtime_sec": float(time.time() - start_time),
                "mip_gap": np.nan,
                "obj_value": float(base_obj),
                "price_buy": p_buy.copy(),
                "price_sell": p_sell.copy(),
                "agg_load": follower["agg_load"].copy(),
                "schedules": follower["schedules"],
            }

        grad_buy = np.zeros(T)
        grad_sell = np.zeros(T)
        eps = cfg.big_hype_fd_eps

        for t in range(T):
            p_buy_eps = p_buy.copy()
            p_buy_eps[t] += eps
            p_buy_eps, p_sell_tmp = project_prices(cfg, market, p_buy_eps, p_sell.copy())

            f_eps = solve_followers_given_price(cfg, data, p_buy_eps, p_sell_tmp)
            if f_eps is not None:
                obj_eps = leader_objective(cfg, data, market, p_buy_eps, p_sell_tmp, f_eps)
                grad_buy[t] = (obj_eps - base_obj) / eps

            p_sell_eps = p_sell.copy()
            p_sell_eps[t] += eps
            p_buy_tmp, p_sell_eps = project_prices(cfg, market, p_buy.copy(), p_sell_eps)

            f_eps = solve_followers_given_price(cfg, data, p_buy_tmp, p_sell_eps)
            if f_eps is not None:
                obj_eps = leader_objective(cfg, data, market, p_buy_tmp, p_sell_eps, f_eps)
                grad_sell[t] = (obj_eps - base_obj) / eps

        step = cfg.big_hype_step / np.sqrt(it + 1.0)

        p_buy = p_buy - step * grad_buy
        p_sell = p_sell - step * grad_sell

        p_buy, p_sell = project_prices(cfg, market, p_buy, p_sell)

    if best is not None:
        best["runtime_sec"] = float(time.time() - start_time)

    return best