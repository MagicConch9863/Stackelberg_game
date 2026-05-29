import numpy as np


def solve_stackelberg_mpec(cfg, data, market, method="sos1"):
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as exc:
        raise RuntimeError("Please install gurobipy and configure a valid Gurobi license.") from exc

    if method not in ("sos1", "bigm", "duality"):
        raise ValueError(f"Unknown MPEC method: {method}")

    T = cfg.horizon
    buses = list(data["buses"])
    net = data["net"]
    load = data["load"]
    pv = data["pv"]

    battery_buses = [b for b in buses if b in cfg.battery_buses]
    pv_buses = [b for b in buses if np.max(pv[b]) > 1e-9]

    baseline = np.sum([load[b] - pv[b] for b in buses], axis=0)
    lref = float(np.mean(baseline))

    m = gp.Model(f"stackelberg_{method}")

    m.Params.OutputFlag = 1
    m.Params.NonConvex = 2
    m.Params.TimeLimit = cfg.time_limit_seconds
    m.Params.MIPGap = cfg.mip_gap
    m.Params.DualReductions = 0

    if hasattr(cfg, "mip_gap_abs"):
        m.Params.MIPGapAbs = cfg.mip_gap_abs

    if hasattr(cfg, "solver_seed"):
        m.Params.Seed = int(cfg.solver_seed)

    # -----------------------------
    # Leader variables
    # -----------------------------
    p_buy = m.addVars(T, lb=cfg.p_min, ub=cfg.p_max, name="p_buy")
    p_sell = m.addVars(T, lb=0.0, ub=cfg.p_max, name="p_sell")
    peak = m.addVar(lb=0.0, name="peak")

    # -----------------------------
    # Follower primal variables
    # -----------------------------
    buy = m.addVars(buses, T, lb=0.0, name="P_buy")
    sell = m.addVars(buses, T, lb=0.0, name="P_sell")
    dl = m.addVars(buses, T, lb=-GRB.INFINITY, name="dL")

    curt = m.addVars(pv_buses, T, lb=0.0, name="P_curt")

    ch = m.addVars(battery_buses, T, lb=0.0, name="P_ch")
    dis = m.addVars(battery_buses, T, lb=0.0, name="P_dis")
    soc = m.addVars(battery_buses, T + 1, lb=-GRB.INFINITY, name="SOC")

    # -----------------------------
    # Follower dual variables
    # -----------------------------
    lam = m.addVars(buses, T, lb=-GRB.INFINITY, name="lambda_balance")
    tau = m.addVars(buses, lb=-GRB.INFINITY, name="tau_dr_energy")

    nu = m.addVars(battery_buses, T, lb=-GRB.INFINITY, name="nu_soc")
    sig0 = m.addVars(battery_buses, lb=-GRB.INFINITY, name="sigma_soc0")
    sigT = m.addVars(battery_buses, lb=-GRB.INFINITY, name="sigma_socT")

    stat = []
    sd_terms = []

    def bounded(x, lb, ub, name):
        """
        Add lb <= x <= ub and its KKT bound-dual terms.

        method="sos1":
            Complementarity by SOS1(slack, dual).

        method="bigm":
            Complementarity by binary + Big-M.

        method="duality":
            Keep strong-duality-like bilinear terms.
            This method is retained but not recommended for large runs.
        """
        slack_lb = m.addVar(lb=0.0, name=f"sl_lb_{name}")
        slack_ub = m.addVar(lb=0.0, name=f"sl_ub_{name}")

        mu_lb = m.addVar(lb=0.0, name=f"mu_lb_{name}")
        mu_ub = m.addVar(lb=0.0, name=f"mu_ub_{name}")

        m.addConstr(slack_lb == x - lb, name=f"def_sl_lb_{name}")
        m.addConstr(slack_ub == ub - x, name=f"def_sl_ub_{name}")

        if method == "sos1":
            m.addSOS(GRB.SOS_TYPE1, [slack_lb, mu_lb])
            m.addSOS(GRB.SOS_TYPE1, [slack_ub, mu_ub])

        elif method == "bigm":
            z_lb = m.addVar(vtype=GRB.BINARY, name=f"z_lb_{name}")
            z_ub = m.addVar(vtype=GRB.BINARY, name=f"z_ub_{name}")

            slack_m = max(float(ub - lb), 1.0)

            m.addConstr(mu_lb <= cfg.big_m_dual * z_lb, name=f"bigm_mu_lb_{name}")
            m.addConstr(slack_lb <= slack_m * (1 - z_lb), name=f"bigm_sl_lb_{name}")

            m.addConstr(mu_ub <= cfg.big_m_dual * z_ub, name=f"bigm_mu_ub_{name}")
            m.addConstr(slack_ub <= slack_m * (1 - z_ub), name=f"bigm_sl_ub_{name}")

        elif method == "duality":
            sd_terms.append(mu_lb * (lb - x))
            sd_terms.append(mu_ub * (x - ub))

        return -mu_lb + mu_ub

    # -----------------------------
    # Leader price constraints
    # -----------------------------
    for t in range(T):
        m.addConstr(p_buy[t] >= market[t], name=f"p_buy_floor_{t}")
        m.addConstr(p_buy[t] <= market[t] + cfg.max_markup, name=f"p_buy_markup_{t}")
        m.addConstr(p_sell[t] <= cfg.sell_discount * p_buy[t], name=f"p_sell_discount_{t}")

        if t > 0:
            m.addConstr(p_buy[t] - p_buy[t - 1] <= cfg.ramp, name=f"p_buy_ramp_up_{t}")
            m.addConstr(p_buy[t - 1] - p_buy[t] <= cfg.ramp, name=f"p_buy_ramp_dn_{t}")
            m.addConstr(p_sell[t] - p_sell[t - 1] <= cfg.ramp, name=f"p_sell_ramp_up_{t}")
            m.addConstr(p_sell[t - 1] - p_sell[t] <= cfg.ramp, name=f"p_sell_ramp_dn_{t}")

    # -----------------------------
    # Follower constraints and stationarity
    # -----------------------------
    for b in buses:
        has_pv = b in pv_buses
        has_bat = b in battery_buses

        # Demand response is energy-neutral.
        m.addConstr(
            gp.quicksum(dl[b, t] for t in range(T)) == 0.0,
            name=f"dr_energy_{b}",
        )

        for t in range(T):
            ch_t = ch[b, t] if has_bat else 0.0
            dis_t = dis[b, t] if has_bat else 0.0
            curt_t = curt[b, t] if has_pv else 0.0

            # Local balance:
            # buy - sell + dis - ch - dl - curt = load - pv
            m.addConstr(
                buy[b, t] - sell[b, t] + dis_t - ch_t - dl[b, t] - curt_t
                == load[b][t] - pv[b][t],
                name=f"balance_{b}_{t}",
            )

            stat.append(
                p_buy[t]
                + lam[b, t]
                + bounded(buy[b, t], 0.0, cfg.buy_max, f"buy_{b}_{t}")
            )

            stat.append(
                -p_sell[t]
                - lam[b, t]
                + bounded(sell[b, t], 0.0, cfg.sell_max, f"sell_{b}_{t}")
            )

            dr_lb = -0.15 * load[b][t]
            dr_ub = 0.15 * load[b][t]

            stat.append(
                cfg.alpha_dr * dl[b, t]
                - lam[b, t]
                + tau[b]
                + bounded(dl[b, t], dr_lb, dr_ub, f"dl_{b}_{t}")
            )

            if has_pv:
                stat.append(
                    cfg.c_curt
                    - lam[b, t]
                    + bounded(curt[b, t], 0.0, pv[b][t], f"curt_{b}_{t}")
                )

            if has_bat:
                m.addConstr(
                    soc[b, t + 1]
                    - soc[b, t]
                    - cfg.eta_ch * cfg.dt * ch[b, t]
                    + cfg.dt / cfg.eta_dis * dis[b, t]
                    == 0.0,
                    name=f"soc_dyn_{b}_{t}",
                )

                stat.append(
                    cfg.beta_bat * ch[b, t]
                    - lam[b, t]
                    - cfg.eta_ch * cfg.dt * nu[b, t]
                    + bounded(ch[b, t], 0.0, cfg.bat_pmax, f"ch_{b}_{t}")
                )

                stat.append(
                    cfg.beta_bat * dis[b, t]
                    + lam[b, t]
                    + cfg.dt / cfg.eta_dis * nu[b, t]
                    + bounded(dis[b, t], 0.0, cfg.bat_pmax, f"dis_{b}_{t}")
                )

        if has_bat:
            m.addConstr(soc[b, 0] == cfg.soc_init, name=f"soc_init_{b}")
            m.addConstr(soc[b, T] == cfg.soc_init, name=f"soc_terminal_{b}")

            for t in range(T + 1):
                bd = bounded(soc[b, t], cfg.soc_min, cfg.soc_max, f"soc_{b}_{t}")

                if t == 0:
                    stat.append(bd + sig0[b] - nu[b, 0])
                elif t == T:
                    stat.append(bd + sigT[b] + nu[b, T - 1])
                else:
                    stat.append(bd + nu[b, t - 1] - nu[b, t])

    for k, expr in enumerate(stat):
        m.addConstr(expr == 0.0, name=f"stationarity_{k}")

    if method == "duality":
        m.addConstr(gp.quicksum(sd_terms) == 0.0, name="strong_duality_terms")

    # -----------------------------
    # LinDistFlow network constraints
    # -----------------------------
    flow = m.addVars(len(net["lines"]), T, lb=-GRB.INFINITY, name="P_line")
    volt = m.addVars(net["buses"], T, lb=-GRB.INFINITY, name="V_lin")

    for t in range(T):
        for lid in range(len(net["lines"])):
            m.addConstr(
                flow[lid, t]
                == gp.quicksum(buy[b, t] - sell[b, t] for b in net["desc"][lid]),
                name=f"flow_{lid}_{t}",
            )

            m.addConstr(
                flow[lid, t] <= cfg.line_margin * net["pmax_kw"][lid],
                name=f"flow_max_{lid}_{t}",
            )

            m.addConstr(
                flow[lid, t] >= -cfg.line_margin * net["pmax_kw"][lid],
                name=f"flow_min_{lid}_{t}",
            )

        for b in net["buses"]:
            drop = gp.quicksum(
                2.0 * net["r_pu_per_kw"][lid] * flow[lid, t]
                for lid in net["path"][b]
            )

            m.addConstr(volt[b, t] == 1.0 - drop, name=f"vlin_{b}_{t}")
            m.addConstr(volt[b, t] >= cfg.v_min, name=f"vmin_{b}_{t}")
            m.addConstr(volt[b, t] <= cfg.v_max, name=f"vmax_{b}_{t}")

        agg = gp.quicksum(buy[b, t] - sell[b, t] for b in buses)
        m.addConstr(peak >= agg, name=f"peak_pos_{t}")
        m.addConstr(peak >= -agg, name=f"peak_neg_{t}")

    # -----------------------------
    # Leader objective
    # -----------------------------
    agg_list = [
        gp.quicksum(buy[b, t] - sell[b, t] for b in buses)
        for t in range(T)
    ]

    retail_profit = gp.quicksum(
        cfg.dt * (p_buy[t] * buy[b, t] - p_sell[t] * sell[b, t])
        for b in buses
        for t in range(T)
    )

    wholesale_cost = gp.quicksum(
        cfg.dt * market[t] * agg_list[t]
        for t in range(T)
    )

    peak_pen = cfg.rho_peak * peak

    flat_pen = cfg.rho_flat * gp.quicksum(
        cfg.dt * (agg_list[t] - lref) * (agg_list[t] - lref)
        for t in range(T)
    )

    price_reg = cfg.rho_price * gp.quicksum(
        cfg.dt * (
            (p_buy[t] - market[t]) * (p_buy[t] - market[t])
            + (p_sell[t] - cfg.sell_discount * market[t])
            * (p_sell[t] - cfg.sell_discount * market[t])
        )
        for t in range(T)
    )

    curt_pen = cfg.rho_curt * gp.quicksum(curt[key] for key in curt.keys())

    m.setObjective(
        wholesale_cost - retail_profit + peak_pen + flat_pen + price_reg + curt_pen,
        GRB.MINIMIZE,
    )

    m.optimize()

    if m.Status in [GRB.INFEASIBLE, GRB.INF_OR_UNBD, GRB.UNBOUNDED]:
        try:
            m.computeIIS()
            m.write(f"stackelberg_infeasible_{method}.ilp")
        except Exception:
            pass
        return None

    if m.SolCount == 0:
        return None

    try:
        mip_gap = float(m.MIPGap)
    except Exception:
        mip_gap = 0.0

    try:
        obj_bound = float(m.ObjBound)
    except Exception:
        obj_bound = np.nan

    try:
        abs_gap = float(abs(m.ObjVal - m.ObjBound))
    except Exception:
        abs_gap = np.nan

    result = {
        "method": method,
        "status": int(m.Status),
        "runtime_sec": float(m.Runtime),
        "mip_gap": mip_gap,
        "obj_value": float(m.ObjVal),
        "obj_bound": obj_bound,
        "abs_gap": abs_gap,
        "price_buy": np.array([p_buy[t].X for t in range(T)]),
        "price_sell": np.array([p_sell[t].X for t in range(T)]),
        "agg_load": np.array([
            sum(buy[b, t].X - sell[b, t].X for b in buses)
            for t in range(T)
        ]),
        "p_peak": float(peak.X),
        "v_lin": np.array([
            [volt[b, t].X for t in range(T)]
            for b in net["buses"]
        ]),
        "line_flow": np.array([
            [flow[lid, t].X for t in range(T)]
            for lid in range(len(net["lines"]))
        ]),
        "bus_order": list(net["buses"]),
        "battery_buses": battery_buses,
        "schedules": {},
    }

    for b in buses:
        result["schedules"][b] = {
            "buy": np.array([buy[b, t].X for t in range(T)]),
            "sell": np.array([sell[b, t].X for t in range(T)]),
            "dload": np.array([dl[b, t].X for t in range(T)]),
            "curt": np.array([curt[b, t].X for t in range(T)]) if b in pv_buses else np.zeros(T),
            "ch": np.array([ch[b, t].X for t in range(T)]) if b in battery_buses else np.zeros(T),
            "dis": np.array([dis[b, t].X for t in range(T)]) if b in battery_buses else np.zeros(T),
            "soc": np.array([soc[b, t].X for t in range(T + 1)]) if b in battery_buses else np.zeros(T + 1),
        }

    return result