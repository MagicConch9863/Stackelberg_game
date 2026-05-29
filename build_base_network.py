import pandapower as pp


def _line_list():
    return [
        ("Bus R0", "Bus R1"),
        ("Bus R1", "Bus R2"),
        ("Bus R2", "Bus R3"),
        ("Bus R3", "Bus R4"),
        ("Bus R4", "Bus R5"),
        ("Bus R5", "Bus R6"),
        ("Bus R6", "Bus R7"),
        ("Bus R7", "Bus R8"),
        ("Bus R8", "Bus R9"),
        ("Bus R9", "Bus R10"),
        ("Bus R3", "Bus R11"),
        ("Bus R4", "Bus R12"),
        ("Bus R12", "Bus R13"),
        ("Bus R13", "Bus R14"),
        ("Bus R14", "Bus R15"),
        ("Bus R6", "Bus R16"),
        ("Bus R9", "Bus R17"),
        ("Bus R10", "Bus R18"),
    ]


def feeder_data(active_buses):
    lines = _line_list()
    all_buses = ["Bus R0"] + [f"Bus R{i}" for i in range(1, 19)]

    parent = {"Bus R0": None}
    parent_line = {}
    children = {b: [] for b in all_buses}

    for idx, (u, v) in enumerate(lines):
        parent[v] = u
        parent_line[v] = idx
        children[u].append(v)

    pmax_kw = [35.0 for _ in lines]
    r_pu_per_kw = [0.00012 for _ in lines]

    path = {}
    for b in active_buses:
        cur = b
        p = []
        while cur != "Bus R0":
            lid = parent_line[cur]
            p.append(lid)
            cur = parent[cur]
        path[b] = list(reversed(p))

    desc = {lid: [] for lid in range(len(lines))}
    for b in active_buses:
        for lid in path[b]:
            desc[lid].append(b)

    return {
        "buses": list(active_buses),
        "lines": lines,
        "path": path,
        "desc": desc,
        "pmax_kw": pmax_kw,
        "r_pu_per_kw": r_pu_per_kw,
    }


def create_pandapower_net():
    net = pp.create_empty_network()

    names = ["Bus 0", "Bus R0"] + [f"Bus R{i}" for i in range(1, 19)]
    bus = {}

    for n in names:
        vn = 20.0 if n == "Bus 0" else 0.4
        bus[n] = pp.create_bus(net, vn_kv=vn, name=n)

    pp.create_ext_grid(net, bus=bus["Bus 0"])

    pp.create_transformer(
        net,
        hv_bus=bus["Bus 0"],
        lv_bus=bus["Bus R0"],
        std_type="0.4 MVA 20/0.4 kV",
    )

    for u, v in _line_list():
        pp.create_line(net, bus[u], bus[v], length_km=0.035, std_type="NAYY 4x150 SE")

    return net, bus