"""
stage1_recourse.py — 지속시간 불확실성을 고려하는 적응형 Stage 2

지속시간별 완전정보 기준 (wait-and-see)
    p30·solve_stage2_fixed(D=30) + p60·solve_stage2_fixed(D=60) + p90·solve_stage2_fixed(D=90)
    → 세 문제를 독립으로 풀고 사후에 가중합. 각 문제는 D 를 '알고' 배차표를 짠다.

적응형 구조
    solve_stage2_adaptive()  하나의 문제
    → 배차변수 y 는 비예견성 정보집합을 공유하고,
      승객 흐름변수 b/a/l/q/w 는 지속시간별로 복제되며,
      목적은  min Σ_D P(D)·Z_D.

비예견성
    장애 시작 후 τ 분에 배차하는 trip 은, 그 시점에 구분 불가능한 지속시간
    시나리오들에서 '같은 결정' 이어야 한다.

        τ ≤ disrupt_start + 30   →  (30, 60, 90)   아직 아무것도 모름
        τ ≤ disrupt_start + 60   →  (60, 90)       30분 종료를 관측함
        그 외                    →  (90,)          60분 종료까지 관측함

    등식 제약을 걸지 않고, 같은 정보집합의 시나리오가 **하나의 변수를 공유**하도록
    변수를 생성한다.

Trip 가용성
    후보 trip 은 90분 배차창으로 한 번만 생성한다.
    지속시간 D 에서 사용 가능한 조건:

        DispatchStart ≤ disrupt_start + D + DISPATCH_BUFFER

    DISPATCH_BUFFER = 0 이면 '장애가 종료된 시나리오에서는 이후 배차 없음' 이
    되고, 이때 가용성 분할과 정보집합 분할이 정확히 일치한다(기본값).
    양수로 두면 종료 직후 잔여 큐 처리를 허용하며, 그 구간의 배차는 이미
    종료를 관측한 뒤이므로 시나리오별 독립 변수가 된다.
"""

import time
from collections import defaultdict

import gurobipy as gp
from gurobipy import GRB

__all__ = ["info_cells", "trip_availability", "build_adaptive_group",
           "solve_stage2_adaptive", "DISPATCH_BUFFER"]

# 장애 종료 후 추가 배차 허용 시간(분). 0 이면 종료 즉시 배차 중단.
DISPATCH_BUFFER = 0


# =============================================================================
# 1. 비예견성 정보집합
# =============================================================================
def info_cells(tau, disrupt_start, durations=(30, 60, 90), buffer=DISPATCH_BUFFER):
    """
    배차시각 tau 에서 만들어야 할 배차변수의 '정보집합' 목록.

    각 원소는 하나의 y 변수를 공유하는 duration 튜플이다.

      - 아직 진행 중인(구분 불가) 시나리오들 → 하나로 묶인다
      - 이미 종료를 관측한 시나리오들       → 각각 독립 (buffer > 0 일 때만 등장)

    buffer=0 이면 결과는 위 비예견성 정보집합과 정확히 같다:
        tau <= start+30 -> [(30,60,90)]
        tau <= start+60 -> [(60,90)]
        그 외           -> [(90,)]
    """
    durations = tuple(sorted(durations))
    avail = tuple(D for D in durations if tau <= disrupt_start + D + buffer)
    if not avail:
        return []
    ongoing = tuple(D for D in avail if tau <= disrupt_start + D)
    ended = tuple(D for D in avail if tau > disrupt_start + D)
    cells = []
    if ongoing:
        cells.append(ongoing)
    cells.extend((D,) for D in ended)      # 종료를 관측한 세계는 서로 구분된다
    return cells


def trip_availability(trip, disrupt_start, durations=(30, 60, 90),
                      buffer=DISPATCH_BUFFER):
    """지속시간 D 에서 이 trip 을 배차할 수 있는가 → {D: bool}."""
    tau = trip["DispatchStart"]
    return {D: (tau <= disrupt_start + D + buffer) for D in durations}


# =============================================================================
# 2. 그룹 구성 — 공통 trip 집합 + 지속시간별 cache
# =============================================================================
def build_adaptive_group(m3, station, start_h, wdata, raw_scenarios,
                         depots, bus_tt, route_lib, durations=(30, 60, 90),
                         buffer=DISPATCH_BUFFER):
    """
    한 사고 위치(station)에 대해 적응형 Stage 2 입력을 만든다.

    m3 : 3MILP 모듈 객체 (select_routes / generate_bus_trips /
         build_stage2_cache / build_scenario_route_library 사용)

    반환 dict
      station, disrupt_start, durations
      active         피해가 있는 지속시간 (나머지는 Z_D ≡ 0)
      scen[D]        지속시간별 시나리오 dict (피해 없으면 None)
      trips          공통 trip 집합 (가장 긴 지속시간 기준으로 1회 생성)
      avail[D]       그 지속시간에서 쓸 수 있는 tid 집합
      cache[D]       지속시간별 Stage 2 인덱스 캐시 (피해 없으면 None)
      cells          {tid: [정보집합, ...]}

    짧은 지속시간에 피해 OD 가 없는 역이 윈도우당 12~18개 있다 (Omega_m 이 비어
    raw 에 아예 없다). 이는 결측이 아니라 '피해 없음' 이므로 Z_D = 0 인 지속시간
    으로 취급한다. 배차변수와 정보집합은 물리적 실행가능성으로만 정해지므로
    durations 전체를 그대로 쓴다 — 30분에 아무도 안 묶인다는 것은 배차 시점에
    알 수 없는 사실이고, 바로 그래서 비예견성이 구속된다.
    """
    by_dur = {s["duration"]: s for s in raw_scenarios if s["station"] == station}
    longest_D = max(durations)
    if longest_D not in by_dur:
        # 임팩트는 지속시간에 단조이므로 최장에 피해가 없으면 어디에도 없다.
        raise ValueError(f"station={station}: duration {longest_D} 시나리오가 없음")

    longest = by_dur[longest_D]
    disrupt_start = longest["DISRUPT_START_T"]

    # 가장 넓은 배차창으로 후보 trip 을 한 번만 생성
    lib, _ = m3.build_scenario_route_library(longest, start_h, depots, bus_tt, route_lib)
    sel = m3.select_routes(longest, depots, lib, bus_tt)
    trips = m3.generate_bus_trips(longest, sel, lib)

    avail, cache, scen, active = {}, {}, {}, []
    for D in durations:
        base = by_dur.get(D)
        if base is None:
            avail[D], scen[D], cache[D] = set(), None, None   # 피해 없음 → Z_D = 0
            continue
        active.append(D)
        s = dict(base)
        ok = {tid: tr for tid, tr in trips.items()
              if trip_availability(tr, disrupt_start, durations, buffer)[D]}
        avail[D] = set(ok)
        s["bus_trips"] = ok
        scen[D] = s
        cache[D] = m3.build_stage2_cache(s, wdata, ok)

    cells = {tid: info_cells(tr["DispatchStart"], disrupt_start, durations, buffer)
             for tid, tr in trips.items()}

    return {"station": station, "disrupt_start": disrupt_start,
            "durations": tuple(sorted(durations)), "active": tuple(active),
            "buffer": buffer, "scen": scen, "trips": trips, "avail": avail,
            "cache": cache, "cells": cells}


# =============================================================================
# 3. 지속시간 하나의 승객 흐름 제약 (solve_stage2_fixed 와 동일 구조)
# =============================================================================
def _add_flow_block(m3, model, D, scenario, cache, wdata, y_of, tag):
    """
    지속시간 D 의 흐름변수와 제약을 모델에 추가한다.
    y_of(tid) : 그 지속시간에서 이 trip 의 배차변수 (없으면 None)
    반환 (b, a, l, q, w, od, b_by_trip, Z_expr)
    """
    Omega_m = scenario["Omega_m"]
    spare_cap = scenario["spare_capacity"]
    Omega_Map = wdata["Omega_Map"]
    Gamma = wdata["Gamma"]
    START_MIN = wdata["START_MIN"]
    station_of = wdata["station_of"]
    train_active_times = wdata["train_active_times"]
    train_dest_map = wdata["train_dest_map"]
    train_dwells = wdata["train_dwells"]
    Vm = wdata["Vm"]
    CumD = wdata["CumD"]
    Dem = wdata["D"]
    T = cache["T"]
    sigma = 2

    b, a, l, q, w, od = {}, {}, {}, {}, {}, {}
    for (om, vid, t, j) in cache["b_var_keys"]:
        ub = CumD.get((om, t), 0.0)
        b[(om, vid, t, j)] = model.addVar(lb=0, ub=max(ub, 0),
                                          vtype=GRB.CONTINUOUS, name=f"b{tag}")
    for (om, vid, t, j) in cache["a_var_keys"]:
        a[(om, vid, t, j)] = model.addVar(lb=0, ub=CumD.get((om, t), 0),
                                          vtype=GRB.CONTINUOUS, name=f"a{tag}")
    for (om, vid, t) in cache["l_var_keys"]:
        cap_v = m3.BUS_CAPACITY if vid.startswith("B_") else m3.TRAIN_CAPACITY
        l[(om, vid, t)] = model.addVar(lb=0, ub=cap_v, vtype=GRB.CONTINUOUS,
                                       name=f"l{tag}")
    for (om, j, t) in cache["q_var_keys"]:
        q[(om, j, t)] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"q{tag}")
    for (om, j, t) in cache["w_var_keys"]:
        w[(om, j, t)] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"w{tag}")
    for (om, t) in cache["od_var_keys"]:
        od[(om, t)] = model.addVar(lb=0, ub=CumD.get((om, t), 0.0),
                                   vtype=GRB.CONTINUOUS, name=f"od{tag}")
    model.update()

    # --- Load balance (버스) ---
    for tid, od_list in cache["trip_od_list"].items():
        for od_info in od_list:
            om = od_info["omega"]
            bt, at = od_info["board_t"], od_info["alight_t"]
            bj, aj = od_info["board_j"], od_info["alight_j"]
            for t in range(bt, at + 1):
                lkey = (om, tid, t)
                if lkey not in l:
                    continue
                board = b.get((om, tid, bt, bj), 0) if t == bt else 0
                alight = a.get((om, tid, at, aj), 0) if t == at else 0
                if t == bt:
                    model.addConstr(l[lkey] == board - alight)
                else:
                    prev = l.get((om, tid, t - 1))
                    if prev is not None:
                        model.addConstr(l[lkey] == prev + board - alight)

    # --- Load balance (열차) ---
    for om in Omega_m:
        if om not in Omega_Map:
            continue
        for v in Vm:
            if v not in train_active_times:
                continue
            for t in train_active_times[v]:
                lkey = (om, v, t)
                if lkey not in l:
                    continue
                s_t = station_of[v].get(t)
                board = b.get((om, v, t, s_t), 0) if s_t else 0
                alight = a.get((om, v, t, s_t), 0) if s_t else 0
                if t == train_active_times[v].start:
                    model.addConstr(l[lkey] == board - alight)
                else:
                    prev = l.get((om, v, t - 1))
                    if prev is not None:
                        model.addConstr(l[lkey] == prev + board - alight)

    # --- Bus capacity : 같은 지속시간의 배차변수를 참조 ---
    l_by_trip_t = defaultdict(list)
    for (om, tid, t), lvar in l.items():
        if tid.startswith("B_"):
            l_by_trip_t[(tid, t)].append(lvar)
    for (tid, t), lvars in l_by_trip_t.items():
        yv = y_of(tid)
        if yv is None:                       # 이 지속시간에서 못 쓰는 trip
            for lv in lvars:
                model.addConstr(lv == 0)
        else:
            model.addConstr(gp.quicksum(lvars) <= m3.BUS_CAPACITY * yv)

    # --- Train capacity ---
    l_by_train_t = defaultdict(list)
    for (om, vid, t), lvar in l.items():
        if not vid.startswith("B_"):
            l_by_train_t[(vid, t)].append(lvar)
    for (vid, t), lvars in l_by_train_t.items():
        model.addConstr(gp.quicksum(lvars) <= spare_cap.get((vid, t), 0.0))

    # --- Queue flow ---
    for om in Omega_m:
        if om not in Omega_Map:
            continue
        orig, dest = Omega_Map[om]
        for j in cache["N_omega"].get(om, set()):
            for ti, t in enumerate(Gamma):
                qvar = q.get((om, j, t))
                if qvar is None:
                    continue
                d_jt = Dem.get((om, j, t), 0.0)
                board_sum = gp.quicksum(
                    b[(om, vid, t, j)]
                    for vid in cache["boarding_by_od_t_j"].get((om, t, j), [])
                    if (om, vid, t, j) in b)
                alight_sum = gp.LinExpr()
                if t >= START_MIN + sigma:
                    for vid in cache["alighting_by_od_t_j"].get((om, t - sigma, j), []):
                        akey = (om, vid, t - sigma, j)
                        if akey in a:
                            alight_sum += a[akey]
                q_prev = q.get((om, j, t - 1), 0) if ti > 0 else 0
                if j == dest:
                    model.addConstr(qvar == q_prev + d_jt + alight_sum
                                    - board_sum - od.get((om, t), 0))
                else:
                    model.addConstr(qvar == q_prev + d_jt + alight_sum - board_sum)

    # --- Walking flow ---
    for om in Omega_m:
        if om not in Omega_Map:
            continue
        orig, dest = Omega_Map[om]
        for j in cache["N_omega"].get(om, set()):
            if j == dest:
                continue
            for ti, t in enumerate(Gamma):
                wvar = w.get((om, j, t))
                if wvar is None:
                    continue
                alight_in = gp.quicksum(
                    a[(om, vid, t, j)]
                    for vid in cache["alighting_by_od_t_j"].get((om, t, j), [])
                    if (om, vid, t, j) in a)
                alight_out = gp.LinExpr()
                if t >= START_MIN + sigma:
                    for vid in cache["alighting_by_od_t_j"].get((om, t - sigma, j), []):
                        akey = (om, vid, t - sigma, j)
                        if akey in a:
                            alight_out += a[akey]
                w_prev = w.get((om, j, t - 1), 0) if ti > 0 else 0
                model.addConstr(wvar == w_prev + alight_in - alight_out)

    # --- 탑승 논리 ---
    for om in Omega_m:
        if om not in Omega_Map:
            continue
        orig, dest = Omega_Map[om]
        for (om2, vid, t, j), bvar in b.items():
            if om2 == om and j == dest:
                model.addConstr(bvar == 0)
        for (om2, vid, t, j), avar in a.items():
            if om2 == om and j == orig:
                model.addConstr(avar == 0)
        for v in Vm:
            if v not in train_active_times:
                continue
            for dwell in train_dwells[v]:
                s_d, times = dwell["station"], dwell["times"]
                inc_load = l.get((om, v, times[0] - 1), 0)
                alight_sum = gp.quicksum(
                    a[(om, v, t, s_d)] for t in times if (om, v, t, s_d) in a)
                if alight_sum.size() > 0:
                    model.addConstr(alight_sum <= inc_load)
            if v in train_dest_map:
                t_last = train_active_times[v].stop - 1
                if (om, v, t_last) in l:
                    model.addConstr(l[(om, v, t_last)] == 0)

    # --- 최소 서비스 ---
    b_by_trip = defaultdict(list)
    for (om, tid, t, j), bvar in b.items():
        if tid.startswith("B_"):
            b_by_trip[tid].append(bvar)
    for tid in cache["trip_ids"]:
        yv = y_of(tid)
        bvars = b_by_trip.get(tid, [])
        if bvars and yv is not None:
            model.addConstr(gp.quicksum(bvars) >= m3.MIN_TRIP_BOARDING * yv)

    Z = (gp.quicksum(q.values()) + gp.quicksum(l.values())
         + gp.quicksum(w.values())) / T
    return b, a, l, q, w, od, b_by_trip, Z


# =============================================================================
# 4. 적응형 Stage 2
# =============================================================================
def solve_stage2_adaptive(m3, env, depots, depot_counts, group, wdata,
                          duration_weight=None, mip_gap=None, time_limit=None,
                          fixed_tids=None):
    """
    한 사고 위치에 대해 30/60/90 을 동시에 담은 적응형 배차 문제를 푼다.

    목적:  min  Σ_D  P(D) · Z_D
    배차:  y[(tid, cell)]  — cell 은 비예견성 정보집합
    흐름:  지속시간별 복제
    """
    t0 = time.time()
    durations = group["durations"]
    # 피해가 없는 지속시간은 Z_D = 0 이므로 흐름 블록을 만들지 않는다.
    # 확률은 durations 전체로 정규화한다 — 그 질량은 사라진 게 아니라 0 을 곱한다.
    active = group.get("active", durations)
    P = duration_weight or m3.DURATION_WEIGHT
    psum = sum(P.get(D, 0.0) for D in durations) or 1.0
    P = {D: P.get(D, 0.0) / psum for D in durations}

    sid = f"{group['station']}_adaptive"
    model = gp.Model(f"Stage2Adaptive_{sid}", env=env)
    model.Params.MIPGap = mip_gap if mip_gap is not None else m3.MIP_GAP
    model.Params.TimeLimit = time_limit if time_limit is not None else m3.TIME_LIMIT
    model.Params.OutputFlag = 0
    model.Params.MIPFocus = 1
    model.Params.Heuristics = 0.5

    # ---- 배차변수: 정보집합 하나당 변수 하나 (등식 제약 없이 공유) ----
    yvar = {}
    for tid, cells in group["cells"].items():
        for cell in cells:
            yvar[(tid, cell)] = model.addVar(
                vtype=GRB.BINARY, name=f"y[{tid}|{'-'.join(map(str, cell))}]")
    model.update()

    y_index = {}                       # (tid, D) -> 변수
    for (tid, cell), v in yvar.items():
        for D in cell:
            y_index[(tid, D)] = v

    # 순차 풀이용: 앞 단계에서 확정된 배차를 고정한다.
    # (단계별로 나눠 풀 때 이전 결정이 뒤 단계에서 번복되지 않게 한다)
    if fixed_tids:
        want = set(fixed_tids)
        for (tid, cell), v in yvar.items():
            if tid in want:
                v.LB = 1.0
        model.update()

    def y_of_factory(D):
        return lambda tid: y_index.get((tid, D))

    # ---- 지속시간별 흐름 블록 ----
    blocks, Zs = {}, {}
    for D in active:
        blocks[D] = _add_flow_block(m3, model, D, group["scen"][D],
                                    group["cache"][D], wdata,
                                    y_of_factory(D), tag=f"_{D}")
        Zs[D] = blocks[D][7]

    # ---- Depot 용량: 지속시간별로 (같은 y 를 참조) ----
    for D in active:
        trips_D = group["scen"][D]["bus_trips"]
        dt = defaultdict(lambda: defaultdict(list))
        for tid, trip in trips_D.items():
            for tau in range(max(trip["DispatchStart"], wdata["Gamma"][0]),
                             min(trip["ResourceEnd"], wdata["Gamma"][-1] + 1)):
                dt[trip["Depot"]][tau].append(tid)
        for d in depots:
            cap = depot_counts.get(d, 0)
            for tau, tids in dt.get(d, {}).items():
                act = [y_index[(tid, D)] for tid in tids if (tid, D) in y_index]
                if act:
                    model.addConstr(gp.quicksum(act) <= cap,
                                    name=f"depot_t[{d},{tau},{D}]")

    # ---- 목적함수 ----
    model.setObjective(gp.quicksum(P[D] * Zs[D] for D in active), GRB.MINIMIZE)
    model.update()
    build_sec = time.time() - t0

    t1 = time.time()
    model.optimize()
    solve_sec = time.time() - t1
    status = m3._status_name(model.Status)
    stats = m3.collect_gurobi_stats(model, solve_sec)
    stats["BuildSec"] = float(build_sec)

    if model.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT) \
            or model.SolCount == 0:
        model.dispose()
        return {"status": status, "objective": 0.0, "Z": {}, "policy": [],
                "y_shared": {}, "stats": stats}

    # ---- 결과 추출 ----
    Zval = {D: (float(Zs[D].getValue()) if D in Zs else 0.0) for D in durations}
    obj = float(model.ObjVal)

    policy = []
    for (tid, cell), v in sorted(yvar.items(),
                                 key=lambda kv: (group["trips"][kv[0][0]]["DispatchStart"],
                                                 kv[0][0])):
        if float(v.X) <= 0.5:
            continue
        tr = group["trips"][tid]
        boarded = {}
        for D in cell:
            bt = blocks[D][6].get(tid, []) if D in blocks else []
            boarded[D] = round(sum(float(x.X) for x in bt), 3)
        policy.append({
            "TripID": tid, "SharedDurations": cell,
            "Depot": tr["Depot"], "Route": tr["Route"],
            "RouteType": tr["RouteType"],
            "StartStation": tr["RouteOrigin"], "EndStation": tr["RouteDest"],
            "DispatchStart": tr["DispatchStart"], "RunStart": tr["RunStart"],
            "RunEnd": tr["RunEnd"], "Boarded": boarded,
        })

    y_shared = {(tid, cell): float(v.X) for (tid, cell), v in yvar.items()}
    model.dispose()
    return {"status": status, "objective": obj, "Z": Zval,
            "policy": policy, "y_shared": y_shared, "stats": stats,
            "weights": P}
