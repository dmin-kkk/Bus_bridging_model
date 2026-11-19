"""normal_generation.py — Normal(무장애) 시나리오 생성

disruption_generation.py 와 동일한 데이터 준비 / rel 계산 / 제약 구조를
쓰되, 다음만 다르다:

    - theta(운행 취소) 없음  -> 모든 열차가 정상 운행, cap_limit = 전체 여유용량
    - Omega 전체(206개)를 한 모델로 풀이  -> disrupted의 Omega_m 필터 없음
    - warm start 없음 (duration 개념이 없으므로)

Disrupted와 같은 가정을 써야 delta 비교가 성립하므로, 아래 3가지는 프로파일의
같은 키에서 읽는다 (disruption_generation.py 와 공유):
    network.max_transfers, scenarios.drain_buffer_min, inputs.walking_times (sigma)

사용법 (BB_PROFILE 로 프로파일 선택):
    python normal_generation.py                # time.windows 전체 순차
    python normal_generation.py --window <h>   # 단일 윈도우 (병렬용)
"""

import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import gurobipy as gp
import pandas as pd
import psutil
from gurobipy import GRB

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_ROOT.parent))
sys.path.insert(0, str(PIPELINE_ROOT))
import network_data  # noqa: E402
from bbconfig import cfg  # noqa: E402

_PROC = psutil.Process(os.getpid())


def _rss_mb():
    return _PROC.memory_info().rss / (1024 * 1024)


total_start_time = time.time()
print(f"[{time.strftime('%H:%M:%S')}] Initializing & Loading Data...")

DIR_NORMAL = str(Path(
    os.environ.get("PAPER_NORMAL_DIR") or cfg.results_dir / cfg.get("paths.normal")
).resolve())
os.makedirs(DIR_NORMAL, exist_ok=True)

env = gp.Env(empty=True)
env.setParam("OutputFlag", 0)
# 병렬 실행 시 워커끼리 스레드 경합이 생기므로 runner가 GRB_THREADS로 제한한다.
# 단독 실행이면 0(=전체 코어) 유지.
env.setParam("Threads", int(os.environ.get("GRB_THREADS") or cfg.get("parallel.normal")["threads"]))
env.start()

schedule_df = pd.read_csv(cfg.path("inputs.timetable"))
schedule_df["Trip"] = schedule_df["Trip"].astype(str)
od_df = pd.read_csv(cfg.path("inputs.od_matrix"))

MIN_PER_HOUR = 60
TIME_STEP = int(cfg.get("scenarios.time_step_min"))
STEPS_PER_HOUR = MIN_PER_HOUR // TIME_STEP
OD_HOURLY_MIN_TRIPS = float(os.environ.get("PAPER_OD_HOURLY_MIN_TRIPS")
                            or cfg.get("operations.od_hourly_min_trips"))
HORIZON_MIN = int(cfg.get("time.horizon_min"))
WINDOW_HOURS = HORIZON_MIN // MIN_PER_HOUR
MAX_TRANSFERS = network_data.max_transfers()
Cv = float(cfg.get("operations.rail_capacity"))
EPS_DEMAND = 1e-9    # per-step demand below this is zero
REPORT_TOL = 1e-4    # flows at or below this are not written
KAPPA = float(cfg.get("operations.out_of_vehicle_weight"))   # Table 3: kappa (Eq. (2))


def _steps(minutes):
    """Minutes -> model time steps (must be a whole number of steps)."""
    steps, remainder = divmod(int(minutes), TIME_STEP)
    if remainder:
        raise ValueError(f"{minutes} min is not a multiple of the {TIME_STEP}-min time step")
    return steps


def parse_time_to_min(t_str, base_hour):
    if pd.isna(t_str):
        return None
    try:
        t = datetime.strptime(t_str, "%H:%M:%S")
        minutes = (t.hour - base_hour) * MIN_PER_HOUR + t.minute
        return int(minutes // TIME_STEP)
    except Exception:
        return None


# --- 병렬 실행 지원 ---
START_HOURS = [int(window) for window in cfg.get("time.windows")]
_STATS_SUFFIX = ""
if "--window" in sys.argv:
    _w = int(sys.argv[sys.argv.index("--window") + 1])
    START_HOURS = [_w]
    _STATS_SUFFIX = f"_w{_w:02d}"
    print(f"[PARALLEL] window {_w:02d} 단독 실행")

STATS_FILE = f"{DIR_NORMAL}/_normal_stats{_STATS_SUFFIX}.csv"

# 재개 시 저장된 통계를 이어쓴다.
# (완료된 윈도우는 skip되어 stat이 새로 안 쌓이는데, 빈 리스트로 시작하면
#  마지막 저장에서 파일이 덮어써져 이전 성능 기록이 전부 날아간다.)
scenario_stats = []
_done_windows = set()
if os.path.exists(STATS_FILE):
    try:
        _prev = pd.read_csv(STATS_FILE)
        scenario_stats = _prev.to_dict("records")
        _done_windows = set(_prev["window"].astype(str))
        print(f"[RESUME] 저장된 통계 {len(scenario_stats)}건 로드 "
              f"({STATS_FILE})")
    except Exception as e:
        print(f"[RESUME] 저장된 통계 로드 실패 (새로 시작): {e}")

for start_h in START_HOURS:
    end_h = start_h + WINDOW_HOURS
    window_name = f"{start_h:02d}_{end_h:02d}"
    print(f"\n{'=' * 80}")
    print(f"Processing Window: {start_h:02d}:00 ~ {end_h:02d}:00")

    out_file = f"{DIR_NORMAL}/Normal_{window_name}.csv"
    if os.path.exists(out_file):
        # 결과가 이미 있으면 건너뛴다. 통계는 위에서 로드해 뒀으므로 보존된다.
        mark = "통계 있음" if window_name in _done_windows else "통계 없음"
        print(f"  > [SKIP] 이미 존재: {out_file}  ({mark})")
        continue

    BASE_HOUR = start_h
    START_MIN = 0
    END_MIN = _steps(HORIZON_MIN)
    # Disrupted와 동일하게 수요 지평 = 시뮬 지평 (경계 불일치 방지)
    DRAIN_BUFFER_MIN = _steps(cfg.get("scenarios.drain_buffer_min"))
    END_MIN_SIM = END_MIN + DRAIN_BUFFER_MIN
    SIM_HOURS = list(range(start_h, end_h))

    current_sched = schedule_df[["Trip", "StationID", "SchArrive", "SchDepart",
                                 "Seq", "Line", "Direction"]].copy()
    min_seq = current_sched.groupby("Trip")["Seq"].transform("min")
    max_seq = current_sched.groupby("Trip")["Seq"].transform("max")
    current_sched["is_first"] = current_sched["Seq"] == min_seq
    current_sched["is_last"] = current_sched["Seq"] == max_seq
    current_sched["event_time"] = current_sched.apply(
        lambda r: r["SchDepart"] if r["is_first"] else r["SchArrive"], axis=1)
    current_sched["t"] = current_sched["event_time"].apply(
        lambda x: parse_time_to_min(x, BASE_HOUR))
    filtered_sched = current_sched[
        (current_sched["t"] >= START_MIN) & (current_sched["t"] <= END_MIN_SIM)
    ].copy()

    if filtered_sched.empty:
        continue

    train_dest_map = filtered_sched[filtered_sched["is_last"]] \
        .set_index("Trip")["StationID"].to_dict()

    # OD Filtering (Disrupted와 동일 기준)
    target_od = (
        od_df[od_df["Hour"].isin(SIM_HOURS)]
        .groupby(["OriginID", "DestinationID", "Hour"], as_index=False)[
            "Trips_Hourly"].sum()
    )
    target_od = target_od[
        target_od["Trips_Hourly"] > OD_HOURLY_MIN_TRIPS].copy()
    valid_pairs = set(target_od[[
        "OriginID", "DestinationID"]].itertuples(index=False, name=None))
    valid_stations = set(filtered_sched["StationID"].unique())

    final_od_list = []
    Omega_Set = set()
    Omega_Map = {}
    for (orig, dest), group in target_od.groupby(["OriginID", "DestinationID"]):
        if (orig, dest) not in valid_pairs:
            continue
        if orig not in valid_stations or dest not in valid_stations:
            continue
        key_str = f"{orig}_{dest}"
        Omega_Set.add(key_str)
        Omega_Map[key_str] = (orig, dest)
        for row in group.itertuples(index=False):
            final_od_list.append({"omega": key_str, "OriginID": orig,
                                  "Hour": row.Hour, "Trips": row.Trips_Hourly})

    Omega = list(Omega_Set)
    N = list(valid_stations)
    Vm = filtered_sched["Trip"].unique().tolist()
    Gamma = list(range(START_MIN, END_MIN_SIM + 1))

    trains_at = defaultdict(lambda: defaultdict(list))
    station_of = defaultdict(dict)
    I_vts = set()
    for row in filtered_sched.itertuples(index=False):
        v, s, t = row.Trip, row.StationID, row.t
        I_vts.add((v, t, s))
        trains_at[t][s].append(v)
        station_of[v][t] = s

    train_active_times = {}
    trip_ranges = filtered_sched.groupby("Trip")["t"].agg(["min", "max"])
    buffer = _steps(cfg.get("scenarios.train_active_buffer_min"))
    for v, row in trip_ranges.iterrows():
        train_active_times[v] = range(max(START_MIN, row["min"] - buffer),
                                      min(END_MIN_SIM, row["max"] + buffer) + 1)

    # Demand
    D = defaultdict(float)
    CumD = defaultdict(float)
    steps_per_h = STEPS_PER_HOUR
    for item in final_od_list:
        omega = item["omega"]
        hour_offset = (item["Hour"] - BASE_HOUR) * STEPS_PER_HOUR
        t_start = max(START_MIN, hour_offset)
        t_end = min(END_MIN, hour_offset + STEPS_PER_HOUR - 1)
        if t_start > t_end:
            continue
        val = item["Trips"] / steps_per_h
        if val > EPS_DEMAND:
            for t in range(t_start, t_end + 1):
                if t in Gamma:
                    D[(omega, item["OriginID"], t)] = val

    for omega in Omega:
        cum = 0
        origin = Omega_Map[omega][0]
        for t in Gamma:
            cum += D.get((omega, origin, t), 0)
            CumD[(omega, t)] = cum

    # Dwell times
    train_dwells = defaultdict(list)
    for v in Vm:
        if v not in train_active_times:
            continue
        times = sorted([t for t in train_active_times[v] if t in station_of[v]])
        if not times:
            continue
        curr_s = station_of[v][times[0]]
        curr_seg = [times[0]]
        for t in times[1:]:
            s = station_of[v][t]
            if s == curr_s and t == curr_seg[-1] + 1:
                curr_seg.append(t)
            else:
                train_dwells[v].append({"station": curr_s, "times": curr_seg})
                curr_s = s
                curr_seg = [t]
        train_dwells[v].append({"station": curr_s, "times": curr_seg})

    valid_keys = list(I_vts)
    # sigma_s: walking transfer time per station [steps] (inputs.walking_times)
    sigma = network_data.transfer_steps(N, TIME_STEP)
    no_board_keys = [(v, t, s) for (v, t, s) in valid_keys
                     if v in train_dest_map and s == train_dest_map[v]]

    trains_at_ts = {}
    for (v, t, s) in valid_keys:
        trains_at_ts.setdefault((t, s), []).append(v)
    EMPTY = []

    def _trains_at(t, s):
        return trains_at_ts.get((t, s), EMPTY)

    train_seq = {}
    for v in Vm:
        _ts = sorted(station_of[v])
        train_seq[v] = ([station_of[v][t] for t in _ts], _ts)
    _seq_items = list(train_seq.items())

    # === 환승 <= MAX_TRANSFERS 경로 기반 sparse rel (Disrupted와 동일 로직) ===
    _MAX_RIDES = MAX_TRANSFERS + 1
    _INF = float("inf")
    _rel_cache = {}

    def _rel_stations(orig, dest):
        fw = [defaultdict(lambda: _INF) for _ in range(_MAX_RIDES + 1)]
        fw[0][orig] = START_MIN
        for r in range(1, _MAX_RIDES + 1):
            prev, cur = fw[r - 1], fw[r]
            for s, t in prev.items():
                if t < cur[s]:
                    cur[s] = t
            for v, (sts, ts) in _seq_items:
                board = _INF
                for i, s in enumerate(sts):
                    tp = prev[s]
                    if tp < _INF:
                        need = tp if (s == orig and tp == START_MIN) else tp + sigma[s]
                        if ts[i] >= need and ts[i] < board:
                            board = ts[i]
                    if board < _INF and ts[i] >= board and ts[i] < cur[s]:
                        cur[s] = ts[i]
        bw = [defaultdict(lambda: -_INF) for _ in range(_MAX_RIDES + 1)]
        bw[0][dest] = END_MIN_SIM
        for r in range(1, _MAX_RIDES + 1):
            prev, cur = bw[r - 1], bw[r]
            for s, t in prev.items():
                if t > cur[s]:
                    cur[s] = t
            for v, (sts, ts) in _seq_items:
                alight = -_INF
                for i in range(len(sts) - 1, -1, -1):
                    s = sts[i]
                    tn = prev[s]
                    if tn > -_INF:
                        lim = tn if s == dest else tn - sigma[s]
                        if ts[i] <= lim and ts[i] > alight:
                            alight = ts[i]
                    if alight > -_INF and ts[i] <= alight and ts[i] > cur[s]:
                        cur[s] = ts[i]
        rel = {orig, dest}
        cand = set()
        for r in range(_MAX_RIDES + 1):
            cand |= fw[r].keys() | bw[r].keys()
        for s in cand:
            for r1 in range(_MAX_RIDES + 1):
                a = fw[r1][s]
                if a < _INF and bw[_MAX_RIDES - r1][s] >= a:
                    rel.add(s)
                    break
        return rel

    valid_by_station = defaultdict(list)
    for (v, t, s) in valid_keys:
        valid_by_station[s].append((v, t, s))

    dwell_index = []
    for v in Vm:
        if v not in train_active_times:
            continue
        for dwell in train_dwells[v]:
            s_d = dwell["station"]
            times = tuple(dwell["times"])
            t0 = times[0] - 1
            has_prev = t0 in train_active_times[v]
            dwell_index.append((v, s_d, times, t0 if has_prev else None))

    print(f"  |N|={len(N)} |Vm|={len(Vm)} |Omega|={len(Omega)} |Gamma|={len(Gamma)} "
          f"|I_vts|={len(valid_keys)}")

    stat = {
        "window": window_name, "t_precompute": 0.0, "t_build": 0.0,
        "t_solve": 0.0, "t_extract": 0.0, "t_total": 0.0,
        "omega_total": len(Omega), "num_vars": 0, "num_constrs": 0, "num_nz": 0,
        "od_hourly_min_trips": OD_HOURLY_MIN_TRIPS,
        "runtime_gurobi": 0.0, "rss_after_solve_mb": 0.0,
        "status": "RUNNING", "obj": float("nan"), "error": "",
        "total_demand": 0.0, "queue_pax_min": 0.0, "load_pax_min": 0.0,
        "walk_pax_min": 0.0, "total_pax_min": 0.0, "checkpoint": "INIT",
    }
    scenario_stats.append(stat)

    def _flush():
        pd.DataFrame(scenario_stats).to_csv(STATS_FILE, index=False)

    _flush()
    t_win_start = time.time()

    # rel 프리컴퓨트
    t_pre = time.time()
    om_rel_stations = {}
    for om in Omega:
        orig, dest = Omega_Map[om]
        c = _rel_cache.get((orig, dest))
        if c is None:
            c = _rel_stations(orig, dest)
            _rel_cache[(orig, dest)] = c
        om_rel_stations[om] = c
    stat["t_precompute"] = time.time() - t_pre
    avg_rel = sum(len(v) for v in om_rel_stations.values()) / max(1, len(om_rel_stations))
    print(f"  [Sparse] avg |rel_stations| per om = {avg_rel:.1f} / |N|={len(N)}")

    # Normal: 취소 없음 -> 전 열차 정상, 여유용량 = Cv
    cap_limit = {}
    for v_id in Vm:
        if v_id in train_active_times:
            for t in train_active_times[v_id]:
                cap_limit[(v_id, t)] = float(Cv)

    Omega_L = list(Omega)

    try:
        t_build = time.time()
        m = gp.Model(f"Normal_{window_name}", env=env)
        m.setParam("MIPGap", cfg.get("scenarios.solver.mip_gap"))
        m.setParam("Presolve", cfg.get("scenarios.solver.presolve"))

        qd_keys = [(om, s, t) for om in Omega_L
                   for s in om_rel_stations[om] for t in Gamma]
        dd_keys = [(om, s, t) for om in Omega_L
                   for s in om_rel_stations[om]
                   if s != Omega_Map[om][1] for t in Gamma]
        l_ind = [(om, v, t) for om in Omega_L for v in Vm
                 if v in train_active_times for t in train_active_times[v]]

        qd = m.addVars(qd_keys, vtype=GRB.CONTINUOUS, lb=0)
        ld = m.addVars(l_ind, vtype=GRB.CONTINUOUS, lb=0)
        ad = m.addVars(Omega_L, valid_keys, vtype=GRB.CONTINUOUS, lb=0)
        bd = m.addVars(Omega_L, valid_keys, vtype=GRB.CONTINUOUS, lb=0)
        od = m.addVars(Omega_L, Gamma, vtype=GRB.CONTINUOUS, lb=0)
        dd = m.addVars(dd_keys, vtype=GRB.CONTINUOUS, lb=0)

        chunk_size = int(cfg.get("scenarios.constraint_block_ods"))
        for i in range(0, len(Omega_L), chunk_size):
            om_chunk = Omega_L[i:i + chunk_size]

            m.addConstrs((qd[om, Omega_Map[om][1], t] ==
                (qd[om, Omega_Map[om][1], t - 1] if t > START_MIN else 0)
                + D.get((om, Omega_Map[om][1], t), 0)
                + gp.quicksum(ad[om, v, t, Omega_Map[om][1]]
                              for v in _trains_at(t, Omega_Map[om][1]))
                - gp.quicksum(bd[om, v, t, Omega_Map[om][1]]
                              for v in _trains_at(t, Omega_Map[om][1]))
                - od[om, t]
                for om in om_chunk for t in Gamma))

            m.addConstrs((qd[om, s, t] ==
                (qd[om, s, t - 1] if t > START_MIN else 0)
                + D.get((om, s, t), 0)
                + gp.quicksum(ad[om, v, t - sg, s]
                              for v in _trains_at(t - sg, s)
                              if t >= START_MIN + sg)
                - gp.quicksum(bd[om, v, t, s] for v in _trains_at(t, s))
                for om in om_chunk
                for s in om_rel_stations[om] if s != Omega_Map[om][1]
                for sg in (sigma[s],)
                for t in Gamma))

            m.addConstrs((dd[om, s, t] ==
                (dd[om, s, t - 1] if t > START_MIN else 0)
                + gp.quicksum(ad[om, v, t, s] for v in _trains_at(t, s))
                - (gp.quicksum(ad[om, v, t - sg, s]
                               for v in _trains_at(t - sg, s))
                   if t >= START_MIN + sg else 0)
                for om in om_chunk
                for s in om_rel_stations[om] if s != Omega_Map[om][1]
                for sg in (sigma[s],)
                for t in Gamma))

            m.addConstrs((ld[om, v, t] ==
                (ld[om, v, t - 1] if t > train_active_times[v].start else 0)
                + (bd[om, v, t, station_of[v][t]] if t in station_of[v] else 0)
                - (ad[om, v, t, station_of[v][t]] if t in station_of[v] else 0)
                for om in om_chunk for v in Vm if v in train_active_times
                for t in train_active_times[v]))

            if no_board_keys:
                m.addConstrs((bd[om, v, t, s] == 0
                              for om in om_chunk for (v, t, s) in no_board_keys))

        m.addConstrs((gp.quicksum(ld[om, v, t] for om in Omega_L) <= cap_limit[(v, t)]
                      for (v, t) in cap_limit))
        m.addConstrs((gp.quicksum(bd[om, v, t, s] for om in Omega_L) <= Cv
                      for (v, t, s) in valid_keys))
        m.addConstrs((ad[om, v, t, s] <= CumD[(om, t)]
                      for om in Omega_L for (v, t, s) in valid_keys))
        m.addConstrs((bd[om, v, t, s] <= CumD[(om, t)]
                      for om in Omega_L for (v, t, s) in valid_keys))

        for om in Omega_L:
            orig, dest = Omega_Map[om]
            if orig in valid_by_station:
                m.addConstrs((ad[om, v, t, s] == 0
                              for (v, t, s) in valid_by_station[orig]))
            if dest in valid_by_station:
                m.addConstrs((bd[om, v, t, s] == 0
                              for (v, t, s) in valid_by_station[dest]))

        # 보존 법칙 보호: rel 밖 승하차 금지 (Disrupted와 동일)
        for om in Omega_L:
            rel_om = om_rel_stations[om]
            outside = [(v, t, s) for (v, t, s) in valid_keys if s not in rel_om]
            if outside:
                m.addConstrs((ad[om, v, t, s] == 0 for (v, t, s) in outside))
                m.addConstrs((bd[om, v, t, s] == 0 for (v, t, s) in outside))

        m.addConstrs((
            gp.quicksum(ad[om, v, t, s_d] for t in times) <=
            (ld[om, v, t0] if t0 is not None else 0)
            for om in Omega_L
            for (v, s_d, times, t0) in dwell_index))

        dest_trains = [v for v in Vm
                       if v in train_dest_map and v in train_active_times]
        m.addConstrs((ld[om, v, train_active_times[v].stop - 1] == 0
                      for om in Omega_L for v in dest_trains))

        # Eq. (3) [eq:obj_n]: Z^N = (1/T) sum_{omega} sum_{t in Gamma} z_t with Eq. (2)
        # z_t = kappa [queue + walking] + onboard load (Table 3: T, kappa).
        m.setObjective(
            (KAPPA * (qd.sum() + dd.sum()) + ld.sum()) / _steps(HORIZON_MIN),
            GRB.MINIMIZE)

        m.update()
        stat["num_vars"] = m.NumVars
        stat["num_constrs"] = m.NumConstrs
        stat["num_nz"] = m.NumNZs
        stat["t_build"] = time.time() - t_build
        stat["checkpoint"] = "BUILT"
        _flush()

        t_solve = time.time()
        m.optimize()
        stat["t_solve"] = time.time() - t_solve
        stat["rss_after_solve_mb"] = _rss_mb()
        try:
            stat["runtime_gurobi"] = m.Runtime
        except Exception:
            pass
        stat["status"] = {
            GRB.OPTIMAL: "OPTIMAL", GRB.INFEASIBLE: "INFEASIBLE",
            GRB.UNBOUNDED: "UNBOUNDED", GRB.TIME_LIMIT: "TIME_LIMIT",
            GRB.SUBOPTIMAL: "SUBOPTIMAL",
        }.get(m.status, f"STATUS_{m.status}")
        try:
            stat["obj"] = m.ObjVal
        except Exception:
            pass
        stat["checkpoint"] = "SOLVED"
        _flush()

        if m.status == GRB.OPTIMAL:
            t_ext = time.time()
            rows = []
            tt_q = tt_l = tt_d = 0.0
            for om in Omega_L:
                orig, dest = Omega_Map[om]
                for s in om_rel_stations[om]:
                    for t in Gamma:
                        qv = qd[om, s, t].X
                        if qv > REPORT_TOL:
                            rows.append(("Queue", om, orig, dest, t, s, None, qv))
                            tt_q += qv
                        if s != dest:
                            dv = dd[om, s, t].X
                            if dv > REPORT_TOL:
                                rows.append(("Walking", om, orig, dest, t, s, None, dv))
                                tt_d += dv
                for v in Vm:
                    if v in train_active_times:
                        for t in train_active_times[v]:
                            lv = ld[om, v, t].X
                            if lv > REPORT_TOL:
                                rows.append(("Load", om, orig, dest, t, None, v, lv))
                                tt_l += lv
                for (v, t, s) in valid_keys:
                    bv = bd[om, v, t, s].X
                    if bv > REPORT_TOL:
                        rows.append(("Boarding", om, orig, dest, t, s, v, bv))
                    av = ad[om, v, t, s].X
                    if av > REPORT_TOL:
                        rows.append(("Alighting", om, orig, dest, t, s, v, av))
                for t in Gamma:
                    ov = od[om, t].X
                    if ov > REPORT_TOL:
                        rows.append(("Exit", om, orig, dest, t, dest, None, ov))

            tot_dem = 0.0
            for om in Omega_L:
                orig = Omega_Map[om][0]
                for t in Gamma:
                    tot_dem += D.get((om, orig, t), 0.0)
            stat["total_demand"] = tot_dem
            stat["queue_pax_min"] = tt_q
            stat["load_pax_min"] = tt_l
            stat["walk_pax_min"] = tt_d
            stat["total_pax_min"] = tt_q + tt_l + tt_d

            pd.DataFrame(rows, columns=["Variable", "Omega", "Origin", "Dest",
                                        "Time", "StationID", "TrainID", "Value"]
                         ).to_csv(out_file, index=False)
            stat["t_extract"] = time.time() - t_ext
            stat["checkpoint"] = "EXTRACTED"
            _flush()
            print(f"  -> Saved {out_file} (vars={stat['num_vars']:,} "
                  f"build={stat['t_build']:.1f}s solve={stat['t_solve']:.1f}s "
                  f"rss={stat['rss_after_solve_mb']:.0f}MB "
                  f"pax_min={stat['total_pax_min']:.0f} dem={tot_dem:.0f})")
        else:
            print(f"  -> 최적해 없음: {stat['status']}")
    except Exception as e:
        print(f"  -> Error: {e}")
        stat["error"] = str(e)[:200]
    finally:
        if "m" in locals():
            m.dispose()
        stat["t_total"] = time.time() - t_win_start
        if stat["checkpoint"] != "EXTRACTED":
            stat["checkpoint"] = "TERMINATED"
        _flush()

env.dispose()

if scenario_stats:
    df = pd.DataFrame(scenario_stats)
    df.to_csv(STATS_FILE, index=False)
    print(f"\nNormal stats saved: {STATS_FILE} ({len(df)} rows)")
    ok = df[df["status"] == "OPTIMAL"]
    if not ok.empty:
        print(f"  build 합계 {ok['t_build'].sum():.1f}s  "
              f"solve 합계 {ok['t_solve'].sum():.1f}s  "
              f"peak RSS {ok['rss_after_solve_mb'].max():.0f}MB")

print(f"\nAll Tasks Completed. Total Time: {time.time() - total_start_time:.2f}s")
