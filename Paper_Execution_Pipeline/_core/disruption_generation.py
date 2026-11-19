"""
disruption_generation.py — 트랙 공유 구간을 반영한 disruption 시나리오 생성

노선(Line, Direction) 단위 정차 순서만으로 판정하면, 한 노선의 열차가 다른 노선
전용역 구간을 무정차 통과할 때 영향 판정이 누락된다.

판정 방식:
    inputs.corridors (트랙 컬럼별 물리적 역 순서; network_data.track_stations)를
    사용하여 "물리적 트랙 위 sequence"로 위치 비교. 열차의 정차역들이 같은 트랙
    위에 있으면(line 무관), target_station이 그 사이에 있을 때 통과 판정.

모든 값(윈도우, 지평, 사고 시작, 지속시간, 용량, 환승시간, 최대 환승 횟수, 입력
파일)은 활성 프로파일(bbconfig.cfg)에서 읽는다.
"""

import os
import sys
import time
from bisect import bisect_left
from collections import defaultdict
from datetime import datetime
from math import atan2, cos, radians, sin, sqrt
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

# ------------------------------------------------------------------------------
# 0. Global Setup
# ------------------------------------------------------------------------------
total_start_time = time.time()
print(f"[{time.strftime('%H:%M:%S')}] Initializing & Loading Data...")

DIR_NORMAL = str(Path(
    os.environ.get("PAPER_NORMAL_DIR") or cfg.results_dir / cfg.get("paths.normal")
).resolve())
DIR_DISRUPT = str(Path(
    os.environ.get("PAPER_DISRUPTION_DIR") or cfg.results_dir / cfg.get("paths.disruptions")
).resolve())
os.makedirs(DIR_DISRUPT, exist_ok=True)

env = gp.Env(empty=True)
env.setParam("OutputFlag", 0)
# 병렬 실행 시 워커끼리 스레드 경합이 생기므로 runner가 GRB_THREADS로 제한한다.
# 단독 실행이면 0(=전체 코어) 유지.
env.setParam("Threads", int(os.environ.get("GRB_THREADS") or cfg.get("parallel.disruptions")["threads"]))
# Method/Crossover/BarConvTol/Presolve/Aggregate/ScaleFlag 모두 기본값 사용
# (Barrier는 메모리를 많이 써서 16시 윈도우 일부 시나리오에서 OOM 발생)
env.start()

schedule_df = pd.read_csv(cfg.path("inputs.timetable"))
schedule_df["Trip"] = schedule_df["Trip"].astype(str)  # normal CSV의 TrainID와 형식 통일
od_df       = pd.read_csv(cfg.path("inputs.od_matrix"))

MIN_PER_HOUR = 60
MIN_PER_DAY = 24 * MIN_PER_HOUR
TIME_STEP = int(cfg.get("scenarios.time_step_min"))
STEPS_PER_HOUR = MIN_PER_HOUR // TIME_STEP
OD_HOURLY_MIN_TRIPS = float(os.environ.get("PAPER_OD_HOURLY_MIN_TRIPS")
                            or cfg.get("operations.od_hourly_min_trips"))
HORIZON_MIN = int(cfg.get("time.horizon_min"))
WINDOW_HOURS = HORIZON_MIN // MIN_PER_HOUR
DAY_WRAP_MIN = int(cfg.get("scenarios.day_wrap_min"))
PASS_THROUGH_FALLBACK = float(cfg.get("scenarios.pass_through_fallback_ratio"))
MAX_TRANSFERS = network_data.max_transfers()
Cv = float(cfg.get("operations.rail_capacity"))
EARTH_RADIUS_KM = 6371.0
EPS_DEMAND = 1e-9    # per-step demand below this is zero
REPORT_TOL = 1e-4    # flows at or below this are not written
KAPPA = float(cfg.get("operations.out_of_vehicle_weight"))   # Table 3: kappa (Eq. (2))
ONSET_MIN = int(cfg.get("time.disruption_onset_min"))        # Table 3: t0
WARM_TOL = 1e-6      # values at or below this are not passed as warm start


def _steps(minutes):
    """Minutes -> model time steps (must be a whole number of steps)."""
    steps, remainder = divmod(int(minutes), TIME_STEP)
    if remainder:
        raise ValueError(f"{minutes} min is not a multiple of the {TIME_STEP}-min time step")
    return steps

# ------------------------------------------------------------------------------
# 0.0 Corridor track topology (inputs.corridors via network_data)
# ------------------------------------------------------------------------------
# 각 column = 하나의 물리적 트랙 sequence. 동일 corridor 내 다른 분기는 별도 column.
# track_position[track][station] = position along the track (0-based, file row order)
# track_stations[track] = ordered list of stations on this track
track_stations = network_data.track_stations()
TRACK_COLUMNS = list(track_stations)
track_position = network_data.track_positions()
# station → tracks containing it
station_to_tracks = network_data.station_tracks()

print(f"Loaded {len(TRACK_COLUMNS)} track columns: {TRACK_COLUMNS}")
print(f"Total stations across tracks: {sum(len(track_stations[c]) for c in TRACK_COLUMNS)} "
      f"({len(station_to_tracks)} unique)")

# ------------------------------------------------------------------------------
# 0.1 Station coordinates (거리 비례 보간용)
# ------------------------------------------------------------------------------
_xl = pd.ExcelFile(cfg.path("inputs.stations"))
_coord_df = (
    pd.concat([_xl.parse(s) for s in _xl.sheet_names], ignore_index=True)
    .drop_duplicates(subset="Code")
)
station_coord_map = (
    _coord_df.set_index("Code")[["Lat", "Long"]].apply(tuple, axis=1).to_dict()
)


def haversine_dist(s1, s2):
    c1 = station_coord_map.get(s1)
    c2 = station_coord_map.get(s2)
    if c1 is None or c2 is None:
        return None
    lat1, lon1 = radians(c1[0]), radians(c1[1])
    lat2, lon2 = radians(c2[0]), radians(c2[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return EARTH_RADIUS_KM * 2 * atan2(sqrt(a), sqrt(1 - a))


def parse_time_to_min(t_str, base_hour):
    if pd.isna(t_str):
        return None
    try:
        t = datetime.strptime(t_str, "%H:%M:%S")
        minutes = (t.hour - base_hour) * MIN_PER_HOUR + t.minute
        # Evening windows may contain trips that continue after midnight.
        # Interpret a clock time more than scenarios.day_wrap_min behind the
        # window start as the following day.
        if minutes < -DAY_WRAP_MIN:
            minutes += MIN_PER_DAY
        return int(minutes // TIME_STEP)
    except Exception:
        return None


# ------------------------------------------------------------------------------
# 0.2 Top stations target
# ------------------------------------------------------------------------------
station_vol = od_df.groupby("OriginID")["Trips_Hourly"].sum().sort_values(ascending=False)
TOP_STATIONS = station_vol.head(int(cfg.get("scenarios.target_station_limit"))).index.tolist()
print(f"Top stations (target for disruption): {len(TOP_STATIONS)}")

# Existing disruption CSVs generated before this version used one timestamp
# both for exposure detection and cancellation.  They must not be silently
# reused after the timing semantics below are corrected.
SCENARIO_LOGIC_VERSION = cfg.get("scenarios.logic_version")


def compute_cancellation_start(t_cross, t_cancel, disruption_start):
    """Return the first minute at which an exposed train is unavailable.

    A train that departed its last safe upstream stop before the disruption
    cannot be canceled retroactively at that departure.  It remains available
    until it reaches the disruption location.  A train that has not departed
    when the disruption starts is canceled at the upstream departure event.
    """
    if t_cancel < disruption_start:
        return int(t_cross)
    return int(t_cancel)


# ------------------------------------------------------------------------------
# Core: corridor-track 기반 disruption 판정
# ------------------------------------------------------------------------------
def compute_disruption_times_for_trip(target_station, trip_stops):
    """열차의 사고 노출시각과 안전 취소시각을 분리하여 반환한다.

    ``t_cross``는 열차가 사고 위치에 도달하거나 통과하는 시각이며 사고
    지속시간과의 중첩 여부를 판정할 때만 사용한다. ``t_cancel``은 사고 위치
    직전의 마지막 안전 정차역 출발시각이며, 노출된 열차의 ``theta``를 이
    시각부터 1로 만든다.

    반환값은 ``(t_cross, t_cancel, case)``이다. ``case``는 정차역이면 ``A``,
    통과역 보간이면 ``B``이다. 해당 열차가 사고역에 도달하지 않으면 None이다.
    """
    def _isnum(x):
        return x is not None and not (isinstance(x, float) and pd.isna(x))

    for i, (seq, sid, t_arr, t_dep) in enumerate(trip_stops):
        if sid == target_station:
            # 사고 노출은 사고역 자체의 도착시각으로 판정한다. 시발역의 t_arr는
            # window timetable 생성 시 출발시각으로 채워져 있다.
            t_cross = t_arr if _isnum(t_arr) else t_dep
            if i > 0:
                _, _, prev_arr, prev_dep = trip_stops[i - 1]
                t_cancel = prev_dep if _isnum(prev_dep) else prev_arr
            else:
                # target이 열차의 시발역이면 출발 자체를 취소한다.
                t_cancel = t_dep if _isnum(t_dep) else t_arr
            if not _isnum(t_cross) or not _isnum(t_cancel):
                return None
            return int(t_cross), int(t_cancel), "A"

    # Case B: 통과 — target_station을 포함한 트랙들 후보
    candidate_tracks = station_to_tracks.get(target_station, [])
    if not candidate_tracks:
        return None

    # 열차 정차역 집합
    train_stations = [(s[1], s[2], s[3]) for s in trip_stops]  # (station, t_arr, t_dep)

    best_candidate = None

    for track in candidate_tracks:
        pos_map = track_position[track]
        tgt_pos = pos_map.get(target_station)
        if tgt_pos is None:
            continue

        # 이 트랙에서 열차가 정차하는 역들의 (track_position, station, t_arr, t_dep)
        on_track = []
        for (sid, t_arr, t_dep) in train_stations:
            p = pos_map.get(sid)
            if p is not None:
                on_track.append((p, sid, t_arr, t_dep))
        if len(on_track) < 2:
            continue   # 이 트랙 위에 정차역이 2개 미만이면 통과 추정 불가

        on_track.sort(key=lambda x: x[0])

        # target_pos를 사이에 두는 인접한 정차역 한 쌍 찾기
        prev_stop = None
        next_stop = None
        for entry in on_track:
            if entry[0] < tgt_pos:
                prev_stop = entry
            elif entry[0] > tgt_pos and next_stop is None:
                next_stop = entry
                break
            elif entry[0] == tgt_pos:
                # 한 트랙에서는 같은 position이 안 나오지만 안전상 처리
                prev_stop = entry
                next_stop = entry
                break

        if prev_stop is None or next_stop is None:
            continue   # target이 열차 운행 구간 밖
        if prev_stop is next_stop:
            # target과 정확히 일치하는 정차역(이미 Case A에서 처리됐어야 함)
            continue

        # 거리 비례 보간 (양방향 지원: D=하행, U=상행 모두)
        p_prev_pos, p_prev_sid, p_prev_arr, p_prev_dep = prev_stop
        n_pos, n_sid, n_arr, n_dep = next_stop

        def _valid(x):
            return x is not None and not (isinstance(x, float) and pd.isna(x))

        t_prev_dep = p_prev_dep if _valid(p_prev_dep) else p_prev_arr
        t_prev_arr = p_prev_arr if _valid(p_prev_arr) else p_prev_dep
        t_next_arr = n_arr if _valid(n_arr) else n_dep
        t_next_dep = n_dep if _valid(n_dep) else n_arr

        # 시간 순서로 (출발측, 도착측) 결정
        if _valid(t_prev_dep) and _valid(t_next_arr) and t_next_arr > t_prev_dep:
            # 하행(D): prev(낮은 pos) → next(높은 pos), 시간상 prev가 먼저
            t_from, t_to = t_prev_dep, t_next_arr
            from_sid = p_prev_sid
        elif _valid(t_next_dep) and _valid(t_prev_arr) and t_prev_arr > t_next_dep:
            # 상행(U): next(높은 pos) → prev(낮은 pos), 시간상 next가 먼저
            t_from, t_to = t_next_dep, t_prev_arr
            from_sid = n_sid
        else:
            continue

        d_total  = haversine_dist(p_prev_sid, n_sid)
        d_to_tgt = haversine_dist(from_sid, target_station)
        if d_total and d_to_tgt and d_total > 0:
            ratio = d_to_tgt / d_total
            ratio = max(0.0, min(1.0, ratio))
        else:
            ratio = PASS_THROUGH_FALLBACK

        t_cross = int(t_from + ratio * (t_to - t_from))
        t_cancel = int(t_from)
        candidate = (t_cross, t_cancel, "B")
        if (best_candidate is None
                or (candidate[0], candidate[1])
                < (best_candidate[0], best_candidate[1])):
            best_candidate = candidate

    return best_candidate


# ------------------------------------------------------------------------------
# 1. Main Loop
# ------------------------------------------------------------------------------
# Duration states and windows of the active profile (time.durations_min, time.windows).
START_HOURS = [int(window) for window in cfg.get("time.windows")]
DURATIONS = [int(value) for value in cfg.get("time.durations_min")]
_env_durations = os.environ.get("PAPER_DURATIONS")
if _env_durations is not None and [
        int(value) for value in _env_durations.split(",") if value.strip()] != DURATIONS:
    raise ValueError(
        f"PAPER_DURATIONS={_env_durations} differs from time.durations_min {DURATIONS}")
# Optional subset of disruption targets (--targets A,B,...): test runs only.
_TEST_TARGET_FILTER = None
if "--targets" in sys.argv:
    _TEST_TARGET_FILTER = [
        token.strip() for token in sys.argv[sys.argv.index("--targets") + 1].split(",")
        if token.strip()]
    print(f"[TARGETS] restricted to {_TEST_TARGET_FILTER}")

# --- 병렬 실행 지원: --window <시작시각> 로 단일 윈도우만 처리 ---
# 윈도우별 프로세스를 병렬로 띄울 때 사용한다.
# 인자가 없으면 전체 윈도우를 순차 처리한다.
_STATS_SUFFIX = ""
if "--window" in sys.argv:
    _w = int(sys.argv[sys.argv.index("--window") + 1])
    START_HOURS = [_w]
    _STATS_SUFFIX = f"_w{_w:02d}"
    print(f"[PARALLEL] window {_w:02d} 단독 실행")

# 시나리오별 통계 수집 — 마지막에 CSV로 저장
# 병렬 실행 시 프로세스별로 파일을 분리해야 서로 덮어쓰지 않는다.
STATS_FILE = f"{DIR_DISRUPT}/_scenario_stats{_STATS_SUFFIX}.csv"

# 재개 시 저장된 통계를 이어쓴다.
# (완료된 시나리오는 CSV 존재로 skip되어 stat이 새로 안 쌓이는데, 빈 리스트로
#  시작하면 _flush_stats()가 파일을 덮어써 이전 성능 기록이 전부 날아간다.)
scenario_stats = []
_done_scenarios = set()
_scenario_logic_versions = {}
_scenario_checkpoints = {}
if os.path.exists(STATS_FILE):
    try:
        _prev = pd.read_csv(STATS_FILE)
        scenario_stats = _prev.to_dict("records")
        _done_scenarios = set(_prev["scenario"].astype(str))
        if "logic_version" in _prev.columns:
            _scenario_logic_versions = {
                str(row.scenario): str(row.logic_version)
                for row in _prev[["scenario", "logic_version"]].itertuples(index=False)
                if pd.notna(row.logic_version)
            }
        if "checkpoint" in _prev.columns:
            _scenario_checkpoints = {
                str(row.scenario): str(row.checkpoint)
                for row in _prev[["scenario", "checkpoint"]].itertuples(index=False)
                if pd.notna(row.checkpoint)
            }
        print(f"[RESUME] 저장된 통계 {len(scenario_stats)}건 로드 ({STATS_FILE})")
    except Exception as e:
        print(f"[RESUME] 저장된 통계 로드 실패 (새로 시작): {e}")

for start_h in START_HOURS:
    end_h = start_h + WINDOW_HOURS
    window_name = f"{start_h:02d}_{end_h:02d}"
    print(f"\n{'=' * 80}")
    print(f"Processing Window: {start_h:02d}:00 ~ {end_h:02d}:00")

    BASE_HOUR = start_h
    START_MIN = 0
    END_MIN   = _steps(HORIZON_MIN)             # 수요 발생 종료 (D 입력 커트오프)
    # 시뮬 지평 = 수요 지평 (240). Normal 결과도 240 기준이므로 지평을 맞춰야
    # t>240 구간에서 normal=0 vs disrupted>0 인 가짜 delta가 생기지 않는다.
    # drain buffer로 건지는 승객은 Exit 기준 0.3%에 불과하고, 그마저도
    # 종료 시점 큐 잔량(=미도착 승객)으로 정확히 집계된다.
    DRAIN_BUFFER_MIN = _steps(cfg.get("scenarios.drain_buffer_min"))
    END_MIN_SIM = END_MIN + DRAIN_BUFFER_MIN    # 시뮬레이션 지평
    SIM_HOURS = list(range(start_h, end_h))

    current_sched = schedule_df[["Trip", "StationID", "SchArrive", "SchDepart",
                                  "Seq", "Line", "Direction"]].copy()
    min_seq = current_sched.groupby("Trip")["Seq"].transform("min")
    max_seq = current_sched.groupby("Trip")["Seq"].transform("max")
    current_sched["is_first"] = current_sched["Seq"] == min_seq
    current_sched["is_last"]  = current_sched["Seq"] == max_seq
    current_sched["event_time"] = current_sched.apply(
        lambda r: r["SchDepart"] if r["is_first"] else r["SchArrive"], axis=1)
    current_sched["t"] = current_sched["event_time"].apply(
        lambda x: parse_time_to_min(x, BASE_HOUR))
    current_sched["t_depart"] = current_sched["SchDepart"].apply(
        lambda x: parse_time_to_min(x, BASE_HOUR))
    filtered_sched = current_sched[
        (current_sched["t"] >= START_MIN) & (current_sched["t"] <= END_MIN_SIM)
    ].copy()

    if filtered_sched.empty:
        continue

    train_dest_map = filtered_sched[filtered_sched["is_last"]] \
        .set_index("Trip")["StationID"].to_dict()

    # 사고 노출/취소시각 계산에는 window 경계 바로 밖의 마지막 안전 정차역도
    # 필요하다. 운행·수요 모형은 filtered_sched를 그대로 쓰되, 사고시각 계산용
    # stop sequence만 window에 걸친 열차의 전체 당일 시간표에서 만든다.
    active_trip_ids = set(filtered_sched["Trip"].unique())
    event_sched = current_sched[current_sched["Trip"].isin(active_trip_ids)]
    trip_stops_seq = {}
    for v, grp in event_sched.groupby("Trip"):
        grp_s = grp.sort_values("Seq")
        trip_stops_seq[v] = list(zip(
            grp_s["Seq"].tolist(),
            grp_s["StationID"].tolist(),
            grp_s["t"].tolist(),
            grp_s["t_depart"].tolist(),
        ))

    # OD Filtering
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
    N     = list(valid_stations)
    Vm    = filtered_sched["Trip"].unique().tolist()
    Gamma = list(range(START_MIN, END_MIN_SIM + 1))

    # Graph lookups
    trains_at  = defaultdict(lambda: defaultdict(list))
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
        t_end   = min(END_MIN, hour_offset + STEPS_PER_HOUR - 1)
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

    l_indices = [(o, v, t) for o in Omega for v in Vm
                 if v in train_active_times for t in train_active_times[v]]
    valid_keys = list(I_vts)
    # sigma_s: walking transfer time per station [steps] (inputs.walking_times)
    sigma = network_data.transfer_steps(N, TIME_STEP)
    no_board_keys = [(v, t, s) for (v, t, s) in valid_keys
                     if v in train_dest_map and s == train_dest_map[v]]

    # === 속도 최적화 캐시 (환승 모델링은 그대로 유지) ===
    # trains_at_ts[(t, s)] = [v...] : 실제 정차 열차 목록을 plain dict로
    # (defaultdict가 quicksum 안에서 매번 빈 리스트를 생성하는 비용 제거)
    trains_at_ts = {}
    for (v, t, s) in valid_keys:
        trains_at_ts.setdefault((t, s), []).append(v)
    EMPTY = []
    # 헬퍼: 빈 (t,s)는 빈 리스트 반환, quicksum이 0으로 처리
    def _trains_at(t, s):
        return trains_at_ts.get((t, s), EMPTY)
    print(f"  Cache: trains_at_ts={len(trains_at_ts)} pairs (was full N x Gamma)")

    # === Sparse indexing 프리컴퓨트 (Phase B) ===
    # train_stops_set[v] = set of stops for train v (om_rel_stations 계산용)
    train_stops_set = {v: set(station_of[v].values()) for v in Vm}
    # train_seq[v] = (stations, times) 시간순 정차 시퀀스 (Phase C: 경로 기반 sparse)
    train_seq = {}
    for v in Vm:
        _ts = sorted(station_of[v])
        train_seq[v] = ([station_of[v][t] for t in _ts], _ts)
    _seq_items = list(train_seq.items())

    # === Phase C: 환승 <= MAX_TRANSFERS 여정이 점유 가능한 역만 남기는 rel 계산 ===
    # orig/dest를 지나는 열차의 모든 정차역 합집합을 쓰면 |N|의 ~50%를
    # 담아 qd/dd 변수가 과다 생성된다. 여기서는 라운드별 relaxation으로
    # "탑승 r회 이내 도달 최소시각"(forward)과 "남은 r회로 dest 도달 최대
    # 출발시각"(backward)을 구해, 두 값이 양립하는 역만 남긴다.
    # avg |rel| 52.4 -> 23.0 (~2.3x), precompute는 OD 전체 0.2초 수준.
    _MAX_RIDES = MAX_TRANSFERS + 1
    _INF = float("inf")
    # (orig,dest)는 target/duration이 달라도 동일하므로 윈도우 단위로 캐시
    _rel_cache = {}

    def _rel_stations(orig, dest):
        # forward[r][s] = 탑승 r회 이내로 s 도달 가능한 최소 시각
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
                        # 출발지에서는 환승 대기(sigma) 없이 승차 가능
                        need = tp if (s == orig and tp == START_MIN) else tp + sigma[s]
                        if ts[i] >= need and ts[i] < board:
                            board = ts[i]
                    if board < _INF and ts[i] >= board and ts[i] < cur[s]:
                        cur[s] = ts[i]
        # backward[r][s] = 남은 탑승 r회로 dest 도달 가능한 최대 시각
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
    # valid_by_station[s] = [(v,t,s)...] : OD-specific 제약 인덱싱용
    valid_by_station = defaultdict(list)
    for (v, t, s) in valid_keys:
        valid_by_station[s].append((v, t, s))
    # dwell 정보를 om-invariant tuple 리스트로 정리
    dwell_index = []  # [(v, s_d, times_tuple, t0_or_None)]
    for v in Vm:
        if v not in train_active_times:
            continue
        for dwell in train_dwells[v]:
            s_d = dwell["station"]
            times = tuple(dwell["times"])
            t0 = times[0] - 1
            has_prev = t0 in train_active_times[v]
            dwell_index.append((v, s_d, times, t0 if has_prev else None))
    print(f"  Sparse cache: train_stops_set for {len(train_stops_set)} trains, "
          f"valid_by_station over {len(valid_by_station)} stations, "
          f"dwell_index={len(dwell_index)}")

    # ==========================================================================
    # STAGE 1: LOAD EXISTING NORMAL RESULTS (모델 풀이 스킵)
    # ==========================================================================
    normal_file = f"{DIR_NORMAL}/Normal_{window_name}.csv"
    normal_l_results = {}
    # Sec. 3.2.2: the disrupted CFM fixes the states immediately before onset to
    # their normal values; the normal actions before t0 are fixed below.
    normal_onset = START_MIN + _steps(ONSET_MIN)
    normal_pre_b, normal_pre_a, normal_pre_od = {}, {}, {}
    normal_pre_stations = defaultdict(set)

    if not os.path.exists(normal_file):
        print(f"  > [SKIP WINDOW] Normal CSV not found: {normal_file}")
        continue

    print(f"  > [Step 1] Loading existing Normal results from {normal_file}")
    try:
        norm_df = pd.read_csv(normal_file)
        load_rows = norm_df[norm_df["Variable"] == "Load"]
        for row in load_rows.itertuples(index=False):
            om   = str(row.Omega)
            v_id = str(row.TrainID)
            if v_id.endswith(".0"):
                v_id = v_id[:-2]
            t_val = int(row.Time)
            normal_l_results[(om, v_id, t_val)] = float(row.Value)
        print(f"    -> Loaded {len(normal_l_results)} Load records")
        pre_rows = norm_df[norm_df["Time"].astype(int) < normal_onset]
        for row in pre_rows.itertuples(index=False):
            om = str(row.Omega)
            t_val = int(row.Time)
            station = (None if pd.isna(row.StationID)
                       else str(row.StationID).strip() or None)
            v_id = None if pd.isna(row.TrainID) else str(row.TrainID)
            if v_id is not None and v_id.endswith(".0"):
                v_id = v_id[:-2]
            if row.Variable == "Boarding":
                normal_pre_b[(om, v_id, t_val, station)] = float(row.Value)
            elif row.Variable == "Alighting":
                normal_pre_a[(om, v_id, t_val, station)] = float(row.Value)
            elif row.Variable == "Exit":
                normal_pre_od[(om, t_val)] = float(row.Value)
            if (row.Variable in ("Boarding", "Alighting", "Queue", "Walking")
                    and station is not None):
                normal_pre_stations[om].add(station)
    except Exception as e:
        print(f"  > Error loading Normal CSV: {e}")
        continue

    # Precompute aggregated loads (occupied_capacity 시나리오별 재계산 비용 제거)
    # total_l[(v,t)] = 전체 om 합산 / l_by_om[om] = 이 om의 (v,t,val) 리스트
    total_l = defaultdict(float)
    l_by_om = defaultdict(list)
    for (om, v, t), val in normal_l_results.items():
        total_l[(v, t)] += val
        l_by_om[om].append((v, t, val))
    print(f"    -> Precomputed total_l ({len(total_l)} keys), l_by_om ({len(l_by_om)} OMs)")

    # ==========================================================================
    # STAGE 2: DISRUPTED SCENARIOS (corridor-track 기반 판정)
    # ==========================================================================
    # TEST_MODE 대상은 TOP_STATIONS 랭킹과 무관하게 실행하기 위해 소스 리스트 교체
    target_list = list(_TEST_TARGET_FILTER) if _TEST_TARGET_FILTER is not None else TOP_STATIONS
    for target_station in target_list:
        if target_station not in N:
            print(f"  [SKIP] target {target_station} not in this window's stations")
            continue

        # target별 (노출시각, 안전취소시각, Case) 캐시 — duration 3개에 걸쳐 재사용
        t_detect_start = time.time()
        disruption_times_by_trip = {}
        for v in Vm:
            trip_stops = trip_stops_seq.get(v, [])
            if not trip_stops:
                continue
            disruption_times = compute_disruption_times_for_trip(
                target_station, trip_stops)
            if disruption_times is not None:
                disruption_times_by_trip[v] = disruption_times
        t_detect = time.time() - t_detect_start  # target별 공유 비용

        # Warm start 힌트 저장소 (동일 target 내 duration 간 재사용)
        # 이전 duration의 최적해 X값을 tuple key로 저장 → 다음 model의 PStart에 주입
        prev_solution = None

        for duration in DURATIONS:
            scenario_name = f"{start_h}_{target_station}_{duration}"
            disrupt_file = f"{DIR_DISRUPT}/Disrupt_{scenario_name}.csv"
            existing_version = _scenario_logic_versions.get(scenario_name)
            if (os.path.exists(disrupt_file)
                    and existing_version == SCENARIO_LOGIC_VERSION
                    and _scenario_checkpoints.get(scenario_name)
                    in {"EXTRACTED", "NO_AFFECTED"}):
                # 결과가 이미 있으면 건너뛴다. 통계는 위에서 로드해 뒀으므로
                # 이 시나리오의 저장된 기록은 그대로 보존된다.
                if scenario_name not in _done_scenarios:
                    print(f"    [SKIP] {scenario_name} (결과 있음, 통계 없음)")
                continue
            if os.path.exists(disrupt_file):
                print(f"    [STALE] {scenario_name}: timing logic "
                      f"{existing_version or 'unversioned'} -> "
                      f"{SCENARIO_LOGIC_VERSION}; regenerating")

            # 동일 시나리오의 이전 logic_version/중단 통계 row가 있으면 새 실행 row로 교체한다.
            scenario_stats = [
                row for row in scenario_stats
                if str(row.get("scenario", "")) != scenario_name
            ]

            # 시나리오 통계 row 초기화 — early append 후 in-place 업데이트 (checkpoint flush 용)
            stat = {
                "window": window_name, "target": target_station, "duration": duration,
                "scenario": scenario_name,
                "logic_version": SCENARIO_LOGIC_VERSION,
                "t_detect_shared": t_detect, "t_filter": 0.0,
                "t_precompute_sparse": 0.0, "t_build": 0.0,
                "t_warmstart": 0.0, "t_solve": 0.0, "t_extract": 0.0, "t_total": 0.0,
                "cpu_scenario": 0.0,
                "canceled_trains": 0,
                "case_a_canceled_trains": 0, "case_b_canceled_trains": 0,
                "mean_cancel_lead_min": 0.0, "max_cancel_lead_min": 0.0,
                "omega_m": 0, "omega_total": len(Omega),
                "od_hourly_min_trips": OD_HOURLY_MIN_TRIPS,
                "num_vars": 0, "num_constrs": 0, "num_nz": 0,
                "runtime_gurobi": 0.0, "presolve_time_gurobi": 0.0,
                "iter_count": 0, "bar_iter_count": 0,
                "warmstart_hits": 0, "warmstart_used": False,
                "rss_before_mb": 0.0, "rss_after_solve_mb": 0.0, "rss_delta_mb": 0.0,
                "status": "RUNNING", "obj": float("nan"), "skipped": False, "error": "",
                # travel time (pax-min) — extract 후 채움
                "affected_demand": 0.0, "queue_pax_min": 0.0,
                "load_pax_min": 0.0, "walk_pax_min": 0.0, "total_pax_min": 0.0,
                "checkpoint": "INIT",
            }
            scenario_stats.append(stat)  # early append: stat 참조로 in-place 업데이트

            def _flush_stats():
                pd.DataFrame(scenario_stats).to_csv(STATS_FILE, index=False)

            _flush_stats()  # 중단 대비 즉시 저장

            t_scenario_start = time.time()
            cpu_scenario_start = time.process_time()
            stat["rss_before_mb"] = _rss_mb()

            DISRUPT_START_T = START_MIN + _steps(cfg.get("time.disruption_onset_min"))
            DISRUPT_END_T   = DISRUPT_START_T + _steps(duration)

            # t_cross로 사고 노출을 판정한다. 사고 전에 이미 전역을 출발한
            # 열차는 과거 출발시각으로 되감지 않고 사고 위치 도달시각부터
            # 차단한다. 아직 출발하지 않은 열차만 t_cancel부터 차단한다.
            t_filter_start = time.time()
            theta = defaultdict(int)
            canceled_trains = set()
            canceled_cases = {"A": 0, "B": 0}
            cancel_leads = []
            for v, (t_cross, t_cancel, event_case) in disruption_times_by_trip.items():
                if not (DISRUPT_START_T <= t_cross <= DISRUPT_END_T):
                    continue
                block_from = compute_cancellation_start(
                    t_cross, t_cancel, DISRUPT_START_T)
                canceled_trains.add(v)
                canceled_cases[event_case] += 1
                cancel_leads.append(max(0, int(t_cross) - int(block_from)))
                if v in train_active_times:
                    for t in train_active_times[v]:
                        if t >= block_from:
                            theta[(v, t)] = 1

            # Omega_m
            Omega_m = set()
            for (om, v, t), val in normal_l_results.items():
                if v in canceled_trains and theta[(v, t)] == 1:
                    Omega_m.add(om)
            Omega_m = list(Omega_m)
            stat["t_filter"] = time.time() - t_filter_start
            stat["canceled_trains"] = len(canceled_trains)
            stat["case_a_canceled_trains"] = canceled_cases["A"]
            stat["case_b_canceled_trains"] = canceled_cases["B"]
            stat["mean_cancel_lead_min"] = (
                float(sum(cancel_leads) / len(cancel_leads)) if cancel_leads else 0.0)
            stat["max_cancel_lead_min"] = (
                float(max(cancel_leads)) if cancel_leads else 0.0)
            stat["omega_m"] = len(Omega_m)

            print(f"  > {target_station} | {duration}m | canceled={len(canceled_trains)}, "
                  f"affected ODs: {len(Omega_m)}/{len(Omega)}")
            if not Omega_m:
                pd.DataFrame({"Info": ["No Affected Pax"]}).to_csv(disrupt_file)
                stat["skipped"] = True
                stat["status"] = "NO_AFFECTED"
                stat["checkpoint"] = "NO_AFFECTED"
                stat["t_total"] = time.time() - t_scenario_start
                stat["cpu_scenario"] = time.process_time() - cpu_scenario_start
                stat["rss_after_solve_mb"] = _rss_mb()
                stat["rss_delta_mb"] = stat["rss_after_solve_mb"] - stat["rss_before_mb"]
                _flush_stats()
                continue

            # occupied_capacity = total_l - sum_{om in Omega_m} l_by_om[om]
            # total_l 전체 스캔 대신 Omega_m 분만 차감 (Omega_m << |normal_l_results|)
            sub_l = defaultdict(float)
            for om in Omega_m:
                for (v, t, val) in l_by_om[om]:
                    sub_l[(v, t)] += val

            def _occupied(v, t):
                return total_l.get((v, t), 0.0) - sub_l.get((v, t), 0.0)

            # === Sparse per-scenario 프리컴퓨트 (Phase B) ===
            # om_rel_stations[om] = orig, dest, 그리고 이들을 방문하는 열차의 모든 stops
            # (1-hop 근사: 대부분 실제 경로 커버. 2-hop transfer는 rare)
            t_precompute_start = time.time()
            # Phase C: 환승 <= MAX_TRANSFERS 경로 기반 sparse rel (윈도우 단위 캐시)
            om_rel_stations = {}
            for om in Omega_m:
                orig, dest = Omega_Map[om]
                cached = _rel_cache.get((orig, dest))
                if cached is None:
                    cached = _rel_stations(orig, dest)
                    _rel_cache[(orig, dest)] = cached
                om_rel_stations[om] = cached | normal_pre_stations.get(om, set())
            # 열차별 capacity 상한 프리컴퓨트 (Phase A)
            cap_limit = {}
            for v_id in Vm:
                if v_id in train_active_times:
                    for t in train_active_times[v_id]:
                        spare_cap = Cv - _occupied(v_id, t)
                        limit = spare_cap * (1 - theta[(v_id, t)])
                        cap_limit[(v_id, t)] = max(0.0, limit)
            stat["t_precompute_sparse"] = time.time() - t_precompute_start
            avg_rel = sum(len(v) for v in om_rel_stations.values()) / max(1, len(om_rel_stations))
            print(f"    [Sparse] avg |rel_stations| per om = {avg_rel:.1f} / |N|={len(N)}")

            try:
                t_build_start = time.time()
                m_dis = gp.Model(f"Disrupt_{scenario_name}", env=env)
                m_dis.setParam("MIPGap", cfg.get("scenarios.solver.mip_gap"))
                m_dis.setParam("Presolve", cfg.get("scenarios.solver.presolve"))
                # Method는 env(=2, Barrier)에서 상속 — LP에 최적

                # === Sparse 인덱스 리스트 (Phase B) ===
                qd_keys = [(om, s, t) for om in Omega_m
                           for s in om_rel_stations[om] for t in Gamma]
                dd_keys = [(om, s, t) for om in Omega_m
                           for s in om_rel_stations[om]
                           if s != Omega_Map[om][1] for t in Gamma]
                l_ind_m = [(om, v, t) for om in Omega_m for v in Vm
                           if v in train_active_times for t in train_active_times[v]]

                qd = m_dis.addVars(qd_keys, vtype=GRB.CONTINUOUS, lb=0)
                ld = m_dis.addVars(l_ind_m, vtype=GRB.CONTINUOUS, lb=0)
                ad = m_dis.addVars(Omega_m, valid_keys, vtype=GRB.CONTINUOUS, lb=0)
                bd = m_dis.addVars(Omega_m, valid_keys, vtype=GRB.CONTINUOUS, lb=0)
                od = m_dis.addVars(Omega_m, Gamma, vtype=GRB.CONTINUOUS, lb=0)
                dd = m_dis.addVars(dd_keys, vtype=GRB.CONTINUOUS, lb=0)

                # === 제약 구성: 청킹 유지 (메모리) + Sparse s + 배치 addConstrs (Phase A+B) ===
                chunk_size = int(cfg.get("scenarios.constraint_block_ods"))
                for i in range(0, len(Omega_m), chunk_size):
                    om_chunk = Omega_m[i:i + chunk_size]

                    # qd @ dest: (om, dest, t) — om당 dest 하나
                    m_dis.addConstrs((qd[om, Omega_Map[om][1], t] ==
                        (qd[om, Omega_Map[om][1], t-1] if t > START_MIN else 0)
                        + D.get((om, Omega_Map[om][1], t), 0)
                        + gp.quicksum(ad[om, v, t, Omega_Map[om][1]]
                                      for v in _trains_at(t, Omega_Map[om][1]))
                        - gp.quicksum(bd[om, v, t, Omega_Map[om][1]]
                                      for v in _trains_at(t, Omega_Map[om][1]))
                        - od[om, t]
                        for om in om_chunk for t in Gamma))

                    # qd @ non-dest: sparse s (om_rel_stations만)
                    m_dis.addConstrs((qd[om, s, t] ==
                        (qd[om, s, t-1] if t > START_MIN else 0)
                        + D.get((om, s, t), 0)
                        + gp.quicksum(ad[om, v, t-sg, s]
                                      for v in _trains_at(t-sg, s)
                                      if t >= START_MIN + sg)
                        - gp.quicksum(bd[om, v, t, s] for v in _trains_at(t, s))
                        for om in om_chunk
                        for s in om_rel_stations[om] if s != Omega_Map[om][1]
                        for sg in (sigma[s],)
                        for t in Gamma))

                    # dd @ non-dest: 개별 addConstr → addConstrs 배치 (Phase A 핵심 개선)
                    m_dis.addConstrs((dd[om, s, t] ==
                        (dd[om, s, t-1] if t > START_MIN else 0)
                        + gp.quicksum(ad[om, v, t, s] for v in _trains_at(t, s))
                        - (gp.quicksum(ad[om, v, t-sg, s]
                                       for v in _trains_at(t-sg, s))
                           if t >= START_MIN + sg else 0)
                        for om in om_chunk
                        for s in om_rel_stations[om] if s != Omega_Map[om][1]
                        for sg in (sigma[s],)
                        for t in Gamma))

                    # ld balance
                    m_dis.addConstrs((ld[om, v, t] ==
                        (ld[om, v, t-1] if t > train_active_times[v].start else 0)
                        + (bd[om, v, t, station_of[v][t]] if t in station_of[v] else 0)
                        - (ad[om, v, t, station_of[v][t]] if t in station_of[v] else 0)
                        for om in om_chunk for v in Vm if v in train_active_times
                        for t in train_active_times[v]))

                    if no_board_keys:
                        m_dis.addConstrs((bd[om, v, t, s] == 0
                            for om in om_chunk for (v, t, s) in no_board_keys))

                # capacity: 개별 addConstr → 배치 (Phase A)
                m_dis.addConstrs((gp.quicksum(ld[om, v, t] for om in Omega_m) <= cap_limit[(v, t)]
                    for (v, t) in cap_limit))

                m_dis.addConstrs((gp.quicksum(bd[om, v, t, s] for om in Omega_m) <= Cv
                    for (v, t, s) in valid_keys))
                m_dis.addConstrs((ad[om, v, t, s] <= CumD[(om, t)]
                    for om in Omega_m for (v, t, s) in valid_keys))
                m_dis.addConstrs((bd[om, v, t, s] <= CumD[(om, t)]
                    for om in Omega_m for (v, t, s) in valid_keys))

                # OD-specific 제약: valid_by_station 프리컴퓨트 활용 (Phase A)
                for om in Omega_m:
                    orig, dest = Omega_Map[om]
                    if orig in valid_by_station:
                        m_dis.addConstrs((ad[om, v, t, s] == 0
                                          for (v, t, s) in valid_by_station[orig]))
                    if dest in valid_by_station:
                        m_dis.addConstrs((bd[om, v, t, s] == 0
                                          for (v, t, s) in valid_by_station[dest]))

                # === 보존 법칙 보호 (Phase C 필수) ===
                # ad/bd는 om_rel_stations와 무관하게 valid_keys 전체에 생성되는데
                # qd/dd 보존 제약은 om_rel_stations 안에서만 만들어진다. rel 밖
                # 역에서 하차하면 받아줄 큐 제약이 없어 승객이 증발하고 목적함수가
                # 과소평가된다(Exit 감소로 확인됨). rel 밖 승하차를 명시적으로 금지.
                for om in Omega_m:
                    rel_om = om_rel_stations[om]
                    outside = [(v, t, s) for (v, t, s) in valid_keys
                               if s not in rel_om]
                    if outside:
                        m_dis.addConstrs((ad[om, v, t, s] == 0
                                          for (v, t, s) in outside))
                        m_dis.addConstrs((bd[om, v, t, s] == 0
                                          for (v, t, s) in outside))

                # dwell + terminal load=0: dwell_index 활용해 개별 addConstr → 배치 (Phase A)
                m_dis.addConstrs((
                    gp.quicksum(ad[om, v, t, s_d] for t in times) <=
                    (ld[om, v, t0] if t0 is not None else 0)
                    for om in Omega_m
                    for (v, s_d, times, t0) in dwell_index))
                # 종점 load = 0 (om × 종착 열차)
                dest_trains = [v for v in Vm
                               if v in train_dest_map and v in train_active_times]
                m_dis.addConstrs((ld[om, v, train_active_times[v].stop - 1] == 0
                                  for om in Omega_m for v in dest_trains))

                # Sec. 3.2.2: actions before onset fixed to the normal trajectory,
                # so the states at t0 - 1 equal their normal values.
                pre_keys = [(v, t, s) for (v, t, s) in valid_keys
                            if t < DISRUPT_START_T]
                pre_minutes = [t for t in Gamma if t < DISRUPT_START_T]
                for om in Omega_m:
                    for (v, t, s) in pre_keys:
                        value = normal_pre_b.get((om, v, t, s), 0.0)
                        bd[om, v, t, s].LB = bd[om, v, t, s].UB = value
                        value = normal_pre_a.get((om, v, t, s), 0.0)
                        ad[om, v, t, s].LB = ad[om, v, t, s].UB = value
                    for t in pre_minutes:
                        value = normal_pre_od.get((om, t), 0.0)
                        od[om, t].LB = od[om, t].UB = value
                omega_m_set = set(Omega_m)
                valid_set = set(valid_keys)
                for (om, v, t, s), value in (*normal_pre_b.items(),
                                             *normal_pre_a.items()):
                    if (om in omega_m_set and t < DISRUPT_START_T and value > 0
                            and (v, t, s) not in valid_set):
                        raise ValueError(
                            f"normal pre-onset action {(om, v, t, s)} outside valid keys")

                # Eq. (14) [eq:obj_d] with Eq. (13) [eq:obj_db], c = D:
                # Z = 1/(T - t0) sum_{omega} sum_{t >= t0} z_t, and Eq. (2)
                # z_t = kappa [queue + walking] + onboard load.
                obj_d = (
                    KAPPA * (gp.quicksum(var for (om, s, t), var in qd.items()
                                         if t >= DISRUPT_START_T)
                             + gp.quicksum(var for (om, s, t), var in dd.items()
                                           if t >= DISRUPT_START_T))
                    + gp.quicksum(var for (om, v, t), var in ld.items()
                                  if t >= DISRUPT_START_T))
                m_dis.setObjective(
                    obj_d / (_steps(HORIZON_MIN) - _steps(ONSET_MIN)),
                    GRB.MINIMIZE)

                # 빌드 완료 — Gurobi 카운터 확정 (update 호출로 pending 변수/제약 반영)
                m_dis.update()
                stat["num_vars"] = m_dis.NumVars
                stat["num_constrs"] = m_dis.NumConstrs
                stat["num_nz"] = m_dis.NumNZs
                stat["t_build"] = time.time() - t_build_start
                stat["checkpoint"] = "BUILT"
                _flush_stats()  # 빌드 완료 시점 저장 (solve 중 크래시 대비)

                # Warm start: 이전 duration 해를 tuple key로 매칭 → PStart 주입
                if prev_solution is not None:
                    t_warm_start = time.time()
                    hits = 0
                    for key, val in prev_solution["l"].items():
                        if key in ld:
                            ld[key].PStart = val
                            hits += 1
                    for key, val in prev_solution["a"].items():
                        if key in ad:
                            ad[key].PStart = val
                            hits += 1
                    for key, val in prev_solution["b"].items():
                        if key in bd:
                            bd[key].PStart = val
                            hits += 1
                    for key, val in prev_solution["q"].items():
                        if key in qd:
                            qd[key].PStart = val
                            hits += 1
                    for key, val in prev_solution["d"].items():
                        if key in dd:
                            dd[key].PStart = val
                            hits += 1
                    for key, val in prev_solution["o"].items():
                        if key in od:
                            od[key].PStart = val
                            hits += 1
                    stat["warmstart_hits"] = hits
                    stat["warmstart_used"] = hits > 0
                    stat["t_warmstart"] = time.time() - t_warm_start

                t_solve_start = time.time()
                m_dis.optimize()
                stat["t_solve"] = time.time() - t_solve_start
                stat["rss_after_solve_mb"] = _rss_mb()
                stat["rss_delta_mb"] = stat["rss_after_solve_mb"] - stat["rss_before_mb"]

                # Gurobi 자체 측정치
                try:
                    stat["runtime_gurobi"] = m_dis.Runtime
                    stat["presolve_time_gurobi"] = m_dis.PresolveTime
                    stat["iter_count"] = int(m_dis.IterCount)
                    stat["bar_iter_count"] = int(m_dis.BarIterCount)
                except Exception:
                    pass

                stat["status"] = {
                    GRB.OPTIMAL: "OPTIMAL", GRB.INFEASIBLE: "INFEASIBLE",
                    GRB.UNBOUNDED: "UNBOUNDED", GRB.TIME_LIMIT: "TIME_LIMIT",
                    GRB.SUBOPTIMAL: "SUBOPTIMAL",
                }.get(m_dis.status, f"STATUS_{m_dis.status}")
                try:
                    stat["obj"] = m_dis.ObjVal
                except Exception:
                    pass
                stat["checkpoint"] = "SOLVED"
                _flush_stats()  # solve 완료 시점 저장

                if m_dis.status == GRB.OPTIMAL:
                    t_extract_start = time.time()
                    dis_results = []
                    # travel time 집계 accumulators (pax-min)
                    tt_q = 0.0
                    tt_l = 0.0
                    tt_d = 0.0
                    for om in Omega_m:
                        orig, dest = Omega_Map[om]
                        # sparse: om_rel_stations[om]만 순회
                        for s in om_rel_stations[om]:
                            for t in Gamma:
                                qv = qd[om, s, t].X
                                if qv > REPORT_TOL:
                                    dis_results.append(("Queue", om, orig, dest, t, s,
                                                        None, qv))
                                    tt_q += qv
                                if s != dest:
                                    dv = dd[om, s, t].X
                                    if dv > REPORT_TOL:
                                        dis_results.append(("Walking", om, orig, dest, t, s,
                                                            None, dv))
                                        tt_d += dv
                        for v in Vm:
                            if v in train_active_times:
                                for t in train_active_times[v]:
                                    lv = ld[om, v, t].X
                                    if lv > REPORT_TOL:
                                        dis_results.append(("Load", om, orig, dest, t,
                                                            None, v, lv))
                                        tt_l += lv
                        for (v, t, s) in valid_keys:
                            if bd[om, v, t, s].X > REPORT_TOL:
                                dis_results.append(("Boarding", om, orig, dest, t, s,
                                                    v, bd[om, v, t, s].X))
                            if ad[om, v, t, s].X > REPORT_TOL:
                                dis_results.append(("Alighting", om, orig, dest, t, s,
                                                    v, ad[om, v, t, s].X))
                        for t in Gamma:
                            if od[om, t].X > REPORT_TOL:
                                dis_results.append(("Exit", om, orig, dest, t, dest,
                                                    None, od[om, t].X))

                    # affected_demand: Omega_m OD의 총 demand
                    aff_dem = 0.0
                    for om in Omega_m:
                        orig = Omega_Map[om][0]
                        for t in Gamma:
                            aff_dem += D.get((om, orig, t), 0.0)
                    stat["affected_demand"] = aff_dem
                    stat["queue_pax_min"] = tt_q
                    stat["load_pax_min"] = tt_l
                    stat["walk_pax_min"] = tt_d
                    stat["total_pax_min"] = tt_q + tt_l + tt_d

                    pd.DataFrame(dis_results, columns=["Variable", "Omega", "Origin",
                        "Dest", "Time", "StationID", "TrainID", "Value"]
                        ).to_csv(disrupt_file, index=False)
                    stat["t_extract"] = time.time() - t_extract_start

                    # 다음 duration용 warm start snapshot (nonzero만 — memory 절감)
                    snap_l, snap_a, snap_b = {}, {}, {}
                    snap_q, snap_d, snap_o = {}, {}, {}
                    for om in Omega_m:
                        dest_om = Omega_Map[om][1]
                        # sparse: om_rel_stations[om]만
                        for s in om_rel_stations[om]:
                            for t in Gamma:
                                v_q = qd[om, s, t].X
                                if v_q > WARM_TOL:
                                    snap_q[(om, s, t)] = v_q
                                if s != dest_om:
                                    v_d = dd[om, s, t].X
                                    if v_d > WARM_TOL:
                                        snap_d[(om, s, t)] = v_d
                        for t in Gamma:
                            v_o = od[om, t].X
                            if v_o > WARM_TOL:
                                snap_o[(om, t)] = v_o
                        for v in Vm:
                            if v in train_active_times:
                                for t in train_active_times[v]:
                                    v_l = ld[om, v, t].X
                                    if v_l > WARM_TOL:
                                        snap_l[(om, v, t)] = v_l
                        for (v, t, s) in valid_keys:
                            v_a = ad[om, v, t, s].X
                            if v_a > WARM_TOL:
                                snap_a[(om, v, t, s)] = v_a
                            v_b = bd[om, v, t, s].X
                            if v_b > WARM_TOL:
                                snap_b[(om, v, t, s)] = v_b
                    prev_solution = {"l": snap_l, "a": snap_a, "b": snap_b,
                                     "q": snap_q, "d": snap_d, "o": snap_o}

                    stat["checkpoint"] = "EXTRACTED"
                    _flush_stats()

                    print(f"    -> Saved {disrupt_file} "
                          f"(vars={stat['num_vars']:,} nz={stat['num_nz']:,} "
                          f"build={stat['t_build']:.1f}s solve={stat['t_solve']:.1f}s "
                          f"iter={stat['iter_count']} bar={stat['bar_iter_count']} "
                          f"rss={stat['rss_after_solve_mb']:.0f}MB "
                          f"warm={stat['warmstart_hits']} "
                          f"pax_min={stat['total_pax_min']:.0f} "
                          f"dem={stat['affected_demand']:.0f})")
            except Exception as e:
                print(f"    -> Error in Disrupted Model: {e}")
                stat["error"] = str(e)[:200]
            finally:
                if "m_dis" in locals():
                    m_dis.dispose()
                stat["t_total"] = time.time() - t_scenario_start
                stat["cpu_scenario"] = time.process_time() - cpu_scenario_start
                if stat["checkpoint"] not in ("EXTRACTED", "NO_AFFECTED"):
                    stat["checkpoint"] = "TERMINATED"  # 예외/미완료 표시
                # stat은 이미 early append됨 → 최종 flush만
                _flush_stats()

env.dispose()

# 시나리오 통계 최종 저장 + 요약
if scenario_stats:
    stats_df = pd.DataFrame(scenario_stats)
    stats_df.to_csv(STATS_FILE, index=False)
    print(f"\nScenario stats saved: {STATS_FILE}  ({len(stats_df)} rows)")
    solved = stats_df[stats_df["status"] == "OPTIMAL"]
    if not solved.empty:
        print(f"  Solved scenarios: {len(solved)}")
        print(f"  build time  : mean={solved['t_build'].mean():.2f}s  "
              f"max={solved['t_build'].max():.2f}s  total={solved['t_build'].sum():.1f}s")
        print(f"  solve time  : mean={solved['t_solve'].mean():.2f}s  "
              f"max={solved['t_solve'].max():.2f}s  total={solved['t_solve'].sum():.1f}s")
        print(f"  gurobi rt   : mean={solved['runtime_gurobi'].mean():.2f}s  "
              f"presolve mean={solved['presolve_time_gurobi'].mean():.2f}s")
        print(f"  iterations  : simplex mean={solved['iter_count'].mean():.0f}  "
              f"barrier mean={solved['bar_iter_count'].mean():.0f}")
        print(f"  vars        : mean={solved['num_vars'].mean():.0f}  "
              f"max={solved['num_vars'].max():,}")
        print(f"  constrs     : mean={solved['num_constrs'].mean():.0f}  "
              f"max={solved['num_constrs'].max():,}")
        print(f"  nonzeros    : mean={solved['num_nz'].mean():.0f}  "
              f"max={solved['num_nz'].max():,}")
        print(f"  peak RSS    : mean={solved['rss_after_solve_mb'].mean():.0f}MB  "
              f"max={solved['rss_after_solve_mb'].max():.0f}MB")
        warm = solved[solved["warmstart_used"] == True]
        if not warm.empty:
            print(f"  warm-start  : used in {len(warm)}/{len(solved)}  "
                  f"avg hits={warm['warmstart_hits'].mean():.0f}")
    skipped = stats_df[stats_df["skipped"] == True]
    if not skipped.empty:
        print(f"  No-affected (skipped LP): {len(skipped)}")

print(f"\nAll Tasks Completed. Total Time: {time.time() - total_start_time:.2f}s")
