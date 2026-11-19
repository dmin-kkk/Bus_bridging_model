"""
milp_core.py — Two-Stage Stochastic MILP for Bus Bridging
           열차 + 버스 공존 세계관 (파일 4 기반)
           ★ 대표 시나리오(reduced)만 Stage 2 실행 버전 ★

파이프라인:
    1. 파일 1 결과 로딩 (Normal_*.csv, Disrupt_*.csv)
    2. 시나리오 축소 (duration별 K-means clustering)
    3. 데이터 준비 (Omega_m, D, spare_capacity, theta)
    4. Route 선택 & Bus Trip 생성
    5. Stage 2 cache 빌드 (N_ω 기반 q/w 인덱스 축소)
    6. Two-Stage Stochastic MILP (대표 시나리오로 depot x[d] 결정)
    7. depot 고정 후 대표 시나리오(reduced)만 Stage 2 재실행 & 결과 출력
"""

import gc
import math
import os
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA  # noqa: F401  (used by legacy reducers only)

from stage2_adaptive import info_cells

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    raise RuntimeError("Gurobi is required.")


# =============================================================================
# 0. Configuration
# =============================================================================
# Every default below comes from the active configuration profile
# (config/profiles/<profile>/*.toml, read through bbconfig).  Callers
# (06_stage2_core.py and the evaluation scripts) may still replace the module
# attributes before loading data.
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from bbconfig import cfg  # noqa: E402

_LEGACY = cfg.get("stage2.core_legacy")
_CANDIDATES = cfg.get("stage2.candidates")

# --- File paths ---
PIPELINE_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = cfg.results_dir
FILE_PREDEFINED_ROUTES  = str(
    RESULT_ROOT / cfg.get("paths.routes") / cfg.get("routes.library_file"))
# The paper pipeline uses the predefined Stage-3 route library only.
ROUTE_SOURCE            = os.getenv(
    "TSB_ROUTE_SOURCE", cfg.get("routes.source_mode")).strip().lower()
# 버스 승차 위치 가정:
#   general — 승객이 실제로 대기 중인 역(Disrupt CSV 의 StationID)에서 태울 수 있다.
#             해당 역에 큐가 있고 목적지가 같은 모든 OD 를 승차 대상으로 삼는다.
#   origin  — (구버전) OD 의 원출발역에서만 태울 수 있다. 열차 쪽과 비대칭.
BUS_BOARD_MODE          = os.getenv(
    "TSB_BUS_BOARD", cfg.get("routes.bus_boarding_mode")).strip().lower()
FILE_DISTANCE           = str(cfg.path("inputs.road_times"))
FILE_SCHEDULE           = str(cfg.path("inputs.timetable"))
FILE_OD                 = str(cfg.path("inputs.od_matrix"))
FILE_EXCEL              = str(cfg.path("inputs.stations"))
# 0normal_generation_corridor.py / 1scenario_generation_corridor.py 산출물.
# 두 폴더는 반드시 같은 가정(240 지평, MAX_TRANSFERS)으로 생성된 짝이어야 한다.
DIR_NORMAL              = str(RESULT_ROOT / cfg.get("paths.normal"))
DIR_DISRUPT             = str(RESULT_ROOT / cfg.get("paths.disruptions"))
SCENARIO_LOGIC_VERSION  = cfg.get("scenarios.logic_version")

# --- Time parameters ---
TIME_STEP          = 1                                   # one-minute grid Gamma (unit, Sec. 3.1)
WINDOW_END_MIN     = int(cfg.get("time.horizon_min"))   # Table 3: planning horizon T
WINDOW_LENGTH_H    = WINDOW_END_MIN // 60                # T in hours (unit conversion)
_env_start_hours = os.getenv("TSB_START_HOURS", "").strip()
if _env_start_hours:
    START_HOURS    = [int(x) for x in _env_start_hours.split(",") if x.strip()]
else:
    START_HOURS    = [int(h) for h in cfg.get("time.windows")]   # Sec. 4.2: planning windows
DISPATCH_STEP      = int(cfg.get("time.dispatch_interval_min"))  # Table 3: eta
DISRUPT_OFFSET     = int(cfg.get("time.disruption_onset_min"))   # Table 3: t0 (disruption start = START_MIN + t0)
# Eq. (22) [eq:start_time_grid]: the first candidate dispatch is t0 + eta.
DISPATCH_START_MIN = DISRUPT_OFFSET + DISPATCH_STEP
DISRUPT_BUFFER     = int(cfg.get("time.clearance_buffer_min"))   # Table 3: clearance buffer delta
# code (not in paper): padding of rail vehicle states around a train's timetable events.
TRAIN_ACTIVE_PADDING_MIN = int(cfg.get("stage2.train_active_padding_min"))

# --- Vehicle parameters ---
BUS_CAPACITY   = cfg.get("operations.bus_capacity")    # Table 3: C_v, v in V_rho
TRAIN_CAPACITY = cfg.get("operations.rail_capacity")   # Table 3: C_v, v in V_r
MAX_TOTAL_BUSES = int(os.getenv("TSB_MAX_BUSES", cfg.get("operations.fleet_size")))  # Table 3: B

# --- Scenario reduction ---
# Same F2 (log1p -> z-score) -> K-means pipeline as k_clustering.py.
# No PCA, no row-normalization. K is the global cluster count; representatives
# are then picked per duration x cluster, so K=10 gives up to 3*10=30 reps/hour.
K_PER_DURATION = int(os.getenv("TSB_K_PER_DURATION", _LEGACY["k_per_duration"]))
# duration 별 K 결정 (Reviewer #5 Q9 대응). 두 축으로 나뉜다.
#
# K_MODE — 배분 방식
#   fixed    : 30/60/90 모두 K_PER_DURATION. 기존 동작.
#   adaptive : duration 마다 K_CRITERION 이 고른 K 를 그대로 쓴다. 총 대표 수가 변하고
#              따라서 Stage 1 규모도 변한다 — 성능 비교시 이 점을 반드시 밝혀야 한다.
#   budget   : K_CRITERION 이 고른 K 의 비율대로 '총 대표 수를 고정한 채' 배분만 바꾼다.
#              계산예산이 fixed 와 같으므로 배분 효과만 분리해 볼 수 있다.
#
# K_CRITERION — duration 하나에서 K 를 고르는 규칙
#   abs       : silhouette 가 K_SIL_ABS 를 처음 넘는 최소 K. (기본)
#               0.5 는 Kaufman & Rousseeuw 의 통상 해석에서 "reasonable structure"
#               경계라 논문에서 인용 가능한 절대 기준이다.
#               ★ 어떤 K 에서도 못 넘으면 argmax 로 폴백한다 — 실제로 장시간(90분)
#                 장애는 임팩트가 넓게 번져 0.5 에 못 미치는 윈도우가 있다.
#   threshold : silhouette 가 최댓값의 K_SIL_TAU 배 이상인 K 중 '가장 큰' K.
#               평탄구간에서 더 촘촘한 분할을 택해 커버리지를 높인다.
#   argmax    : silhouette 최댓값 K. 평탄구간에서 불안정.
#   elbow     : inertia 곡선의 최대 곡률점(kneedle). silhouette 를 안 쓰는 대안.
# 군집화 방법 (Reviewer #3 Q3 대응 — k-medoids / 대안 거리 검토 요구)
#   kmeans   : 현행. 대표는 '중심에 가장 가까운 시나리오'로 사후 근사한다.
#   kmedoids : 대표(medoid)가 정의상 실제 시나리오다. 목적함수 Σ d(x_i, medoid(i)) 가
#              곧 시나리오 축소오차이므로, 원 특징공간 거리로 돌리면 축소오차를
#              직접 최소화한다 (Dupacova/Heitsch-Romisch 계열과 같은 형태).
# 군집화 공간
#   z    : log1p -> z-score (현행)
#   orig : log1p 만. 역별 임팩트의 원 스케일을 유지 — 축소오차 기준이 이 공간이다.
# 거리
#   euclidean / cosine (cosine 은 임팩트의 공간 패턴만 보고 총 규모는 무시)
# 시나리오 공간확률 p(역) — Reviewer #3 (2) 대응
#   uniform   : 모든 역이 등확률 (현행). 대표 확률 = 군집크기 / duration내 시나리오 수
#   frequency : 사고 기록의 역별 발생 빈도 (Laplace 평활). 가장 직접적인 추정
#   exposure  : 그 윈도우에 열차가 그 역을 지나는 횟수 (passenger-exposure weighting)
# 세 설정을 바꿔가며 돌리면 심사자가 요구한 민감도 비교가 그대로 나온다.
SCENARIO_PRIOR = os.getenv("TSB_SCENARIO_PRIOR", _LEGACY["scenario_prior"]).strip().lower()
PRIOR_ALPHA    = float(os.getenv("TSB_PRIOR_ALPHA", _LEGACY["prior_alpha"]))   # Laplace 평활 강도
FILE_INCIDENT  = os.getenv("TSB_INCIDENT_FILE", str(cfg.path("inputs.incidents_raw")))

CLUSTER_METHOD = os.getenv("TSB_CLUSTER_METHOD", _LEGACY["cluster_method"]).strip().lower()
CLUSTER_SPACE  = os.getenv("TSB_CLUSTER_SPACE", _LEGACY["cluster_space"]).strip().lower()
CLUSTER_METRIC = os.getenv("TSB_CLUSTER_METRIC", _LEGACY["cluster_metric"]).strip().lower()

K_MODE      = os.getenv("TSB_K_MODE", _LEGACY["k_mode"]).strip().lower()
K_CRITERION = os.getenv("TSB_K_CRITERION", _LEGACY["k_criterion"]).strip().lower()
K_SIL_ABS   = float(os.getenv("TSB_K_SIL_ABS", _LEGACY["k_sil_abs"]))
K_SIL_TAU   = float(os.getenv("TSB_K_SIL_TAU", _LEGACY["k_sil_tau"]))
K_MIN    = int(os.getenv("TSB_K_MIN", _LEGACY["k_min"]))
K_MAX    = int(os.getenv("TSB_K_MAX", _LEGACY["k_max"]))
K_BUDGET = int(os.getenv("TSB_K_BUDGET", _LEGACY["k_budget"]))   # 0 → len(DURATIONS) * K_PER_DURATION
# 축소 단위 (Reviewer #1 (2), #3 (3), #5 (9))
#   duration : duration 별 독립 K-means. 현행. 대표 30개가 서로 다른 21~26개 역에
#              흩어져, 한 역에 세 지속시간이 모두 붙는 경우가 윈도우당 0~2개다.
#              그래서 Stage 1 축소집합 위에서는 "역은 실현, 지속시간은 미실현"
#              정보집합을 만들 수 없다.
#   station  : 역을 공동 선택하고 그 역의 30/60/90 을 통째로 블록화한다.
#              duration 축은 유지하되 비용을 P(D) 로 결합한다.
REDUCTION  = os.getenv("TSB_REDUCTION", _LEGACY["reduction"]).strip().lower()
K_STATIONS = int(os.getenv("TSB_K_STATIONS", _LEGACY["k_stations"]))
# Sec. 3.1 / Table 3: duration states P and the Stage 1 prior pi_tau (callers override
# DURATION_WEIGHT with the Stage 2 probabilities pi_hat_tau, Eq. (41)).
DURATIONS      = [int(d) for d in cfg.get("time.durations_min")]
_PRIOR_COUNTS = {int(k): int(v) for k, v in cfg.get("probability.duration_counts").items()}
DURATION_WEIGHT = {d: _PRIOR_COUNTS[d] / sum(_PRIOR_COUNTS.values()) for d in DURATIONS}

# --- Route/trip filtering ---
# 후보 (노선, 차고지) 를 거르는 두 규칙. 버스측 재차변수 l 이 모델의 54.5% 이고
# trip 수에 선형이라 여기가 최대 레버다 (DESIGN §3.6).
#
# DEPOT_ACCESS_BOUND_MIN — Eq. (21) [eq:deadhead_bound] mu_max (Table 3: 120 min).
#   차고지에서 노선 기점까지 허용 회송시간 (operations.depot_access_bound_min).
#   운영 규칙이라 노선에 무관하게 일관되고 Reviewer #5 (4) 의 "장거리 버스가
#   현실적인가" 에 직접 답한다. 배차는 t=20 부터이므로 deadhead d 인 버스는
#   t=20+d 에야 태우기 시작한다. 실측 분포는 중앙 47 / p75 62 / p90 77 / 최대 90 분.
#   ★ 상한이 낮으면 차고지가 하나도 없는 노선이 생겨 커버가 사라진다.
#     값은 {90, 60, 45, 30} 감도로 정한다 (Reviewer #3 (4)).
#
# MAX_DEPOTS_PER_ROUTE — 노선당 차고지 개수 상한 (가까운 순). 0 = 무제한.
#   노선마다 거리 분포가 달라 자의적이다. 시간 상한을 우선 쓰고 이건 보조.
#   window 6 실측(6블록 표본): 무제한 22.9M -> N=4 61% -> N=3 56% -> N=2 51%
DEPOT_ACCESS_BOUND_MIN     = float(os.getenv(
    "TSB_DEPOT_ACCESS_BOUND", cfg.get("operations.depot_access_bound_min")))   # Eq. (21) deadhead bound (see operations.toml)
MAX_DEPOTS_PER_ROUTE = int(os.getenv(
    "TSB_MAX_DEPOTS_PER_ROUTE", _CANDIDATES["max_depots_per_route"]))

# --- Solver ---
MIP_GAP    = float(cfg.get("stage2.solver.mip_gap"))
TIME_LIMIT = float(_LEGACY["time_limit_s"])     # legacy default; Stage 2 callers set it from stage2.solver
QUEUE_EPS = 1e-6                                # float tolerance (comparison only)
MIN_TRIP_BOARDING = cfg.get("operations.min_trip_boarding")   # Table 3: epsilon

# --- OD filtering ---
OD_HOURLY_MIN_TRIPS = float(os.getenv(
    "PAPER_OD_HOURLY_MIN_TRIPS", cfg.get("operations.od_hourly_min_trips")))

# --- Output ---
# TSB_OUTPUT_SUFFIX 가 지정되면 결과 폴더명에 suffix 추가 (sensitivity run 분리용).
_OUT_SUFFIX = os.getenv("TSB_OUTPUT_SUFFIX", "").strip()
OUTPUT_ROOT       = str(RESULT_ROOT / cfg.get("paths.stage2_proposed") / "_core_unused")
DIR_OUT_ALL       = os.path.join(OUTPUT_ROOT, "All")
DIR_OUT_ALLOC     = os.path.join(OUTPUT_ROOT, "Allocation")
DIR_OUT_LOG       = os.path.join(OUTPUT_ROOT, "Logs")
DIR_OUT_TIMETABLE = os.path.join(OUTPUT_ROOT, "Generated_Timetables")
DIR_OUT_DETAIL    = os.path.join(OUTPUT_ROOT, "Decision_Variables")
DIR_OUT_PERF      = os.path.join(OUTPUT_ROOT, "Performance")

# Direct Stage-2 imports this module as a library.  Do not create unrelated
# legacy output directories merely by importing it.
if __name__ == "__main__":
    for _d in [OUTPUT_ROOT, DIR_OUT_ALL, DIR_OUT_ALLOC, DIR_OUT_LOG,
               DIR_OUT_TIMETABLE, DIR_OUT_DETAIL, DIR_OUT_PERF]:
        os.makedirs(_d, exist_ok=True)

VERBOSE = os.getenv("TSB_VERBOSE", "1").strip().lower() not in {"", "0", "false", "no"}

def _log(msg):
    if VERBOSE:
        print(msg, flush=True)

def _status_name(code):
    return {
        GRB.OPTIMAL: "OPTIMAL", GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.TIME_LIMIT: "TIME_LIMIT", GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
    }.get(code, str(code))


def _safe_attr(model, name, default=None):
    try:
        return getattr(model, name)
    except Exception:
        return default


def collect_gurobi_stats(model, solve_sec):
    """Gurobi 모델에서 보고용 통계를 추출."""
    has_sol = _safe_attr(model, "SolCount", 0) > 0
    return {
        "NumVars":     int(_safe_attr(model, "NumVars", 0)),
        "NumIntVars":  int(_safe_attr(model, "NumIntVars", 0)),
        "NumBinVars":  int(_safe_attr(model, "NumBinVars", 0)),
        "NumConstrs":  int(_safe_attr(model, "NumConstrs", 0)),
        "Runtime":     float(_safe_attr(model, "Runtime", solve_sec)),
        "WallSec":     float(solve_sec),
        "Status":      _status_name(_safe_attr(model, "Status", -1)),
        "SolCount":    int(_safe_attr(model, "SolCount", 0)),
        "ObjVal":      float(_safe_attr(model, "ObjVal", float("nan"))) if has_sol else float("nan"),
        "ObjBound":    float(_safe_attr(model, "ObjBound", float("nan"))),
        "MIPGap":      float(_safe_attr(model, "MIPGap", float("nan"))) if has_sol else float("nan"),
        "NodeCount":   int(_safe_attr(model, "NodeCount", 0)),
    }


# =============================================================================
# 1. Static Data Loading
# =============================================================================
def load_schedule(path):
    df = pd.read_csv(path)
    df["Trip"] = df["Trip"].astype(str)
    return df

def load_od(path):
    return pd.read_csv(path)

def load_distance_matrix(path):
    tt = {}
    if not os.path.exists(path):
        print(f"[WARN] Distance matrix not found: {path}")
        return tt
    df = pd.read_csv(path)
    df.columns = df.columns.str.lstrip("\ufeff")
    for row in df.itertuples(index=False):
        try:
            o, d = str(row.Origin_Code), str(row.Destination_Code)
            dur = int(round(float(row.Duration_min)))
        except Exception:
            continue
        if dur >= 0:
            if tt.get((o, d)) is None or dur < tt[(o, d)]:
                tt[(o, d)] = dur
    return tt

def get_travel_time(tt, o, d):
    if o == d:
        return 0
    return tt.get((o, d)) or tt.get((d, o))

def _load_directed_route_times(path):
    """Load unrounded, directed road times used only for route timing.

    ``load_distance_matrix`` intentionally rounds individual OD travel times
    for depot/deadhead decisions.  Route construction follows a different
    convention: raw directed segment times are accumulated first and each
    cumulative arrival is rounded afterwards.  Keeping this small loader
    separate avoids changing any non-route coefficient in the MILP.
    """
    if path is None or not os.path.isfile(path):
        raise FileNotFoundError(f"Directed route-time matrix not found: {path}")
    frame = pd.read_csv(path)
    frame.columns = frame.columns.str.lstrip("\ufeff")
    required = {"Origin_Code", "Destination_Code", "Duration_min"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"Directed route-time matrix is missing columns: {sorted(missing)}")
    times = {}
    for row in frame.itertuples(index=False):
        try:
            origin = str(row.Origin_Code).strip()
            destination = str(row.Destination_Code).strip()
            duration = float(row.Duration_min)
        except (TypeError, ValueError):
            continue
        if not origin or not destination or not math.isfinite(duration) or duration < 0:
            continue
        key = (origin, destination)
        if key not in times or duration < times[key]:
            times[key] = duration
    if not times:
        raise ValueError(f"No usable directed route times in {path}")
    return times


def _directed_route_timing(stops, directed_times, route_name):
    """Return cumulative-rounded model times for one directed stop sequence."""
    raw_cumulative = [0.0]
    for origin, destination in zip(stops, stops[1:]):
        key = (str(origin), str(destination))
        if key not in directed_times:
            raise ValueError(
                f"{route_name}: missing directed road time {origin}->{destination}")
        raw_cumulative.append(raw_cumulative[-1] + directed_times[key])
    cumulative = [int(round(value)) for value in raw_cumulative]
    if any(right <= left for left, right in zip(cumulative, cumulative[1:])):
        raise ValueError(
            f"{route_name}: cumulative rounding creates a nonpositive route leg: "
            f"{cumulative}")
    segments = [right - left for left, right in zip(cumulative, cumulative[1:])]
    return cumulative, segments


def load_route_library(path, *, distance_path=None,
                       validate_forward_times=False):
    """Load routes and, when requested, construct truly directed reversals.

    Passing ``distance_path`` activates the production timing convention.  A
    reverse route is recomputed from the reverse-direction entries in
    ``Station_Distance_Matrix`` and is cumulative-rounded; it is never made by
    merely reversing the forward integer segments.  With
    ``validate_forward_times=True`` the CSV's forward cumulative times must
    already follow the same convention.  The no-keyword legacy behavior is
    retained for historical scripts, while the Stage 1/2 production wrappers
    opt into the validated path explicitly.
    """
    library = {}
    seen_seqs = set()
    if not os.path.exists(path):
        print(f"[WARN] Route file not found: {path}")
        return library
    directed_times = (
        None if distance_path is None
        else _load_directed_route_times(distance_path)
    )
    if validate_forward_times and directed_times is None:
        raise ValueError(
            "validate_forward_times=True requires a directed distance_path")

    df = pd.read_csv(path)
    df.columns = df.columns.str.lstrip("\ufeff")
    for col in ["Route", "BaseLine", "Type", "StationCode"]:
        df[col] = df[col].astype(str)
    df["StopSeq"]    = pd.to_numeric(df["StopSeq"],    errors="coerce")
    df["CumTime_min"]= pd.to_numeric(df["CumTime_min"],errors="coerce")
    df = df.dropna(subset=["StopSeq", "CumTime_min"])

    for route_name, grp in df.groupby("Route", sort=False):
        g = grp.sort_values("StopSeq").reset_index(drop=True)
        stops, cum, seg_times = [], [], []
        ok, prev_cum = True, None
        for row in g.itertuples(index=False):
            station = str(row.StationCode)
            c = int(round(float(row.CumTime_min)))
            if not station or (prev_cum is not None and c <= prev_cum):
                ok = False; break
            stops.append(station)
            cum.append(c)
            if prev_cum is not None:
                seg_times.append(c - prev_cum)
            prev_cum = c
        if not ok or len(stops) < 2 or any(s <= 0 for s in seg_times):
            continue
        if cum[0] != 0:
            offset = cum[0]; cum = [c - offset for c in cum]

        if directed_times is not None:
            matrix_cum, _matrix_segments = _directed_route_timing(
                stops, directed_times, str(route_name))
            if validate_forward_times and matrix_cum != cum:
                raise ValueError(
                    f"{route_name}: route CSV does not match directed "
                    "cumulative-rounded Station_Distance_Matrix timing: "
                    f"file={cum}, matrix={matrix_cum}")

        entry = {
            "route": str(route_name), "route_type": str(g.iloc[0]["Type"]).upper(),
            "direction": f"{stops[0]}->{stops[-1]}", "base_line": str(g.iloc[0]["BaseLine"]),
            "stops": stops, "cum": cum, "seg_durations": seg_times,
        }
        library[str(route_name)] = entry
        seen_seqs.add(tuple(stops))

        # Auto-generate reverse
        rev_stops = list(reversed(stops))
        if tuple(rev_stops) not in seen_seqs:
            rev_key = f"{route_name}__REV"
            if directed_times is None:
                rev_seg = list(reversed(seg_times))
                rev_cum = [0]
                for s in rev_seg:
                    rev_cum.append(rev_cum[-1] + s)
            else:
                rev_cum, rev_seg = _directed_route_timing(
                    rev_stops, directed_times, rev_key)
            library[rev_key] = {
                "route": str(route_name), "route_type": entry["route_type"],
                "direction": f"{rev_stops[0]}->{rev_stops[-1]}", "base_line": entry["base_line"],
                "stops": rev_stops, "cum": rev_cum, "seg_durations": rev_seg,
            }
            seen_seqs.add(tuple(rev_stops))

    print(f"  Route library: {len(library)} routes")
    return library


# =============================================================================
# 2. Time Window Setup
# =============================================================================
def parse_time_to_min(t_str, base_hour):
    if pd.isna(t_str):
        return None
    try:
        t = datetime.strptime(t_str, "%H:%M:%S")
        minutes = (t.hour - base_hour) * 60 + t.minute
        # Evening windows may contain trains that continue after midnight.
        # A clock time more than 12 hours behind the window start belongs to
        # the following day (00:04 at base_hour=20 is minute 244).
        if minutes < -12 * 60:
            minutes += 24 * 60
        return int(minutes // TIME_STEP)
    except Exception:
        return None

def build_window_data(start_h, schedule_df, od_df):
    """시간대별 열차 스케줄, OD, 수요 전처리."""
    end_h    = start_h + WINDOW_LENGTH_H
    BASE_H   = start_h
    START_MIN, END_MIN = 0, WINDOW_END_MIN
    SIM_HOURS = list(range(start_h, end_h))
    Gamma = list(range(START_MIN, END_MIN + 1))

    # Schedule
    sched = schedule_df[["Trip","StationID","SchArrive","SchDepart","Seq","Line","Direction"]].copy()
    min_seq = sched.groupby("Trip")["Seq"].transform("min")
    max_seq = sched.groupby("Trip")["Seq"].transform("max")
    sched["is_first"] = sched["Seq"] == min_seq
    sched["is_last"]  = sched["Seq"] == max_seq
    sched["event_time"] = sched.apply(lambda r: r["SchDepart"] if r["is_first"] else r["SchArrive"], axis=1)
    sched["t"]        = sched["event_time"].apply(lambda x: parse_time_to_min(x, BASE_H))
    sched["t_depart"] = sched["SchDepart"].apply(lambda x: parse_time_to_min(x, BASE_H))
    filtered = sched[(sched["t"] >= START_MIN) & (sched["t"] <= END_MIN)].copy()
    if filtered.empty:
        return None

    Vm = filtered["Trip"].unique().tolist()
    N  = list(filtered["StationID"].unique())

    trains_at   = defaultdict(lambda: defaultdict(list))
    station_of  = defaultdict(dict)
    I_vts       = set()
    for row in filtered.itertuples(index=False):
        v, s, t = str(row.Trip), row.StationID, row.t
        I_vts.add((v, t, s))
        trains_at[t][s].append(v)
        station_of[v][t] = s

    buffer = TRAIN_ACTIVE_PADDING_MIN
    train_active_times = {}
    trip_ranges = filtered.groupby("Trip")["t"].agg(["min","max"])
    for v, row in trip_ranges.iterrows():
        train_active_times[v] = range(max(START_MIN, row["min"]-buffer),
                                      min(END_MIN,   row["max"]+buffer)+1)

    train_dest_map = filtered[filtered["is_last"]].set_index("Trip")["StationID"].to_dict()

    # trip_line_dir_map, trip_stops_seq (급행 통과 판별용)
    trip_line_dir_map = (
        filtered[["Trip","Line","Direction"]].drop_duplicates("Trip")
        .set_index("Trip").apply(tuple, axis=1).to_dict()
    )
    # Cancellation timing needs the last safe upstream stop even when it lies
    # immediately outside the optimization window.  Keep the optimization
    # support filtered, but construct the timing sequence from the complete
    # schedule of every train that is active in this window.  This matches
    # 1scenario_generation_corridor.py causal_cancel_start_v3_20260809.
    active_trip_ids = set(Vm)
    timing_sched = sched[sched["Trip"].isin(active_trip_ids)]
    trip_stops_seq = {}
    for v, grp in timing_sched.groupby("Trip"):
        grp_s = grp.sort_values("Seq")
        trip_stops_seq[v] = list(zip(
            grp_s["Seq"].tolist(), grp_s["StationID"].tolist(),
            grp_s["t"].tolist(),   grp_s["t_depart"].tolist()
        ))

    # OD
    target_od = (
        od_df[od_df["Hour"].isin(SIM_HOURS)]
        .groupby(["OriginID", "DestinationID", "Hour"], as_index=False)[
            "Trips_Hourly"].sum()
    )
    target_od = target_od[
        target_od["Trips_Hourly"] > OD_HOURLY_MIN_TRIPS].copy()
    valid_pairs = set(target_od[[
        "OriginID", "DestinationID"]].itertuples(index=False, name=None))
    valid_stations = set(N)

    Omega_Set, Omega_Map = set(), {}
    final_od_list = []
    for (orig, dest), group in target_od.groupby(["OriginID","DestinationID"]):
        if (orig, dest) not in valid_pairs: continue
        if orig not in valid_stations or dest not in valid_stations: continue
        key = f"{orig}_{dest}"
        Omega_Set.add(key)
        Omega_Map[key] = (orig, dest)
        for row in group.itertuples(index=False):
            final_od_list.append({"omega": key, "OriginID": orig,
                                   "Hour": row.Hour, "Trips": row.Trips_Hourly})

    Omega = list(Omega_Set)

    # D[(omega, origin, t)]
    D    = defaultdict(float)
    CumD = defaultdict(float)
    steps_per_h = 60 / TIME_STEP
    for item in final_od_list:
        omega = item["omega"]
        hour_offset = (item["Hour"] - BASE_H) * 60
        t_s = max(START_MIN, hour_offset)
        t_e = min(END_MIN,   hour_offset + 59)
        if t_s > t_e: continue
        val = item["Trips"] / steps_per_h
        if val > 1e-9:
            for t in range(t_s, t_e + 1):
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
        if v not in train_active_times: continue
        times = sorted([t for t in train_active_times[v] if t in station_of[v]])
        if not times: continue
        curr_s, curr_seg = station_of[v][times[0]], [times[0]]
        for t in times[1:]:
            s = station_of[v][t]
            if s == curr_s and t == curr_seg[-1] + 1:
                curr_seg.append(t)
            else:
                train_dwells[v].append({"station": curr_s, "times": curr_seg})
                curr_s, curr_seg = s, [t]
        train_dwells[v].append({"station": curr_s, "times": curr_seg})

    valid_keys    = list(I_vts)
    no_board_keys = [(v, t, s) for (v, t, s) in valid_keys
                     if v in train_dest_map and s == train_dest_map[v]]

    return {
        "start_h": start_h, "end_h": end_h, "BASE_H": BASE_H,
        "START_MIN": START_MIN, "END_MIN": END_MIN,
        "Gamma": Gamma, "Vm": Vm, "N": N,
        "trains_at": trains_at, "station_of": station_of, "I_vts": I_vts,
        "train_active_times": train_active_times, "train_dest_map": train_dest_map,
        "train_dwells": train_dwells, "valid_keys": valid_keys,
        "no_board_keys": no_board_keys,
        "Omega": Omega, "Omega_Map": Omega_Map,
        "D": D, "CumD": CumD,
        "trip_line_dir_map": trip_line_dir_map, "trip_stops_seq": trip_stops_seq,
    }


# =============================================================================
# 3. Load Normal Results
# =============================================================================
_NORMAL_TRAJECTORY_CACHE = {}


def load_normal_trajectory(start_h):
    """Load the positive normal passenger trajectory by variable family.

    The normal-result CSV stores only positive rows.  Missing keys therefore
    have the exact modelling meaning zero.  Keeping all six families is
    required by the adaptive Stage 2: unaffected passengers are fixed to this
    trajectory, and the common-OD train residual capacity must remove their
    normal load exactly once.
    """
    window_name = f"{start_h:02d}_{start_h+WINDOW_LENGTH_H:02d}"
    fpath = os.path.join(DIR_NORMAL, f"Normal_{window_name}.csv")
    cache_key = (os.path.abspath(fpath), int(start_h))
    cached = _NORMAL_TRAJECTORY_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if not os.path.exists(fpath):
        print(f"  [WARN] Normal file not found: {fpath}")
        return {name: {} for name in ("b", "a", "l", "q", "w", "od")}
    df = pd.read_csv(fpath)
    required = {"Variable", "Omega", "Time", "Value",
                "StationID", "TrainID"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"normal trajectory {fpath} is missing columns: "
            f"{sorted(missing)}")

    result = {name: {} for name in ("b", "a", "l", "q", "w", "od")}
    family = {
        "Boarding": "b", "Alighting": "a", "Load": "l",
        "Queue": "q", "Walking": "w", "Exit": "od",
    }

    def _text(value):
        if pd.isna(value):
            return None
        text = str(value).strip()
        return text or None

    for row in df.itertuples(index=False):
        target = family.get(str(row.Variable))
        if target is None:
            continue
        omega = str(row.Omega)
        minute = int(row.Time)
        value = float(row.Value)
        station = _text(row.StationID)
        vehicle = _text(row.TrainID)
        if vehicle and vehicle.endswith(".0"):
            vehicle = vehicle[:-2]
        if target in ("b", "a"):
            key = (omega, vehicle, minute, station)
        elif target == "l":
            key = (omega, vehicle, minute)
        elif target in ("q", "w"):
            key = (omega, station, minute)
        else:
            key = (omega, minute)
        result[target][key] = value

    _NORMAL_TRAJECTORY_CACHE[cache_key] = result
    return result


def load_normal_results(start_h):
    """Normal_*.csv → normal_l_results[(omega, v, t)] = load value."""
    return load_normal_trajectory(start_h)["l"]


# =============================================================================
# 4. Disruption Processing — Corridor-based (SAME_CORRIDOR.csv)
# =============================================================================
# Corridor topology + station coords — 모듈 로드시 1회
FILE_CORRIDOR = str(cfg.path("inputs.corridors"))

_corridor_df = pd.read_csv(FILE_CORRIDOR)
_TRACK_COLUMNS = list(_corridor_df.columns)
_track_position = {}
_track_stations = {}
for _col in _TRACK_COLUMNS:
    _stations = [s for s in _corridor_df[_col].dropna().tolist() if str(s).strip()]
    _track_stations[_col] = _stations
    _track_position[_col] = {s: i for i, s in enumerate(_stations)}
_station_to_tracks = defaultdict(list)
for _col in _TRACK_COLUMNS:
    for _s in _track_stations[_col]:
        _station_to_tracks[_s].append(_col)

try:
    _xl_coord = pd.ExcelFile(FILE_EXCEL)
    _coord_df_full = pd.concat([_xl_coord.parse(s) for s in _xl_coord.sheet_names],
                                ignore_index=True).drop_duplicates(subset="Code")
    _station_coord_map = _coord_df_full.set_index("Code")[["Lat", "Long"]].apply(tuple, axis=1).to_dict()
except Exception:
    _station_coord_map = {}


def _haversine_dist(s1, s2):
    from math import radians, sin, cos, sqrt, atan2
    c1, c2 = _station_coord_map.get(s1), _station_coord_map.get(s2)
    if not c1 or not c2: return None
    lat1, lon1 = radians(c1[0]), radians(c1[1])
    lat2, lon2 = radians(c2[0]), radians(c2[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 6371.0 * 2 * atan2(sqrt(a), sqrt(1 - a))


def compute_t_event_for_trip(target_station, trip_stops):
    """
    트랙(corridor) 기반 통과/정차 시각 추정. Window 필터링은 호출 측.

    Case A) target_station이 정차역 → arrival(or departure) 시각
    Case B) 정차역은 아니나 동일 트랙 column 위에 정차역들이 (앞, 뒤) 모두 존재
            → 거리 비례 보간. 여러 트랙 후보면 min(t_event).
            상행(U)/하행(D) 모두 시간 순서로 출발측/도착측 판별.
    Returns: int t_event 또는 None
    """
    def _isnum(x):
        return x is not None and not (isinstance(x, float) and pd.isna(x))

    # Case A
    for (seq, sid, t_arr, t_dep) in trip_stops:
        if sid == target_station:
            t_event = t_arr if _isnum(t_arr) else t_dep
            if not _isnum(t_event):
                return None
            return int(t_event)

    # Case B
    candidate_tracks = _station_to_tracks.get(target_station, [])
    if not candidate_tracks:
        return None

    train_stations = [(s[1], s[2], s[3]) for s in trip_stops]
    best_event = None

    for track in candidate_tracks:
        pos_map = _track_position[track]
        tgt_pos = pos_map.get(target_station)
        if tgt_pos is None:
            continue

        on_track = []
        for (sid, t_arr, t_dep) in train_stations:
            p = pos_map.get(sid)
            if p is not None:
                on_track.append((p, sid, t_arr, t_dep))
        if len(on_track) < 2:
            continue
        on_track.sort(key=lambda x: x[0])

        prev_stop = None
        next_stop = None
        for entry in on_track:
            if entry[0] < tgt_pos:
                prev_stop = entry
            elif entry[0] > tgt_pos and next_stop is None:
                next_stop = entry
                break
            elif entry[0] == tgt_pos:
                prev_stop = entry
                next_stop = entry
                break
        if prev_stop is None or next_stop is None:
            continue
        if prev_stop is next_stop:
            continue

        p_prev_pos, p_prev_sid, p_prev_arr, p_prev_dep = prev_stop
        n_pos, n_sid, n_arr, n_dep = next_stop

        def _v(x):
            return x is not None and not (isinstance(x, float) and pd.isna(x))

        t_prev_dep = p_prev_dep if _v(p_prev_dep) else p_prev_arr
        t_prev_arr = p_prev_arr if _v(p_prev_arr) else p_prev_dep
        t_next_arr = n_arr if _v(n_arr) else n_dep
        t_next_dep = n_dep if _v(n_dep) else n_arr

        if _v(t_prev_dep) and _v(t_next_arr) and t_next_arr > t_prev_dep:
            # 하행(D): prev → next
            t_from, t_to = t_prev_dep, t_next_arr
            from_sid = p_prev_sid
        elif _v(t_next_dep) and _v(t_prev_arr) and t_prev_arr > t_next_dep:
            # 상행(U): next → prev
            t_from, t_to = t_next_dep, t_prev_arr
            from_sid = n_sid
        else:
            continue

        d_total = _haversine_dist(p_prev_sid, n_sid)
        d_to_tgt = _haversine_dist(from_sid, target_station)
        if d_total and d_to_tgt and d_total > 0:
            ratio = d_to_tgt / d_total
            ratio = max(0.0, min(1.0, ratio))
        else:
            ratio = 0.5

        t_event = int(t_from + ratio * (t_to - t_from))
        if best_event is None or t_event < best_event:
            best_event = t_event

    return best_event


def compute_cancellation_start(t_cross, t_cancel, disruption_start):
    """Return the first minute at which an exposed train is unavailable.

    A train that already departed its last safe upstream stop cannot be
    canceled retroactively.  It remains usable until it reaches the disruption
    location.  A train that has not departed when the disruption starts is
    blocked from its upstream departure event.
    """
    if t_cancel < disruption_start:
        return int(t_cross)
    return int(t_cancel)


def compute_disruption_times_for_trip(target_station, trip_stops):
    """Return ``(t_cross, t_cancel, case)`` using the causal v3 semantics."""
    def _isnum(value):
        return value is not None and not (
            isinstance(value, float) and pd.isna(value))

    # Case A: the train is scheduled to stop at the disrupted station.
    for index, (_seq, station, arrival, departure) in enumerate(trip_stops):
        if station != target_station:
            continue
        t_cross = arrival if _isnum(arrival) else departure
        if index > 0:
            _pseq, _pstation, previous_arrival, previous_departure = (
                trip_stops[index - 1])
            t_cancel = (
                previous_departure
                if _isnum(previous_departure) else previous_arrival)
        else:
            # If the target is the trip origin, prevent its own departure.
            t_cancel = departure if _isnum(departure) else arrival
        if not _isnum(t_cross) or not _isnum(t_cancel):
            return None
        return int(t_cross), int(t_cancel), "A"

    # Case B: the train passes the target between two stops on the same track.
    candidate_tracks = _station_to_tracks.get(target_station, [])
    if not candidate_tracks:
        return None
    train_stations = [
        (station, arrival, departure)
        for _seq, station, arrival, departure in trip_stops
    ]
    best_candidate = None

    for track in candidate_tracks:
        position = _track_position[track]
        target_position = position.get(target_station)
        if target_position is None:
            continue
        on_track = []
        for station, arrival, departure in train_stations:
            station_position = position.get(station)
            if station_position is not None:
                on_track.append(
                    (station_position, station, arrival, departure))
        if len(on_track) < 2:
            continue
        on_track.sort(key=lambda item: item[0])

        previous_stop = None
        next_stop = None
        for entry in on_track:
            if entry[0] < target_position:
                previous_stop = entry
            elif entry[0] > target_position and next_stop is None:
                next_stop = entry
                break
            elif entry[0] == target_position:
                previous_stop = entry
                next_stop = entry
                break
        if (previous_stop is None or next_stop is None or
                previous_stop is next_stop):
            continue

        _ppos, previous_station, previous_arrival, previous_departure = (
            previous_stop)
        _npos, next_station, next_arrival, next_departure = next_stop
        t_previous_departure = (
            previous_departure
            if _isnum(previous_departure) else previous_arrival)
        t_previous_arrival = (
            previous_arrival
            if _isnum(previous_arrival) else previous_departure)
        t_next_arrival = (
            next_arrival if _isnum(next_arrival) else next_departure)
        t_next_departure = (
            next_departure if _isnum(next_departure) else next_arrival)

        if (_isnum(t_previous_departure) and _isnum(t_next_arrival) and
                t_next_arrival > t_previous_departure):
            t_from, t_to = t_previous_departure, t_next_arrival
            from_station = previous_station
        elif (_isnum(t_next_departure) and _isnum(t_previous_arrival) and
              t_previous_arrival > t_next_departure):
            t_from, t_to = t_next_departure, t_previous_arrival
            from_station = next_station
        else:
            continue

        total_distance = _haversine_dist(previous_station, next_station)
        target_distance = _haversine_dist(from_station, target_station)
        if total_distance and target_distance and total_distance > 0:
            ratio = max(0.0, min(1.0, target_distance / total_distance))
        else:
            ratio = 0.5
        t_cross = int(t_from + ratio * (t_to - t_from))
        t_cancel = int(t_from)
        candidate = (t_cross, t_cancel, "B")
        if (best_candidate is None or
                (candidate[0], candidate[1]) <
                (best_candidate[0], best_candidate[1])):
            best_candidate = candidate

    return best_candidate


def _load_scenario_logic_manifest(start_h):
    """Return the latest recorded logic/checkpoint for each disruption CSV.

    The individual result CSVs predate logic-version metadata.  The scenario
    generator records it in its checkpoint statistics, so consumers must use
    that manifest to avoid mixing stale passenger-impact files with newly
    computed cancellation masks.
    """
    manifest = {}
    paths = [os.path.join(DIR_DISRUPT, "_scenario_stats.csv")]
    paths.append(os.path.join(
        DIR_DISRUPT, f"_scenario_stats_w{int(start_h):02d}.csv"))
    for path in paths:
        if not os.path.isfile(path):
            continue
        try:
            stats = pd.read_csv(path, low_memory=False)
        except Exception as exc:
            raise RuntimeError(
                f"cannot read disruption scenario manifest {path}: {exc}") from exc
        required = {"scenario", "logic_version", "checkpoint"}
        if not required.issubset(stats.columns):
            continue
        for row in stats.itertuples(index=False):
            scenario_id = str(getattr(row, "scenario", "")).strip()
            if not scenario_id:
                continue
            version = getattr(row, "logic_version", None)
            checkpoint = getattr(row, "checkpoint", None)
            manifest[scenario_id] = (
                "" if pd.isna(version) else str(version).strip(),
                "" if pd.isna(checkpoint) else str(checkpoint).strip(),
                path,
            )
    return manifest


def load_disruption_scenarios(start_h, wdata):
    """
    DIR_DISRUPT 내 Disrupt_{start_h}_{station}_{duration}.csv 파일을 스캔하여
    시나리오별 canceled_trains, theta, Omega_m, spare_capacity 생성.

    Patches vs original:
      1) compute_t_event_for_trip — corridor 양방향(U/D) 통과 판정
      2) total_l/l_by_om 한 번 precompute (시나리오마다 normal_l 전체 스캔 제거)
      3) t_event_by_trip 을 station별로 캐싱 (duration 3개에 걸쳐 공유)
    """
    normal_l = load_normal_results(start_h)
    Vm       = wdata["Vm"]
    trip_stops_seq    = wdata["trip_stops_seq"]
    train_active_times = wdata["train_active_times"]
    START_MIN = wdata["START_MIN"]

    scenarios = []

    if not os.path.isdir(DIR_DISRUPT):
        return scenarios

    logic_manifest = _load_scenario_logic_manifest(start_h)

    # (1) Precompute aggregated loads — spare_capacity 시나리오별 재계산 비용 제거
    total_l = defaultdict(float)
    l_by_om = defaultdict(list)
    for (om, v, t), val in normal_l.items():
        total_l[(v, t)] += val
        l_by_om[om].append((v, t, val))

    # (2) station별 t_event 캐시 — 같은 station의 30/60/90 duration이 공유
    disruption_times_cache_by_station = {}

    prefix = f"Disrupt_{start_h}_"
    for fname in sorted(os.listdir(DIR_DISRUPT)):
        if not fname.startswith(prefix) or not fname.endswith(".csv"):
            continue
        middle = fname[len(prefix):-4]
        last_us = middle.rfind("_")
        if last_us == -1: continue
        try:
            station  = middle[:last_us]
            duration = int(middle[last_us+1:])
        except ValueError:
            continue
        if duration not in DURATIONS:
            continue

        scenario_id = f"{start_h}_{station}_{duration}"
        recorded = logic_manifest.get(scenario_id)
        # A v3 NO_AFFECTED checkpoint supersedes any stale result CSV left by
        # an older run.  Ignore that file without deleting user data.
        if (recorded is not None and
                recorded[0] == SCENARIO_LOGIC_VERSION and
                recorded[1] == "NO_AFFECTED"):
            continue
        if (recorded is None or recorded[0] != SCENARIO_LOGIC_VERSION or
                recorded[1] != "EXTRACTED"):
            actual = "unrecorded" if recorded is None else (
                f"version={recorded[0] or 'unversioned'}, "
                f"checkpoint={recorded[1] or 'missing'}")
            raise RuntimeError(
                f"{scenario_id}: stale disruption CSV ({actual}); regenerate "
                f"it with {SCENARIO_LOGIC_VERSION} before running CG")

        DISRUPT_START_T = START_MIN + DISRUPT_OFFSET
        DISRUPT_END_T   = DISRUPT_START_T + duration
        DISPATCH_END_MIN = DISRUPT_END_T + DISRUPT_BUFFER

        # station별 t_event 한 번만 계산
        if station not in disruption_times_cache_by_station:
            timing_map = {}
            for v in Vm:
                stops = trip_stops_seq.get(v, [])
                if not stops:
                    continue
                timing = compute_disruption_times_for_trip(station, stops)
                if timing is not None:
                    timing_map[v] = timing
            disruption_times_cache_by_station[station] = timing_map
        disruption_times_by_trip = disruption_times_cache_by_station[station]

        # --- Identify canceled trains (window 필터링만) ---
        theta = defaultdict(int)
        canceled_trains = set()
        exposure_time_by_train = {}
        cancellation_start_by_train = {}
        cancellation_case_by_train = {}
        for v, (t_cross, t_cancel, event_case) in disruption_times_by_trip.items():
            # Exposure is determined at the disrupted location.  The actual
            # block may start at the last safe upstream departure, but never
            # retroactively if the train already departed before the event.
            if not (DISRUPT_START_T <= t_cross <= DISRUPT_END_T):
                continue
            block_from = compute_cancellation_start(
                t_cross, t_cancel, DISRUPT_START_T)
            canceled_trains.add(v)
            exposure_time_by_train[v] = int(t_cross)
            cancellation_start_by_train[v] = int(block_from)
            cancellation_case_by_train[v] = event_case
            for t in train_active_times.get(v, []):
                if t >= block_from:
                    theta[(v, t)] = 1

        # --- Omega_m ---
        Omega_m = set()
        for (om, v, t), val in normal_l.items():
            if v in canceled_trains and theta[(v,t)] == 1:
                Omega_m.add(om)
        Omega_m = list(Omega_m)
        if not Omega_m:
            continue

        # --- spare_capacity (total_l - sub_l 패턴) ---
        sub_l = defaultdict(float)
        for om in Omega_m:
            for (v, t, val) in l_by_om[om]:
                sub_l[(v, t)] += val

        spare_capacity = {}
        for v in Vm:
            for t in train_active_times.get(v, []):
                occ = total_l.get((v, t), 0.0) - sub_l.get((v, t), 0.0)
                limit = (TRAIN_CAPACITY - occ) * (1 - theta[(v, t)])
                spare_capacity[(v, t)] = max(0.0, limit)

        # --- pair_score: Σ_t D[(ω, origin, t)] ---
        D = wdata["D"]
        Omega_Map = wdata["Omega_Map"]
        pair_score = {}
        for om in Omega_m:
            if om not in Omega_Map: continue
            origin, dest = Omega_Map[om]
            score = sum(D.get((om, origin, t), 0) for t in wdata["Gamma"])
            pair_score[(origin, dest)] = pair_score.get((origin, dest), 0) + score

        # --- impact_by_station: 시나리오 클러스터링 feature (F2 spatial) ---
        # --- strand_score:      (대기역, 목적지)별 Queue+Walking 합계 ---
        # Disrupt CSV 한 번 읽어 둘 다 뽑는다.
        #
        # pair_score 는 (원출발역, 목적지) 기준이라 "승객이 실제로 발이 묶인 역"을
        # 담지 못한다. 표본 40개 시나리오에서 잔류 수요의 49.3% 가 원출발역이 아닌
        # 중간역(환승 간선역)에 몰려 있었다. 버스가 그 사람들을
        # 태우려면 대기역 기준 수요가 필요하다.
        impact_by_station = {}
        strand_score = {}
        try:
            ddf = pd.read_csv(os.path.join(DIR_DISRUPT, fname))
            if "Variable" in ddf.columns and "StationID" in ddf.columns:
                sp = ddf[ddf["Variable"].isin(["Queue", "Walking"])]
                if not sp.empty:
                    impact_by_station = (sp.groupby("StationID")["Value"]
                                         .sum().to_dict())
                    sp = sp[sp["Value"] > 0]
                    if not sp.empty and "Dest" in sp.columns:
                        gs = sp.groupby([sp["StationID"].astype(str).str.strip(),
                                         sp["Dest"].astype(str).str.strip()])["Value"].sum()
                        strand_score = {(o, d): float(v) for (o, d), v in gs.items()
                                        if o and d and o != d}
        except Exception:
            pass

        scenarios.append({
            "id": scenario_id,
            "logic_version": SCENARIO_LOGIC_VERSION,
            "logic_manifest": recorded[2],
            "station": station,
            "duration": duration,
            "DISRUPT_START_T": DISRUPT_START_T,
            "DISRUPT_END_T": DISRUPT_END_T,
            "DISPATCH_END_MIN": DISPATCH_END_MIN,
            "canceled_trains": canceled_trains,
            "exposure_time_by_train": exposure_time_by_train,
            "cancellation_start_by_train": cancellation_start_by_train,
            "cancellation_case_by_train": cancellation_case_by_train,
            "theta": dict(theta),
            "Omega_m": Omega_m,
            "spare_capacity": spare_capacity,
            "pair_score": pair_score,
            "strand_score": strand_score,
            "impact_by_station": impact_by_station,
        })

    return scenarios


# =============================================================================
# 5. Scenario Reduction (methodology-style F2 -> PCA -> K-means)
# =============================================================================
def _load_station_coords():
    """역 좌표 로드 (k_clustering.py와 동일)"""
    try:
        xl = pd.ExcelFile(FILE_EXCEL)
        coord_map = {}
        for sheet in xl.sheet_names:
            df = xl.parse(sheet)
            for _, row in df.iterrows():
                if row["Code"] not in coord_map:
                    coord_map[row["Code"]] = (float(row["Lat"]), float(row["Long"]))
        return coord_map
    except Exception:
        return {}

# 역 좌표 캐시 (최초 1회 로드)
_COORD_CACHE = None

def _get_coord_map():
    global _COORD_CACHE
    if _COORD_CACHE is None:
        _COORD_CACHE = _load_station_coords()
    return _COORD_CACHE


def _reduce_scenarios_duration_legacy(scenarios, k_per_duration=K_PER_DURATION):
    """
    duration별로 시나리오를 K개 cluster로 축소.

    Feature pipeline (F2 spatial):
      1) 역별 (Queue + Walking) 임팩트 벡터  (impact_by_station)
      2) log(1+x)               — long-tail 압축
      3) row-normalize          — 시나리오 절대 크기 제거 (분포만 유지)
      4) z-score                — 역별 동등 가중
      5) PCA(min(5, n-1, d))    — 차원의 저주 회피, EVR ~80%
      6) K-means(k_per_duration)

    대표: centroid에 가장 가까운 실제 시나리오.
    probability: cluster 크기 / duration 내 전체 시나리오 수.
    """
    if not scenarios:
        return []

    by_duration = defaultdict(list)
    for s in scenarios:
        by_duration[s["duration"]].append(s)

    reduced = []
    for dur, dur_scenarios in sorted(by_duration.items()):
        n = len(dur_scenarios)
        k = min(k_per_duration, n)
        dur_weight = DURATION_WEIGHT.get(dur, 1.0 / len(DURATIONS))

        # 1) 역별 임팩트 매트릭스
        all_st = sorted({st for s in dur_scenarios
                         for st in s.get("impact_by_station", {}).keys()})
        if not all_st:
            # impact_by_station 가 없는 경우 (CSV 결측) → 그대로 모두 사용
            for s in dur_scenarios:
                rep = s.copy()
                rep["probability"] = dur_weight / n
                reduced.append(rep)
            _log(f"    duration={dur}: impact_by_station 없음, "
                 f"{n}개 그대로 사용")
            continue

        idx = {st: i for i, st in enumerate(all_st)}
        X = np.zeros((n, len(all_st)), dtype=float)
        for i, s in enumerate(dur_scenarios):
            for st, v in s.get("impact_by_station", {}).items():
                if st in idx:
                    X[i, idx[st]] = v

        # 2) log(1+x)
        X = np.log1p(X)
        # 3) row-normalize
        rs = X.sum(axis=1, keepdims=True); rs[rs < 1e-9] = 1.0
        X = X / rs
        # 4) z-score
        mu = X.mean(0); sd = X.std(0); sd[sd < 1e-9] = 1.0
        X = (X - mu) / sd

        # 5) PCA
        n_comp = min(5, n - 1, X.shape[1])
        if n_comp >= 2:
            pca = PCA(n_components=n_comp, random_state=42)
            X_red = pca.fit_transform(X)
            evr = float(pca.explained_variance_ratio_.sum())
        else:
            X_red = X
            evr = 1.0

        if k == 1 or n == 1:
            rep = dur_scenarios[0].copy()
            rep["probability"] = dur_weight
            reduced.append(rep)
            continue

        # 6) K-means
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_red)

        for cluster_id in range(k):
            members = [i for i, lb in enumerate(labels) if lb == cluster_id]
            if not members:
                continue
            centroid = km.cluster_centers_[cluster_id]
            dists = [np.linalg.norm(X_red[i] - centroid) for i in members]
            rep_idx = members[int(np.argmin(dists))]
            rep = dur_scenarios[rep_idx].copy()
            rep["probability"] = dur_weight * (len(members) / n)
            reduced.append(rep)

        _log(f"    duration={dur}: {n}개 → {k}개 대표 시나리오 "
             f"(F2 -> PCA({n_comp}), EVR={evr:.1%})")

    return reduced


def _reduce_scenarios_total_legacy(scenarios, k_total=K_PER_DURATION):
    """
    make_methodology_figures.py의 F2 -> PCA(5) -> K-means 방법을 실제 MILP
    scenario reduction에 적용한다.

    Difference from the legacy reducer:
      - duration별로 K개씩 뽑지 않는다.
      - 한 start-hour의 30/60/90 raw scenarios 전체를 같이 놓고 총 K개로 줄인다.
      - feature transform은 methodology figure와 동일하게
        raw station impact -> log1p -> row-normalize -> z-score -> PCA(5) 순서다.

    MILP에는 representative probability가 필요하므로, 각 raw scenario에
    DURATION_WEIGHT[duration] / count(duration)을 base probability로 주고
    cluster representative probability는 member base probabilities의 합으로 둔다.
    """
    if not scenarios:
        return []

    n = len(scenarios)
    k = min(k_total, n)

    by_duration = defaultdict(list)
    for s in scenarios:
        by_duration[s["duration"]].append(s)

    base_prob = []
    for s in scenarios:
        dur = s["duration"]
        dur_count = max(len(by_duration[dur]), 1)
        dur_weight = DURATION_WEIGHT.get(dur, 1.0 / len(DURATIONS))
        base_prob.append(dur_weight / dur_count)
    base_prob = np.asarray(base_prob, dtype=float)

    all_st = sorted({st for s in scenarios
                     for st in s.get("impact_by_station", {}).keys()})
    reduced = []
    if not all_st:
        for s, prob in zip(scenarios, base_prob):
            rep = s.copy()
            rep["probability"] = float(prob)
            reduced.append(rep)
        _log(f"    all durations: impact_by_station 없음, {n}개 그대로 사용")
        return reduced

    idx = {st: i for i, st in enumerate(all_st)}
    X = np.zeros((n, len(all_st)), dtype=float)
    for i, s in enumerate(scenarios):
        for st, v in s.get("impact_by_station", {}).items():
            if st in idx:
                X[i, idx[st]] = v

    # Same transform chain as make_methodology_figures.transform_chain.
    X = np.log1p(X)
    rs = X.sum(axis=1, keepdims=True)
    rs[rs < 1e-9] = 1.0
    X = X / rs
    mu = X.mean(0)
    sd = X.std(0)
    sd[sd < 1e-9] = 1.0
    X = (X - mu) / sd

    n_comp = min(5, n - 1, X.shape[1])
    if n_comp >= 2:
        pca = PCA(n_components=n_comp, random_state=42)
        X_red = pca.fit_transform(X)
        evr = float(pca.explained_variance_ratio_.sum())
    else:
        X_red = X
        evr = 1.0

    if k == 1 or n == 1:
        centroid = X_red.mean(axis=0)
        dists = np.linalg.norm(X_red - centroid, axis=1)
        rep_idx = int(np.argmin(dists))
        rep = scenarios[rep_idx].copy()
        rep["probability"] = float(base_prob.sum())
        rep["cluster_id"] = 0
        rep["cluster_size"] = n
        reduced.append(rep)
        return reduced

    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = km.fit_predict(X_red)

    for cluster_id in range(k):
        members = [i for i, lb in enumerate(labels) if lb == cluster_id]
        if not members:
            continue
        centroid = km.cluster_centers_[cluster_id]
        dists = [np.linalg.norm(X_red[i] - centroid) for i in members]
        rep_idx = members[int(np.argmin(dists))]
        rep = scenarios[rep_idx].copy()
        rep["probability"] = float(base_prob[members].sum())
        rep["cluster_id"] = int(cluster_id)
        rep["cluster_size"] = int(len(members))
        reduced.append(rep)

    dur_counts = ", ".join(f"{dur}:{len(by_duration[dur])}" for dur in sorted(by_duration))
    _log(f"    all durations: {n}개 ({dur_counts}) -> {len(reduced)}개 대표 시나리오 "
         f"(F2 -> PCA({n_comp}), EVR={evr:.1%}, K={k})")

    return reduced


def _methodology_stage0_raw_matrix(scenarios):
    all_st = sorted({st for s in scenarios
                     for st in s.get("impact_by_station", {}).keys()})
    idx = {st: i for i, st in enumerate(all_st)}
    X = np.zeros((len(scenarios), len(all_st)), dtype=float)
    for i, scenario in enumerate(scenarios):
        for st, value in scenario.get("impact_by_station", {}).items():
            X[i, idx[st]] = value
    return X, all_st


def _methodology_transform_chain(X):
    """
    Match k_clustering.py exactly: log1p -> z-score (no row-normalization, no PCA).
    """
    X1 = np.log1p(X)
    mu = X1.mean(axis=0)
    sd = X1.std(axis=0)
    sd[sd < 1e-9] = 1.0
    return (X1 - mu) / sd


def _methodology_select_representatives_within_duration(dur_scenarios, dur_labels,
                                                        X_dur, k, dur, medoids=None,
                                                        prior=None):
    """
    Within a single duration, pick one representative per cluster:
    the scenario closest to the cluster's feature-space centroid.

    Probability = dur_weight * (군집이 차지하는 공간확률 비중)

      prior 가 None(=uniform)  : cluster_size / n_dur          ← 기존 동작
      prior 가 주어지면        : Σ_{i∈군집} p(역_i) / Σ_j p(역_j)

    prior 는 scenario_station_prior() 가 준 {역: 확률} 이며, Reviewer #3 (2) 의
    "uniform spatial probabilities" 지적에 대응하는 지점이다.
    """
    dur_labels = np.asarray(dur_labels)
    X_dur = np.asarray(X_dur, dtype=float)
    n_dur = len(dur_scenarios)
    dur_weight = DURATION_WEIGHT.get(dur, 1.0 / len(DURATIONS))
    out = []

    for cid in range(k):
        local_members = [i for i, lb in enumerate(dur_labels) if lb == cid]
        if not local_members:
            _log(f"    [WARN] duration={dur}, cluster={cid}: no member")
            continue

        X_members = X_dur[local_members, :]
        if medoids is not None:
            # k-medoids: 대표가 이미 실제 시나리오다. 근사할 필요가 없다.
            rep_idx = int(medoids[cid])
            rep_local_idx = local_members.index(rep_idx) \
                if rep_idx in local_members else 0
            dists = np.linalg.norm(X_members - X_dur[rep_idx], axis=1)
        else:
            center = X_members.mean(axis=0)
            dists = np.linalg.norm(X_members - center, axis=1)
            rep_local_idx = int(np.argmin(dists))
            rep_idx = local_members[rep_local_idx]

        rep = dur_scenarios[rep_idx].copy()
        if prior:
            # 공간확률 가중: 군집이 차지하는 p(역) 비중
            num = sum(prior.get(dur_scenarios[i]["station"], 0.0)
                      for i in local_members)
            den = sum(prior.get(s["station"], 0.0) for s in dur_scenarios) or 1.0
            share = num / den
        else:
            share = len(local_members) / n_dur          # 균등 가정 (기존)
        rep["probability"] = dur_weight * share
        rep["prob_share"] = float(share)
        rep["cluster_id"] = int(cid)
        rep["cluster_size"] = int(len(local_members))
        rep["representative_id"] = rep["id"]
        rep["distance_to_feature_centroid"] = float(dists[rep_local_idx])
        out.append(rep)

    return out


def kmedoids(D, k, seed=42, n_init=10, max_iter=300):
    """
    PAM 근사 k-medoids: k-medoids++ 초기화 → Voronoi 반복 → swap 개선.

    D    : 사전계산 거리행렬 (n x n)
    반환 : (labels, medoid_indices, cost)

    KMeans 와 달리 대표가 데이터에 실재하는 점이므로, 시나리오 축소에서
    '대표 = 실제 시나리오' 조건이 근사 없이 만족된다.
    """
    rng = np.random.default_rng(seed)
    n = D.shape[0]
    k = min(k, n)
    best = None

    for _ in range(n_init):
        med = [int(rng.integers(n))]
        for _ in range(k - 1):
            d = D[:, med].min(axis=1) ** 2
            s = d.sum()
            med.append(int(rng.choice(n, p=d / s)) if s > 0 else int(rng.integers(n)))
        med = list(dict.fromkeys(med))
        while len(med) < k:
            c = int(rng.integers(n))
            if c not in med:
                med.append(c)
        med = np.array(med)

        for _ in range(max_iter):
            lab = np.argmin(D[:, med], axis=1)
            new = med.copy()
            for c in range(k):
                mem = np.where(lab == c)[0]
                if len(mem):
                    new[c] = mem[int(np.argmin(D[np.ix_(mem, mem)].sum(axis=1)))]
            if np.array_equal(np.sort(new), np.sort(med)):
                med = new
                break
            med = new

        cost = D[:, med].min(axis=1).sum()
        improved = True
        while improved:                       # PAM swap
            improved = False
            for ci in range(k):
                for cand in range(n):
                    if cand in med:
                        continue
                    trial = med.copy(); trial[ci] = cand
                    c2 = D[:, trial].min(axis=1).sum()
                    if c2 < cost - 1e-12:
                        med, cost, improved = trial, c2, True
                        break
                if improved:
                    break

        lab = np.argmin(D[:, med], axis=1)
        if best is None or cost < best[2]:
            best = (lab, med, cost)
    return best


def _cluster_once(X, k):
    """
    CLUSTER_METHOD / CLUSTER_METRIC 에 따라 한 번 군집화.
    반환 (labels, medoids or None).
    """
    if k <= 1 or X.shape[0] <= 1:
        return np.zeros(X.shape[0], dtype=int), None
    if CLUSTER_METHOD == "kmedoids":
        from scipy.spatial.distance import pdist, squareform
        metric = CLUSTER_METRIC
        if metric == "cosine" and np.any(np.linalg.norm(X, axis=1) < 1e-9):
            metric = "euclidean"       # 영벡터가 있으면 코사인이 정의되지 않는다
        D = squareform(pdist(X, metric=metric))
        lab, med, _ = kmedoids(D, k)
        return lab, med
    return KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(X), None


def cluster_curves(X, k_min=K_MIN, k_max=K_MAX):
    """
    duration 하나의 특징행렬에 대해 K 별 평균 silhouette 와 inertia 를 계산.
    반환: (sil {k: score}, inertia {k: value}). 2 <= k <= min(k_max, n-1).
    """
    from sklearn.metrics import silhouette_score

    n = X.shape[0]
    hi = min(k_max, n - 1)
    lo = max(2, k_min)
    sil, inertia = {}, {}
    for k in range(lo, hi + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=20).fit(X)
        if len(set(km.labels_)) < 2:
            continue
        try:
            sil[k] = float(silhouette_score(X, km.labels_))
        except ValueError:
            continue
        inertia[k] = float(km.inertia_)
    return sil, inertia


def silhouette_curve(X, k_min=K_MIN, k_max=K_MAX):
    """하위호환 래퍼."""
    return cluster_curves(X, k_min, k_max)[0]


def _knee_k(inertia):
    """
    kneedle: inertia 곡선의 양끝을 잇는 직선에서 가장 멀리 떨어진 K.
    silhouette 를 쓰지 않는 대안 기준.
    """
    ks = sorted(inertia)
    if len(ks) < 3:
        return ks[0] if ks else None
    x = np.array(ks, dtype=float)
    y = np.array([inertia[k] for k in ks], dtype=float)
    x = (x - x.min()) / max(x.max() - x.min(), 1e-12)
    y = (y - y.min()) / max(y.max() - y.min(), 1e-12)
    # 양끝을 잇는 직선까지의 수직 거리
    d = np.abs((y[-1] - y[0]) * x - (x[-1] - x[0]) * y + x[-1] * y[0] - y[-1] * x[0])
    d /= max(np.hypot(y[-1] - y[0], x[-1] - x[0]), 1e-12)
    return ks[int(np.argmax(d))]


def pick_k(sil, inertia, criterion=None, tau=None):
    """
    곡선 하나에서 K 를 고른다. 반환: (k, 설명문자열). 곡선이 비면 (None, "").
    """
    criterion = (criterion or K_CRITERION)
    tau = K_SIL_TAU if tau is None else tau
    if not sil:
        return None, ""

    smax = max(sil.values())
    if criterion == "argmax":
        k = max(sil, key=sil.get)
        return k, f"argmax s={smax:.3f}"
    if criterion == "elbow":
        k = _knee_k(inertia) or max(sil, key=sil.get)
        return k, f"elbow s={sil.get(k, float('nan')):.3f}"
    if criterion == "threshold":
        cand = [k for k in sil if sil[k] >= tau * smax]
        k = max(cand) if cand else max(sil, key=sil.get)
        return k, f"s>={tau:.0%}max -> s={sil[k]:.3f} (max {smax:.3f})"
    # abs (기본): s 가 K_SIL_ABS 를 처음 넘는 최소 K
    cand = [k for k in sorted(sil) if sil[k] >= K_SIL_ABS]
    if cand:
        k = cand[0]
        return k, f"first s>={K_SIL_ABS:.2f} at K={k} (s={sil[k]:.3f})"
    # 한 번도 못 넘음 → argmax 폴백. 이 폴백이 얼마나 자주 걸리는지는 보고해야 한다.
    k = max(sil, key=sil.get)
    return k, f"never reached {K_SIL_ABS:.2f}, argmax K={k} (s={smax:.3f})"


def _largest_remainder(weights, total, lo, hi):
    """
    weights 비율대로 total 을 정수 배분하되 각 항목을 [lo[i], hi[i]] 로 제한.
    최대잔여법 — 반올림만 하면 합이 total 과 어긋난다.
    """
    keys = list(weights)
    wsum = sum(weights[k] for k in keys) or 1.0
    raw = {k: total * weights[k] / wsum for k in keys}
    alloc = {k: max(lo[k], min(hi[k], int(raw[k]))) for k in keys}

    # 남거나 모자란 만큼을 잔여가 큰/작은 순으로 조정
    for _ in range(10 * len(keys) + total):
        diff = total - sum(alloc.values())
        if diff == 0:
            break
        if diff > 0:
            cand = [k for k in keys if alloc[k] < hi[k]]
            if not cand:
                break
            k = max(cand, key=lambda k: raw[k] - alloc[k])
            alloc[k] += 1
        else:
            cand = [k for k in keys if alloc[k] > lo[k]]
            if not cand:
                break
            k = min(cand, key=lambda k: raw[k] - alloc[k])
            alloc[k] -= 1
    return alloc


def resolve_k_per_duration(X_by_dur, k_per_duration=K_PER_DURATION):
    """
    duration 별 K 를 결정한다 (Reviewer #5 Q9).

    fixed    : 전부 k_per_duration.
    adaptive : duration 마다 K_CRITERION 이 고른 K 를 그대로 사용.
               → 총 대표 수가 달라지므로 Stage 1 규모도 달라진다.
    budget   : 그 K 들의 비율대로 총 대표 수를 재배분. 총합이 fixed 와 같으므로
               '같은 계산예산에서 배분만 바꿨을 때 무엇이 나아지는가'를 볼 수 있다.
               심사 대응에는 이쪽이 핵심이다.

    반환: ({duration: K}, 진단문자열)
    """
    durs = [d for d in DURATIONS if d in X_by_dur]
    if not durs:
        return {}, ""

    if K_MODE == "fixed":
        return ({d: min(k_per_duration, X_by_dur[d].shape[0]) for d in durs}, "")

    best, notes = {}, []
    for d in durs:
        sil, inertia = cluster_curves(X_by_dur[d])
        k, why = pick_k(sil, inertia)
        best[d] = k or min(k_per_duration, X_by_dur[d].shape[0])
        notes.append(f"{d}:K={best[d]} n={X_by_dur[d].shape[0]} [{why}]")
    diag = f"{K_CRITERION} | " + " | ".join(notes)

    if K_MODE == "adaptive":
        return ({d: max(1, min(best[d], X_by_dur[d].shape[0])) for d in durs}, diag)

    if K_MODE == "budget":
        total = K_BUDGET or (len(DURATIONS) * k_per_duration)
        lo = {d: min(K_MIN, X_by_dur[d].shape[0]) for d in durs}
        hi = {d: min(K_MAX, X_by_dur[d].shape[0]) for d in durs}
        alloc = _largest_remainder({d: float(best[d]) for d in durs}, total, lo, hi)
        return alloc, diag + f" | budget={total} -> " + ", ".join(
            f"{d}:{alloc[d]}" for d in durs)

    print(f"[WARN] unknown TSB_K_MODE={K_MODE!r}, falling back to fixed")
    return ({d: min(k_per_duration, X_by_dur[d].shape[0]) for d in durs}, "")


_INCIDENT_FREQ_CACHE = None


def incident_frequency():
    """
    사고 기록에서 역별 발생 건수를 센다 (ItemId 단위로 유일화).

    모집단은 '전체 사고' 다. 지연 15분 이상으로 좁히면 시나리오 역의 1/4 이
    0건이 되어 확률 0 이 생긴다. 지속시간 분포(DURATION_WEIGHT)는 조건부로
    추정하되, '어느 역이 끊기나' 는 모든 사고가 정보를 준다.
    """
    global _INCIDENT_FREQ_CACHE
    if _INCIDENT_FREQ_CACHE is not None:
        return _INCIDENT_FREQ_CACHE
    freq = {}
    if os.path.exists(FILE_INCIDENT):
        try:
            df = pd.read_csv(FILE_INCIDENT, usecols=["ItemId", "LocationCode"],
                             low_memory=False)
            first = (df.dropna(subset=["LocationCode"])
                       .groupby("ItemId")["LocationCode"].first())
            freq = first.astype(str).str.strip().value_counts().to_dict()
        except Exception as e:
            print(f"[WARN] 사고 빈도 로딩 실패: {e}")
    else:
        print(f"[WARN] 사고 파일 없음: {FILE_INCIDENT}")
    _INCIDENT_FREQ_CACHE = freq
    return freq


def station_exposure(wdata):
    """그 윈도우 동안 각 역을 지나는 열차 수 (trip 기준 유일화)."""
    cnt, seen = {}, set()
    for (v, t, s) in wdata["I_vts"]:
        if (v, s) in seen:
            continue
        seen.add((v, s))
        cnt[s] = cnt.get(s, 0) + 1
    return cnt


def scenario_station_prior(scenarios, wdata=None, mode=None):
    """
    시나리오 역별 사전확률 p(역) 을 반환한다 (합 1).

    uniform   : 1/n
    frequency : (사고 건수 + alpha) 정규화
    exposure  : (열차 통과 수 + alpha) 정규화 — wdata 필요
    """
    mode = (mode or SCENARIO_PRIOR)
    stations = sorted({s["station"] for s in scenarios})
    n = len(stations)
    if n == 0:
        return {}
    if mode == "uniform":
        return {st: 1.0 / n for st in stations}

    if mode == "frequency":
        raw = incident_frequency()
    elif mode == "exposure":
        if wdata is None:
            print("[WARN] exposure prior 에 wdata 가 필요함 → uniform 사용")
            return {st: 1.0 / n for st in stations}
        raw = station_exposure(wdata)
    else:
        print(f"[WARN] unknown TSB_SCENARIO_PRIOR={mode!r} → uniform 사용")
        return {st: 1.0 / n for st in stations}

    vals = {st: float(raw.get(st, 0.0)) + PRIOR_ALPHA for st in stations}
    tot = sum(vals.values()) or 1.0
    return {st: v / tot for st, v in vals.items()}


def _reduce_scenarios_station_blocks(scenarios, k_stations=K_STATIONS, wdata=None):
    """
    역 단위 축소 — 대표역 하나가 30/60/90 을 통째로 가져온다.

        Z_D     = zscore(log1p(imp_D))       duration 블록마다 따로 적합
        X_joint = hstack( √P(D) · Z_D )      ← 확률가중이 거리에 내장된다
        선택     KMeans(random_state=42, n_init=20), 중심 최근접 관측점
        확률     P(k, D) = P(k) · P(D)

    √P(D) 스케일링의 요점: 결합공간의 제곱거리가 duration 별 제곱거리의
    확률가중합과 정확히 같아진다.

        ‖x_i − x_j‖²_joint = Σ_D P(D)·‖x_i^D − x_j^D‖²

    K-means 가 제곱거리를 최소화하므로, **평범한 K-means 를 그대로 쓰면서**
    duration 가중이 P(D) 가 된다. 새 알고리즘을 도입할 필요가 없고 논문
    방법론(K-means, random_state=42, n_init=20, 중심 최근접 대표)이 연속된다.

    concat 을 한 번에 표준화하는 순진한 대안은 세 duration 을 사실상 1/3 씩
    균등 가중한다 (실측 거리기여 0.320 / 0.337 / 0.342). 실제 P(D) 는
    (0.506, 0.222, 0.272) 이라 30분이 크게 과소 가중된다. √P(D) 가 이를 고친다.

    같은 기계가 `common_station_kmeans_k10.py` / `station_bundle_clustering.py`
    에도 있다. 두 경로가 어긋나면 안 되므로 변환·시드·대표선택 규칙을 일치시킨다.

    duration 을 축으로 유지하는 근거: duration 별 독립 K-means 파티션끼리의
    ARI 가 30-60 0.328, 30-90 0.276, 60-90 0.597 로 낮다. 특히 30분과 90분의
    군집 구조는 거의 다르다. concat 은 이를 좌표 평균으로 뭉갠다.

    ★ 더 단순한 대안 — 가중치를 곱해 한 벡터로 접기 — 는 시험 후 기각했다.

        A1  log1p(Σ_D P(D)·imp_D)         변환 전에 접기
        A2  Σ_D P(D)·log1p(imp_D)         변환 후 좌표 평균
        B   Σ_D P(D)·c(z(log1p(imp_D)))   거리 평균 (여기서 쓰는 것)

    가중은 셋 다 P(D) 로 같고 차이는 '접는 시점' 뿐이다. 7개 윈도우 K=15 실측:

        공간        TV(차고지 재현)      W1(r) (duration 프로파일 재현)
        A1              0.0495                   0.0930
        A2              0.0490                   0.0820
        B               0.0549                   0.0479

    주 척도인 TV 는 사실상 무승부다 (A1 이 윈도우 4승 3패, 평균차 0.006 은
    윈도우별 변동 0.026~0.070 에 묻힌다). 갈리는 것은 두 번째다.

    r = Σimp30 / Σimp90 은 '30분 시점에 최종 피해의 몇 %가 이미 나타났나' 이며,
    r 이 작은 역(모집단의 27~44%, 평균 34.3%)이 곧 "지금 배차할까 관측하고
    기다릴까" 가 실제 결정이 되는 곳이다. 접으면 이 분포 재현도가 2배 나빠지고
    (B 6승 1패), 대표집합이 r 을 못 담으면 EVPI 가 0 으로 나와 Reviewer #1 (2)
    에 대한 답 자체가 성립하지 않는다.

    Jensen 에 의해 dist(평균) ≤ 평균(dist) 이므로 좌표를 접으면 역 간 분리가
    줄어든다. A2 처럼 접는 시점을 늦춰도 소용없고 거리를 평균해야 보존된다.
    재현: scratchpad/collapse.py

    C_agg 가 Kantorovich 비용이므로 확률계획 안정성 정리가 그대로 적용된다.
    K-means 에는 없는 보장이며 Reviewer #3 (3) 이 요구한 대안 축소 알고리즘이다.

    임팩트가 0 인 (역, duration) 은 raw 에 존재하지 않는다 — Omega_m 이 비면
    load_disruption_scenarios 가 건너뛴다. 결측이 아니라 '피해 없음' 이므로
    특징행렬에서는 0 벡터로 두되 블록은 만들지 않는다 (Z_D = 0 이라 목적함수
    기여가 없다). 윈도우당 12~18개 역이 여기 해당하며 패턴은 전부 (90,) 또는
    (60,90) 로 단조다. 이 역들을 버리면 '30분이면 무해하고 90분이면 심각한'
    사례가 사라지는데, 이는 비예견성이 실제로 구속되는 바로 그 경우다.

    반환 확률의 합은 1 이하이며, 모자란 만큼이 '아무도 발이 묶이지 않을'
    확률질량이다 (해당 블록의 Z 가 0 이므로 기대값은 정확하다).
    """
    if not scenarios:
        return []

    by_st = defaultdict(dict)
    for s in scenarios:
        by_st[s["station"]][s["duration"]] = s
    stations = sorted(by_st)
    n_st = len(stations)

    cols = sorted({c for per_dur in by_st.values() for s in per_dur.values()
                   for c in s.get("impact_by_station", {})})
    if not cols:
        reduced = []
        for st in stations:
            for dur, s in by_st[st].items():
                rep = s.copy()
                rep["probability"] = (DURATION_WEIGHT.get(dur, 1.0 / len(DURATIONS))
                                      / n_st)
                reduced.append(rep)
        _log(f"    impact_by_station missing: using all {len(reduced)} scenarios")
        return reduced

    col_ix = {c: i for i, c in enumerate(cols)}
    w_dur = np.array([DURATION_WEIGHT.get(d, 1.0 / len(DURATIONS))
                      for d in DURATIONS], dtype=float)
    w_dur = w_dur / w_dur.sum()

    # duration 블록마다 따로 표준화한 뒤 √P(D) 로 스케일해 이어붙인다.
    Zs = []
    for di, dur in enumerate(DURATIONS):
        X = np.zeros((n_st, len(cols)), dtype=float)
        for r, st in enumerate(stations):
            s = by_st[st].get(dur)
            if s is None:
                continue                      # 피해 없음 → 0 벡터 유지
            for c, v in s.get("impact_by_station", {}).items():
                if c in col_ix:
                    X[r, col_ix[c]] = v
        Z = (np.log1p(X) if CLUSTER_SPACE == "orig"
             else _methodology_transform_chain(X))
        Zs.append(np.sqrt(w_dur[di]) * Z)
    X_joint = np.hstack(Zs)

    # 공간확률 p(역) — uniform | frequency | exposure (Reviewer #3 (2))
    prior = scenario_station_prior(scenarios, wdata)
    p_st = np.array([prior.get(st, 0.0) for st in stations], dtype=float)
    if p_st.sum() <= 0:
        p_st = np.full(n_st, 1.0 / n_st)
    p_st = p_st / p_st.sum()

    k = int(min(max(k_stations, 1), n_st))
    km = KMeans(n_clusters=k, random_state=42, n_init=20).fit(X_joint)
    labels = km.labels_.astype(int)

    reduced, n_rep = [], 0
    for cid in range(k):
        members = np.flatnonzero(labels == cid)
        if members.size == 0:
            continue
        d = np.linalg.norm(X_joint[members] - km.cluster_centers_[cid], axis=1)
        i = int(members[int(np.argmin(d))])
        st = stations[i]
        n_rep += 1
        p_k = float(p_st[members].sum())      # 군집이 차지하는 공간확률
        for di, dur in enumerate(DURATIONS):
            s = by_st[st].get(dur)
            if s is None:
                continue                      # Z_D = 0 → 블록 불필요
            rep = s.copy()
            rep["probability"] = p_k * float(w_dur[di])
            rep["prob_station"] = p_k
            rep["block_station"] = st
            rep["cluster_id"] = int(cid)
            rep["cluster_size"] = int(members.size)
            rep["representative_id"] = rep["id"]
            rep["distance_to_joint_centroid"] = float(d.min())
            reduced.append(rep)

    n_full = n_rep * len(DURATIONS)
    p_sum = sum(r["probability"] for r in reduced)
    _log(f"    station-blocks/kmeans K={k} prior={SCENARIO_PRIOR}: "
         f"역 {n_st} -> {n_rep}, 블록 {len(reduced)}/{n_full} "
         f"(0임팩트 {n_full - len(reduced)}개 생략), "
         f"inertia={km.inertia_:.1f}, Σp={p_sum:.4f}")
    return reduced


def reduce_scenarios(scenarios, k_per_duration=K_PER_DURATION, wdata=None):
    """
    Per-duration K-means clustering so every duration yields exactly K reps:
      F2 raw matrix -> log1p -> z-score (whole-data fit),
      then KMeans(K, random_state=42, n_init=20) is run INDEPENDENTLY for
      each duration on its own sub-matrix.

    Representative = scenario closest to its cluster's feature-space centroid
    (k_clustering.py rule). This guarantees up to len(DURATIONS) * K reps and
    avoids the "empty (duration, cluster) cell" artifact that occurs when a
    single global K-means is sliced by duration.

    TSB_REDUCTION=station 이면 역 단위 블록 축소로 넘긴다.
    """
    if not scenarios:
        return []

    if REDUCTION == "station":
        return _reduce_scenarios_station_blocks(scenarios, K_STATIONS, wdata)

    X_raw, stations = _methodology_stage0_raw_matrix(scenarios)
    if not stations:
        reduced = []
        by_duration = defaultdict(list)
        for scenario in scenarios:
            by_duration[scenario["duration"]].append(scenario)
        for dur, dur_scenarios in sorted(by_duration.items()):
            dur_weight = DURATION_WEIGHT.get(dur, 1.0 / len(DURATIONS))
            for scenario in dur_scenarios:
                rep = scenario.copy()
                rep["probability"] = dur_weight / len(dur_scenarios)
                reduced.append(rep)
        _log(f"    impact_by_station missing: using all {len(reduced)} scenarios")
        return reduced

    # CLUSTER_SPACE="orig" 는 z-score 를 건너뛴다. 축소오차를 재는 공간과
    # 군집화 공간을 일치시키면 k-medoids 가 그 오차를 직접 최소화한다.
    X_final = (np.log1p(X_raw) if CLUSTER_SPACE == "orig"
               else _methodology_transform_chain(X_raw))

    # 공간확률 p(역) — SCENARIO_PRIOR 가 uniform 이면 None (기존 동작 유지)
    prior = None
    if SCENARIO_PRIOR != "uniform":
        prior = scenario_station_prior(scenarios, wdata)
        if prior:
            u = 1.0 / len(prior)
            _log(f"    prior={SCENARIO_PRIOR}: 역 {len(prior)}개, "
                 f"최대 {max(prior.values())/u:.2f}x / 최소 "
                 f"{min(prior.values())/u:.2f}x (균등 대비)")

    X_by_dur, scen_by_dur = {}, {}
    for dur in DURATIONS:
        dur_idx = [i for i, s in enumerate(scenarios) if s["duration"] == dur]
        if not dur_idx:
            continue
        scen_by_dur[dur] = [scenarios[i] for i in dur_idx]
        X_by_dur[dur] = X_final[dur_idx, :]

    k_by_dur, k_diag = resolve_k_per_duration(X_by_dur, k_per_duration)

    reduced = []
    for dur in DURATIONS:
        if dur not in scen_by_dur:
            continue

        dur_scenarios = scen_by_dur[dur]
        X_dur = X_by_dur[dur]

        k = min(k_by_dur[dur], len(dur_scenarios))
        dur_labels, medoids = _cluster_once(X_dur, k)

        reduced.extend(_methodology_select_representatives_within_duration(
            dur_scenarios, dur_labels, X_dur, k, dur, medoids=medoids,
            prior=prior
        ))

    dur_counts = ", ".join(
        f"{dur}:{sum(1 for s in scenarios if s['duration'] == dur)}"
        for dur in DURATIONS
    )
    rep_counts = ", ".join(
        f"{dur}:{sum(1 for s in reduced if s['duration'] == dur)}"
        for dur in DURATIONS
    )
    k_txt = ", ".join(f"{d}:{k_by_dur.get(d)}" for d in DURATIONS if d in k_by_dur)
    _log(f"    {CLUSTER_METHOD}/{CLUSTER_SPACE}/{CLUSTER_METRIC} "
         f"K_MODE={K_MODE} K=({k_txt}): raw {len(scenarios)} ({dur_counts}) "
         f"-> reps {len(reduced)} ({rep_counts})")
    if k_diag:
        _log(f"      {k_diag}")

    return reduced


# =============================================================================
# 6. Route Selection
# =============================================================================
def _ceil_step(val, step=DISPATCH_STEP):
    return int(math.ceil(float(val) / step) * step)

def _floor_step(val, step=DISPATCH_STEP):
    return int(math.floor(float(val) / step) * step)

def bus_demand_map(scenario):
    """
    버스가 실을 수 있는 수요 맵 {(승차역, 목적지): weight}.

    BUS_BOARD_MODE == "general" 이면 대기역 기준(strand_score),
    아니면 원출발역 기준(pair_score). select_routes / generate_bus_trips /
    노선 생성이 모두 이 맵 하나를 공유해야 후보 노선과 MILP 인덱스가 어긋나지 않는다.
    """
    if BUS_BOARD_MODE == "general":
        s = scenario.get("strand_score")
        if s:
            return s
        # strand_score 를 못 읽은 시나리오는 조용히 구버전 기준으로 되돌린다
    return scenario["pair_score"]


def build_scenario_route_library(scenario, start_h, depots, bus_tt, fallback_lib):
    """Return the predefined local-plus-express route library.

    The paper execution bundle deliberately excludes legacy on-the-fly route
    generation.  Routes are produced once by Stage 3 and then treated as the
    finite candidate library in both Stage 1 and Stage 2.
    """
    return fallback_lib, None


def select_routes(scenario, depots, route_library, bus_tt):
    """
    Omega_m 기반 수요맵으로 route 선별.
    기존 파일 3의 select_routes와 동일 구조, scoring 기준만 교체.
    """
    pair_score   = bus_demand_map(scenario)
    dispatch_end = scenario["DISPATCH_END_MIN"]
    end_min      = WINDOW_END_MIN

    selected     = {}

    for rk, rd in route_library.items():
        total = rd["cum"][-1]
        if total <= 0:
            continue

        origin = rd["stops"][0]
        dest   = rd["stops"][-1]

        # depot 실현 가능성 확인
        feasible = {}
        for depot in depots:
            dh_out = get_travel_time(bus_tt, depot, origin)
            if dh_out is None or dh_out > DEPOT_ACCESS_BOUND_MIN:
                continue
            # Eq. (22) [eq:start_time_grid]: t_start in {t0+eta, t0+2eta, ...} with
            # t0+eta <= t_start - mu <= t0+max_tau+delta (= dispatch_end) and, for the
            # complete service, t_end <= T (checked in generate_bus_trips).
            earliest = DISRUPT_OFFSET + _ceil_step(DISPATCH_STEP + dh_out)
            latest   = DISRUPT_OFFSET + _floor_step(
                min(dispatch_end + dh_out, end_min) - DISRUPT_OFFSET)
            if latest < earliest:
                continue
            feasible[depot] = {
                "dh_out": int(dh_out),
                "earliest": int(earliest), "latest": int(latest),
            }

        # 노선당 차고지 상한 (Reviewer #3 (4) 후보생성 감도, #3 (8) 전처리).
        # deadhead <= 90분이 유일한 필터라 노선당 평균 9개 차고지가 붙는데,
        # 후보 (노선,차고지) 의 deadhead 중앙값이 47분이다. 배차는 t=20 부터라
        # 그 버스는 t=67 에야 태우기 시작하고, 30분 장애는 t=40 에 끝나 있다.
        # 계산 낭비이기 전에 운영상 성립하지 않는 배차다.
        # 0 이면 무제한(기존 동작). 동률은 이름순으로 깨 결정론을 유지한다.
        if MAX_DEPOTS_PER_ROUTE > 0 and len(feasible) > MAX_DEPOTS_PER_ROUTE:
            keep = sorted(feasible, key=lambda d: (feasible[d]["dh_out"], d))
            feasible = {d: feasible[d] for d in keep[:MAX_DEPOTS_PER_ROUTE]}

        if not feasible:
            continue

        # pair coverage
        covered = []
        stops, cum = rd["stops"], rd["cum"]
        for bi, bs in enumerate(stops[:-1]):
            for ai in range(bi+1, len(stops)):
                ds = stops[ai]
                score = pair_score.get((bs, ds), 0.0)
                if score <= QUEUE_EPS:
                    continue
                covered.append({
                    "board_station": bs, "dest": ds, "score": score,
                    "board_idx": bi, "alight_idx": ai,
                    "travel_time": cum[ai] - cum[bi],
                })

        if not covered:
            continue

        route_score = sum(c["score"] for c in covered)
        selected[rk] = {"data": rd, "score": route_score, "depots": feasible, "pairs": covered}

    # Sec. 3.4.4: fully enumerated finite candidate set. Every library route with an
    # admissible depot (Eq. (21)) is kept; a route that covers no positive-demand
    # pair has no boarding variable, so Eq. (38) fixes its trips at zero and it is
    # omitted as a structural zero.
    return {rk: selected[rk] for rk in sorted(selected)}


# =============================================================================
# 7. Bus Trip Generation (단방향 + 240분 잘림)
# =============================================================================
def generate_bus_trips(scenario, selected_routes, route_library):
    """
    단방향 trip 기반. 240분 초과 정류장은 잘림.
    왕복 시도: forward leg → (시간 남으면) reverse leg.
    DISPATCH_END_MIN 이후 출발 trip은 생성 안 함.
    """
    pair_score   = bus_demand_map(scenario)
    all_trips    = []

    for rk, sel in selected_routes.items():
        rd        = sel["data"]
        stops_fwd = rd["stops"]
        cum_fwd   = rd["cum"]

        for depot, di in sel["depots"].items():
            dh_out    = di["dh_out"]
            run_start = di["earliest"]

            while run_start <= di["latest"]:
                dispatch_start = run_start - dh_out
                if dispatch_start < DISPATCH_START_MIN:
                    run_start += DISPATCH_STEP
                    continue

                # --- Forward leg: 240분까지 잘림 ---
                fwd_stop_times = [run_start + c for c in cum_fwd]
                # A candidate is a complete one-way service, never an
                # arbitrary prefix created by the planning boundary.
                if (len(fwd_stop_times) < 2 or
                        fwd_stop_times[-1] > WINDOW_END_MIN):
                    run_start += DISPATCH_STEP
                    continue
                fwd_stops_complete = list(zip(stops_fwd, fwd_stop_times))

                fwd_end = fwd_stop_times[-1]

                # Forward OD coverage
                fwd_od = []
                for bi, (bs, bt) in enumerate(fwd_stops_complete[:-1]):
                    for ai in range(bi+1, len(fwd_stops_complete)):
                        ds, at = fwd_stops_complete[ai]
                        if (bs, ds) not in [tuple(x) for x in [(p["board_station"], p["dest"]) for p in sel["pairs"]]]:
                            continue
                        score = pair_score.get((bs, ds), 0.0)
                        if score <= QUEUE_EPS: continue
                        fwd_od.append({
                            "board_station": bs, "dest": ds,
                            "board_idx": bi, "alight_idx": ai,
                            "board_time": bt, "alight_time": at,
                            "travel_time": at - bt,
                            "leg_index": 0,
                        })

                # --- Reverse leg: 시간 남으면, 240분까지 잘림 ---
                all_od = fwd_od
                if not all_od:
                    run_start += DISPATCH_STEP
                    continue

                # Inter-trip vehicle circulation is handled by chain variables.
                service_end = fwd_end
                service_end_station = stops_fwd[-1]
                resource_end = service_end

                # Build legs
                fwd_leg = {
                    "leg_index": 0, "route_key": rk,
                    "direction": rd["direction"],
                    "stops": [s for s,t in fwd_stops_complete],
                    "stop_times": [t for s,t in fwd_stops_complete],
                    "start_time": run_start, "end_time": fwd_end,
                    "od_coverage": fwd_od,
                }
                legs = [fwd_leg]

                trip_id = f"B_{depot}_{rk}_{run_start}"
                all_trips.append({
                    "TripID":        trip_id,
                    "Depot":         depot,
                    "RouteKey":      rk,
                    "Route":         rd["route"],
                    "RouteType":     rd["route_type"],
                    "Direction":     rd["direction"],
                    "RouteOrigin":   stops_fwd[0],
                    "RouteDest":     stops_fwd[-1],
                    "ServiceEndStation": service_end_station,
                    "DeadheadOut":   dh_out,
                    "RunStart":      run_start,
                    "RunEnd":        service_end,
                    "ServiceEnd":    service_end,
                    "DispatchStart": dispatch_start,
                    "ResourceEnd":   resource_end,
                    "RouteStops":    [s for s,t in fwd_stops_complete],
                    "StopTimeList":  [t for s,t in fwd_stops_complete],
                    "ODCoverage":    all_od,
                    "Legs":          legs,
                    "NumLegs":       1,
                })

                run_start += DISPATCH_STEP

    bus_trips = {t["TripID"]: t for t in all_trips}
    return bus_trips


# =============================================================================
# 8. Stage 2 Cache (N_ω 기반 인덱스 축소)
# =============================================================================
def build_stage2_cache(scenario, wdata, bus_trips,
                       bus_stop_support=None, support_only=False,
                       include_bus_load_keys=True):
    """
    q, w 변수 인덱스를 N_ω (OD별 실제 경유 가능 역)으로 축소.

    N_ω = {origin(ω)}
        ∪ {canceled_trains 중 ω 구간 정차역}
        ∪ {버스 trip stops 중 ω ODCoverage에 포함된 역}
        ∪ {dest(ω)}

    ── 열 생성용 인자 (DESIGN_column_generation.md §2, §9-B) ──────────────
    N_ω 가 bus_trips 에 의존하므로, trip 부분집합으로 이 함수를 다시 부르면
    q/w 행 집합 자체가 줄어든다. 그러면 RMP 가 전체 LP 의 제한 문제가 아니게
    되어 열 생성의 전제가 깨진다.

      support_only=True   N_ω 만 만들고 즉시 반환 (흐름변수 키를 만들지 않아 싸다)
      bus_stop_support    주어지면 버스 정차역을 이것으로 대체한다

    사용법:

        support = build_stage2_cache(scen, wdata, 전체_trips, support_only=True)["N_omega"]
        cache   = build_stage2_cache(scen, wdata, 부분집합,  bus_stop_support=support)

    같은 코드 경로로 만들어지므로 규칙이 어긋날 수 없다. 기본 인자에서는
    기존 동작과 완전히 동일하다.
    """
    Omega_m   = scenario["Omega_m"]
    Omega_Map = wdata["Omega_Map"]
    Gamma     = wdata["Gamma"]
    T         = len(Gamma)

    canceled_trains    = scenario["canceled_trains"]
    train_active_times = wdata["train_active_times"]
    station_of         = wdata["station_of"]

    # --- N_ω 구성 ---
    # 목표: File 1과 동일한 환승 탐색 공간을 보장하되, 불필요한 역을 제외하여
    #        변수 수(q, w, b, a)를 줄이고 풀이 속도를 확보한다.
    #
    # 방법: OD(om)의 승객이 실제로 이용 가능한 역만 수집.
    #   1) origin, dest는 무조건 포함
    #   2) 모든 열차가 정차하는 역 중, origin 또는 dest를 지나는 열차의
    #      정차역을 모두 추가 (환승 가능역 포함)
    #      → 열차 A가 origin만 지나고, 열차 B가 dest만 지나더라도
    #        A와 B가 공통으로 정차하는 환승역이 포함됨
    #   3) 버스 정차역 추가
    #
    # 이렇게 하면 "origin을 지나는 열차의 모든 정차역" ∪ "dest를 지나는 열차의 모든 정차역"
    # 이므로 환승이 필요한 경우에도 경유 가능한 역이 빠지지 않는다.
    # (열차가 전혀 지나지 않는 역은 queue/walking 변수를 만들어도 항상 0이므로 제외해도 무방)

    N_omega = defaultdict(set)

    # 사전 계산: 각 역을 지나는 열차 목록 & 각 열차의 정차역 집합
    trains_passing_station = defaultdict(set)   # station → {train_ids}
    stations_of_train = {}                       # train → {stations}
    for v in wdata["Vm"]:
        sof = station_of.get(v, {})
        st_set = set(sof.values())
        stations_of_train[v] = st_set
        for s in st_set:
            trains_passing_station[s].add(v)

    Omega_m_set = set(Omega_m)
    for om in Omega_m:
        if om not in Omega_Map: continue
        origin, dest = Omega_Map[om]
        N_omega[om].add(origin)
        N_omega[om].add(dest)

        # origin을 지나는 열차들의 모든 정차역
        for v in trains_passing_station.get(origin, set()):
            N_omega[om].update(stations_of_train[v])
        # dest를 지나는 열차들의 모든 정차역
        for v in trains_passing_station.get(dest, set()):
            N_omega[om].update(stations_of_train[v])

    # Passenger non-anticipativity can place an OD in every duration branch
    # even when only another branch is physically affected.  Preserve every
    # station used by that OD's positive normal trajectory so normal-fixed
    # b/a decisions, q/w states, and their conservation rows have matching
    # variable support in each branch.
    for om, stations in scenario.get("required_station_support", {}).items():
        if om in Omega_m_set:
            N_omega[om].update(
                station for station in stations if station is not None)

    # --- 버스 커버리지 → om 매핑 -------------------------------------------
    # BUS_BOARD_MODE == "general":
    #   버스가 역 j 에서 태워 역 d 에 내려주는 구간은, 목적지가 d 인 모든 OD 중
    #   j 에서 대기 가능한(j ∈ N_ω) 승객을 실을 수 있다. 열차 쪽은 이미 N_ω 의
    #   모든 역에서 승차를 허용하므로 이렇게 해야 두 수단이 대칭이 된다.
    #   (표본 40개 시나리오 기준 잔류 수요의 49.3% 가 원출발역이 아닌 중간역에 있다)
    # BUS_BOARD_MODE == "origin" (구버전):
    #   om 을 board_station 에서 역산 → OD 원출발역에서만 승차 가능.
    oms_by_dest = defaultdict(list)
    if BUS_BOARD_MODE == "general":
        for om in Omega_m:
            if om in Omega_Map:
                oms_by_dest[Omega_Map[om][1]].append(om)

    # 승차 가능 판정은 '열차로 도달 가능한 역'(= 버스 정차역을 넣기 전의 N_ω)을
    # 기준으로 한 번만 내리고 고정한다. 아래에서 N_ω 에 버스 정차역을 더하는데,
    # 그 뒤에 다시 판정하면 om 집합이 커져서 q/w 변수가 없는 역에 a 변수가 생긴다.
    base_N = {om: frozenset(N_omega[om]) for om in Omega_m}

    def _oms_for(cov):
        if BUS_BOARD_MODE != "general":
            om = f"{cov['board_station']}_{cov['dest']}"
            return (om,) if om in Omega_m_set else ()
        j = cov["board_station"]
        return tuple(om for om in oms_by_dest.get(cov["dest"], ())
                     if j in base_N.get(om, ()))

    # 버스 커버리지별 om 을 한 번 확정해 두고 이후 전 단계가 이것만 쓴다
    cov_oms = {}
    for tid, trip in bus_trips.items():
        for ci, cov in enumerate(trip["ODCoverage"]):
            cov_oms[(tid, ci)] = _oms_for(cov)

    # 버스 정차역 추가
    # bus_stop_support 가 주어지면 전체 후보로 미리 구한 지지집합을 쓴다.
    # 그래야 trip 부분집합을 넣어도 q/w 행이 그대로다 (열 생성의 전제).
    if bus_stop_support is None:
        for tid, trip in bus_trips.items():
            legs_by_idx = {leg["leg_index"]: leg for leg in trip["Legs"]}
            for ci, cov in enumerate(trip["ODCoverage"]):
                leg = legs_by_idx.get(cov["leg_index"])
                if leg is None:
                    continue
                for om in cov_oms[(tid, ci)]:
                    N_omega[om].update(leg["stops"])
    else:
        for om, stops in bus_stop_support.items():
            N_omega[om].update(stops)

    if support_only:
        # 흐름변수 키를 만들지 않고 N_ω 만 돌려준다 (전체 후보로 1회 호출용).
        return {"N_omega": {om: set(s) for om, s in N_omega.items()}}

    # --- 변수 인덱스 ---
    trip_ids = sorted(bus_trips)
    trains_at  = wdata["trains_at"]

    b_var_keys, a_var_keys, l_var_keys = [], [], []
    boarding_by_od_t_j   = defaultdict(list)
    alighting_by_od_t_j  = defaultdict(list)
    od_trips             = defaultdict(set)
    trip_od_list         = defaultdict(list)

    seen_b, seen_a, seen_l = set(), set(), set()

    # Strict passenger non-anticipativity distinguishes service withdrawal
    # from physical passenger clearance.  A clearance can occur at a scheduled
    # departure minute inside a dwell, which is absent from the ordinary
    # arrival-only event support.  Bus-only scenarios carry no such metadata
    # and retain their historical cache shape and boarding support.
    termination_events = {}
    for raw_vehicle, raw_event in scenario.get(
            "termination_alight_event_by_train", {}).items():
        vehicle = str(raw_vehicle)
        if isinstance(raw_event, dict):
            minute = int(raw_event["minute"])
            station = str(raw_event["station"])
        else:
            minute, station = raw_event
            minute, station = int(minute), str(station)
        if vehicle not in train_active_times:
            raise ValueError(
                f"termination train {vehicle} has no active load stream")
        if minute not in train_active_times[vehicle]:
            raise ValueError(
                f"termination event ({vehicle}, {minute}, {station}) lies "
                "outside its active load stream")
        termination_events[vehicle] = (minute, station)
    termination_event_keys = {
        (vehicle, minute, station)
        for vehicle, (minute, station) in termination_events.items()
    }
    raw_termination_support = scenario.get(
        "required_termination_alight_support")
    if raw_termination_support is None:
        termination_action_support = set(termination_event_keys)
    else:
        termination_action_support = {
            (str(vehicle), int(minute), str(station))
            for vehicle, minute, station in raw_termination_support
        }
    preserve_common_rail_support = bool(
        scenario.get("preserve_common_rail_event_support", False))

    # 버스 boarding/alighting
    for tid, trip in bus_trips.items():
        for ci, cov in enumerate(trip["ODCoverage"]):
            bt, bj = cov["board_time"], cov["board_station"]
            at, aj = cov["alight_time"], cov["dest"]

            for om in cov_oms[(tid, ci)]:
                bkey = (om, tid, bt, bj)
                if bkey not in seen_b:
                    seen_b.add(bkey); b_var_keys.append(bkey)
                    boarding_by_od_t_j[(om, bt, bj)].append(tid)

                akey = (om, tid, at, aj)
                if akey not in seen_a:
                    seen_a.add(akey); a_var_keys.append(akey)
                    alighting_by_od_t_j[(om, at, aj)].append(tid)

                # Exact CG reconstructs these bus-load streams lazily from
                # trip_od_list.  The legacy/default path still materializes
                # every bus l key so existing callers remain unchanged.
                if include_bus_load_keys:
                    for t in range(bt, at+1):
                        lkey = (om, tid, t)
                        if lkey not in seen_l:
                            seen_l.add(lkey); l_var_keys.append(lkey)

                od_trips[om].add(tid)
                trip_od_list[tid].append({
                    "omega": om, "board_t": bt, "board_j": bj,
                    "alight_t": at, "alight_j": aj,
                    "board_idx": cov["board_idx"], "alight_idx": cov["alight_idx"],
                })

    # 열차 boarding/alighting (Omega_m 범위)
    I_vts      = wdata["I_vts"]
    valid_keys = wdata["valid_keys"]
    Vm         = wdata["Vm"]

    # valid_keys를 station 기준으로 사전 인덱싱 (이중 루프 제거)
    valid_keys_by_station = defaultdict(list)
    for vts in valid_keys:
        valid_keys_by_station[vts[2]].append(vts)

    # N_omega의 모든 역 집합 (전체 합집합)
    all_N_stations = set()
    for om in Omega_m:
        all_N_stations.update(N_omega[om])

    train_l_keys = []
    for om in Omega_m:
        if om not in Omega_Map: continue
        N_om_set = N_omega[om]
        for v in Vm:
            if v not in train_active_times: continue
            for t in train_active_times[v]:
                lkey = (om, v, t)
                if lkey not in seen_l:
                    seen_l.add(lkey); train_l_keys.append(lkey)
        for s in N_om_set:
            for (v, t, _) in valid_keys_by_station.get(s, []):
                bkey = (om, v, t, s)
                service_withdrawn = int(scenario.get("theta", {}).get(
                    (str(v), int(t)), 0))
                if ((preserve_common_rail_support
                        or not termination_events or not service_withdrawn)
                        and (preserve_common_rail_support
                             or (str(v), int(t), str(s))
                             not in termination_event_keys)
                        and bkey not in seen_b):
                    seen_b.add(bkey); b_var_keys.append(bkey)
                    boarding_by_od_t_j[(om, t, s)].append(v)
                akey = (om, v, t, s)
                if akey not in seen_a:
                    seen_a.add(akey); a_var_keys.append(akey)
                    alighting_by_od_t_j[(om, t, s)].append(v)

    # Materialize one alighting-only action for each forced clearance.  It is
    # authoritative even when the event minute is not in ``valid_keys``.
    for vehicle, minute, station in termination_action_support:
        for om in Omega_m:
            if om not in Omega_Map:
                continue
            if station not in N_omega[om]:
                raise ValueError(
                    f"termination station {station} is missing from "
                    f"N_omega[{om}]")
            akey = (om, vehicle, minute, station)
            if akey not in seen_a:
                seen_a.add(akey); a_var_keys.append(akey)
                alighting_by_od_t_j[(om, minute, station)].append(vehicle)

    l_var_keys = l_var_keys + train_l_keys

    # q, w, od 인덱스 (N_ω 기반) — list comprehension으로 생성
    q_var_keys = [
        (om, j, t)
        for om in Omega_m if om in Omega_Map
        for j in N_omega[om]
        for t in Gamma
    ]
    w_var_keys = [
        (om, j, t)
        for om in Omega_m if om in Omega_Map
        for j in N_omega[om] if j != Omega_Map[om][1]
        for t in Gamma
    ]
    od_var_keys = [
        (om, t)
        for om in Omega_m if om in Omega_Map
        for t in Gamma
    ]

    result = {
        "trip_ids":    trip_ids,
        "Gamma":       Gamma,
        "T":           T,
        "N_omega":     dict(N_omega),
        "b_var_keys":  b_var_keys,
        "a_var_keys":  a_var_keys,
        "l_var_keys":  l_var_keys,
        "q_var_keys":  q_var_keys,
        "w_var_keys":  w_var_keys,
        "od_var_keys": od_var_keys,
        "boarding_by_od_t_j":  dict(boarding_by_od_t_j),
        "alighting_by_od_t_j": dict(alighting_by_od_t_j),
        "od_trips":    dict(od_trips),
        "trip_od_list":dict(trip_od_list),
    }
    if termination_events:
        result["termination_alight_event_by_train"] = termination_events
    if termination_action_support:
        result["required_termination_alight_support"] = tuple(sorted(
            termination_action_support))
    return result


def build_block_trips(scenarios, start_h, depots, bus_tt, route_lib):
    """
    역 블록의 30/60/90 이 **같은 trip 집합**을 쓰도록 만든다. y 공유의 전제조건.

    시나리오마다 따로 노선을 고르면 duration 별로 trip id 가 달라져 공유할 수
    없다. 최장 duration 으로 한 번만 생성한 뒤 duration 별 배차창으로 거른다.
    select_routes / generate_bus_trips 가 둘 다 scenario["DISPATCH_END_MIN"] 만
    배차상한으로 쓰므로, 이 필터가 개별 생성과 정확히 같은 창을 준다.

    다만 **노선 선택은 최장 duration 의 수요 기준**이 된다. 짧은 duration 은
    자기 수요로 골랐을 때보다 노선이 많아지므로 흐름변수가 늘어난다. 이것이
    비예견성의 계산 비용이다.

    반환 {sid: (selected_routes, bus_trips)}. block_station 이 없으면 빈 dict
    (= duration 축소 경로 → 호출부가 기존대로 시나리오별 생성).
    """
    blocks = defaultdict(list)
    for s in scenarios:
        st = s.get("block_station")
        if st is not None:
            blocks[st].append(s)
    if not blocks:
        return {}

    out = {}
    for st, block in blocks.items():
        # A representative-station union may contain a probability-zero,
        # resource-only longest-duration branch.  Route coverage must still be
        # selected from the longest branch that carries real disrupted demand,
        # while the physical trip universe must extend through the longest
        # dispatch horizon in the block.  With three ordinary raw branches this
        # is exactly the historical longest-duration behavior.
        demand_branches = [
            scenario for scenario in block
            if (not scenario.get("zero_impact_branch", False) and
                bus_demand_map(scenario))
        ]
        demand_template = max(
            demand_branches or block, key=lambda s: s["duration"])
        physical_template = demand_template.copy()
        physical_template["DISPATCH_END_MIN"] = max(
            int(scenario["DISPATCH_END_MIN"]) for scenario in block)
        lib, _ = build_scenario_route_library(
            physical_template, start_h, depots,
                                               bus_tt, route_lib)
        sel = select_routes(physical_template, depots, lib, bus_tt)
        trips = generate_bus_trips(physical_template, sel, lib)
        for s in block:
            end = s["DISPATCH_END_MIN"]
            out[s["id"]] = (sel, {tid: tr for tid, tr in trips.items()
                                  if tr["DispatchStart"] <= end})
    return out


def _build_shared_y(model, scenarios, buffer=DISRUPT_BUFFER, vtype=GRB.BINARY,
                    active_columns=None, column_ub=None):
    """
    역 블록별 비예견성 배차변수. 반환 {sid: {tid: var}}.

    같은 역의 30/60/90 은 '역은 실현, 지속시간은 미실현' 정보집합을 이룬다.
    배차시각 tau 에서 아직 구분할 수 없는 지속시간들이 **하나의 변수를 공유**
    한다 (등식 제약이 아니라 변수 자체를 공유 — stage2_adaptive 와 같은 방식).

        tau <= start+30      -> (30, 60, 90)   아직 아무것도 모름
        tau <= start+60      -> (60, 90)       30분 종료를 관측
        그 외                -> (90,)          60분 종료까지 관측

    buffer > 0 이면 종료를 관측한 뒤의 배차가 생기며, 그 구간은 시나리오별
    독립 변수가 된다 (info_cells 가 (30,) 같은 단일 셀로 분리).

    block_station 이 없으면 빈 dict → 호출부가 시나리오별 독립 변수를 만든다
    (기존 완전정보 리코스).
    """
    blocks = defaultdict(list)
    for s in scenarios:
        st = s.get("block_station")
        if st is not None:
            blocks[st].append(s)
    if not blocks:
        return {}

    # ``active_columns`` is used only by exact block column generation.  A key
    # is (block_station, trip_id, information_cell).  Keeping this filter here,
    # where the shared variables are actually created, prevents a scenario-wise
    # trip subset from accidentally breaking nonanticipativity.  ``None`` keeps
    # the historical behaviour byte-for-byte at the model level.
    active_columns = (None if active_columns is None else {
        (str(k), str(tid), tuple(cell)) for k, tid, cell in active_columns
    })
    column_ub = column_ub or {}

    out, n_shared, n_indep = {}, 0, 0
    for st, block in blocks.items():
        durs = tuple(sorted(s["duration"] for s in block))
        ds = block[0]["DISRUPT_START_T"]
        trips = {}
        for s in block:                      # 최장 duration 이 상위집합
            trips.update(s.get("bus_trips", {}))
            n_indep += len(s.get("bus_trips", {}))

        var = {}
        for tid, tr in trips.items():
            for cell in info_cells(tr["DispatchStart"], ds, durs, buffer):
                key = (str(st), str(tid), tuple(cell))
                if active_columns is not None and key not in active_columns:
                    continue
                ub = float(column_ub.get(key, 1.0))
                var[(tid, cell)] = model.addVar(
                    lb=0.0, ub=ub, vtype=vtype,
                    name=f"y[{st},{tid}|{'-'.join(map(str, cell))}]")
        n_shared += len(var)

        # 셀은 가용 duration 을 분할하므로 각 D 는 정확히 한 셀에만 속한다.
        for s in block:
            D = s["duration"]
            out[s["id"]] = {tid: v for (tid, cell), v in var.items() if D in cell}

    _log(f"    [비예견성] 역블록 {len(blocks)}개 · 배차변수 {n_shared:,} "
         f"(독립이면 {n_indep:,}, {1 - n_shared / max(n_indep, 1):.1%} 감소)")
    return out


# =============================================================================
# 9. Two-Stage Stochastic MILP
# =============================================================================
def _aggregate_bus_load_groups(cache, trip_ids=None):
    """Return one mass-balance event stream per ``(trip, omega)``.

    ``trip_od_list`` is coverage-indexed, while the load variable is indexed
    only by ``(omega, trip, minute)``. In general boarding mode several
    coverage records can therefore refer to the same load stream. The legacy
    formulation wrote one recurrence per coverage record, imposing conflicting
    equalities on the same load variable. Exact-CG uses this helper to
    deduplicate actual b/a keys and write one continuous recurrence per stream.
    """
    allowed = None if trip_ids is None else {str(tid) for tid in trip_ids}
    raw = {}
    for tid, od_list in cache.get("trip_od_list", {}).items():
        if allowed is not None and str(tid) not in allowed:
            continue
        for od_info in od_list:
            om = od_info["omega"]
            bt, at = int(od_info["board_t"]), int(od_info["alight_t"])
            bj, aj = od_info["board_j"], od_info["alight_j"]
            group = raw.setdefault((tid, om), {
                "first": min(bt, at), "last": max(bt, at),
                "board": defaultdict(set), "alight": defaultdict(set),
            })
            group["first"] = min(group["first"], bt, at)
            group["last"] = max(group["last"], bt, at)
            group["board"][bt].add((om, tid, bt, bj))
            group["alight"][at].add((om, tid, at, aj))

    groups = defaultdict(dict)
    for (tid, om), group in raw.items():
        groups[tid][om] = {
            "times": tuple(range(group["first"], group["last"] + 1)),
            "board": {
                t: tuple(sorted(keys, key=repr))
                for t, keys in group["board"].items()
            },
            "alight": {
                t: tuple(sorted(keys, key=repr))
                for t, keys in group["alight"].items()
            },
        }
    return {tid: dict(by_om) for tid, by_om in groups.items()}


def solve_milp(env, depots, scenarios, all_caches, wdata,
               lp_relax=False, duals_out=None, active_columns=None,
               column_ub=None, depot_row_keys=None, extract_solution=True,
               aggregate_bus_load_balance=False):
    """
    Stage 1: x[d]
    Stage 2: y[v], b, a, l, q, w
    목적함수: min (1/T) Σ_s p_s [Q + L + W]

    lp_relax : x, y 를 연속으로 완화한다. 열 생성의 RMP 용.
    duals_out: dict 를 주면 결합제약의 **듀얼 값**을 담는다
               (DESIGN_column_generation.md §1). 핸들이 아니라 float 다 —
               model.dispose() 후에는 핸들이 무효이므로 반환 전에 값으로 뽑는다.
                 duals_out["q"][(sid, om, j, t)]      큐균형   π
                 duals_out["w"][(sid, om, j, t)]      도보     ρ
                 duals_out["depot"][(sid, d, tau)]    차고지   μ
               lp_relax=False 이면 MIP 라 듀얼이 없으므로 빈 dict 가 된다.
    active_columns:
               None 이면 기존처럼 모든 trip-cell 을 만든다. 집합을 주면
               (block_station, trip_id, information_cell) 단위로 버스의
               y/b/a/l 및 국소제약만 제한한다. q/w/od, 열차 변수와 그 행은
               all_caches 의 full-pool 지지집합을 그대로 유지한다. 따라서
               부분집합마다 cache 를 재생성해서는 안 된다.
    column_ub: 검증용 선택 인자. column key 별 y 상한(기본 1.0).
    depot_row_keys:
               exact CG에서 고정할 full-pool (sid, depot, minute) 행 집합.
               활성 trip이 아직 없어도 0 <= x[d] 행을 만들어 μ=0으로
               full dual에 확장할 수 있게 한다. None이면 기존 행 생성 방식.
    extract_solution:
               False이면 최적값/듀얼/통계만 반환하고 큰 해 추출을 생략한다.
               exact CG의 반복 RMP용이며 기본값은 기존과 동일한 True.
    aggregate_bus_load_balance:
               True이면 (omega, trip)별 단일 재귀식에 시간별 unique b/a를
               합산한다. exact CG가 사용한다. 기본 False는 과거 duration
               경로의 모형과 결과를 보존한다.

    기본 인자에서는 기존 동작과 완전히 동일하다.
    """
    _collect = duals_out is not None
    if _collect:
        duals_out.setdefault("q", {})
        duals_out.setdefault("w", {})
        duals_out.setdefault("depot", {})
    if not scenarios or not depots:
        return {d: 0 for d in depots}, "NO_SCENARIO", {}, None

    t_func_start = time.time()
    model = gp.Model("TwoStage_BusBridging", env=env)
    model.Params.MIPGap    = MIP_GAP
    model.Params.TimeLimit = TIME_LIMIT
    if lp_relax and active_columns is not None:
        # Exact-CG certification uses RC_EPS=1e-7, so the LP dual tolerance
        # must be tighter than the certificate threshold (Gurobi default is
        # 1e-6).  This branch never changes the historical solve path.
        model.Params.OptimalityTol = float(os.getenv("TSB_CG_OPT_TOL", _LEGACY["cg_optimality_tol"]))
        model.Params.FeasibilityTol = float(os.getenv("TSB_CG_FEAS_TOL", _LEGACY["cg_feasibility_tol"]))
    # ==========================================
    # 🚀 Gurobi 휴리스틱 몰빵 세팅 추가
    # ==========================================
    # 기본값은 기존 그대로다 (duration 경로 결과 보존).
    # station 경로는 모델이 1.7배로 커지고 근 LP 가 병목이라 반대 튜닝이 필요하다:
    #   MIPFocus=1 + Heuristics=0.5 는 '해를 하나라도 찾자' 설정이라
    #   갭을 닫아야 하는 상황에서는 시간의 절반을 낭비한다.
    #   TSB_MIP_FOCUS=3 TSB_HEURISTICS=0.05 TSB_CROSSOVER=0 TSB_NODEFILE=8
    model.Params.MIPFocus = int(os.getenv("TSB_MIP_FOCUS", _LEGACY["mip_focus"]))
    model.Params.Heuristics = float(os.getenv("TSB_HEURISTICS", _LEGACY["heuristics"]))
    _method = int(os.getenv("TSB_LP_METHOD", _LEGACY["lp_method"]))
    if _method >= 0:
        model.Params.Method = _method          # 2 = barrier
    _cross = int(os.getenv("TSB_CROSSOVER", _LEGACY["crossover"]))
    if _cross >= 0:
        model.Params.Crossover = _cross        # 0 = crossover 생략
    _nodefile = float(os.getenv("TSB_NODEFILE", _LEGACY["nodefile"]))
    if _nodefile > 0:
        model.Params.NodefileStart = _nodefile  # GB. B&B 트리를 디스크로
    # ==========================================

    sigma = 2  # 환승 대기 시간 (분)

    # Stage 1
    _vt_x = GRB.CONTINUOUS if lp_relax else GRB.INTEGER
    _vt_y = GRB.CONTINUOUS if lp_relax else GRB.BINARY
    x = {d: model.addVar(lb=0, ub=MAX_TOTAL_BUSES, vtype=_vt_x, name=f"x[{d}]")
         for d in depots}
    model.addConstr(gp.quicksum(x[d] for d in depots) <= MAX_TOTAL_BUSES, "fleet_budget")

    obj_terms    = []
    scen_vars    = {}
    Omega_Map    = wdata["Omega_Map"]
    trains_at    = wdata["trains_at"]
    station_of   = wdata["station_of"]
    train_active_times = wdata["train_active_times"]
    train_dest_map     = wdata["train_dest_map"]
    train_dwells       = wdata["train_dwells"]
    valid_keys         = wdata["valid_keys"]
    no_board_keys      = wdata["no_board_keys"]
    Vm                 = wdata["Vm"]
    Gamma              = wdata["Gamma"]
    START_MIN          = wdata["START_MIN"]
    CumD               = wdata["CumD"]

    # exact CG key를 한 번 정규화한다.  기본값 None은 기존 전체 모델이다.
    _active_columns = (None if active_columns is None else {
        (str(k), str(tid), tuple(cell)) for k, tid, cell in active_columns
    })
    _column_ub = {
        (str(k), str(tid), tuple(cell)): float(ub)
        for (k, tid, cell), ub in (column_ub or {}).items()
    }
    _depot_row_keys = (None if depot_row_keys is None else {
        (str(sid), str(d), int(tau)) for sid, d, tau in depot_row_keys
    })
    _depot_rows_by_sid_d = defaultdict(set)
    if _depot_row_keys is not None:
        for row_sid, row_d, tau in _depot_row_keys:
            _depot_rows_by_sid_d[(row_sid, row_d)].add(tau)

    # 역 블록이면 배차변수를 정보집합 단위로 미리 만든다 (비예견성).
    # duration 축소 경로에서는 빈 dict 이므로 아래에서 기존대로 시나리오별 생성.
    y_shared = _build_shared_y(
        model, scenarios, buffer=DISRUPT_BUFFER, vtype=_vt_y,
        active_columns=_active_columns, column_ub=_column_ub)

    for scenario in scenarios:
        sid  = scenario["id"]
        prob = scenario.get("probability", 1.0 / len(scenarios))
        cache = all_caches.get(sid)
        if (cache is None or
                (_active_columns is None and not cache["trip_ids"])):
            continue

        T         = cache["T"]
        Omega_m   = scenario["Omega_m"]
        spare_cap = scenario["spare_capacity"]
        theta     = scenario["theta"]
        bus_trips_s = scenario["bus_trips"]

        # 버스 국소블록만 제한하고 결합행/열차 지지집합은 full cache 그대로 둔다.
        # 한 information cell의 열을 택하면 그 cell에 속한 모든 duration에서
        # 같은 tid가 동시에 활성화된다.
        if _active_columns is None:
            active_trip_ids = tuple(cache["trip_ids"])
        else:
            block_key = str(scenario.get("block_station", sid))
            duration = int(scenario.get("duration", 0))
            selected_trip_ids = {
                tid for k, tid, cell in _active_columns
                if k == block_key and duration in cell
            }
            active_trip_ids = tuple(
                tid for tid in cache["trip_ids"]
                if tid in selected_trip_ids and tid in bus_trips_s
            )
        active_trip_set = set(active_trip_ids)

        # --- y[v]: 버스 trip 활성화 ---
        # 역 블록이면 같은 정보집합의 지속시간들이 하나의 변수를 공유한다.
        shared = y_shared.get(sid)
        if shared is None:
            y = {}
            for tid in active_trip_ids:
                key = (str(sid), str(tid), (int(scenario.get("duration", 0)),))
                y[tid] = model.addVar(
                    lb=0, ub=float(_column_ub.get(key, 1.0)), vtype=_vt_y,
                    name=f"y[{sid},{tid}]")
        else:
            y = {}
            for tid in active_trip_ids:
                v = shared.get(tid)
                if v is None:
                    # 블록 내 trip 집합이 어긋났다는 뜻. 조용히 넘기면 배차변수
                    # 없는 trip 이 생겨 모형이 틀린다.
                    raise RuntimeError(
                        f"{sid}: trip {tid} 이 공유 배차변수에 없음 — "
                        f"build_block_trips 를 거치지 않았는지 확인")
                y[tid] = v

        # --- b, a (버스 + 열차) ---
        b, a = {}, {}
        for (om, vid, t, j) in cache["b_var_keys"]:
            if str(vid).startswith("B_") and vid not in active_trip_set:
                continue
            ub = CumD.get((om, t), 0.0)
            b[(om, vid, t, j)] = model.addVar(lb=0, ub=max(ub, 0), vtype=GRB.CONTINUOUS,
                                               name=f"b[{sid},{om},{vid},{t}]")
        for (om, vid, t, j) in cache["a_var_keys"]:
            if str(vid).startswith("B_") and vid not in active_trip_set:
                continue
            a[(om, vid, t, j)] = model.addVar(lb=0, ub=CumD.get((om,t),0),
                                               vtype=GRB.CONTINUOUS,
                                               name=f"a[{sid},{om},{vid},{t}]")

        # --- l ---
        l = {}
        for (om, vid, t) in cache["l_var_keys"]:
            if str(vid).startswith("B_") and vid not in active_trip_set:
                continue
            cap = BUS_CAPACITY if vid.startswith("B_") else TRAIN_CAPACITY
            l[(om, vid, t)] = model.addVar(lb=0, ub=cap, vtype=GRB.CONTINUOUS,
                                            name=f"l[{sid},{om},{vid},{t}]")

        # The exact formulation needs a continuous minute-indexed load stream
        # from the first unique event through the last one.  Legacy cache keys
        # are the union of coverage intervals and can contain gaps when the
        # same omega is represented by several coverage records.
        aggregate_load_groups = (
            _aggregate_bus_load_groups(cache, active_trip_set)
            if aggregate_bus_load_balance else None
        )
        if aggregate_load_groups is not None:
            for tid, by_om in aggregate_load_groups.items():
                for om, group in by_om.items():
                    for t in group["times"]:
                        lkey = (om, tid, t)
                        if lkey not in l:
                            l[lkey] = model.addVar(
                                lb=0, ub=BUS_CAPACITY,
                                vtype=GRB.CONTINUOUS,
                                name=f"l[{sid},{om},{tid},{t}]",
                            )

        # --- q ---
        q = {}
        for (om, j, t) in cache["q_var_keys"]:
            q[(om, j, t)] = model.addVar(lb=0, vtype=GRB.CONTINUOUS,
                                          name=f"q[{sid},{om},{j},{t}]")

        # --- w ---
        w = {}
        for (om, j, t) in cache["w_var_keys"]:
            w[(om, j, t)] = model.addVar(lb=0, vtype=GRB.CONTINUOUS,
                                          name=f"w[{sid},{om},{j},{t}]")

        # --- od (Exit): 목적지 도달 승객 소멸 변수 (File 1의 od에 대응) ---
        od = {}
        for (om, t) in cache["od_var_keys"]:
            od[(om, t)] = model.addVar(lb=0, ub=CumD.get((om, t), 0.0),
                                        vtype=GRB.CONTINUOUS,
                                        name=f"od[{sid},{om},{t}]")

        model.update()

        # ── (C2) Depot-시간 capacity ──
        depot_time_trips = defaultdict(lambda: defaultdict(list))
        for tid, trip in bus_trips_s.items():
            d     = trip["Depot"]
            t_s   = trip["DispatchStart"]
            t_e   = trip["ResourceEnd"]
            for tau in range(max(t_s, Gamma[0]), min(t_e, Gamma[-1] + 1)):
                depot_time_trips[d][tau].append(tid)

        for d in depots:
            if _depot_row_keys is None:
                # Preserve the legacy insertion order on the default path so
                # time-limited MIP search remains reproducible.
                row_items = depot_time_trips.get(d, {}).items()
            else:
                row_minutes = set(depot_time_trips.get(d, {}))
                row_minutes.update(_depot_rows_by_sid_d[(str(sid), str(d))])
                row_items = ((tau, depot_time_trips.get(d, {}).get(tau, ()))
                             for tau in sorted(row_minutes))
            for tau, tids in row_items:
                active = [y[tid] for tid in tids if tid in y]
                if active or (_depot_row_keys is not None and
                              (str(sid), str(d), int(tau)) in _depot_row_keys):
                    _c = model.addConstr(gp.quicksum(active) <= x[d],
                                         name=f"depot_t[{sid},{d},{tau}]")
                    if _collect:
                        duals_out["depot"][(sid, d, tau)] = _c

        # ── Load balance (버스) ──
        if aggregate_load_groups is not None:
            for tid, by_om in aggregate_load_groups.items():
                for om, group in by_om.items():
                    previous = 0
                    for index, t in enumerate(group["times"]):
                        lkey = (om, tid, t)
                        if lkey not in l:
                            continue
                        board_term = gp.quicksum(
                            b[key] for key in group["board"].get(t, ())
                            if key in b
                        )
                        alight_term = gp.quicksum(
                            a[key] for key in group["alight"].get(t, ())
                            if key in a
                        )
                        model.addConstr(
                            l[lkey] == previous + board_term - alight_term,
                            name=(f"lb0[{sid},{om},{tid},{t}]" if index == 0
                                  else f"lb[{sid},{om},{tid},{t}]"),
                        )
                        previous = l[lkey]
                    if group["times"]:
                        last = int(group["times"][-1])
                        last_key = (om, tid, last)
                        if last_key in l:
                            model.addConstr(
                                l[last_key] == 0.0,
                                name=f"bus_empty[{sid},{om},{tid},{last}]")
        else:
            for tid, od_list in cache["trip_od_list"].items():
                if tid not in active_trip_set:
                    continue
                for od_info in od_list:
                    om = od_info["omega"]
                    bt, at = od_info["board_t"], od_info["alight_t"]
                    bj, aj = od_info["board_j"], od_info["alight_j"]
                    for t in range(bt, at+1):
                        lkey = (om, tid, t)
                        if lkey not in l: continue
                        board_term  = b.get((om, tid, bt, bj), 0) if t == bt else 0
                        alight_term = a.get((om, tid, at, aj), 0) if t == at else 0
                        if t == bt:
                            model.addConstr(l[lkey] == board_term - alight_term,
                                            name=f"lb0[{sid},{om},{tid},{t}]")
                        else:
                            prev = l.get((om, tid,t-1))
                            if prev is not None:
                                model.addConstr(l[lkey] == prev + board_term - alight_term,
                                                name=f"lb[{sid},{om},{tid},{t}]")

        # ── Load balance (열차) ──
        for om in Omega_m:
            if om not in Omega_Map: continue
            orig, dest = Omega_Map[om]
            for v in Vm:
                if v not in train_active_times: continue
                for t in train_active_times[v]:
                    lkey = (om, v, t)
                    if lkey not in l: continue
                    s_t = station_of[v].get(t)
                    board_t  = b.get((om, v, t, s_t), 0) if s_t else 0
                    alight_t = a.get((om, v, t, s_t), 0) if s_t else 0
                    t_start  = train_active_times[v].start
                    if t == t_start:
                        model.addConstr(l[lkey] == board_t - alight_t,
                                        name=f"trlb0[{sid},{om},{v},{t}]")
                    else:
                        prev = l.get((om, v, t-1))
                        if prev is not None:
                            model.addConstr(l[lkey] == prev + board_t - alight_t,
                                            name=f"trlb[{sid},{om},{v},{t}]")

        # ── Bus capacity ──
        l_by_trip_t = defaultdict(list)
        for (om, tid, t), lvar in l.items():
            if tid.startswith("B_"):
                l_by_trip_t[(tid, t)].append(lvar)
        for (tid, t), lvars in l_by_trip_t.items():
            model.addConstr(gp.quicksum(lvars) <= BUS_CAPACITY * y[tid],
                            name=f"bus_cap[{sid},{tid},{t}]")

        # ── Train capacity (spare_capacity 반영) ──
        l_by_train_t = defaultdict(list)
        for (om, vid, t), lvar in l.items():
            if not vid.startswith("B_"):
                l_by_train_t[(vid, t)].append(lvar)
        for (vid, t), lvars in l_by_train_t.items():
            cap = spare_cap.get((vid, t), 0.0)
            model.addConstr(gp.quicksum(lvars) <= cap,
                            name=f"tr_cap[{sid},{vid},{t}]")

        # ── Queue flow (q) ──
        sigma_int = int(sigma)
        for om in Omega_m:
            if om not in Omega_Map: continue
            orig, dest = Omega_Map[om]
            N_om = cache["N_omega"].get(om, set())
            D    = wdata["D"]

            for j in N_om:
                for ti, t in enumerate(Gamma):
                    qvar = q.get((om, j, t))
                    if qvar is None: continue

                    d_jt = D.get((om, j, t), 0.0)

                    # 탑승 합계
                    board_sum  = gp.quicksum(
                        b[(om, vid, t, j)]
                        for vid in (cache["boarding_by_od_t_j"].get((om, t, j), []))
                        if (om, vid, t, j) in b
                    )
                    # 하차 합계 (sigma분 전)
                    alight_sum = gp.LinExpr()
                    if t >= START_MIN + sigma_int:
                        t_sig = t - sigma_int
                        for vid in cache["alighting_by_od_t_j"].get((om, t_sig, j), []):
                            akey = (om, vid, t_sig, j)
                            if akey in a: alight_sum += a[akey]

                    q_prev = q.get((om, j, t-1), 0) if ti > 0 else 0

                    if j == dest:
                        # 목적지 queue: File 1과 동일하게 od[om,t]로 도달 승객 소멸
                        # + alight_sum (환승 후 목적지에 하차한 승객도 반영)
                        od_sink = od.get((om, t), 0)
                        _c = model.addConstr(
                            qvar == q_prev + d_jt + alight_sum - board_sum - od_sink,
                            name=f"q_dest[{sid},{om},{j},{t}]"
                        )
                    else:
                        _c = model.addConstr(
                            qvar == q_prev + d_jt + alight_sum - board_sum,
                            name=f"q_trans[{sid},{om},{j},{t}]"
                        )
                    if _collect:
                        duals_out["q"][(sid, om, j, t)] = _c

        # ── Walking/Transfer flow (w) ──
        for om in Omega_m:
            if om not in Omega_Map: continue
            orig, dest = Omega_Map[om]
            N_om = cache["N_omega"].get(om, set())

            for j in N_om:
                if j == dest: continue
                for ti, t in enumerate(Gamma):
                    wvar = w.get((om, j, t))
                    if wvar is None: continue

                    alight_in = gp.quicksum(
                        a[(om, vid, t, j)]
                        for vid in cache["alighting_by_od_t_j"].get((om, t, j), [])
                        if (om, vid, t, j) in a
                    )
                    alight_out = gp.LinExpr()
                    if t >= START_MIN + sigma_int:
                        t_sig = t - sigma_int
                        for vid in cache["alighting_by_od_t_j"].get((om, t_sig, j), []):
                            akey = (om, vid, t_sig, j)
                            if akey in a: alight_out += a[akey]

                    w_prev = w.get((om, j, t-1), 0) if ti > 0 else 0
                    _c = model.addConstr(
                        wvar == w_prev + alight_in - alight_out,
                        name=f"w[{sid},{om},{j},{t}]"
                    )
                    if _collect:
                        duals_out["w"][(sid, om, j, t)] = _c

        # ── 탑승 논리 제약 ──
        for om in Omega_m:
            if om not in Omega_Map: continue
            orig, dest = Omega_Map[om]
            # 목적지에서 탑승 불가
            for (om2, vid, t, j), bvar in b.items():
                if om2 == om and j == dest:
                    model.addConstr(bvar == 0, name=f"no_board_dest[{sid},{om},{vid},{t}]")
            # 출발역에서 하차 불가
            for (om2, vid, t, j), avar in a.items():
                if om2 == om and j == orig:
                    model.addConstr(avar == 0, name=f"no_alight_orig[{sid},{om},{vid},{t}]")
            # 열차 dwell 제약
            for v in Vm:
                if v not in train_active_times: continue
                for dwell in train_dwells[v]:
                    s_d, times = dwell["station"], dwell["times"]
                    inc_load = l.get((om, v, times[0]-1), 0)
                    alight_sum = gp.quicksum(
                        a[(om, v, t, s_d)] for t in times if (om, v, t, s_d) in a
                    )
                    if alight_sum.size() > 0:
                        model.addConstr(alight_sum <= inc_load,
                                        name=f"dwell[{sid},{om},{v},{s_d}]")
                if v in train_dest_map:
                    t_last = train_active_times[v].stop - 1
                    if (om, v, t_last) in l:
                        model.addConstr(l[(om, v, t_last)] == 0,
                                        name=f"tr_empty[{sid},{om},{v}]")

        # ── 최소 서비스 ──
        b_by_trip = defaultdict(list)
        for (om, tid, t, j), bvar in b.items():
            if tid.startswith("B_"):
                b_by_trip[tid].append(bvar)
        for tid in cache["trip_ids"]:
            bvars = b_by_trip.get(tid, [])
            if bvars:
                model.addConstr(gp.quicksum(bvars) >= MIN_TRIP_BOARDING * y[tid],
                                name=f"min_svc[{sid},{tid}]")

        # ── 목적함수 기여 ──
        Q = gp.quicksum(q.values())
        L = gp.quicksum(l.values())
        W = gp.quicksum(w.values())
        obj_terms.append(prob * (1.0 / T) * (Q + L + W))

        scen_vars[sid] = {
            "y": y, "b": b, "a": a, "l": l, "q": q, "w": w, "od": od,
            "b_by_trip": dict(b_by_trip),
            "active_trip_ids": tuple(sorted(active_trip_ids)),
            "cache": cache, "scenario": scenario,
        }

    if not obj_terms:
        return {d: 0 for d in depots}, "NO_TRIPS", {}, None

    model.setObjective(gp.quicksum(obj_terms), GRB.MINIMIZE)
    model.update()
    t_build_sec = time.time() - t_func_start
    print(f"  [MILP] vars={model.NumVars:,}, constrs={model.NumConstrs:,}, "
          f"binary={model.NumIntVars:,}, build={t_build_sec:.1f}s")

    t0 = time.time()
    model.optimize()
    solve_sec = time.time() - t0
    status = _status_name(model.Status)
    stage1_stats = collect_gurobi_stats(model, solve_sec)
    stage1_stats["BuildSec"]   = float(t_build_sec)
    stage1_stats["ExtractSec"] = 0.0
    stage1_stats["AggregateBusLoadBalance"] = bool(
        aggregate_bus_load_balance)
    print(f"  [MILP] status={status}, time={solve_sec:.1f}s, "
          f"gap={stage1_stats['MIPGap']*100 if not math.isnan(stage1_stats['MIPGap']) else float('nan'):.2f}%")

    # 듀얼은 model.dispose() 전에 값으로 뽑아야 한다. 핸들은 dispose 후 무효다.
    # 완화하지 않은 MIP 에는 Pi 가 없으므로 lp_relax 일 때만 채운다.
    if _collect:
        for kind in ("q", "w", "depot"):
            src = duals_out.get(kind, {})
            # A TIME_LIMIT/SUBOPTIMAL LP dual is not an optimality certificate
            # for full-pool pricing.  Return duals only for an OPTIMAL LP so a
            # caller cannot accidentally label an interrupted RMP as converged.
            duals_out[kind] = ({k: float(c.Pi) for k, c in src.items()}
                               if lp_relax and model.Status == GRB.OPTIMAL
                               else {})

    if model.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT):
        model.dispose()
        return {d: 0 for d in depots}, status, {}, stage1_stats

    # TIME_LIMIT이지만 feasible solution이 없는 경우
    if model.SolCount == 0:
        print("  [MILP] No feasible solution found.")
        model.dispose()
        return {d: 0 for d in depots}, status, {}, stage1_stats

    if not extract_solution:
        model.dispose()
        return {d: 0 for d in depots}, status, {}, stage1_stats

    t_extract_start = time.time()
    depot_counts = {d: int(round(x[d].X)) for d in depots}

    # Extract results
    scenario_results = {}
    for sid, sv in scen_vars.items():
        y, b, a, l, q, w, od = sv["y"], sv["b"], sv["a"], sv["l"], sv["q"], sv["w"], sv["od"]
        cache    = sv["cache"]
        scenario = sv["scenario"]

        boarded      = sum(float(bv.X) for bv in b.values())
        queue_area   = sum(float(qv.X) for qv in q.values())
        onboard_sum  = sum(float(lv.X) for lv in l.values())
        walking_sum  = sum(float(wv.X) for wv in w.values())
        obj_val      = (queue_area + onboard_sum + walking_sum) / cache["T"]

        activated, serving = 0, 0
        dispatched = []
        for tid in sv["active_trip_ids"]:
            if float(y[tid].X) <= 0.5: continue
            activated += 1
            trip = scenario["bus_trips"][tid]
            trip_boarded = sum(float(bv.X) for bv in sv["b_by_trip"].get(tid, []))
            if trip_boarded <= QUEUE_EPS: continue
            serving += 1
            dispatched.append({
                "TripID":        tid,
                "AssignedDepot": trip["Depot"],
                "Route":         trip["Route"],
                "RouteDir":      trip["RouteKey"],
                "RouteType":     trip["RouteType"],
                "Direction":     trip["Direction"],
                "StartStation":  trip["RouteOrigin"],
                "StartTime":     trip["RunStart"],
                "EndStation":    trip["RouteDest"],
                "EndTime":       trip["RunEnd"],
                "DeadheadToStart": trip["DeadheadOut"],
                "PassengerCount":  round(trip_boarded, 4),
                "NumLegs":         trip.get("NumLegs", 1),
            })

        dispatched.sort(key=lambda r: (r["AssignedDepot"], r["StartTime"]))

        # --- DecisionVars: 모델 살아있는 동안 float으로 추출 ---
        Omega_Map_local = wdata["Omega_Map"]
        dv_rows = []
        for (om, vid, t, j), bvar in b.items():
            val = float(bvar.X)
            if val > 1e-4:
                orig, dest = Omega_Map_local.get(om, (om.split("_")[0], om.split("_")[-1]))
                dv_rows.append(("Boarding", om, orig, dest, t, j, vid, val))
        for (om, vid, t, j), avar in a.items():
            val = float(avar.X)
            if val > 1e-4:
                orig, dest = Omega_Map_local.get(om, (om.split("_")[0], om.split("_")[-1]))
                dv_rows.append(("Alighting", om, orig, dest, t, j, vid, val))
        for (om, vid, t), lvar in l.items():
            val = float(lvar.X)
            if val > 1e-4:
                orig, dest = Omega_Map_local.get(om, (om.split("_")[0], om.split("_")[-1]))
                dv_rows.append(("Load", om, orig, dest, t, None, vid, val))
        for (om, j, t), qvar in q.items():
            val = float(qvar.X)
            if val > 1e-4:
                orig, dest = Omega_Map_local.get(om, (om.split("_")[0], om.split("_")[-1]))
                dv_rows.append(("Queue", om, orig, dest, t, j, None, val))
        for (om, j, t), wvar in w.items():
            val = float(wvar.X)
            if val > 1e-4:
                orig, dest = Omega_Map_local.get(om, (om.split("_")[0], om.split("_")[-1]))
                dv_rows.append(("Walking", om, orig, dest, t, j, None, val))
        for (om, t), odvar in od.items():
            val = float(odvar.X)
            if val > 1e-4:
                orig, dest = Omega_Map_local.get(om, (om.split("_")[0], om.split("_")[-1]))
                dv_rows.append(("Exit", om, orig, dest, t, dest, None, val))

        scenario_results[sid] = {
            "status":          status,
            "objective":       float(obj_val),
            "dispatched":      dispatched,
            "solve_sec":       solve_sec,
            "bus_trips_used":  scenario["bus_trips"],
            "boarded_total":   float(boarded),
            "queue_area":      float(queue_area),
            "onboard_sum":     float(onboard_sum),
            "walking_sum":     float(walking_sum),
            "activated_trips": activated,
            "serving_trips":   serving,
            "dv_rows":         dv_rows,
        }

    stage1_stats["ExtractSec"] = float(time.time() - t_extract_start)
    model.dispose()
    return depot_counts, status, scenario_results, stage1_stats



# =============================================================================
# 9.5 Stage 2 Fixed (depot 고정 후 개별 시나리오 풀기)
# =============================================================================
def solve_stage2_fixed(env, depots, depot_counts, scenario, cache, wdata):
    """
    depot_counts (x[d]) 를 고정값으로 두고 Stage 2만 풀기.
    Two-Stage MILP에서 결정된 depot 배치를 유지한 채
    raw_scenarios 전체에 대해 개별 스케줄링 결과를 도출.
    """
    t_func_start = time.time()
    sid       = scenario["id"]
    Omega_m   = scenario["Omega_m"]
    spare_cap = scenario["spare_capacity"]
    theta     = scenario["theta"]
    bus_trips_s = scenario["bus_trips"]
    Omega_Map = wdata["Omega_Map"]
    Gamma     = wdata["Gamma"]
    START_MIN = wdata["START_MIN"]
    trains_at = wdata["trains_at"]
    station_of = wdata["station_of"]
    train_active_times = wdata["train_active_times"]
    train_dest_map     = wdata["train_dest_map"]
    train_dwells       = wdata["train_dwells"]
    valid_keys         = wdata["valid_keys"]
    Vm                 = wdata["Vm"]
    CumD               = wdata["CumD"]
    D                  = wdata["D"]
    T                  = cache["T"]
    sigma              = 2

    if not cache["trip_ids"]:
        return _empty_s2_result(sid)

    model = gp.Model(f"Stage2_{sid}", env=env)
    model.Params.MIPGap      = MIP_GAP
    model.Params.TimeLimit   = TIME_LIMIT
    model.Params.OutputFlag  = 0
    model.Params.MIPFocus    = 1
    model.Params.Heuristics  = 0.5

    # y[v]: 버스 trip 활성화
    y = {tid: model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name=f"y[{tid}]")
         for tid in cache["trip_ids"]}

    # Depot-시간 capacity (x[d] 고정값으로 상한 설정)
    depot_time_trips = defaultdict(lambda: defaultdict(list))
    for tid, trip in bus_trips_s.items():
        d   = trip["Depot"]
        t_s = trip["DispatchStart"]
        t_e = trip["ResourceEnd"]
        for tau in range(max(t_s, Gamma[0]), min(t_e, Gamma[-1] + 1)):
            depot_time_trips[d][tau].append(tid)

    for d in depots:
        cap = depot_counts.get(d, 0)
        for tau, tids in depot_time_trips.get(d, {}).items():
            active = [y[tid] for tid in tids if tid in y]
            if active:
                model.addConstr(gp.quicksum(active) <= cap,
                                name=f"depot_t[{d},{tau}]")

    # b, a, l, q, w, od
    b, a, l, q, w, od = {}, {}, {}, {}, {}, {}
    for (om, vid, t, j) in cache["b_var_keys"]:
        ub = CumD.get((om, t), 0.0)
        b[(om, vid, t, j)] = model.addVar(lb=0, ub=max(ub, 0), vtype=GRB.CONTINUOUS)
    for (om, vid, t, j) in cache["a_var_keys"]:
        a[(om, vid, t, j)] = model.addVar(lb=0, ub=CumD.get((om, t), 0), vtype=GRB.CONTINUOUS)
    for (om, vid, t) in cache["l_var_keys"]:
        cap_v = BUS_CAPACITY if vid.startswith("B_") else TRAIN_CAPACITY
        l[(om, vid, t)] = model.addVar(lb=0, ub=cap_v, vtype=GRB.CONTINUOUS)
    for (om, j, t) in cache["q_var_keys"]:
        q[(om, j, t)] = model.addVar(lb=0, vtype=GRB.CONTINUOUS)
    for (om, j, t) in cache["w_var_keys"]:
        w[(om, j, t)] = model.addVar(lb=0, vtype=GRB.CONTINUOUS)
    for (om, t) in cache["od_var_keys"]:
        od[(om, t)] = model.addVar(lb=0, ub=CumD.get((om, t), 0.0), vtype=GRB.CONTINUOUS)

    model.update()

    # 제약 (solve_milp와 동일)
    # Load balance (버스)
    for tid, od_list in cache["trip_od_list"].items():
        for od_info in od_list:
            om = od_info["omega"]
            bt, at = od_info["board_t"], od_info["alight_t"]
            bj, aj = od_info["board_j"], od_info["alight_j"]
            for t in range(bt, at + 1):
                lkey = (om, tid, t)
                if lkey not in l: continue
                board  = b.get((om, tid, bt, bj), 0) if t == bt else 0
                alight = a.get((om, tid, at, aj), 0) if t == at else 0
                if t == bt:
                    model.addConstr(l[lkey] == board - alight)
                else:
                    prev = l.get((om, tid, t-1))
                    if prev is not None:
                        model.addConstr(l[lkey] == prev + board - alight)

    # Load balance (열차)
    for om in Omega_m:
        if om not in Omega_Map: continue
        for v in Vm:
            if v not in train_active_times: continue
            for t in train_active_times[v]:
                lkey = (om, v, t)
                if lkey not in l: continue
                s_t    = station_of[v].get(t)
                board  = b.get((om, v, t, s_t), 0) if s_t else 0
                alight = a.get((om, v, t, s_t), 0) if s_t else 0
                t_start = train_active_times[v].start
                if t == t_start:
                    model.addConstr(l[lkey] == board - alight)
                else:
                    prev = l.get((om, v, t-1))
                    if prev is not None:
                        model.addConstr(l[lkey] == prev + board - alight)

    # Bus capacity
    l_by_trip_t = defaultdict(list)
    for (om, tid, t), lvar in l.items():
        if tid.startswith("B_"):
            l_by_trip_t[(tid, t)].append(lvar)
    for (tid, t), lvars in l_by_trip_t.items():
        model.addConstr(gp.quicksum(lvars) <= BUS_CAPACITY * y[tid])

    # Train capacity
    l_by_train_t = defaultdict(list)
    for (om, vid, t), lvar in l.items():
        if not vid.startswith("B_"):
            l_by_train_t[(vid, t)].append(lvar)
    for (vid, t), lvars in l_by_train_t.items():
        model.addConstr(gp.quicksum(lvars) <= spare_cap.get((vid, t), 0.0))

    # Queue flow
    for om in Omega_m:
        if om not in Omega_Map: continue
        orig, dest = Omega_Map[om]
        N_om = cache["N_omega"].get(om, set())
        for j in N_om:
            for ti, t in enumerate(Gamma):
                qvar = q.get((om, j, t))
                if qvar is None: continue
                d_jt = D.get((om, j, t), 0.0)
                board_sum = gp.quicksum(
                    b[(om, vid, t, j)]
                    for vid in cache["boarding_by_od_t_j"].get((om, t, j), [])
                    if (om, vid, t, j) in b
                )
                alight_sum = gp.LinExpr()
                if t >= START_MIN + sigma:
                    t_sig = t - sigma
                    for vid in cache["alighting_by_od_t_j"].get((om, t_sig, j), []):
                        akey = (om, vid, t_sig, j)
                        if akey in a: alight_sum += a[akey]
                q_prev = q.get((om, j, t-1), 0) if ti > 0 else 0
                if j == dest:
                    od_sink = od.get((om, t), 0)
                    model.addConstr(qvar == q_prev + d_jt + alight_sum - board_sum - od_sink)
                else:
                    model.addConstr(qvar == q_prev + d_jt + alight_sum - board_sum)

    # Walking flow
    for om in Omega_m:
        if om not in Omega_Map: continue
        orig, dest = Omega_Map[om]
        N_om = cache["N_omega"].get(om, set())
        for j in N_om:
            if j == dest: continue
            for ti, t in enumerate(Gamma):
                wvar = w.get((om, j, t))
                if wvar is None: continue
                alight_in = gp.quicksum(
                    a[(om, vid, t, j)]
                    for vid in cache["alighting_by_od_t_j"].get((om, t, j), [])
                    if (om, vid, t, j) in a
                )
                alight_out = gp.LinExpr()
                if t >= START_MIN + sigma:
                    t_sig = t - sigma
                    for vid in cache["alighting_by_od_t_j"].get((om, t_sig, j), []):
                        akey = (om, vid, t_sig, j)
                        if akey in a: alight_out += a[akey]
                w_prev = w.get((om, j, t-1), 0) if ti > 0 else 0
                model.addConstr(wvar == w_prev + alight_in - alight_out)

    # 탑승 논리
    for om in Omega_m:
        if om not in Omega_Map: continue
        orig, dest = Omega_Map[om]
        for (om2, vid, t, j), bvar in b.items():
            if om2 == om and j == dest:
                model.addConstr(bvar == 0)
        for (om2, vid, t, j), avar in a.items():
            if om2 == om and j == orig:
                model.addConstr(avar == 0)
        for v in Vm:
            if v not in train_active_times: continue
            for dwell in train_dwells[v]:
                s_d, times = dwell["station"], dwell["times"]
                inc_load = l.get((om, v, times[0]-1), 0)
                alight_sum = gp.quicksum(
                    a[(om, v, t, s_d)] for t in times if (om, v, t, s_d) in a
                )
                if alight_sum.size() > 0:
                    model.addConstr(alight_sum <= inc_load)
            if v in train_dest_map:
                t_last = train_active_times[v].stop - 1
                if (om, v, t_last) in l:
                    model.addConstr(l[(om, v, t_last)] == 0)

    # 최소 서비스
    b_by_trip = defaultdict(list)
    for (om, tid, t, j), bvar in b.items():
        if tid.startswith("B_"): b_by_trip[tid].append(bvar)
    for tid in cache["trip_ids"]:
        bvars = b_by_trip.get(tid, [])
        if bvars:
            model.addConstr(gp.quicksum(bvars) >= MIN_TRIP_BOARDING * y[tid])

    # 목적함수
    Q = gp.quicksum(q.values())
    L = gp.quicksum(l.values())
    W = gp.quicksum(w.values())
    model.setObjective((Q + L + W) / T, GRB.MINIMIZE)
    model.update()
    t_build_sec = time.time() - t_func_start

    t0 = time.time()
    model.optimize()
    solve_sec = time.time() - t0
    status = _status_name(model.Status)
    stage2_stats = collect_gurobi_stats(model, solve_sec)
    stage2_stats["BuildSec"]   = float(t_build_sec)
    stage2_stats["ExtractSec"] = 0.0

    if model.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT) or model.SolCount == 0:
        model.dispose()
        return _empty_s2_result(sid, status, stage2_stats)

    # 결과 추출 (모델 살아있는 동안 float 변환)
    t_extract_start = time.time()
    boarded     = sum(float(bv.X) for bv in b.values())
    queue_area  = sum(float(qv.X) for qv in q.values())
    onboard_sum = sum(float(lv.X) for lv in l.values())
    walking_sum = sum(float(wv.X) for wv in w.values())
    obj_val     = (queue_area + onboard_sum + walking_sum) / T

    activated, serving = 0, 0
    dispatched = []
    for tid in cache["trip_ids"]:
        if float(y[tid].X) <= 0.5: continue
        activated += 1
        trip = bus_trips_s[tid]
        trip_boarded = sum(float(bv.X) for bv in b_by_trip.get(tid, []))
        if trip_boarded <= QUEUE_EPS: continue
        serving += 1
        dispatched.append({
            "TripID":          tid,
            "AssignedDepot":   trip["Depot"],
            "Route":           trip["Route"],
            "RouteDir":        trip["RouteKey"],
            "RouteType":       trip["RouteType"],
            "Direction":       trip["Direction"],
            "StartStation":    trip["RouteOrigin"],
            "StartTime":       trip["RunStart"],
            "EndStation":      trip["RouteDest"],
            "EndTime":         trip["RunEnd"],
            "DeadheadToStart": trip["DeadheadOut"],
            "PassengerCount":  round(trip_boarded, 4),
            "NumLegs":         trip.get("NumLegs", 1),
        })

    dispatched.sort(key=lambda r: (r["AssignedDepot"], r["StartTime"]))

    # DecisionVars float 추출
    dv_rows = []
    for (om, vid, t, j), bvar in b.items():
        val = float(bvar.X)
        if val > 1e-4:
            orig, dest = Omega_Map.get(om, (om.split("_")[0], om.split("_")[-1]))
            dv_rows.append(("Boarding", om, orig, dest, t, j, vid, val))
    for (om, vid, t, j), avar in a.items():
        val = float(avar.X)
        if val > 1e-4:
            orig, dest = Omega_Map.get(om, (om.split("_")[0], om.split("_")[-1]))
            dv_rows.append(("Alighting", om, orig, dest, t, j, vid, val))
    for (om, vid, t), lvar in l.items():
        val = float(lvar.X)
        if val > 1e-4:
            orig, dest = Omega_Map.get(om, (om.split("_")[0], om.split("_")[-1]))
            dv_rows.append(("Load", om, orig, dest, t, None, vid, val))
    for (om, j, t), qvar in q.items():
        val = float(qvar.X)
        if val > 1e-4:
            orig, dest = Omega_Map.get(om, (om.split("_")[0], om.split("_")[-1]))
            dv_rows.append(("Queue", om, orig, dest, t, j, None, val))
    for (om, j, t), wvar in w.items():
        val = float(wvar.X)
        if val > 1e-4:
            orig, dest = Omega_Map.get(om, (om.split("_")[0], om.split("_")[-1]))
            dv_rows.append(("Walking", om, orig, dest, t, j, None, val))
    for (om, t), odvar in od.items():
        val = float(odvar.X)
        if val > 1e-4:
            orig, dest = Omega_Map.get(om, (om.split("_")[0], om.split("_")[-1]))
            dv_rows.append(("Exit", om, orig, dest, t, dest, None, val))

    stage2_stats["ExtractSec"] = float(time.time() - t_extract_start)
    model.dispose()

    return {
        "status":          status,
        "objective":       float(obj_val),
        "dispatched":      dispatched,
        "bus_trips_used":  bus_trips_s,
        "boarded_total":   float(boarded),
        "queue_area":      float(queue_area),
        "onboard_sum":     float(onboard_sum),
        "walking_sum":     float(walking_sum),
        "activated_trips": activated,
        "serving_trips":   serving,
        "dv_rows":         dv_rows,
        "stats":           stage2_stats,
    }


def _empty_s2_result(sid, status="NO_SOLUTION", stats=None):
    return {
        "status": status, "objective": 0.0, "dispatched": [],
        "bus_trips_used": {}, "boarded_total": 0.0, "queue_area": 0.0,
        "onboard_sum": 0.0, "walking_sum": 0.0,
        "activated_trips": 0, "serving_trips": 0, "dv_rows": [],
        "stats": stats,
    }


# =============================================================================
# 10. Output Export
# =============================================================================
def export_results(scenario, res):
    """
    Schedule_*.csv, Timetable_*.csv, DecisionVars_*.csv 출력.
    DecisionVars 포맷: Variable, Omega, Origin, Dest, Time, StationID, TrainID, Value
    (파일 4 Recovery model 출력과 동일한 구조)
    """
    fname      = f"Affected_Disrupt_{scenario['id']}_PosQueueDiff_WithStation.csv"
    sched_path = os.path.join(DIR_OUT_ALL,       f"Schedule_{fname}")
    tt_path    = os.path.join(DIR_OUT_TIMETABLE, f"Timetable_{fname}")
    dv_path    = os.path.join(DIR_OUT_DETAIL,    f"DecisionVars_{fname}")

    if os.path.exists(sched_path) and os.path.exists(tt_path):
        _log(f"    [SKIP] {scenario['id']}: output files already exist")
        try:
            return len(pd.read_csv(sched_path))
        except Exception:
            return 0

    dispatched = res.get("dispatched", [])
    bus_trips  = res.get("bus_trips_used", {})

    # --- DecisionVars (solve_milp 안에서 float으로 추출된 값) ---
    dv_rows = res.get("dv_rows", [])
    if dv_rows:
        pd.DataFrame(dv_rows, columns=[
            "Variable","Omega","Origin","Dest","Time","StationID","TrainID","Value"
        ]).to_csv(dv_path, index=False, encoding="utf-8-sig")

    if not dispatched:
        pd.DataFrame().to_csv(sched_path, index=False)
        pd.DataFrame().to_csv(tt_path,    index=False)
        return 0

    # Assign BusID
    counter = defaultdict(int)
    for row in dispatched:
        depot = row["AssignedDepot"]
        counter[depot] += 1
        row["BusID"]        = f"Bus_{depot}_{counter[depot]:02d}"
        row["TripSequence"] = counter[depot]

    # Schedule CSV
    sched_rows = [{
        "BusID":        r["BusID"],
        "TripSequence": r["TripSequence"],
        "TripID":       r["TripID"],
        "AssignedDepot":r["AssignedDepot"],
        "Line":         r["Route"],
        "RouteDir":     r["RouteDir"],
        "RouteType":    r["RouteType"],
        "StartStation": r["StartStation"],
        "StartTime":    r["StartTime"],
        "EndStation":   r["EndStation"],
        "EndTime":      r["EndTime"],
        "DeadheadToStart": r["DeadheadToStart"],
        "PassengerCount":  r["PassengerCount"],
        "NumLegs":      r["NumLegs"],
    } for r in dispatched]
    pd.DataFrame(sched_rows).to_csv(sched_path, index=False, encoding="utf-8-sig")

    # Timetable CSV
    tt_rows = []
    for r in dispatched:
        trip = bus_trips.get(r["TripID"])
        if not trip: continue
        seq = 0
        for leg in trip.get("Legs", []):
            for station, arr_t in zip(leg["stops"], leg["stop_times"]):
                seq += 1
                tt_rows.append({
                    "BusID":          r["BusID"],
                    "TripID":         r["TripID"],
                    "Route":          r["Route"],
                    "LegIndex":       leg["leg_index"],
                    "Direction":      leg.get("direction", ""),
                    "StopSeq":        seq,
                    "StationID":      station,
                    "ArrivalTime_min":arr_t,
                })
    pd.DataFrame(tt_rows).to_csv(tt_path, index=False, encoding="utf-8-sig")
    return len(sched_rows)


# =============================================================================
# 11. Main
# =============================================================================
def main():
    overall_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] === Two-Stage Stochastic MILP (Reduced Scenarios Only) ===")
    print(f"  Fleet: {MAX_TOTAL_BUSES} buses | Methodology K/duration: {K_PER_DURATION} -> max {len(DURATIONS) * K_PER_DURATION} representative scenarios/hour")
    print(f"  OUTPUT_ROOT: {OUTPUT_ROOT}")

    # Static data
    schedule_df = load_schedule(FILE_SCHEDULE)
    od_df       = load_od(FILE_OD)
    bus_tt      = load_distance_matrix(FILE_DISTANCE)
    route_lib   = load_route_library(FILE_PREDEFINED_ROUTES)
    if not route_lib:
        print("[ERROR] No routes loaded.")
        return

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 1)
    env.start()

    try:
        # 재개 시 기존 통계를 이어쓴다.
        # (완료된 윈도우는 DepotAlloc_{HH}.csv 존재로 skip되어 row가 새로 안
        #  쌓이는데, 빈 리스트로 시작하면 마지막 저장에서 파일이 덮어써져
        #  이전 성능 기록이 전부 날아간다.)
        stage1_perf_rows = []
        all_scenario_rows = []
        _prev_perf = os.path.join(DIR_OUT_PERF, "Stage1_Perf_All.csv")
        if os.path.exists(_prev_perf):
            try:
                stage1_perf_rows = pd.read_csv(_prev_perf).to_dict("records")
                print(f"  [RESUME] 기존 Stage1 perf {len(stage1_perf_rows)}건 로드")
            except Exception as e:
                print(f"  [RESUME] Stage1 perf 로드 실패 (새로 시작): {e}")
        _prev_scen = os.path.join(DIR_OUT_PERF, "_scenario_stats.csv")
        if os.path.exists(_prev_scen):
            try:
                all_scenario_rows = pd.read_csv(_prev_scen).to_dict("records")
                print(f"  [RESUME] 기존 시나리오 통계 {len(all_scenario_rows)}건 로드")
            except Exception as e:
                print(f"  [RESUME] 시나리오 통계 로드 실패 (새로 시작): {e}")

        for start_h in START_HOURS:
            print(f"\n{'='*70}")
            print(f"  Window {start_h:02d}:00 ~ {start_h+WINDOW_LENGTH_H:02d}:00")

            # Skip if Stage 1 already completed.
            # Stage 1의 유일한 산출물은 DepotAlloc_{HH}.csv 이므로 이것만 체크한다.
            # (이전엔 Stage 1이 만들지도 않는 Summary_{HH}.csv 도 AND 조건으로 체크해서
            #  실제로는 스킵이 절대 발동하지 않던 버그가 있었음.)
            _alloc_path = os.path.join(DIR_OUT_ALLOC, f"DepotAlloc_{start_h:02d}.csv")
            if os.path.exists(_alloc_path):
                print(f"  -> Already done (DepotAlloc_{start_h:02d}.csv exists), skipping.")
                continue

            # 1. Window data
            wdata = build_window_data(start_h, schedule_df, od_df)
            if wdata is None:
                print("  -> No trains, skipping.")
                continue

            # 2. Load disruption scenarios
            _t_load = time.time()
            raw_scenarios = load_disruption_scenarios(start_h, wdata)
            load_disrupt_sec = time.time() - _t_load
            if not raw_scenarios:
                print("  -> No disruption scenarios found, skipping.")
                continue
            print(f"  Raw scenarios: {len(raw_scenarios)} (loaded in {load_disrupt_sec:.1f}s)")

            # 3. Scenario reduction
            _t_red = time.time()
            reduced = reduce_scenarios(raw_scenarios, K_PER_DURATION, wdata)
            reduce_sec = time.time() - _t_red
            print(f"  Reduced to: {len(reduced)} representative scenarios ({reduce_sec:.1f}s)")
            if not reduced:
                continue

            # 4. Depot selection (route endpoints)
            depots = sorted(set(
                s for rd in route_lib.values() for s in [rd["stops"][0], rd["stops"][-1]]
            ))

            # 5. Preprocess per scenario — 시나리오별 cache build time 기록
            all_caches = {}
            cache_build_stats = []
            # 역 블록이면 30/60/90 이 같은 trip 집합을 써야 y 를 공유할 수 있다.
            block_trips = build_block_trips(reduced, start_h, depots,
                                            bus_tt, route_lib)
            for si, scenario in enumerate(reduced, 1):
                t0 = time.time()
                if scenario["id"] in block_trips:
                    sel_routes, bus_trips = block_trips[scenario["id"]]
                else:
                    scen_lib, gen_stats = build_scenario_route_library(
                        scenario, start_h, depots, bus_tt, route_lib)
                    sel_routes = select_routes(scenario, depots, scen_lib, bus_tt)
                    bus_trips  = generate_bus_trips(scenario, sel_routes, scen_lib)
                scenario["bus_trips"]       = bus_trips
                scenario["selected_routes"] = sel_routes
                cache = build_stage2_cache(scenario, wdata, bus_trips)
                all_caches[scenario["id"]] = cache
                # selected_routes는 캐시 빌드 후 불필요 → 해제
                del scenario["selected_routes"]
                cb_sec = time.time() - t0
                cache_build_stats.append({
                    "Window":     f"{start_h:02d}_{start_h+WINDOW_LENGTH_H:02d}",
                    "Scenario":   scenario["id"],
                    "Station":    scenario["station"],
                    "Duration":   scenario["duration"],
                    "Omega_m":    len(scenario["Omega_m"]),
                    "NumRoutes":  len(sel_routes),
                    "NumTrips":   len(bus_trips),
                    "b_vars":     len(cache["b_var_keys"]),
                    "q_vars":     len(cache["q_var_keys"]),
                    "w_vars":     len(cache["w_var_keys"]),
                    "CacheBuildSec": cb_sec,
                })
                _log(f"    [{si}/{len(reduced)}] {scenario['id']}: "
                     f"routes={len(sel_routes)}, trips={len(bus_trips)}, "
                     f"b_vars={len(cache['b_var_keys'])}, "
                     f"q_vars={len(cache['q_var_keys'])}, "
                     f"w_vars={len(cache['w_var_keys'])}, "
                     f"{cb_sec:.1f}s")

            # 6. Solve Two-Stage MILP (대표 시나리오로 depot x[d] 결정)
            depot_counts, milp_status, _, stage1_stats = solve_milp(
                env, depots, reduced, all_caches, wdata
            )
            print(f"  Status: {milp_status}")

            if stage1_stats is not None:
                stage1_row = {
                    "Window":     f"{start_h:02d}_{start_h+WINDOW_LENGTH_H:02d}",
                    "StartHour":  start_h,
                    "NumScenarios": len(reduced),
                    "LoadDisruptSec": load_disrupt_sec,
                    "ReduceSec":      reduce_sec,
                    **stage1_stats,
                }
                stage1_perf_rows.append(stage1_row)
                pd.DataFrame([stage1_row]).to_csv(
                    os.path.join(DIR_OUT_PERF, f"Stage1_Perf_{start_h:02d}.csv"),
                    index=False, encoding="utf-8-sig"
                )
                # 시나리오별 stats (cache build + shared Stage 1 stats)
                scen_rows = []
                for cb in cache_build_stats:
                    scen_rows.append({**cb, **{
                        "Stage1_NumVars":    stage1_stats.get("NumVars", 0),
                        "Stage1_NumConstrs": stage1_stats.get("NumConstrs", 0),
                        "Stage1_BuildSec":   stage1_stats.get("BuildSec", 0),
                        "Stage1_SolveSec":   stage1_stats.get("WallSec", 0),
                        "Stage1_ExtractSec": stage1_stats.get("ExtractSec", 0),
                        "Stage1_Status":     stage1_stats.get("Status", ""),
                        "Stage1_ObjVal":     stage1_stats.get("ObjVal", float("nan")),
                    }})
                pd.DataFrame(scen_rows).to_csv(
                    os.path.join(DIR_OUT_PERF, f"_scenario_stats_{start_h:02d}.csv"),
                    index=False, encoding="utf-8-sig"
                )
                all_scenario_rows.extend(scen_rows)
            active_depots = {d: c for d, c in depot_counts.items() if c > 0}
            print(f"  Depot allocation: {active_depots}")

            # Save depot allocation
            pd.DataFrame([{"Depot": d, "Count": depot_counts.get(d, 0)} for d in depots]).to_csv(
                os.path.join(DIR_OUT_ALLOC, f"DepotAlloc_{start_h:02d}.csv"),
                index=False, encoding="utf-8-sig"
            )

            # Stage 1만 풀고 종료 (시나리오별 Stage 2는 생략)
            all_caches.clear()

            if not active_depots:
                print("  -> No buses allocated.")

            # 윈도우 종료 시 명시적 메모리 해제 (다음 시간대 시작 전 회수)
            # raw_scenarios / reduced 안의 spare_capacity, theta, bus_trips 같은
            # 큰 dict가 다음 윈도우까지 유지되지 않도록 제거 후 GC 강제 실행.
            for _s in raw_scenarios:
                _s.clear()
            for _s in reduced:
                _s.clear()
            raw_scenarios.clear()
            reduced.clear()
            wdata.clear()
            cache_build_stats.clear()
            del raw_scenarios, reduced, wdata, all_caches, cache_build_stats
            gc.collect()

        # 재개로 이어붙인 경우 중복이 생길 수 있으므로 키 기준으로
        # 마지막 값만 남긴다.
        def _dedup(rows, key):
            df = pd.DataFrame(rows)
            if key in df.columns:
                df = df.drop_duplicates(subset=key, keep="last")
            return df

        # 전체 Stage 1 성능 통계 통합 저장
        if stage1_perf_rows:
            _pdf = _dedup(stage1_perf_rows, "StartHour")
            _pdf.to_csv(
                os.path.join(DIR_OUT_PERF, "Stage1_Perf_All.csv"),
                index=False, encoding="utf-8-sig"
            )
            print(f"\n  Stage1 perf → {DIR_OUT_PERF}/Stage1_Perf_All.csv "
                  f"({len(_pdf)} windows)")

        # 시나리오별 통계 통합 저장
        if all_scenario_rows:
            _sdf = _dedup(all_scenario_rows, "Scenario")
            _sdf.to_csv(
                os.path.join(DIR_OUT_PERF, "_scenario_stats.csv"),
                index=False, encoding="utf-8-sig"
            )
            print(f"  Scenario stats → {DIR_OUT_PERF}/_scenario_stats.csv "
                  f"({len(_sdf)} rows)")

    finally:
        env.dispose()

    print(f"\n{'='*70}")
    print(f"  Done in {time.time()-overall_start:.1f}s")


if __name__ == "__main__":
    main()
