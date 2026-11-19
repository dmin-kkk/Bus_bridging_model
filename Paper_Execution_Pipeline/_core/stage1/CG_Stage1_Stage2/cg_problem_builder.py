"""3MILP_integrated_reduced_corridor.py 의 K-means 축소를 그대로 써서
cg_stage1.CGProblem 을 만든다.

원본 파일은 수정하지 않고 import 만 한다. 축소·노선선택·trip 생성·캐시 빌드는
전부 원본 함수를 호출하므로, 여기서 나온 CGProblem 은 원 파이프라인의
3~5 단계 결과와 같은 것이다.

    2. load_disruption_scenarios(start_h, wdata)
    3. reduce_scenarios(raw, K, wdata)          ← K-means. REDUCTION 모드에 따라
                                                   duration 별 / 역 블록
    4. depots = 명시적 후보 파일에 열거된 역
    5. build_block_trips → build_stage2_cache

축소 모드
    station   역 단위 공동선택. 같은 역의 30/60/90 이 한 블록이 되어
              배차변수 y 를 정보집합에서 공유할 수 있다. CG 는 이 모드를 쓴다.
    duration  duration 별 독립 K-means (논문 식 16-17 의 현재 방식).
              블록이 없으므로 정보집합이 duration 하나짜리로 퇴화하고
              비예견성이 사라진다. 비교용으로만 쓴다.

0임팩트 duration
    역 블록 축소는 Z_D = 0 인 duration 의 블록을 만들지 않는다. 기대값은
    정확하지만(Z=0), 정보트리가 (60,90) 처럼 잘려서 "아직 30분인지 모르는"
    구간이 사라진다. restore_zero_impact=True 로 두면 승객흐름 없이 배차·차고지
    점유만 갖는 resource-only 자식으로 되살려 30/60/90 트리를 유지한다.
"""
from __future__ import annotations

import csv
import os
import sys
import time
import importlib.util
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Mapping, Sequence

# Avoid the documented Windows MKL small-problem leak.  An explicitly supplied
# environment setting is respected.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

import cg_stage1 as cg


CORE_FILENAME = "3MILP_integrated_reduced_corridor.py"
EXPECTED_SCENARIO_LOGIC_VERSION = "causal_cancel_start_v3_20260809"
# 원본 core 파일은 이 폴더의 부모(프로젝트 루트)에 있다. 이동해도 안 된다.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.insert(0, os.path.dirname(PIPELINE_ROOT))   # package root (bbconfig)
from bbconfig import cfg  # noqa: E402
PIPELINE_INPUTS = str(os.environ.get("PAPER_PIPELINE_INPUT_DIR") or cfg.data_dir)
PIPELINE_RESULTS = str(os.environ.get("PAPER_PIPELINE_RESULTS_DIR") or cfg.results_dir)
ROUTE_FILE = os.path.join(
    PIPELINE_RESULTS, cfg.get("paths.routes"), cfg.get("routes.library_file"))
DEPOT_CANDIDATE_FILE = os.path.join(PIPELINE_INPUTS, cfg.get("inputs.depots"))
# Legacy exposure-prior file of the in-engine reduction (not read by the paper pipeline,
# which passes the pinned representative branches).
SCENARIO_PROBABILITY_FILE = os.path.join(
    PIPELINE_RESULTS, cfg.get("paths.clustering"), "scenario_probabilities_exposure.csv")

# Reviewer-validated production scenario reduction.  Scenario probabilities
# are deliberately excluded from K-means fitting and are reassigned only after
# the partition is fixed.
DEFAULT_K_PER_DURATION = 12
CLUSTER_RANDOM_STATE = 42
CLUSTER_N_INIT = 20
CLUSTER_FEATURE = "log1p_time_aggregated_q_plus_w"
CLUSTER_METHOD = "unweighted_euclidean_kmeans"

# Provenance label for the fixed, all-period K3--5 production library.
ROUTE_POOL_PROVENANCE = "static-all-od-k3-5"


def _require_scenario_logic(core) -> None:
    actual = getattr(core, "SCENARIO_LOGIC_VERSION", None)
    if actual != EXPECTED_SCENARIO_LOGIC_VERSION:
        raise RuntimeError(
            "CG Stage 1 requires the causal train-cancellation logic "
            f"{EXPECTED_SCENARIO_LOGIC_VERSION!r}; loaded {actual!r}")


def _load_candidate_depots(path=DEPOT_CANDIDATE_FILE):
    depots = []
    seen = set()
    with open(path, "r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = {"Code"} - fields
        if missing:
            raise ValueError(
                f"depot candidate file is missing columns: {sorted(missing)}")
        for row in reader:
            depot = str(row.get("Code", "")).strip().upper()
            if depot and depot not in seen:
                seen.add(depot)
                depots.append(depot)
    if not depots:
        raise ValueError("depot candidate file contains no station")
    return sorted(depots)


_SCENARIO_PROBABILITY_CACHE = None


def _load_scenario_probabilities(window_hour: int,
                                 path=SCENARIO_PROBABILITY_FILE):
    """Load the positive-exposure support for one start-hour window."""
    global _SCENARIO_PROBABILITY_CACHE
    if _SCENARIO_PROBABILITY_CACHE is None:
        frame = pd.read_csv(path, encoding="utf-8-sig")
        required = {
            "window", "duration", "station", "p_station_in_duration",
            "duration_weight", "p_scenario",
        }
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(
                f"scenario probability file is missing columns: {sorted(missing)}")
        frame = frame[list(required)].copy()
        frame["window"] = frame["window"].astype(int)
        frame["duration"] = frame["duration"].astype(int)
        frame["station"] = frame["station"].astype(str).str.strip().str.upper()
        for column in ("p_station_in_duration", "duration_weight", "p_scenario"):
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        frame = frame[frame["p_scenario"].gt(0)].copy()
        if frame.duplicated(["window", "duration", "station"]).any():
            raise ValueError("duplicate positive-probability scenario rows")
        _SCENARIO_PROBABILITY_CACHE = frame

    result = _SCENARIO_PROBABILITY_CACHE[
        _SCENARIO_PROBABILITY_CACHE["window"].eq(int(window_hour))
    ].copy()
    if result.empty:
        raise ValueError(f"window {window_hour}: no positive scenario probabilities")
    conditional_mass = result.groupby("duration")[
        "p_station_in_duration"
    ].sum()
    if not np.allclose(conditional_mass.to_numpy(), 1.0, atol=1e-10):
        raise ValueError(
            f"window {window_hour}: conditional station probabilities do not "
            f"sum to one: {conditional_mass.to_dict()}")
    return result.sort_values(["duration", "station"]).reset_index(drop=True)


def reduce_duration_scenarios_kmeans(scenarios, k: int, window_hour: int):
    """Apply the validated duration-wise unweighted K-means reduction.

    Features are station-level cumulative Queue + Walking inventories already
    carried in ``scenario['impact_by_station']``.  The only transform is
    ``log1p``.  Exposure probabilities do not affect cluster assignments;
    after fitting, every representative receives the probability mass of its
    cluster members.
    """
    k = int(k)
    if k <= 0:
        raise ValueError("K must be positive")
    if not scenarios:
        return []

    probabilities = _load_scenario_probabilities(window_hour)
    probability_by_key = {
        (int(row.duration), str(row.station)):
        (float(row.p_station_in_duration), float(row.duration_weight),
         float(row.p_scenario))
        for row in probabilities.itertuples(index=False)
    }
    scenario_by_key = {}
    for scenario in scenarios:
        key = (int(scenario["duration"]),
               str(scenario["station"]).strip().upper())
        if key in scenario_by_key:
            raise ValueError(f"duplicate raw disruption scenario: {key}")
        scenario_by_key[key] = scenario

    probability_keys = set(probability_by_key)
    scenario_keys = set(scenario_by_key)
    missing_raw = sorted(probability_keys - scenario_keys)
    extra_raw = sorted(scenario_keys - probability_keys)
    if missing_raw or extra_raw:
        raise RuntimeError(
            "raw disruption support and exposure-probability support differ: "
            f"missing_raw={missing_raw[:8]} ({len(missing_raw)} total), "
            f"extra_raw={extra_raw[:8]} ({len(extra_raw)} total). "
            "Regenerate the probability table and disruption scenarios with "
            "the same cancellation logic before solving.")

    feature_names = sorted({
        str(station).strip()
        for scenario in scenarios
        for station in scenario.get("impact_by_station", {})
    })
    if not feature_names:
        raise RuntimeError(
            f"window {window_hour}: passenger-inventory clustering features are empty")
    feature_index = {station: index for index, station in enumerate(feature_names)}

    reduced = []
    for duration in sorted(probabilities["duration"].unique()):
        duration = int(duration)
        duration_scenarios = sorted(
            (scenario for scenario in scenarios
             if int(scenario["duration"]) == duration),
            key=lambda scenario: str(scenario["station"]).strip().upper(),
        )
        if len(duration_scenarios) < k:
            raise RuntimeError(
                f"window {window_hour}, duration {duration}: "
                f"{len(duration_scenarios)} scenarios cannot support K={k}")

        raw_features = np.zeros(
            (len(duration_scenarios), len(feature_names)), dtype=float,
        )
        for row, scenario in enumerate(duration_scenarios):
            for station, value in scenario.get("impact_by_station", {}).items():
                station = str(station).strip()
                if station in feature_index:
                    raw_features[row, feature_index[station]] = float(value)
        transformed = np.log1p(raw_features)
        model = KMeans(
            n_clusters=k,
            random_state=CLUSTER_RANDOM_STATE,
            n_init=CLUSTER_N_INIT,
            algorithm="lloyd",
        ).fit(transformed)
        labels = model.labels_.astype(int)
        overall = transformed.mean(axis=0)
        total_sse = float(np.square(transformed - overall).sum())
        explained = (
            100.0 * (1.0 - float(model.inertia_) / total_sse)
            if total_sse > 1e-12 else 100.0
        )

        for cluster_id in range(k):
            members = np.flatnonzero(labels == cluster_id)
            if members.size == 0:
                raise AssertionError(
                    f"window {window_hour}/{duration}: empty cluster {cluster_id}")
            distances = np.linalg.norm(
                transformed[members] - model.cluster_centers_[cluster_id], axis=1,
            )
            representative_row = int(members[int(np.argmin(distances))])
            representative = duration_scenarios[representative_row].copy()

            member_keys = [
                (duration, str(duration_scenarios[index]["station"]).strip().upper())
                for index in members
            ]
            conditional_probability = float(sum(
                probability_by_key[key][0] for key in member_keys
            ))
            joint_probability = float(sum(
                probability_by_key[key][2] for key in member_keys
            ))
            representative["probability"] = joint_probability
            representative["prob_station"] = conditional_probability
            representative["prob_share"] = conditional_probability
            representative["cluster_id"] = int(cluster_id)
            representative["cluster_size"] = int(members.size)
            representative["representative_id"] = representative["id"]
            representative["distance_to_feature_centroid"] = float(
                distances[int(np.argmin(distances))]
            )
            representative["explained_variation_percent"] = explained
            representative["feature_transform"] = CLUSTER_FEATURE
            representative["cluster_method"] = CLUSTER_METHOD
            representative["cluster_probability_weighted"] = False
            # Duration outcomes are linked nonanticipatively only when their
            # independently selected observed representative station matches.
            representative["block_station"] = str(
                representative["station"]
            ).strip().upper()
            reduced.append(representative)

    duration_mass = defaultdict(float)
    for scenario in reduced:
        duration_mass[int(scenario["duration"])] += float(scenario["probability"])
    expected_mass = probabilities.groupby("duration")["p_scenario"].sum().to_dict()
    for duration, expected in expected_mass.items():
        if not np.isclose(duration_mass[int(duration)], float(expected), atol=1e-12):
            raise AssertionError(
                f"duration {duration} probability changed during reduction: "
                f"{duration_mass[int(duration)]} != {expected}")
    if not np.isclose(sum(duration_mass.values()), 1.0, atol=1e-12):
        raise AssertionError(
            f"reduced scenario probabilities sum to {sum(duration_mass.values())}")

    station_counts = defaultdict(int)
    for scenario in reduced:
        station_counts[str(scenario["station"]).strip().upper()] += 1
    for scenario in reduced:
        scenario["selected_duration_count"] = station_counts[
            str(scenario["station"]).strip().upper()
        ]
    return reduced


def load_core(window_hour: int, module_name: str = "tsb_core"):
    """원본 모듈을 로드한다. 창 시각은 import 전에 환경변수로 넣어야 한다."""
    if module_name in sys.modules:
        module = sys.modules[module_name]
        _require_scenario_logic(module)
        return module
    os.environ.setdefault("TSB_VERBOSE", "0")
    # sklearn/MKL 은 스레드보다 청크가 적으면 Windows 에서 누수 경고를 낸다.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ["TSB_START_HOURS"] = str(int(window_hour))
    # core 는 stage2_adaptive 등을 bare import 하므로 루트가 sys.path 에 있어야 한다.
    if BASE_DIR not in sys.path:
        sys.path.insert(0, BASE_DIR)
    path = os.path.join(BASE_DIR, CORE_FILENAME)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    _require_scenario_logic(module)
    return module


@dataclass
class BuildReport:
    """무엇이 만들어졌는지. 논문 표와 사전점검에 그대로 쓴다."""

    window_hour: int
    reduction: str
    k: int
    n_raw: int
    n_stations_raw: int
    n_blocks: int
    n_scenarios: int
    probability_sum: float
    restored_zero_impact: int
    macro_columns: int
    n_trips_union: int
    cells_by_block: dict = field(default_factory=dict)
    var_keys: dict = field(default_factory=dict)
    seconds: dict = field(default_factory=dict)

    def format(self) -> str:
        lines = [
            f"window {self.window_hour}  reduction={self.reduction}  K={self.k}",
            f"  원시 시나리오      {self.n_raw:,}  (사고역 {self.n_stations_raw:,})",
            f"  블록 / 시나리오    {self.n_blocks:,} / {self.n_scenarios:,}",
            f"  확률합 Σp          {self.probability_sum:.6f}",
        ]
        if self.restored_zero_impact:
            lines.append(
                f"  0임팩트 복원       {self.restored_zero_impact:,} "
                f"(resource-only)")
        lines += [
            f"  물리 trip (합집합) {self.n_trips_union:,}",
            f"  매크로열           {self.macro_columns:,}",
        ]
        if self.var_keys:
            lines.append("  캐시 키            " + "  ".join(
                f"{name}={count:,}" for name, count in sorted(self.var_keys.items())))
        shapes = defaultdict(int)
        for cells in self.cells_by_block.values():
            shapes[tuple(sorted(cells))] += 1
        if shapes:
            lines.append("  정보셀 구성")
            for shape, count in sorted(shapes.items(), key=lambda kv: -kv[1]):
                lines.append(f"    {str(list(shape)):<34} 블록 {count:,}개")
        if self.seconds:
            lines.append("  소요(s)            " + "  ".join(
                f"{name}={value:.1f}" for name, value in self.seconds.items()))
        return "\n".join(lines)


def _config_from_core(core, **overrides) -> cg.CGConfig:
    """CGConfig 를 원본 상수에서 만든다. 하드코딩하면 조용히 어긋난다."""
    values = dict(
        sigma=2,                       # solve_milp 내부 상수
        disrupt_buffer=int(core.DISRUPT_BUFFER),
        bus_capacity=float(core.BUS_CAPACITY),
        train_capacity=float(core.TRAIN_CAPACITY),
        min_trip_boarding=float(core.MIN_TRIP_BOARDING),
        max_total_buses=int(core.MAX_TOTAL_BUSES),
        mip_gap=float(core.MIP_GAP),
        time_limit=float(core.TIME_LIMIT),
        ignore_names=True,
    )
    values.update(overrides)
    return cg.CGConfig(**values)


def _restore_zero_impact_children(core, reduced, buffer):
    """블록에서 빠진 duration 을 승객흐름 없는 resource-only 자식으로 복원.

    Z_D = 0 이라 목적함수 기여는 없지만, 배차시각에는 그 세계와 구분이 안 되므로
    공유 y 와 차고지 점유가 살아 있어야 비예견성이 성립한다.

    피해는 duration 에 대해 단조라고 보고, 관측된 최대 duration 보다 짧은 것만
    복원한다. 최장 duration 은 노선 선택의 기준이라 비면 안 된다.
    """
    durations = [int(d) for d in core.DURATIONS]
    weight = {d: float(core.DURATION_WEIGHT.get(d, 1.0 / len(durations)))
              for d in durations}
    by_block = defaultdict(dict)
    for scenario in reduced:
        block = scenario.get("block_station")
        if block is None:
            continue
        by_block[block][int(scenario["duration"])] = scenario

    restored = []
    for block, per_duration in by_block.items():
        if not per_duration:
            continue
        longest = max(per_duration)
        template = per_duration[longest]
        station_probability = float(template.get(
            "prob_station", template.get("probability", 0.0)))
        for duration in durations:
            if duration in per_duration or duration > longest:
                continue
            start = int(template["DISRUPT_START_T"])
            child = {
                "id": f"{template['id']}__zero{duration}",
                "logic_version": template.get(
                    "logic_version", EXPECTED_SCENARIO_LOGIC_VERSION),
                "station": template.get("station", block),
                "block_station": block,
                "duration": duration,
                "DISRUPT_START_T": start,
                "DISRUPT_END_T": start + duration,
                "DISPATCH_END_MIN": start + duration + int(buffer),
                "Omega_m": [],
                "theta": {},
                "spare_capacity": {},
                "canceled_trains": set(),
                "impact_by_station": {},
                "probability": station_probability * weight[duration],
                "prob_station": station_probability,
                "cluster_id": template.get("cluster_id"),
                "representative_id": template.get("representative_id"),
                "zero_impact_branch": True,
            }
            restored.append(child)
    return restored


def _validate_scenarios(scenarios) -> None:
    """CGProblem.__post_init__ 이 검사하지 않는 것들.

    Omega_m 이 없으면 흐름 제약이 하나도 안 붙고, spare_capacity 가 없으면
    모든 열차 용량이 0 이 된다. 둘 다 예외 없이 그럴듯한 숫자를 낸다.
    """
    required = ("id", "duration", "DISRUPT_START_T", "bus_trips")
    for scenario in scenarios:
        missing = [key for key in required if key not in scenario]
        if missing:
            raise ValueError(f"{scenario.get('id', '?')}: 시나리오 키 누락 {missing}")
        if "Omega_m" not in scenario:
            raise ValueError(f"{scenario['id']}: Omega_m 없음 — 흐름 제약이 사라진다")
        if "spare_capacity" not in scenario:
            raise ValueError(
                f"{scenario['id']}: spare_capacity 없음 — 열차 용량이 전부 0 이 된다")
        actual_logic = scenario.get("logic_version")
        if actual_logic != EXPECTED_SCENARIO_LOGIC_VERSION:
            raise ValueError(
                f"{scenario['id']}: train-cancellation logic {actual_logic!r}; "
                f"expected {EXPECTED_SCENARIO_LOGIC_VERSION!r}")
        if scenario.get("zero_impact_branch"):
            continue
        if not scenario["Omega_m"]:
            raise ValueError(
                f"{scenario['id']}: Omega_m 이 비었는데 zero_impact_branch 가 아니다")


def build_cg_problem(window_hour: int, *, reduction: str | None = None,
                     k: int | None = None, env=None,
                     disrupt_buffer: int = 20,
                     restore_zero_impact: bool = False,
                     verbose: bool = True):
    """원 파이프라인 2~5 단계를 그대로 밟아 CGProblem 을 만든다.

    반환 ``(problem, config, report)``.
    """
    core = load_core(window_hour)
    if int(disrupt_buffer) < 0:
        raise ValueError("disrupt_buffer must be nonnegative")
    # This builder is callable without the root runner, so do not inherit the
    # legacy core module's process-global default implicitly.
    core.DISRUPT_BUFFER = int(disrupt_buffer)
    log = (lambda text: print(text)) if verbose else (lambda text: None)
    seconds = {}

    reduction = (reduction or core.REDUCTION).strip().lower()
    if reduction not in ("station", "duration"):
        raise ValueError(f"알 수 없는 축소 모드: {reduction}")
    if k is None:
        k = core.K_STATIONS if reduction == "station" else DEFAULT_K_PER_DURATION
    k = int(k)
    # reduce_scenarios 는 호출 시점의 모듈 전역을 읽는다.
    core.REDUCTION = reduction
    core.K_STATIONS = k
    core.K_PER_DURATION = k

    started = time.time()
    schedule = core.load_schedule(core.FILE_SCHEDULE)
    od = core.load_od(core.FILE_OD)
    bus_tt = core.load_distance_matrix(core.FILE_DISTANCE)
    core.FILE_PREDEFINED_ROUTES = ROUTE_FILE
    route_lib = core.load_route_library(core.FILE_PREDEFINED_ROUTES)
    depots = _load_candidate_depots()
    wdata = core.build_window_data(window_hour, schedule, od)
    seconds["load"] = time.time() - started

    started = time.time()
    raw = core.load_disruption_scenarios(window_hour, wdata)
    if not raw:
        raise ValueError(f"window {window_hour}: 장애 시나리오 없음")
    seconds["scenarios"] = time.time() - started

    started = time.time()
    if reduction == "duration":
        reduced = reduce_duration_scenarios_kmeans(raw, k, window_hour)
    else:
        # Retained only for the separate common-station-block experiment.
        reduced = core.reduce_scenarios(raw, k, wdata)
    if not reduced:
        raise ValueError(f"window {window_hour}: 축소 결과가 비었다")
    # Some legacy reducers synthesize zero-impact children with no post-end
    # buffer.  Normalize every branch before building the common trip pool so
    # passenger and resource-only worlds use exactly the same policy cutoff.
    for scenario in reduced:
        scenario["DISPATCH_END_MIN"] = (
            int(scenario["DISRUPT_END_T"]) + int(disrupt_buffer))
    seconds["reduce"] = time.time() - started

    config = _config_from_core(core)
    restored_count = 0
    if restore_zero_impact:
        if reduction != "station":
            raise ValueError("0임팩트 복원은 역 블록 축소에서만 의미가 있다")
        children = _restore_zero_impact_children(
            core, reduced, config.disrupt_buffer)
        reduced = list(reduced) + children
        restored_count = len(children)

    # 역 블록이면 30/60/90 이 같은 trip 집합을 써야 y 를 공유할 수 있다.
    started = time.time()
    block_trips = core.build_block_trips(
        reduced, window_hour, depots, bus_tt, route_lib)
    for scenario in reduced:
        sid = scenario["id"]
        if sid in block_trips:
            selected, trips = block_trips[sid]
        else:
            library, _stats = core.build_scenario_route_library(
                scenario, window_hour, depots, bus_tt, route_lib)
            selected = core.select_routes(scenario, depots, library, bus_tt)
            trips = core.generate_bus_trips(scenario, selected, library)
        scenario["bus_trips"] = trips
        scenario["selected_routes"] = selected
    seconds["trips"] = time.time() - started

    started = time.time()
    full_caches = {}
    var_keys = defaultdict(int)
    for scenario in reduced:
        # Build support once from the complete physical trip union.  Bus-load
        # minutes are reconstructed lazily as one continuous (trip, OD)
        # ledger by the standalone engine, so materialising all candidate l
        # keys here wastes memory without changing the model.
        cache = core.build_stage2_cache(
            scenario, wdata, scenario["bus_trips"],
            include_bus_load_keys=False)
        cache = cg.compact_full_pool_cache(
            cache, scenario, wdata, sigma=config.sigma,
            preserve_trip_ids=True)
        full_caches[str(scenario["id"])] = cache
        for name in ("b_var_keys", "a_var_keys", "l_var_keys",
                     "q_var_keys", "w_var_keys"):
            var_keys[name.replace("_var_keys", "")] += len(cache.get(name, ()))
        del scenario["selected_routes"]
    seconds["cache"] = time.time() - started

    _validate_scenarios(reduced)
    problem = cg.CGProblem(
        depots=tuple(depots), scenarios=tuple(reduced),
        full_caches=full_caches, wdata=wdata, env=env,
        travel_times=bus_tt)

    blocks = defaultdict(set)
    trips_union = set()
    for scenario in reduced:
        blocks[str(scenario.get("block_station", scenario["id"]))].add(
            int(scenario["duration"]))
        trips_union.update(str(tid) for tid in scenario["bus_trips"])
    cells_by_block = {}
    for block, durations in blocks.items():
        found = set()
        sorted_durations = tuple(sorted(durations))
        start = next(int(s["DISRUPT_START_T"]) for s in reduced
                     if str(s.get("block_station", s["id"])) == block)
        for scenario in reduced:
            if str(scenario.get("block_station", scenario["id"])) != block:
                continue
            for trip in scenario["bus_trips"].values():
                for cell in cg.info_cells(int(trip["DispatchStart"]), start,
                                          sorted_durations,
                                          config.disrupt_buffer):
                    found.add(cell)
        cells_by_block[block] = sorted(found)

    report = BuildReport(
        window_hour=int(window_hour), reduction=reduction, k=k,
        n_raw=len(raw),
        n_stations_raw=len({s["station"] for s in raw}),
        n_blocks=len(blocks), n_scenarios=len(reduced),
        probability_sum=sum(float(s.get("probability", 0.0)) for s in reduced),
        restored_zero_impact=restored_count,
        macro_columns=cg.count_macro_columns(problem, config),
        n_trips_union=len(trips_union),
        cells_by_block=cells_by_block, var_keys=dict(var_keys),
        seconds=seconds,
    )
    if verbose:
        log(report.format())
        if reduction == "duration":
            log("  [주의] duration 모드는 블록이 없어 정보셀이 duration 하나로 "
                "퇴화한다. 비예견성이 사라지므로 CG 의 전제가 성립하지 않는다.")
        degenerate = [block for block, cells in cells_by_block.items()
                      if all(len(cell) == 1 for cell in cells)]
        if reduction == "station" and degenerate:
            log(f"  [주의] 정보셀이 전부 단일 duration 인 블록 {len(degenerate)}개. "
                f"restore_zero_impact=True 로 트리를 복원할 수 있다.")
    return problem, config, report


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(
        description="cg_stage1 용 CGProblem 을 원본 K-means 축소로 만든다")
    parser.add_argument("--window", type=int, default=17)
    parser.add_argument("--reduction", choices=("station", "duration"),
                        default="duration")
    parser.add_argument("--k", type=int, default=DEFAULT_K_PER_DURATION)
    parser.add_argument("--restore-zero-impact", action="store_true")
    parser.add_argument("--buffer", type=int, default=20,
                        help="장애 종료 후 배차 허용 시간(분), 기본 20")
    args = parser.parse_args(argv)

    sys.stdout.reconfigure(encoding="utf-8")
    build_cg_problem(
        args.window, reduction=args.reduction, k=args.k,
        disrupt_buffer=args.buffer,
        restore_zero_impact=args.restore_zero_impact, verbose=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
