"""
build_local_routes.py — inputs.local_routes 에 도로 소요시간을 붙여 노선표로 만든다

inputs.local_routes 는 열 하나가 로컬(정차형) 노선 하나이고 정류장 순서만 담겨 있다.
여기에 inputs.road_times 의 **도로 주행시간**을 계산해
inputs.route_topology 와 동일한 스키마의 CSV 를 만든다.

왜 도로시간인가
--------------
inputs.route_topology 의 CumTime 은 철도 소요시간이다. 브리징 버스는 도로를 달리므로
그 값으로는 실현가능성을 판정할 수 없다. 급행(build_express_variants.py)도 도로시간을
쓰므로, 두 노선 타입이 같은 근거 위에서 비교되어야 MILP 후보집합이 일관된다.

정차시간
--------
넣지 않는다. 순수 도로 주행시간만 쓴다 (급행과 동일 규약).

출력: Local_route_Detail.csv
"""
import os
import sys
from pathlib import Path

import pandas as pd

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_ROOT.parent))
from bbconfig import cfg  # noqa: E402

ROUTE_DIR = Path(
    os.environ.get("PAPER_ROUTE_DIR") or cfg.results_dir / cfg.get("paths.routes")
).resolve()
FILE_IN = cfg.path("inputs.local_routes")
FILE_DISTANCE = cfg.path("inputs.road_times")
FILE_OUT = ROUTE_DIR / cfg.get("routes.local_detail_file")

# 로컬 노선은 corridor 트랙(inputs.corridors)의 연속 구간이며 입력 순서를 그대로 쓴다.
# 노선별로 제외할 정류장은 없다 (빈 매핑).
DROP = {}


def load_road_times(path):
    frame = pd.read_csv(
        path, usecols=["Origin_Code", "Destination_Code", "Duration_min"]
    ).dropna()
    return {
        (str(origin), str(destination)): float(minutes)
        for origin, destination, minutes in zip(
            frame["Origin_Code"], frame["Destination_Code"],
            frame["Duration_min"],
        )
    }


def road_time(times, origin, destination):
    if origin == destination:
        return 0.0
    return times.get((origin, destination), times.get((destination, origin)))


def route_road_cum(stops, times):
    cumulative = [0.0]
    for origin, destination in zip(stops, stops[1:]):
        minutes = road_time(times, origin, destination)
        if minutes is None:
            return None
        cumulative.append(cumulative[-1] + minutes)
    return cumulative


def main():
    ROUTE_DIR.mkdir(parents=True, exist_ok=True)
    rt = load_road_times(FILE_DISTANCE)

    df = pd.read_csv(FILE_IN)
    rows = []
    print(f"{'노선':6s} {'정거장':>4s} {'소요(분)':>8s}  경로")
    print("-" * 78)

    for line in df.columns:
        stops = [str(s).strip() for s in df[line] if pd.notna(s) and str(s).strip()]
        drop = DROP.get(line, set())
        if drop:
            stops = [s for s in stops if s not in drop]
        if len(stops) < 2:
            print(f"{line:6s}  정류장 부족, 스킵")
            continue

        cum = route_road_cum(stops, rt)
        if cum is None:
            miss = [(stops[i], stops[i + 1]) for i in range(len(stops) - 1)
                    if road_time(rt, stops[i], stops[i + 1]) is None]
            print(f"{line:6s}  도로 연결 없음 {miss}, 스킵")
            continue

        total = cum[-1]
        skipped = []
        for i, (s, c) in enumerate(zip(stops, cum)):
            rows.append({
                "Route": f"{line}_LOC",
                "BaseLine": line,
                "Type": "LOC",
                "StopSeq": i + 1,
                "StationCode": s,
                "CumTime_min": int(round(c)),
                "SegTime_min": 0 if i == 0 else int(round(c - cum[i - 1])),
                "SkippedStations": "",
                "OrigLine_NumStops": len(stops),
                "OrigLine_TotalTime": int(round(total)),
                "Route_TotalTime": int(round(total)),
                "TimeSaving_pct": 0.0,
                "DisruptionDemand": 0.0,
            })
        print(f"{line:6s} {len(stops):4d} {total:8.0f}  {'-'.join(stops)}")

    out = pd.DataFrame(rows)
    out.to_csv(FILE_OUT, index=False)
    print("-" * 78)
    print(f"저장: {FILE_OUT}  ({out['Route'].nunique()}개 노선, {len(out)}행)")

    t = out.groupby("Route")["Route_TotalTime"].first()
    horizon = int(cfg.get("time.horizon_min"))
    print(f"소요시간: 평균 {t.mean():.0f}분 / 중앙 {t.median():.0f} / "
          f"최소 {t.min()} / 최대 {t.max()}")
    print(f"{horizon}분 초과: {(t > horizon).sum()}개  " +
          (", ".join(t[t > horizon].index) if (t > horizon).any() else ""))


if __name__ == "__main__":
    main()
