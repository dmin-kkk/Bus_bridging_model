"""Generate the synthetic input data set of the reproducibility package.

    python scripts/make_synthetic_inputs.py [--seed 20261118] [--out data/synthetic]

The script writes every input file named in ``config/profiles/synthetic/inputs.toml``
with the same columns, value formats and encodings as the corresponding file of the
paper profile (``config/profiles/paper/inputs.toml``), for a fictitious 12-station,
2-corridor regional rail network.  All values are produced here from the design
constants in ``scripts/synthetic_design.toml`` and a seeded random generator; no value is
copied, resampled or perturbed from the real (restricted) inputs.  The output is
deterministic: the same seed gives byte-identical files (``MANIFEST.sha256``).
"""
from __future__ import annotations

import argparse
import hashlib
import math
import re
import sys
import tomllib
import zipfile
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import bbconfig  # noqa: E402

DESIGN_FILE = Path(__file__).with_name("synthetic_design.toml")
with DESIGN_FILE.open("rb") as _handle:
    DESIGN = tomllib.load(_handle)

PROFILE = bbconfig.load("synthetic")
NEWLINE = "\n"
FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)

# ----------------------------------------------------------------------------------
# Column lists = schema of the real files (column names only).
# ----------------------------------------------------------------------------------
SCHEDULE_COLUMNS = [
    "Unnamed: 0", "TimetableDate", "Trip", "Seq", "LineDescription", "Line", "Direction",
    "Origin", "Destination", "ej", "ea", "tob", "dis_inc", "dis_ex_melb", "penalties",
    "StationID", "SchDepart", "SchArrive", "ActDepart", "ActArrive", "Dep_Delay(s)",
    "Arr_Delay(s)", "SchDwell(s)", "ActDwell(s)", "Join_rate(p/s)", "Alight_rate(p/s)",
    "Name", "Stop", "ItemId", "ItemCreated", "ItemDescription", "TripId", "TripDate",
    "PrimaryServiceForItem", "ServiceStatusCode", "ServiceShortStatusCode",
    "ItemTypeTaxonomy", "OrgText", "PathLocationSeqNo", "LocationCode", "EventType",
    "DelayMinutes", "ActualDelayMinutes", "Attribution_Description", "VehicleID",
    "VehicleName", "VehicleClassCode", "VehicleClassDescription", "VehicleCategoryCode",
    "VehicleCategoryDescription", "Corridor", "Direction.1", "KeyValue", "Line.1",
    "TimetableDate.1", "MasterRoute", "SchedOrig", "SchedOrigName", "SchedDest",
    "SchedDestName", "SchedPathLocCount", "ActualPathLocCount", "IsPunctual", "IsReliable",
    "Index", "STATION", "Avg_Dep_Delay(s)_per_Trip_Station",
    "Avg_Arr_Delay(s)_per_Trip_Station", "mean_arr_delay_punctual",
    "std_arr_delay_punctual", "mean_dep_delay_punctual_x", "std_dep_delay_punctual_x",
    "Prior_Depdelay", "Prior_arrdelay", "x", "y", "is_delayed", "dep_hour",
    "station_delay_rate", "trip_delay_rate", "hourly_delay_rate", "ItemCreated_dt",
    "ItemCreated_time_only", "LINE_NAME_count", "line_connected_count", "Hour", "DayOfWeek",
    "IsWeekend", "IsPeak", "AverageDelayByItemType", "avg_ej", "avg_ea",
    "avg_join_rate(p/s)", "avg_alight_rate(p/s)", "BetweennessCentrality",
    "Corrected_Dep_Delay(s)", "Prior_Depdelay_Corrected", "mean_dep_delay_punctual",
    "std_dep_delay_punctual", "TopTaxonomy", "AvgDelayByTopTaxonomy",
    "Corrected_Arr_Delay(s)", "Prior_Corrected_Arr_Delay(s)", "RunTime",
]
# Incident columns of the timetable: blank (one space) on every row, as on the
# no-incident rows of the real file.
SCHEDULE_INCIDENT_BLANK = [
    "ItemId", "ItemCreated", "ItemDescription", "TripId", "TripDate",
    "PrimaryServiceForItem", "ServiceStatusCode", "ServiceShortStatusCode",
    "ItemTypeTaxonomy", "OrgText", "PathLocationSeqNo", "LocationCode", "EventType",
    "DelayMinutes", "Attribution_Description", "VehicleID", "VehicleName",
    "VehicleClassCode", "VehicleClassDescription", "VehicleCategoryCode",
    "VehicleCategoryDescription", "Corridor", "Direction.1", "KeyValue", "Line.1",
    "TimetableDate.1",
]
RAW_INCIDENT_COLUMNS = [
    "ItemId", "ItemCreated", "ItemDescription", "TripId", "TripDate",
    "PrimaryServiceForItem", "ServiceStatusCode", "ServiceShortStatusCode",
    "ItemTypeTaxonomy", "OrgText", "PathLocationSeqNo", "LocationCode", "EventType",
    "DelayMinutes", "ActualDelayMinutes", "Attribution_Description", "VehicleID",
    "VehicleName", "VehicleClassCode", "VehicleClassDescription", "VehicleCategoryCode",
    "VehicleCategoryDescription", "Corridor", "Direction", "KeyValue", "Line",
    "TimetableDate",
]
AUGMENTED_COLUMNS = [
    "Year", "IncidentID", "EventDate", "EventDateSource", "Hour", "Station",
    "MaxDelayMinutes", "StationCorridor", "CorridorSource", "CorridorIsImputed",
    "CorridorEvidence", "TripIDs", "RawLines", "IsWeekend", "DailyIsRain",
]
COHORT_COLUMNS = ["IncidentID", "Eligible", "Hour", "Station", "RealizedState"]
WEATHER_COLUMNS = [
    "WeatherLocation", "RequestedLat", "RequestedLong", "WeatherGridLat",
    "WeatherGridLong", "WeatherGridElevationM", "WeatherHour", "TemperatureC",
    "ApparentTemperatureC", "RelativeHumidityPct", "PrecipitationMm", "RainMm",
    "CloudCoverPct", "WeatherCode", "WindSpeed10mKmh", "WeatherCategory", "IsRain",
]
ROUTE_COLUMNS = [
    "Route", "BaseLine", "Type", "StopSeq", "StationCode", "CumTime_min", "SegTime_min",
    "SkippedStations", "OrigLine_NumStops", "OrigLine_TotalTime", "Route_TotalTime",
    "TimeSaving_pct", "DisruptionDemand",
]
# Files whose real counterpart starts with a UTF-8 byte-order mark.
BOM_KEYS = {"road_times", "rail_travel_times", "route_topology", "incidents_gt10",
            "incidents_ge20", "weather"}


# ----------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------
def clock(minutes: int) -> str:
    """Minutes after midnight -> 'H:MM:SS' (no leading zero, as in the real timetable)."""
    if not 0 <= minutes < 24 * 60:
        raise ValueError(f"time outside the service day: {minutes} min")
    return f"{minutes // 60}:{minutes % 60:02d}:00"


def clock_seconds(seconds: int) -> str:
    seconds = int(seconds)
    if not 0 <= seconds < 24 * 3600:
        raise ValueError(f"time outside the service day: {seconds} s")
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def token(rng: np.random.Generator, prefix: str = "tok") -> str:
    """Meaningless letters-only token (no digits, so it can never look like an identifier)."""
    return prefix + "_" + "".join(chr(ord("a") + int(k)) for k in rng.integers(0, 26, size=4))


def write_csv(frame: pd.DataFrame, path: Path, bom: bool) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig" if bom else "utf-8",
                 lineterminator=NEWLINE)


def normalize_xlsx(path: Path) -> None:
    """Rewrite an .xlsx with fixed zip timestamps and document dates (determinism)."""
    with zipfile.ZipFile(path) as archive:
        items = [(info.filename, archive.read(info.filename)) for info in archive.infolist()]
    stamp = DESIGN["output"]["document_timestamp"].encode()
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in items:
            if name == "docProps/core.xml":
                for tag in (b"created", b"modified"):
                    data = re.sub(
                        rb"(<dcterms:" + tag + rb"[^>]*>)[^<]*(</dcterms:" + tag + rb">)",
                        rb"\g<1>" + stamp + rb"\g<2>", data)
            info = zipfile.ZipInfo(name, date_time=FIXED_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, data)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ----------------------------------------------------------------------------------
# Network
# ----------------------------------------------------------------------------------
class Network:
    def __init__(self, rng: np.random.Generator):
        net = DESIGN["network"]
        geo = DESIGN["geometry"]
        self.hub = net["hub"]
        self.central = net["central"]
        self.corridors = net["corridors"]          # list of tables
        self.names = {self.hub: net["hub_name"], self.central: net["central_name"]}
        self.xy = {self.hub: (0.0, 0.0), self.central: (geo["hub_to_central_km"], 0.0)}
        self.line_stops: dict[str, list[str]] = {}
        self.station_corridor: dict[str, str] = {}
        cx, cy = self.xy[self.central]
        for corridor in self.corridors:
            theta = math.radians(corridor["bearing_deg"])
            ux, uy = math.cos(theta), math.sin(theta)
            nx, ny = -uy, ux
            along = 0.0
            stops = [self.hub, self.central]
            for k, base in enumerate(geo["link_km"], start=1):
                along += base + rng.uniform(-geo["link_jitter_km"], geo["link_jitter_km"])
                offset = rng.uniform(-geo["lateral_jitter_km"], geo["lateral_jitter_km"])
                code = f"{corridor['prefix']}{k}"
                self.xy[code] = (cx + along * ux + offset * nx, cy + along * uy + offset * ny)
                self.names[code] = f"{corridor['label']} {k}"
                self.station_corridor[code] = corridor["id"]
                stops.append(code)
            self.line_stops[corridor["line"]] = stops
        self.stations = [self.hub, self.central] + [
            s for c in self.corridors for s in self.line_stops[c["line"]][2:]]
        self.central_group = {self.hub, self.central}
        self.line_of_corridor = {c["id"]: c["line"] for c in self.corridors}
        self.corridor_of_line = {c["line"]: c["id"] for c in self.corridors}
        self.label_of_corridor = {c["id"]: c["label"] for c in self.corridors}
        self.line_name = {c["line"]: c["line_name"] for c in self.corridors}
        # rail run time per physical link [min], shared links get one value
        self.run: dict[frozenset, int] = {}
        lo, hi = geo["rail_link_min"]
        for stops in self.line_stops.values():
            for a, b in zip(stops, stops[1:]):
                key = frozenset((a, b))
                if key not in self.run:
                    minutes = self.euclid(a, b) / geo["rail_speed_kmh"] * 60
                    minutes += rng.uniform(0.0, geo["rail_slack_min"])
                    self.run[key] = int(min(hi, max(lo, round(minutes))))
        # lat/long: the plane is anchored in open ocean (no city), local tangent scale
        lat0, lon0 = geo["origin_lat"], geo["origin_lon"]
        self.latlon = {
            s: (round(lat0 + y / 110.574, 6),
                round(lon0 + x / (111.320 * math.cos(math.radians(lat0))), 6))
            for s, (x, y) in self.xy.items()}

    def euclid(self, a: str, b: str) -> float:
        (xa, ya), (xb, yb) = self.xy[a], self.xy[b]
        return math.hypot(xa - xb, ya - yb)

    def lines_at(self, station: str) -> list[str]:
        return [line for line, stops in self.line_stops.items() if station in stops]

    def corridor_label(self, station: str) -> str:
        if station in self.central_group:
            return DESIGN["incidents"]["central_corridor_label"]
        return self.label_of_corridor[self.station_corridor[station]]

    def rail_position(self) -> dict[str, float]:
        """Scheduled rail minutes from the hub (run + dwell), per station."""
        dwell = DESIGN["service"]["dwell_min"]
        pos = {self.hub: 0.0}
        for stops in self.line_stops.values():
            t = 0.0
            for a, b in zip(stops, stops[1:]):
                t += self.run[frozenset((a, b))]
                pos[b] = t
                t += dwell
        return pos

    def betweenness(self) -> dict[str, float]:
        """Normalized betweenness centrality of the (tree-shaped) rail graph."""
        adjacency = defaultdict(set)
        for stops in self.line_stops.values():
            for a, b in zip(stops, stops[1:]):
                adjacency[a].add(b)
                adjacency[b].add(a)
        n = len(self.stations)
        result = {}
        for v in self.stations:
            sizes = []
            for start in adjacency[v]:
                seen, stack = {v, start}, [start]
                while stack:
                    node = stack.pop()
                    for nxt in adjacency[node]:
                        if nxt not in seen:
                            seen.add(nxt)
                            stack.append(nxt)
                sizes.append(len(seen) - 1)
            pairs = (sum(sizes) ** 2 - sum(s * s for s in sizes)) / 2
            result[v] = round(pairs / ((n - 1) * (n - 2) / 2), 10)
        return result


# ----------------------------------------------------------------------------------
# Timetable
# ----------------------------------------------------------------------------------
def build_trips(net: Network) -> list[dict]:
    service = DESIGN["service"]
    dwell = service["dwell_min"]
    trips = []
    for corridor in net.corridors:
        line = corridor["line"]
        stops_down = net.line_stops[line]
        departures = []   # (direction, minute after midnight at the first stop)
        for hour in range(service["first_hour"], service["last_departure_hour"] + 1):
            departures.append(("D", hour * 60 + corridor["down_minute"]))
            departures.append(("U", hour * 60 + corridor["up_minute"]))
        for hhmm in corridor["peak_up"]:
            departures.append(("U", int(hhmm[:2]) * 60 + int(hhmm[3:])))
        for hhmm in corridor["peak_down"]:
            departures.append(("D", int(hhmm[:2]) * 60 + int(hhmm[3:])))
        departures.sort(key=lambda item: (item[1], item[0]))
        counters = {"D": 0, "U": 0}
        for direction, start in departures:
            counters[direction] += 1
            offset = 2 * counters[direction] - (1 if direction == "D" else 0)
            number = corridor["trip_base"] + offset
            stops = stops_down if direction == "D" else stops_down[::-1]
            events, t = [], start
            for idx, station in enumerate(stops):
                if idx == 0:
                    arrive, depart = None, t
                else:
                    t += net.run[frozenset((stops[idx - 1], station))]
                    arrive = t
                    depart = t + dwell if idx < len(stops) - 1 else None
                    if depart is not None:
                        t = depart
                events.append({"seq": idx + 1, "station": station,
                               "arrive": arrive, "depart": depart})
            if events[-1]["arrive"] >= 24 * 60:
                raise ValueError(f"trip {number} runs past midnight; shorten the service day")
            trips.append({"trip": number, "line": line, "direction": direction,
                          "events": events})
    return trips


def trips_departing(trips: list[dict]) -> dict:
    """(line, direction, station, hour) -> list of trips that depart there in that hour."""
    index = defaultdict(list)
    for trip in trips:
        for event in trip["events"][:-1]:
            index[(trip["line"], trip["direction"], event["station"],
                   event["depart"] // 60)].append(trip)
    return index


# ----------------------------------------------------------------------------------
# Demand
# ----------------------------------------------------------------------------------
def build_od(net: Network, trips: list[dict], rng: np.random.Generator):
    demand = DESIGN["demand"]
    hours = list(range(demand["first_hour"], demand["last_hour"] + 1))
    w_in = np.asarray(demand["profile_inbound"], dtype=float)
    w_out = np.asarray(demand["profile_outbound"], dtype=float)
    if len(w_in) != len(hours) or len(w_out) != len(hours):
        raise ValueError("demand profiles must have one weight per hour")
    w_in, w_out = w_in / w_in.sum(), w_out / w_out.sum()
    w_cross = 0.5 * (w_in + w_out)
    mass = {s: demand["mass"][s] * rng.uniform(1 - demand["mass_jitter"],
                                               1 + demand["mass_jitter"])
            for s in net.stations}
    pos = net.rail_position()
    transfer = demand["transfer_penalty_min"]

    def corridor_of(s):
        return None if s in net.central_group else net.station_corridor[s]

    def rail_minutes(i, j):
        ci, cj = corridor_of(i), corridor_of(j)
        if ci is None or cj is None or ci == cj:
            return abs(pos[i] - pos[j])
        return (pos[i] - pos[net.central]) + (pos[j] - pos[net.central]) + transfer

    daily = {}
    for i in net.stations:
        for j in net.stations:
            if i == j:
                continue
            daily[(i, j)] = mass[i] * mass[j] * math.exp(-rail_minutes(i, j) / demand["decay_min"])
    hourly = {}
    for (i, j), value in daily.items():
        ci, cj = corridor_of(i), corridor_of(j)
        if ci is not None and cj is not None and ci != cj:
            profile = w_cross
        elif pos[j] < pos[i]:
            profile = w_in
        else:
            profile = w_out
        for k, hour in enumerate(hours):
            hourly[(i, j, hour)] = value * profile[k]

    loads = assign_loads(net, trips, hourly)
    peak = max(max(trip_loads["tob"].values()) for trip_loads in loads.values())
    scale = demand["target_peak_train_load"] / peak
    hourly = {key: value * scale for key, value in hourly.items()}
    loads = assign_loads(net, trips, hourly)
    rows = []
    for i in sorted(net.stations):
        for j in sorted(net.stations):
            for hour in hours:
                value = 0.0 if i == j else round(hourly[(i, j, hour)], 4)
                rows.append({"OriginID": i, "DestinationID": j, "Hour": hour,
                             "Trips_Hourly": value, "OriginName": 1, "DestinationName": 1})
    return pd.DataFrame(rows), loads, hourly


def assign_loads(net: Network, trips: list[dict], hourly: dict) -> dict:
    """Spread hourly OD flows over the trains that serve them (descriptive loads)."""
    index = trips_departing(trips)
    board = defaultdict(lambda: defaultdict(float))
    alight = defaultdict(lambda: defaultdict(float))
    pos = net.rail_position()

    def legs(i, j):
        if i in net.central_group and j in net.central_group:
            return [(i, j, list(net.line_stops))]
        ci = None if i in net.central_group else net.station_corridor[i]
        cj = None if j in net.central_group else net.station_corridor[j]
        if ci is None or cj is None or ci == cj:
            line = net.line_of_corridor[ci or cj]
            return [(i, j, [line])]
        return [(i, net.central, [net.line_of_corridor[ci]]),
                (net.central, j, [net.line_of_corridor[cj]])]

    for (i, j, hour), value in hourly.items():
        if value <= 0:
            continue
        for origin, destination, lines in legs(i, j):
            direction = "D" if pos[destination] > pos[origin] else "U"
            serving = [t for line in lines for t in index.get((line, direction, origin, hour), [])]
            if not serving:
                continue
            share = value / len(serving)
            for trip in serving:
                board[trip["trip"]][origin] += share
                alight[trip["trip"]][destination] += share
    loads = {}
    for trip in trips:
        on, tob = 0.0, {}
        ej, ea = {}, {}
        for event in trip["events"]:
            s = event["station"]
            ea[s] = alight[trip["trip"]].get(s, 0.0)
            ej[s] = board[trip["trip"]].get(s, 0.0)
            on = max(0.0, on - ea[s] + ej[s])
            tob[s] = on
        loads[trip["trip"]] = {"ej": ej, "ea": ea, "tob": tob}
    return loads


# ----------------------------------------------------------------------------------
# Files: network
# ----------------------------------------------------------------------------------
def network_frames(net: Network) -> dict[str, object]:
    geo = DESIGN["geometry"]
    out: dict[str, object] = {}
    rows = []
    for o in net.stations:
        for d in net.stations:
            km = round(geo["road_detour_factor"] * net.euclid(o, d), 2)
            rows.append({"Origin_Code": o, "Origin_Name": net.names[o],
                         "Destination_Code": d, "Destination_Name": net.names[d],
                         "Distance_km": km,
                         "Duration_min": round(km / geo["road_speed_kmh"] * 60, 2)})
    out["road_times"] = pd.DataFrame(rows).sort_values(
        ["Origin_Code", "Destination_Code"]).reset_index(drop=True)
    rows = []
    for corridor in net.corridors:
        stops = net.line_stops[corridor["line"]]
        for a, b in zip(stops, stops[1:]):
            rows.append({"Line": corridor["line"], "Origin_Code": a, "Origin_Name": net.names[a],
                         "Destination_Code": b, "Destination_Name": net.names[b],
                         "Travel_Time_Min": net.run[frozenset((a, b))]})
    out["rail_travel_times"] = pd.DataFrame(rows)
    out["corridors"] = pd.DataFrame({c["id"]: net.line_stops[c["line"]] for c in net.corridors})
    out["corridor_tracks"] = pd.DataFrame(
        [{"Track": c["id"], "Corridor": c["id"], "CorridorLabel": c["label"]}
         for c in net.corridors])
    out["depots"] = pd.DataFrame(
        [{"Code": s, "Station": net.names[s], "Latitude": net.latlon[s][0],
          "Longitude": net.latlon[s][1]} for s in DESIGN["network"]["depots"]])
    out["walking_times"] = pd.DataFrame(
        [{"StationCode": s, "WalkingMinutes": int(PROFILE.get("operations.walking_transfer_min"))}
         for s in sorted(net.stations)])
    width = max(len(stops) for stops in net.line_stops.values())
    out["local_routes"] = pd.DataFrame(
        {c["line"]: net.line_stops[c["line"]] + [None] * (width - len(net.line_stops[c["line"]]))
         for c in net.corridors})
    sheets = {}
    for corridor in net.corridors:
        sheets[corridor["line"]] = pd.DataFrame(
            [{"Code": s, "Name": net.names[s], "Lat": net.latlon[s][0], "Long": net.latlon[s][1]}
             for s in net.line_stops[corridor["line"]]])
    out["stations"] = sheets
    return out


def route_topology(net: Network, od_hourly: dict) -> pd.DataFrame:
    """Local route per line with scheduled rail times (topology file, as BusRoutes_Detail)."""
    dwell = DESIGN["service"]["dwell_min"]
    rows = []
    for corridor in net.corridors:
        line = corridor["line"]
        stops = net.line_stops[line]
        cum = [0]
        for a, b in zip(stops, stops[1:]):
            cum.append(cum[-1] + net.run[frozenset((a, b))] + (dwell if len(cum) > 1 else 0))
        on_route = set(stops)
        demand = round(sum(v for (i, j, _h), v in od_hourly.items()
                           if i in on_route and j in on_route), 1)
        for k, (station, c) in enumerate(zip(stops, cum)):
            rows.append({"Route": f"{line}_LOC", "BaseLine": line, "Type": "LOC",
                         "StopSeq": k + 1, "StationCode": station, "CumTime_min": c,
                         "SegTime_min": 0 if k == 0 else c - cum[k - 1],
                         "SkippedStations": None, "OrigLine_NumStops": len(stops),
                         "OrigLine_TotalTime": cum[-1], "Route_TotalTime": cum[-1],
                         "TimeSaving_pct": 0.0, "DisruptionDemand": demand})
    return pd.DataFrame(rows, columns=ROUTE_COLUMNS)


def timetable_frame(net: Network, trips: list[dict], loads: dict,
                    rng: np.random.Generator) -> pd.DataFrame:
    service = DESIGN["service"]
    peak_hours = set(DESIGN["incidents"]["peak_hours"])
    between = net.betweenness()
    km_from_hub = {}
    for stops in net.line_stops.values():
        dist = 0.0
        km_from_hub[stops[0]] = 0.0
        for a, b in zip(stops, stops[1:]):
            dist += net.euclid(a, b)
            km_from_hub[b] = dist
    reference = date.fromisoformat(service["timetable_date"])
    rows = []
    for trip in sorted(trips, key=lambda t: t["trip"]):
        events = trip["events"]
        first, last = events[0]["station"], events[-1]["station"]
        load = loads[trip["trip"]]
        dep_delay_prev, arr_delay_prev = 0, 0
        origin_delay = int(max(0, round(rng.normal(*service["origin_delay_s"]) / 10)) * 10)
        delay = origin_delay
        for k, event in enumerate(events):
            s = event["station"]
            arr, dep = event["arrive"], event["depart"]
            arr_delay = dep_delay = None
            if arr is not None:
                delay = int(max(-60, delay + round(rng.normal(0, service["link_delay_sd_s"]))))
                arr_delay = delay
            if dep is not None:
                dwell_extra = int(max(0, round(rng.normal(0, service["dwell_delay_sd_s"]))))
                dep_delay = max(0, (arr_delay if arr_delay is not None else delay)) + dwell_extra
                delay = dep_delay
            sch_dwell = None if (arr is None or dep is None) else float((dep - arr) * 60)
            act_dwell = None
            if sch_dwell is not None:
                act_dwell = float(sch_dwell + dep_delay - arr_delay)
            ej = int(round(load["ej"].get(s, 0.0)))
            ea = int(round(load["ea"].get(s, 0.0)))
            tob = int(round(load["tob"].get(s, 0.0)))
            dwell_s = sch_dwell if sch_dwell else 60.0
            prev = events[k - 1] if k else None
            run_time = None
            if prev is not None:
                run_time = float((arr - prev["depart"]) * 60)
            dis_inc = 0 if prev is None else int(round(net.euclid(prev["station"], s)))
            dep_hour = None if dep is None else float(dep // 60)
            rows.append({
                "TimetableDate": service["timetable_date"], "Trip": trip["trip"],
                "Seq": event["seq"], "LineDescription": net.line_name[trip["line"]],
                "Line": trip["line"], "Direction": trip["direction"], "Origin": first,
                "Destination": last, "ej": ej, "ea": ea, "tob": tob, "dis_inc": dis_inc,
                "dis_ex_melb": int(round(km_from_hub[s])), "penalties": 0, "StationID": s,
                "SchDepart": None if dep is None else clock(dep),
                "SchArrive": None if arr is None else clock(arr),
                "ActDepart": None if dep is None else clock_seconds(dep * 60 + dep_delay),
                "ActArrive": None if arr is None else clock_seconds(arr * 60 + arr_delay),
                "Dep_Delay(s)": None if dep is None else float(dep_delay),
                "Arr_Delay(s)": None if arr is None else float(arr_delay),
                "SchDwell(s)": sch_dwell, "ActDwell(s)": act_dwell,
                "Join_rate(p/s)": round(ej / dwell_s, 2), "Alight_rate(p/s)": round(ea / dwell_s, 2),
                "Name": net.names[s], "Stop": 1,
                "MasterRoute": f"{first}{last}", "SchedOrig": first,
                "SchedOrigName": net.names[first], "SchedDest": last,
                "SchedDestName": net.names[last], "SchedPathLocCount": len(events),
                "ActualPathLocCount": len(events), "Index": 1, "STATION": net.names[s],
                "Prior_Depdelay": float(dep_delay_prev), "Prior_arrdelay": float(arr_delay_prev),
                "x": net.latlon[s][1], "y": net.latlon[s][0],
                "is_delayed": bool((dep_delay or arr_delay or 0) > service["delayed_threshold_s"]),
                "dep_hour": dep_hour,
                "LINE_NAME_count": float(len(net.lines_at(s))),
                "line_connected_count": len(net.lines_at(s)),
                "Hour": dep_hour, "DayOfWeek": reference.weekday(),
                "IsWeekend": int(reference.weekday() >= 5),
                "IsPeak": int(dep_hour is not None and int(dep_hour) in peak_hours),
                "BetweennessCentrality": between[s],
                "Corrected_Dep_Delay(s)": None if dep is None else float(dep_delay),
                "Prior_Depdelay_Corrected": float(dep_delay_prev),
                "Corrected_Arr_Delay(s)": None if arr is None else float(arr_delay),
                "Prior_Corrected_Arr_Delay(s)": int(arr_delay_prev),
                "RunTime": run_time,
                "_dep_delay": dep_delay, "_arr_delay": arr_delay,
            })
            if dep_delay is not None:
                dep_delay_prev = dep_delay
            if arr_delay is not None:
                arr_delay_prev = arr_delay
    frame = pd.DataFrame(rows)
    frame.insert(0, "Unnamed: 0", np.arange(len(frame)))
    # trip/station delay statistics (descriptive columns)
    frame["Avg_Dep_Delay(s)_per_Trip_Station"] = frame["_dep_delay"].astype(float)
    frame["Avg_Arr_Delay(s)_per_Trip_Station"] = frame["_arr_delay"].astype(float)
    by_station = frame.groupby("StationID")
    for name, source in (("arr", "_arr_delay"), ("dep", "_dep_delay")):
        values = frame[source].astype(float)
        mean = by_station[source].transform(lambda x: x.astype(float).mean()).round(6)
        std = by_station[source].transform(lambda x: x.astype(float).std()).round(6)
        mean = mean.where(values.notna())
        std = std.where(values.notna())
        if name == "arr":
            frame["mean_arr_delay_punctual"], frame["std_arr_delay_punctual"] = mean, std
        else:
            frame["mean_dep_delay_punctual_x"], frame["std_dep_delay_punctual_x"] = mean, std
            frame["mean_dep_delay_punctual"], frame["std_dep_delay_punctual"] = mean, std
    delayed = frame["is_delayed"].astype(float)
    frame["station_delay_rate"] = delayed.groupby(frame["StationID"]).transform("mean").round(10)
    frame["trip_delay_rate"] = delayed.groupby(frame["Trip"]).transform("mean").round(10)
    hourly_rate = delayed.groupby(frame["dep_hour"]).transform("mean").round(10)
    frame["hourly_delay_rate"] = hourly_rate.where(frame["dep_hour"].notna())
    for column, source in (("avg_ej", "ej"), ("avg_ea", "ea"),
                           ("avg_join_rate(p/s)", "Join_rate(p/s)"),
                           ("avg_alight_rate(p/s)", "Alight_rate(p/s)")):
        frame[column] = frame.groupby("StationID")[source].transform("mean").round(6)
    punctual = frame.groupby("Trip")["_arr_delay"].transform(
        lambda x: int(x.dropna().iloc[-1] <= service["punctual_threshold_s"]))
    frame["IsPunctual"] = punctual.astype(int)
    frame["IsReliable"] = 1
    for column in SCHEDULE_INCIDENT_BLANK:
        frame[column] = " "
    for column in ("ActualDelayMinutes", "ItemCreated_dt", "ItemCreated_time_only",
                   "AverageDelayByItemType", "TopTaxonomy", "AvgDelayByTopTaxonomy"):
        frame[column] = np.nan
    return frame[SCHEDULE_COLUMNS]


# ----------------------------------------------------------------------------------
# Weather
# ----------------------------------------------------------------------------------
def weather_frame(net: Network, rng: np.random.Generator) -> pd.DataFrame:
    spec = DESIGN["weather"]
    start = datetime.fromisoformat(spec["start"])
    end = datetime.fromisoformat(spec["end"])
    hours = int((end - start).total_seconds() // 3600) + 1
    lat, lon = net.latlon[net.hub]
    grid_lat, grid_lon = round(lat * 4) / 4, round(lon * 4) / 4
    temp_noise, cloud_latent, wet = 0.0, 0.0, False
    rows = []
    for k in range(hours):
        stamp = start + timedelta(hours=k)
        day = (stamp - start).total_seconds() / 86400
        seasonal = spec["temp_mean_c"] + spec["temp_trend_c_per_day"] * day
        diurnal = spec["temp_diurnal_c"] * math.cos(2 * math.pi * (stamp.hour - 15) / 24)
        temp_noise = 0.9 * temp_noise + rng.normal(0, spec["temp_noise_c"])
        p_wet = spec["p_wet_wet"] if wet else spec["p_dry_wet"]
        wet = bool(rng.random() < p_wet)
        rain = round(float(rng.gamma(spec["rain_shape"], spec["rain_scale_mm"])), 1) if wet else 0.0
        cloud_latent = 0.9 * cloud_latent + rng.normal(0, spec["cloud_noise"])
        cloud = 100.0 / (1.0 + math.exp(-(cloud_latent + (3.0 if wet else 0.0))))
        temperature = round(seasonal + diurnal + temp_noise - (1.0 if wet else 0.0), 1)
        wind = round(float(abs(rng.normal(spec["wind_mean_kmh"], spec["wind_sd_kmh"]))), 1)
        apparent = round(temperature - 0.12 * wind, 1)
        humidity = int(min(100, max(35, round(70 + (20 if wet else 0) + rng.normal(0, 8)))))
        if rain >= 0.1:
            category = "Rain"
            code = 51 if rain < 0.3 else 53 if rain < 0.6 else 55 if rain < 1.0 else 61 if rain < 2.5 else 63
        elif cloud >= 85:
            category, code = "CloudyOrFog", 3
        elif cloud >= 45:
            category, code = "PartlyCloudy", 2
        else:
            category, code = "Clear", (1 if cloud >= 20 else 0)
        rows.append({
            "WeatherLocation": spec["location_label"], "RequestedLat": lat, "RequestedLong": lon,
            "WeatherGridLat": grid_lat, "WeatherGridLong": grid_lon,
            "WeatherGridElevationM": spec["grid_elevation_m"],
            "WeatherHour": stamp.strftime("%Y-%m-%d %H:%M:%S"), "TemperatureC": temperature,
            "ApparentTemperatureC": apparent, "RelativeHumidityPct": humidity,
            "PrecipitationMm": rain, "RainMm": rain, "CloudCoverPct": int(round(cloud)),
            "WeatherCode": code, "WindSpeed10mKmh": wind, "WeatherCategory": category,
            "IsRain": int(category == "Rain"),
        })
    return pd.DataFrame(rows, columns=WEATHER_COLUMNS)


# ----------------------------------------------------------------------------------
# Incidents
# ----------------------------------------------------------------------------------
def station_events(trips: list[dict]) -> dict[str, list[tuple]]:
    """station -> sorted (minute, trip number, line, direction, seq, event type)."""
    events = defaultdict(list)
    for trip in trips:
        for event in trip["events"]:
            minute = event["arrive"] if event["arrive"] is not None else event["depart"]
            kind = "Arr" if event["arrive"] is not None else "Dep"
            events[event["station"]].append(
                (minute, trip["trip"], trip["line"], trip["direction"], event["seq"], kind))
    return {s: sorted(v) for s, v in events.items()}


def build_incidents(net: Network, trips: list[dict], od_hourly: dict,
                    weather: pd.DataFrame, rng: np.random.Generator):
    spec = DESIGN["incidents"]
    windows = [int(w) for w in PROFILE.get("time.windows")]
    counts = {int(k): int(v) for k, v in PROFILE.get("probability.duration_counts").items()}
    cutoff = int(PROFILE.get("probability.duration_cutoff_min"))
    durations = sorted(counts)
    n_short, n_long = counts[durations[0]], counts[durations[-1]]
    n_regular = n_short + n_long
    first_day = date.fromisoformat(spec["first_date"])
    n_days = int(spec["days"])
    types = spec["types"]
    type_p = np.asarray([t["share"] for t in types], dtype=float)
    type_p /= type_p.sum()
    rain_by_hour = dict(zip(weather["WeatherHour"], weather["RainMm"]))
    daily_rain = (weather.assign(day=weather["WeatherHour"].str[:10])
                  .groupby("day")["RainMm"].sum())
    events = station_events(trips)

    # demand-weighted (hour, station) cells for the regular cohort
    throughput = defaultdict(float)
    for (i, j, hour), value in od_hourly.items():
        throughput[(hour, i)] += value
        throughput[(hour, j)] += value
    cells = [(h, s) for h in windows for s in net.stations]
    cell_w = np.asarray([throughput[c] ** spec["demand_weight_exponent"] for c in cells])
    cell_w /= cell_w.sum()

    def created_time(day_index, hour):
        minute = int(rng.integers(0, 60))
        second = int(rng.integers(0, 60))
        micro = int(rng.integers(0, 1000)) * 1000
        return datetime.combine(first_day + timedelta(days=day_index), datetime.min.time()) + \
            timedelta(hours=hour, minutes=minute, seconds=second, microseconds=micro)

    incidents = []
    # ---- regular incidents (the cohort): regular 8xxx trips, hours = planning windows
    for _ in range(n_regular):
        hour, station = cells[int(rng.choice(len(cells), p=cell_w))]
        created = created_time(int(rng.integers(0, n_days)), hour)
        kind = types[int(rng.choice(len(types), p=type_p))]
        rain = float(rain_by_hour[created.strftime("%Y-%m-%d %H:00:00")])
        corridor = net.corridor_label(station)
        latent = (spec["log_mu"] + kind["effect"] + spec["hour_effect"].get(str(hour), 0.0)
                  + (spec["rain_effect"] if rain > 0 else 0.0)
                  + spec["corridor_effect"].get(corridor, 0.0)
                  + rng.normal(0, spec["log_sigma"]))
        incidents.append({"group": "regular", "created": created, "hour": hour,
                          "station": station, "type": kind, "latent": latent})
    # exact class sizes by construction: the n_long largest latent durations are long
    order = np.argsort([-inc["latent"] for inc in incidents], kind="stable")
    long_ids, short_ids = order[:n_long], order[n_long:]
    lo_s, hi_s = spec["short_range_min"]
    lo_l, hi_l = spec["long_range_min"]
    if not (hi_s < cutoff <= lo_l and lo_s > spec["gt_threshold_min"]):
        raise ValueError("duration ranges inconsistent with the cutoff / gt10 threshold")

    def spread(ids, lo, hi):
        values = np.asarray([incidents[i]["latent"] for i in ids])
        lo_v, hi_v = values.min(), values.max()
        for i, v in zip(ids, values):
            share = 0.0 if hi_v == lo_v else (v - lo_v) / (hi_v - lo_v)
            incidents[i]["max_delay"] = float(round(math.exp(
                math.log(lo) + share * (math.log(hi) - math.log(lo)))))
    spread(short_ids, lo_s, hi_s)
    spread(long_ids, lo_l, hi_l)

    # ---- non-regular incidents (special services): in the >10 min file, never in the cohort
    for k in range(int(spec["n_nonregular"])):
        hour = int(rng.integers(spec["service_hours"][0], spec["service_hours"][1] + 1))
        station = net.stations[int(rng.integers(0, len(net.stations)))]
        blank_station = k < int(spec["n_nonregular_blank_station"])
        incidents.append({
            "group": "nonregular", "created": created_time(int(rng.integers(0, n_days)), hour),
            "hour": hour, "station": station, "blank_station": blank_station,
            "type": types[int(rng.choice(len(types), p=type_p))],
            "max_delay": float(int(rng.integers(spec["gt_threshold_min"] + 1,
                                                spec["nonregular_max_delay_min"] + 1))),
            "special_trip": f"{spec['special_trip_prefix']}{101 + k}"})
    # ---- minor incidents (<= 10 min): only in the raw registry
    for _ in range(int(spec["n_minor"])):
        hour = int(rng.integers(spec["service_hours"][0], spec["service_hours"][1] + 1))
        station = net.stations[int(rng.integers(0, len(net.stations)))]
        incidents.append({
            "group": "minor", "created": created_time(int(rng.integers(0, n_days)), hour),
            "hour": hour, "station": station,
            "type": types[int(rng.choice(len(types), p=type_p))],
            "max_delay": float(int(rng.integers(1, spec["gt_threshold_min"] + 1)))})

    incidents.sort(key=lambda inc: (inc["created"], inc["station"], inc["group"]))
    for k, inc in enumerate(incidents):
        inc["id"] = int(spec["first_incident_id"]) + k

    raw_rows, aug_rows = [], []
    for inc in incidents:
        day = inc["created"].date().isoformat()
        created_text = inc["created"].strftime("%Y-%m-%d %H:%M:%S.%f")
        minute_of_day = inc["created"].hour * 60 + inc["created"].minute
        taxonomy = f"{inc['type']['root']} ; {inc['type']['sub']}"
        description, org = token(rng), token(rng, "org")
        station = inc["station"]
        candidates = events[station]
        after = [e for e in candidates if e[0] >= minute_of_day - spec["primary_window_min"]]
        primary = after[0] if after else candidates[-1]
        n_knock = min(spec["max_knock_on"], 1 + int(inc["max_delay"] // spec["knock_on_every_min"]))
        later = [e for e in candidates if e[0] > primary[0] and e[1] != primary[1]]
        knock = []
        for e in later:
            if e[1] not in {k[1] for k in knock}:
                knock.append(e)
            if len(knock) >= n_knock:
                break
        two_primary = inc["group"] == "regular" and rng.random() < spec["share_two_primary"]
        trip_rows = []
        if inc["group"] == "nonregular":
            line = net.lines_at(station)[int(rng.integers(0, len(net.lines_at(station))))]
            trip_rows.append({"trip": inc["special_trip"], "line": line,
                              "direction": "U" if rng.random() < 0.5 else "D",
                              "seq": None if inc["blank_station"] else 1,
                              "kind": None if inc["blank_station"] else "Dep",
                              "location": None if inc["blank_station"] else station,
                              "primary": 1, "delay": None if inc["blank_station"] else inc["max_delay"]})
            # knock-on on regular trains; carries the maximum delay when the primary row has no location
            for k, e in enumerate(knock[:2]):
                delay = inc["max_delay"] if (inc["blank_station"] and k == 0) else \
                    float(max(1, round(inc["max_delay"] * spec["knock_on_decay"] ** (k + 1))))
                trip_rows.append({"trip": str(e[1]), "line": e[2], "direction": e[3],
                                  "seq": e[4], "kind": e[5], "location": station,
                                  "primary": 0, "delay": delay})
        else:
            trip_rows.append({"trip": str(primary[1]), "line": primary[2], "direction": primary[3],
                              "seq": primary[4], "kind": primary[5], "location": station,
                              "primary": 1, "delay": inc["max_delay"]})
            for k, e in enumerate(knock if inc["group"] == "regular" else knock[:1]):
                delay = float(max(1, round(inc["max_delay"] * spec["knock_on_decay"] ** (k + 1))))
                trip_rows.append({"trip": str(e[1]), "line": e[2], "direction": e[3],
                                  "seq": e[4], "kind": e[5], "location": station,
                                  "primary": 1 if (two_primary and k == 0) else 0,
                                  "delay": delay})
        vehicle = token(rng, "veh") if rng.random() < spec["share_vehicle_id"] else None
        for r in trip_rows:
            status = "A" if rng.random() > spec["share_status_other"] else \
                ["R", "C", "T", "P"][int(rng.integers(0, 4))]
            has_loc = r["location"] is not None
            key = (f"{r['trip']}_{day}_{inc['id']}_{r['location']}_{r['kind']}" if has_loc
                   else f"{r['trip']}_{day}_{inc['id']}_NA")
            raw_rows.append({
                "ItemId": inc["id"], "ItemCreated": created_text, "ItemDescription": description,
                "TripId": r["trip"], "TripDate": day, "PrimaryServiceForItem": r["primary"],
                "ServiceStatusCode": status,
                "ServiceShortStatusCode": "SD" if (status == "A" and rng.random() < 0.03) else None,
                "ItemTypeTaxonomy": taxonomy, "OrgText": org,
                "PathLocationSeqNo": None if r["seq"] is None else float(r["seq"]),
                "LocationCode": r["location"], "EventType": r["kind"],
                "DelayMinutes": r["delay"],
                "ActualDelayMinutes": (r["delay"] if (r["delay"] is not None
                                                      and rng.random() > 0.1) else None),
                "Attribution_Description": token(rng, "att") if rng.random() < 0.2 else None,
                "VehicleID": vehicle if r["primary"] == 1 else None,
                "VehicleName": None,
                "VehicleClassCode": spec["vehicle_class_code"] if (vehicle and r["primary"] == 1) else None,
                "VehicleClassDescription": spec["vehicle_class_description"] if (vehicle and r["primary"] == 1) else None,
                "VehicleCategoryCode": spec["vehicle_category_code"] if (vehicle and r["primary"] == 1) else None,
                "VehicleCategoryDescription": spec["vehicle_category_description"] if (vehicle and r["primary"] == 1) else None,
                "Corridor": net.label_of_corridor[net.corridor_of_line[r["line"]]],
                "Direction": r["direction"], "KeyValue": key, "Line": r["line"],
                "TimetableDate": day,
            })
        if inc["group"] == "minor":
            continue
        derived_max = max(r["delay"] for r in trip_rows if r["delay"] is not None)
        if derived_max != inc["max_delay"]:
            raise AssertionError(f"incident {inc['id']}: max delay not carried by its rows")
        blank = inc.get("blank_station", False)
        lines = sorted({r["line"] for r in trip_rows})
        weekday = inc["created"].date().weekday()
        aug_rows.append({
            "Year": inc["created"].year, "IncidentID": inc["id"], "EventDate": day,
            "EventDateSource": spec["event_date_source"], "Hour": inc["hour"],
            "Station": None if blank else station, "MaxDelayMinutes": inc["max_delay"],
            "StationCorridor": (net.label_of_corridor[net.corridor_of_line[lines[0]]]
                                if blank else net.corridor_label(station)),
            "CorridorSource": "raw_line" if blank else "observed_station",
            "CorridorIsImputed": int(blank),
            "CorridorEvidence": lines[0] if blank else station,
            "TripIDs": "|".join(sorted({r["trip"] for r in trip_rows})),
            "RawLines": "|".join(lines), "IsWeekend": int(weekday >= 5),
            "DailyIsRain": int(daily_rain.get(day, 0.0) >= spec["daily_rain_threshold_mm"]),
            "_group": inc["group"],
        })
    raw = pd.DataFrame(raw_rows, columns=RAW_INCIDENT_COLUMNS)
    aug = pd.DataFrame(aug_rows).sort_values(["EventDate", "Hour", "IncidentID"],
                                             kind="stable").reset_index(drop=True)
    regular = aug[aug["_group"] == "regular"]
    cohort = pd.DataFrame({
        "IncidentID": regular["IncidentID"], "Eligible": True, "Hour": regular["Hour"],
        "Station": regular["Station"],
        "RealizedState": np.where(regular["MaxDelayMinutes"] >= cutoff,
                                  durations[-1], durations[0]),
    }).sort_values("IncidentID").reset_index(drop=True)[COHORT_COLUMNS]
    gt10 = aug[aug["MaxDelayMinutes"] > spec["gt_threshold_min"]][AUGMENTED_COLUMNS]
    ge20 = aug[aug["MaxDelayMinutes"] >= spec["ge_threshold_min"]][AUGMENTED_COLUMNS]
    observed = cohort["RealizedState"].value_counts().to_dict()
    if observed != {durations[0]: n_short, durations[-1]: n_long}:
        raise AssertionError(f"cohort duration counts {observed} differ from the profile {counts}")
    return raw, gt10.reset_index(drop=True), ge20.reset_index(drop=True), cohort



# ----------------------------------------------------------------------------------
def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=int(DESIGN["seed"]))
    parser.add_argument("--out", type=Path, default=Path(PROFILE.get("profile.data_dir")))
    args = parser.parse_args(argv)
    out = args.out if args.out.is_absolute() else (ROOT / args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    files = {key: str(value) for key, value in PROFILE.get("inputs").items()}

    net = Network(rng)
    trips = build_trips(net)
    od, loads, od_hourly = build_od(net, trips, rng)
    frames: dict[str, object] = network_frames(net)
    frames["od_matrix"] = od
    frames["route_topology"] = route_topology(net, od_hourly)
    frames["timetable"] = timetable_frame(net, trips, loads, rng)
    weather = weather_frame(net, rng)
    frames["weather"] = weather
    raw, gt10, ge20, cohort = build_incidents(net, trips, od_hourly, weather, rng)
    frames.update({"incidents_raw": raw, "incidents_gt10": gt10,
                   "incidents_ge20": ge20, "incident_cohort": cohort})

    missing = sorted(set(files) - set(frames))
    if missing:
        raise KeyError(f"no generator for input key(s) {missing}")
    for key, name in files.items():
        path = out / name
        frame = frames[key]
        if key == "stations":
            with pd.ExcelWriter(path, engine="openpyxl") as writer:
                for sheet, table in frame.items():
                    table.to_excel(writer, sheet_name=sheet, index=False)
            normalize_xlsx(path)
        else:
            write_csv(frame, path, bom=key in BOM_KEYS)
    names = sorted(files.values())
    manifest = "".join(f"{sha256(out / name)}  {name}\n" for name in names)
    (out / "MANIFEST.sha256").write_text(manifest, encoding="utf-8", newline=NEWLINE)
    print(f"wrote {len(names)} files to {out}")
    print(manifest, end="")


if __name__ == "__main__":
    main()
