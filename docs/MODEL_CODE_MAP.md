# Model ↔ code map

Correspondence between the formulation of the paper (Sec. 3; Table 3 = `tab:settings`; printed equation numbers) and the code.
Code column: file (abbreviation below), then function, then the name used in the code (variable, row name, constant or config key); several entries jointly implement one element.
Files (below `Paper_Execution_Pipeline/`): **CG** `_core/stage1/CG_Stage1_Stage2/cg_stage1.py` · **CR** `_core/stage1/stage1_exact_cg.py` · **C1** `_core/stage1/stage1_network_model.py` · **SA** `_core/stage2/stage2_adaptive.py` · **MC** `_core/stage2/milp_core.py` · **NG** `_core/normal_generation.py` · **DG** `_core/disruption_generation.py` · **QC/SC/WE** `CG_Stage1_Stage2/{queue_compression,state_compression,walking_elimination}.py` · **ER** `Sensitivity/source/evaluate_ri.py` · **04/06/06D** the runners `04_stage1_positioning.py`, `06_stage2_core.py`, `06_stage2_dispatch.py`.

## 1. Sets, parameters and variables

| Paper element | Code location |
|---|---|
| N, platforms j (rail platform and paired virtual bus boarding point) | C1/MC `build_window_data`: station codes `N`, `station_of`; a bus stop carries the station code of its platform |
| N^ω, OD-specific platform subset | C1/MC `build_stage2_cache`: `N_omega` |
| A_w, σ_ij, walking transfer time (Table 3: 2 min) | `operations.walking_transfer_min` → CG `SIGMA`; `walking_times.load_uniform_walking_minutes`; WE `WalkingEliminationPlan` (σ-minute walking interval after an alighting at j ≠ e(ω)) |
| A_ρ, μ_ij, road links and times | `inputs.road_times` → C1/MC `load_distance_matrix`, `get_travel_time` |
| 𝓗, h(ξ), planning windows | `time.windows`; one model per window |
| Γ, T, t₀ (Table 3: 240, 10) | `time.horizon_min`, `time.disruption_onset_min` → C1/MC `build_window_data` `Gamma`, `WINDOW_END_MIN`, `DISRUPT_OFFSET`; CG `OBJECTIVE_ONSET_MIN`, `OBJECTIVE_NORMALISER` (T − t₀) |
| η, δ (Table 3: 10, 30) | `time.dispatch_interval_min`, `time.clearance_buffer_min` → C1/MC `DISPATCH_STEP`, `DISRUPT_BUFFER`, `DISPATCH_END_MIN` |
| 𝓟, τ(ξ), duration states | `time.durations_min` → `core.DURATIONS` |
| Ξᴺ, Ξᴰ, Ξᴿ, scenario sets | NG (one normal case per window); DG scenario loop (platform × window × duration); CR `prepare_window`, `_load_duration_representatives`, `_expand_terminal_station_union` |
| Ω, Ω_m(ξ), Ω_n(ξ) | `build_window_data` `Omega`, `Omega_Map`; DG `Omega_m`; scenario `Omega_m` |
| V_r, V_ρ, V | `build_window_data` `Vm`, `train_active_times`; scenario `bus_trips` (MC/C1 `generate_bus_trips`) |
| C_v (Table 3: 500 rail, 50 bus) | `operations.rail_capacity`, `operations.bus_capacity` → CG `CGConfig.train_capacity`, `CGConfig.bus_capacity`; C1/MC `TRAIN_CAPACITY`, `BUS_CAPACITY` |
| 𝒢, candidate depots | CR `load_candidate_depots`; `inputs` depot list |
| B (Table 3: 30) | `operations.fleet_size` → CG `CGConfig.max_total_buses`, `MAX_TOTAL_BUSES` |
| ε (Table 3: 1) | `operations.min_trip_boarding` → CG `CGConfig.min_trip_boarding`, MC `MIN_TRIP_BOARDING` |
| κ (Table 3: 1.0) | `operations.out_of_vehicle_weight` → CG `OUT_OF_VEHICLE_WEIGHT`; NG/DG `KAPPA`; SA `reweight_flow_block` (κ cases of Sec. 5.7) |
| μ_max (Table 3: 120) | `operations.depot_access_bound_min` → C1/MC `DEPOT_ACCESS_BOUND_MIN`, CR `EXACT_DEPOT_ACCESS_BOUND_MIN`, env `TSB_DEPOT_ACCESS_BOUND` |
| n_s, express one-way bound (Table 3: 5, 120 min) | `routes.express_max_stops`, `routes.express_max_travel_min` → `03_bus_routes.py`, `_core/build_express_variants.py`; library `routes.library_file` |
| D_{ω,j,t}, Σ_{t′≤t} D_{ω,s(ω),t′} | `build_window_data` `D`, `CumD` |
| θ_{v,t}(ξ) | DG `compute_cancellation_start`, `theta`; C1/MC `load_disruption_scenarios` scenario `theta` |
| S_{v,t}(ξ) | DG `cap_limit`; scenario `spare_capacity` |
| I_{v,t,j} | `build_window_data` `I_vts`, `trains_at`; bus trip `StopTimeList` |
| g_v, s(v), e(v), t_v^start, t_v^end | bus trip record `Depot`, `RouteOrigin`, `ServiceEndStation`, `RunStart`, `RunEnd`; mobilization time t_v^start − μ_{g_v,s(v)} = `DispatchStart` |
| α_{v,τ} | C1 `build_block_trips` (`DispatchStart` ≤ t₀+τ+δ); SA `trip_availability`; CG/SA `info_cells` |
| Ψ | CG `_add_nonanticipative_vehicle_chain_constraints` (successor lists); SA `_compatible_trip_arcs` |
| 𝒫_v(ξ), 𝒫_{v,v′}(ξ) | CG/SA `info_cells`, `_chain_arc_decision_time` |
| π_τ (Table 3: 0.686, 0.314) | `probability.duration_counts` → 04 `reweight_duration_prior`; 08 `HISTORICAL_WEIGHTS` |
| π̂_τ | `05_duration_probabilities.py`; 06D `run_single` (`p60`) → SA `solve_stage2_adaptive` `duration_weight` |
| β, n_j, π_λ, π_j | `probability.station_smoothing_beta` → `02_reduction_method_comparison.py` `historical_corridor_probability`, `apply_corridor_probability`; `02_scenario_reduction.py` `recompute` |
| p_ξ | 04 `reweight_duration_prior` (cluster mass × π_τ) → scenario `probability` |
| x_g | CG `solve_master` `x[g]`; Stage 2: 06 `load_allocation` → `depot_counts` |
| y_v(ξ) | CG `_build_shared_y` `y[block,trip|cell]`; SA `solve_stage2_adaptive` `yvar` `y[tid|cell]` |
| u_v(ξ) | CG `first`; SA `first_vars` |
| ψ_{v,v′}(ξ) | CG `link`; SA `link_vars` |
| q, l, b, a, d, o | CG `solve_master` `q_{sid}` (QC segments), `l_{sid}` (SC segments) and hybrid bus states `L_{sid}_{trip}`, `b_{sid}`, `a_{sid}`, walking d from `a` via WE, `od_{sid}`; SA `_add_flow_block_exact` (same families); NG/DG `qd`, `ld`, `bd`, `ad`, `dd`, `od` |
| z_t^ω | CG `eq13_inventories` (queue, onboard, walking terms) |

## 2. Stage 1 equations (with the CFM, scenario selection and network construction)

| Paper equation | Code location |
|---|---|
| (1) Ξᴰ, disruption scenario set | DG scenario loop over target platforms, windows and `time.durations_min`; C1 `load_disruption_scenarios` |
| (2) [eq:inventory] generalized inventory z_t | CG `eq13_inventories`, `eq13_objective` (κ); NG and DG objective expressions (`KAPPA`) |
| (3) [eq:obj_n] normal objective Z^N | NG `m.setObjective`: (κ[qd + dd] + ld) / T |
| (4) [eq:q_dest_n] destination queue balance | NG/DG `qd` destination rows; CG destination absorption (`CGConfig.eliminate_destination_states`) |
| (5) [eq:q_nd_n] platform queue balance | CG `q_flow[...]` rows on QC segments; NG/DG `qd` non-destination rows |
| (6) [eq:l_n] onboard load balance | CG `_add_compressed_load_states`, `_add_hybrid_bus_load_states`; NG/DG `ld` rows |
| (7) [eq:cap_n] rail capacity | CG `_add_interval_load_capacities`, `tr_cap[...]`; NG/DG capacity rows |
| (8) [eq:dem_ab_n] a, b ≤ cumulative demand | CG `b_ub`, `a_ub` (`CumD`); NG/DG `ad`, `bd` ≤ `CumD` |
| (9) [eq:dem_o_n] exit ≤ cumulative demand | CG `od_ub`; NG/DG `od` bounds |
| (10) [eq:nn_n] nonnegativity | lb = 0 on every flow variable (CG `_batch_add_passenger_vars`, NG/DG `addVars(lb=0)`) |
| (11) [eq:omega_m] affected OD set | DG `Omega_m` (normal loads on services with θ = 1); scenario `Omega_m` |
| (12) [eq:spare] residual rail capacity | DG `cap_limit`; scenario `spare_capacity`; CG `load_cap_*` rows and boarding fixings on services with θ = 1 |
| (13) [eq:obj_db] Z_m^c over t ≥ t₀, 1/(T − t₀) | CG `eq13_inventories`, `eq13_objective`; per-scenario `objective` in `solve_master` results |
| (14) [eq:obj_d] disrupted objective | DG `m_dis.setObjective` (Eq. (13) with c = D) |
| Sec. 3.2.2, states before onset fixed to normal values | CR `_attach_pre_onset_normal_actions` (`normal_pre_onset`, `required_station_support`); CG `fix_pre_onset_to_normal`, `pre_onset_normal_station_support`; C1 `load_normal_trajectory`; DG `normal_pre_b`, `normal_pre_a`, `normal_pre_od` |
| (15) [eq:location_prob] smoothed location probability | `02_reduction_method_comparison.py` `historical_corridor_probability`, `apply_corridor_probability`; `02_scenario_reduction.py` `recompute` |
| (16) [eq:feature_component] clustering feature | `02_reduction_method_comparison.py` `read_impact`, `load_impact_features` |
| (17) [eq:k_allocation_obj] K allocation objective | `02_reduction_method_comparison.py` `allocate_k`; `02_scenario_reduction.py` `allocate` |
| (18) [eq:k_allocation_total] total K | `allocate_k`, `allocate` (`clustering.k_per_window`) |
| (19) [eq:k_allocation_bound] K bounds | `allocate_k`, `allocate`, `fit_options` |
| (20) [eq:arrival_indicator] I_{v,t,j} | C1 `build_window_data` `I_vts`; C1 `generate_bus_trips` `StopTimeList` |
| (21) [eq:deadhead_bound] μ ≤ μ_max | C1 `select_routes` (`DEPOT_ACCESS_BOUND_MIN`) |
| (22) [eq:start_time_grid] start grid and mobilization window | C1 `select_routes` (`earliest`, `latest`), `generate_bus_trips` (t_v^end ≤ T) |
| (23) [eq:availability] α_{v,τ} | C1 `build_block_trips`; CG `info_cells` |
| (24) [eq:chain_set] Ψ | CG `_add_nonanticipative_vehicle_chain_constraints` (`successors`, `link`) |
| (25) [eq:shared_trip_branches] 𝒫_v(ξ) | CG `info_cells`, `_build_shared_y` |
| (26) [eq:shared_chain_branches] 𝒫_{v,v′}(ξ) | CG `_chain_arc_decision_time` |
| (27) [eq:obj_milp] min Σ p_ξ Z_m^B | CG `solve_master` `setObjective` (Σ `probability` × `eq13_objective`) |
| (28) [eq:fleet] Σ x_g = B | CG `fleet_budget` |
| (29) [eq:fleet_domain] x_g ∈ ℤ≥0 | CG `x[g]` (integer, ub B) |
| (30) [eq:avail_cons] y ≤ α | CG `_build_shared_y` (y exists only in cells with α = 1) |
| (31) [eq:na_dispatch] shared dispatch | CG `_build_shared_y` (one `y` per information cell) |
| (32) [eq:na_duty] shared duty start | CG `first` (one per information cell) |
| (33) [eq:na_chain] shared chaining | CG `link` (one per chain information cell) |
| (34) [eq:chain] Σψ_in + u = y | CG `chain_pred[...]` (u = y substituted when Ψ has no predecessor) |
| (35) [eq:chain_successor] Σψ_out ≤ y | CG `chain_succ[...]` |
| (36) [eq:depot_vehicles] Σ_{g_v = g} u_v ≤ x_g | CG `chain_fleet[...]` |
| (37) [eq:cap_bus] bus capacity | CG `_add_hybrid_bus_load_states` (capacity × y), `bus_cap[...]` |
| (38) [eq:min_svc] Σ b ≥ ε y | CG `min_svc[...]`, `min_svc_cell[...]` |
| (39) [eq:domain_yu] y, u binary | CG `y_type` binary, `first` binary |
| (40) [eq:domain_psi] ψ binary | CG `link` binary |

## 3. Stage 2 equations (and resilience assessment)

| Paper equation | Code location |
|---|---|
| (2) [eq:inventory] and (13) [eq:obj_db] Z_m^B((j, τ)) | SA `_add_flow_block_exact` → CG `eq13_inventories`, `eq13_objective`; SA `reweight_flow_block` |
| (4) [eq:q_dest_n] | SA `_add_flow_block_exact` (destination absorption) |
| (5) [eq:q_nd_n] | SA `_add_flow_block_exact` queue-segment rows (QC) |
| (6) [eq:l_n] | CG `_add_compressed_load_states` (rail), `_add_hybrid_bus_load_states` (bus), called by SA `_add_flow_block_exact` |
| (7) [eq:cap_n] | CG `_add_interval_load_capacities` via SA `_add_flow_block_exact` |
| (8) [eq:dem_ab_n] | SA `_add_flow_block_exact` `b_upper`, `a_upper` (`CumD`) |
| (9) [eq:dem_o_n] | SA `_add_flow_block_exact` exit bounds |
| (10) [eq:nn_n] | SA `_add_flow_block_exact` (lb = 0) |
| (11) [eq:omega_m] | SA `build_adaptive_group` `scen[D]["Omega_m"]` |
| (12) [eq:spare] | scenario `spare_capacity`, `theta`; SA `_add_flow_block_exact` (b = 0 on services with θ = 1) |
| Sec. 3.2.2, states before onset fixed to normal values | SA `build_adaptive_group` (`required_station_support`), `solve_stage2_adaptive` → CG `fix_pre_onset_to_normal`; MC `load_normal_trajectory` |
| (20) [eq:arrival_indicator] | MC `build_window_data` `I_vts`; MC `generate_bus_trips` `StopTimeList` |
| (21) [eq:deadhead_bound] | MC `select_routes` (`DEPOT_ACCESS_BOUND_MIN`) |
| (22) [eq:start_time_grid] | MC `select_routes`, `generate_bus_trips` |
| (23) [eq:availability] | SA `trip_availability`, `info_cells` |
| (24) [eq:chain_set] | SA `_compatible_trip_arcs` |
| (25) [eq:shared_trip_branches], (26) [eq:shared_chain_branches] | SA `info_cells`, `_chain_arc_decision_time` |
| (30) [eq:avail_cons], (31) [eq:na_dispatch] | SA `solve_stage2_adaptive` `yvar` (one per available information cell) |
| (32) [eq:na_duty], (33) [eq:na_chain] | SA `_add_nonanticipative_vehicle_chains` `first_vars`, `link_vars` |
| (34) [eq:chain] | SA `eq34_chain[...]` |
| (35) [eq:chain_successor] | SA `eq35_chain_successor[...]` |
| (36) [eq:depot_vehicles] with x_g = x*_g | SA `eq36_depot_vehicles[...]` (`depot_counts` from 06 `load_allocation`) |
| (37) [eq:cap_bus] | CG `_add_hybrid_bus_load_states` via SA `_add_flow_block_exact` |
| (38) [eq:min_svc] | SA `_add_flow_block_exact` (Σ b ≥ ε y) |
| (39) [eq:domain_yu], (40) [eq:domain_psi] | SA `yvar`, `first_vars`, `link_vars` (binary) |
| (41) [eq:s2_eval] min Σ π̂_τ Z_m^B | SA `solve_stage2_adaptive` `setObjective` (`P[D] * Zs[D]`); π̂ from 06D `run_single` |
| (42) [eq:ap] AP_t^c | ER `performance_ratio` |
| (43) [eq:ri] RI^c | ER `resilience_index` |
| (44) [eq:passenger_time] PT^c | ER `passenger_time_metrics` |
