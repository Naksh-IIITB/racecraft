import sys
from hashlib import sha256
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from racecraft.circuits import CIRCUITS
from racecraft.data.snapshot import (
    policies_from_snapshot,
    profiles_from_snapshot,
    validate_snapshot,
)
from racecraft.ocr.f1_timing import extract_f1_timing_snapshot
from racecraft.simulation.engine import (
    BASE_DEGRADATION,
    RaceState,
    Strategy,
    candidate_strategies,
    evaluate_strategies,
)
from racecraft.simulation.twin import (
    EventHazard,
    PitPolicy,
    demo_grid,
    simulate_twin,
    simulate_twin_ensemble,
)
from racecraft.visualization.charts import strategy_chart

st.set_page_config(page_title="RaceCraft Strategy Lab", page_icon="🏁", layout="wide")
st.markdown(
    """
    <style>
      .stApp { background: radial-gradient(circle at 85% 3%, #25395b 0, #101622 34%, #090c13 72%); }
      [data-testid="stSidebar"] { background: linear-gradient(180deg, #171f2e 0%, #0e1420 100%); }
      [data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: .55rem; }
      .racecraft-hero { border: 1px solid rgba(90, 215, 191, .35); border-left: 6px solid #ff4655;
        border-radius: 18px; padding: 1.35rem 1.6rem; margin-bottom: 1rem;
        background: linear-gradient(105deg, rgba(255,70,85,.16), rgba(20,211,180,.11) 55%, rgba(24,35,55,.6)); }
      .racecraft-hero h1 { margin: 0; font-size: clamp(2rem, 4vw, 3.5rem); letter-spacing: -.045em; }
      .racecraft-kicker { color: #65e5d1; font-size: .76rem; font-weight: 800; letter-spacing: .16em; text-transform: uppercase; }
      .racecraft-subtitle { color: #cbd5e1; margin: .4rem 0 0; font-size: 1.06rem; }
      .racecraft-pill { display: inline-block; margin-top: .8rem; padding: .27rem .7rem; border-radius: 99px;
        color: #ffced3; background: rgba(255,70,85,.15); font-size: .82rem; font-weight: 650; }
      [data-testid="stMetric"] { background: rgba(20, 29, 44, .83); border: 1px solid rgba(113, 155, 184, .28);
        border-radius: 14px; padding: .7rem .85rem; }
      [data-testid="stMetricValue"] { color: #65e5d1; }
      [data-testid="stDataFrame"] { border: 1px solid rgba(113, 155, 184, .26); border-radius: 12px; overflow: hidden; }
      div[data-testid="stButton"] > button[kind="primary"] { background: linear-gradient(105deg, #ff4655, #ff714d); border: 0; font-weight: 750; }
      div[data-testid="stButton"] > button[kind="secondary"]:hover { border-color: #65e5d1; color: #65e5d1; }
      .racecraft-call { border-radius: 14px; border: 1px solid rgba(101,229,209,.38); background: rgba(44, 210, 181, .10);
        padding: .7rem .9rem; margin: .6rem 0 .9rem; color: #e5fffa; }
      .racecraft-call b { color: #65e5d1; }
    </style>
    <section class="racecraft-hero">
      <div class="racecraft-kicker">Live decision simulator · v0.3</div>
      <h1>🏁 RaceCraft Strategy Lab</h1>
      <p class="racecraft-subtitle">Turn race state into a clear pit-call comparison—fast enough for the pit wall, transparent enough for a project demo.</p>
      <span class="racecraft-pill">What-if analysis, not live team radio</span>
    </section>
    """,
    unsafe_allow_html=True,
)

with st.expander("New here? Read this first", expanded=True):
    st.markdown(
        """
**Start with the sidebar:** circuit, current lap, tyre age, and expected clean lap time.

RaceCraft asks: **would pitting now, later, or not at all be faster than staying out?**

- **Time vs staying out:** positive means the option is projected to save time.
- **Chance it is faster:** how often the option won in alternate race scenarios.
The default screen deliberately does **not** forecast finishing position. It only compares pit calls with staying out. A credible finish forecast needs the pace and gaps of the whole field; use **Real race snapshot mode** below when you have that information.

Results improve when you replace defaults with what you can see in the session: current tyre wear, recent clean pace, and gaps around the car.
"""
    )

with st.sidebar:
    st.header("Circuit and decision state")
    circuit_name = st.selectbox(
        "Circuit",
        list(CIRCUITS),
        index=7,
        help="Sets initial assumptions for lap count, pit-lane time, and tyre wear.",
    )
    circuit = CIRCUITS[circuit_name]
    st.caption(f"{circuit.overtaking} overtaking · editable simulation prior")
    total_laps = st.number_input(
        "Race distance",
        min_value=10,
        value=circuit.total_laps,
        help="Total laps in this race. The circuit default is prefilled.",
    )
    lap = st.number_input(
        "Current lap",
        min_value=1,
        max_value=int(total_laps - 2),
        value=min(35, int(total_laps - 2)),
        help="The lap that has just finished. RaceCraft projects the race from here.",
    )
    position = st.number_input(
        "Current position",
        min_value=1,
        max_value=20,
        value=8,
        help="Your car's current position; P1 means leading.",
    )
    tyre_age = st.number_input(
        "Tyre age",
        min_value=0,
        value=18,
        help="Laps already completed on the current tyre set.",
    )
    compound = st.selectbox(
        "Current compound",
        ["SOFT", "MEDIUM", "HARD"],
        index=1,
        help="The tyre currently on the car.",
    )
    clean_lap = st.number_input(
        "Expected clean lap (seconds)",
        min_value=60.0,
        value=circuit.clean_lap_seconds,
        step=0.1,
        help="A normal green-flag lap with no traffic. Use a recent clean lap—not a pit, Safety Car, or traffic-affected lap.",
    )
    sc_probability = st.slider(
        "Safety-car / VSC chance in remaining race",
        0.0,
        0.8,
        circuit.safety_car_probability,
        0.01,
        help="Your estimate of a Safety Car or Virtual Safety Car before the finish. A higher value makes waiting for a cheap pit stop more attractive.",
    )
    st.subheader("Tyre observation meter")
    use_observed_deg = st.checkbox("Use observed degradation", value=False)
    observed_default = max(
        0.01,
        min(
            0.20,
            round(BASE_DEGRADATION[compound] * circuit.degradation_prior / 0.005)
            * 0.005,
        ),
    )
    observed_deg = st.slider(
        "Observed loss per tyre lap (sec)",
        0.01,
        0.20,
        observed_default,
        0.005,
        help="Enter the recent green-flag lap-time loss you observed. It replaces the circuit prior.",
    )
    st.subheader("Projected field")
    field_gap = st.slider(
        "Typical gap per position (sec)",
        0.5,
        5.0,
        1.8,
        0.1,
        help="Use the live gaps around your car. Smaller gaps make a strategy time loss cost more positions.",
    )
    field_uncertainty = st.slider(
        "Field uncertainty (sec)",
        0.0,
        10.0,
        2.5,
        0.5,
        help="How much unknown rival pace and traffic could change the result.",
    )
    analysis_depth = st.select_slider(
        "Analysis depth",
        options=[
            "Quick (250 scenarios)",
            "Balanced (1,000 scenarios)",
            "Deep (5,000 scenarios)",
        ],
        value="Balanced (1,000 scenarios)",
        help="RaceCraft tries alternate Safety Car and traffic outcomes. More scenarios make percentages steadier but take longer.",
    )
    runs = {
        "Quick (250 scenarios)": 250,
        "Balanced (1,000 scenarios)": 1000,
        "Deep (5,000 scenarios)": 5000,
    }[analysis_depth]

degradation_multiplier = (
    observed_deg / BASE_DEGRADATION[compound]
    if use_observed_deg
    else circuit.degradation_prior
)
total_laps = int(total_laps)
# Hot reload can retain a previous UI state whose lap was legal under the old
# upper bound. Clamp it before any strategy is built so an end-of-race click
# never creates an invalid pit lap.
current_lap = min(max(1, int(lap)), total_laps - 2)
if current_lap != int(lap):
    st.warning(
        f"Decision lap reset to {current_lap}: a pit call must leave at least one racing lap."
    )
st.markdown(
    f"<div class='racecraft-call'><b>Race state · {circuit_name}</b> &nbsp; | &nbsp; "
    f"Pit-loss prior <b>{circuit.pit_lane_loss:.1f}s</b> &nbsp; | &nbsp; "
    f"Tyre loss <b>{degradation_multiplier:.2f}×</b> &nbsp; | &nbsp; "
    f"Traffic gap <b>{field_gap:.1f}s/position</b></div>",
    unsafe_allow_html=True,
)

state = RaceState(
    current_lap, total_laps, int(tyre_age), compound, int(position), float(clean_lap)
)
pit_next = current_lap + 1
extend_lap = min(total_laps - 1, current_lap + 5)
quick_calls = {
    "Hold track position — pit next lap for Hard": Strategy(
        pit_next, "HARD", f"Pit lap {pit_next} → Hard"
    ),
    "Attack undercut — pit next lap for Soft": Strategy(
        pit_next, "SOFT", f"Pit lap {pit_next} → Soft"
    ),
    f"Extend — pit lap {extend_lap} for Medium": Strategy(
        extend_lap, "MEDIUM", f"Pit lap {extend_lap} → Medium"
    ),
    "Commit — no further stop": Strategy(None, None, "Stay out"),
}
st.subheader("Quick strategy calls")
selected_call = st.radio(
    "Apply a strategy instantly",
    list(quick_calls),
    horizontal=False,
    label_visibility="collapsed",
)
selected_strategy = quick_calls[selected_call]
strategies = [
    selected_strategy,
    *candidate_strategies(state, horizon=min(10, total_laps - current_lap)),
]
# Preserve order while removing duplicate calls.
strategies = list({(x.pit_lap, x.next_compound): x for x in strategies}.values())
results, failed_strategies = evaluate_strategies(
    state,
    strategies,
    n_simulations=runs,
    safety_car_probability=sc_probability,
    pit_lane_loss=circuit.pit_lane_loss,
    degradation_multiplier=degradation_multiplier,
    field_gap_seconds=field_gap,
    field_uncertainty_seconds=field_uncertainty,
    seed=2026,
)
if failed_strategies:
    st.warning("Skipped invalid strategy inputs: " + " | ".join(failed_strategies))
if not results:
    st.error(
        "No valid strategy could be evaluated. Reset the decision inputs and try again."
    )
    st.stop()
results.sort(key=lambda x: x.delta_to_baseline_seconds, reverse=True)

winner = results[0]
selected_result = next(
    x
    for x in results
    if (x.strategy.pit_lap, x.strategy.next_compound)
    == (selected_strategy.pit_lap, selected_strategy.next_compound)
)
left, right = st.columns((2, 1))
with left:
    st.plotly_chart(strategy_chart(results[:6]), use_container_width=True)
with right:
    st.subheader("Selected call")
    selected_colour = {
        "SOFT": "🔴",
        "MEDIUM": "🟡",
        "HARD": "⚪",
    }.get(selected_result.strategy.next_compound, "🟢")
    st.markdown(
        f"<div class='racecraft-call'>{selected_colour} <b>{selected_result.strategy.label}</b></div>",
        unsafe_allow_html=True,
    )
    metric_a, metric_b = st.columns(2)
    metric_a.metric(
        "Time vs stay out", f"{selected_result.delta_to_baseline_seconds:+.1f}s"
    )
    metric_b.metric(
        "Chance it beats stay out", f"{selected_result.p_beat_baseline:.0%}"
    )
    st.divider()
    st.caption("Highest-value call under these assumptions")
    st.write(winner.strategy.label)
    st.metric("Value", f"{winner.delta_to_baseline_seconds:+.1f}s")
    if selected_result.delta_to_baseline_seconds > 0:
        st.success("Green light: this call is faster than staying out in the model.")
    else:
        st.warning("Hold: this call loses time versus staying out under these assumptions.")

st.subheader("Candidate details")
st.caption(
    "The six strongest choices in plain language. Positive time means faster than staying out."
)
table = pd.DataFrame(
    [
        {
            "Call": r.strategy.label,
            "Saves / loses time": f"{r.delta_to_baseline_seconds:+.1f}s",
            "Chance it is faster": f"{r.p_beat_baseline:.0%}",
        }
        for r in results[:6]
    ]
)
st.dataframe(table, hide_index=True, use_container_width=True)
st.info(
    "Scenario mode compares time, not finishing positions. It shares a baseline and Safety Car draws across every option, but cannot know what the other cars will do."
)

st.divider()
st.subheader("Real race snapshot mode")
st.caption(
    "A finish forecast needs every car's current position, gap, pace and tyres. Import a timing screenshot or a CSV, then review the editable table before forecasting."
)
snapshot_mode = st.radio(
    "Snapshot source",
    ["Timing screenshot", "CSV file"],
    horizontal=True,
    help="A screenshot is read locally using Apple Vision on macOS. CSV is best when timing data is already available.",
)
snapshot: pd.DataFrame | None = None
if snapshot_mode == "Timing screenshot":
    timing_image = st.file_uploader(
        "Upload an F1 timing screenshot",
        type=["png", "jpg", "jpeg"],
        help="Use a sharp leaderboard or tyre screen with all rows visible. The image stays on this computer.",
    )
    if timing_image is not None:
        image_bytes = timing_image.getvalue()
        image_fingerprint = sha256(image_bytes).hexdigest()
        st.image(timing_image, caption="Screenshot to read", width=600)
        if st.button("Read timing screenshot"):
            try:
                with st.spinner("Reading positions, gaps and tyre details locally…"):
                    st.session_state["timing_snapshot"] = extract_f1_timing_snapshot(
                        image_bytes, float(clean_lap)
                    )
                    st.session_state["timing_snapshot_source"] = image_fingerprint
            except (RuntimeError, ValueError) as error:
                st.error(f"Could not read this screenshot: {error}")
        if st.session_state.get("timing_snapshot_source") == image_fingerprint:
            st.caption(
                "Review the extracted table. Correct any missing gap or `UNKNOWN` tyre compound before forecasting."
            )
            snapshot = st.data_editor(
                st.session_state["timing_snapshot"],
                hide_index=True,
                use_container_width=True,
                key="timing_snapshot_editor",
                column_config={
                    "compound": st.column_config.SelectboxColumn(
                        "Compound", options=["SOFT", "MEDIUM", "HARD", "UNKNOWN"]
                    ),
                    "clean_lap_seconds": st.column_config.NumberColumn(
                        "Clean lap (sec)", min_value=1.0
                    ),
                    "gap_to_leader_seconds": st.column_config.NumberColumn(
                        "Gap to leader (sec)", min_value=0.0
                    ),
                },
            )
else:
    snapshot_file = st.file_uploader(
        "Upload race snapshot CSV",
        type="csv",
        help="Required: driver_id, position, gap_to_leader_seconds, clean_lap_seconds, tyre_age, compound. Optional: planned_pit_lap, next_compound.",
    )
    if snapshot_file is not None:
        snapshot = pd.read_csv(snapshot_file)

if snapshot is not None:
    try:
        validate_snapshot(snapshot)
        rival_policies = policies_from_snapshot(
            snapshot, current_lap=current_lap, total_laps=total_laps
        )
        target_driver = st.selectbox(
            "Target car",
            snapshot.driver_id.astype(str).tolist(),
            help="The car whose selected strategy you want to test.",
        )
        with st.expander("How rival strategies are handled"):
            policy_table = pd.DataFrame(
                [
                    {
                        "Car": driver,
                        "Modelled rival call": (
                            f"Pit lap {policy.planned_pit_lap} → {policy.next_compound.title()}"
                            if policy.planned_pit_lap
                            else "No planned stop"
                        ),
                    }
                    for driver, policy in rival_policies.items()
                    if driver != target_driver
                ]
            )
            st.caption(
                "The model runs every car in every scenario. Blank rival plans are inferred from current tyre age and compound; Safety Car windows can pull a scheduled stop forward. Edit `planned_pit_lap` in the snapshot table to override a rival."
            )
            st.dataframe(policy_table, hide_index=True, use_container_width=True)
        field_runs = {
            "Quick (250 scenarios)": 80,
            "Balanced (1,000 scenarios)": 200,
            "Deep (5,000 scenarios)": 500,
        }[analysis_depth]
        full_grid = len(snapshot) == 20
        if not full_grid:
            st.warning(
                f"This snapshot has {len(snapshot)} cars. A real F1 expected position needs all 20 cars, so the finish forecast is disabled until the grid is complete."
            )
        if st.button(
            f"Run {field_runs} full-field finish scenarios",
            type="primary",
            disabled=not full_grid,
        ):
            profiles = profiles_from_snapshot(snapshot)
            remaining_laps = total_laps - current_lap
            per_lap_hazard = 1 - (1 - sc_probability) ** (1 / max(remaining_laps, 1))
            target_policy = PitPolicy(
                selected_strategy.pit_lap, selected_strategy.next_compound or "HARD"
            )
            snapshot_twin = simulate_twin_ensemble(
                profiles,
                rival_policies,
                total_laps=total_laps,
                start_lap=current_lap,
                target_id=target_driver,
                target_strategy=target_policy,
                hazard=EventHazard(per_lap_hazard),
                n_simulations=field_runs,
                seed=2026,
            )
            target_row = snapshot_twin.classification.query(
                "driver_id == @target_driver"
            ).iloc[0]
            one, two, three = st.columns(3)
            one.metric(
                "Expected target finish", f"P{target_row.expected_position:.1f}"
            )
            two.metric(
                "Likely range (80%)",
                f"P{int(target_row.best_case)}–P{int(target_row.worst_case)}",
            )
            three.metric("Points chance", f"{target_row.points_chance:.0%}")
            st.caption(
                f"Projected classification from {snapshot_twin.scenario_count} full-field scenarios. This is a scenario distribution, not a promise of a race result."
            )
            display_classification = snapshot_twin.classification.rename(
                columns={
                    "driver_id": "Car",
                    "expected_position": "Expected finish",
                    "median_position": "Most likely finish",
                    "best_case": "Best 10%",
                    "worst_case": "Worst 10%",
                    "points_chance": "Points chance",
                    "win_chance": "Win chance",
                }
            )
            st.dataframe(
                display_classification,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Points chance": st.column_config.ProgressColumn(
                        format="%d%%", min_value=0, max_value=1
                    ),
                    "Win chance": st.column_config.ProgressColumn(
                        format="%d%%", min_value=0, max_value=1
                    ),
                },
            )
    except (ValueError, KeyError, pd.errors.ParserError) as error:
        st.error(f"Snapshot could not be used: {error}")
else:
    st.code(
        "driver_id,position,gap_to_leader_seconds,clean_lap_seconds,tyre_age,compound\nTARGET,8,18.2,77.8,18,MEDIUM",
        language="csv",
    )
    st.caption(
        "Need a CSV instead? Example file: data/sample/race_snapshot_example.csv"
    )

with st.expander("Developer demo: synthetic all-car control room"):
    st.caption(
        "This uses invented cars. It demonstrates the simulator only and is not a prediction."
    )
    target_pit = st.slider(
        "Target car planned pit lap",
        current_lap + 1,
        total_laps - 1,
        min(current_lap + 5, total_laps - 1),
        key="twin_pit",
    )
    run_demo = st.button("Run synthetic demo")
    if run_demo:
        profiles, policies = demo_grid()
        twin = simulate_twin(
            profiles,
            policies,
            total_laps=total_laps,
            start_lap=current_lap,
            target_id="CAR-08",
            target_strategy=PitPolicy(target_pit, "HARD"),
            seed=2026,
        )
        col1, col2 = st.columns(2)
        with col1:
            st.write("Final simulated classification")
            st.dataframe(twin.classification, hide_index=True, use_container_width=True)
        with col2:
            st.write("Sampled race events")
            event_rows = [
                {"lap": x.lap, "state": x.state, "duration": x.duration_laps}
                for x in twin.events
            ]
            st.dataframe(
                pd.DataFrame(event_rows)
                if event_rows
                else pd.DataFrame(
                    [{"lap": "—", "state": "None sampled", "duration": "—"}]
                ),
                hide_index=True,
                use_container_width=True,
            )
        trajectory = twin.laps[twin.laps.driver_id == "CAR-08"]
        st.line_chart(trajectory.set_index("lap")[["position"]], y_label="Position")
