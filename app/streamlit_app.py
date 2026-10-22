import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from racecraft.circuits import CIRCUITS
from racecraft.simulation.engine import (
    BASE_DEGRADATION,
    RaceState,
    Strategy,
    candidate_strategies,
    evaluate_strategies,
)
from racecraft.simulation.twin import PitPolicy, demo_grid, simulate_twin
from racecraft.visualization.charts import strategy_chart

st.set_page_config(page_title="RaceCraft Strategy Lab", page_icon="🏁", layout="wide")
st.title("🏁 RaceCraft Strategy Lab")
st.caption(
    "A decision simulator: compare every pit call against the same projected no-stop race baseline."
)

with st.sidebar:
    st.header("Circuit and decision state")
    circuit_name = st.selectbox("2026 circuit", list(CIRCUITS), index=7)
    circuit = CIRCUITS[circuit_name]
    st.caption(f"{circuit.overtaking} overtaking · editable simulation prior")
    total_laps = st.number_input(
        "Race distance", min_value=10, value=circuit.total_laps
    )
    lap = st.number_input(
        "Current lap",
        min_value=1,
        max_value=int(total_laps - 2),
        value=min(35, int(total_laps - 2)),
    )
    position = st.number_input("Current position", min_value=1, max_value=20, value=8)
    tyre_age = st.number_input("Tyre age", min_value=0, value=18)
    compound = st.selectbox("Current compound", ["SOFT", "MEDIUM", "HARD"], index=1)
    clean_lap = st.number_input(
        "Expected clean lap (seconds)",
        min_value=60.0,
        value=circuit.clean_lap_seconds,
        step=0.1,
    )
    sc_probability = st.slider(
        "Safety-car / VSC chance in remaining race",
        0.0,
        0.8,
        circuit.safety_car_probability,
        0.01,
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
    runs = st.select_slider(
        "Monte Carlo runs", [250, 500, 1000, 2000, 5000], value=1000
    )

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
st.caption(
    f"**{circuit_name}** · pit-loss prior: {circuit.pit_lane_loss:.1f}s · degradation: **{degradation_multiplier:.2f}×** · field gap: **{field_gap:.1f}s/position**"
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
    st.plotly_chart(strategy_chart(results[:12]), use_container_width=True)
with right:
    st.subheader("Selected call")
    st.write(selected_result.strategy.label)
    metric_a, metric_b = st.columns(2)
    metric_a.metric(
        "Time vs stay out", f"{selected_result.delta_to_baseline_seconds:+.1f}s"
    )
    metric_b.metric(
        "Chance it beats stay out", f"{selected_result.p_beat_baseline:.0%}"
    )
    st.metric("Projected finish", f"P{selected_result.expected_finish:.1f}")
    st.caption(
        f"80% range: P{selected_result.p10_finish:.0f}–P{selected_result.p90_finish:.0f}"
    )
    st.divider()
    st.caption("Highest-value call under these assumptions")
    st.write(winner.strategy.label)
    st.metric("Value", f"{winner.delta_to_baseline_seconds:+.1f}s")

st.subheader("Candidate details")
table = pd.DataFrame(
    [
        {
            "Strategy": r.strategy.label,
            "Expected finish": round(r.expected_finish, 2),
            "Time vs stay out (s)": round(r.delta_to_baseline_seconds, 2),
            "Beat stay-out chance": round(r.p_beat_baseline, 3),
            "Points probability": round(r.p_points, 3),
            "P10 finish": r.p10_finish,
            "P90 finish": r.p90_finish,
        }
        for r in results
    ]
)
st.dataframe(table, hide_index=True, use_container_width=True)
st.info(
    "Accuracy mode: every option shares the same no-stop baseline and safety-car draws. These are scenario estimates, not a claimed live prediction. For a real race, set clean pace, tyre loss, safety-car probability, and field gap from current session data."
)

st.divider()
st.subheader("Digital twin: synthetic all-car control room")
st.caption(
    "A compact 20-car simulator that exposes opponent pit plans and event scenarios. Replace the synthetic grid with sourced, pre-decision race state for research use."
)
target_pit = st.slider(
    "Target car planned pit lap",
    current_lap + 1,
    total_laps - 1,
    min(current_lap + 5, total_laps - 1),
    key="twin_pit",
)
if st.button("Run all-car counterfactual", type="primary"):
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
            else pd.DataFrame([{"lap": "—", "state": "None sampled", "duration": "—"}]),
            hide_index=True,
            use_container_width=True,
        )
    trajectory = twin.laps[twin.laps.driver_id == "CAR-08"]
    st.line_chart(trajectory.set_index("lap")[["position"]], y_label="Position")
