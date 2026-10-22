from pathlib import Path
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from racecraft.simulation.engine import RaceState, candidate_strategies, simulate_strategy
from racecraft.simulation.twin import PitPolicy, demo_grid, simulate_twin
from racecraft.visualization.charts import strategy_chart

st.set_page_config(page_title="RaceCraft", page_icon="🏁", layout="wide")
st.title("🏁 RaceCraft")
st.caption("Counterfactual strategy comparisons with explicit uncertainty—not claims of historical certainty.")

with st.sidebar:
    st.header("Decision state")
    total_laps = st.number_input("Race distance", min_value=10, value=57)
    lap = st.number_input("Current lap", min_value=1, max_value=int(total_laps - 1), value=min(35, int(total_laps - 1)))
    position = st.number_input("Current position", min_value=1, max_value=20, value=8)
    tyre_age = st.number_input("Tyre age", min_value=0, value=18)
    compound = st.selectbox("Current compound", ["SOFT", "MEDIUM", "HARD"], index=1)
    clean_lap = st.number_input("Expected clean lap (seconds)", min_value=60.0, value=92.0, step=0.1)
    sc_probability = st.slider("Safety-car / VSC chance in remaining race", 0.0, 0.8, 0.22, 0.01)
    runs = st.select_slider("Monte Carlo runs", [250, 500, 1000, 2000, 5000], value=1000)

state = RaceState(int(lap), int(total_laps), int(tyre_age), compound, int(position), float(clean_lap))
strategies = candidate_strategies(state, horizon=min(10, int(total_laps - lap)))
results = [simulate_strategy(state, x, n_simulations=runs, safety_car_probability=sc_probability, seed=2026) for x in strategies]
results.sort(key=lambda x: x.expected_finish)

winner = results[0]
left, right = st.columns((2, 1))
with left:
    st.plotly_chart(strategy_chart(results[:12]), use_container_width=True)
with right:
    st.subheader("Recommended under these assumptions")
    st.metric("Strategy", winner.strategy.label)
    st.metric("Expected finish", f"P{winner.expected_finish:.1f}")
    st.metric("Points probability", f"{winner.p_points:.0%}")
    st.write(f"80% simulated interval: P{winner.p10_finish:.0f}–P{winner.p90_finish:.0f}")

st.subheader("Candidate details")
table = pd.DataFrame([{"Strategy": r.strategy.label, "Expected finish": round(r.expected_finish, 2),
                       "Points probability": round(r.p_points, 3), "P10 finish": r.p10_finish, "P90 finish": r.p90_finish}
                      for r in results])
st.dataframe(table, hide_index=True, use_container_width=True)
st.info("Baseline simulator uses transparent degradation and pit-loss assumptions. Add calibrated pace, safety-car, and multi-car traffic models before treating results as research conclusions.")

st.divider()
st.subheader("Digital twin: synthetic all-car control room")
st.caption("A compact 20-car simulator that exposes opponent pit plans and event scenarios. Replace the synthetic grid with sourced, pre-decision race state for research use.")
target_pit = st.slider("Target car planned pit lap", int(lap + 1), int(total_laps - 1), min(int(lap + 5), int(total_laps - 1)), key="twin_pit")
if st.button("Run all-car counterfactual", type="primary"):
    profiles, policies = demo_grid()
    twin = simulate_twin(profiles, policies, total_laps=int(total_laps), start_lap=int(lap),
                         target_id="CAR-08", target_strategy=PitPolicy(target_pit, "HARD"), seed=2026)
    col1, col2 = st.columns(2)
    with col1:
        st.write("Final simulated classification")
        st.dataframe(twin.classification, hide_index=True, use_container_width=True)
    with col2:
        st.write("Sampled race events")
        event_rows = [{"lap": x.lap, "state": x.state, "duration": x.duration_laps} for x in twin.events]
        st.dataframe(pd.DataFrame(event_rows) if event_rows else pd.DataFrame([{"lap": "—", "state": "None sampled", "duration": "—"}]), hide_index=True, use_container_width=True)
    trajectory = twin.laps[twin.laps.driver_id == "CAR-08"]
    st.line_chart(trajectory.set_index("lap")[["position"]], y_label="Position")
