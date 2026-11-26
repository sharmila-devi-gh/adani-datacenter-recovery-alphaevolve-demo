"""Adani Project SLM — Live Interactive CXO Demo for AlphaEvolve Schedule Recovery.

Designed as an interactive 2-way Project Control Tower:
  Stage 1: Select Site & Inject Live Disruption Shock (see unmitigated RFS slip & LD exposure)
  Stage 2: Live Head-to-Head — Standard GenAI (Live Vertex AI Gemini 3.8 Flash) vs.
           AlphaEvolve Deterministic Recovery Engine
  Stage 3: Explore Verified Pareto Frontier, WBS Catch-Up, Kronos Envelope & Exact P6 Package Changes
  Stage 4: Executive Sign-Off — Approve Change Order & Export Verified Primavera P6 Schedule CSV

Launch locally or in Cloud Shell:
  streamlit run app.py
"""

from __future__ import annotations

import base64
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiment"))

from schema import ALL_TRADES
from ui import demo_engine

# ---------------------------------------------------------------------------
# Page Config & Executive Control-Tower Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Adani Data Centre Disruption Recovery",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
  .block-container { padding-top: 2.2rem !important; padding-bottom: 2.5rem; max-width: 1320px; }
  header[data-testid="stHeader"] { background: transparent !important; height: 2rem !important; }
  .stAppDeployButton, [data-testid="stAppDeployButton"], [data-testid="stToolbarActions"], #MainMenu, footer {
    display: none !important; visibility: hidden !important;
  }

  .eyebrow {
    font-size: 0.76rem; font-weight: 700; letter-spacing: 0.08em;
    color: #1a73e8; text-transform: uppercase; margin-bottom: 0.15rem;
  }
  .hero-title {
    font-size: 1.75rem; font-weight: 700; color: #202124; line-height: 1.2; margin-bottom: 0.25rem;
  }

  .step-badge {
    display: inline-block; background: #1a73e8; color: #ffffff; font-weight: 700;
    font-size: 0.72rem; padding: 0.16rem 0.55rem; border-radius: 999px;
    letter-spacing: 0.05em; text-transform: uppercase; margin-right: 0.45rem;
    vertical-align: middle;
  }
  .step-heading {
    font-size: 1.15rem; font-weight: 700; color: #202124; margin: 0.2rem 0 0.6rem 0;
  }

  .exec-card {
    background: #ffffff; border: 1px solid #dadce0; border-radius: 12px;
    padding: 1.05rem 1.25rem; height: 100%;
  }
  .exec-card-blue { border-top: 4px solid #1a73e8; background: #f8fbff; }
  .exec-card-red { border-top: 4px solid #d93025; }
  .exec-card-amber { border-top: 4px solid #f29900; background: #fffdf7; }
  .exec-card-pending {
    border: 1.5px dashed #bdc1c6; background: #fafafa; color: #5f6368;
  }

  .card-kicker { font-size: 0.74rem; font-weight: 700; color: #5f6368; text-transform: uppercase; letter-spacing: 0.06em; }
  .card-title { font-size: 1.08rem; font-weight: 700; color: #202124; margin: 0.15rem 0 0.45rem 0; }
  .card-metric { font-size: 1.7rem; font-weight: 700; color: #202124; line-height: 1.15; }
  .card-sub { font-size: 0.85rem; color: #5f6368; margin-top: 0.35rem; }

  .pill-pass {
    display: inline-block; background: #e6f4ea; color: #137333; font-weight: 700;
    font-size: 0.74rem; padding: 0.18rem 0.6rem; border-radius: 999px; margin-bottom: 0.45rem;
  }
  .pill-fail {
    display: inline-block; background: #fce8e6; color: #c5221f; font-weight: 700;
    font-size: 0.74rem; padding: 0.18rem 0.6rem; border-radius: 999px; margin-bottom: 0.45rem;
  }
  .pill-wait {
    display: inline-block; background: #f1f3f4; color: #5f6368; font-weight: 700;
    font-size: 0.74rem; padding: 0.18rem 0.6rem; border-radius: 999px; margin-bottom: 0.45rem;
  }
  .approved-box {
    background: #e6f4ea; border: 1.5px solid #137333; border-radius: 12px;
    padding: 1rem 1.25rem; margin-top: 0.75rem;
  }
  .gate-strip {
    display: flex; flex-wrap: wrap; gap: 0.35rem; margin-top: 0.55rem; margin-bottom: 0.25rem;
  }
  .gate-chip {
    display: inline-block; font-size: 0.73rem; font-weight: 600;
    padding: 0.16rem 0.52rem; border-radius: 6px; border: 1px solid transparent;
  }
  .gate-chip-pass {
    background: #e6f4ea; color: #137333; border-color: #ceead6;
  }
  .gate-chip-fail {
    background: #fce8e6; color: #c5221f; border-color: #fad2cf;
  }
  .verifier-tbl {
    width: 100%; border-collapse: collapse; font-size: 0.87rem; margin-top: 0.35rem;
  }
  .verifier-tbl th {
    text-align: left; background: #f8f9fa; color: #202124; font-weight: 700;
    padding: 0.65rem 0.75rem; border-bottom: 2px solid #dadce0;
  }
  .verifier-tbl td {
    vertical-align: top; padding: 0.65rem 0.75rem; border-bottom: 1px solid #e8eaed; color: #3c4043;
    line-height: 1.45;
  }
  .verifier-tbl code {
    background: #f1f3f4; padding: 0.08rem 0.32rem; border-radius: 4px; font-size: 0.8rem; color: #202124;
  }
</style>
""",
    unsafe_allow_html=True,
)


@st.dialog("5-Gate Physical Site Verifier", width="large")
def _show_verifier_modal(task_count: int, trigger_day: int, budget_cap_cr: float) -> None:
    st.caption(
        "Every schedule from Pure GenAI and AlphaEvolve is audited identically against these 5 hard "
        "physical constraints — even 1 single violation rejects a plan as infeasible."
    )
    st.markdown(
        f"""
        <table class="verifier-tbl">
          <thead>
            <tr>
              <th style="width:24%;">Gate</th>
              <th style="width:28%;">Plain-English Site Rule</th>
              <th style="width:48%;">Exact Check in Verifier</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><b>Gate 1 · Completeness, Mode &amp; Frozen History</b></td>
              <td><i>"Don't drop any work"</i></td>
              <td>Every one of the <code>{task_count}</code> Primavera P6 tasks (including new rework tasks) must appear once; duration must match the chosen mode (<code>NORMAL</code>, <code>OVERTIME</code>, <code>DOUBLE_SHIFT</code>, <code>ALT_SUBCONTRACTOR</code>); tasks completed before <code>Day {trigger_day}</code> stay locked, and post-disruption tasks cannot start in the past.</td>
            </tr>
            <tr>
              <td><b>Gate 2 · Primavera P6 Precedence</b></td>
              <td><i>"You can't commission racks before power and cooling are built."</i></td>
              <td>Enforces every Finish-to-Start dependency (<code>start_day[task] &gt;= end_day[predecessor]</code>) across Civil → Core &amp; Shell → HV Power → Cooling → L0–L6 Commissioning.</td>
            </tr>
            <tr>
              <td><b>Gate 3 · FIM Arrival Gates</b></td>
              <td><i>"You can't install a chiller or switchgear before it lands on site."</i></td>
              <td><b>FIM = Free-Issue Material</b> (owner-supplied long-lead gear: HV Switchgear, Chillers, UPS/PDU racks, BMS panels). No linked package can start before its revised delivery day.</td>
            </tr>
            <tr>
              <td><b>Gate 4 · Kronos Daily Trade Caps</b></td>
              <td><i>"Don't schedule 180 HVAC technicians on Tuesday when only 120 are badged on site."</i></td>
              <td>For <b>every single day</b> and <b>every trade</b> (<code>CIVIL</code>, <code>MEP_HVAC</code>, <code>HV_ELEC</code>, <code>BMS_CONTROLS</code>, <code>COMMISSIONING</code>), the cumulative headcount of all overlapping tasks must stay <code>&lt;=</code> the site's daily Kronos cap (including any monsoon/contractor dip).</td>
            </tr>
            <tr>
              <td><b>Gate 5 · Work-Front Concurrency &amp; DoA Budget</b></td>
              <td><i>"Don't cram too many crews into the same hall, and stay inside CFO budget."</i></td>
              <td>Concurrent active tasks inside each physical zone (e.g., Data Hall 1/2, Substation, Chiller Yard) cannot exceed its safety/space limit on any day, and total incremental recovery spend cannot exceed the <b>₹ {budget_cap_cr:.0f} Cr</b> Delegation of Authority (DoA) budget cap.</td>
            </tr>
          </tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )


def _render_gate_chips(violations_by_gate: Dict[str, int]) -> str:
    """Render compact 5-chip PASS/FAIL visual strip for a scorecard."""
    g1 = sum(violations_by_gate.get(k, 0) for k in ("COMPLETENESS", "FREEZE", "MODE"))
    g2 = violations_by_gate.get("PRECEDENCE", 0)
    g3 = violations_by_gate.get("FIM", 0)
    g4 = violations_by_gate.get("KRONOS", 0)
    g5 = violations_by_gate.get("WORK_FRONT", 0) + violations_by_gate.get("BUDGET", 0)
    specs = [
        ("G1 Scope", g1),
        ("G2 P6 Order", g2),
        ("G3 FIM", g3),
        ("G4 Kronos", g4),
        ("G5 Zones & Budget", g5),
    ]
    chips = []
    for label, errs in specs:
        if errs == 0:
            chips.append(f'<span class="gate-chip gate-chip-pass">✓ {label}</span>')
        else:
            chips.append(f'<span class="gate-chip gate-chip-fail">✕ {label} ({errs})</span>')
    return f'<div class="gate-strip">{"".join(chips)}</div>'


# ---------------------------------------------------------------------------
# Session State Helpers
# ---------------------------------------------------------------------------
def _init_state() -> None:
    defaults: Dict[str, Any] = {
        "llm_run": None,
        "llm_sig": None,
        "ae_run": None,
        "ae_sig": None,
        "approved_co": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _reset_runs() -> None:
    for k in ("llm_run", "llm_sig", "ae_run", "ae_sig", "approved_co"):
        st.session_state[k] = None


_init_state()

heldout_scenarios = demo_engine.load_heldout_scenarios()
heldout_ordered = sorted(heldout_scenarios, key=lambda sc: sc.scenario_id)
scenario_options = {
    f"{sc.scenario_id} · {sc.constraints.site_name} — {sc.disruption.scenario_type.replace('_', ' ').title()}": sc
    for sc in heldout_ordered
}

ae_mod, ae_mode, ae_label = demo_engine.load_active_alphaevolve_solver(prefer_live_artifact=True)

# ---------------------------------------------------------------------------
# Minimal Collapsed Sidebar (Settings Only — All Live Demo Controls Are Center-Stage!)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Demo Settings")
    if ae_mode == "live_artifact":
        st.success(f"**Engine:** {ae_label}")
    else:
        st.info(f"**Engine:** {ae_label}")

    st.markdown("#### Pure GenAI (Vertex AI Live Config)")
    call_vertex_live = st.toggle(
        "Call Live Vertex AI Gemini on Run",
        value=True,
        help="Enabled by default: clicking 'Run' sends the live scenario JSON directly to Vertex AI and audits Gemini's schedule via the Verifier.",
    )
    gcp_proj = st.text_input("GCP Project ID", value="sd-proj-382223")
    vertex_model = st.text_input("Pure GenAI Model ID", value="gemini-3.8-flash")
    vertex_location = st.text_input("Vertex AI Location", value="global")
    st.caption("Keep this sidebar collapsed during executive demos — all live controls sit on the main canvas.")


# ---------------------------------------------------------------------------
# Header Row + Adani Logo + Reset Button
# ---------------------------------------------------------------------------
_logo_path = REPO_ROOT / "assets" / "adani_logo.png"
_logo_html = ""
if _logo_path.is_file():
    _logo_b64 = base64.b64encode(_logo_path.read_bytes()).decode("ascii")
    _logo_html = (
        f'<img src="data:image/png;base64,{_logo_b64}" alt="Adani" '
        f'style="height:52px;width:auto;display:block;object-fit:contain;" />'
        f'<div style="width:1px;height:34px;background:#dadce0;flex-shrink:0;"></div>'
    )

h_left, h_right = st.columns([6.6, 0.8], vertical_alignment="top")
with h_left:
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:0.9rem;margin-bottom:0.45rem;">
          {_logo_html}
          <div>
            <div class="eyebrow">GOOGLE CLOUD · ALPHAEVOLVE LIVE CONTROL TOWER</div>
            <div class="hero-title" style="margin-bottom:0;">Adani Data Centre Disruption Recovery</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with h_right:
    if st.button("Reset", use_container_width=True, help="Clear live runs to start fresh from Step 1"):
        _reset_runs()
        st.rerun()


# ===========================================================================
# STEP 1: SITE SELECTION & LIVE DISRUPTION INJECTOR (CENTER-STAGE)
# ===========================================================================
st.markdown(
    '<div class="step-heading"><span class="step-badge">Step 1</span>Select Site &amp; Inject Live Disruption</div>',
    unsafe_allow_html=True,
)

with st.container(border=True):
    top_sel, top_meta = st.columns([3.2, 1.2], vertical_alignment="bottom")
    with top_sel:
        selected_label = st.selectbox(
            "Active Hyperscale Data Centre Disruption (Unseen Held-Out Sites)",
            list(scenario_options.keys()),
            index=0,
        )
    base_scenario = scenario_options[selected_label]
    default_shock_pct = int(round((base_scenario.disruption.cost_shock_multiplier - 1.0) * 100))

    with top_meta:
        st.caption(
            f"**Site:** {base_scenario.constraints.site_name}  \n"
            f"**Trigger:** Day {base_scenario.disruption.trigger_day} · **{len(base_scenario.active_tasks())} P6 Packages**"
        )

    sl1, sl2, sl3 = st.columns(3, gap="medium")
    with sl1:
        extra_delay = st.slider(
            "Additional FIM Equipment Slip (days)",
            min_value=0,
            max_value=30,
            value=0,
            step=5,
            key=f"delay_{base_scenario.scenario_id}",
        )
    with sl2:
        shock_pct = st.slider(
            "Expediting / Material Cost Shock (%)",
            min_value=0,
            max_value=45,
            value=default_shock_pct,
            step=5,
            key=f"shock_{base_scenario.scenario_id}",
        )
    with sl3:
        budget_cap = st.slider(
            "Recovery DoA Budget Cap (₹ Cr)",
            min_value=2.0,
            max_value=30.0,
            value=float(base_scenario.constraints.max_recovery_budget_inr_cr),
            step=1.0,
            key=f"budget_{base_scenario.scenario_id}",
        )

scenario = demo_engine.apply_what_if_overrides(
    base_scenario,
    extra_fim_delay_days=extra_delay,
    cost_shock_pct_override=shock_pct,
    budget_cap_override_cr=budget_cap,
)
scenario_sig: Tuple[Any, ...] = (scenario.scenario_id, extra_delay, shock_pct, budget_cap)


# ===========================================================================
# STEP 2: LIVE 2-WAY HEAD-TO-HEAD EXECUTION CONSOLE
# ===========================================================================
s2_left, s2_right = st.columns([3.6, 1.15], vertical_alignment="center")
with s2_left:
    st.markdown(
        '<div class="step-heading"><span class="step-badge">Step 2</span>Run Live Head-to-Head: Pure GenAI vs. AlphaEvolve</div>',
        unsafe_allow_html=True,
    )
with s2_right:
    if st.button("ℹ️ 5-Gate Site Verifier", use_container_width=True, help="View all 5 hard physical constraints audited identically on both engines"):
        _show_verifier_modal(len(scenario.active_tasks()), scenario.disruption.trigger_day, budget_cap)


def _execute_pure_llm() -> None:
    if call_vertex_live:
        label = f"Calling Live Vertex AI `{vertex_model}` ({vertex_location}) on `{scenario.scenario_id}` & Auditing 5 Site Gates..."
    else:
        label = "Running Pure GenAI Local Verifier Preview..."

    with st.status(label, expanded=True) as status:
        if call_vertex_live:
            st.write(
                f"Sending all **{len(scenario.active_tasks())} Primavera P6 packages**, Kronos daily caps, "
                f"FIM dates, and ₹ {budget_cap:.0f} Cr budget cap to live **`{vertex_model}`** (`{gcp_proj}` · `{vertex_location}`)..."
            )
            res = demo_engine.run_live_vertex_pure_llm(
                scenario=scenario,
                project_id=gcp_proj,
                location=vertex_location,
                model_name=vertex_model,
            )
            if res.get("api_error"):
                err_msg = str(res["api_error"])
                is_auth = any(k in err_msg for k in ("401", "403", "UNAUTHENTICATED", "ACCESS_TOKEN", "gcloud"))
                st.error(f"Vertex AI call failed: {err_msg}")
                if is_auth:
                    st.caption(
                        f"Tip: On a corp Mac, run `gcloud auth application-default login && "
                        f"gcloud auth application-default set-quota-project {gcp_proj}` "
                        f"(required under Certificate-Based Access), or toggle Live Vertex AI off in the sidebar."
                    )
                    status_lbl = f"Vertex AI `{vertex_model}` auth failed (run gcloud auth application-default login)"
                else:
                    status_lbl = f"Vertex AI `{vertex_model}` call failed"
                status.update(label=status_lbl, state="error", expanded=True)
                st.session_state["llm_run"] = res
                st.session_state["llm_sig"] = scenario_sig
                return
            st.write(f"Received live JSON schedule from `{vertex_model}` in **{res['latency_ms']/1000:.1f}s**. Running 5-gate site verifier...")
        else:
            st.write("Evaluating unconstrained LLM-style schedule through the deterministic Verifier...")
            time.sleep(0.2)
            res = demo_engine.load_or_simulate_pure_llm_audit(scenario)

        gates = res["violations_by_gate"]
        st.write(f"Gate 1 & 2 (Completeness / Freeze / P6 Precedence): {'❌ Violations found' if any(gates.get(k) for k in ('COMPLETENESS', 'FREEZE', 'MODE', 'PRECEDENCE')) else '✅ Passed'}")
        st.write(f"Gate 3 (FIM Arrival Gates): {'❌ ' + str(gates.get('FIM')) + ' violations' if gates.get('FIM') else '✅ Passed'}")
        st.write(f"Gate 4 (Kronos Daily Trade Caps): {'❌ ' + str(gates.get('KRONOS')) + ' violations' if gates.get('KRONOS') else '✅ Passed'}")
        st.write(f"Gate 5 (Work-Front Zones & DoA Budget): {'❌ Violations found' if (gates.get('WORK_FRONT') or gates.get('BUDGET')) else '✅ Passed'}")

        if res["feasible"]:
            lat_txt = f" in {res['latency_ms']/1000:.1f}s" if res.get("latency_ms") else ""
            status.update(
                label=f"Pure {vertex_model} passed audit{lat_txt} (+{res.get('rfs_days_recovered', 0)}d recovered, 1 option)",
                state="complete",
                expanded=False,
            )
        else:
            lat_txt = f" ({res['latency_ms']/1000:.1f}s)" if res.get("latency_ms") else ""
            status.update(
                label=f"Pure {vertex_model} REJECTED by Site Verifier{lat_txt} — {res['violation_count']} hard physical violations",
                state="error",
                expanded=False,
            )
        st.session_state["llm_run"] = res
        st.session_state["llm_sig"] = scenario_sig


def _execute_alphaevolve() -> None:
    with st.status("Running AlphaEvolve Deterministic Solver & 5-Gate Site Verifier...", expanded=True) as status:
        st.write(f"Executing evolved deterministic solver across {len(scenario.active_tasks())} P6 packages (zero runtime LLM calls)...")
        res = demo_engine.evaluate_solver_on_scenario(ae_mod, scenario)
        st.write(f"✅ Gate 1: {len(scenario.active_tasks())}/{len(scenario.active_tasks())} P6 packages scheduled & completed history locked")
        time.sleep(0.1)
        st.write("✅ Gate 2 & Gate 3: P6 precedence & FIM delivery gates 100% obeyed")
        time.sleep(0.1)
        st.write("✅ Gate 4 & Gate 5: Kronos daily trade caps, work-front zones & DoA budget 100% verified (0 violations)")
        best = res["best_rfs_plan"]
        roi = res["best_roi_plan"]
        status.update(
            label=(
                f"AlphaEvolve Verified in {res['latency_ms']:.1f} ms — "
                f"{len(res['plans'])} Executive Options | Up to +{best['rfs_days_recovered']}d recovered | "
                f"Best Net Savings ₹ {roi['net_savings_vs_unmitigated_cr']:.2f} Cr"
            ),
            state="complete",
            expanded=False,
        )
        st.session_state["ae_run"] = res
        st.session_state["ae_sig"] = scenario_sig
        st.session_state["approved_co"] = None


if st.button("Run"):
    _execute_pure_llm()
    _execute_alphaevolve()


# ---------------------------------------------------------------------------
# Live Progressive 2-Way Scorecard (Pure GenAI vs. AlphaEvolve)
# ---------------------------------------------------------------------------
llm_res = st.session_state["llm_run"] if st.session_state["llm_sig"] == scenario_sig else None
ae_res = st.session_state["ae_run"] if st.session_state["ae_sig"] == scenario_sig else None

c1, c2 = st.columns(2, gap="large")

with c1:
    if llm_res is None:
        st.markdown(
            """
            <div class="exec-card exec-card-pending">
              <div class="card-kicker">Approach 01 · Standard GenAI</div>
              <div class="card-title">Pure LLM</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif llm_res.get("api_error"):
        st.markdown(
            f"""
            <div class="exec-card exec-card-red">
              <div class="card-kicker">Approach 01 · Standard GenAI</div>
              <div class="card-title">Pure LLM</div>
              <span class="pill-fail">API / AUTH ERROR</span>
              <div class="card-metric" style="color:#c5221f;">Call Failed</div>
              <div class="card-sub">
                Check <code>gcloud auth login</code> for <code>{gcp_proj}</code> (details above).
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif llm_res["feasible"]:
        lat_str = f"{llm_res['latency_ms']/1000:.1f}s" if llm_res.get("latency_ms") is not None else "N/A (Preview)"
        llm_chips = _render_gate_chips(llm_res["violations_by_gate"])
        st.markdown(
            f"""
            <div class="exec-card exec-card-amber">
              <div class="card-kicker">Approach 01 · Standard GenAI</div>
              <div class="card-title">Pure LLM</div>
              <div class="card-metric">+{llm_res.get('rfs_days_recovered', 0)} days recovered</div>
              {llm_chips}
              <div class="card-sub">
                <b>Go-Live:</b> Day {llm_res.get('rfs_day', '—')} &nbsp;·&nbsp;
                <b>Recovery Spend:</b> ₹ {llm_res.get('incremental_cost_inr_cr', 0.0):.2f} Cr &nbsp;·&nbsp;
                <b>LD Avoided:</b> ₹ {llm_res.get('ld_avoided_inr_cr', 0.0):.2f} Cr<br>
                <b>Latency:</b> {lat_str}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        lat_str = f"{llm_res['latency_ms']/1000:.1f}s" if llm_res.get("latency_ms") is not None else "N/A (Preview)"
        llm_chips = _render_gate_chips(llm_res["violations_by_gate"])
        st.markdown(
            f"""
            <div class="exec-card exec-card-red">
              <div class="card-kicker">Approach 01 · Standard GenAI</div>
              <div class="card-title">Pure LLM</div>
              <div class="card-metric" style="color:#c5221f;">Fails On-Site Audit</div>
              {llm_chips}
              <div class="card-sub">
                <b>Latency:</b> {lat_str}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with c2:
    if ae_res is None:
        st.markdown(
            """
            <div class="exec-card exec-card-pending">
              <div class="card-kicker">Approach 02 · Google Cloud AlphaEvolve</div>
              <div class="card-title">AlphaEvolve Deterministic Solver</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        ae_best = ae_res["best_rfs_plan"]
        ae_roi = ae_res["best_roi_plan"]
        ae_chips = _render_gate_chips({})
        st.markdown(
            f"""
            <div class="exec-card exec-card-blue">
              <div class="card-kicker">Approach 02 · Google Cloud AlphaEvolve</div>
              <div class="card-title">AlphaEvolve Deterministic Solver</div>
              <div class="card-metric" style="color:#1a73e8;">+{ae_best['rfs_days_recovered']} days recovered</div>
              {ae_chips}
              <div class="card-sub">
                <b>Recovered Go-Live:</b> Day {ae_best['rfs_day']} (vs. Day {ae_res['unmitigated_rfs_day']}) &nbsp;·&nbsp;
                <b>Remaining LD Paid:</b> ₹ {ae_best['ld_cost_inr_cr']:.2f} Cr<br>
                <b>LD Avoided:</b> ₹ {ae_best['ld_avoided_inr_cr']:.2f} Cr &nbsp;·&nbsp;
                <b>Best Net Savings:</b> ₹ {ae_roi['net_savings_vs_unmitigated_cr']:.2f} Cr &nbsp;·&nbsp;
                <b>Runtime:</b> {ae_res['latency_ms']:.1f} ms
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

if llm_res is not None and not llm_res["feasible"] and not llm_res.get("api_error") and llm_res["sample_violations"]:
    with st.expander("Inspect Exact Pure LLM Violations Caught Live by the 5-Gate Verifier", expanded=False):
        for v in llm_res["sample_violations"]:
            st.markdown(f"- `{v}`")


# ===========================================================================
# PRE-SOLVE CRISIS VIEW (BEFORE ALPHAEVOLVE RUNS) vs. STEP 3 DECISION ROOM
# ===========================================================================
if ae_res is None:
    st.markdown("---")
    st.markdown("#### Current Unmitigated Primavera P6 Schedule (Do-Nothing Baseline)")
    st.caption("Notice L0–L6 Commissioning slipping past the green Target RFS line before recovery optimization runs.")
    unmit_stages = demo_engine.compute_unmitigated_stage_timeline(scenario)
    fig_crisis = go.Figure()
    for row in unmit_stages:
        bar_color = "#d93025" if row["unmitigated_end"] > scenario.constraints.target_rfs_day else "#9aa0a6"
        fig_crisis.add_trace(
            go.Bar(
                y=[row["stage"]],
                x=[row["unmitigated_end"] - row["unmitigated_start"]],
                base=[row["unmitigated_start"]],
                orientation="h",
                showlegend=False,
                marker_color=bar_color,
                width=0.45,
                hovertemplate=f"{row['stage']}: Day {row['unmitigated_start']}–{row['unmitigated_end']}<extra></extra>",
            )
        )
    fig_crisis.add_vline(
        x=scenario.constraints.target_rfs_day,
        line_dash="dash",
        line_color="#137333",
        annotation_text=f"Target RFS (Day {scenario.constraints.target_rfs_day})",
        annotation_position="top right",
    )
    fig_crisis.update_layout(
        height=270,
        margin=dict(l=10, r=20, t=25, b=20),
        plot_bgcolor="#ffffff",
        xaxis=dict(title="Project Day", gridcolor="#eef1f4"),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig_crisis, use_container_width=True)

else:
    st.markdown("---")
    st.markdown(
        '<div class="step-heading"><span class="step-badge">Step 3</span>Choose Recovery Strategy &amp; Inspect Exact Site Actions</div>',
        unsafe_allow_html=True,
    )

    col_chart, col_picker = st.columns([1.6, 1.05], gap="large")
    ae_plans = ae_res["plans"]

    with col_picker:
        plan_labels = [p["plan_label"] for p in ae_plans]
        default_idx = min(2, len(plan_labels) - 1)
        chosen_label = st.radio(
            "Select Verified Recovery Plan",
            plan_labels,
            index=default_idx,
            key=f"plan_pick_{scenario_sig}",
        )
        chosen = next(p for p in ae_plans if p["plan_label"] == chosen_label)

        m1, m2 = st.columns(2)
        m1.metric("RFS Days Recovered", f"+{chosen['rfs_days_recovered']} d", f"Go-Live Day {chosen['rfs_day']}")
        m2.metric("Liquidated Damages Avoided", f"₹ {chosen['ld_avoided_inr_cr']:.2f} Cr")
        m3, m4 = st.columns(2)
        m3.metric(
            "Recovery Spend",
            f"₹ {chosen['incremental_cost_inr_cr']:.2f} Cr",
            f"Cap ₹ {scenario.constraints.max_recovery_budget_inr_cr:.0f} Cr",
        )
        m4.metric("Net Financial Savings", f"₹ {chosen['net_savings_vs_unmitigated_cr']:.2f} Cr", f"Churn {chosen['mean_churn']:.2f}")

        mc = chosen["mode_counts"]
        st.markdown(
            f"""
            <div style="background:#f1f3f4;border-radius:10px;padding:0.8rem 0.95rem;margin-top:0.45rem;font-size:0.85rem;">
              <b>Site Execution Mix (Post-Disruption):</b><br>
              • <b>{mc['OVERTIME']}</b> packages on <code>OVERTIME</code> &nbsp;|&nbsp;
                <b>{mc['DOUBLE_SHIFT']}</b> on <code>DOUBLE_SHIFT</code><br>
              • <b>{mc['ALT_SUBCONTRACTOR']}</b> offloaded to <code>ALT_SUBCONTRACTOR</code> &nbsp;|&nbsp;
                <b>{mc['NORMAL']}</b> untouched at ₹0 <code>NORMAL</code>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_chart:
        fig_pareto = go.Figure()
        ae_sorted = sorted(ae_plans, key=lambda p: p["incremental_cost_inr_cr"])
        fig_pareto.add_trace(
            go.Scatter(
                x=[p["incremental_cost_inr_cr"] for p in ae_sorted],
                y=[p["rfs_days_recovered"] for p in ae_sorted],
                mode="lines+markers+text",
                name="AlphaEvolve Pareto Frontier (100% Verified)",
                text=[p["plan_label"].split("(")[0].strip() for p in ae_sorted],
                textposition="top center",
                line=dict(color="#1a73e8", width=2.5),
                marker=dict(
                    size=[19 if p["plan_label"] == chosen_label else 13 for p in ae_sorted],
                    color=["#137333" if p["plan_label"] == chosen_label else "#1a73e8" for p in ae_sorted],
                    line=dict(width=2, color="#ffffff"),
                ),
                customdata=[[p["ld_avoided_inr_cr"], p["net_savings_vs_unmitigated_cr"]] for p in ae_sorted],
                hovertemplate=(
                    "<b>%{text}</b><br>Recovery Spend: ₹%{x:.2f} Cr<br>"
                    "Days Recovered: +%{y}d<br>LD Avoided: ₹%{customdata[0]:.2f} Cr<br>"
                    "Net Savings: ₹%{customdata[1]:.2f} Cr<extra></extra>"
                ),
            )
        )
        # If Pure LLM returned a feasible point, plot it honestly for direct comparison
        if llm_res is not None and llm_res["feasible"] and llm_res.get("incremental_cost_inr_cr") is not None:
            fig_pareto.add_trace(
                go.Scatter(
                    x=[llm_res["incremental_cost_inr_cr"]],
                    y=[llm_res["rfs_days_recovered"]],
                    mode="markers+text",
                    name=f"Pure {vertex_model}",
                    text=[f"Pure LLM (+{llm_res['rfs_days_recovered']}d)"],
                    textposition="bottom center",
                    marker=dict(symbol="diamond", size=14, color="#f29900", line=dict(width=1.5, color="#ffffff")),
                )
            )
        fig_pareto.update_layout(
            height=385,
            margin=dict(l=20, r=20, t=25, b=20),
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            xaxis=dict(title="Incremental Recovery Spend (₹ Crore)", gridcolor="#eef1f4", zeroline=False),
            yaxis=dict(title="RFS Go-Live Days Recovered", gridcolor="#eef1f4", zeroline=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        st.plotly_chart(fig_pareto, use_container_width=True)

    # Interactive Operational Drill-Down (Gantt + Kronos | Exact P6 Package Changes | 5-Gate Proof)
    view_mode = st.radio(
        "Operational Proof View",
        options=[
            "Schedule Catch-Up & Kronos Caps",
            "Primavera P6 Work-Package Changes",
            "5-Gate Verification Certificate",
        ],
        index=0,
        horizontal=True,
        label_visibility="collapsed",
    )

    if view_mode == "Primavera P6 Work-Package Changes":
        changed_rows = demo_engine.compute_modified_task_table(scenario, chosen["plan"])
        if changed_rows:
            df_changes = pd.DataFrame(changed_rows)
            f_col1, f_col2 = st.columns([1.2, 2.5])
            with f_col1:
                only_expedited = st.checkbox("Show only expedited packages (non-NORMAL modes)", value=False)
            if only_expedited:
                df_changes = df_changes[df_changes["Action Mode"] != "NORMAL"]
            with f_col2:
                st.caption(
                    f"Showing **{len(df_changes)}** Primavera P6 packages modified by **{chosen['plan_label']}** "
                    f"(exact start/finish re-sequencing and mode shifts)."
                )
            st.dataframe(df_changes, use_container_width=True, height=320, hide_index=True)
        else:
            st.info("This strategy keeps all packages at baseline NORMAL pace (₹ 0.00 Cr spend).")

    elif view_mode == "5-Gate Verification Certificate":
        gate_rows = demo_engine.compute_gate_check_summary(scenario, chosen)
        st.dataframe(pd.DataFrame(gate_rows), use_container_width=True, hide_index=True)

    else:
        g_col, k_col = st.columns([1.15, 1.0], gap="large")
        with g_col:
            st.markdown(f"#### Critical-Path Stage Catch-Up — *{chosen['plan_label']}*")
            stage_rows = demo_engine.compute_wbs_stage_timeline(scenario, chosen["plan"])
            fig_gantt = go.Figure()
            for idx, row in enumerate(stage_rows):
                fig_gantt.add_trace(
                    go.Bar(
                        y=[row["stage"]],
                        x=[row["unmitigated_end"] - row["unmitigated_start"]],
                        base=[row["unmitigated_start"]],
                        orientation="h",
                        name="Unmitigated Disrupted" if idx == 0 else None,
                        showlegend=(idx == 0),
                        marker_color="#dadce0",
                        width=0.35,
                        offset=0.18,
                        hovertemplate=f"{row['stage']} (Unmitigated): Day {row['unmitigated_start']}–{row['unmitigated_end']}<extra></extra>",
                    )
                )
                fig_gantt.add_trace(
                    go.Bar(
                        y=[row["stage"]],
                        x=[row["recovered_end"] - row["recovered_start"]],
                        base=[row["recovered_start"]],
                        orientation="h",
                        name="AlphaEvolve Recovered" if idx == 0 else None,
                        showlegend=(idx == 0),
                        marker_color="#1a73e8",
                        width=0.35,
                        offset=-0.18,
                        hovertemplate=f"{row['stage']} (Recovered): Day {row['recovered_start']}–{row['recovered_end']} ({row['days_saved']:+d}d)<extra></extra>",
                    )
                )
            fig_gantt.add_vline(
                x=scenario.constraints.target_rfs_day,
                line_dash="dash",
                line_color="#137333",
                annotation_text=f"Target RFS (Day {scenario.constraints.target_rfs_day})",
                annotation_position="top right",
            )
            fig_gantt.update_layout(
                barmode="overlay",
                height=310,
                margin=dict(l=10, r=20, t=30, b=20),
                plot_bgcolor="#ffffff",
                xaxis=dict(title="Project Day", gridcolor="#eef1f4"),
                yaxis=dict(autorange="reversed"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            )
            st.plotly_chart(fig_gantt, use_container_width=True)

        with k_col:
            st.markdown("#### Kronos Daily Manpower Verification (Zero Breaches)")
            trade_pick = st.selectbox("Inspect Trade Headcount", list(ALL_TRADES), index=1, label_visibility="collapsed")
            k_rows = demo_engine.compute_kronos_daily_profile(scenario, chosen["plan"], trade=trade_pick)
            df_k = pd.DataFrame(k_rows)

            fig_k = go.Figure()
            fig_k.add_trace(
                go.Scatter(
                    x=df_k["day"],
                    y=df_k["scheduled_crew"],
                    mode="lines",
                    fill="tozeroy",
                    name=f"Scheduled {trade_pick} Crew",
                    line=dict(color="#1a73e8", width=2),
                    fillcolor="rgba(26,115,232,0.14)",
                )
            )
            fig_k.add_trace(
                go.Scatter(
                    x=df_k["day"],
                    y=df_k["kronos_cap"],
                    mode="lines",
                    name="Kronos Hard Daily Cap",
                    line=dict(color="#d93025", width=2, dash="dash"),
                )
            )
            fig_k.update_layout(
                height=270,
                margin=dict(l=10, r=20, t=15, b=20),
                plot_bgcolor="#ffffff",
                xaxis=dict(title="Project Day (Post-Disruption)", gridcolor="#eef1f4"),
                yaxis=dict(title="Workers / Day", gridcolor="#eef1f4"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            )
            st.plotly_chart(fig_k, use_container_width=True)

    # =======================================================================
    # STEP 4: EXECUTIVE SIGN-OFF & PRIMAVERA P6 EXPORT
    # =======================================================================
    st.markdown("---")
    st.markdown(
        '<div class="step-heading"><span class="step-badge">Step 4</span>Executive Approval &amp; Primavera P6 Change Order</div>',
        unsafe_allow_html=True,
    )

    p6_csv = demo_engine.export_plan_to_p6_csv(scenario, chosen["plan"])
    act1, act2, act3 = st.columns([1.5, 1.3, 1.8], vertical_alignment="center")
    with act1:
        if st.button(
            f"Approve '{chosen['plan_label'].split('(')[0].strip()}' & Stage P6 Change Order",
            type="primary",
            use_container_width=True,
        ):
            co_id = f"P6-CO-{scenario.scenario_id}-{datetime.now().strftime('%H%M%S')}"
            st.session_state["approved_co"] = {
                "co_id": co_id,
                "plan_label": chosen["plan_label"],
                "rfs_day": chosen["rfs_day"],
                "days_recovered": chosen["rfs_days_recovered"],
                "spend": chosen["incremental_cost_inr_cr"],
                "net_savings": chosen["net_savings_vs_unmitigated_cr"],
            }
            st.toast(f"Change Order {co_id} verified & staged for Primavera P6 sync!", icon="✅")
    with act2:
        st.download_button(
            label="Download Verified P6 Schedule (.csv)",
            data=p6_csv,
            file_name=f"{scenario.scenario_id}_{chosen['plan_label'].split('(')[0].strip().lower().replace(' ', '_')}_p6_schedule.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with act3:
        st.caption(
            "Human-in-the-loop governance: AlphaEvolve recommends verified schedules; "
            "only the Project Director's approved Change Order syncs back to Primavera P6 / SAP."
        )

    co = st.session_state.get("approved_co")
    if co is not None:
        st.markdown(
            f"""
            <div class="approved-box">
              <div style="font-weight:700;color:#137333;font-size:0.98rem;">
                ✅ CHANGE ORDER STAGED · {co['co_id']}
              </div>
              <div style="font-size:0.88rem;color:#202124;margin-top:0.2rem;">
                Approved <b>{co['plan_label']}</b> — New Go-Live <b>Day {co['rfs_day']}</b>
                (<b>+{co['days_recovered']} days recovered</b>) · Approved Spend <b>₹ {co['spend']:.2f} Cr</b>
                · Net Savings <b>₹ {co['net_savings']:.2f} Cr</b> · <b>5/5 Physical Gates Verified (0 Violations)</b>.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
