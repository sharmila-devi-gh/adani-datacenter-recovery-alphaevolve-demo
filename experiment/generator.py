"""Deterministic synthetic Data Centre capital-project benchmark generator.

Generates 12 evolution scenarios (train_scenarios/) and 6 unseen validation
scenarios (heldout_scenarios/), including:
  - H01: 30-day Chiller/Switchgear FIM slip + 25% steel/copper cost shock
  - H02: Mid-build Water-to-Air Cooling spec redesign blocking Data Hall 2

All figures are 100% synthetic (zero confidential customer records).
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from schema import (
    ALL_TRADES,
    DisruptionEvent,
    ExecutionMode,
    Scenario,
    ScenarioType,
    SiteConstraints,
    TaskNode,
    Trade,
    WbsStage,
    scenario_to_dict,
)

ZONES = (
    "SUBSTRUCTURE",
    "DATA_HALL_1",
    "DATA_HALL_2",
    "CHILLER_YARD",
    "SUBSTATION_HV",
    "COMMISSIONING_BUS",
)

FIM_ITEMS = (
    "HV_SWITCHGEAR",
    "CHILLER_PLANT_A",
    "CHILLER_PLANT_B",
    "UPS_PDU_RACKS",
    "GENSET_BANK",
    "BMS_PANELS",
)

STAGE_CONFIG: List[Tuple[WbsStage, str, Tuple[Trade, ...], Tuple[str, ...]]] = [
    ("CIVIL", "Civil & Foundation", ("CIVIL",), ("SUBSTRUCTURE",)),
    ("CORE_SHELL", "Core & Shell", ("CIVIL", "HV_ELEC"), ("DATA_HALL_1", "DATA_HALL_2")),
    ("MEP_POWER", "HV Power & UPS/PDU", ("HV_ELEC", "BMS_CONTROLS"), ("SUBSTATION_HV", "DATA_HALL_1", "DATA_HALL_2")),
    ("COOLING", "Chiller & Cooling Plant", ("MEP_HVAC", "HV_ELEC"), ("CHILLER_YARD", "DATA_HALL_1", "DATA_HALL_2")),
    ("L0_L6_CX", "L0-L6 Commissioning", ("COMMISSIONING", "BMS_CONTROLS", "MEP_HVAC"), ("COMMISSIONING_BUS", "DATA_HALL_1", "DATA_HALL_2")),
]


def _build_modes(
    rng: random.Random,
    base_dur: int,
    primary_trade: Trade,
    secondary_trade: Optional[Trade],
    base_crew: int,
    base_cost_unit: float,
) -> Tuple[ExecutionMode, ...]:
    """Create 4 realistic execution modes with deliberate Kronos vs. cost trade-offs."""
    normal_crew: Dict[str, int] = {primary_trade: base_crew}
    if secondary_trade and secondary_trade != primary_trade:
        normal_crew[secondary_trade] = max(2, base_crew // 3)

    ot_dur = max(2, int(round(base_dur * rng.uniform(0.80, 0.86))))
    ds_dur = max(2, int(round(base_dur * rng.uniform(0.64, 0.72))))
    sub_dur = max(2, int(round(base_dur * rng.uniform(0.72, 0.80))))

    # Ensure strictly faster crashing modes where duration permits
    if base_dur >= 6:
        ot_dur = min(ot_dur, base_dur - 1)
        ds_dur = min(ds_dur, max(2, ot_dur - 1))
        sub_dur = min(sub_dur, ot_dur)

    # DOUBLE_SHIFT uses ~1.55x internal Kronos crew; ALT_SUBCONTRACTOR uses ~0.45x internal crew
    ds_crew = {k: max(2, int(round(v * 1.55))) for k, v in normal_crew.items()}
    sub_crew = {k: max(1, int(round(v * 0.45))) for k, v in normal_crew.items()}

    return (
        ExecutionMode(
            mode="NORMAL",
            duration_days=base_dur,
            incremental_cost_inr_cr=0.0,
            crew_by_trade=normal_crew,
            churn_penalty=0.0,
        ),
        ExecutionMode(
            mode="OVERTIME",
            duration_days=ot_dur,
            incremental_cost_inr_cr=round(base_cost_unit * rng.uniform(0.35, 0.55), 3),
            crew_by_trade=dict(normal_crew),
            churn_penalty=0.3,
        ),
        ExecutionMode(
            mode="DOUBLE_SHIFT",
            duration_days=ds_dur,
            incremental_cost_inr_cr=round(base_cost_unit * rng.uniform(0.75, 1.05), 3),
            crew_by_trade=ds_crew,
            churn_penalty=0.7,
        ),
        ExecutionMode(
            mode="ALT_SUBCONTRACTOR",
            duration_days=sub_dur,
            incremental_cost_inr_cr=round(base_cost_unit * rng.uniform(1.15, 1.55), 3),
            crew_by_trade=sub_crew,
            churn_penalty=1.0,
        ),
    )


def _generate_single_scenario(
    scenario_id: str,
    split: str,
    site_name: str,
    scenario_type: ScenarioType,
    num_tasks: int,
    seed: int,
    custom_summary: Optional[str] = None,
    slip_days: int = 30,
    cost_shock: float = 1.25,
) -> Scenario:
    rng = random.Random(seed)

    # Distribute tasks across 5 realistic DC build stages
    stage_weights = [0.16, 0.20, 0.24, 0.22, 0.18]
    stage_counts = [max(4, int(round(num_tasks * w))) for w in stage_weights]
    stage_counts[-1] += num_tasks - sum(stage_counts)

    # Base Kronos daily headcount caps (tight enough that naive double-shift saturates them!)
    kronos_caps: Dict[str, int] = {
        "CIVIL": rng.randint(46, 56),
        "MEP_HVAC": rng.randint(36, 44),
        "HV_ELEC": rng.randint(38, 46),
        "BMS_CONTROLS": rng.randint(24, 30),
        "COMMISSIONING": rng.randint(26, 34),
    }
    zone_caps: Dict[str, int] = {
        "SUBSTRUCTURE": 3,
        "DATA_HALL_1": 4,
        "DATA_HALL_2": 4,
        "CHILLER_YARD": 3,
        "SUBSTATION_HV": 3,
        "COMMISSIONING_BUS": 3,
    }

    # Build task DAG stage by stage
    raw_tasks: List[Dict] = []
    stage_task_ids: Dict[WbsStage, List[str]] = {cfg[0]: [] for cfg in STAGE_CONFIG}
    task_end_est: Dict[str, int] = {}

    global_idx = 1
    for stage_idx, (stage, stage_title, trades, zones) in enumerate(STAGE_CONFIG):
        count = stage_counts[stage_idx]
        prev_stage_ids = stage_task_ids[STAGE_CONFIG[stage_idx - 1][0]] if stage_idx > 0 else []

        for local_i in range(count):
            tid = f"P6-{stage[:3]}-{global_idx:03d}"
            global_idx += 1
            primary_trade = trades[local_i % len(trades)]
            secondary_trade = trades[(local_i + 1) % len(trades)] if len(trades) > 1 and rng.random() < 0.45 else None
            zone = zones[local_i % len(zones)]

            base_dur = rng.randint(8, 20)
            base_crew = rng.randint(10, 18) if primary_trade in ("CIVIL", "MEP_HVAC", "HV_ELEC") else rng.randint(7, 13)
            base_cost_unit = round(rng.uniform(0.45, 1.35), 2)

            preds: List[str] = []
            if prev_stage_ids:
                # Connect to 1-2 predecessor tasks from previous stage
                k_prev = min(len(prev_stage_ids), rng.choice([1, 1, 2]))
                preds.extend(rng.sample(prev_stage_ids, k_prev))
            if local_i >= 2 and rng.random() < 0.55:
                # Intra-stage sequential dependency
                preds.append(stage_task_ids[stage][local_i - rng.choice([1, 2])])
            preds = sorted(set(preds))

            # Assign FIM items to key MEP_POWER and COOLING tasks
            fim_item: Optional[str] = None
            if stage == "MEP_POWER" and local_i in (0, 2, 4):
                fim_item = ("HV_SWITCHGEAR", "UPS_PDU_RACKS", "GENSET_BANK")[local_i // 2]
            elif stage == "COOLING" and local_i in (0, 2):
                fim_item = ("CHILLER_PLANT_A", "CHILLER_PLANT_B")[local_i // 2]
            elif stage == "L0_L6_CX" and local_i == 0:
                fim_item = "BMS_PANELS"

            modes = _build_modes(rng, base_dur, primary_trade, secondary_trade, base_crew, base_cost_unit)
            raw_tasks.append(
                {
                    "task_id": tid,
                    "name": f"{stage_title} Pkg {local_i + 1:02d} ({zone})",
                    "wbs_stage": stage,
                    "work_front_zone": zone,
                    "predecessors": tuple(preds),
                    "fim_item_id": fim_item,
                    "modes": modes,
                }
            )
            stage_task_ids[stage].append(tid)

    # Compute resource-feasible NORMAL baseline schedule so baseline itself has 0 violations
    trade_use: Dict[int, Dict[str, int]] = {}
    zone_use: Dict[int, Dict[str, int]] = {}
    baseline_tasks: List[TaskNode] = []
    fim_base_arrival: Dict[str, int] = {}

    for rt in raw_tasks:
        normal = rt["modes"][0]
        earliest = max([0] + [task_end_est[p] for p in rt["predecessors"]])
        start_d = earliest
        dur = normal.duration_days
        while True:
            ok = True
            for d in range(start_d, start_d + dur):
                if zone_use.setdefault(d, {}).get(rt["work_front_zone"], 0) + 1 > zone_caps[rt["work_front_zone"]]:
                    ok = False
                    break
                for tr, cnt in normal.crew_by_trade.items():
                    if trade_use.setdefault(d, {}).get(tr, 0) + cnt > kronos_caps[tr]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                break
            start_d += 1

        end_d = start_d + dur
        task_end_est[rt["task_id"]] = end_d
        for d in range(start_d, end_d):
            zone_use[d][rt["work_front_zone"]] = zone_use[d].get(rt["work_front_zone"], 0) + 1
            for tr, cnt in normal.crew_by_trade.items():
                trade_use[d][tr] = trade_use[d].get(tr, 0) + cnt

        if rt["fim_item_id"] is not None:
            # FIM arrives 3 days before baseline task start
            cur_arr = fim_base_arrival.get(rt["fim_item_id"])
            cand_arr = max(0, start_d - 3)
            fim_base_arrival[rt["fim_item_id"]] = min(cur_arr, cand_arr) if cur_arr is not None else cand_arr

        baseline_tasks.append(
            TaskNode(
                task_id=rt["task_id"],
                name=rt["name"],
                wbs_stage=rt["wbs_stage"],
                work_front_zone=rt["work_front_zone"],
                predecessors=rt["predecessors"],
                fim_item_id=rt["fim_item_id"],
                baseline_start_day=start_d,
                baseline_end_day=end_d,
                baseline_mode="NORMAL",
                available_modes=rt["modes"],
            )
        )

    for fim in FIM_ITEMS:
        fim_base_arrival.setdefault(fim, 35)

    target_rfs = max(t.baseline_end_day for t in baseline_tasks)
    # Trigger disruption ~28% into the project (after Civil/early Core is done)
    trigger_day = max(20, int(round(target_rfs * 0.28)))

    # Construct disruption event
    fim_overrides: Dict[str, int] = {}
    dur_mults: Dict[str, float] = {}
    mp_overrides: Dict[str, Tuple[int, int, int]] = {}
    rework_tasks: List[TaskNode] = []

    post_mep = [t for t in baseline_tasks if t.wbs_stage == "MEP_POWER" and t.baseline_start_day >= trigger_day]
    post_cool = [t for t in baseline_tasks if t.wbs_stage == "COOLING" and t.baseline_start_day >= trigger_day]
    post_cx = [t for t in baseline_tasks if t.wbs_stage == "L0_L6_CX" and t.baseline_start_day >= trigger_day]

    if scenario_type == "MATERIAL_SLIP_COST_SHOCK":
        fim_overrides["HV_SWITCHGEAR"] = max(trigger_day + 10, fim_base_arrival["HV_SWITCHGEAR"] + slip_days)
        fim_overrides["CHILLER_PLANT_A"] = max(trigger_day + 12, fim_base_arrival["CHILLER_PLANT_A"] + slip_days)
        for t in post_mep[:3]:
            dur_mults[t.task_id] = 1.15
        summary = custom_summary or (
            f"{slip_days}-day HV Switchgear & Chiller FIM slip + {int(round((cost_shock - 1)*100))}% material/expediting cost shock"
        )
    elif scenario_type == "COOLING_SPEC_CHANGE":
        fim_overrides["CHILLER_PLANT_A"] = max(trigger_day + 10, fim_base_arrival["CHILLER_PLANT_A"] + 22)
        fim_overrides["CHILLER_PLANT_B"] = max(trigger_day + 12, fim_base_arrival["CHILLER_PLANT_B"] + 26)
        for t in post_cool[:4]:
            dur_mults[t.task_id] = 1.35
        # Add 3 rework packages (Design -> Piping -> Rack Provisioning tie-in)
        anchor_pred = post_cool[0].predecessors[0] if (post_cool and post_cool[0].predecessors) else baseline_tasks[0].task_id
        rw1 = TaskNode(
            task_id="P6-RWK-901",
            name="Water-to-Air Cooling Redesign & CFD Sign-off",
            wbs_stage="COOLING",
            work_front_zone="CHILLER_YARD",
            predecessors=(anchor_pred,),
            fim_item_id=None,
            baseline_start_day=trigger_day,
            baseline_end_day=trigger_day + 14,
            baseline_mode="NORMAL",
            available_modes=_build_modes(rng, 14, "MEP_HVAC", "BMS_CONTROLS", 10, 0.95),
        )
        rw2 = TaskNode(
            task_id="P6-RWK-902",
            name="Data Hall 2 Air-Cooled Plenum & Duct Rework",
            wbs_stage="COOLING",
            work_front_zone="DATA_HALL_2",
            predecessors=("P6-RWK-901",),
            fim_item_id="CHILLER_PLANT_B",
            baseline_start_day=trigger_day + 14,
            baseline_end_day=trigger_day + 32,
            baseline_mode="NORMAL",
            available_modes=_build_modes(rng, 18, "MEP_HVAC", "HV_ELEC", 14, 1.35),
        )
        rw3 = TaskNode(
            task_id="P6-RWK-903",
            name="Revised Cooling Controls & L3 Integration Re-Test",
            wbs_stage="L0_L6_CX",
            work_front_zone="COMMISSIONING_BUS",
            predecessors=("P6-RWK-902",) + ((post_cx[0].task_id,) if post_cx else ()),
            fim_item_id="BMS_PANELS",
            baseline_start_day=trigger_day + 32,
            baseline_end_day=trigger_day + 46,
            baseline_mode="NORMAL",
            available_modes=_build_modes(rng, 14, "COMMISSIONING", "BMS_CONTROLS", 11, 1.15),
        )
        rework_tasks = [rw1, rw2, rw3]
        summary = custom_summary or (
            "Mid-build customer spec change (Water-to-Air cooling redesign) adding 3 rework packages & 26d chiller slip"
        )
    elif scenario_type == "MANPOWER_DIP":
        mp_overrides["MEP_HVAC"] = (trigger_day, trigger_day + 45, int(round(kronos_caps["MEP_HVAC"] * 0.72)))
        mp_overrides["HV_ELEC"] = (trigger_day + 10, trigger_day + 50, int(round(kronos_caps["HV_ELEC"] * 0.75)))
        fim_overrides["UPS_PDU_RACKS"] = max(trigger_day + 8, fim_base_arrival["UPS_PDU_RACKS"] + 16)
        summary = custom_summary or (
            "45-day monsoon/contractor Kronos manpower dip (-28% MEP_HVAC, -25% HV_ELEC) + 16d UPS/PDU slip"
        )
    else:  # COMPOUND
        fim_overrides["HV_SWITCHGEAR"] = max(trigger_day + 10, fim_base_arrival["HV_SWITCHGEAR"] + 25)
        fim_overrides["CHILLER_PLANT_A"] = max(trigger_day + 10, fim_base_arrival["CHILLER_PLANT_A"] + 20)
        mp_overrides["HV_ELEC"] = (trigger_day + 5, trigger_day + 40, int(round(kronos_caps["HV_ELEC"] * 0.78)))
        for t in (post_cool[:2] + post_mep[:2]):
            dur_mults[t.task_id] = 1.22
        summary = custom_summary or (
            f"Compound disruption: 25d switchgear slip, 20d chiller slip, -22% HV_ELEC dip, +{int(round((cost_shock - 1)*100))}% cost shock"
        )

    ld_rate = round(rng.uniform(0.55, 0.95), 2)  # ₹0.55–0.95 Cr / day of RFS slip
    max_budget = round(rng.uniform(14.0, 22.0), 1)  # ₹14–22 Cr recovery DoA budget

    constraints = SiteConstraints(
        site_name=site_name,
        target_rfs_day=target_rfs,
        ld_grace_days=5,
        ld_rate_inr_cr_per_day=ld_rate,
        kronos_daily_cap=kronos_caps,
        fim_base_arrival_day=fim_base_arrival,
        zone_max_concurrent=zone_caps,
        max_recovery_budget_inr_cr=max_budget,
    )
    disruption = DisruptionEvent(
        event_id=f"EVT-{scenario_id}",
        scenario_type=scenario_type,
        summary=summary,
        trigger_day=trigger_day,
        fim_arrival_overrides=fim_overrides,
        duration_multipliers=dur_mults,
        cost_shock_multiplier=cost_shock,
        manpower_cap_overrides=mp_overrides,
        added_rework_tasks=tuple(rework_tasks),
    )

    return Scenario(
        scenario_id=scenario_id,
        split=split,  # type: ignore[arg-type]
        description=f"{site_name} ({len(baseline_tasks) + len(rework_tasks)} tasks): {summary}",
        tasks=tuple(baseline_tasks),
        constraints=constraints,
        disruption=disruption,
    )


def generate_benchmark_suite(data_root: Path) -> Tuple[List[Scenario], List[Scenario]]:
    """Generate 12 train scenarios + 6 unseen held-out scenarios deterministically."""
    train_dir = data_root / "train_scenarios"
    heldout_dir = data_root / "heldout_scenarios"
    train_dir.mkdir(parents=True, exist_ok=True)
    heldout_dir.mkdir(parents=True, exist_ok=True)

    train_specs: List[Tuple[str, str, ScenarioType, int, int, int, float]] = [
        ("T01", "Synthetic DC West-1", "MATERIAL_SLIP_COST_SHOCK", 80, 101, 22, 1.20),
        ("T02", "Synthetic DC West-2", "MATERIAL_SLIP_COST_SHOCK", 95, 102, 32, 1.28),
        ("T03", "Synthetic DC South-1", "MATERIAL_SLIP_COST_SHOCK", 110, 103, 38, 1.30),
        ("T04", "Synthetic DC West-3", "COOLING_SPEC_CHANGE", 82, 104, 24, 1.18),
        ("T05", "Synthetic DC South-2", "COOLING_SPEC_CHANGE", 96, 105, 28, 1.22),
        ("T06", "Synthetic DC North-1", "COOLING_SPEC_CHANGE", 108, 106, 26, 1.25),
        ("T07", "Synthetic DC East-1", "MANPOWER_DIP", 84, 107, 18, 1.12),
        ("T08", "Synthetic DC West-4", "MANPOWER_DIP", 98, 108, 20, 1.15),
        ("T09", "Synthetic DC South-3", "MANPOWER_DIP", 112, 109, 22, 1.18),
        ("T10", "Synthetic DC North-2", "COMPOUND", 88, 110, 25, 1.24),
        ("T11", "Synthetic DC West-5", "COMPOUND", 100, 111, 28, 1.26),
        ("T12", "Synthetic DC South-4", "COMPOUND", 114, 112, 30, 1.30),
    ]

    heldout_specs: List[Tuple[str, str, ScenarioType, int, int, int, float, Optional[str]]] = [
        (
            "H01",
            "Navi Mumbai DC-2 (Synthetic Hero 1)",
            "MATERIAL_SLIP_COST_SHOCK",
            96,
            2001,
            30,
            1.25,
            "30-day HV Switchgear & Chiller FIM slip + 25% steel/copper cost shock",
        ),
        (
            "H02",
            "Chennai DC-1 (Synthetic Hero 2)",
            "COOLING_SPEC_CHANGE",
            102,
            2002,
            26,
            1.20,
            "Mid-build Water-to-Air cooling redesign blocking Data Hall 2 rack provisioning",
        ),
        ("H03", "Hyderabad DC-1 (Synthetic)", "MANPOWER_DIP", 90, 2003, 20, 1.15, None),
        ("H04", "Noida DC-3 (Synthetic)", "COMPOUND", 108, 2004, 28, 1.28, None),
        ("H05", "Pune DC-2 (Synthetic)", "MATERIAL_SLIP_COST_SHOCK", 86, 2005, 35, 1.30, None),
        ("H06", "Navi Mumbai DC-4 (Synthetic)", "COMPOUND", 116, 2006, 30, 1.25, None),
    ]

    train_scenarios: List[Scenario] = []
    for sid, site, stype, n_tasks, seed, slip, shock in train_specs:
        sc = _generate_single_scenario(sid, "train", site, stype, n_tasks, seed, slip_days=slip, cost_shock=shock)
        train_scenarios.append(sc)
        (train_dir / f"{sid}.json").write_text(json.dumps(scenario_to_dict(sc), indent=2), encoding="utf-8")

    heldout_scenarios: List[Scenario] = []
    for sid, site, stype, n_tasks, seed, slip, shock, summary in heldout_specs:
        sc = _generate_single_scenario(
            sid, "heldout", site, stype, n_tasks, seed, custom_summary=summary, slip_days=slip, cost_shock=shock
        )
        heldout_scenarios.append(sc)
        (heldout_dir / f"{sid}.json").write_text(json.dumps(scenario_to_dict(sc), indent=2), encoding="utf-8")

    return train_scenarios, heldout_scenarios


if __name__ == "__main__":
    root = Path(__file__).resolve().parent / "data"
    train, heldout = generate_benchmark_suite(root)
    print(f"Generated {len(train)} train scenarios and {len(heldout)} held-out scenarios in {root}")
