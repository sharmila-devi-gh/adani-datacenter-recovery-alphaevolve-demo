"""Deterministic 5-gate hard feasibility verifier and KPI calculator.

Shared byte-for-byte between:
  1. Offline AlphaEvolve client-side evaluator (`experiment/evaluator.py`)
  2. Online Pure LLM vs. Baseline vs. AlphaEvolve benchmark (`scripts/run_benchmark.py`)
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from schema import (
    ALL_TRADES,
    ConstraintViolation,
    PlanMetrics,
    RecoveryPlan,
    Scenario,
    TaskAssignment,
    TaskNode,
)


def compute_unmitigated_rfs_and_ld(scenario: Scenario) -> Tuple[int, float]:
    """Compute the 'do nothing' (NORMAL mode only) disrupted RFS day and LD cost.

    Uses a deterministic earliest-feasible topological schedule in NORMAL mode so
    every recovery plan is measured against an identical disruption baseline.
    """
    tasks = scenario.active_tasks()
    task_map = {t.task_id: t for t in tasks}
    fim_arrivals = scenario.effective_fim_arrivals()
    trigger_day = scenario.disruption.trigger_day

    # Kahn topological ordering deterministic by (baseline_start_day, task_id)
    indeg: Dict[str, int] = {t.task_id: 0 for t in tasks}
    children: Dict[str, List[str]] = defaultdict(list)
    for t in tasks:
        for p in t.predecessors:
            if p in task_map:
                indeg[t.task_id] += 1
                children[p].append(t.task_id)

    ready = sorted(
        [tid for tid, deg in indeg.items() if deg == 0],
        key=lambda tid: (task_map[tid].baseline_start_day, tid),
    )
    order: List[str] = []
    while ready:
        cur = ready.pop(0)
        order.append(cur)
        for nxt in children[cur]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                ready.append(nxt)
                ready.sort(key=lambda tid: (task_map[tid].baseline_start_day, tid))

    trade_usage: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    zone_usage: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    end_days: Dict[str, int] = {}

    for tid in order:
        t = task_map[tid]
        normal_mode = t.get_mode("NORMAL") or t.available_modes[0]
        if t.baseline_end_day <= trigger_day and t.task_id not in {
            rt.task_id for rt in scenario.disruption.added_rework_tasks
        }:
            start_d = t.baseline_start_day
            end_d = t.baseline_end_day
            end_days[tid] = end_d
            for d in range(start_d, end_d):
                for tr, cnt in normal_mode.crew_by_trade.items():
                    trade_usage[d][tr] += cnt
                zone_usage[d][t.work_front_zone] += 1
            continue

        dur = scenario.effective_duration(t, normal_mode)
        earliest = max(
            [trigger_day if t.baseline_start_day >= trigger_day else t.baseline_start_day]
            + [end_days[p] for p in t.predecessors if p in end_days]
        )
        if t.fim_item_id and t.fim_item_id in fim_arrivals:
            earliest = max(earliest, fim_arrivals[t.fim_item_id])

        # Find earliest resource-feasible start day
        start_d = earliest
        while True:
            ok = True
            z_cap = scenario.constraints.zone_max_concurrent.get(t.work_front_zone, 999)
            for d in range(start_d, start_d + dur):
                if zone_usage[d][t.work_front_zone] + 1 > z_cap:
                    ok = False
                    break
                for tr, cnt in normal_mode.crew_by_trade.items():
                    if cnt > 0 and trade_usage[d][tr] + cnt > scenario.effective_kronos_cap(tr, d):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                break
            start_d += 1

        end_d = start_d + dur
        end_days[tid] = end_d
        for d in range(start_d, end_d):
            for tr, cnt in normal_mode.crew_by_trade.items():
                trade_usage[d][tr] += cnt
            zone_usage[d][t.work_front_zone] += 1

    rfs_day = max(end_days.values()) if end_days else scenario.constraints.target_rfs_day
    ld_cost = compute_ld_cost(scenario, rfs_day)
    return rfs_day, ld_cost


def compute_ld_cost(scenario: Scenario, rfs_day: int) -> float:
    """Contractual Liquidated Damages (INR Cr) for RFS slip past grace period."""
    slip = max(0, rfs_day - scenario.constraints.target_rfs_day)
    billable_days = max(0, slip - scenario.constraints.ld_grace_days)
    return round(billable_days * scenario.constraints.ld_rate_inr_cr_per_day, 4)


def verify_plan(
    scenario: Scenario,
    plan: RecoveryPlan,
    unmitigated_rfs: Optional[int] = None,
    unmitigated_ld: Optional[float] = None,
    max_violations: int = 20,
) -> PlanMetrics:
    """Audit a RecoveryPlan against all 5 hard feasibility gates and compute KPIs."""
    if unmitigated_rfs is None or unmitigated_ld is None:
        unmitigated_rfs, unmitigated_ld = compute_unmitigated_rfs_and_ld(scenario)

    tasks = scenario.active_tasks()
    task_map: Dict[str, TaskNode] = {t.task_id: t for t in tasks}
    rework_ids = {t.task_id for t in scenario.disruption.added_rework_tasks}
    fim_arrivals = scenario.effective_fim_arrivals()
    trigger_day = scenario.disruption.trigger_day

    violations: List[ConstraintViolation] = []

    def add_v(gate: str, msg: str, task_id: Optional[str] = None, day: Optional[int] = None) -> None:
        if len(violations) < max_violations:
            violations.append(ConstraintViolation(gate=gate, message=msg, task_id=task_id, day=day))

    # Gate 1: Completeness, valid mode, duration match, and past-task freeze
    expected_ids = set(task_map.keys())
    assigned_ids = set(plan.assignments.keys())
    missing_ids = sorted(expected_ids - assigned_ids)
    extra_ids = sorted(assigned_ids - expected_ids)
    if missing_ids:
        add_v(
            "COMPLETENESS",
            f"Missing {len(missing_ids)} required task(s): {', '.join(missing_ids[:5])}",
            task_id=missing_ids[0],
        )
    if extra_ids:
        add_v(
            "COMPLETENESS",
            f"Unknown extra task_id(s): {', '.join(extra_ids[:5])}",
            task_id=extra_ids[0],
        )

    trade_usage: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    zone_usage: Dict[int, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    trade_tasks: Dict[int, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))

    total_cost = 0.0
    total_churn = 0.0
    active_post_trigger_count = 0
    max_end_day = 0

    for tid in sorted(expected_ids & assigned_ids):
        t = task_map[tid]
        asgn: TaskAssignment = plan.assignments[tid]
        mode_obj = t.get_mode(asgn.selected_mode)
        if mode_obj is None:
            add_v(
                "MODE",
                f"Task {tid} selected invalid mode '{asgn.selected_mode}'",
                task_id=tid,
            )
            continue

        if asgn.start_day < 0 or asgn.end_day <= asgn.start_day:
            add_v(
                "MODE",
                f"Task {tid} has invalid interval [{asgn.start_day}, {asgn.end_day})",
                task_id=tid,
            )
            continue

        is_frozen = (t.baseline_end_day <= trigger_day) and (tid not in rework_ids)
        if is_frozen:
            if (
                asgn.start_day != t.baseline_start_day
                or asgn.end_day != t.baseline_end_day
                or asgn.selected_mode != t.baseline_mode
            ):
                add_v(
                    "FREEZE",
                    f"Completed task {tid} (baseline end Day {t.baseline_end_day} <= trigger Day {trigger_day}) was altered",
                    task_id=tid,
                    day=asgn.start_day,
                )
        else:
            expected_dur = scenario.effective_duration(t, mode_obj)
            if (asgn.end_day - asgn.start_day) != expected_dur:
                add_v(
                    "MODE",
                    f"Task {tid} duration mismatch for mode {asgn.selected_mode}: got {asgn.end_day - asgn.start_day}d, expected {expected_dur}d",
                    task_id=tid,
                )
            # In-progress tasks (started before trigger_day, finish after) cannot shift start earlier than baseline
            if tid not in rework_ids and t.baseline_start_day < trigger_day:
                if asgn.start_day < t.baseline_start_day:
                    add_v(
                        "FREEZE",
                        f"In-progress task {tid} shifted earlier than baseline start Day {t.baseline_start_day}",
                        task_id=tid,
                        day=asgn.start_day,
                    )
            else:
                if asgn.start_day < trigger_day:
                    add_v(
                        "FREEZE",
                        f"Post-disruption task {tid} started on Day {asgn.start_day} before disruption trigger Day {trigger_day}",
                        task_id=tid,
                        day=asgn.start_day,
                    )

            total_cost += scenario.effective_cost(t, mode_obj)
            total_churn += mode_obj.churn_penalty
            active_post_trigger_count += 1

        # Gate 3: FIM material arrival gate
        if not is_frozen and t.fim_item_id and t.fim_item_id in fim_arrivals:
            req_day = fim_arrivals[t.fim_item_id]
            if asgn.start_day < req_day:
                add_v(
                    "FIM",
                    f"Task {tid} starts Day {asgn.start_day} before FIM '{t.fim_item_id}' arrives on Day {req_day}",
                    task_id=tid,
                    day=asgn.start_day,
                )

        max_end_day = max(max_end_day, asgn.end_day)
        # Guard against pathological multi-thousand-day hallucinated dates
        if asgn.end_day - asgn.start_day > 365 or asgn.end_day > 2000:
            add_v(
                "MODE",
                f"Task {tid} scheduled beyond 2000-day horizon: [{asgn.start_day}, {asgn.end_day})",
                task_id=tid,
                day=asgn.start_day,
            )
            continue

        for d in range(asgn.start_day, asgn.end_day):
            zone_usage[d][t.work_front_zone].append(tid)
            for tr, cnt in mode_obj.crew_by_trade.items():
                if cnt > 0:
                    trade_usage[d][tr] += cnt
                    trade_tasks[d][tr].append(tid)

    # Gate 2: P6 Finish-to-Start precedence
    for tid in sorted(expected_ids & assigned_ids):
        t = task_map[tid]
        asgn = plan.assignments[tid]
        for pred_id in t.predecessors:
            if pred_id in plan.assignments:
                pred_end = plan.assignments[pred_id].end_day
                if asgn.start_day < pred_end:
                    add_v(
                        "PRECEDENCE",
                        f"Task {tid} starts Day {asgn.start_day} before predecessor {pred_id} finishes Day {pred_end}",
                        task_id=tid,
                        day=asgn.start_day,
                    )

    # Gate 4: Kronos daily trade headcount caps (checked post-trigger and overall)
    for d in sorted(trade_usage.keys()):
        if d < trigger_day:
            continue
        for tr in ALL_TRADES:
            used = trade_usage[d].get(tr, 0)
            cap = scenario.effective_kronos_cap(tr, d)
            if used > cap:
                offenders = ", ".join(trade_tasks[d][tr][:4])
                add_v(
                    "KRONOS",
                    f"Day {d}: {tr} headcount {used} exceeds Kronos cap {cap} (tasks: {offenders})",
                    day=d,
                )

    # Gate 5: Physical work-front concurrency & recovery budget ceiling
    for d in sorted(zone_usage.keys()):
        if d < trigger_day:
            continue
        for zone, active_tids in zone_usage[d].items():
            z_cap = scenario.constraints.zone_max_concurrent.get(zone, 999)
            if len(active_tids) > z_cap:
                add_v(
                    "WORK_FRONT",
                    f"Day {d}: Zone '{zone}' has {len(active_tids)} concurrent tasks > limit {z_cap} ({', '.join(active_tids[:4])})",
                    day=d,
                )

    total_cost = round(total_cost, 4)
    if total_cost > scenario.constraints.max_recovery_budget_inr_cr + 1e-6:
        add_v(
            "BUDGET",
            f"Recovery cost ₹{total_cost:.2f} Cr exceeds site cap ₹{scenario.constraints.max_recovery_budget_inr_cr:.2f} Cr",
        )

    feasible = len(violations) == 0
    rfs_day = max_end_day if max_end_day > 0 else unmitigated_rfs
    rfs_slip = max(0, rfs_day - scenario.constraints.target_rfs_day)
    rfs_recovered = unmitigated_rfs - rfs_day
    ld_cost = compute_ld_cost(scenario, rfs_day)
    ld_avoided = round(max(0.0, unmitigated_ld - ld_cost), 4)
    mean_churn = round(total_churn / max(1, active_post_trigger_count), 4)

    return PlanMetrics(
        plan_label=plan.plan_label,
        feasible=feasible,
        rfs_day=rfs_day,
        rfs_slip_days=rfs_slip,
        rfs_days_recovered=rfs_recovered,
        incremental_cost_inr_cr=total_cost,
        ld_cost_inr_cr=ld_cost,
        ld_avoided_inr_cr=ld_avoided,
        net_total_cost_inr_cr=round(total_cost + ld_cost, 4),
        mean_churn=mean_churn,
        violations=tuple(violations),
    )
