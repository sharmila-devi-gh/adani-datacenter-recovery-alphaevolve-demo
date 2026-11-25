"""Seed recovery solver for AlphaEvolve optimization.

AlphaEvolve mutates ONLY the code inside the marked mutable region below to
discover better multi-objective Data Centre recovery algorithms while preserving
the external function signature `generate_recovery_plans(scenario, max_plans=5)`.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

from schema import (
    ExecutionMode,
    RecoveryPlan,
    Scenario,
    TaskAssignment,
    TaskNode,
)


def _find_earliest_feasible_start(
    task: TaskNode,
    mode: ExecutionMode,
    earliest: int,
    scenario: Scenario,
    trade_usage: Dict[int, Dict[str, int]],
    zone_usage: Dict[int, Dict[str, int]],
) -> Tuple[int, int]:
    """Return earliest (start_day, end_day) respecting Kronos and zone caps."""
    dur = scenario.effective_duration(task, mode)
    z_cap = scenario.constraints.zone_max_concurrent.get(task.work_front_zone, 999)
    start_d = earliest
    while True:
        ok = True
        for d in range(start_d, start_d + dur):
            if zone_usage[d][task.work_front_zone] + 1 > z_cap:
                ok = False
                break
            for tr, cnt in mode.crew_by_trade.items():
                if cnt > 0 and trade_usage[d][tr] + cnt > scenario.effective_kronos_cap(tr, d):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return start_d, start_d + dur
        start_d += 1


def _commit_task(
    task: TaskNode,
    mode: ExecutionMode,
    start_d: int,
    end_d: int,
    trade_usage: Dict[int, Dict[str, int]],
    zone_usage: Dict[int, Dict[str, int]],
) -> None:
    for d in range(start_d, end_d):
        zone_usage[d][task.work_front_zone] += 1
        for tr, cnt in mode.crew_by_trade.items():
            if cnt > 0:
                trade_usage[d][tr] += cnt


def generate_recovery_plans(scenario: Scenario, max_plans: int = 5) -> List[RecoveryPlan]:
    """Return 3 to `max_plans` strictly feasible, Pareto-diverse recovery plans.

    Available optimization levers inside EVOLVE-BLOCK:
      - Smarter topological priority rules (e.g. Latest Finish Time, downstream
        critical-path depth, FIM-unblock urgency, resource bottleneck weight).
      - Smarter mode selection per task:
          * NORMAL: 0 cost, 0 churn
          * OVERTIME: ~18% faster, low cost, 1.0x crew
          * DOUBLE_SHIFT: ~32% faster, medium cost, 1.55x crew (pressures Kronos!)
          * ALT_SUBCONTRACTOR: ~25% faster, higher cost, 0.45x crew (relieves Kronos bottlenecks!)
      - Fast post-schedule critical-path local search / neighborhood repair.
      - Diverse scalarization weights + non-dominated Pareto pruning across
        (RFS days recovered, incremental cost, LD avoided, crew churn).
    """
# EVOLVE-BLOCK-START
    tasks = scenario.active_tasks()
    task_map: Dict[str, TaskNode] = {t.task_id: t for t in tasks}
    rework_ids = {t.task_id for t in scenario.disruption.added_rework_tasks}
    fim_arrivals = scenario.effective_fim_arrivals()
    trigger_day = scenario.disruption.trigger_day
    max_budget = scenario.constraints.max_recovery_budget_inr_cr

    # Fixed topological order sorted by baseline_start_day (Gen-0 naive ordering)
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
    topo_order: List[str] = []
    while ready:
        cur = ready.pop(0)
        topo_order.append(cur)
        for nxt in children[cur]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                ready.append(nxt)
                ready.sort(key=lambda tid: (task_map[tid].baseline_start_day, tid))

    def build_greedy_plan(label: str, budget_fraction: float, crash_mode: str) -> RecoveryPlan:
        trade_usage: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        zone_usage: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        assignments: Dict[str, TaskAssignment] = {}
        spent = 0.0
        budget_limit = max_budget * budget_fraction

        for tid in topo_order:
            t = task_map[tid]
            is_frozen = (t.baseline_end_day <= trigger_day) and (tid not in rework_ids)
            if is_frozen:
                mode_obj = t.get_mode(t.baseline_mode) or t.available_modes[0]
                assignments[tid] = TaskAssignment(
                    task_id=tid,
                    start_day=t.baseline_start_day,
                    end_day=t.baseline_end_day,
                    selected_mode=mode_obj.mode,
                )
                _commit_task(t, mode_obj, t.baseline_start_day, t.baseline_end_day, trade_usage, zone_usage)
                continue

            normal_mode = t.get_mode("NORMAL") or t.available_modes[0]
            chosen_mode = normal_mode
            if budget_fraction > 0.0:
                cand_mode = t.get_mode(crash_mode)
                if cand_mode is not None:
                    c_cost = scenario.effective_cost(t, cand_mode)
                    if spent + c_cost <= budget_limit + 1e-9:
                        chosen_mode = cand_mode

            earliest = trigger_day if (tid in rework_ids or t.baseline_start_day >= trigger_day) else t.baseline_start_day
            for p in t.predecessors:
                if p in assignments:
                    earliest = max(earliest, assignments[p].end_day)
            if t.fim_item_id and t.fim_item_id in fim_arrivals:
                earliest = max(earliest, fim_arrivals[t.fim_item_id])

            start_d, end_d = _find_earliest_feasible_start(
                t, chosen_mode, earliest, scenario, trade_usage, zone_usage
            )
            assignments[tid] = TaskAssignment(
                task_id=tid,
                start_day=start_d,
                end_day=end_d,
                selected_mode=chosen_mode.mode,
            )
            _commit_task(t, chosen_mode, start_d, end_d, trade_usage, zone_usage)
            spent += scenario.effective_cost(t, chosen_mode)

        return RecoveryPlan(plan_label=label, assignments=assignments)

    plans = [
        build_greedy_plan("Gen0_Normal", 0.0, "NORMAL"),
        build_greedy_plan("Gen0_Greedy_OT_50pct", 0.5, "OVERTIME"),
        build_greedy_plan("Gen0_Greedy_DoubleShift_100pct", 1.0, "DOUBLE_SHIFT"),
    ]
    return plans[:max_plans]
# EVOLVE-BLOCK-END
