"""Seed recovery solver for AlphaEvolve optimization.

AlphaEvolve mutates ONLY the top-level function inside the marked mutable region
below while reusing the immutable helpers above it.

Available schema attributes (imported from `schema.py`):
  scenario.active_tasks() -> List[TaskNode]
  scenario.effective_fim_arrivals() -> Dict[str, int]
  scenario.effective_duration(task, mode) -> int
  scenario.effective_cost(task, mode) -> float
  scenario.effective_kronos_cap(trade, day) -> int
  scenario.constraints: .target_rfs_day, .ld_grace_days, .ld_rate_inr_cr_per_day,
      .kronos_daily_cap, .zone_max_concurrent, .max_recovery_budget_inr_cr
  scenario.disruption: .trigger_day, .added_rework_tasks
  task: .task_id, .wbs_stage, .work_front_zone, .predecessors, .fim_item_id,
      .baseline_start_day, .baseline_end_day, .baseline_mode, .available_modes, .get_mode(name)
  mode: .mode ('NORMAL'|'OVERTIME'|'DOUBLE_SHIFT'|'ALT_SUBCONTRACTOR'),
      .duration_days, .incremental_cost_inr_cr, .crew_by_trade, .churn_penalty
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


def _topological_order(scenario: Scenario) -> List[str]:
    """Deterministic baseline topological order of all active tasks."""
    tasks = scenario.active_tasks()
    task_map = {t.task_id: t for t in tasks}
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
    return order


# EVOLVE-BLOCK-START
def generate_recovery_plans(scenario: Scenario, max_plans: int = 5) -> List[RecoveryPlan]:
    """Return 3 to `max_plans` strictly feasible, Pareto-diverse recovery plans."""
    tasks = scenario.active_tasks()
    task_map: Dict[str, TaskNode] = {t.task_id: t for t in tasks}
    rework_ids = {t.task_id for t in scenario.disruption.added_rework_tasks}
    fim_arrivals = scenario.effective_fim_arrivals()
    trigger_day = scenario.disruption.trigger_day
    max_budget = scenario.constraints.max_recovery_budget_inr_cr
    topo_order = _topological_order(scenario)

    def build_plan(label: str, budget_fraction: float, crash_mode: str) -> RecoveryPlan:
        trade_usage: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        zone_usage: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        assignments: Dict[str, TaskAssignment] = {}
        spent = 0.0
        budget_limit = max_budget * budget_fraction

        for tid in topo_order:
            t = task_map[tid]
            is_frozen = (t.baseline_end_day <= trigger_day) and (tid not in rework_ids)
            if is_frozen:
                m = t.get_mode(t.baseline_mode) or t.available_modes[0]
                assignments[tid] = TaskAssignment(tid, t.baseline_start_day, t.baseline_end_day, m.mode)
                _commit_task(t, m, t.baseline_start_day, t.baseline_end_day, trade_usage, zone_usage)
                continue

            chosen = t.get_mode("NORMAL") or t.available_modes[0]
            if budget_fraction > 0.0:
                cand = t.get_mode(crash_mode)
                if cand is not None and spent + scenario.effective_cost(t, cand) <= budget_limit + 1e-9:
                    chosen = cand

            earliest = trigger_day if (tid in rework_ids or t.baseline_start_day >= trigger_day) else t.baseline_start_day
            for p in t.predecessors:
                if p in assignments:
                    earliest = max(earliest, assignments[p].end_day)
            if t.fim_item_id and t.fim_item_id in fim_arrivals:
                earliest = max(earliest, fim_arrivals[t.fim_item_id])

            start_d, end_d = _find_earliest_feasible_start(t, chosen, earliest, scenario, trade_usage, zone_usage)
            assignments[tid] = TaskAssignment(tid, start_d, end_d, chosen.mode)
            _commit_task(t, chosen, start_d, end_d, trade_usage, zone_usage)
            spent += scenario.effective_cost(t, chosen)

        return RecoveryPlan(plan_label=label, assignments=assignments)

    plans = [
        build_plan("Gen0_Normal", 0.0, "NORMAL"),
        build_plan("Gen0_Greedy_OT_50pct", 0.5, "OVERTIME"),
        build_plan("Gen0_Greedy_DoubleShift_100pct", 1.0, "DOUBLE_SHIFT"),
    ]
    return plans[:max_plans]
# EVOLVE-BLOCK-END
