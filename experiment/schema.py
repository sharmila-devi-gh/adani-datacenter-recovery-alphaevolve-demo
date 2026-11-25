"""Immutable domain schema for Data Centre capital-project schedule recovery.

This file is included in `seed_files` WITHOUT EVOLVE-BLOCK markers so AlphaEvolve
has full visibility into data structures while keeping the contract frozen.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

Trade = Literal["CIVIL", "MEP_HVAC", "HV_ELEC", "BMS_CONTROLS", "COMMISSIONING"]
ModeType = Literal["NORMAL", "OVERTIME", "DOUBLE_SHIFT", "ALT_SUBCONTRACTOR"]
WbsStage = Literal["CIVIL", "CORE_SHELL", "MEP_POWER", "COOLING", "L0_L6_CX"]
ScenarioType = Literal[
    "MATERIAL_SLIP_COST_SHOCK",
    "COOLING_SPEC_CHANGE",
    "MANPOWER_DIP",
    "COMPOUND",
]

ALL_TRADES: Tuple[Trade, ...] = (
    "CIVIL",
    "MEP_HVAC",
    "HV_ELEC",
    "BMS_CONTROLS",
    "COMMISSIONING",
)


@dataclass(frozen=True)
class ExecutionMode:
    """One selectable execution mode for a P6 project task."""

    mode: ModeType
    duration_days: int
    incremental_cost_inr_cr: float
    crew_by_trade: Dict[str, int]
    churn_penalty: float  # 0.0 (normal), 0.3 (OT), 0.7 (2x shift), 1.0 (subcon)


@dataclass(frozen=True)
class TaskNode:
    """Single Primavera P6 activity node with multi-mode execution options."""

    task_id: str
    name: str
    wbs_stage: WbsStage
    work_front_zone: str
    predecessors: Tuple[str, ...]
    fim_item_id: Optional[str]
    baseline_start_day: int
    baseline_end_day: int
    baseline_mode: ModeType
    available_modes: Tuple[ExecutionMode, ...]

    def get_mode(self, mode_name: str) -> Optional[ExecutionMode]:
        for m in self.available_modes:
            if m.mode == mode_name:
                return m
        return None


@dataclass(frozen=True)
class DisruptionEvent:
    """Mid-build disruption injected at trigger_day."""

    event_id: str
    scenario_type: ScenarioType
    summary: str
    trigger_day: int
    fim_arrival_overrides: Dict[str, int] = field(default_factory=dict)
    duration_multipliers: Dict[str, float] = field(default_factory=dict)
    cost_shock_multiplier: float = 1.0
    manpower_cap_overrides: Dict[str, Tuple[int, int, int]] = field(default_factory=dict)
    # manpower_cap_overrides maps trade -> (start_day, end_day_exclusive, reduced_cap)
    added_rework_tasks: Tuple[TaskNode, ...] = ()


@dataclass(frozen=True)
class SiteConstraints:
    """Site-level resource, FIM, work-front, budget, and LD constraints."""

    site_name: str
    target_rfs_day: int
    ld_grace_days: int
    ld_rate_inr_cr_per_day: float
    kronos_daily_cap: Dict[str, int]
    fim_base_arrival_day: Dict[str, int]
    zone_max_concurrent: Dict[str, int]
    max_recovery_budget_inr_cr: float


@dataclass(frozen=True)
class Scenario:
    """Complete self-contained synthetic Data Centre recovery scenario."""

    scenario_id: str
    split: Literal["train", "heldout"]
    description: str
    tasks: Tuple[TaskNode, ...]
    constraints: SiteConstraints
    disruption: DisruptionEvent

    def active_tasks(self) -> List[TaskNode]:
        """Return original tasks plus any rework tasks introduced by the disruption."""
        return list(self.tasks) + list(self.disruption.added_rework_tasks)

    def effective_fim_arrivals(self) -> Dict[str, int]:
        merged = dict(self.constraints.fim_base_arrival_day)
        merged.update(self.disruption.fim_arrival_overrides)
        return merged

    def effective_duration(self, task: TaskNode, mode: ExecutionMode) -> int:
        mult = self.disruption.duration_multipliers.get(task.task_id, 1.0)
        return max(1, int(round(mode.duration_days * mult)))

    def effective_cost(self, task: TaskNode, mode: ExecutionMode) -> float:
        if mode.incremental_cost_inr_cr <= 0.0:
            return 0.0
        return round(mode.incremental_cost_inr_cr * self.disruption.cost_shock_multiplier, 4)

    def effective_kronos_cap(self, trade: str, day: int) -> int:
        base_cap = self.constraints.kronos_daily_cap[trade]
        override = self.disruption.manpower_cap_overrides.get(trade)
        if override is not None:
            start_d, end_d, reduced_cap = override
            if start_d <= day < end_d:
                return min(base_cap, reduced_cap)
        return base_cap


@dataclass(frozen=True)
class TaskAssignment:
    """Scheduled start/end and mode for a single task."""

    task_id: str
    start_day: int
    end_day: int
    selected_mode: ModeType


@dataclass(frozen=True)
class RecoveryPlan:
    """Candidate schedule recovery option returned by a solver."""

    plan_label: str
    assignments: Dict[str, TaskAssignment]


@dataclass(frozen=True)
class ConstraintViolation:
    """Single deterministic feasibility violation found by verifier.py."""

    gate: str  # COMPLETENESS | FREEZE | PRECEDENCE | FIM | KRONOS | WORK_FRONT | BUDGET | MODE
    message: str
    task_id: Optional[str] = None
    day: Optional[int] = None


@dataclass(frozen=True)
class PlanMetrics:
    """Computed KPIs for a verified feasible RecoveryPlan."""

    plan_label: str
    feasible: bool
    rfs_day: int
    rfs_slip_days: int
    rfs_days_recovered: int
    incremental_cost_inr_cr: float
    ld_cost_inr_cr: float
    ld_avoided_inr_cr: float
    net_total_cost_inr_cr: float
    mean_churn: float
    violations: Tuple[ConstraintViolation, ...] = ()


def scenario_to_dict(scenario: Scenario) -> Dict[str, Any]:
    """Serialize a Scenario to JSON-safe dict."""
    return asdict(scenario)


def _task_from_dict(d: Dict[str, Any]) -> TaskNode:
    modes = tuple(
        ExecutionMode(
            mode=m["mode"],
            duration_days=int(m["duration_days"]),
            incremental_cost_inr_cr=float(m["incremental_cost_inr_cr"]),
            crew_by_trade={str(k): int(v) for k, v in m["crew_by_trade"].items()},
            churn_penalty=float(m["churn_penalty"]),
        )
        for m in d["available_modes"]
    )
    return TaskNode(
        task_id=str(d["task_id"]),
        name=str(d["name"]),
        wbs_stage=d["wbs_stage"],
        work_front_zone=str(d["work_front_zone"]),
        predecessors=tuple(str(p) for p in d["predecessors"]),
        fim_item_id=d.get("fim_item_id"),
        baseline_start_day=int(d["baseline_start_day"]),
        baseline_end_day=int(d["baseline_end_day"]),
        baseline_mode=d["baseline_mode"],
        available_modes=modes,
    )


def scenario_from_dict(d: Dict[str, Any]) -> Scenario:
    """Deserialize a Scenario from JSON dict."""
    c = d["constraints"]
    constraints = SiteConstraints(
        site_name=str(c["site_name"]),
        target_rfs_day=int(c["target_rfs_day"]),
        ld_grace_days=int(c["ld_grace_days"]),
        ld_rate_inr_cr_per_day=float(c["ld_rate_inr_cr_per_day"]),
        kronos_daily_cap={str(k): int(v) for k, v in c["kronos_daily_cap"].items()},
        fim_base_arrival_day={str(k): int(v) for k, v in c["fim_base_arrival_day"].items()},
        zone_max_concurrent={str(k): int(v) for k, v in c["zone_max_concurrent"].items()},
        max_recovery_budget_inr_cr=float(c["max_recovery_budget_inr_cr"]),
    )
    dis = d["disruption"]
    raw_mp = dis.get("manpower_cap_overrides", {})
    mp_overrides = {
        str(k): (int(v[0]), int(v[1]), int(v[2])) for k, v in raw_mp.items()
    }
    disruption = DisruptionEvent(
        event_id=str(dis["event_id"]),
        scenario_type=dis["scenario_type"],
        summary=str(dis["summary"]),
        trigger_day=int(dis["trigger_day"]),
        fim_arrival_overrides={
            str(k): int(v) for k, v in dis.get("fim_arrival_overrides", {}).items()
        },
        duration_multipliers={
            str(k): float(v) for k, v in dis.get("duration_multipliers", {}).items()
        },
        cost_shock_multiplier=float(dis.get("cost_shock_multiplier", 1.0)),
        manpower_cap_overrides=mp_overrides,
        added_rework_tasks=tuple(
            _task_from_dict(t) for t in dis.get("added_rework_tasks", [])
        ),
    )
    return Scenario(
        scenario_id=str(d["scenario_id"]),
        split=d["split"],
        description=str(d["description"]),
        tasks=tuple(_task_from_dict(t) for t in d["tasks"]),
        constraints=constraints,
        disruption=disruption,
    )
