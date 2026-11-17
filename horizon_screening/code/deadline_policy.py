"""Execution budgets and certificates; no changes to the energy model."""
import math


def accept_mpc_stage(stage_index, valid, gap, target=.10, strict=False):
    """Stage two accepts feasible plans except during strict quality review."""
    return valid and math.isfinite(gap) and ((stage_index >= 1 and not strict) or gap <= target)


def relative_gap(upper, lower):
    if not math.isfinite(upper) or not math.isfinite(lower):
        return math.inf
    if lower > upper + 1e-7 * max(1., abs(upper)):
        raise ValueError('Lower bound exceeds the objective of the saved feasible plan')
    return abs(upper - min(upper, lower)) / abs(upper) if upper else (0. if lower == 0 else math.inf)


def certificate(original_upper, original_lower, new_lower, target=.10, acceptable=.20):
    """A bound proves quality of the ORIGINAL plan, never of a replacement plan."""
    if not all(math.isfinite(v) for v in (original_upper, original_lower, new_lower)):
        raise ValueError('Nonfinite certificate objective or bound')
    bound = max(original_lower, new_lower)
    gap = relative_gap(original_upper, bound)
    return dict(original_objective=original_upper, best_bound=bound, gap=gap,
                quality=('target_met' if gap <= target else
                         'acceptable_20pct' if gap <= acceptable else 'needs_further_review'),
                actions_changed=False)


def task_priority(task):
    if task.get('review_mode') == 'completion':
        return 0
    if task.get('review_mode') == 'certificate':
        return 1 if task.get('priority_gap_gt20') else 2
    if task.get('review_mode') == 'budget_retry':
        return 2
    if task.get('review'):
        return 3
    return 4
