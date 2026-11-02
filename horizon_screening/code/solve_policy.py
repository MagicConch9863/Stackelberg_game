"""Budget policy for one fixed mathematical window; no solver dependencies."""
import math


def merge_candidate(best, result):
    """Keep the smallest incumbent and strongest mutually consistent bound."""
    result=dict(result)
    result['solver_gap']=float(result['gap'])
    result['solver_best_bound']=float(result['best_bound'])
    result['solver_status']=int(result['status'])
    candidate=result if best is None or result['objective']<best['objective'] else dict(best)
    bound=max(result['best_bound'],best['best_bound'] if best else -math.inf)
    upper=float(candidate['objective'])
    tolerance=1e-7*max(1.,abs(upper))
    if bound>upper+tolerance:
        # Retain a valid certificate from the selected solve; do not silently
        # manufacture a zero gap from inconsistent cross-solve bounds.
        candidate['cross_solve_bound_rejected']=True
        bound=float(candidate['solver_best_bound'])
    bound=min(bound,upper)
    candidate['best_bound']=bound
    candidate['gap']=abs(upper-bound)/abs(upper) if upper else (0. if bound==0 else math.inf)
    return candidate


def stop_after(limit,best,cfg):
    if best is None:return False
    if limit==60:return best['gap']<=cfg['target_gap']
    if limit==120:return best['gap']<=cfg['acceptance_gap_after_second']
    # 300 seconds exhausts the quality retry. 600 seconds is only needed
    # when no valid incumbent exists after all preceding attempts.
    return limit>=300


def quality(gap):
    if gap<=.10:return 'gap_le_10'
    if gap<=.20:return 'gap_10_to_20'
    if gap<=.50:return 'review_gap_20_to_50'
    return 'priority_review_gap_above_50'
