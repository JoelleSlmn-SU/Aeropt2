# =========================
# Objective Evaluator
# =========================
import math
import re

_EPS = 1e-12


def _safe_eval(expr, metrics):
    """
    Evaluate `expr` (a string) against a per-condition metrics dict in a
    restricted namespace (no builtins). `metrics` keys are used directly as
    identifiers -- this is what makes generated symbols like "PR_s111" or
    "DC60_s111" (produced by the objective/monitor editor's surface-aware
    terms, see ObjectiveEditor._make_symbol in Aeropt.py) usable in
    expressions without ObjectiveEvaluator needing to know their names in
    advance. CL/CD/CM always default to 0.0 even if absent, matching the
    previous hardcoded behaviour, so existing Drag/Lift/custom-CL-CD-CM
    configs keep working unchanged.
    """
    env = {"CL": 0.0, "CD": 0.0, "CM": 0.0}
    env.update(metrics or {})
    env.update({"abs": abs, "max": max, "min": min})
    return float(eval(expr, {"__builtins__": {}}, env))


class ObjectiveEvaluator:
    """
    Computes weighted objective from per-condition metrics.
    Supports:
      - "Drag": sum_i w_i * CD_i
      - "Lift": -sum_i w_i * CL_i  (minimise → maximise lift)
      - "Lift-to-Drag": sum_i w_i * (CD_i / max(CL_i, eps))
      - custom expression using ANY key present in the per-condition metrics
        dict (CL, CD, CM, or generated monitor symbols like PR_s111,
        DC60_s111, ...). Since BO here always minimises, maximise a metric
        by giving it a negative coefficient, e.g. expression="-PR_s111"
        (this matches the convention already used by the GUI's Lift/
        Lift-to-Drag presets and its help text).
    """
    def __init__(self, obj_cfg):
        self.cfg = obj_cfg or {}
        self.expr = self.cfg.get("expression", "Drag")
        self.conds = self.cfg.get("conditions", [])

    def compute(self, per_cond_metrics):
        eps = _EPS
        if self.expr.lower() == "drag":
            return sum(c["Weight"] * m.get("CD", 0.0) for c, m in zip(self.conds, per_cond_metrics))
        if self.expr.lower() == "lift":
            return -sum(c["Weight"] * m.get("CL", 0.0) for c, m in zip(self.conds, per_cond_metrics))
        if "lift-to-drag" in self.expr.lower():
            return sum(c["Weight"] * (m.get("CD", 0.0) / max(m.get("CL", eps), eps))
                       for c, m in zip(self.conds, per_cond_metrics))
        # Custom expression (very light sandbox)
        total = 0.0
        for c, m in zip(self.conds, per_cond_metrics):
            try:
                total += c["Weight"] * _safe_eval(self.expr, m)
            except Exception:
                total += 1e9  # penalise invalid
        return total


class ConstraintSet:
    """
    Parses the free-text constraint lines produced by the GUI's
    ObjectiveEditor ("Constraints (optional, one per line; e.g.
    'CL >= 0.3', 'CD <= 0.02')" -> objective_config["constraints"]) and
    evaluates them against per-condition metrics for ONE design point.

    NOTE: as of this change these strings are only consumed here -- nothing
    in the original codebase (ObjectiveEvaluator/BayesianOptimiser) read
    objective_config["constraints"] before. Wiring is:
      Aeropt.py (run_optimisation) -> bo_settings.json["constraints"]
      -> BayesianOptimiser.__init__ -> constraint GP(s)
      -> whatever builds `eval_func` (not in this file) must call
         ConstraintSet(...).evaluate(per_cond_metrics) and return the
         result as the second element of eval_func's return value.

    Each line may be any expression the objective supports (same _safe_eval
    sandbox, so it can reference generated symbols like DC60_s111), followed
    by "<=" or ">=" and a numeric limit, e.g. "DC60_s111 <= 0.30".

    Multiple flow conditions: unlike the objective (a weighted SUM across
    conditions, appropriate for performance metrics you want on average),
    a constraint is aggregated as the WORST CASE across conditions --
    max(values) for "<=" constraints, min(values) for ">=" constraints --
    because a physical tolerance (e.g. distortion an engine can swallow)
    has to hold in the worst flight condition it sees, not on average.
    If you actually want a per-condition or averaged constraint instead,
    say so explicitly; worst-case is the conservative default here.
    """
    _PATTERN = re.compile(r"^(.*?)\s*(<=|>=)\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$")

    def __init__(self, constraint_lines):
        self.constraints = []  # list of {"label", "expr", "sense", "limit"}
        for raw in (constraint_lines or []):
            raw = raw.strip()
            if not raw:
                continue
            match = self._PATTERN.match(raw)
            if not match:
                raise ValueError(
                    f"Could not parse constraint line: {raw!r}. "
                    f"Expected '<expression> <= <number>' or '<expression> >= <number>'."
                )
            expr, sense, limit = match.group(1).strip(), match.group(2), float(match.group(3))
            self.constraints.append({"label": expr, "expr": expr, "sense": sense, "limit": limit})

    def as_settings_list(self):
        """The structured form BayesianOptimiser expects in settings['constraints']."""
        return [{"metric": c["label"], "limit": c["limit"], "sense": c["sense"]} for c in self.constraints]

    def evaluate(self, per_cond_metrics):
        """
        Returns {label: worst_case_value} for ONE design point, where
        per_cond_metrics is the same list-of-dicts (one per flow condition)
        that ObjectiveEvaluator.compute() consumes.
        """
        out = {}
        for c in self.constraints:
            values = []
            for m in per_cond_metrics:
                try:
                    values.append(_safe_eval(c["expr"], m))
                except Exception:
                    # Missing/invalid metric for this condition: treat as a
                    # constraint violation rather than silently dropping it,
                    # so a broken monitor can't masquerade as "feasible".
                    values.append(float("inf") if c["sense"] == "<=" else float("-inf"))
            if not values:
                continue
            out[c["label"]] = max(values) if c["sense"] == "<=" else min(values)
        return out