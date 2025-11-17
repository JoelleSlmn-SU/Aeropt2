# =========================
# Objective Evaluator
# =========================
import math
import re

class ObjectiveEvaluator:
    """
    Computes weighted objective from per-condition metrics.
    Supports:
      - "Drag": sum_i w_i * CD_i
      - "Lift": -sum_i w_i * CL_i  (minimise → maximise lift)
      - "Lift-to-Drag": sum_i w_i * (CD_i / max(CL_i, eps))
      - custom expression with variables CL, CD, CM (safe-eval-like).
    """
    def __init__(self, obj_cfg):
        self.cfg = obj_cfg or {}
        self.expr = self.cfg.get("expression", "Drag")
        self.conds = self.cfg.get("conditions", [])

    def compute(self, per_cond_metrics):
        import math
        eps = 1e-12
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
            env = {"CL": m.get("CL", 0.0), "CD": m.get("CD", 0.0), "CM": m.get("CM", 0.0), "abs": abs, "max": max, "min": min}
            try:
                total += c["Weight"] * float(eval(self.expr, {"__builtins__": {}}, env))
            except Exception:
                total += 1e9  # penalise invalid
        return total
