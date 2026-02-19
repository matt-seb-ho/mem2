from mem2.branches.evaluator.arc_exec import ArcExecutionEvaluator
from mem2.branches.evaluator.lcb_exec import LcbExecutionEvaluator
from mem2.branches.evaluator.math_ps_exec import MathPsExecutionEvaluator

EVALUATORS = {
    "arc_exec": ArcExecutionEvaluator,
    "arc_exec_v1": ArcExecutionEvaluator,
    "lcb_exec": LcbExecutionEvaluator,
    "math_ps_exec": MathPsExecutionEvaluator,
}
