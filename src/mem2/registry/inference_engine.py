from mem2.branches.inference_engine.lcb_solve import LcbSolveInferenceEngine
from mem2.branches.inference_engine.gepa_hsea import GepaHseaInferenceEngine
from mem2.branches.inference_engine.math_ps_solve import MathPsSolveInferenceEngine
from mem2.branches.inference_engine.math_reason import MathReasonInferenceEngine
from mem2.branches.inference_engine.python_transform_retry import PythonTransformRetryInferenceEngine

INFERENCE_ENGINES = {
    "gepa_hsea": GepaHseaInferenceEngine,
    "python_transform_retry": PythonTransformRetryInferenceEngine,
    "python_transform_retry_v1": PythonTransformRetryInferenceEngine,
    "math_ps_solve": MathPsSolveInferenceEngine,
    "math_reason": MathReasonInferenceEngine,
    "lcb_solve": LcbSolveInferenceEngine,
}
