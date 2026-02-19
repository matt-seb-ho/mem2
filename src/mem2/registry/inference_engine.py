from mem2.branches.inference_engine.lcb_solve import LcbSolveInferenceEngine
from mem2.branches.inference_engine.math_ps_solve import MathPsSolveInferenceEngine
from mem2.branches.inference_engine.python_transform_retry import PythonTransformRetryInferenceEngine

INFERENCE_ENGINES = {
    "python_transform_retry": PythonTransformRetryInferenceEngine,
    "python_transform_retry_v1": PythonTransformRetryInferenceEngine,
    "math_ps_solve": MathPsSolveInferenceEngine,
    "lcb_solve": LcbSolveInferenceEngine,
}
