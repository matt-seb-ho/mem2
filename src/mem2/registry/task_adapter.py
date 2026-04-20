from mem2.branches.task_adapter.arc_grid import ArcGridTaskAdapter
from mem2.branches.task_adapter.arc3 import Arc3TaskAdapter
from mem2.branches.task_adapter.livecodebench import LiveCodeBenchTaskAdapter
from mem2.branches.task_adapter.math_ps import MathPsTaskAdapter

TASK_ADAPTERS = {
    "arc_grid": ArcGridTaskAdapter,
    "arc_grid_v1": ArcGridTaskAdapter,
    "arc3": Arc3TaskAdapter,
    "livecodebench": LiveCodeBenchTaskAdapter,
    "math_ps": MathPsTaskAdapter,
}
