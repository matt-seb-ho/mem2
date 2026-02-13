from mem2.branches.task_adapter.arc_grid import ArcGridTaskAdapter
from mem2.branches.task_adapter.math_ps import MathPsTaskAdapter

TASK_ADAPTERS = {
    "arc_grid": ArcGridTaskAdapter,
    "arc_grid_v1": ArcGridTaskAdapter,
    "math_ps": MathPsTaskAdapter,
}
