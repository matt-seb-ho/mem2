from mem2.branches.benchmark.arc_agi import ArcAgiBenchmarkAdapter
from mem2.branches.benchmark.competition_math_ps import CompetitionMathPsBenchmarkAdapter
from mem2.branches.benchmark.livecodebench import LiveCodeBenchAdapter
from mem2.branches.task_adapter.arc3 import Arc3SdkBenchmark

BENCHMARKS = {
    "arc_agi": ArcAgiBenchmarkAdapter,
    "arc_agi_v1": ArcAgiBenchmarkAdapter,
    "arc3_sdk": Arc3SdkBenchmark,
    "competition_math_ps": CompetitionMathPsBenchmarkAdapter,
    "livecodebench": LiveCodeBenchAdapter,
}
