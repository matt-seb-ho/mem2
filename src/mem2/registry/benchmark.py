from mem2.branches.benchmark.arc_agi import ArcAgiBenchmarkAdapter
from mem2.branches.benchmark.competition_math_ps import CompetitionMathPsBenchmarkAdapter
from mem2.branches.benchmark.livecodebench import LiveCodeBenchAdapter

BENCHMARKS = {
    "arc_agi": ArcAgiBenchmarkAdapter,
    "arc_agi_v1": ArcAgiBenchmarkAdapter,
    "competition_math_ps": CompetitionMathPsBenchmarkAdapter,
    "livecodebench": LiveCodeBenchAdapter,
}
