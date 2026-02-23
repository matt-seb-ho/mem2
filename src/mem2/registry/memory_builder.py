from mem2.branches.memory_builder.arcmemo_oe import ArcMemoOeMemoryBuilder
from mem2.branches.memory_builder.arcmemo_ps import ArcMemoPsMemoryBuilder
from mem2.branches.memory_builder.none import NoneMemoryBuilder

MEMORY_BUILDERS = {
    "none": NoneMemoryBuilder,
    "arcmemo_oe": ArcMemoOeMemoryBuilder,
    "arcmemo_ps": ArcMemoPsMemoryBuilder,
    "concept_ps": ArcMemoPsMemoryBuilder,      # legacy alias
    "arcmemo_ps_v1": ArcMemoPsMemoryBuilder,   # legacy alias
}
