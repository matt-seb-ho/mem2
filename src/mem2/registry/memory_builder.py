from mem2.branches.memory_builder.arcmemo_ps import ArcMemoPsMemoryBuilder
from mem2.branches.memory_builder.concept_ps import ConceptPsMemoryBuilder

MEMORY_BUILDERS = {
    "arcmemo_ps": ArcMemoPsMemoryBuilder,
    "arcmemo_ps_v1": ArcMemoPsMemoryBuilder,
    "concept_ps": ConceptPsMemoryBuilder,
}
