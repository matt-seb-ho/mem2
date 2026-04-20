from mem2.branches.memory_builder.alma_style_metaedit import ALMAStyleMetaEditMemoryBuilder
from mem2.branches.memory_builder.arcmemo_oe import ArcMemoOeMemoryBuilder
from mem2.branches.memory_builder.arcmemo_ps import ArcMemoPsMemoryBuilder
from mem2.branches.memory_builder.arcmemo_reorg import ArcMemoReorgMemoryBuilder
from mem2.branches.memory_builder.barc_ingest import BARCIngestMemoryBuilder
from mem2.branches.memory_builder.none import NoneMemoryBuilder
from mem2.branches.memory_builder.variant_formats import VariantFormatBuilder

MEMORY_BUILDERS = {
    "none": NoneMemoryBuilder,
    "arcmemo_oe": ArcMemoOeMemoryBuilder,
    "arcmemo_ps": ArcMemoPsMemoryBuilder,
    "arcmemo_reorg": ArcMemoReorgMemoryBuilder,
    "alma_style_metaedit": ALMAStyleMetaEditMemoryBuilder,
    "barc_ingest": BARCIngestMemoryBuilder,
    "variant_format": VariantFormatBuilder,
    "concept_ps": ArcMemoPsMemoryBuilder,      # legacy alias
    "arcmemo_ps_v1": ArcMemoPsMemoryBuilder,   # legacy alias
}
