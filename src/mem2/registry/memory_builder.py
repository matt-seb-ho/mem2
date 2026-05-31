from mem2.branches.memory_builder.accretive_prune import AccretivePruneMemoryBuilder
from mem2.branches.memory_builder.adas_style_search import ADASMetaSearchBuilder
from mem2.branches.memory_builder.alma_style_metaedit import ALMAStyleMetaEditMemoryBuilder
from mem2.branches.memory_builder.arcmemo_oe import ArcMemoOeMemoryBuilder
from mem2.branches.memory_builder.arcmemo_ps import ArcMemoPsMemoryBuilder
from mem2.branches.memory_builder.arcmemo_reorg import ArcMemoReorgMemoryBuilder
from mem2.branches.memory_builder.barc_ingest import BARCIngestMemoryBuilder
from mem2.branches.memory_builder.none import NoneMemoryBuilder
from mem2.branches.memory_builder.reorg_amem import AMEMAgenticMemoryBuilder
from mem2.branches.memory_builder.reorg_dreamcoder import DreamCoderReorgBuilder
from mem2.branches.memory_builder.reorg_evolver import EvolveRDedupBuilder
from mem2.branches.memory_builder.reorg_lilo import LILOLibraryGrowthBuilder
from mem2.branches.memory_builder.reorg_lrll import LRLLWakeSleepBuilder
from mem2.branches.memory_builder.reorg_memp import MempProceduralMemoryBuilder
from mem2.branches.memory_builder.reorg_memtree import MemTreeHierarchicalBuilder
from mem2.branches.memory_builder.reorg_sleepgate import SleepGateForgettingBuilder
from mem2.branches.memory_builder.reorg_stitch import StitchReorgBuilder
from mem2.branches.memory_builder.variant_dspy_opt import DSPyOptFormatBuilder
from mem2.branches.memory_builder.variant_formats import VariantFormatBuilder
from mem2.branches.memory_builder.variant_gepa import GEPAFormatBuilder
from mem2.branches.memory_builder.variant_parse import PARSESchemaBuilder
# COLM 2026 rebuttal — new method ports (added 2026-05-27)
from mem2.branches.memory_builder.ace import AceMemoryBuilder
from mem2.branches.memory_builder.dc import DcMemoryBuilder
from mem2.branches.memory_builder.reasoning_bank import ReasoningBankMemoryBuilder

MEMORY_BUILDERS = {
    "none": NoneMemoryBuilder,
    "arcmemo_oe": ArcMemoOeMemoryBuilder,
    "arcmemo_ps": ArcMemoPsMemoryBuilder,
    "arcmemo_reorg": ArcMemoReorgMemoryBuilder,
    "alma_style_metaedit": ALMAStyleMetaEditMemoryBuilder,
    "adas_style_search": ADASMetaSearchBuilder,
    "accretive_prune": AccretivePruneMemoryBuilder,
    "barc_ingest": BARCIngestMemoryBuilder,
    "reorg_amem": AMEMAgenticMemoryBuilder,
    "reorg_dreamcoder": DreamCoderReorgBuilder,
    "reorg_evolver": EvolveRDedupBuilder,
    "reorg_lilo": LILOLibraryGrowthBuilder,
    "reorg_lrll": LRLLWakeSleepBuilder,
    "reorg_memp": MempProceduralMemoryBuilder,
    "reorg_memtree": MemTreeHierarchicalBuilder,
    "reorg_sleepgate": SleepGateForgettingBuilder,
    "reorg_stitch": StitchReorgBuilder,
    "variant_format": VariantFormatBuilder,
    "variant_dspy_opt": DSPyOptFormatBuilder,
    "variant_gepa": GEPAFormatBuilder,
    "variant_parse": PARSESchemaBuilder,
    "concept_ps": ArcMemoPsMemoryBuilder,      # legacy alias
    "arcmemo_ps_v1": ArcMemoPsMemoryBuilder,   # legacy alias
    # COLM 2026 rebuttal — added 2026-05-27.
    "ace": AceMemoryBuilder,
    "dc": DcMemoryBuilder,
    "reasoning_bank": ReasoningBankMemoryBuilder,
}
