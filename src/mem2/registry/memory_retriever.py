from mem2.branches.memory_retriever.none import NoneMemoryRetriever
from mem2.branches.memory_retriever.oe_selector import OeSelectorRetriever
from mem2.branches.memory_retriever.ps_selector import PsSelectorRetriever
from mem2.branches.memory_retriever.oe_topk import OeTopKRetriever

MEMORY_RETRIEVERS = {
    "none": NoneMemoryRetriever,
    "oe_topk": OeTopKRetriever,
    "oe_selector": OeSelectorRetriever,
    "ps_selector": PsSelectorRetriever,
    "lesson_topk": OeTopKRetriever,            # legacy alias
    "lesson_topk_v1": OeTopKRetriever,         # legacy alias
    "lesson_selector": OeSelectorRetriever,    # legacy alias
    "arcmemo_selector": OeSelectorRetriever,   # legacy alias
    "arcmemo_selector_v1": OeSelectorRetriever,# legacy alias
    "concept_selector": PsSelectorRetriever,   # legacy alias
}
