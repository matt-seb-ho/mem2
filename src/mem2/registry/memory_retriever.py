from mem2.branches.memory_retriever.arcmemo_selector import ArcMemoStyleSelectorRetriever
from mem2.branches.memory_retriever.concept_selector import ConceptSelectorRetriever
from mem2.branches.memory_retriever.lesson_topk import LessonTopKRetriever

MEMORY_RETRIEVERS = {
    "arcmemo_selector": ArcMemoStyleSelectorRetriever,
    "arcmemo_selector_v1": ArcMemoStyleSelectorRetriever,
    "lesson_topk": LessonTopKRetriever,
    "lesson_topk_v1": LessonTopKRetriever,
    "concept_selector": ConceptSelectorRetriever,
}
