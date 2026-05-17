from mem2.branches.memory_retriever.colbert_rerank import ColBERTRerankRetriever
from mem2.branches.memory_retriever.graph_traversal import GraphTraversalRetriever
from mem2.branches.memory_retriever.graphrag import GraphRAGRetriever
from mem2.branches.memory_retriever.hipporag2 import HippoRAG2FilterRetriever
from mem2.branches.memory_retriever.hmem_hierarchical import HMEMHierarchicalRetriever
from mem2.branches.memory_retriever.hipporag_ppr import HippoRAGPPRRetriever
from mem2.branches.memory_retriever.lightrag import LightRAGRetriever
from mem2.branches.memory_retriever.magma import MAGMAMultiGraphRetriever
from mem2.branches.memory_retriever.mediq_policy import MediQPolicyRetriever
from mem2.branches.memory_retriever.pathrag import PathRAGRetriever
from mem2.branches.memory_retriever.none import NoneMemoryRetriever
from mem2.branches.memory_retriever.oe_selector import OeSelectorRetriever
from mem2.branches.memory_retriever.ps_selector import PsSelectorRetriever
from mem2.branches.memory_retriever.ps_topk import PsTopKRetriever
from mem2.branches.memory_retriever.ps_topk_query import PsTopKQueryRetriever
from mem2.branches.memory_retriever.oe_topk import OeTopKRetriever
from mem2.branches.memory_retriever.raptor import RAPTORRetriever
from mem2.branches.memory_retriever.rrmc_interactive import RRMCInteractiveRetriever
from mem2.branches.memory_retriever.uot_entropy import UoTEntropyRetriever

MEMORY_RETRIEVERS = {
    "none": NoneMemoryRetriever,
    "oe_topk": OeTopKRetriever,
    "oe_selector": OeSelectorRetriever,
    "ps_selector": PsSelectorRetriever,
    "ps_topk": PsTopKRetriever,
    "ps_topk_query": PsTopKQueryRetriever,
    "colbert_rerank": ColBERTRerankRetriever,
    "graph_traversal": GraphTraversalRetriever,
    "graphrag": GraphRAGRetriever,
    "hipporag_ppr": HippoRAGPPRRetriever,
    "hipporag2_filter": HippoRAG2FilterRetriever,
    "hmem_hierarchical": HMEMHierarchicalRetriever,
    "lightrag": LightRAGRetriever,
    "magma_multigraph": MAGMAMultiGraphRetriever,
    "mediq_policy": MediQPolicyRetriever,
    "pathrag": PathRAGRetriever,
    "raptor": RAPTORRetriever,
    "rrmc_interactive": RRMCInteractiveRetriever,
    "uot_entropy": UoTEntropyRetriever,
    "lesson_topk": OeTopKRetriever,            # legacy alias
    "lesson_topk_v1": OeTopKRetriever,         # legacy alias
    "lesson_selector": OeSelectorRetriever,    # legacy alias
    "arcmemo_selector": OeSelectorRetriever,   # legacy alias
    "arcmemo_selector_v1": OeSelectorRetriever,# legacy alias
    "concept_selector": PsSelectorRetriever,   # legacy alias
}
