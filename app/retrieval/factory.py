"""按检索模式创建检索器。

问答编排只接收 Retriever，不在这里判断用向量、关键词还是混合。
默认仍是 vector。bm25、hybrid、hybrid_rerank 只在明确指定时创建。
hybrid_rerank 只接受已配置的重排服务，不接受 MockReranker。
"""

from __future__ import annotations

from app.core.exceptions import ConfigurationError
from app.embeddings.base import EmbeddingProvider
from app.retrieval.base import Retriever
from app.retrieval.reranker import Reranker, require_production_reranker
from app.retrieval.tokenizer import Tokenizer
from app.retrieval.vector_retriever import VectorRetriever
from app.vectorstore.base import VectorStore

VECTOR_MODE = "vector"
BM25_MODE = "bm25"
HYBRID_MODE = "hybrid"
HYBRID_RERANK_MODE = "hybrid_rerank"
_SUPPORTED = f"{VECTOR_MODE}、{BM25_MODE}、{HYBRID_MODE}、{HYBRID_RERANK_MODE}"


class RetrieverFactory:
    def __init__(
        self,
        embedder: EmbeddingProvider,
        store: VectorStore,
        *,
        max_distance: float | None = None,
        tokenizer: Tokenizer | None = None,
        hybrid_vector_top_k: int = 20,
        hybrid_bm25_top_k: int = 20,
        hybrid_final_top_k: int = 10,
        rrf_k: int = 60,
        reranker: Reranker | None = None,
        reranker_candidate_top_k: int = 20,
        reranker_final_top_k: int = 5,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._max_distance = max_distance
        self._tokenizer = tokenizer
        self._hybrid_vector_top_k = hybrid_vector_top_k
        self._hybrid_bm25_top_k = hybrid_bm25_top_k
        self._hybrid_final_top_k = hybrid_final_top_k
        self._rrf_k = rrf_k
        self._reranker = reranker
        self._reranker_candidate_top_k = reranker_candidate_top_k
        self._reranker_final_top_k = reranker_final_top_k

    def create(self, mode: str = VECTOR_MODE) -> Retriever:
        selected = mode.strip()
        if selected == VECTOR_MODE:
            return VectorRetriever(
                self._embedder,
                self._store,
                max_distance=self._max_distance,
            )
        if selected == BM25_MODE:
            from app.retrieval.bm25_retriever import BM25Retriever

            return BM25Retriever(self._store, self._tokenizer_or_default())
        if selected == HYBRID_MODE:
            return self._create_hybrid()
        if selected == HYBRID_RERANK_MODE:
            from app.retrieval.reranker import RerankingRetriever

            reranker = require_production_reranker(self._reranker)
            return RerankingRetriever(
                self._create_hybrid(),
                reranker,
                candidate_top_k=self._reranker_candidate_top_k,
                final_top_k=self._reranker_final_top_k,
            )
        raise ConfigurationError(f"未知检索模式：{mode}。当前只支持 {_SUPPORTED}。")

    def _create_hybrid(self) -> Retriever:
        from app.retrieval.bm25_retriever import BM25Retriever
        from app.retrieval.hybrid_retriever import HybridRetriever

        return HybridRetriever(
            VectorRetriever(
                self._embedder,
                self._store,
                max_distance=self._max_distance,
            ),
            BM25Retriever(self._store, self._tokenizer_or_default()),
            vector_top_k=self._hybrid_vector_top_k,
            bm25_top_k=self._hybrid_bm25_top_k,
            final_top_k=self._hybrid_final_top_k,
            rrf_k=self._rrf_k,
        )

    def _tokenizer_or_default(self) -> Tokenizer:
        if self._tokenizer is not None:
            return self._tokenizer
        from app.retrieval.tokenizer import JiebaTokenizer

        return JiebaTokenizer()
