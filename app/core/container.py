"""把配置、向量库和模型客户端装进同一个对象，供路由使用。

ENABLE_EXTERNAL_MODELS=false 时不打开向量库，也不创建模型客户端。
这样健康检查和单元测试不需要 API Key。上传和问答会返回明确的配置错误。
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings
from app.core.exceptions import ConfigurationError
from app.embeddings.base import EmbeddingProvider
from app.embeddings.openai_compatible import OpenAICompatibleEmbeddingProvider
from app.ingestion.pipeline import IngestionPipeline
from app.llm.base import LLMProvider
from app.llm.openai_compatible import OpenAICompatibleLLMProvider
from app.rag.context_builder import ContextBuilder
from app.rag.pipeline import RAGPipeline
from app.rag.prompt import PromptBuilder
from app.retrieval.vector_retriever import VectorRetriever
from app.vectorstore.base import VectorStore
from app.vectorstore.chroma_store import ChromaVectorStore

_STORE_HINT = "请设置 ENABLE_EXTERNAL_MODELS=true，并填写 Embedding 配置。"


@dataclass
class AppContainer:
    settings: Settings
    embedder: EmbeddingProvider | None = None
    vector_store: VectorStore | None = None
    llm: LLMProvider | None = None

    def require_store(self) -> VectorStore:
        if self.vector_store is None:
            raise ConfigurationError(f"向量库未打开。{_STORE_HINT}")
        return self.vector_store

    def require_ingestion(self) -> IngestionPipeline:
        store = self.require_store()
        if self.embedder is None:
            raise ConfigurationError(f"导入文档需要 Embedding。{_STORE_HINT}")
        return IngestionPipeline(
            self.settings,
            embedder=self.embedder,
            vector_store=store,
        )

    def require_rag(self) -> RAGPipeline:
        store = self.require_store()
        if self.embedder is None or self.llm is None:
            raise ConfigurationError(
                "问答需要同时填写 Embedding 和 LLM 配置。"
                "请设置 ENABLE_EXTERNAL_MODELS=true，并分别填写两套地址、密钥和模型名。"
            )
        return RAGPipeline(
            retriever=VectorRetriever(
                self.embedder,
                store,
                max_distance=self.settings.retrieval_max_distance,
            ),
            context_builder=ContextBuilder(self.settings.max_context_chars),
            prompt_builder=PromptBuilder(),
            llm=self.llm,
            default_top_k=self.settings.default_top_k,
            max_top_k=self.settings.max_top_k,
            max_question_chars=self.settings.max_question_chars,
        )

    def close(self) -> None:
        if self.vector_store is not None:
            self.vector_store.close()
            self.vector_store = None
        for client in (self.embedder, self.llm):
            close = getattr(client, "close", None)
            if close is not None:
                close()
        self.embedder = None
        self.llm = None


def build_container(settings: Settings) -> AppContainer:
    """按配置创建客户端。未启用外部模型时返回空容器。"""
    if not settings.enable_external_models:
        return AppContainer(settings=settings)
    if settings.embedding_dimension is None:
        raise ConfigurationError("缺少 EMBEDDING_DIMENSION")
    embedder = OpenAICompatibleEmbeddingProvider.from_settings(settings)
    store: ChromaVectorStore | None = None
    try:
        store = ChromaVectorStore(
            persist_dir=settings.chroma_path,
            collection_name=settings.chroma_collection,
            embedding_model=settings.embedding_model,
            embedding_dimension=settings.embedding_dimension,
        )
        llm = OpenAICompatibleLLMProvider.from_settings(settings)
    except Exception:
        if store is not None:
            store.close()
        embedder.close()
        raise
    return AppContainer(
        settings=settings,
        embedder=embedder,
        vector_store=store,
        llm=llm,
    )
