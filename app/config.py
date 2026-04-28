import os
import logging
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

load_dotenv()
logger = logging.getLogger(__name__)

# ====================== 基础配置 ======================
DB_URL = os.getenv("DB_URL", "postgresql://postgres:postgres123@localhost:5432/spatial_kb")
API_KEY = os.getenv("API_KEY", "test-key")

# ====================== LLM（工厂模式，支持多种供应商） ======================
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL = os.getenv("MODEL_NAME", "gpt-4o")

EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", "d:/work/text2sql2/models/bge-m3")

from app.llm_provider import create_llm

llm = create_llm()

# ====================== 统一 BAAI/bge-m3（只加载一次） ======================
class BGEEmbeddings(Embeddings):
    """自定义 LangChain 兼容的 BGE Embedding（避免重复加载模型）"""

    def __init__(self, model_path: str = EMBEDDING_MODEL_PATH, device: str = "cpu"):
        self.model = SentenceTransformer(model_path, device=device)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

# 全局共享的 embedding 对象
logger.info("正在加载 Embedding 模型: %s ...", EMBEDDING_MODEL_PATH)
try:
    bge_embeddings = BGEEmbeddings(model_path=EMBEDDING_MODEL_PATH, device="cpu")
    bge_embedding_model = bge_embeddings.model
    logger.info("Embedding 模型加载完成")
except Exception as e:
    logger.error("Embedding 模型加载失败: %s", e, exc_info=True)
    raise

logger.info("配置加载完成 | Provider: %s | Model: %s | Embedding: %s", LLM_PROVIDER, LLM_MODEL, EMBEDDING_MODEL_PATH)
