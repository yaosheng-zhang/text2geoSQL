"""
LLM Provider 工厂模块

通过 LLM_PROVIDER 环境变量选择 AI 供应商，统一返回 LangChain BaseChatModel 实例。

支持供应商:
  - openai        : OpenAI 官方 / 兼容 OpenAI 协议的中转站（默认）
  - azure_openai  : Azure OpenAI
  - deepseek      : DeepSeek（兼容 OpenAI 协议）
  - zhipu         : 智谱 AI (GLM-4 等)
  - ollama        : Ollama 本地推理
  - vllm          : vLLM 本地推理（兼容 OpenAI 协议）
  - anthropic     : Anthropic Claude
  - dashscope     : 阿里云百炼 / 通义千问

用法:
    from app.llm_provider import create_llm
    llm = create_llm()           # 自动读取环境变量
    llm = create_llm(provider="ollama", model="qwen2.5:7b")
"""

from __future__ import annotations

import os
import logging
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)


# ====================== 供应商实现 ======================

def _create_openai(
    model: str,
    api_key: Optional[str],
    base_url: Optional[str],
    temperature: float,
    **kwargs,
) -> BaseChatModel:
    """OpenAI 官方 / 兼容 OpenAI 协议的中转站（OneAPI、硅基流动等）。"""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url or "https://api.openai.com/v1",
        temperature=temperature,
        **kwargs,
    )


def _create_azure_openai(
    model: str,
    api_key: Optional[str],
    base_url: Optional[str],
    temperature: float,
    **kwargs,
) -> BaseChatModel:
    """Azure OpenAI 服务。"""
    from langchain_openai import AzureChatOpenAI

    azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", model)
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")

    return AzureChatOpenAI(
        azure_deployment=azure_deployment,
        api_key=api_key,
        azure_endpoint=base_url,
        api_version=api_version,
        temperature=temperature,
        **kwargs,
    )


def _create_deepseek(
    model: str,
    api_key: Optional[str],
    base_url: Optional[str],
    temperature: float,
    **kwargs,
) -> BaseChatModel:
    """DeepSeek（兼容 OpenAI 协议）。"""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model or "deepseek-chat",
        api_key=api_key,
        base_url=base_url or "https://api.deepseek.com/v1",
        temperature=temperature,
        **kwargs,
    )


def _create_zhipu(
    model: str,
    api_key: Optional[str],
    base_url: Optional[str],
    temperature: float,
    **kwargs,
) -> BaseChatModel:
    """智谱 AI (GLM-4 等)。langchain-community 内置。"""
    try:
        from langchain_community.chat_models import ChatZhipuAI
    except ImportError:
        raise ImportError(
            "智谱 AI 需要安装依赖: pip install langchain-community zhipuai"
        )

    return ChatZhipuAI(
        model=model or "glm-4",
        api_key=api_key,
        temperature=temperature,
        **kwargs,
    )


def _create_ollama(
    model: str,
    api_key: Optional[str],
    base_url: Optional[str],
    temperature: float,
    **kwargs,
) -> BaseChatModel:
    """Ollama 本地推理。"""
    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        raise ImportError(
            "Ollama 需要安装依赖: pip install langchain-ollama"
        )

    return ChatOllama(
        model=model or "qwen2.5:7b",
        base_url=base_url or "http://localhost:11434",
        temperature=temperature,
        
        **kwargs,
    )


def _create_vllm(
    model: str,
    api_key: Optional[str],
    base_url: Optional[str],
    temperature: float,
    **kwargs,
) -> BaseChatModel:
    """vLLM 本地推理（兼容 OpenAI 协议，走 ChatOpenAI）。"""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model,
        api_key=api_key or "EMPTY",
        base_url=base_url or "http://localhost:8000/v1",
        temperature=temperature,
        **kwargs,
    )


def _create_anthropic(
    model: str,
    api_key: Optional[str],
    base_url: Optional[str],
    temperature: float,
    **kwargs,
) -> BaseChatModel:
    """Anthropic Claude。"""
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError:
        raise ImportError(
            "Anthropic 需要安装依赖: pip install langchain-anthropic"
        )

    return ChatAnthropic(
        model=model or "claude-sonnet-4-20250514",
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        **kwargs,
    )


def _create_dashscope(
    model: str,
    api_key: Optional[str],
    base_url: Optional[str],
    temperature: float,
    **kwargs,
) -> BaseChatModel:
    """阿里云百炼 / 通义千问（兼容 OpenAI 协议）。"""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model or "qwen-plus",
        api_key=api_key,
        base_url=base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=temperature,
        **kwargs,
    )


# ====================== 工厂注册表 ======================

_PROVIDER_REGISTRY: dict[str, callable] = {
    "openai": _create_openai,
    "azure_openai": _create_azure_openai,
    "deepseek": _create_deepseek,
    "zhipu": _create_zhipu,
    "ollama": _create_ollama,
    "vllm": _create_vllm,
    "anthropic": _create_anthropic,
    "dashscope": _create_dashscope,
}


def register_provider(name: str, factory_fn: callable):
    """注册自定义供应商（供外部插件扩展用）。

    Args:
        name: 供应商标识符（小写）
        factory_fn: 工厂函数，签名为 (model, api_key, base_url, temperature, **kwargs) -> BaseChatModel
    """
    _PROVIDER_REGISTRY[name.lower()] = factory_fn
    logger.info("已注册自定义 LLM 供应商: %s", name)


# ====================== 工厂入口 ======================

def create_llm(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: Optional[float] = None,
    **kwargs,
) -> BaseChatModel:
    """创建 LLM 实例（工厂入口）。

    优先级：显式参数 > 环境变量 > 默认值。

    环境变量:
        LLM_PROVIDER   : 供应商标识（openai / deepseek / ollama / vllm / ...）
        MODEL_NAME     : 模型名称
        LLM_API_KEY    : API Key（未设置则回退到 OPENAI_API_KEY）
        LLM_BASE_URL   : API 基础地址（未设置则回退到 OPENAI_BASE_URL）
        LLM_TEMPERATURE: 温度参数
    """
    _provider = (provider or os.getenv("LLM_PROVIDER", "openai")).lower().strip()
    _model = model or os.getenv("MODEL_NAME", "gpt-4o")
    _api_key = api_key or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    _base_url = base_url or os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    _temperature = temperature if temperature is not None else float(
        os.getenv("LLM_TEMPERATURE", "0.0")
    )

    factory_fn = _PROVIDER_REGISTRY.get(_provider)
    if factory_fn is None:
        available = ", ".join(sorted(_PROVIDER_REGISTRY.keys()))
        raise ValueError(
            f"不支持的 LLM 供应商: '{_provider}'，可选: {available}"
        )

    llm = factory_fn(
        model=_model,
        api_key=_api_key,
        base_url=_base_url,
        temperature=_temperature,
        **kwargs,
    )

    logger.info(
        "LLM 已创建 | provider=%s | model=%s | base_url=%s | temperature=%.2f",
        _provider, _model, _base_url or "(default)", _temperature,
    )
    return llm
