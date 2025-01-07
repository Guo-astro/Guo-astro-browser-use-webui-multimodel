# -*- coding: utf-8 -*-
"""
Filename: utils.py

Description:
    Utility functions for loading language models (LLMs) from various providers
    (OpenAI, Anthropic, DeepSeek, Gemini, Ollama, Azure OpenAI, etc.) and for
    encoding image files into Base64 strings.
"""

import os
import base64
import logging
from typing import Optional, Union

# Your custom langchain-like imports
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

logger = logging.getLogger(__name__)


def get_llm_model(provider: str, **kwargs) -> Union[
    ChatOpenAI, AzureChatOpenAI, ChatAnthropic, ChatGoogleGenerativeAI, ChatOllama
]:
    """
    Obtain a language model instance based on the specified provider.

    Parameters
    ----------
    provider : str
        The LLM provider identifier (e.g., 'openai', 'anthropic', etc.).
    **kwargs :
        Arbitrary keyword arguments, including base_url, api_key, model_name,
        temperature, etc.

    Returns
    -------
    Union[ChatOpenAI, AzureChatOpenAI, ChatAnthropic, ChatGoogleGenerativeAI, ChatOllama]
        A configured LLM client from one of the supported providers.

    Raises
    ------
    ValueError
        If an unsupported provider is specified.
    """
    logger.debug("Fetching LLM model for provider: %s with kwargs: %s", provider, kwargs)

    # -------------------------
    # Anthropic
    # -------------------------
    if provider == 'anthropic':
        base_url = kwargs.get("base_url") or os.getenv("ANTHROPIC_ENDPOINT", "https://api.anthropic.com")
        api_key = kwargs.get("api_key") or os.getenv("ANTHROPIC_API_KEY", "")
        model_name = kwargs.get("model_name", "claude-3-5-sonnet-20240620")
        temperature = kwargs.get("temperature", 0.0)

        logger.debug("Configuring ChatAnthropic with model=%s, base_url=%s", model_name, base_url)
        return ChatAnthropic(
            model_name=model_name,
            temperature=temperature,
            base_url=base_url,
            api_key=api_key
        )

    # -------------------------
    # OpenAI
    # -------------------------
    elif provider == 'openai':
        base_url = kwargs.get("base_url") or os.getenv("OPENAI_ENDPOINT", "https://api.openai.com/v1")
        api_key = kwargs.get("api_key") or os.getenv("OPENAI_API_KEY", "")
        model_name = kwargs.get("model_name", "gpt-4o")
        temperature = kwargs.get("temperature", 0.0)

        logger.debug("Configuring ChatOpenAI with model=%s, base_url=%s", model_name, base_url)
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            base_url=base_url,
            api_key=api_key
        )

    # -------------------------
    # DeepSeek
    # -------------------------
    elif provider == 'deepseek':
        base_url = kwargs.get("base_url") or os.getenv("DEEPSEEK_ENDPOINT", "")
        api_key = kwargs.get("api_key") or os.getenv("DEEPSEEK_API_KEY", "")
        model_name = kwargs.get("model_name", "deepseek-chat")
        temperature = kwargs.get("temperature", 0.0)

        logger.debug("Configuring DeepSeek ChatOpenAI with model=%s, base_url=%s", model_name, base_url)
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            base_url=base_url,
            api_key=api_key
        )

    # -------------------------
    # Gemini (Google Generative AI)
    # -------------------------
    elif provider == 'gemini':
        api_key = kwargs.get("api_key") or os.getenv("GOOGLE_API_KEY", "")
        model_name = kwargs.get("model_name", "gemini-2.0-flash-exp")
        temperature = kwargs.get("temperature", 0.0)

        logger.debug("Configuring ChatGoogleGenerativeAI with model=%s", model_name)
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=api_key
        )

    # -------------------------
    # Ollama
    # -------------------------
    elif provider == 'ollama':
        model_name = kwargs.get("model_name", "qwen2.5:7b")
        temperature = kwargs.get("temperature", 0.0)

        logger.debug("Configuring ChatOllama with model=%s", model_name)
        return ChatOllama(
            model=model_name,
            temperature=temperature,
        )

    # -------------------------
    # Azure OpenAI
    # -------------------------
    elif provider == 'azure_openai':
        base_url = kwargs.get("base_url") or os.getenv("AZURE_OPENAI_ENDPOINT", "")
        api_key = kwargs.get("api_key") or os.getenv("AZURE_OPENAI_API_KEY", "")
        model_name = kwargs.get("model_name", "gpt-4o")
        temperature = kwargs.get("temperature", 0.0)

        logger.debug("Configuring AzureChatOpenAI with model=%s, base_url=%s", model_name, base_url)
        return AzureChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_version="2024-05-01-preview",
            azure_endpoint=base_url,
            api_key=api_key
        )

    else:
        error_msg = f"Unsupported provider: {provider}"
        logger.error(error_msg)
        raise ValueError(error_msg)


def encode_image(img_path: Optional[str]) -> Optional[str]:
    """
    Encode an image file into a Base64 string.

    Parameters
    ----------
    img_path : Optional[str]
        The path to the image file on disk. If None or empty, returns None.

    Returns
    -------
    Optional[str]
        A Base64-encoded string of the image's contents, or None if the path
        was not provided.
    """
    if not img_path:
        logger.debug("No image path provided; returning None.")
        return None

    if not os.path.isfile(img_path):
        logger.warning("Image file not found at path: %s", img_path)
        return None

    logger.debug("Encoding image from path: %s", img_path)
    with open(img_path, "rb") as fin:
        image_data = base64.b64encode(fin.read()).decode("utf-8")
    return image_data
