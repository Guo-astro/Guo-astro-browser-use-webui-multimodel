# -*- coding: utf-8 -*-
"""
Filename: test_llm_api.py

Description:
    Test scripts for verifying functionality of various LLM providers (OpenAI,
    Gemini, Azure OpenAI, DeepSeek, Ollama). Each test function sets up an LLM,
    encodes an image (if needed), invokes the model, and displays the response.
"""

import os
import sys
import logging

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Optionally, set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Ensure local paths are recognized
sys.path.append(".")


def test_openai_model() -> None:
    """
    Test the OpenAI GPT model, passing an image plus text prompt and printing the LLM response.
    """
    from langchain_core.messages import HumanMessage
    from src.utils import utils

    llm = utils.get_llm_model(
        provider="openai",
        model_name="gpt-4o",
        temperature=0.8,
        base_url=os.getenv("OPENAI_ENDPOINT", ""),
        api_key=os.getenv("OPENAI_API_KEY", "")
    )

    image_path = "assets/examples/test.png"
    image_data = utils.encode_image(image_path)

    message = HumanMessage(
        content=[
            {"type": "text", "text": "describe this image"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
            },
        ]
    )

    logger.info("Invoking OpenAI GPT model...")
    ai_msg = llm.invoke([message])
    logger.info("Model response: %s", ai_msg.content)
    print(ai_msg.content)


def test_gemini_model() -> None:
    """
    Test the Google Generative AI (Gemini) model, providing an image-based prompt
    and retrieving its textual description.
    """
    from langchain_core.messages import HumanMessage
    from src.utils import utils

    llm = utils.get_llm_model(
        provider="deepseek",  # or 'gemini' if you directly support 'gemini' as a provider
        model_name="gemini-2.0-flash-exp",
        temperature=0.8,
        api_key=os.getenv("GOOGLE_API_KEY", "")
    )

    image_path = "assets/examples/test.png"
    image_data = utils.encode_image(image_path)

    message = HumanMessage(
        content=[
            {"type": "text", "text": "describe this image"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
            },
        ]
    )

    logger.info("Invoking Gemini model...")
    ai_msg = llm.invoke([message])
    logger.info("Model response: %s", ai_msg.content)
    print(ai_msg.content)


def test_azure_openai_model() -> None:
    """
    Test the Azure OpenAI model by sending an image-based prompt and printing the LLM response.
    """
    from langchain_core.messages import HumanMessage
    from src.utils import utils

    llm = utils.get_llm_model(
        provider="azure_openai",
        model_name="gpt-4o",
        temperature=0.8,
        base_url=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        api_key=os.getenv("AZURE_OPENAI_API_KEY", "")
    )

    image_path = "assets/examples/test.png"
    image_data = utils.encode_image(image_path)

    message = HumanMessage(
        content=[
            {"type": "text", "text": "describe this image"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
            },
        ]
    )

    logger.info("Invoking Azure OpenAI GPT model...")
    ai_msg = llm.invoke([message])
    logger.info("Model response: %s", ai_msg.content)
    print(ai_msg.content)


def test_deepseek_model() -> None:
    """
    Test the DeepSeek model, sending a simple textual prompt and printing the LLM response.
    """
    from langchain_core.messages import HumanMessage
    from src.utils import utils

    llm = utils.get_llm_model(
        provider="deepseek",
        model_name="deepseek-chat",
        temperature=0.8,
        base_url=os.getenv("DEEPSEEK_ENDPOINT", ""),
        api_key=os.getenv("DEEPSEEK_API_KEY", "")
    )

    message = HumanMessage(
        content=[
            {"type": "text", "text": "Who are you?"}
        ]
    )

    logger.info("Invoking DeepSeek model...")
    ai_msg = llm.invoke([message])
    logger.info("Model response: %s", ai_msg.content)
    print(ai_msg.content)


def test_ollama_model() -> None:
    """
    Test the Ollama model by sending a textual prompt and printing its response.
    """
    from langchain_ollama import ChatOllama

    logger.info("Invoking Ollama model...")
    llm = ChatOllama(model="qwen2.5:7b")
    ai_msg = llm.invoke("Sing a ballad of LangChain.")
    logger.info("Model response: %s", ai_msg.content)
    print(ai_msg.content)


if __name__ == "__main__":
    # Uncomment the tests you want to run:
    # test_openai_model()
    # test_gemini_model()
    # test_azure_openai_model()
    # test_deepseek_model()
    test_ollama_model()
