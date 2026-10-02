"""
src/generator.py — LLM Generator (Hugging Face Inference API / Chat Completion)

Module: Module 9 (LLM Generator)

Responsibility:
    - Initialize and configure the Hugging Face Chat LLM (HuggingFaceChatModel)
    - Handle conversational message formatting (System and Human messages) cleanly
    - Connect via Hugging Face InferenceClient chat.completions to avoid parameter mismatch errors (400 Bad Request)
    - Default to LLM_MODEL (HuggingFaceH4/zephyr-7b-beta), LLM_TEMPERATURE (0.3), and LLM_MAX_TOKENS (1024) from src/config.py
    - Expose get_llm() factory function for LCEL pipelines (prompt | llm | StrOutputParser)
    - Provide generate_response() helper to generate textual answers

What is the Generator?
    The Generator is the "brain/speaker" of the RAG assistant:
    1. It receives the grounded prompt prepared by Module 8 (containing retrieved context + question).
    2. Sends the conversational prompt to the open-source LLM via Hugging Face Inference API.
    3. Returns a well-formatted, factual natural language answer to the user.

Imported by:
    - src/rag_pipeline.py (in Module 10)
"""

from typing import Any, List, Optional
# pyrefly: ignore [missing-import]
from langchain_core.callbacks import CallbackManagerForLLMRun
# pyrefly: ignore [missing-import]
from langchain_core.language_models.chat_models import BaseChatModel
# pyrefly: ignore [missing-import]
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
# pyrefly: ignore [missing-import]
from langchain_core.outputs import ChatGeneration, ChatResult
# pyrefly: ignore [missing-import]
from huggingface_hub import InferenceClient

from src.config import (
    HUGGINGFACEHUB_API_TOKEN,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
)


class HuggingFaceChatModel(BaseChatModel):
    """
    Robust LangChain BaseChatModel wrapper around the Hugging Face Inference API.

    Directly interacts with Hugging Face InferenceClient chat completions, ensuring
    clean message formatting and preventing invalid parameter rejection (400 Bad Request)
    from downstream inference providers.
    """

    model_name: str = LLM_MODEL
    api_token: str
    temperature: float = LLM_TEMPERATURE
    max_tokens: int = LLM_MAX_TOKENS
    timeout: int = 120

    @property
    def _llm_type(self) -> str:
        return "huggingface-chat"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        client = InferenceClient(api_key=self.api_token, timeout=self.timeout)

        hf_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                hf_messages.append({"role": "system", "content": str(msg.content)})
            elif isinstance(msg, HumanMessage):
                hf_messages.append({"role": "user", "content": str(msg.content)})
            elif isinstance(msg, AIMessage):
                hf_messages.append({"role": "assistant", "content": str(msg.content)})
            else:
                hf_messages.append({"role": "user", "content": str(msg.content)})

        response = client.chat.completions.create(
            model=self.model_name,
            messages=hf_messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop=stop,
        )

        content = response.choices[0].message.content or ""
        generation = ChatGeneration(message=AIMessage(content=content.strip()))
        return ChatResult(generations=[generation])


def get_llm(
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    api_key: Optional[str] = None,
    max_new_tokens: Optional[int] = None,
) -> HuggingFaceChatModel:
    """
    Initialize and return a Hugging Face Chat LLM instance configured for conversational tasks.

    Args:
        model_name (str, optional): Hugging Face model repository identifier.
                                     Defaults to LLM_MODEL from config.py ('HuggingFaceH4/zephyr-7b-beta').
        temperature (float, optional): Sampling temperature (0.0 to 1.0).
                                       Defaults to LLM_TEMPERATURE from config.py (0.3).
        api_key (str, optional): Hugging Face API token.
                                 Defaults to HUGGINGFACEHUB_API_TOKEN from config.py.
        max_new_tokens (int, optional): Maximum new tokens to generate (default: 1024 from config.py).

    Returns:
        HuggingFaceChatModel: Configured LangChain Chat model instance.

    Raises:
        ValueError: If no valid Hugging Face API token is configured.
    """
    token = api_key or HUGGINGFACEHUB_API_TOKEN
    if not token:
        raise ValueError(
            "Hugging Face API token not found. Please ensure HUGGINGFACEHUB_API_TOKEN "
            "is set in your .env file."
        )

    model = model_name or LLM_MODEL
    temp = temperature if temperature is not None else LLM_TEMPERATURE
    tokens = max_new_tokens if max_new_tokens is not None else LLM_MAX_TOKENS

    return HuggingFaceChatModel(
        model_name=model,
        api_token=token,
        temperature=temp,
        max_tokens=tokens,
    )


def generate_response(
    prompt: str,
    llm: Optional[HuggingFaceChatModel] = None,
) -> str:
    """
    Send a direct text prompt to the Hugging Face LLM and return the generated text response.

    Args:
        prompt (str): Text prompt or formatted message.
        llm (HuggingFaceChatModel, optional): Active LLM instance.

    Returns:
        str: Clean string content of the model's response.
    """
    model = llm or get_llm()
    response = model.invoke(prompt)
    if hasattr(response, "content"):
        return str(response.content).strip()
    return str(response).strip()
