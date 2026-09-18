"""One place that builds chat models, so the provider is a config value.

Groq is the default because it has a free tier. Because every other common
provider (OpenAI, Ollama, vLLM, Together, OpenRouter) speaks the OpenAI API,
switching is a config change plus one optional dependency rather than a code
change:

    LLM_PROVIDER=openai  OPENAI_BASE_URL=http://localhost:11434/v1  ...

Install the extra only if you switch:  pip install langchain-openai
"""
import os

from config import LLM_FALLBACK_MODELS, LLM_PROVIDER

# Errors that mean "try the next model" rather than "give up".
RETRYABLE = ("rate_limit", "429", "model_not_found", "404", "does not exist",
             "overloaded", "503", "timeout")


def build_chat_model(model, temperature=0):
    """Return a chat model for the configured provider."""
    if LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=model,
            temperature=temperature,
            api_key=os.getenv("GROQ_API_KEY"),
        )

    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:      # pragma: no cover - optional dependency
        raise RuntimeError(
            f"LLM_PROVIDER='{LLM_PROVIDER}' needs the OpenAI-compatible client. "
            "Install it with: pip install langchain-openai"
        ) from exc

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=os.getenv("OPENAI_API_KEY", "not-needed"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )


def _is_retryable(error):
    text = str(error).lower()
    return any(marker in text for marker in RETRYABLE)


def with_fallback(model, model_call, temperature=0):
    """Call `model_call(model)`; on a provider failure, try the fallbacks.

    A retired or rate-limited model should degrade to the next one rather than
    taking the whole app down — that is exactly how this project died before.
    """
    errors = []
    for candidate in [model, *LLM_FALLBACK_MODELS]:
        try:
            return model_call(build_chat_model(candidate, temperature))
        except Exception as exc:                    # noqa: BLE001
            errors.append(f"{candidate}: {exc}")
            if not _is_retryable(exc):
                raise
    raise RuntimeError("every configured model failed:\n" + "\n".join(errors))


def available_models():
    """Model ids the provider reports, or [] when it cannot be asked."""
    if LLM_PROVIDER == "groq":
        try:
            from groq import Groq

            client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            return sorted(m.id for m in client.models.list().data)
        except Exception:                           # noqa: BLE001
            return []
    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", "not-needed"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )
        return sorted(m.id for m in client.models.list().data)
    except Exception:                               # noqa: BLE001
        return []


def validate_models(models, strict=False):
    """Check configured models still exist. Returns (missing, available).

    Providers that cannot be listed from (or that are unreachable) return an
    empty `available`, which callers treat as "unverified", not "broken".
    """
    available = available_models()
    if not available:
        return [], []
    missing = [m for m in models if m not in available]
    if missing:
        message = (
            "These configured models are no longer available: "
            + ", ".join(missing)
            + "\nAvailable: "
            + ", ".join(available)
            + "\nUpdate the LLM_MODEL_* values in config.py."
        )
        if strict:
            raise RuntimeError(message)
        print("WARNING: " + message)
    return missing, available
