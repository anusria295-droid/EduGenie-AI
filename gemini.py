import time
from functools import lru_cache

from google import genai
from google.genai import types

from config import settings


class GeminiNotConfiguredError(RuntimeError):
    pass


@lru_cache
def get_client():

    if not settings gemini_configured:
        raise GeminiNotConfiguredError(
            "GEMINI_API_KEY is not configured. "
            "Add your Gemini API key to the .env file."
        )

    return genai.Client(
        api_key=settings.gemini_api_key
    )


def _generate_with_retry(
    client,
    prompt: str,
    config,
):
    max_attempts = 4

    for attempt in range(max_attempts):
        try:
            return client.models.generate_content(
                model=settings.gemini_model,
                contents=prompt,
                config=config,
            )

        except Exception as e:

            error_text = str(e)

            temporary_error = (
                "503" in error_text
                or "UNAVAILABLE" in error_text
                or "high demand" in error_text.lower()
            )

            if not temporary_error:
                raise

            if attempt == max_attempts - 1:
                raise RuntimeError(
                    "Gemini is temporarily unavailable because "
                    "the model is experiencing high demand. "
                    "Please try again in a few minutes."
                )

            wait_seconds = 2 ** attempt

            print(
                f"Gemini temporarily unavailable. "
                f"Retrying in {wait_seconds} seconds..."
            )

            time.sleep(wait_seconds)


def generate_text(
    prompt: str,
    *,
    system_instruction: str | None = None,
    temperature: float = 0.3,
    max_output_tokens: int = 1200,
) -> str:

    client = get_client()

    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        system_instruction=system_instruction,
    )

    response = _generate_with_retry(
        client,
        prompt,
        config,
    )

    text = getattr(
        response,
        "text",
        None
    )

    if not text:
        raise RuntimeError(
            "Gemini returned an empty response."
        )

    return text.strip()


def generate_json(
    prompt: str,
    schema,
    *,
    system_instruction: str | None = None,
    temperature: float = 0.2,
    max_output_tokens: int = 1800,
):

    client = get_client()

    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        system_instruction=system_instruction,
        response_mime_type="application/json",
        response_schema=schema,
    )

    response = _generate_with_retry(
        client,
        prompt,
        config,
    )

    text = getattr(
        response,
        "text",
        None
    )

    if not text:
        raise RuntimeError(
            "Gemini returned an empty JSON response."
        )

    return text.strip()
