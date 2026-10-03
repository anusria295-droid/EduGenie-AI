from config import settings
from gemini import generate_text


_local_pipeline = None


def _local_explain(
    topic: str
) -> str:

    global _local_pipeline

    if _local_pipeline is None:

        try:

            from transformers import pipeline

            _local_pipeline = pipeline(
                "text2text-generation",
                model=settings.local_explainer_model,
            )

        except Exception as exc:

            raise RuntimeError(
                "Local LaMini model could not "
                f"be loaded: {exc}"
            ) from exc

    prompt = (
        f"Explain {topic} to a beginner "
        "in simple language. "
        "Include a short definition, "
        "how it works, and one example."
    )

    result = _local_pipeline(
        prompt,
        max_new_tokens=300,
        do_sample=False,
    )

    return result[0][
        "generated_text"
    ].strip()


def explain_topic(
    topic: str
) -> str:

    # Try local model if enabled.
    if settings.use_local_explainer:

        try:

            return _local_explain(
                topic
            )

        except Exception:

            # If local model fails,
            # fall back to Gemini.
            pass

    prompt = f"""
Explain the following educational
topic for a beginner:

{topic}

Structure your explanation as:

1. Simple definition
2. How it works
3. Easy analogy or example
4. Key points to remember

Keep the explanation clear,
correct and concise.
"""

    return generate_text(
        prompt,
        system_instruction=(
            "You are EduGenie. "
            "Make difficult concepts easy "
            "without losing correctness."
        ),
        temperature=0.3,
        max_output_tokens=1400,
    )
