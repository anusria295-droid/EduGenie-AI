from .gemini import generate_text


def answer_question(
    question: str
) -> str:

    prompt = f"""
Answer the student's question accurately
and concisely.

QUESTION:
{question}

Rules:

- Explain at a learner-friendly level.
- Use short sections or bullet points when useful.
- If the question is ambiguous, state the assumption briefly.
- Do not invent citations or sources.
- Do not unnecessarily make the answer very long.
"""

    return generate_text(
        prompt,
        system_instruction=(
            "You are EduGenie, a careful "
            "educational question-answering "
            "assistant."
        ),
        temperature=0.3,
        max_output_tokens=1200,
    )