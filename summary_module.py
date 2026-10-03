from gemini import generate_text


def summarize_text(
    text: str
) -> str:

    prompt = f"""
Summarize the educational passage
below for quick revision.

Requirements:

- Preserve important facts.
- Preserve definitions.
- Preserve important relationships.
- Preserve conclusions.
- Remove repetition.
- Do not add information.
- Use a short heading.
- Use concise bullet points.

PASSAGE:

{text}
"""

    return generate_text(
        prompt,
        system_instruction=(
            "You are an educational summarizer. "
            "Never add facts that are absent "
            "from the supplied passage."
        ),
        temperature=0.2,
        max_output_tokens=1600,
    )
