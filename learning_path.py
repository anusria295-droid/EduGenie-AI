from gemini import generate_text


def get_learning_recommendations(
    topic: str,
    level: str,
    goal: str,
) -> str:

    prompt = f"""
Create a personalized learning path.

TOPIC:
{topic}

CURRENT LEVEL:
{level}

GOAL:
{goal}

Organize the learning path
from beginner to advanced.

For each stage include:

- Topics to learn
- What to practice
- Suggested realistic timeline
- Suggested resource types
- Practice ideas

Useful resource types include:

- Official documentation
- Videos
- Articles
- Books
- Practice platforms

Do not fabricate exact URLs.

At the end include:

1. A simple weekly routine
2. A small project idea
3. A short checklist for progress
"""

    return generate_text(
        prompt,
        system_instruction=(
            "You are an educational mentor. "
            "Create practical, progressive "
            "learning paths suitable for "
            "self-learners."
        ),
        temperature=0.4,
        max_output_tokens=2200,
    )
