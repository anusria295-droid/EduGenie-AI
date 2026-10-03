from typing import List

from pydantic import (
    BaseModel,
    Field,
    field_validator,
)

from .gemini import generate_json


class QuizQuestion(BaseModel):

    question: str = Field(
        min_length=1
    )

    options: List[str] = Field(
        min_length=4,
        max_length=4
    )

    correct_answer: int = Field(
        ge=0,
        le=3,
        description=(
            "Zero-based index of "
            "the correct option"
        ),
    )

    explanation: str = Field(
        min_length=1
    )

    @field_validator("options")
    @classmethod
    def validate_options(
        cls,
        value
    ):

        if len(value) != 4:

            raise ValueError(
                "Each question must contain "
                "exactly four options."
            )

        cleaned = [
            option.strip()
            for option in value
        ]

        if any(
            not option
            for option in cleaned
        ):

            raise ValueError(
                "Quiz options cannot be empty."
            )

        return cleaned


class QuizResponse(BaseModel):

    title: str

    questions: List[
        QuizQuestion
    ] = Field(
        min_length=3,
        max_length=3
    )


def generate_quiz(
    passage: str
) -> dict:

    prompt = f"""
Create an educational quiz from
the passage below.

Generate EXACTLY 3 multiple-choice
questions.

For every question:

- Provide exactly 4 options.
- Make the options plausible.
- Have exactly one correct answer.
- correct_answer must be the zero-based
  option index.
- Provide a short explanation.
- Questions must be based only on
  information contained in the passage.

PASSAGE:

{passage}
"""

    raw = generate_json(
        prompt,
        QuizResponse,
        system_instruction=(
            "You create fair and unambiguous "
            "educational quizzes. "
            "Use only information supported "
            "by the supplied passage."
        ),
        temperature=0.2,
        max_output_tokens=2200,
    )

    quiz = QuizResponse.model_validate_json(
        raw
    )

    return quiz.model_dump()