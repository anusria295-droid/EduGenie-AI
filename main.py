from pathlib import Path

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from config import settings
from explanation_module import explain_topic
from learning_path import get_learning_recommendations
from qna import answer_question
from quiz_module import generate_quiz
from summary_module import summarize_text


BASE_DIR = Path(__file__).resolve().parent


app = FastAPI(
    title="EduGenie",
    version="1.0.0",
    description="Google Gemini Powered Learning Assistant",
)


app.mount(
    "/static",
    StaticFiles(
        directory=BASE_DIR / "services" / "templates" / "static"
    ),
    name="static",
)


templates = Jinja2Templates(directory="services/templates")


# ---------------------------------------------------------
# Request Models
# ---------------------------------------------------------

class QARequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=12000
    )


class ExplainRequest(BaseModel):
    topic: str = Field(
        ...,
        min_length=1,
        max_length=12000
    )


class TextRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=30000
    )


class LearningRequest(BaseModel):
    topic: str = Field(
        ...,
        min_length=1,
        max_length=5000
    )

    level: str = Field(
        default="beginner",
        max_length=50
    )

    goal: str = Field(
        default="Build a strong understanding",
        max_length=5000
    )


# ---------------------------------------------------------
# Gemini Error Handler
# ---------------------------------------------------------

def handle_gemini_error(error):

    error_text = str(error)

    if (
        "429" in error_text
        or "RESOURCE_EXHAUSTED" in error_text
    ):
        raise HTTPException(
            status_code=429,
            detail=(
                "Gemini API quota exceeded. "
                "Please wait and try again later."
            )
        )

    raise HTTPException(
        status_code=500,
        detail=(
            "Gemini API request failed. "
            "Please try again."
        )
    )


# ---------------------------------------------------------
# Frontend
# ---------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request,
            "app_name": settings.app_name,
        }
    )


# ---------------------------------------------------------
# Health Check
# ---------------------------------------------------------

@app.get("/health")
async def health():

    return {
        "status": "ok",
        "gemini_configured": settings.gemini_configured,
        "local_explainer_enabled": settings.use_local_explainer,
    }


# ---------------------------------------------------------
# Question Answering
# ---------------------------------------------------------

@app.post("/qa")
async def qa(payload: QARequest):

    answer = answer_question(
        payload.question
    )

    return {
        "answer": answer
    }


# ---------------------------------------------------------
# Concept Explanation
# ---------------------------------------------------------

@app.post("/explain")
async def explain(payload: ExplainRequest):

    explanation = explain_topic(
        payload.topic
    )

    return {
        "explanation": explanation
    }


# ---------------------------------------------------------
# Quiz Generation
# ---------------------------------------------------------

@app.post("/quiz")
async def quiz(payload: TextRequest):

    try:

        return generate_quiz(
            payload.text
        )

    except Exception as error:

        handle_gemini_error(error)


# ---------------------------------------------------------
# Summarization
# ---------------------------------------------------------

@app.post("/summarize")
async def summarize(payload: TextRequest):

    try:

        summary = summarize_text(
            payload.text
        )

        return {
            "summary": summary
        }

    except Exception as error:

        handle_gemini_error(error)


# ---------------------------------------------------------
# Learning Path
# ---------------------------------------------------------

@app.post("/learn/recommendations")
async def learning_recommendations(
    payload: LearningRequest
):

    try:

        learning_path = get_learning_recommendations(
            topic=payload.topic,
            level=payload.level,
            goal=payload.goal,
        )

        return {
            "learning_path": learning_path
        }

    except Exception as error:

        handle_gemini_error(error)
