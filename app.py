import html
import json
import os
import random
from datetime import datetime
from pathlib import Path

import streamlit as st
from google import genai
from google.genai import types
from pydantic import BaseModel

# Settings
HISTORY_FILE = Path(__file__).with_name("quiz_history.json")
DEFAULT_MODEL = "gemini-3.5-flash-lite" 

FALLBACK_MODELS = ["gemini-3.1-flash-lite", "gemini-3.6-flash", "gemini-2.5-flash-lite"]

DIFFICULTY_LABELS = {"Easy": "🟢 Easy", "Medium": "🟡 Medium", "Hard": "🔴 Hard"}
DIFFICULTY_GUIDE = {
    "Easy": "basic definitions and well-known facts that a beginner can answer",
    "Medium": "understanding of concepts and how things work; the student must "
    "apply what they know, not just remember a definition",
    "Hard": "deep reasoning, comparisons, edge cases and application; the wrong "
    "options must be very plausible so the student has to think carefully",
}
LANGUAGES = {"English": "English", "မြန်မာ (Burmese)": "Burmese"}
SUGGESTED_TOPICS = [
    "Artificial Intelligence",
    "Python Programming",
    "World History",
    "Space & Astronomy",
]

# AI part: generate questions with Gemini

class QuizQuestion(BaseModel):
    """Shape of one question. Gemini is forced to answer in this JSON shape."""

    question: str
    options: list[str]
    correct_index: int
    explanation: str


class QuizError(Exception):
    """An error message that is safe to show to the student."""


def build_prompt(topic: str, difficulty: str, count: int, language: str) -> str:
    return (
        f"You are a teacher writing a multiple-choice quiz.\n"
        f"Topic: {topic}\n"
        f"Difficulty: {difficulty} - questions should test {DIFFICULTY_GUIDE[difficulty]}.\n"
        f"Number of questions: {count}\n"
        f"Language: write everything in {language}.\n\n"
        "Rules:\n"
        "- Each question has exactly 4 different answer options.\n"
        "- Exactly one option is correct. Put its position (0 to 3) in correct_index.\n"
        "- Do not put letters like 'A)' in front of the options.\n"
        "- Questions must be factually correct and clearly worded.\n"
        "- explanation: one or two short sentences saying why the answer is correct.\n"
        "- Do not repeat questions."
    )


def clean_questions(data, count: int) -> list[dict]:
    """Check the AI output and convert it to the format the UI uses."""
    if isinstance(data, dict):
        data = data.get("questions", [])
    if not isinstance(data, list):
        raise ValueError("AI answer is not a list")

    cleaned = []
    for item in data:
        try:
            question = str(item["question"]).strip()
            options = [str(o).strip() for o in item["options"]]
            correct_index = int(item["correct_index"])
            explanation = str(item.get("explanation", "")).strip()
        except (KeyError, TypeError, ValueError):
            continue  # skip a broken question
        if not question or len(options) != 4 or len(set(options)) != 4:
            continue
        if not 0 <= correct_index < 4:
            continue
        answer = options[correct_index]
        random.shuffle(options)  # AI often puts the answer in the same spot
        cleaned.append(
            {
                "question": question,
                "options": options,
                "answer": answer,
                "explanation": explanation,
            }
        )

    if not cleaned:
        raise ValueError("No valid questions in AI answer")
    return cleaned[:count]


def friendly_error(exc: Exception) -> str:
    text = str(exc)
    low = text.lower()
    if "429" in text or "resource_exhausted" in low or "quota" in low:
        return (
            "The free-tier rate limit was reached. "
            "Please wait about a minute and try again."
        )
    if "api key" in low or "api_key" in low or "401" in text or "403" in text:
        return "Gemini rejected the API key. Please check that it is correct."
    if "404" in text or "not found" in low:
        return "That model was not found. Change the model name in Advanced settings."
    return f"Could not get questions from Gemini: {text[:200]}"


def generate_quiz(api_key, model, topic, difficulty, count, language) -> list[dict]:
    client = genai.Client(api_key=api_key)
    prompt = build_prompt(topic, difficulty, count, language)
    candidates = [model] + [m for m in FALLBACK_MODELS if m != model]
    last_error = None

    for candidate in candidates:  # next model only if this one does not exist
        for _ in range(3):  # retry if the AI returns broken JSON
            try:
                response = client.models.generate_content(
                    model=candidate,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=list[QuizQuestion],
                        temperature=0.7,
                    ),
                )
                return clean_questions(json.loads(response.text), count)
            except (ValueError, TypeError):  # includes JSONDecodeError
                continue
            except Exception as exc:  # network, key, rate limit ...
                if "404" in str(exc) or "not found" in str(exc).lower():
                    last_error = exc
                    break  # model missing -> try the next model
                raise QuizError(friendly_error(exc)) from exc
        else:
            raise QuizError(
                "Gemini returned an unreadable answer 3 times. Please try again."
            )

    raise QuizError(friendly_error(last_error))


def resolve_api_key(typed: str) -> str:
    """Order: sidebar box -> Streamlit secrets -> environment variable."""
    if typed and typed.strip():
        return typed.strip()
    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        key = ""
    return key or os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")


# Storage part: JSON file
def load_history() -> list[dict]:
    try:
        return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []


def write_history(history: list[dict]) -> None:
    try:
        HISTORY_FILE.write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError:
        pass  # saving is a bonus feature, never crash the quiz


def save_attempt(entry: dict) -> None:
    history = load_history()
    history.append(entry)
    write_history(history[-50:])  # keep the latest 50


# State handling (callbacks)
def init_state() -> None:
    defaults = {
        "questions": None,
        "meta": {},
        "submitted": False,
        "answers": [],
        "score": 0,
        "quiz_id": 0,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def start_quiz(questions: list[dict], meta: dict) -> None:
    st.session_state.questions = questions
    st.session_state.meta = meta
    st.session_state.submitted = False
    st.session_state.answers = []
    st.session_state.score = 0
    st.session_state.quiz_id += 1  # new widget keys -> clean radio buttons


def retry_quiz() -> None:
    start_quiz(st.session_state.questions, st.session_state.meta)


def reset_quiz() -> None:
    st.session_state.questions = None
    st.session_state.submitted = False


def set_topic(topic: str) -> None:
    st.session_state["topic_input"] = topic


def load_saved_quiz(index: int) -> None:
    item = list(reversed(load_history()))[index]
    start_quiz(
        item["questions"],
        {
            "topic": item["topic"],
            "difficulty": item["difficulty"],
            "language": item.get("language", "English"),
        },
    )


def clear_history() -> None:
    write_history([])


def finish_quiz(answers: list[str]) -> None:
    questions = st.session_state.questions
    meta = st.session_state.meta
    score = sum(a == q["answer"] for a, q in zip(answers, questions))
    st.session_state.answers = answers
    st.session_state.score = score
    st.session_state.submitted = True
    save_attempt(
        {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "topic": meta["topic"],
            "difficulty": meta["difficulty"],
            "language": meta.get("language", "English"),
            "score": score,
            "total": len(questions),
            "questions": questions,
        }
    )


# UI part
CSS = """
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {max-width: 820px; padding-top: 2rem;}

.hero {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 55%, #ec4899 100%);
    border-radius: 22px; padding: 28px 32px; margin-bottom: 1.5rem;
    box-shadow: 0 12px 32px rgba(99, 102, 241, 0.35);
}
.hero h1 {color: #fff; margin: 0; font-size: 2.1rem; padding: 0;}
.hero p {color: rgba(255,255,255,.92); margin: .5rem 0 0; font-size: 1.05rem;}

.chip {
    display: inline-block; padding: 4px 14px; margin: 0 8px 12px 0;
    border-radius: 999px; font-size: .82rem; font-weight: 600;
    background: rgba(99, 102, 241, .15); color: #6366f1;
}

.stButton > button, .stFormSubmitButton > button {
    width: 100%; border-radius: 12px; font-weight: 600; padding: .6rem 1rem;
}

.score-card {
    display: flex; align-items: center; gap: 28px; color: #fff;
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    border-radius: 22px; padding: 26px 30px; margin: .5rem 0 1.5rem;
    box-shadow: 0 12px 32px rgba(79, 70, 229, .35);
}
.score-card h2 {color: #fff; margin: 0; padding: 0; font-size: 1.7rem;}
.score-card p {margin: .3rem 0 0; color: rgba(255,255,255,.9);}
.ring {
    flex: none; width: 112px; height: 112px; border-radius: 50%;
    display: grid; place-items: center; position: relative;
    background: conic-gradient(#fff calc(var(--pct) * 1%), rgba(255,255,255,.25) 0);
}
.ring::before {
    content: ""; position: absolute; inset: 10px; border-radius: 50%;
    background: #5b47e4;
}
.ring span {position: relative; font-size: 1.5rem; font-weight: 800;}

.review {
    border-left: 5px solid; border-radius: 12px; padding: 14px 18px;
    margin: 8px 0 18px; line-height: 1.6;
}
.review.ok  {border-color: #22c55e; background: rgba(34, 197, 94, .10);}
.review.bad {border-color: #ef4444; background: rgba(239, 68, 68, .10);}
.review .q {font-weight: 700; margin-bottom: 6px;}
.review .tag {font-weight: 700;}
.review .why {margin-top: 8px; opacity: .85; font-style: italic;}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def hero() -> None:
    st.markdown(
        '<div class="hero"><h1>🧠 AI Quiz Generator</h1>'
        "<p>Type any topic, pick a level, and let Gemini write a quiz for you.</p></div>",
        unsafe_allow_html=True,
    )


def sidebar() -> None:
    with st.sidebar:
        st.header("⚙️ Settings")
        st.text_input(
            "Gemini API key",
            type="password",
            key="api_key_input",
            help="Not saved anywhere. You can also set GEMINI_API_KEY instead.",
        )
        st.caption("Get a free key: aistudio.google.com/apikey")
        with st.expander("Advanced"):
            st.text_input("Model name", value=DEFAULT_MODEL, key="model_input")

        st.divider()
        st.subheader("📚 Saved quizzes")
        history = list(reversed(load_history()))
        if not history:
            st.caption("Finished quizzes will appear here.")
            return
        labels = [
            f"{h['topic']} · {h['difficulty']} · {h['score']}/{h['total']}"
            for h in history
        ]
        choice = st.selectbox(
            "Pick a quiz",
            range(len(history)),
            format_func=lambda i: labels[i],
            label_visibility="collapsed",
        )
        st.caption(f"Taken on {history[choice]['date']}")
        st.button("🔁 Retake this quiz", on_click=load_saved_quiz, args=(choice,))
        st.button("🗑️ Clear history", on_click=clear_history)


def setup_view() -> None:
    with st.container(border=True):
        st.markdown("#### What do you want to practice?")
        topic = st.text_input(
            "Topic", key="topic_input", placeholder="e.g. Artificial Intelligence"
        )
        st.caption("Need an idea? Tap one:")
        cols = st.columns(len(SUGGESTED_TOPICS))
        for col, suggestion in zip(cols, SUGGESTED_TOPICS):
            col.button(
                suggestion, key=f"suggest_{suggestion}", on_click=set_topic, args=(suggestion,)
            )

        st.markdown("**Difficulty**")
        difficulty = st.radio(
            "Difficulty",
            list(DIFFICULTY_LABELS),
            index=1,
            format_func=DIFFICULTY_LABELS.get,
            horizontal=True,
            label_visibility="collapsed",
        )
        left, right = st.columns(2)
        count = left.slider("Number of questions", 3, 10, 5)
        language = right.selectbox("Quiz language", list(LANGUAGES))

    if not st.button("✨ Generate quiz", type="primary"):
        return

    topic = topic.strip()
    api_key = resolve_api_key(st.session_state.get("api_key_input", ""))
    model = (st.session_state.get("model_input") or DEFAULT_MODEL).strip()

    if not topic:
        st.warning("Please type a topic first.")
        return
    if not api_key:
        st.error("Please paste your Gemini API key in the sidebar (left side).")
        return

    with st.spinner("Gemini is writing your questions..."):
        try:
            questions = generate_quiz(
                api_key, model, topic, difficulty, count, LANGUAGES[language]
            )
        except QuizError as exc:
            st.error(str(exc))
            return

    start_quiz(
        questions,
        {"topic": topic, "difficulty": difficulty, "language": LANGUAGES[language]},
    )
    st.rerun()


def meta_chips(meta: dict, total: int) -> None:
    st.markdown(
        f'<span class="chip">📖 {html.escape(meta["topic"])}</span>'
        f'<span class="chip">{DIFFICULTY_LABELS[meta["difficulty"]]}</span>'
        f'<span class="chip">{total} questions</span>',
        unsafe_allow_html=True,
    )


def quiz_view() -> None:
    questions = st.session_state.questions
    quiz_id = st.session_state.quiz_id
    meta_chips(st.session_state.meta, len(questions))

    with st.form(f"quiz_form_{quiz_id}"):
        for i, q in enumerate(questions):
            with st.container(border=True):
                st.markdown(f"**Question {i + 1} of {len(questions)}**")
                st.markdown(q["question"])
                st.radio(
                    "Choose one answer",
                    q["options"],
                    index=None,
                    key=f"q_{quiz_id}_{i}",
                    label_visibility="collapsed",
                )
        submitted = st.form_submit_button("✅ Submit answers", type="primary")

    st.button("← Cancel and go back", on_click=reset_quiz)

    if submitted:
        answers = [st.session_state.get(f"q_{quiz_id}_{i}") for i in range(len(questions))]
        missing = [str(i + 1) for i, a in enumerate(answers) if a is None]
        if missing:
            st.warning("Please answer question " + ", ".join(missing) + " first.")
        else:
            finish_quiz(answers)
            st.rerun()


def result_view() -> None:
    questions = st.session_state.questions
    answers = st.session_state.answers
    score = st.session_state.score
    total = len(questions)
    pct = round(score / total * 100)

    if pct >= 80:
        title = "Excellent work! 🎉"
    elif pct >= 50:
        title = "Good job! 👍"
    else:
        title = "Keep practicing! 💪"

    meta_chips(st.session_state.meta, total)
    st.markdown(
        f'<div class="score-card"><div class="ring" style="--pct:{pct}">'
        f"<span>{score}/{total}</span></div>"
        f"<div><h2>{title}</h2><p>You got {score} of {total} correct ({pct}%).</p></div></div>",
        unsafe_allow_html=True,
    )

    st.markdown("### Review your answers")
    for i, (q, picked) in enumerate(zip(questions, answers), start=1):
        correct = picked == q["answer"]
        css_class = "ok" if correct else "bad"
        icon = "✅ Correct" if correct else "❌ Wrong"
        lines = [
            f'<div class="q">{i}. {html.escape(q["question"])}</div>',
            f'<span class="tag">{icon}</span>',
            f"<br>Your answer: {html.escape(picked)}",
        ]
        if not correct:
            lines.append(f"<br>Correct answer: <b>{html.escape(q['answer'])}</b>")
        if q["explanation"]:
            lines.append(f'<div class="why">💡 {html.escape(q["explanation"])}</div>')
        st.markdown(
            f'<div class="review {css_class}">{"".join(lines)}</div>',
            unsafe_allow_html=True,
        )

    left, right = st.columns(2)
    left.button("🔁 Retry this quiz", on_click=retry_quiz)
    right.button("✨ New quiz", type="primary", on_click=reset_quiz)


def main() -> None:
    st.set_page_config(page_title="AI Quiz Generator", page_icon="🧠", layout="centered")
    inject_css()
    init_state()
    sidebar()
    hero()

    if st.session_state.questions is None:
        setup_view()
    elif not st.session_state.submitted:
        quiz_view()
    else:
        result_view()


if __name__ == "__main__":
    main()