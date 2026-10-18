# 🧠 AI Quiz Generator

A small web app that writes multiple-choice quizzes with Google Gemini.

1. The student enters a topic (e.g. *Artificial Intelligence*).
2. The student picks a difficulty: Easy, Medium or Hard.
3. Gemini generates the multiple-choice questions.
4. The student selects answers and submits.
5. The app checks the answers and shows the score with explanations.

**Tech:** Python, Streamlit, Google Gemini API, JSON file storage.
**AI technique:** Generative AI / NLP. A Large Language Model creates the questions, and prompt engineering controls the topic, difficulty, language and output format.

## Run it

```powershell
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

## Get a free Gemini API key

1. Open https://aistudio.google.com/apikey and sign in with a Google account.
2. Click **Create API key** and copy it.
3. Paste it into the sidebar of the app (it is not saved anywhere).

Instead of pasting it every time, you can set an environment variable before running:

- Windows (PowerShell): `$env:GEMINI_API_KEY="your-key"`
- macOS / Linux: `export GEMINI_API_KEY="your-key"`

Or create `.streamlit/secrets.toml` with `GEMINI_API_KEY = "your-key"`.
**Never upload your key to GitHub.**

Free tier limits change over time and depend on the model and your country. Check the current limits in Google AI Studio. If you see a "rate limit" message, wait one minute and try again. The app asks for all questions in a single request to stay inside the free limits.

## How it works

| Part | Where in `app.py` |
|------|-------------------|
| Prompt and Gemini call (forced JSON output) | `build_prompt`, `generate_quiz` |
| Checking and cleaning AI output | `clean_questions` |
| Storing quizzes and scores in JSON | `load_history`, `save_attempt` |
| Quiz and result screens | `quiz_view`, `result_view` |

Finished quizzes are saved to `quiz_history.json` next to `app.py`. You can retake them from the sidebar.

## Notes for your report

- AI answers can be wrong, so the explanation shown after each question helps the student verify it.
- The app validates the AI output (4 unique options, valid correct answer) and retries up to 3 times if the JSON is broken.
- Options are shuffled locally so the correct answer is not always in the same position.

## Ideas to extend

- Timer for each quiz
- Score chart over time
- Export a quiz to PDF
