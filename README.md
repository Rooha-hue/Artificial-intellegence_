# Artificial-intellegence_
# 📄 ATS Resume Checker

Upload a resume (PDF or DOCX), optionally paste a job description, and get:

- An overall **ATS score (0-100)** with a category breakdown
- Strengths, prioritized **improvements**, and bullet **rewrite examples**
- **Matched / missing keywords** against the job description
- Rule-based quick checks (email, phone, LinkedIn, sections, word count)

Built with [Streamlit](https://streamlit.io) and the Google Gemini Flash API.

## How scoring works

Gemini rates five categories (keywords, content quality, formatting, sections,
readability). The overall score is computed in code as a weighted average
(30 / 25 / 20 / 15 / 10), so it is consistent and not decided by the model alone.
Scores are an estimate, not the output of any real employer's ATS.

## Run locally

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Get a free API key at https://aistudio.google.com, then either:

```bash
export GEMINI_API_KEY="your-key"     # Windows PowerShell: $env:GEMINI_API_KEY="your-key"
streamlit run app.py
```

or create `.streamlit/secrets.toml` (never commit it):

```toml
GEMINI_API_KEY = "your-key"
```

You can also paste the key into the app's sidebar.

## Changing the model

The default model is `gemini-2.5-flash`. Google releases newer Flash models
often, so you can change it in the sidebar or by setting the `GEMINI_MODEL`
environment variable. Check https://ai.google.dev/gemini-api/docs/models for
current names.

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app**, choose your repo, branch `main`, main file `app.py`.
4. Open **Advanced settings → Secrets** and add:
   ```toml
   GEMINI_API_KEY = "your-key"
   ```
5. Click **Deploy**.

## Notes

- Scanned/image-only PDFs have no extractable text; export a text-based PDF or DOCX.
- Resume text is sent to Google's Gemini API for analysis.
