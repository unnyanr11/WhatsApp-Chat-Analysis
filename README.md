# WhatsApp Chat Intelligence

A privacy-first, local-first analytics platform for exported WhatsApp conversations.

## Features

- Robust TXT and ZIP ingestion
- Multiline-message parsing
- Common 12/24-hour WhatsApp timestamp formats
- Day-first date parsing for ambiguous exports
- Preserved system/media events with message-type classification
- Message, word, character, emoji, question and URL metrics
- Participant profiles and message-share analysis
- Hour/day/month timelines and activity heatmaps
- Conversation/session detection and initiator tracking
- Response-time statistics
- Messaging streaks and inactivity periods
- Keyword and regex search with participant/date/type filters
- Top words, bigrams and trigrams
- Transparent lexical sentiment estimates
- Optional language detection
- Optional TF-IDF/NMF topic discovery
- Participant interaction matrix
- Milestones
- CSV, Excel, HTML and PDF reports
- Interactive Streamlit dashboard
- Automated parser and analytics tests
- Reusable Python engine independent of the UI

## Run

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Open the local Streamlit URL, upload a WhatsApp TXT export or a ZIP containing one, and explore the dashboard.

## Architecture

```
WhatsApp TXT/ZIP
      |
      v
WhatsAppParser
      |
      v
Normalized message dataframe
      |
      +--> ChatAnalytics
      |       +--> activity
      |       +--> participants
      |       +--> sessions/responses
      |       +--> text/NLP
      |       +--> interactions
      |
      +--> exporters
      |
      v
Streamlit dashboard
```

## Privacy

The application is designed for local processing. Chat data is kept in the current application session and is only written when the user explicitly downloads an export. Do not deploy private chat data to a public server unless the deployment's storage, logging and access model are understood.

Sentiment is intentionally a lightweight lexical estimate. It should not be interpreted as a psychological assessment.

## Backwards compatibility

The original `chat_analysis.py` entry point remains available as a compatibility layer. New projects should import:

```python
from whatsapp_analyzer import WhatsAppParser, ChatAnalytics
```

## Testing

```bash
pytest -q
```

## Project structure

- `app.py` — Streamlit dashboard
- `whatsapp_analyzer.py` — parser, feature engineering and analytics
- `exporters.py` — Excel, HTML and PDF generation
- `tests/` — regression tests
- `chat_analysis.py` — legacy-compatible API

## Future extensions

The architecture is ready for media-folder indexing, richer WhatsApp system-event parsing, reply/quote extraction where available, semantic embeddings, local LLM querying, richer network graphs, and additional export formats.
