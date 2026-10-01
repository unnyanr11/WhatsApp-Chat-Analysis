# WhatsApp Chat Intelligence

> A privacy-first, local-first analytics platform for turning WhatsApp chat exports into an interactive data-analysis workspace.

Analyze conversation volume, participant activity, timelines, conversation sessions, response patterns, text usage, links, emojis, topics, language, interaction patterns, anomalies and more — without requiring a remote backend for the core workflow.

---

## Table of Contents

- Overview
- Features
- Dashboard
- Architecture
- Project Structure
- Requirements
- Installation
- Running the Application
- Using a WhatsApp Export
- Analytics Reference
- Search
- Advanced Analytics
- Exports
- Python API
- Testing and CI
- Privacy and Security
- Limitations
- Extending the Project
- Roadmap

---

## Overview

WhatsApp Chat Intelligence started as a simple Python/Colab script for calculating basic chat statistics. It has been restructured into a reusable analytics engine plus an interactive Streamlit application.

The project separates the workflow into four layers:

1. **Parsing** — convert WhatsApp exports into normalized message records.
2. **Analytics** — calculate statistics, patterns and text features.
3. **Presentation** — display results through an interactive dashboard.
4. **Export** — generate machine-readable and human-readable reports.

The core workflow is local-first: your chat export can be analyzed on your own machine without a database or external API.

---

# Features

## Chat ingestion

- TXT WhatsApp exports
- ZIP files containing a TXT export
- Multiline messages
- Common 12-hour timestamps
- Common 24-hour timestamps
- Day-first date handling
- System-message recognition
- Media placeholder recognition
- Call-message classification
- URL detection
- Message-type classification

The parser preserves useful system/media records instead of blindly deleting them.

### Normalized message data

The analytics dataframe contains fields such as:

- Message_ID
- Date
- Time
- Author
- Message
- DateTime
- Is_System
- Message_Type
- Is_Deleted
- Message_Length
- Word_Count
- Emoji_Count
- URL_Count
- Is_Question
- Is_Exclamation
- Has_Emoji
- Has_URL
- Date_Only
- Year
- Month
- Day
- Hour
- Is_Weekend
- Domain

---

## Participant analytics

For each participant:

- Message count
- Percentage of total messages
- Total words
- Average words per message
- Average message length
- Emoji count
- Question count
- Link count
- Media count
- Peak messaging hour

---

## Time analytics

Explore activity by:

- Day
- Hour
- Weekday
- Month
- Weekend vs weekday

The dashboard also provides a weekday/hour activity heatmap.

---

## Conversation sessions

Messages can be grouped into conversation sessions using a configurable inactivity threshold.

Example:

~~~text
09:00  A: Hello
09:03  B: Hi
09:12  A: How are you?

        ↓ inactivity threshold

11:10  A: What are you doing?
~~~

Each session can expose:

- Session ID
- Start
- End
- Duration
- Message count
- Participant count
- Initiator

---

## Response-time analytics

The engine detects timestamp transitions where the sender changes and calculates elapsed time.

Available statistics:

- Response count
- Average response time
- Median response time
- Fastest observed response

A maximum-gap threshold prevents very long inactive periods from being treated as direct responses.

---

## Streaks and inactivity

The application calculates:

- Active days
- Longest consecutive active-day streak
- Longest inactive period

These are based on dates present in the exported dataset.

---

## Text analytics

### Top words

Extracts frequently used words while ignoring a basic set of common English stopwords.

### Bigrams

Finds frequently occurring two-word sequences.

### Trigrams

Finds frequently occurring three-word sequences.

These are frequency statistics, not semantic conclusions.

---

## Emoji analytics

The parser detects Unicode emoji characters and calculates:

- Emoji count per message
- Total emoji count
- Participant emoji usage

Emoji counts should not be interpreted as definitive indicators of emotion.

---

## Link analytics

The application detects messages containing URLs and extracts:

- Timestamp
- Author
- Domain
- Original message

---

## Sentiment estimation

A transparent lexical sentiment estimate is available using a small positive/negative vocabulary.

Results are categorized as:

- Positive
- Neutral
- Negative

This is intentionally lightweight and explainable.

It is **not** intended to determine psychological state, actual emotion, personality, intention or relationship quality.

---

## Language detection

When the optional language-detection dependency is available, messages can be classified by detected language.

Very short messages, emoji-only messages and ambiguous text can produce unreliable language classifications.

---

## Topic discovery

Optional TF-IDF/NMF analysis can surface groups of frequently associated terms.

Example:

~~~text
Topic 1 — project, meeting, code, deadline
Topic 2 — travel, hotel, train, booking
Topic 3 — dinner, food, restaurant, tomorrow
~~~

These are statistical term groups rather than authoritative semantic categories.

---

## Participant interaction matrix

The application tracks transitions between participants, such as:

~~~text
A → B
B → A
A → C
C → A
~~~

The resulting interaction matrix can be visualized as a heatmap.

---

## Milestones

The dashboard identifies dataset-level milestones including:

- First message
- Last message
- Busiest day
- Longest message
- First detected emoji

---

## Activity anomaly detection

Daily message counts can be converted into simple z-scores.

Days exceeding the anomaly threshold are surfaced for investigation.

This identifies unusual message volume; it does not explain why activity changed.

---

# Dashboard

The Streamlit application is divided into focused sections.

### Overview

- Message count
- Word count
- Participant count
- Duration
- Question count
- Link count
- Messages-over-time chart
- Hourly activity
- Weekday activity
- Heatmap
- Milestones

### Participants

- Participant statistics
- Message-volume visualization

### Conversations

- Configurable session threshold
- Session table
- Response-time summary
- Streak statistics

### Text & NLP

- Top words
- Bigrams/trigrams
- Lexical sentiment
- Language detection
- Topic discovery

### Search

- Keyword search
- Regex search
- Message-type filtering
- Result table

### Network

- Participant interaction matrix

### Advanced

- Built-in natural-language analytics shortcuts
- Activity anomaly detection
- Optional local media-folder indexing

### Export

- CSV
- Excel
- HTML
- PDF

---

# Architecture

~~~text
                         WhatsApp Export
                         TXT / ZIP
                              |
                              v
                    +-------------------+
                    |   WhatsAppParser  |
                    +-------------------+
                              |
                              v
                  Normalized Message Data
                              |
             +----------------+----------------+
             |                |                |
             v                v                v
      ChatAnalytics       Advanced         Exporters
             |                |                |
      +------+------+      +--+--+       +----+----+
      |      |      |      |     |       |    |   |
   Activity Text  People  NLP  Anomaly   CSV Excel HTML PDF
      |      |      |
      +------+------+
             |
             v
       Streamlit UI
~~~

### Core modules

**whatsapp_analyzer.py**

Parser, feature engineering and reusable analytics engine.

**advanced.py**

Anomaly detection, media-folder indexing and deterministic natural-language shortcuts.

**exporters.py**

Excel, HTML and PDF report generation.

**app.py**

Interactive Streamlit dashboard.

**chat_analysis.py**

Compatibility layer for the original project API.

---

# Project Structure

~~~text
WhatsApp-Chat-Analysis/
│
├── app.py
├── whatsapp_analyzer.py
├── advanced.py
├── exporters.py
├── chat_analysis.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── tests/
│   └── test_parser.py
│
└── .github/
    └── workflows/
        └── tests.yml
~~~

---

# Requirements

| Package | Purpose |
|---|---|
| pandas | Data processing |
| numpy | Numerical operations |
| plotly | Interactive charts |
| streamlit | Dashboard |
| openpyxl | Excel export |
| reportlab | PDF export |
| scikit-learn | TF-IDF/NMF |
| langdetect | Language detection |
| wordcloud | Text visualization support |
| pytest | Testing |

Python 3.12 is used by the CI workflow.

---

# Installation

~~~bash
git clone https://github.com/unnyanr11/WhatsApp-Chat-Analysis.git
cd WhatsApp-Chat-Analysis

python -m venv .venv
~~~

### Windows

~~~bash
.venv\Scripts\activate
~~~

### macOS / Linux

~~~bash
source .venv/bin/activate
~~~

Install dependencies:

~~~bash
pip install -r requirements.txt
~~~

---

# Running the Application

~~~bash
streamlit run app.py
~~~

Open the local Streamlit address shown in the terminal, normally:

~~~text
http://localhost:8501
~~~

No database setup is required.

---

# Using a WhatsApp Export

Export a conversation using WhatsApp's normal chat-export functionality.

The application accepts:

- A TXT chat export
- A ZIP containing a TXT chat export

The filename does not need to follow a particular naming convention.

Typical workflow:

~~~text
WhatsApp
   ↓
Export chat
   ↓
TXT / ZIP
   ↓
Upload
   ↓
Parser
   ↓
Normalized dataframe
   ↓
Analytics
   ↓
Interactive dashboard
~~~

---

# Analytics Reference

| Metric | Meaning |
|---|---|
| Messages | Parsed message records |
| Words | Detected word tokens |
| Characters | Message character count |
| Avg words/message | Mean message word count |
| Questions | Messages containing ? |
| Emojis | Detected Unicode emoji characters |
| Links | URL-containing messages |
| Media | Non-text message classifications |
| Participants | Distinct non-system authors |
| Duration | Difference between first and last timestamps |
| Active days | Dates containing messages |
| Longest streak | Consecutive active dates |
| Longest silence | Largest gap between active dates |

---

# Search

Normal search:

~~~text
meeting
~~~

Regex mode examples:

~~~regex
https?://
~~~

~~~regex
\bhello\b
~~~

Message-type filters include:

~~~text
text
image
video
audio
document
sticker
link
system
~~~

Search results include timestamp, author, message and message type.

---

# Advanced Analytics

## Natural-language shortcuts

The Advanced tab supports deterministic questions such as:

~~~text
Who sent the most messages?
How many messages are there?
How many participants are there?
What was the busiest day?
What is the average response time?
What was the longest message?
How many emojis were used?
How many links were shared?
~~~

This feature does not call an external AI service.

## Media-folder indexing

A local media directory can be indexed and classified into:

- Images
- Videos
- Audio
- Documents
- Other

The index records path, filename, type, extension and file size.

---

# Exports

## CSV

Exports the normalized message dataframe for use in pandas, Excel, R, SQL, Power BI or Tableau.

## Excel

The workbook contains sheets for major datasets such as:

- Messages
- Participants
- Daily Activity
- Responses
- Sessions
- Links
- Interactions

## HTML

Generates a standalone browser-readable report containing overview metrics and participant statistics.

## PDF

Generates a compact report containing overview statistics and participant data.

---

# Python API

The analytics engine can be used without Streamlit.

~~~python
from whatsapp_analyzer import WhatsAppParser, ChatAnalytics

parser = WhatsAppParser(dayfirst=True)
result = parser.parse_file("chat.txt")

df = result.dataframe
analytics = ChatAnalytics(df)

print(analytics.overview())
print(analytics.participant_stats())
~~~

### Daily activity

~~~python
daily = analytics.daily_activity()
~~~

### Conversation sessions

~~~python
sessions = analytics.sessions(gap_minutes=60)
~~~

### Response statistics

~~~python
responses = analytics.response_summary()
~~~

### Search

~~~python
results = analytics.search("meeting", author="John")
~~~

### Top words

~~~python
words = analytics.top_words(n=50)
~~~

### Interaction matrix

~~~python
matrix = analytics.interaction_matrix()
~~~

### Sentiment estimate

~~~python
sentiment = analytics.sentiment()
~~~

### Topic discovery

~~~python
topics = analytics.topics()
~~~

---

# Backwards Compatibility

The original chat_analysis.py entry point remains available.

Existing code can continue using:

~~~python
from chat_analysis import (
    WhatsAppChatProcessor,
    WhatsAppChatAnalysis,
    get_chat_insights,
    search_messages,
    get_author_stats,
    analyze_whatsapp_chat,
)
~~~

New development should generally use:

~~~python
from whatsapp_analyzer import WhatsAppParser, ChatAnalytics
~~~

This keeps the original project workflow usable while providing a cleaner architecture for future development.

---

# Testing and CI

Run tests locally:

~~~bash
pytest -q
~~~

The regression suite covers:

- Multiline messages
- URL detection
- Emoji detection
- Participant extraction
- Basic analytics
- Session detection

GitHub Actions runs the test suite automatically on pushes and pull requests.

Workflow:

~~~text
.github/workflows/tests.yml
        ↓
Install dependencies
        ↓
pytest -q
~~~

---

# Privacy and Security

WhatsApp exports may contain highly sensitive personal information.

This project is designed around local processing.

Recommended practices:

- Analyze private exports locally.
- Never commit chat exports to Git.
- Do not place private exports in the public repository.
- Do not deploy the dashboard publicly with private data unless the hosting model is understood.
- Keep credentials outside version control.
- Treat generated CSV, Excel, HTML and PDF reports as sensitive.
- Delete temporary exports when they are no longer needed.

The core analytics engine does not require a remote database or external AI API.

---

# Limitations

Analytics are limited to information contained in the exported dataset.

Examples:

- Deleted content may be unavailable.
- Media placeholders may not contain the original media.
- WhatsApp export formats can vary.
- Ambiguous dates can affect timestamp interpretation.
- Very short messages can make language detection unreliable.
- Emoji detection is intentionally approximate.
- Response time is inferred from timestamps.
- Conversation sessions depend on the selected inactivity threshold.
- Topic modeling produces statistical term groups.
- Lexical sentiment is not psychological or emotional assessment.
- Activity anomalies identify unusual message volume, not their causes.

The project should therefore be treated as a data-analysis tool rather than a system for inferring private mental states, intentions or relationship characteristics.

---

# Extending the Project

The architecture is intentionally modular.

### Parser extensions

Possible additions:

- More WhatsApp export variants
- More system-event patterns
- Additional date formats
- Better media metadata matching

### Analytics extensions

Possible additions:

- Monthly and quarterly reports
- Conversation similarity
- Co-occurrence analysis
- More advanced response statistics
- Participant comparison reports

### NLP extensions

Possible additions:

- Advanced multilingual sentiment
- Named-entity recognition
- Keyword extraction
- Semantic embeddings
- Optional local LLM integration

### Visualization extensions

Possible additions:

- Network graphs
- Calendar views
- Message-length distributions
- Cumulative activity
- Conversation replay
- Participant comparison dashboards

### Export extensions

Possible additions:

- JSON
- Markdown
- SQLite
- Parquet
- PowerPoint

---

# Roadmap

- [ ] Richer WhatsApp system-event recognition
- [ ] Improved media-to-message matching
- [ ] Richer participant network graphs
- [ ] Conversation replay mode
- [ ] Advanced multilingual NLP
- [ ] Optional local semantic search
- [ ] Optional local LLM querying
- [ ] More comprehensive PDF reports
- [ ] JSON and Parquet exports
- [ ] Configurable analytics profiles
- [ ] Larger regression-test suite
- [ ] Performance optimization for very large exports

---

# Contributing

Contributions are welcome.

For a pull request:

1. Keep the change focused.
2. Add documentation for new functionality.
3. Add tests for new parser or analytics behavior.
4. Never commit private WhatsApp exports.
5. Never commit credentials or API keys.
6. Run the test suite before submitting.

~~~bash
pip install -r requirements.txt
pytest -q
~~~

---

# License

No license has currently been specified for this repository.

Until a license is added, the repository should not be assumed to grant broad permissions to reuse, modify or redistribute the code.

---

# Project Philosophy

### Local-first

Keep personal conversation data under the user's control.

### Modular

Parsing, analytics, visualization and exports should remain independently reusable.

### Transparent

Statistics should be explainable from the underlying messages rather than presented as unsupported conclusions.

### Extensible

The project should be useful as both a finished dashboard and a foundation for more advanced WhatsApp analytics.
