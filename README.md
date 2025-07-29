# SEO News Bot

This project contains a Telegram bot that fetches posts from specified channels,
filters them using OpenAI, and reposts approved content.

## Installation

1. Install Python 3.11 or newer.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

The bot reads API keys and tokens from environment variables:

- `OPENAI_API_KEY` – OpenAI API key
- `TELEGRAM_BOT_TOKEN` – Telegram bot token
- `TG_API_ID` – Telegram API ID
- `TG_API_HASH` – Telegram API hash

Create these variables in your shell or in a `.env` file before running the bot.

## Running

Run the bot with:

```bash
python main.py
```

The bot stores its state in `processed_ids.json` and uses `seo_news_session.session`
for the Telethon client session.
