# Telegram Bot

## Configuration
Copy `.env.example` to `.env` and fill in the required API keys.

```
cp .env.example .env
nano .env
```

The `TG_API_ID` and `TG_API_HASH` values must come from a Telegram API
application created at <https://my.telegram.org>. These credentials must match
any `seo_news_session.session` file you use; otherwise Telethon will raise an
`ApiIdInvalidError` on startup.

The `ADMIN_ID` variable is optional and allows the bot to send error tracebacks
to your Telegram user.

## Installation
```
pip install -r requirements.txt
```

## Running
```
python main.py
```

### Using a user session (for private channels)
1. Generate a Telethon session once:
   ```
   python - <<'PY'
   from telethon import TelegramClient
   api_id = TG_API_ID
   api_hash = 'TG_API_HASH'
   with TelegramClient('seo_news_session', api_id, api_hash) as client:
       print('Session saved:', client.session.save())
   PY
   ```
   Follow the prompts to log in; a `seo_news_session.session` file will appear.
2. Place this session file next to `main.py` and make sure it is copied into the
   container when deploying.

The bot will automatically use this session; otherwise it connects via the
`TELEGRAM_BOT_TOKEN`.

For persistent running on a server via SSH, consider using `screen` or `tmux`, or configure a systemd service.
