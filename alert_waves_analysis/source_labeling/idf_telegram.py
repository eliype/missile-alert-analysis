"""Download and parse the official IDF Telegram channel's public HTML pages."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import date, datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import pandas as pd


CHANNEL = "idfofficial"
BASE_URL = f"https://t.me/s/{CHANNEL}"
ISRAEL_TZ = ZoneInfo("Asia/Jerusalem")
USER_AGENT = (
    "Mozilla/5.0 (compatible; HUJI-course-research/1.0; "
    "+https://t.me/idfofficial)"
)


@dataclass
class TelegramMessage:
    post_id: int
    timestamp_utc: datetime
    text: str

    @property
    def timestamp_local(self) -> datetime:
        return self.timestamp_utc.astimezone(ISRAEL_TZ)

    @property
    def source_url(self) -> str:
        return f"https://t.me/{CHANNEL}/{self.post_id}"


class TelegramPageParser(HTMLParser):
    """Small purpose-built parser for Telegram's public channel markup."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.messages: list[TelegramMessage] = []
        self._current_post_id: int | None = None
        self._current_timestamp: datetime | None = None
        self._text_parts: list[str] = []
        self._text_depth = 0

    def _finish_current(self) -> None:
        if self._current_post_id is None:
            return
        if self._current_timestamp is not None:
            text = " ".join("".join(self._text_parts).split())
            self.messages.append(
                TelegramMessage(
                    post_id=self._current_post_id,
                    timestamp_utc=self._current_timestamp,
                    text=text,
                )
            )
        self._current_post_id = None
        self._current_timestamp = None
        self._text_parts = []
        self._text_depth = 0

    def handle_starttag(self, tag: str, attrs_list: list[tuple[str, str | None]]) -> None:
        attrs = dict(attrs_list)
        post = attrs.get("data-post")
        if tag == "div" and post and post.startswith(f"{CHANNEL}/"):
            self._finish_current()
            self._current_post_id = int(post.rsplit("/", 1)[1])
            return

        if self._current_post_id is None:
            return

        classes = set((attrs.get("class") or "").split())
        if tag == "div" and "tgme_widget_message_text" in classes:
            self._text_depth = 1
            return
        if self._text_depth:
            if tag == "br":
                self._text_parts.append("\n")
            elif tag in {"div", "p", "blockquote"}:
                self._text_depth += 1

        if tag == "time" and attrs.get("datetime"):
            parsed = datetime.fromisoformat(attrs["datetime"])
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=ZoneInfo("UTC"))
            self._current_timestamp = parsed

    def handle_endtag(self, tag: str) -> None:
        if self._text_depth and tag in {"div", "p", "blockquote"}:
            self._text_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._text_depth:
            self._text_parts.append(data)

    def close(self) -> None:
        super().close()
        self._finish_current()


def parse_page(html_text: str) -> tuple[list[TelegramMessage], int | None]:
    parser = TelegramPageParser()
    parser.feed(html_text)
    parser.close()
    older_match = re.search(
        r'<link\s+rel="prev"\s+href="/s/idfofficial\?before=(\d+)"',
        html_text,
    )
    older_cursor = int(older_match.group(1)) if older_match else None
    if older_cursor is None and parser.messages:
        older_cursor = min(message.post_id for message in parser.messages)
    return parser.messages, older_cursor


def download_page(cursor: int, cache_dir: Path, refresh: bool = False) -> str:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"before_{cursor}.html"
    if cache_path.exists() and not refresh:
        return cache_path.read_text(encoding="utf-8")

    url = f"{BASE_URL}?before={cursor}"
    error: Exception | None = None
    for attempt in range(3):
        try:
            request = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(request, timeout=30) as response:
                body = response.read().decode("utf-8", errors="replace")
            if "tgme_widget_message" not in body:
                raise RuntimeError(f"Telegram returned no message markup for {url}")
            cache_path.write_text(body, encoding="utf-8")
            return body
        except (HTTPError, URLError, TimeoutError, RuntimeError) as exc:
            error = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Could not download {url}: {error}")


def crawl_messages(
    start_date: date,
    end_date: date,
    cache_dir: Path,
    initial_before: int = 18_800,
    refresh: bool = False,
    max_pages: int = 250,
) -> pd.DataFrame:
    """Crawl backward until messages older than ``start_date`` are reached."""
    cursor = initial_before
    seen_cursors: set[int] = set()
    collected: dict[int, TelegramMessage] = {}

    for page_number in range(1, max_pages + 1):
        if cursor in seen_cursors:
            raise RuntimeError(f"Telegram pagination loop at before={cursor}")
        seen_cursors.add(cursor)

        html_text = download_page(cursor, cache_dir, refresh=refresh)
        messages, older_cursor = parse_page(html_text)
        if not messages:
            raise RuntimeError(f"No messages parsed from before={cursor}")

        for message in messages:
            collected[message.post_id] = message

        oldest_local_date = min(message.timestamp_local.date() for message in messages)
        print(
            f"page {page_number:03d}: before={cursor}, "
            f"messages={len(messages)}, oldest={oldest_local_date}",
            flush=True,
        )
        if oldest_local_date < start_date:
            break
        if older_cursor is None or older_cursor >= cursor:
            raise RuntimeError(
                f"Invalid older cursor {older_cursor!r} after before={cursor}"
            )
        cursor = older_cursor
    else:
        raise RuntimeError(
            f"Reached max_pages={max_pages} before reaching {start_date}"
        )

    records = []
    for message in sorted(collected.values(), key=lambda item: item.post_id):
        local_time = message.timestamp_local
        if not (start_date <= local_time.date() <= end_date):
            continue
        records.append(
            {
                "post_id": message.post_id,
                "timestamp_utc": message.timestamp_utc.isoformat(),
                "timestamp_local": local_time.isoformat(),
                "text": message.text,
                "source_url": message.source_url,
            }
        )
    return pd.DataFrame(records)
