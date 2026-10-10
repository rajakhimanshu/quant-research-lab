"""Paper monitor — weekly arXiv + central-bank research sweep for M5–M30 ideas.

Sources: arXiv q-fin queries (paced, retried on 429), recent Semantic Scholar
citations of papers that documented FX/gold forced flows, and RSS feeds from BIS,
the Federal Reserve (FEDS, IFDP), ECB and Bank of England working papers. Items are kept only
if the title/abstract mentions FX, gold, or dealer/liquidity terms.

Also supports SSRN manual-drop: paste a URL, tool fetches title/abstract via
read_url_content-style HTTP (no login required for SSRN abstract pages).
NBER blocks automated feed reads; drop NBER papers by hand.

Usage:
    python -m ats intake paper-monitor            # dry run, prints only
    python -m ats intake paper-monitor --save     # saves raw leads to intake.yaml
    python -m ats intake paper-monitor --ssrn <URL>  # manually drop one SSRN paper
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

from ats.config import DATA_DIR
from ats.ideas.dedup import check_dedup
from ats.ideas.intake_store import add_idea, list_rows

NS = {"a": "http://www.w3.org/2005/Atom"}
OUT = DATA_DIR / "ideas"

# ── M5–M30 targeted arXiv queries ────────────────────────────────────────────
# Curated for microstructure mechanisms that plausibly work at retail MT5 latency.
# Do not add queries that would return HFT/co-location literature — already screened.

M5_M30_QUERIES: list[str] = [
    'abs:intraday AND abs:liquidity AND (abs:forex OR abs:"foreign exchange")',
    'abs:"order flow" AND (abs:"foreign exchange" OR abs:currency)',
    "abs:inventory AND abs:dealer AND abs:FX",
    'abs:"mean reversion" AND abs:intraday AND (abs:currency OR abs:FX)',
    'abs:"lead-lag" AND (abs:currency OR abs:FX OR abs:gold)',
    'abs:announcement AND abs:drift AND abs:"exchange rate"',
    "abs:microstructure AND abs:gold",
    'abs:"stop loss" AND abs:liquidity',
    'abs:fixing AND (abs:"exchange rate" OR abs:"foreign exchange" OR abs:gold)',
    'abs:"hedging demand" AND (abs:currency OR abs:"exchange rate")',
    'abs:rebalancing AND (abs:currency OR abs:"exchange rate" OR abs:dollar)',
    '(abs:"option expiry" OR abs:"option expiration") AND (abs:currency OR abs:FX OR abs:gold)',
    'abs:"month-end" AND (abs:currency OR abs:dollar OR abs:"exchange rate")',
]

ARXIV_PAUSE_S = 3.0

# Papers that documented a forced flow in FX or gold. Recent papers citing them are
# likely to study the same flows (Semantic Scholar, free API, no key).
S2_SEEDS: tuple[str, ...] = (
    "Foreign Exchange Fixings and Returns Around the Clock",
    "Equity hedging and exchange rates at the London 4 p.m. fix",
    "Stop-loss orders and price cascades in currency markets",
    "Currency orders and exchange rate dynamics: an explanation for the predictive success of technical analysis",
    "Segmentation and time-of-day patterns in foreign exchange markets",
    "Micro effects of macro announcements: real-time price discovery in foreign exchange",
    "Puzzles in the Tokyo fixing in the forex market: order imbalances and biased pricing",
)
S2_API = "https://api.semanticscholar.org/graph/v1"
S2_RECENT_YEARS = 3

INSTITUTIONAL_FEEDS: dict[str, str] = {
    "BIS working papers": "https://www.bis.org/doclist/wppubls.rss",
    "BIS research hub": "https://www.bis.org/doclist/reshub_papers.rss",
    "Fed FEDS": "https://www.federalreserve.gov/feeds/feds.xml",
    "Fed IFDP": "https://www.federalreserve.gov/feeds/ifdp.xml",
    "ECB working papers": "https://www.ecb.europa.eu/rss/wppub.html",
    "Bank of England": "https://www.bankofengland.co.uk/rss/publications",
}

RELEVANCE_TERMS: tuple[str, ...] = (
    "exchange rate", "foreign exchange", " fx ", "fx market", "currenc", "dollar",
    "gold", "bullion", "dealer", "order flow", "intraday", "market mak", "liquidity",
    "fixing", "benchmark rate", "swap line", "carry trade", "settlement",
)


def _get(url: str, accept: str = "*/*", timeout: int = 30) -> bytes:
    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 proofbook-intake", "Accept": accept}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _get_with_backoff(url: str, tries: int = 3, wait_s: float = 10.0) -> bytes | None:
    """GET with retries on HTTP 429/503 (arXiv and Semantic Scholar rate limits)."""
    import time
    import urllib.error

    for attempt in range(tries):
        try:
            return _get(url)
        except urllib.error.HTTPError as exc:
            if exc.code not in {429, 503} or attempt == tries - 1:
                return None
            time.sleep(wait_s * (attempt + 1))
        except Exception:
            return None
    return None


def _fetch_s2_citations(seed_title: str, limit: int = 50) -> list[dict]:
    """Recent papers citing one seed paper, via the Semantic Scholar graph API."""
    import time

    match = _get_with_backoff(f"{S2_API}/paper/search/match?" + urllib.parse.urlencode(
        {"query": seed_title, "fields": "paperId,title"}))
    data = json.loads(match).get("data") if match else None
    if not data:
        print(f"  [WARN] Semantic Scholar: seed not found: {seed_title[:50]}")
        return []
    time.sleep(1.5)
    raw = _get_with_backoff(f"{S2_API}/paper/{data[0]['paperId']}/citations?" + urllib.parse.urlencode(
        {"fields": "title,abstract,year,url,externalIds", "limit": limit}))
    if not raw:
        return []
    min_year = datetime.now(timezone.utc).year - S2_RECENT_YEARS
    rows = []
    for item in json.loads(raw).get("data") or []:
        p = item.get("citingPaper") or {}
        if not p.get("title") or (p.get("year") or 0) < min_year:
            continue
        doi = (p.get("externalIds") or {}).get("DOI")
        rows.append({
            "title": " ".join(p["title"].split()),
            "summary": " ".join((p.get("abstract") or "").split())[:500],
            "link": f"https://doi.org/{doi}" if doi else (p.get("url") or ""),
            "published": str(p.get("year") or ""),
            "feed": "Semantic Scholar citations",
        })
    return rows


def is_relevant(title: str, summary: str) -> bool:
    text = f" {title} {summary} ".lower()
    return any(term in text for term in RELEVANCE_TERMS)


# ── arXiv fetch ───────────────────────────────────────────────────────────────

def _fetch_arxiv(query: str, max_results: int = 6) -> list[dict]:
    """Fetch papers from arXiv for a single query string."""
    params = {
        "search_query": f"cat:q-fin* AND ({query})",
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(params)
    xml_bytes = _get_with_backoff(url)
    if xml_bytes is None:
        print(f"  [WARN] arXiv fetch failed for query '{query[:50]}'")
        return []

    root = ET.fromstring(xml_bytes)
    rows: list[dict] = []
    for entry in root.findall("a:entry", NS):
        title = (entry.findtext("a:title", default="", namespaces=NS) or "").strip()
        summary = (entry.findtext("a:summary", default="", namespaces=NS) or "").strip()
        link = ""
        for lnk in entry.findall("a:link", NS):
            if lnk.get("type") == "text/html" or lnk.get("rel") == "alternate":
                link = lnk.get("href") or link
        published = (entry.findtext("a:published", default="", namespaces=NS) or "")[:10]
        rows.append({
            "title": " ".join(title.split()),
            "summary": " ".join(summary.split())[:500],
            "link": link,
            "published": published,
        })
    return rows


# ── Institutional RSS feeds ──────────────────────────────────────────────────

def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def parse_feed(xml_bytes: bytes, source: str) -> list[dict]:
    """Parse RSS 2.0, RSS 1.0 (RDF) or Atom into title/summary/link/published."""
    import re

    root = ET.fromstring(xml_bytes)
    rows: list[dict] = []
    for node in root.iter():
        if _local(node.tag) not in {"item", "entry"}:
            continue
        fields: dict[str, str] = {}
        for child in node:
            name = _local(child.tag)
            if name == "link" and not (child.text or "").strip():
                fields.setdefault("link", child.get("href") or "")
            elif name in {"title", "link", "description", "summary", "pubdate", "date", "published", "updated"}:
                fields.setdefault(name, (child.text or "").strip())
        summary = fields.get("description") or fields.get("summary") or ""
        summary = re.sub(r"<[^>]+>", " ", summary)
        rows.append({
            "title": " ".join(fields.get("title", "").split()),
            "summary": " ".join(summary.split())[:500],
            "link": fields.get("link", ""),
            "published": (fields.get("published") or fields.get("date") or fields.get("pubdate")
                          or fields.get("updated") or "")[:32],
            "feed": source,
        })
    return rows


def _fetch_feed(name: str, url: str) -> list[dict]:
    try:
        return parse_feed(_get(url), name)
    except Exception as exc:
        print(f"  [WARN] feed fetch failed for {name}: {exc}")
        return []


# ── SSRN manual-drop ─────────────────────────────────────────────────────────

def _fetch_ssrn(url: str) -> dict | None:
    """Fetch title and abstract from an SSRN abstract page (no login needed)."""
    try:
        html = _get(url, accept="text/html").decode("utf-8", errors="replace")
    except Exception as exc:
        print(f"  [WARN] SSRN fetch failed for {url}: {exc}")
        return None

    # Extract title from <title> tag (usually "SSRN - Paper Title")
    import re
    title_match = re.search(r"<title[^>]*>([^<]+)</title>", html, re.IGNORECASE)
    title = ""
    if title_match:
        raw_title = title_match.group(1).strip()
        title = re.sub(r"^SSRN\s*[-–—]\s*", "", raw_title).strip()

    # Extract abstract from meta description or og:description
    abstract = ""
    meta_match = re.search(
        r'<meta[^>]+name=["\']description["\'][^>]+content=["\']([^"\']+)["\']',
        html,
        re.IGNORECASE,
    )
    if meta_match:
        abstract = meta_match.group(1).strip()
    if not abstract:
        og_match = re.search(
            r'<meta[^>]+property=["\']og:description["\'][^>]+content=["\']([^"\']+)["\']',
            html,
            re.IGNORECASE,
        )
        if og_match:
            abstract = og_match.group(1).strip()

    if not title and not abstract:
        print(f"  [WARN] Could not extract title/abstract from {url}")
        return None

    return {"title": title, "summary": abstract[:500], "link": url, "published": ""}


# ── Main runner ───────────────────────────────────────────────────────────────

def run_paper_monitor(
    max_per_query: int = 6,
    save: bool = True,
    ssrn_url: str | None = None,
) -> int:
    """Run all M5–M30 arXiv queries (+ optional SSRN drop) and add raw leads.

    Args:
        max_per_query: Papers to fetch per arXiv query.
        save:          If True, write raw records to intake.yaml.
        ssrn_url:      If provided, fetch this specific SSRN paper instead of running the full arXiv sweep.

    Returns:
        Number of papers processed.
    """
    OUT.mkdir(parents=True, exist_ok=True)

    if ssrn_url:
        paper = _fetch_ssrn(ssrn_url)
        papers = [paper] if paper else []
        print(f"SSRN drop: {'found' if paper else 'failed'} — {ssrn_url}")
    else:
        import time

        print(f"Paper monitor: running {len(M5_M30_QUERIES)} arXiv queries...")
        papers: list[dict] = []
        for q in M5_M30_QUERIES:
            batch = _fetch_arxiv(q, max_results=max_per_query)
            papers.extend(batch)
            print(f"  query: {q[:60]!r} -> {len(batch)} papers")
            time.sleep(ARXIV_PAUSE_S)
        print(f"Semantic Scholar: recent citations of {len(S2_SEEDS)} forced-flow papers...")
        for seed in S2_SEEDS:
            batch = _fetch_s2_citations(seed)
            papers.extend(batch)
            print(f"  cites {seed[:55]!r} -> {len(batch)} recent")
            time.sleep(1.5)
        print(f"Institutional feeds: {len(INSTITUTIONAL_FEEDS)} sources...")
        for name, url in INSTITUTIONAL_FEEDS.items():
            batch = _fetch_feed(name, url)
            papers.extend(batch)
            print(f"  feed: {name} -> {len(batch)} items")
        before = len(papers)
        papers = [p for p in papers if is_relevant(p.get("title", ""), p.get("summary", ""))]
        print(f"  relevance filter kept {len(papers)}/{before}")

    # Deduplicate by URL within this run
    seen_links: set[str] = set()
    unique_papers: list[dict] = []
    for p in papers:
        link = p.get("link", "")
        if link and link not in seen_links:
            seen_links.add(link)
            unique_papers.append(p)

    # Save raw audit file
    run_record = {
        "fetched_utc": datetime.now(timezone.utc).isoformat(),
        "papers": unique_papers,
    }
    (OUT / "paper_monitor_run.json").write_text(
        json.dumps(run_record, indent=2), encoding="utf-8"
    )
    print(f"  {len(unique_papers)} unique papers found (saved to data/ideas/paper_monitor_run.json)")

    if not save:
        print("[DRY RUN] --save not set. Re-run with --save to add to intake.yaml.")
        return len(unique_papers)

    known = {(r.get("source_ref") or "").strip() for r in list_rows()}
    unique_papers = [p for p in unique_papers if p.get("link", "") not in known]
    added = 0
    for p in unique_papers:
        dedup = check_dedup(p.get("summary", ""), p.get("title", ""))
        iid = add_idea({
            "source": "institutional" if p.get("feed") else "paper",
            "feed": p.get("feed", "arXiv"),
            "source_ref": p.get("link", ""),
            "name": p.get("title", "")[:80],
            "title": p.get("title", ""),
            "summary_excerpt": p.get("summary", "")[:300],
            "published": p.get("published", ""),
            "causal_category": "",          # owner fills at review
            "causal_actor": "",             # REQUIRED before promotion
            "timeframe_tier": "",           # owner assigns at review
            "instrument": "",
            "timeframe": "",
            "expected_holding_bars": None,
            "expected_trades_month": None,
            "dedup_check": dedup,
            "status": "raw",
        })
        added += 1
        dedup_note = f" [dedup: {dedup[:2]}]" if dedup else ""
        print(f"  Added {iid}{dedup_note}: {p.get('title', '')[:65]}")

    print(f"\nPaper monitor done: {added} leads added to intake.yaml with status=raw.")
    print("Next: python -m ats intake list --status raw  ->  review and assign tier/causal_actor.")
    return added
