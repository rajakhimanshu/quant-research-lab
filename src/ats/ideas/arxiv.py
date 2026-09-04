"""arXiv q-fin search. Returns paper leads, not trade rules."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

from ats.config import DATA_DIR

NS = {"a": "http://www.w3.org/2005/Atom"}
OUT = DATA_DIR / "ideas"


def parse_arxiv_xml(xml: bytes) -> list[dict]:
    root = ET.fromstring(xml)
    rows = []
    for entry in root.findall("a:entry", NS):
        title = (entry.findtext("a:title", default="", namespaces=NS) or "").strip().replace("\n", " ")
        summary = (entry.findtext("a:summary", default="", namespaces=NS) or "").strip().replace("\n", " ")
        link = ""
        for l in entry.findall("a:link", NS):
            if l.get("type") == "text/html" or l.get("rel") == "alternate":
                link = l.get("href") or link
        rows.append(
            {
                "title": " ".join(title.split()),
                "summary": " ".join(summary.split())[:400],
                "link": link,
                "published": (entry.findtext("a:published", default="", namespaces=NS) or "")[:10],
            }
        )
    return rows


def search_arxiv(query: str, max_results: int = 8) -> list[dict]:
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "ats-research-lab/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        xml = resp.read()
    rows = parse_arxiv_xml(xml)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "arxiv_leads.json"
    path.write_text(
        json.dumps({"query": query, "fetched_utc": datetime.now(timezone.utc).isoformat(), "papers": rows}, indent=2),
        encoding="utf-8",
    )
    return rows


def save_arxiv_leads(rows: list[dict]) -> int:
    from ats.ideas.inbox import upsert_lead

    n = 0
    for p in rows:
        aid = (p.get("link") or "").rstrip("/").rsplit("/", 1)[-1] or f"{p.get('published')}_{n}"
        upsert_lead(
            {
                "id": f"LEAD_arxiv_{aid}",
                "status": "lead",
                "source": "academic",
                "title": p.get("title"),
                "url": p.get("link"),
                "published": p.get("published"),
                "why": "",
                "note": "Write a causal why and freeze params in hypotheses.yaml before python -m ats test.",
            }
        )
        n += 1
    return n


def print_arxiv(rows: list[dict], query: str) -> None:
    print(f"arXiv leads for: {query}")
    print("These are papers, not hypotheses. Pick one, write a why, freeze params, then test.")
    if not rows:
        print("No hits. Try --query all:forex or all:\"currency carry\"")
        return
    for i, p in enumerate(rows, 1):
        print(f"\n{i}. {p['title']} ({p['published']})")
        print(f"   {p['link']}")
        print(f"   {p['summary'][:220]}...")
