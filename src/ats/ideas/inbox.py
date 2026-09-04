"""Idea inbox. A lead is not a test and not an edge."""

from __future__ import annotations

import yaml

from ats.config import CONFIG_DIR

IDEAS_PATH = CONFIG_DIR / "ideas.yaml"


def load_ideas() -> dict:
    if not IDEAS_PATH.exists():
        return {"protocol": "", "ideas": []}
    with IDEAS_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {"ideas": []}


def save_ideas(blob: dict) -> None:
    with IDEAS_PATH.open("w", encoding="utf-8") as f:
        yaml.safe_dump(blob, f, sort_keys=False, allow_unicode=True)


def list_rows() -> list[dict]:
    return list(load_ideas().get("ideas") or [])


def upsert_lead(idea: dict) -> None:
    blob = load_ideas()
    ideas = list(blob.get("ideas") or [])
    iid = idea["id"]
    ideas = [x for x in ideas if x.get("id") != iid] + [idea]
    blob["ideas"] = ideas
    save_ideas(blob)


def print_inbox() -> None:
    rows = list_rows()
    print("Idea intake automates sourcing and frozen-spec discipline. It does not find an edge.")
    print("A lead needs a causal why, frozen params, then python -m ats test (OOS locked).")
    print(f"{'id':28} {'status':10} source")
    for r in rows:
        print(f"{r.get('id','')[:28]:28} {r.get('status',''):10} {r.get('source','')}")
        why = (r.get("why") or "").strip().replace("\n", " ")
        if why:
            print(f"  why: {why[:160]}")
