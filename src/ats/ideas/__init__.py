from ats.ideas.arxiv import print_arxiv, search_arxiv
from ats.ideas.blogs import print_blogs
from ats.ideas.cot import gold_series, pull_cot
from ats.ideas.cross_market import snapshot
from ats.ideas.inbox import list_rows, print_inbox
from ats.ideas.sentiment import print_sentiment

# ── Intake pipeline (upstream of lab) ────────────────────────────────────────
from ats.ideas.mechanism_library import load_library, print_library
from ats.ideas.intake_store import (
    add_idea,
    get_idea,
    list_rows as intake_list_rows,
    print_intake,
    update_status,
)
from ats.ideas.dedup import check_dedup, print_dedup_report
from ats.ideas.paper_monitor import run_paper_monitor
from ats.ideas.groq_brainstorm import run_groq_brainstorm, run_groq_brainstorm_all
from ats.ideas.promoter import promote
from ats.ideas.ka04_log import log_trade
from ats.ideas.prompts import CATEGORIES, _PROMPTS

__all__ = [
    # Legacy ideas inbox
    "print_arxiv",
    "search_arxiv",
    "print_blogs",
    "gold_series",
    "pull_cot",
    "snapshot",
    "list_rows",
    "print_inbox",
    "print_sentiment",
    # Intake pipeline
    "load_library",
    "print_library",
    "add_idea",
    "get_idea",
    "intake_list_rows",
    "print_intake",
    "update_status",
    "check_dedup",
    "print_dedup_report",
    "run_paper_monitor",
    "run_groq_brainstorm",
    "run_groq_brainstorm_all",
    "promote",
    "log_trade",
    "CATEGORIES",
    "_PROMPTS",
]
