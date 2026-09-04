"""Methodology reading list. Not signals, not an API scrape."""

BLOGS = [
    (
        "QuantConnect research",
        "https://www.quantconnect.com/research",
        "Look for how they define treatment vs baseline, not ticker alerts.",
    ),
    (
        "QuantInsti blog",
        "https://blog.quantinsti.com/",
        "Use posts that spell out a testable claim and a hold period.",
    ),
    (
        "Robot Wealth",
        "https://robotwealth.com/blog/",
        "Process posts (research hygiene) are the lead. Entry recipes are not.",
    ),
]


def print_blogs() -> None:
    print("Methodology sources — if it tells you where to buy, skip it.")
    for name, url, note in BLOGS:
        print(f"- {name}: {url}")
        print(f"  {note}")
    print("Next: write a why, freeze params in hypotheses.yaml, then python -m ats test --id ...")
