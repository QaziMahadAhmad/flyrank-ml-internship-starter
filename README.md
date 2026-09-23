# FlyRank Content Opportunity Scoring

## Ranking Content Pages for SEO Review Using Position-Relative Click-Through Rate

A machine learning system for helping SEO and content teams prioritize pages for review using search-performance signals.

This project was developed as the capstone for the FlyRank ML Internship. It investigates whether position-relative click-through rate (CTR) can provide a useful signal for ranking content pages for SEO review.

The system is designed as **decision support for human review**, not as an automated SEO decision maker or a causal model of Google's ranking system.

---

## 1. Problem

SEO and content teams may have many pages that could potentially benefit from review or optimization. Reviewing every page manually is inefficient, so a practical question is:

> Can search-performance signals be used to rank content pages so that the pages most deserving of review can be prioritized first?

This project focuses on **position-relative CTR**.

A page's observed CTR depends strongly on where it appears in search results. Therefore, comparing a page's CTR with the CTR typically observed at its search position provides a more useful signal than looking at raw CTR alone.

The project compares:

1. A transparent CTR-gap baseline.
2. A machine-learning ranking model using 21 pre-decision features.

---

## 2. Who This Is For

The project is intended for:

- SEO and content teams
- Content strategists
- Search-performance analysts
- ML engineers working on ranking and prioritization systems
- Reviewers interested in reproducible applied ML workflows

The intended workflow is:

```text
Search-performance data
        ↓
Feature engineering
        ↓
Position-relative CTR signal
        ↓
ML ranking model
        ↓
Held-out evaluation
        ↓
Prioritized content pages
        ↓
Human SEO review
