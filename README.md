# FlyRank Content Opportunity Scoring

## Ranking Content Pages for SEO Review Using Position-Relative Click-Through Rate

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Scikit--learn](https://img.shields.io/badge/scikit--learn-ML-F7931E.svg)](https://scikit-learn.org/)
[![FlyRank](https://img.shields.io/badge/FlyRank-ML%20Internship-111827.svg)](https://flyrank.ai/)

> An ML-based decision-support system for ranking content pages that may deserve SEO review, using position-relative click-through-rate signals and client-grouped validation.

---

## Overview

This project was developed as part of the **FlyRank ML Internship** and addresses a content opportunity scoring problem:

**How can search-performance signals be used to prioritize content pages for SEO review?**

The project investigates whether a page's observed click-through rate (CTR), relative to the CTR expected at its search position, can provide a useful signal for identifying content that may deserve review.

The final system combines:

- A transparent position-relative CTR baseline
- Feature engineering from search-performance data
- A `HistGradientBoostingRegressor` machine-learning model
- 5-fold `GroupKFold` validation grouped by client
- A top-50 ranking evaluation
- An action-oriented recommendation layer

The system is designed as **decision support**, not as an automated replacement for SEO judgment.

---

## Who Is This For?

This project is intended for:

- SEO teams prioritizing pages for content review
- Content strategists deciding which pages deserve attention
- Search-performance analysts
- ML engineers working on ranking and prioritization problems
- Researchers evaluating ML systems on grouped observational data
- Teams looking for transparent, reproducible content-scoring workflows

---

## Problem Statement

SEO teams can have thousands of content pages competing for limited optimization time.

A simple approach is to inspect pages based on metrics such as:

- Search position
- Click-through rate
- Impressions
- Clicks
- Historical CTR
- Content metadata

However, raw CTR is strongly influenced by search position.

A page ranking near position 1 naturally has a much higher opportunity to receive clicks than a page ranking near position 50.

Therefore, comparing pages using raw CTR alone can produce misleading priorities.

### Research Question

> Can position-relative CTR signals and machine learning be used to rank content pages for SEO review more effectively than a transparent rule-based baseline?

---

# Approach

The project follows a research-to-model-to-evaluation workflow.

```text
FlyRank Search Warehouse
          |
          v
Data Filtering & Validation
          |
          v
Feature Engineering
          |
          v
Signal Audit
          |
          +----------------------+
          |                      |
          v                      v
Transparent Baseline      ML Ranking Model
          |                      |
          |              HistGradientBoosting
          |                      |
          +----------+-----------+
                     |
                     v
        Client-Grouped Validation
             5-Fold GroupKFold
                     |
                     v
             Top-50 Evaluation
                     |
                     v
       Content Review Recommendations
