# Chapter 2 · From branches to branching strategies

> **Question.** Nobody writes "we use Git Flow" into their CI configuration. Can a
> project's branching strategy be recognised from its branch structure alone, and
> reliably enough to build on?
>
> **Short answer.** Yes. Three structural measurements and four rules reproduce a blind
> human reviewer's judgement for **97.5 %** of 948 projects (Cohen's **κ = 0.96**).

## The idea

Strategies differ in *shape*, not in name. We measure three things per project:

| Measurement | What it captures |
|---|---|
| **Long-lived branches** | how many foundational lines of development run in parallel (main, develop, release…) |
| **Feeder branches** | branches whose work flows into the main line |
| **Median feeder lifetime** | how long a typical unit of parallel work lives before it is integrated |

and apply four rules, in this order:

```text
no feeder branches                         →  One Branch
3 or more long-lived branches              →  Git Flow
feeder branches live < 7 days (median)     →  Trunk-based
feeder branches live ≥ 7 days (median)     →  Feature Branching
```

GitHub Flow (one long-lived branch, short feature branches) falls under trunk-based or
feature branching; GitLab Flow (extra environment/release branches) under Git Flow.
Chapter 6 tests how much the 7-day boundary matters.

## How each measurement is taken

**Branch lifetime** is the time between the first and the last build on a branch.

**Long-lived branches.** The longest-living branch is the main line (L1). Another branch
counts as long-lived when its lifetime is a large enough fraction of L1's. The fraction is
learned from the data, not picked by hand:

* TravisTorrent: among all projects' ratios of second-to-first longest branch, take the
  ones in the top quartile and use their first quartile → **0.59**.
* GHALogs: the 75th percentile of the same ratio → **0.19**.

**Feeder branches** are measured differently, because the datasets record different
things:

* **TravisTorrent** records the commits each build built, so we rebuild each project's
  **commit graph** and count the branches whose commits are merged into the main line.
* **GHALogs** records runs but not parent commits, so every branch that is not long-lived
  counts as a feeder branch.

## How much to trust the labels

A reviewer labelled all 948 TravisTorrent projects with a purpose-built tool
([`review_tool.py`](review_tool.py)). The tool shows the branch timeline and merge events,
records the reviewer's choice, and **only then** reveals the automated label.

![Agreement between automated and blind human labels](../../figures/labeling_agreement.png)

| | One Branch | Trunk-based | Feature Branching | Git Flow |
|---|---:|---:|---:|---:|
| projects (human label) | 397 | 447 | 82 | 22 |
| F1 of the automated rule | 0.977 | 0.981 | 0.976 | **0.800** |

The rule is least reliable for the rarest strategy, Git Flow. The downstream analyses use
the human labels.

## Is this reproducible?

Yes, exactly, from the public raw data:

| Check | Result |
|---|---|
| TravisTorrent rule labels and metrics recomputed from the raw release | **948 / 948** identical |
| GHALogs branch lifetimes recomputed from `runs.json.gz` | **136,339 / 136,339** identical |
| GHALogs strategies recomputed | **28,658 / 28,658** identical |

## How builds split into long- and short-lived branches (descriptive)

Before strategies, the paper describes the branches themselves. Rules applied in order
(name pattern → lifetime ≥ 30 days → PR-focused → short merge lifecycle → k-means):

| Branch class | Share of builds | Branch names |
|---|---:|---:|
| long-lived, by name pattern (master, develop, release-*…) | 78.8 % | 471 |
| long-lived, by lifetime ≥ 30 days | 10.9 % | 4,177 |
| short-lived, by clustering | 7.3 % | 55,894 |
| long-lived, by clustering | 2.9 % | 2,116 |
| short-lived, merge lifecycle / PR-focused | 0.1 % | 171 |

A few foundational branches carry most builds; many short-lived branches carry the rest.
(Statistics are pooled by branch name across projects, as in the original analysis.)

## Run it

| Script | Does | Time |
|---|---|---|
| [`validate_labels.py`](validate_labels.py) | agreement, κ, confusion matrix, figure | seconds |
| [`travis_classify.py`](travis_classify.py) | label all TravisTorrent projects from raw builds | ~25 min |
| [`ghalogs_classify.py`](ghalogs_classify.py) | branch lifetimes + labels for GHALogs (`--from-derived` to skip the raw scan) | ~15 min |
| [`travis_branch_lifecycle.py`](travis_branch_lifecycle.py) | long/short-lived branch distribution | ~5 min |
| [`review_tool.py`](review_tool.py) | the blind review GUI (needs a display) | — |

The rules live in [`src/bsci/labeling.py`](../../src/bsci/labeling.py).

**Paper:** Section V-C · **Limitations:** [measurement differs across datasets; GHALogs labels not human-validated](../../docs/limitations.md#about-the-labels)
