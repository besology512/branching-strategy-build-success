# Chapter 1 · The data: 4.5 million CI builds from two eras of continuous integration

> **Question.** Where do the builds come from, and how do you get them?

The study uses two public datasets. They differ on purpose: different CI services,
languages and years. A finding that holds in both is less likely to be an artefact of
one corpus.

|  | **TravisTorrent** | **GHALogs** |
|---|---|---|
| CI service | Travis CI | GitHub Actions |
| Period | builds from Apr 2011 to Aug 2016 | runs collected in 2023 |
| Unit | build job | workflow run (completed: success / failure) |
| Size used here | **3,881,992** builds · **948** projects | **573,993** runs · **28,658** repositories |
| Languages | Ruby (248 projects), Python (261), Java (241), Go (198) | 20 languages; Python, TypeScript, Go most common |
| Selection | non-fork, non-toy, >10 watchers, ≥50 Travis builds | >100 stars, ≥30 Actions runs in 90 days |
| Source | Beller, Gousios & Zaidman, MSR 2017 | Moriconi et al., MSR 2025 |
| Download | [figshare 19314170](https://figshare.com/articles/dataset/TravisTorrent/19314170) (`final-2017-01-25.csv`) | [Zenodo 10154920](https://zenodo.org/records/10154920) (`runs.json.gz`, `repositories.json.gz`) |

> The TravisTorrent paper describes an earlier snapshot (2.64 M builds, 1,359 Ruby and
> Java projects). This study uses the **final public release of 25 January 2017**, whose
> 948 projects also include Python and Go.

## Put the files here

```
data/raw/
├── travistorrent/final-2017-01-25.csv      # unzip the figshare download (~3.6 GB)
└── ghalogs/
    ├── runs.json.gz                        # ~1 GB; the 142 GB log archive is NOT needed
    └── repositories.json.gz                # only needed to rebuild the repository metadata
```

Or keep them anywhere and point to that folder with `BSCI_RAW_DIR`:

```bash
export BSCI_RAW_DIR=/path/to/raw          # PowerShell: $env:BSCI_RAW_DIR = "D:\path\to\raw"
```

## What is already in the repository

Everything the analyses produce *about* the projects (as opposed to the raw builds) is
small and versioned in [`data/derived/`](../../data/derived):

| File | Rows | What it is | Made by |
|---|---:|---|---|
| `travis_project_labels.csv` | 948 | strategy per project (blind human label + automated rule label) and the branching metrics behind it | [chapter 2](../2_labeling) |
| `ghalogs_branch_lifetimes.csv.gz` | 136,339 | first/last run of every repository branch | [chapter 2](../2_labeling) |
| `ghalogs_repo_strategies.csv` | 28,658 | strategy per repository and the branching metrics behind it | [chapter 2](../2_labeling) |
| `ghalogs_repo_metadata.csv` | 28,658 | main language, contributors, creation date (for the confounder analysis) | `bsci.ghalogs.build_repo_metadata` |

With these files, chapters 2 (validation), 5 and 6 need only the raw builds for pass/fail
outcomes. Chapters 3 and 4 rebuild the modelling tables from the raw files and cache
them in `data/cache/` (git-ignored).
