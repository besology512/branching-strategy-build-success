# What this study can and cannot tell you

A reference repository should make the weak points as easy to find as the results. This
page collects them in one place; each chapter links here where it matters.

## About the question

**Association, not causation.** Every result is observational. Projects choose their
branching strategy for reasons (team size, release obligations, maturity) that can also
affect their builds. Chapter 5 adjusts for the confounders we could measure; unmeasured
ones remain.

**Practices we cannot see.** Trunk-based development usually comes with feature flags,
dark launches and canary releases. Neither dataset records them, so any advantage of
short-lived branches includes whatever those accompanying practices contribute.

**Build success is a proxy.** A passing build says the test suite passed. Nothing more:
not that the software is defect-free, maintainable, or good after release. Build
outcomes are also noisy (Gallaba et al., ASE 2018).

## About the data

**Two corpora, not the whole world.** TravisTorrent (2011–2016; Ruby, Python, Java, Go)
and GHALogs (2023; mostly Python, TypeScript, Go) were selected by popularity and CI
activity. Small, private and industrial projects are not represented.

**Few Git Flow projects on TravisTorrent.** Only 22 of 948 projects are Git Flow, and
three of them contribute 80.6 % of all Git Flow builds. Build-weighted pass rates
(Table 8) therefore describe those projects more than a typical Git Flow project; the
project-level comparison in chapter 5 shows no difference on TravisTorrent. GHALogs, with
683 Git Flow repositories, gives the more reliable estimate of the penalty.

**The feature-branching group on GHALogs is tiny** (56 repositories, 1,202 runs), so its
pass rate is indicative only.

## About the labels

**Feeder branches are measured differently on the two datasets.** On TravisTorrent,
feeder branches are branches whose commits are merged into the main line, traced through
the commit graph. GHALogs does not record parent commits, so every branch that is not
long-lived counts as a feeder branch there. The strategy rules are identical; the
measurement of one input is not. See chapter 2.

**Validation covers TravisTorrent only.** The blind human labels (97.5 % agreement,
κ = 0.96) validate the TravisTorrent labeller. The GHALogs labels were not validated
against human judgement.

**Branch lifecycle statistics are pooled by branch name.** The descriptive lifecycle
distribution (long- vs short-lived branches, §V-C of the paper) aggregates branches by
name across projects, as in the original analysis. The strategy labels are computed per
project and do not depend on it.

## About the models

**SMOTE before the split (TravisTorrent).** The four strategies were balanced with SMOTE
on the full table, before the 80/20 split. The test set therefore contains synthetic
builds interpolated from real ones, some of whose neighbours are in the training set.
This favours flexible models (an unlimited-depth Random Forest reaches AUC 0.976) and
makes the TravisTorrent scores in Tables 3, 6 and 9 optimistic. The GHALogs models use
real runs only and are not affected. Comparisons *between* feature sets and models are
made on the same split and remain informative; the absolute TravisTorrent scores should
be read as upper bounds.

**Hyperparameters were chosen on the evaluation split.** The sensitivity curves in
chapter 6 were computed on the same held-out split that Table 3 reports; there was no
separate validation split.

**Near-outcome features.** The number of failed tests is, unsurprisingly, the strongest
predictor of a failed build. It is kept because it is a standard CI-prediction feature,
but it means the models describe *which information co-occurs with* failure, not what
predicts it before a build runs.

**Table 6 uses a lighter configuration.** The cross-validation and Friedman/Nemenyi tests
used 100 trees (Random Forest depth 12, AdaBoost learning rate 0.5) and a 300,000-row
TravisTorrent sample, lighter than the Table 5 settings. `chapters/3_prediction/significance.py`
reproduces it as it was run.

**CPU vs GPU.** The published Tables 3–4 trained XGBoost and CatBoost on a GPU; this
repository runs everything on the CPU, which changes those two models' scores in the
third decimal.

**Library versions.** The results in this repository were produced with the versions in
`requirements.txt` (scikit-learn 1.7.1, NumPy 2.3.2). The paper's Table 2 lists
scikit-learn 1.3.2 and NumPy 1.26.4. AdaBoost is the one model this matters for:
scikit-learn 1.6 removed its SAMME.R algorithm (the default up to 1.3), and the
cross-validated AdaBoost scores come out 0.02–0.04 AUC lower here than in Table 6.
Its rank does not change.
