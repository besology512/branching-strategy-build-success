"""Chapter 5 · Is it the strategy, or the kind of project that uses it?

Language, team size and project maturity may each go with both the chosen strategy and
the build outcome. Three complementary checks:

  1. Logistic regression of build success on strategy (reference: trunk-based),
     language, log2 team size and project age, with standard errors clustered by project
     (builds are nested in projects).
  2. The same model plus strategy x team-size interaction terms (joint Wald test).
  3. Project level: every project contributes its own pass rate once (Kruskal-Wallis
     across strategies; Mann-Whitney U and Cliff's delta for trunk-based vs Git Flow).
Also: pass rates within every language, team-size band and maturity band.

TravisTorrent: language, team size and repository age as recorded for each build.
GHALogs: main language, number of contributors and repository age at collection.

Paper: Section VI-C, Table 10; RQ3.
Output: results/5_strategy_and_success/confounders_*.csv, figures/gitflow_team_size.png
Time  : ~20 min
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from bsci import travis, ghalogs, style
from bsci.paths import results_dir, figure_path

REF = "Trunk-based"
ORDER = ["Trunk-based", "One Branch", "Feature Branching", "Git Flow"]


def travis_builds():
    df = travis.builds_with_labels(["tr_status", "gh_lang", "gh_team_size", "gh_repo_age"])
    return pd.DataFrame({"project": df["gh_project_name"], "strategy": df["human_label"],
                         "success": (df["tr_status"] == "passed").astype(int),
                         "language": df["gh_lang"].str.capitalize(),
                         "team_size": pd.to_numeric(df["gh_team_size"], errors="coerce"),
                         "age_years": pd.to_numeric(df["gh_repo_age"], errors="coerce") / 365.25})


def ghalogs_runs():
    runs = ghalogs.run_table()[["repository_name", "build_is_success"]]
    strat = ghalogs.read_strategies().assign(repository_name=lambda d: d.repository_name.str.lower())
    meta = ghalogs.read_repo_metadata()
    meta["age_years"] = (pd.Timestamp("2023-10-01", tz="UTC") - pd.to_datetime(meta["created_at"], utc=True)).dt.days / 365.25
    df = runs.merge(strat[["repository_name", "strategy"]], on="repository_name").merge(meta, on="repository_name", how="left")
    top = df.drop_duplicates("repository_name")["main_language"].value_counts().head(5).index
    return pd.DataFrame({"project": df["repository_name"], "strategy": df["strategy"],
                         "success": df["build_is_success"].astype(int),
                         "language": np.where(df["main_language"].isin(top), df["main_language"], "Other"),
                         "team_size": pd.to_numeric(df["contributors"], errors="coerce"),
                         "age_years": df["age_years"]})


def add_bands(df):
    qt = df["team_size"].quantile([1 / 3, 2 / 3]).round().tolist()
    qa = df["age_years"].quantile([1 / 3, 2 / 3]).round(1).tolist()
    df["team_band"] = pd.cut(df["team_size"], [-np.inf, *qt, np.inf],
                             labels=[f"small (≤{qt[0]:.0f})", f"medium ({qt[0]:.0f}–{qt[1]:.0f})", f"large (>{qt[1]:.0f})"])
    df["maturity_band"] = pd.cut(df["age_years"], [-np.inf, *qa, np.inf],
                                 labels=[f"young (≤{qa[0]} y)", f"mid ({qa[0]}–{qa[1]} y)", f"mature (>{qa[1]} y)"])
    return df


def regressions(ds, df):
    d = df.dropna(subset=["team_size", "age_years", "language"]).copy()
    d["log2_team"] = np.log2(d["team_size"].clip(lower=1))
    d["strategy"] = pd.Categorical(d["strategy"], categories=[s for s in ORDER if s in d["strategy"].unique()])
    groups = pd.factorize(d["project"])[0]
    base = f"success ~ C(strategy, Treatment('{REF}')) + C(language) + log2_team + age_years"
    fit = lambda f: smf.logit(f, d).fit(disp=0, cov_type="cluster", cov_kwds={"groups": groups}, maxiter=100)
    m1 = fit(base)
    m2 = fit(base + f" + C(strategy, Treatment('{REF}')):log2_team")
    inter = [p for p in m2.params.index if ":log2_team" in p]
    wald = m2.wald_test(", ".join(f"{p} = 0" for p in inter), scalar=True)
    ci = np.exp(m1.conf_int())
    coef = pd.DataFrame({"dataset": ds, "term": m1.params.index, "odds_ratio": np.exp(m1.params.values),
                         "ci_low": ci[0].values, "ci_high": ci[1].values, "p_value": m1.pvalues.values})
    info = {"dataset": ds, "builds": len(d), "projects": d["project"].nunique(),
            "interaction_chi2": float(wald.statistic), "interaction_df": len(inter), "interaction_p": float(wald.pvalue)}
    return coef, info


def cliffs_delta(x, y):
    x, y = np.asarray(x), np.asarray(y)
    return float(((x[:, None] > y[None, :]).sum() - (x[:, None] < y[None, :]).sum()) / (len(x) * len(y)))


def project_level(ds, df):
    p = df.groupby(["project", "strategy"])["success"].agg(rate="mean", builds="size").reset_index()
    rows = []
    for s in ORDER:
        g = p[p.strategy == s].sort_values("builds", ascending=False)
        rows.append({"dataset": ds, "strategy": s, "projects": len(g), "median_project_rate": g["rate"].median(),
                     "top3_share_of_builds": g["builds"].head(3).sum() / g["builds"].sum()})
    tbd, gf = p.loc[p.strategy == REF, "rate"], p.loc[p.strategy == "Git Flow", "rate"]
    kw = stats.kruskal(*[p.loc[p.strategy == s, "rate"] for s in ORDER])
    mw = stats.mannwhitneyu(tbd, gf, alternative="two-sided")
    test = {"dataset": ds, "kruskal_H": kw.statistic, "kruskal_p": kw.pvalue, "tbd_vs_gitflow_U": mw.statistic,
            "tbd_vs_gitflow_p": mw.pvalue, "cliffs_delta": cliffs_delta(tbd, gf)}
    return rows, test


def stratified(ds, df):
    out = []
    for by in ["language", "team_band", "maturity_band"]:
        g = df.groupby([by, "strategy"], observed=True).agg(builds=("success", "size"),
                                                           projects=("project", "nunique"),
                                                           success_rate=("success", "mean")).reset_index()
        out.append(g.rename(columns={by: "stratum"}).assign(dataset=ds, stratifier=by))
    return pd.concat(out)


def main():
    out = results_dir("5_strategy_and_success")
    coefs, infos, plevel, tests, strata = [], [], [], [], []
    for ds, loader in [("TravisTorrent", travis_builds), ("GHALogs", ghalogs_runs)]:
        df = add_bands(loader())
        c, i = regressions(ds, df)
        rows, t = project_level(ds, df)
        coefs.append(c); infos.append(i); plevel += rows; tests.append(t); strata.append(stratified(ds, df))
        print(c.round(4).to_string(index=False)); print(i); print(t)
    pd.concat(coefs).to_csv(out / "confounders_odds_ratios.csv", index=False)
    pd.DataFrame(infos).to_csv(out / "confounders_interaction.csv", index=False)
    pd.DataFrame(plevel).to_csv(out / "confounders_project_level.csv", index=False)
    pd.DataFrame(tests).to_csv(out / "confounders_project_level_tests.csv", index=False)
    s = pd.concat(strata)
    s.to_csv(out / "confounders_stratified_rates.csv", index=False)
    figure(s)


def figure(s=None):
    if s is None:
        s = pd.read_csv(results_dir("5_strategy_and_success") / "confounders_stratified_rates.csv")
    """The headline: Git Flow's penalty depends on team size."""
    style.use()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)
    for ax, ds in zip(axes, ["TravisTorrent", "GHALogs"]):
        d = s[(s.dataset == ds) & (s.stratifier == "team_band")]
        bands = list(dict.fromkeys(d["stratum"].astype(str)))
        series = {k: d[d.strategy == k].set_index(d[d.strategy == k]["stratum"].astype(str)).loc[bands]
                  for k in ["Trunk-based", "Git Flow"]}
        for strat, marker in [("Trunk-based", "o"), ("Git Flow", "D")]:
            r, other = series[strat], series["Git Flow" if strat == "Trunk-based" else "Trunk-based"]
            ax.plot(range(3), r["success_rate"] * 100, marker=marker, ms=7, lw=2,
                    color=style.STRATEGY_COLORS[strat], mec="white", label=strat)
            for x, (v, o) in enumerate(zip(r["success_rate"], other["success_rate"])):
                ax.annotate(f"{v:.0%}", (x, v * 100), textcoords="offset points",
                            xytext=(0, 9 if v >= o else -16), ha="center", fontsize=8.5, color=style.INK)
        gf_n = series["Git Flow"]["projects"].astype(int).tolist()
        ax.set_xticks(range(3), [f"{b}\n{n} Git Flow projects" for b, n in zip(bands, gf_n)])
        ax.set_xlim(-0.4, 2.4)
        ax.set_title(ds, loc="left")
        ax.set_xlabel("Team size (contributors), tertiles of builds")
        style.grid(ax, "y")
    axes[0].set_ylabel("Builds that pass (%)")
    axes[0].set_ylim(40, 95)
    axes[0].legend(loc="lower left")
    fig.suptitle("Git Flow's penalty grows with team size on TravisTorrent; on GHALogs it is small but constant",
                 x=0.02, ha="left", fontsize=12, fontweight="semibold")
    fig.tight_layout()
    style.save(fig, figure_path("gitflow_team_size.png"))


if __name__ == "__main__":
    main()
