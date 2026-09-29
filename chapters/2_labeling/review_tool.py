"""Chapter 2 · The blind human-review tool used to create the ground-truth labels.

For each project the tool opens an interactive branch-lifetime timeline (with merge
events) in the browser and shows the long-lived-branch and feeder-branch analysis. The
reviewer presses 1-4 to choose a strategy. Only *after* the choice is recorded does the
tool reveal the automated rule's label, so the human judgement is independent of it.

This is how data/derived/travis_project_labels.csv (column strategy_human) was produced.
Progress is saved after every project; closing and re-opening resumes where you left off.

Usage : python review_tool.py            (requires a display; uses tkinter + plotly)
Output: results/2_labeling/review_progress.csv
"""
import os
import sys
import webbrowser
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import tkinter as tk
from tkinter import ttk

from bsci import travis, labeling
from bsci.labeling import _commit_graph
from bsci.paths import results_dir

KEYS = {"1": "One Branch", "2": "Trunk-based", "3": "Feature Branching", "4": "Git Flow"}
COLS = ["gh_project_name", "git_branch", "gh_build_started_at", "git_all_built_commits", "git_trigger_commit"]


class Review:
    def __init__(self, root, builds, threshold):
        self.root, self.builds, self.threshold = root, builds, threshold
        self.out = results_dir("2_labeling") / "review_progress.csv"
        self.done = pd.read_csv(self.out) if self.out.exists() else pd.DataFrame()
        done = set(self.done.get("project", []))
        self.queue = [p for p in sorted(builds["gh_project_name"].unique()) if p not in done]
        self.info = ttk.Label(root, font=("Consolas", 10), justify=tk.LEFT)
        self.info.pack(padx=12, pady=10, fill="x")
        self.reveal = ttk.Label(root, font=("Segoe UI", 11))
        self.reveal.pack(pady=6)
        bar = ttk.Frame(root); bar.pack(pady=8)
        for k, name in KEYS.items():
            ttk.Button(bar, text=f"[{k}] {name}", command=lambda n=name: self.choose(n)).pack(side=tk.LEFT, padx=4)
            root.bind(f"<Key-{k}>", lambda e, n=name: self.choose(n))
        ttk.Button(bar, text="Next project →", command=self.next).pack(side=tk.LEFT, padx=16)
        self.next()

    def next(self):
        if not self.queue:
            self.info.config(text="All projects reviewed.")
            return
        self.project = self.queue.pop(0)
        p = self.builds[self.builds.gh_project_name == self.project]
        self.metrics = labeling.travis_project_metrics(p, self.threshold)
        m = self.metrics
        self.info.config(text=(f"{self.project}\n\nlongest branch: {m['l1_lifetime_days']:.1f} days\n"
                               f"long-lived branches (ratio > {self.threshold:.2f} of longest): {m['num_long_lived']}\n"
                               f"feeder branches merging into the main line: {m['num_feeder_branches']}\n"
                               f"median feeder-branch lifetime: {m['median_feeder_lifetime_days']:.1f} days"))
        self.reveal.config(text="Choose a strategy (1-4). The automated label is shown afterwards.")
        self.timeline(p)

    def timeline(self, p):
        life = p.groupby("git_branch")["gh_build_started_at"].agg(["min", "max"]).reset_index()
        fig = px.timeline(life, x_start="min", x_end="max", y="git_branch", title=f"Branch timeline — {self.project}")
        g = _commit_graph(p)
        merges = [(g.nodes[u]["branch"], g.nodes[v]["branch"], g.nodes[v].get("time")) for u, v in g.edges()
                  if "branch" in g.nodes[u] and "branch" in g.nodes[v] and g.nodes[u]["branch"] != g.nodes[v]["branch"]]
        merges = pd.DataFrame(merges, columns=["source", "target", "time"]).dropna()
        if not merges.empty:
            fig.add_trace(go.Scatter(x=merges["time"], y=merges["target"], mode="markers", name="merge",
                                     marker=dict(symbol="circle-open", size=8), customdata=merges[["source"]],
                                     hovertemplate="merge from %{customdata[0]}<br>into %{y}<extra></extra>"))
        fig.update_layout(height=max(600, 18 * len(life)))
        path = results_dir("2_labeling") / "timelines" / f"{self.project.replace('/', '_')}.html"
        path.parent.mkdir(exist_ok=True)
        fig.write_html(path)
        webbrowser.open(path.resolve().as_uri())

    def choose(self, strategy):
        m = self.metrics
        rule = labeling.travis_rule(m["num_long_lived"], m["num_feeder_branches"], m["median_feeder_lifetime_days"])
        row = {"project": self.project, "strategy_human": strategy, "strategy_rule": rule, **m}
        self.done = pd.concat([self.done[self.done.get("project", pd.Series(dtype=str)) != self.project]
                               if len(self.done) else self.done, pd.DataFrame([row])], ignore_index=True)
        self.done.to_csv(self.out, index=False)
        self.reveal.config(text=f"You chose: {strategy}    |    automated rule: {rule}")


def main():
    builds = travis.read_raw(COLS)
    builds["gh_build_started_at"] = pd.to_datetime(builds["gh_build_started_at"], errors="coerce")
    threshold = labeling.travis_significance_threshold(builds)
    root = tk.Tk()
    root.title("Branching-strategy review (blind)")
    Review(root, builds, threshold)
    root.mainloop()


if __name__ == "__main__":
    main()
