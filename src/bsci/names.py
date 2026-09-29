"""Readable names for dataset identifiers, used in every figure and table."""

FEATURE_LABELS = {
    # TravisTorrent — standard
    "tr_log_num_tests_failed": "Failed tests", "tr_duration": "Build duration", "gh_repo_age": "Repository age",
    "gh_team_size": "Team size", "tr_log_num_tests_ok": "Passed tests", "gh_sloc": "Source lines of code",
    "tr_log_num_tests_run": "Tests run", "tr_log_testduration": "Test duration",
    "gh_num_commits_on_files_touched": "Prior commits on touched files", "gh_is_pr": "Pull-request build",
    "gh_diff_files_modified": "Files modified", "git_diff_src_churn": "Source-code churn",
    "tr_log_num_tests_skipped": "Skipped tests", "gh_by_core_team_member": "Core-team author",
    "gh_diff_files_added": "Files added", "gh_diff_files_deleted": "Files deleted", "git_diff_test_churn": "Test-code churn",
    "build_language": "Build language",
    # TravisTorrent — branching
    "num_feeder_branches": "Feeder-branch count", "median_feeder_lifetime_days": "Median feeder-branch lifetime",
    "num_long_lived": "Long-lived branch count",
    # GHALogs — standard
    "trigger_event": "Trigger event", "duration_sec": "Run duration", "logs_size": "Log size",
    "run_number": "Run number", "is_pr": "Pull-request run", "run_attempt": "Run attempt",
    # GHALogs — branching
    "max_branch_life_sec": "Maximum branch lifetime", "avg_branch_life_sec": "Mean branch lifetime",
    "std_branch_life_sec": "Branch-lifetime variability", "min_branch_life_sec": "Minimum branch lifetime",
    "feeder_count": "Feeder-branch count", "median_feeder_seconds": "Median feeder-branch lifetime",
    "long_lived_count": "Long-lived branch count",
    # both
    "strategy_label": "Branching-strategy label",
}


def label(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature.replace("_", " ").capitalize())
