"""IDSSE Dataset Analysis Package."""

__version__ = "1.0.0"

# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
from .download import download_match, download_all_matches, validate_downloads

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
from .parser import (
    parse_events, parse_codes, parse_teamsheets, parse_team_names,
    parse_tracking, parse_ball_tracking, parse_all_ball_tracking,
    parse_match, parse_all_matches,
)

# ---------------------------------------------------------------------------
# NPT analysis
# ---------------------------------------------------------------------------
from .npt_analysis import (
    compute_npt_frame,
    compute_npt_interval,
    load_official_npt,
    compare_npt_methods,
    npt_summary_all,
    build_episode_table,
    time_loss_by_restart,
    cause_by_restart,
    infer_cause_from_restart,
    infer_cause_from_ball,
    infer_cause,
    fmt_mmss,
    fmt_seconds_cols,
)

# ---------------------------------------------------------------------------
# Visualisation (optional — package stays importable without matplotlib)
# ---------------------------------------------------------------------------
try:
    from .visualization import (
        draw_pitch,
        plot_pass_map,
        plot_shot_chart,
        plot_heatmap,
        plot_possession_timeline,
        plot_restart_map,
    )
except ModuleNotFoundError:
    pass
