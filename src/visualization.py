"""src/visualization.py — Football pitch drawing and match data visualisation."""

from __future__ import annotations

import ast
from typing import Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

# IDSSE coordinate system: x ∈ [0, 105], y ∈ [0, 68] (metres)
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _require_columns(df: pd.DataFrame, required: list[str], df_name: str) -> None:
    """Raise a clear error if required columns are missing."""
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{df_name} missing required columns: {missing}")


def _parse_qualifier(value) -> dict:
    """Parse qualifier payload into a dict; return {} on failure."""
    if isinstance(value, dict):
        return value
    if pd.isna(value):
        return {}
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            return parsed if isinstance(parsed, dict) else {}
        except (ValueError, SyntaxError):
            return {}
    return {}


def _fmt_mmss(seconds: float) -> str:
    """Format seconds as ``mm:ss``."""
    m, s = divmod(int(round(seconds)), 60)
    return f"{m}:{s:02d}"


def _mirror_events(df: pd.DataFrame, segment_col: str = "segment") -> pd.DataFrame:
    """Mirror first-half ``at_x/y`` and ``to_x/y`` so Home always attacks left → right."""
    df = df.copy()
    fh = df[segment_col] == "firstHalf"
    for col in ["at_x", "to_x"]:
        if col in df.columns:
            df.loc[fh, col] = PITCH_LENGTH - df.loc[fh, col]
    for col in ["at_y", "to_y"]:
        if col in df.columns:
            df.loc[fh, col] = PITCH_WIDTH - df.loc[fh, col]
    return df


def _mirror_tracking(df: pd.DataFrame, segment_col: str = "segment") -> pd.DataFrame:
    """Mirror first-half tracking coordinates (player ``p*_x``, ``p*_y``)."""
    df = df.copy()
    fh = df[segment_col] == "firstHalf"
    for col in df.columns:
        if col.endswith("_x"):
            df.loc[fh, col] = PITCH_LENGTH - df.loc[fh, col]
        elif col.endswith("_y"):
            df.loc[fh, col] = PITCH_WIDTH - df.loc[fh, col]
    return df


# ---------------------------------------------------------------------------
# Pitch drawing
# ---------------------------------------------------------------------------

def draw_pitch(ax: Axes, color: str = "white", linecolor: str = "#333333") -> Axes:
    """Draw a standard 105×68m pitch on *ax*."""
    ax.set_facecolor(color)
    ax.set_xlim(-2, PITCH_LENGTH + 2)
    ax.set_ylim(-2, PITCH_WIDTH + 2)
    ax.set_aspect("equal")
    ax.axis("off")

    lw = 1.2
    kw = dict(color=linecolor, linewidth=lw, zorder=1)

    # Outline & centre line
    ax.add_patch(mpatches.Rectangle((0, 0), PITCH_LENGTH, PITCH_WIDTH, fill=False, **kw))
    ax.plot([PITCH_LENGTH / 2] * 2, [0, PITCH_WIDTH], **kw)

    # Centre circle
    ax.add_patch(mpatches.Circle((PITCH_LENGTH / 2, PITCH_WIDTH / 2), 9.15, fill=False, **kw))
    ax.plot(PITCH_LENGTH / 2, PITCH_WIDTH / 2, "o", color=linecolor, markersize=2, zorder=2)

    # Penalty areas
    for x0, sign in [(0, 1), (PITCH_LENGTH, -1)]:
        ax.add_patch(mpatches.Rectangle(
            (x0, (PITCH_WIDTH - 40.32) / 2), sign * 16.5, 40.32, fill=False, **kw))
        ax.add_patch(mpatches.Rectangle(
            (x0, (PITCH_WIDTH - 18.32) / 2), sign * 5.5, 18.32, fill=False, **kw))
        ax.plot(x0 + sign * 11, PITCH_WIDTH / 2, "o", color=linecolor, markersize=2, zorder=2)
        theta = np.linspace(
            np.arccos((16.5 - 11) / 9.15) * np.sign(sign),
            -np.arccos((16.5 - 11) / 9.15) * np.sign(sign), 100)
        ax.plot(x0 + sign * 11 + 9.15 * np.cos(theta),
                PITCH_WIDTH / 2 + 9.15 * np.sin(theta), **kw)

    return ax


# ---------------------------------------------------------------------------
# 1. Event-based: Pass Map
# ---------------------------------------------------------------------------

def plot_pass_map(
    events_df: pd.DataFrame,
    team: Optional[str] = None,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    team_names: Optional[dict[str, str]] = None,
) -> Figure:
    """Pass map coloured by outcome (green = success, red = fail).

    Parameters
    ----------
    team_names : dict, optional
        ``{"Home": "Team A", "Away": "Team B"}`` for display labels.
    """
    _require_columns(
        events_df, ["eID", "team", "at_x", "at_y", "to_x", "to_y"], "events_df"
    )

    # Mirror first-half so Home always attacks left → right
    events_df = _mirror_events(events_df)

    # Filter pass/cross events
    df = events_df[events_df["eID"].str.contains("Pass|Cross", na=False)].copy()
    if team:
        df = df[df["team"] == team]
    df = df.dropna(subset=["at_x", "at_y", "to_x", "to_y"])

    # Determine pass success from qualifier
    qualifiers = df["qualifier"] if "qualifier" in df.columns else pd.Series(None, index=df.index)

    def is_successful(q):
        qdict = _parse_qualifier(q)
        eval_text = str(qdict.get("Evaluation", "")).strip().lower()
        if not eval_text:
            return True          # default to success when qualifier absent
        return eval_text.startswith("success")

    df["success"] = qualifiers.apply(is_successful) if len(df) else pd.Series(dtype=bool)

    # Create figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 7.5))
    else:
        fig = ax.figure

    draw_pitch(ax, color="#f8f8f8")

    # Draw arrows
    for _, row in df.iterrows():
        color = "#2ecc71" if row["success"] else "#e74c3c"
        alpha = 0.7 if row["success"] else 0.5
        ax.annotate(
            "", xy=(row["to_x"], row["to_y"]), xytext=(row["at_x"], row["at_y"]),
            arrowprops=dict(arrowstyle="->", color=color, alpha=alpha, lw=0.8),
            zorder=3
        )

    # Title with pass stats
    n_success = df["success"].sum()
    n_total = len(df)
    pass_pct = (n_success / n_total * 100) if n_total else 0
    display_name = (team_names or {}).get(team, team) if team else None
    team_label = f" ({display_name})" if display_name else ""
    ax.set_title(
        title or f"Pass Map{team_label} — {n_success}/{n_total} successful ({pass_pct:.0f}%)",
        fontweight="bold",
    )

    # Legend
    legend_elements = [
        Line2D([0], [0], color="#2ecc71", lw=2, label="Successful"),
        Line2D([0], [0], color="#e74c3c", lw=2, label="Failed"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

    return fig


# ---------------------------------------------------------------------------
# 2. Event-based: Shot Chart with xG
# ---------------------------------------------------------------------------

def plot_shot_chart(
    events_df: pd.DataFrame,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    team_names: Optional[dict[str, str]] = None,
) -> Figure:
    """Shot chart with xG-scaled markers. Goals = stars, shots = circles.

    Parameters
    ----------
    team_names : dict, optional
        ``{"Home": "Team A", "Away": "Team B"}`` for display labels.
    """
    _require_columns(events_df, ["eID", "team", "at_x", "at_y"], "events_df")

    # Mirror first-half so Home always attacks left → right
    events_df = _mirror_events(events_df)

    # Filter shot events
    df = events_df[
        events_df["eID"].str.contains("ShotAtGoal", na=False)
    ].copy()
    df = df.dropna(subset=["at_x", "at_y"])

    # Snap penalty shots to the correct penalty-spot coordinate
    pen_mask = df["eID"].str.startswith("Penalty", na=False)
    df.loc[pen_mask & (df["team"] == "Home"), "at_x"] = PITCH_LENGTH - 11
    df.loc[pen_mask & (df["team"] == "Home"), "at_y"] = PITCH_WIDTH / 2
    df.loc[pen_mask & (df["team"] == "Away"), "at_x"] = 11
    df.loc[pen_mask & (df["team"] == "Away"), "at_y"] = PITCH_WIDTH / 2

    # Extract xG and goal flag from qualifiers
    qualifiers = df["qualifier"] if "qualifier" in df.columns else pd.Series(None, index=df.index)

    # Extract xG from qualifier
    def get_xg(q):
        qdict = _parse_qualifier(q)
        try:
            return float(qdict.get("xG", 0.05))
        except (TypeError, ValueError):
            return 0.05

    def is_goal(row: pd.Series) -> bool:
        return "Successful" in str(row.get("eID", ""))

    df["xG"] = qualifiers.apply(get_xg) if len(df) else pd.Series(dtype=float)
    df["is_goal"] = df.apply(is_goal, axis=1) if len(df) else pd.Series(dtype=bool)

    # Create figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 7.5))
    else:
        fig = ax.figure

    draw_pitch(ax, color="#f8f8f8")

    # Team colours and scatter by team
    colors = {"Home": "#e74c3c", "Away": "#7aa9e6"}
    _tn = team_names or {}

    teams = list(dict.fromkeys(df["team"].dropna().astype(str))) if len(df) else []
    if not teams:
        teams = ["Home", "Away"]

    for team in teams:
        team_df = df[df["team"] == team]
        display = _tn.get(team, team)
        goals = team_df[team_df["is_goal"]]
        non_goals = team_df[~team_df["is_goal"]]

        # Non-goals: circles
        if len(non_goals) > 0:
            ax.scatter(non_goals["at_x"], non_goals["at_y"],
                      s=non_goals["xG"] * 1000 + 30, c=colors.get(team, "#555555"),
                      alpha=0.6, edgecolors="white", linewidths=0.5, zorder=3,
                      label=f"{display} shots")

        # Goals: stars
        if len(goals) > 0:
            ax.scatter(goals["at_x"], goals["at_y"],
                      s=goals["xG"] * 1000 + 100, c=colors.get(team, "#555555"),
                      marker="*", alpha=0.9, edgecolors="white", linewidths=0.5, zorder=4,
                      label=f"{display} goals")

    # xG summary
    home_name = _tn.get("Home", "Home")
    away_name = _tn.get("Away", "Away")
    xg_home = df[df["team"] == "Home"]["xG"].sum()
    xg_away = df[df["team"] == "Away"]["xG"].sum()
    goals_home = df[(df["team"] == "Home") & df["is_goal"]].shape[0]
    goals_away = df[(df["team"] == "Away") & df["is_goal"]].shape[0]

    # xG labels near each team's attacking goal
    ax.text(
        PITCH_LENGTH * 0.76, PITCH_WIDTH * 0.92, f"{xg_home:.1f} xG",
        color=colors["Home"], fontsize=24, fontweight="bold",
        ha="center", va="center", alpha=0.95, zorder=5,
    )
    ax.text(
        PITCH_LENGTH * 0.24, PITCH_WIDTH * 0.92, f"{xg_away:.1f} xG",
        color=colors["Away"], fontsize=24, fontweight="bold",
        ha="center", va="center", alpha=0.95, zorder=5,
    )

    # Title with goal/xG summary
    if "Home" in df["team"].values or "Away" in df["team"].values:
        default_title = (
            f"Shot Chart — {home_name} {goals_home}G (xG={xg_home:.2f}) "
            f"vs {away_name} {goals_away}G (xG={xg_away:.2f})"
        )
    else:
        default_title = f"Shot Chart — {len(df)} shots"

    ax.set_title(title or default_title, fontweight="bold")

    # Legend: shots vs goals per team
    legend_handles = []
    if "Home" in teams:
        legend_handles.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor=colors["Home"],
                   markeredgecolor="white", markersize=9, label=f"{home_name} shots")
        )
        legend_handles.append(
            Line2D([0], [0], marker="*", color="w", markerfacecolor=colors["Home"],
                   markeredgecolor="white", markersize=9, label=f"{home_name} goals")
        )
    if "Away" in teams:
        legend_handles.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor=colors["Away"],
                   markeredgecolor="white", markersize=9, label=f"{away_name} shots")
        )
        legend_handles.append(
            Line2D([0], [0], marker="*", color="w", markerfacecolor=colors["Away"],
                   markeredgecolor="white", markersize=9, label=f"{away_name} goals")
        )
    if legend_handles:
        ax.legend(handles=legend_handles, loc="lower left", fontsize=8, framealpha=0.95)

    return fig


# ---------------------------------------------------------------------------
# 3. Tracking-based: Position Heatmap
# ---------------------------------------------------------------------------

def plot_heatmap(
    tracking_df: pd.DataFrame,
    team: str,
    segment: Optional[str] = None,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    bins: int = 40,
    team_names: Optional[dict[str, str]] = None,
) -> Figure:
    """2D positional density heatmap from tracking data.

    Parameters
    ----------
    team_names : dict, optional
        ``{"Home": "Team A", "Away": "Team B"}`` for display labels.
    """
    _require_columns(tracking_df, ["team"], "tracking_df")

    # Mirror first-half so Home always attacks left → right
    tracking_df = _mirror_tracking(tracking_df)

    # Filter by team and segment
    df = tracking_df[tracking_df["team"] == team]
    if segment:
        df = df[df["segment"] == segment]

    x_cols = [c for c in df.columns if c.endswith("_x")]
    y_cols = [c for c in df.columns if c.endswith("_y")]
    if not x_cols or not y_cols:
        raise ValueError("tracking_df must include player coordinate columns ending with '_x' and '_y'")

    # Flatten all player coordinates into 1D arrays
    xs = df[x_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float).ravel()
    ys = df[y_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float).ravel()
    mask = ~(np.isnan(xs) | np.isnan(ys))
    xs, ys = xs[mask], ys[mask]

    # Create figure with dark background
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 7.5))
    else:
        fig = ax.figure

    draw_pitch(ax, color="#1a1a2e", linecolor="#444")

    # Compute and render 2D density histogram
    if len(xs) > 0 and len(ys) > 0:
        h, _, _ = np.histogram2d(
            xs, ys, bins=bins, range=[[0, PITCH_LENGTH], [0, PITCH_WIDTH]]
        )
        ax.imshow(
            h.T, origin="lower", extent=[0, PITCH_LENGTH, 0, PITCH_WIDTH],
            cmap="hot", alpha=0.75, aspect="auto", zorder=2
        )
    else:
        ax.text(
            0.5, 0.5, "No tracking points",
            transform=ax.transAxes, ha="center", va="center",
            color="white", fontsize=11, alpha=0.8,
        )

    seg_label = f" · {segment}" if segment else ""
    display_name = (team_names or {}).get(team, team)
    ax.set_title(title or f"{display_name}{seg_label} — Position Density", fontweight="bold", color="black", pad=10)
    return fig


# ---------------------------------------------------------------------------
# 4. Possession Timeline
# ---------------------------------------------------------------------------

def plot_possession_timeline(
    codes_df: pd.DataFrame,
    bin_s: float = 60.0,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    team_names: Optional[dict[str, str]] = None,
) -> Figure:
    """Momentum-bar possession chart.

    For every *bin_s*-second bin the Home possession share is computed.
    Bars extend **upward** (Home colour) when the Home share exceeds 50 %
    and **downward** (Away colour) otherwise.

    Parameters
    ----------
    codes_df : DataFrame
        Output of ``parse_codes()`` (single match).
        Required columns: segment, gameclock_s, possession.
    bin_s : float
        Width of each time bin in seconds (default 60 = 1 min).
    team_names : dict, optional
        ``{"Home": "Team A", "Away": "Team B"}`` for display labels.
    """
    _require_columns(codes_df, ["segment", "gameclock_s", "possession"], "codes_df")

    home_name = (team_names or {}).get("Home", "Home")
    away_name = (team_names or {}).get("Away", "Away")

    home_color = "#c0392b"
    away_color = "#7ba5cc"

    # build continuous match time & home flag
    ordered_segs = [s for s in ["firstHalf", "secondHalf"]
                    if s in codes_df["segment"].values]
    times, flags = [], []
    offset = 0.0
    for seg in ordered_segs:
        seg_df = codes_df[codes_df["segment"] == seg].sort_values("gameclock_s")
        gc = seg_df["gameclock_s"].to_numpy(dtype=float)
        times.append(gc + offset)
        flags.append((seg_df["possession"].to_numpy(dtype=float) == 1).astype(float))
        offset = float(times[-1][-1]) if len(times[-1]) else offset

    all_t = np.concatenate(times)
    all_f = np.concatenate(flags)

    # bin into time buckets
    max_t = float(all_t[-1]) if len(all_t) else 1.0
    edges = np.arange(0, max_t + bin_s, bin_s)
    bin_idx = np.digitize(all_t, edges) - 1
    n_bins = len(edges) - 1

    home_pct = np.full(n_bins, 50.0)
    for i in range(n_bins):
        mask = bin_idx == i
        if mask.any():
            home_pct[i] = float(all_f[mask].mean()) * 100

    centres = (edges[:-1] + edges[1:]) / 2.0
    centres_min = centres / 60.0
    bar_w = bin_s / 60.0 * 0.85          

    # draw
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 3.0))
    else:
        fig = ax.figure

    fig.patch.set_facecolor("#faf9f6")
    ax.set_facecolor("#faf9f6")

    for i, (cm, hp) in enumerate(zip(centres_min, home_pct)):
        height = hp - 50            # positive = Home dominance, negative = Away
        color = home_color if height >= 0 else away_color
        ax.bar(cm, height, width=bar_w, bottom=50, color=color,
               edgecolor="none", alpha=0.85, zorder=2)

    # centre line
    ax.axhline(50, color="#aaa", lw=0.6, zorder=1)

    # x-axis: minute ticks
    total_min = max_t / 60.0
    tick_positions = np.arange(0, total_min + 1, 15)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([f"{int(t)}'" for t in tick_positions], fontsize=8)
    ax.set_xlim(-0.5, total_min + 0.5)

    # y-axis: hide numbers, label teams on left side
    y_max = max(abs(home_pct - 50).max() + 5, 15)
    ax.set_ylim(50 - y_max, 50 + y_max)
    ax.set_yticks([])

    # Team labels on the left
    ax.text(-0.01, 0.95, home_name, transform=ax.transAxes,
            color=home_color, fontsize=9, fontweight="bold", ha="right", va="top")
    ax.text(-0.01, 0.05, away_name, transform=ax.transAxes,
            color=away_color, fontsize=9, fontweight="bold", ha="right", va="bottom")

    # Overall possession text on the right
    overall_home = float(all_f.mean()) * 100
    ax.text(1.01, 0.95, f"{overall_home:.0f}%", transform=ax.transAxes,
            color=home_color, fontsize=10, fontweight="bold", ha="left", va="top")
    ax.text(1.01, 0.05, f"{100 - overall_home:.0f}%", transform=ax.transAxes,
            color=away_color, fontsize=10, fontweight="bold", ha="left", va="bottom")

    # Clean spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)

    ax.set_title(
        title or f"Possession Momentum — {home_name} vs {away_name}",
        fontweight="bold", fontsize=11, pad=8,
    )
    return fig


# ---------------------------------------------------------------------------
# 5. NPT-related: Restart Map (where stoppages restart)
# ---------------------------------------------------------------------------

def plot_restart_map(
    events_df: pd.DataFrame,
    codes_df: Optional[pd.DataFrame] = None,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
) -> Figure:
    """
    Three-panel restart analysis figure:

    Panel A (top)    : Pitch map — marker shape+colour = restart type,
                       bubble size = preceding dead-ball duration (s).
    Panel B (middle) : Horizontal bar — % of total dead-ball time per restart type.
    Panel C (bottom) : Horizontal bar — count of stoppage reasons.
    """
    _require_columns(events_df, ["eID", "at_x", "at_y", "gameclock"], "events_df")

    # ------------------------------------------------------------------
    # Restart-type definitions
    # ------------------------------------------------------------------
    RESTART_PREFIXES = {
        "KickOff":    ("#3498db", "P"),
        "ThrowIn":    ("#9b59b6", "o"),
        "FreeKick":   ("#e74c3c", "s"),
        "CornerKick": ("#f39c12", "D"),
        "GoalKick":   ("#2ecc71", "^"),
        "RefereeBall":("#1abc9c", "X"),
        "Penalty":    ("#e67e22", "*"),
    }

    RESTART_EIDS = {
        "Penalty_ShotAtGoal_SuccessfulShot",
        "KickOff_Play_Pass",
        "ThrowIn", "ThrowIn_Play_Pass", "ThrowIn_Play_Cross",
        "FreeKick_Play_Pass", "FreeKick_Play_Cross",
        "FreeKick_ShotAtGoal_SavedShot", "FreeKick_ShotAtGoal_BlockedShot",
        "FreeKick_ShotAtGoal_ShotWide",
        "CornerKick_Play_Pass", "CornerKick_Play_Cross",
        "GoalKick_Play_Pass",
        "RefereeBall",
    }

    def _restart_group(eid: str) -> Optional[str]:
        for prefix in RESTART_PREFIXES:
            if eid.startswith(prefix):
                return prefix
        return None

    # ------------------------------------------------------------------
    # Stoppage reason counts (deduplicated by gameclock × eID)
    # ------------------------------------------------------------------
    ev_dedup = events_df.drop_duplicates(subset=["gameclock", "eID"])

    var_unique = int(
        ev_dedup[ev_dedup["eID"].isin(["VideoAssistantAction", "GoalDisallowed"])].shape[0]
    )
    foul_unique  = int((ev_dedup["eID"] == "Foul").sum())
    sub_unique   = int((ev_dedup["eID"] == "OutSubstitution").sum())
    goal_unique  = int(ev_dedup["eID"].isin(
        ["ShotAtGoal_SuccessfulShot", "Penalty_ShotAtGoal_SuccessfulShot"]
    ).sum())
    offside_unique     = int((ev_dedup["eID"] == "Offside").sum())
    disciplinary_unique = int(ev_dedup["eID"].isin(["Caution", "CautionTeamofficial"]).sum())
    ballout_unique = int(ev_dedup["eID"].isin([
        "ThrowIn", "ThrowIn_Play_Pass", "ThrowIn_Play_Cross",
        "GoalKick_Play_Pass",
        "CornerKick_Play_Pass", "CornerKick_Play_Cross",
    ]).sum())
    dropball_unique = int((ev_dedup["eID"] == "RefereeBall").sum())

    STOPPAGE_GROUPS: dict = {
        "Goal Celebration": goal_unique,
        "Foul":             foul_unique,
        "Offside":          offside_unique,
        "Disciplinary":     disciplinary_unique,
        "VAR":              var_unique,
        "Substitution":     sub_unique,
        "Ball Out":         ballout_unique,
        "Medical / Drop Ball": dropball_unique,
    }

    # ------------------------------------------------------------------
    # Dead-ball duration per restart (from codes_df transitions)
    # ------------------------------------------------------------------
    # Filter restart events and assign group labels
    restart_df = events_df[events_df["eID"].isin(RESTART_EIDS)].copy()
    restart_df = restart_df.dropna(subset=["at_x", "at_y"])
    restart_df["restart_group"] = restart_df["eID"].apply(_restart_group)

    if codes_df is not None and "gameclock_s" in codes_df.columns:
        codes_sorted = codes_df.sort_values("gameclock_s")
        bs = codes_sorted["ballstatus"].to_numpy()
        ts = codes_sorted["gameclock_s"].to_numpy()

        # Detect dead↔alive transitions and pair intervals
        transitions = np.diff(bs)
        dead_starts_idx = np.where(transitions == -1)[0] + 1
        dead_ends_idx   = np.where(transitions == 1)[0] + 1
        pairs = []
        ei_ptr = 0
        for si in dead_starts_idx:
            while ei_ptr < len(dead_ends_idx) and dead_ends_idx[ei_ptr] <= si:
                ei_ptr += 1
            if ei_ptr < len(dead_ends_idx):
                pairs.append((float(ts[si]), float(ts[dead_ends_idx[ei_ptr]])))
        dead_intervals = np.array(pairs) if pairs else np.empty((0, 2))

        def _dead_duration(t_restart: float) -> float:
            """Dead-ball duration (s) ending at *t_restart* (3 s tolerance)."""
            if len(dead_intervals) == 0:
                return 5.0
            idx = np.searchsorted(dead_intervals[:, 1], t_restart, side="left")
            if idx < len(dead_intervals):
                t0, t1 = dead_intervals[idx]
                if abs(t1 - t_restart) < 3.0:
                    return max(1.0, t1 - t0)
            return 5.0

        restart_df["dead_s"] = restart_df["gameclock"].apply(
            lambda t: _dead_duration(float(t)) if pd.notna(t) else 5.0
        )
    else:
        restart_df["dead_s"] = 5.0

    # Normalise bubble sizes to [30, 400]
    d = restart_df["dead_s"].values.astype(float)
    d_min, d_max = d.min(), d.max()
    if d_max > d_min:
        sizes = 30 + (d - d_min) / (d_max - d_min) * 370
    else:
        sizes = np.full(len(d), 100.0)
    restart_df["bubble_s"] = sizes

    # Dead-ball time totals per restart group (Panel B)
    dead_by_group: dict = {}
    for grp in RESTART_PREFIXES:
        mask = restart_df["restart_group"] == grp
        dead_by_group[grp] = restart_df.loc[mask, "dead_s"].sum()
    total_dead = sum(dead_by_group.values()) or 1.0

    # ------------------------------------------------------------------
    # Layout: 3 panels
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(15, 16))
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 1.2, 1.4], hspace=0.45)
    ax_pitch = fig.add_subplot(gs[0])
    ax_dead  = fig.add_subplot(gs[1])
    ax_stop  = fig.add_subplot(gs[2])

    # ------------------------------------------------------------------
    # Panel A — pitch map
    # ------------------------------------------------------------------
    draw_pitch(ax_pitch, color="#f8f8f8")
    legend_handles = []

    # Scatter each restart group with distinct marker
    for grp, (color, marker) in RESTART_PREFIXES.items():
        sub = restart_df[restart_df["restart_group"] == grp]
        if len(sub) == 0:
            continue
        ax_pitch.scatter(
            sub["at_x"], sub["at_y"],
            c=color, marker=marker,
            s=sub["bubble_s"], alpha=0.75,
            edgecolors="white", linewidths=0.5, zorder=3,
        )
        legend_handles.append(
            Line2D([0], [0], marker=marker, color="w", markerfacecolor=color,
                   markersize=8, label=f"{grp} ({len(sub)})"))

    ax_pitch.set_title(
        title or "Restart Locations  ·  bubble size = preceding dead-ball duration",
        fontweight="bold", pad=8,
    )
    if legend_handles:
        ax_pitch.legend(handles=legend_handles,
                        loc="upper left", bbox_to_anchor=(1.01, 1.0),
                        fontsize=8, framealpha=0.85, borderaxespad=0)

    # Marker-size legend (second legend via add_artist)
    if d_max > d_min:
        size_vals = [d_min, (d_min + d_max) / 2, d_max]
    else:
        size_vals = [float(d_min)]
    size_handles = []
    for sv in size_vals:
        if d_max > d_min:
            ms_area = 30 + (sv - d_min) / (d_max - d_min) * 370
        else:
            ms_area = 100.0
        size_handles.append(
            ax_pitch.scatter([], [], s=ms_area, c="#888", alpha=0.6,
                             edgecolors="white", linewidths=0.5,
                             label=_fmt_mmss(sv))
        )
    size_legend = ax_pitch.legend(
        handles=size_handles, title="Dead-ball duration",
        loc="lower left", bbox_to_anchor=(1.01, 0.0),
        fontsize=8, title_fontsize=8, framealpha=0.85, borderaxespad=0,
    )
    ax_pitch.add_artist(size_legend)
    # Re-add restart-type legend (overwritten by size_legend call above)
    if legend_handles:
        ax_pitch.legend(handles=legend_handles,
                        loc="upper left", bbox_to_anchor=(1.01, 1.0),
                        fontsize=8, framealpha=0.85, borderaxespad=0)

    # ------------------------------------------------------------------
    # Panel B — % dead-ball time by restart type
    # ------------------------------------------------------------------
    # Sort by percentage descending
    b_items = sorted(
        [(k, v / total_dead * 100) for k, v in dead_by_group.items() if v > 0],
        key=lambda x: x[1], reverse=True,
    )
    if b_items:
        b_labels, b_pcts = zip(*b_items)
        b_colors = [RESTART_PREFIXES[k][0] for k in b_labels]
        bars = ax_dead.barh(b_labels, b_pcts, color=b_colors, edgecolor="white")
        ax_dead.set_xlabel("% of total dead-ball time", fontsize=9)
        ax_dead.set_title("Dead-Ball Time by Restart Type", fontweight="bold")
        for bar, pct in zip(bars, b_pcts):
            ax_dead.text(bar.get_width() + 0.4, bar.get_y() + bar.get_height() / 2,
                         f"{pct:.1f}%", va="center", fontsize=8)
        ax_dead.set_xlim(0, max(b_pcts) * 1.18)

    # ------------------------------------------------------------------
    # Panel C — stoppage reasons by count
    # ------------------------------------------------------------------
    # Sort by count descending
    c_items = sorted(
        [(k, v) for k, v in STOPPAGE_GROUPS.items() if v > 0],
        key=lambda x: x[1], reverse=True,
    )
    if c_items:
        c_labels, c_counts = zip(*c_items)
        c_colors = ["#e74c3c", "#e67e22", "#3498db", "#f39c12",
                    "#8e44ad", "#16a085", "#9b59b6", "#1abc9c"]
        bars2 = ax_stop.barh(c_labels, c_counts,
                              color=c_colors[:len(c_labels)], edgecolor="white")
        ax_stop.set_xlabel("Count", fontsize=9)
        ax_stop.set_title("Stoppage Reasons by Count", fontweight="bold")
        for bar, cnt in zip(bars2, c_counts):
            ax_stop.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                         str(cnt), va="center", fontsize=8)
        ax_stop.set_xlim(0, max(c_counts) * 1.12)

    return fig
