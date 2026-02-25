"""src/npt_analysis.py — NPT quantification, episode decomposition, and cause inference."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd


def _require_columns(df: pd.DataFrame, required: list[str], df_name: str) -> None:
    """Raise ``ValueError`` if required columns are missing."""
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{df_name} missing required columns: {missing}")


def fmt_mmss(seconds) -> str:
    """Format seconds as ``mm:ss`` (returns ``"—"`` for NaN)."""
    if pd.isna(seconds):
        return "—"
    m, s = divmod(int(round(float(seconds))), 60)
    return f"{m}:{s:02d}"


def fmt_seconds_cols(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """Return a display copy with ``_s`` columns formatted as ``mm:ss``."""
    out = df.copy()
    if cols is None:
        cols = [c for c in out.columns if c.endswith("_s")]
    for c in cols:
        if c in out.columns:
            out[c] = out[c].apply(fmt_mmss)
    return out


# ---------------------------------------------------------------------------
# Method 1 · Frame-counting NPT
# ---------------------------------------------------------------------------

def compute_npt_frame(codes_df: pd.DataFrame) -> pd.DataFrame:
    """Method 1 — frame-counting NPT (alive_frames × dt)."""
    _require_columns(
        codes_df,
        ["match_id", "segment", "gameclock_s", "ballstatus"],
        "codes_df",
    )
    rows = []
    for (match_id, segment), grp in codes_df.groupby(["match_id", "segment"]):
        total_frames = len(grp)
        alive_frames = int((grp["ballstatus"] == 1).sum())
        dead_frames  = total_frames - alive_frames

        # Infer dt from gameclock_s spacing (should be 1/25 = 0.04 s)
        dt = float(grp["gameclock_s"].diff().dropna().median()) if total_frames > 1 else 0.04
        total_s = total_frames * dt
        npt_s   = alive_frames * dt
        dead_s  = dead_frames  * dt

        rows.append({
            "match_id":     match_id,
            "segment":      segment,
            "total_frames": total_frames,
            "alive_frames": alive_frames,
            "dead_frames":  dead_frames,
            "total_s":      round(total_s, 3),
            "npt_s":        round(npt_s,   3),
            "dead_s":       round(dead_s,  3),
            "npt_pct":      round(npt_s / total_s * 100, 2) if total_s else 0.0,
        })

    df = pd.DataFrame(rows)
    return _add_totals(df)


# ---------------------------------------------------------------------------
# Method 2 · Alive-interval summation NPT
# ---------------------------------------------------------------------------

def compute_npt_interval(codes_df: pd.DataFrame) -> pd.DataFrame:
    """Method 2 — alive-interval summation NPT."""
    _require_columns(
        codes_df,
        ["match_id", "segment", "gameclock_s", "ballstatus"],
        "codes_df",
    )
    rows = []
    for (match_id, segment), grp in codes_df.groupby(["match_id", "segment"]):
        grp    = grp.sort_values("gameclock_s").reset_index(drop=True)
        status = grp["ballstatus"].to_numpy()
        times  = grp["gameclock_s"].to_numpy()

        # Per-frame dt (handles minor jitter in gameclock_s)
        dt = np.empty(len(status))
        dt[:-1] = np.diff(times)
        dt[-1]  = dt[-2] if len(dt) > 1 else 0.04

        # Find alive intervals: contiguous runs of status == 1
        padded = np.concatenate(([False], status == 1, [False]))
        starts = np.where(~padded[:-1] & padded[1:])[0]
        ends   = np.where(padded[:-1] & ~padded[1:])[0]

        # Sum duration per interval
        intervals = [float(dt[s:e].sum()) for s, e in zip(starts, ends)]

        total_s = float(times[-1] - times[0] + dt[-1]) if len(times) > 0 else 0.0
        npt_s   = sum(intervals)
        dead_s  = total_s - npt_s

        rows.append({
            "match_id":          match_id,
            "segment":           segment,
            "n_intervals":       len(intervals),
            "total_s":           round(total_s,  3),
            "npt_s":             round(npt_s,    3),
            "dead_s":            round(dead_s,   3),
            "npt_pct":           round(npt_s / total_s * 100, 2) if total_s else 0.0,
            "mean_interval_s":   round(float(np.mean(intervals)),   2) if intervals else 0.0,
            "median_interval_s": round(float(np.median(intervals)), 2) if intervals else 0.0,
        })

    df = pd.DataFrame(rows)
    return _add_totals(df)


# ---------------------------------------------------------------------------
# Official NPT from matchinformation XML
# ---------------------------------------------------------------------------

def load_official_npt(match_id: str, data_dir: Path) -> pd.DataFrame:
    """Parse official playing-time values (ms → s) from the matchinformation XML."""
    # Locate matchinformation XML
    candidates = list(Path(data_dir).glob(f"DFL_02_01_matchinformation_*{match_id}*.xml"))
    if not candidates:
        raise FileNotFoundError(
            f"Matchinfo XML not found for {match_id} in {data_dir}"
        )
    xml_path = candidates[0]

    # Extract <OtherGameInformation> attributes
    ogi = ET.parse(xml_path).getroot().find(".//OtherGameInformation")
    if ogi is None:
        raise ValueError(f"<OtherGameInformation> not found in {xml_path}")

    def _ms_to_s(attr: str) -> float:
        return int(ogi.attrib[attr]) / 1000.0

    rows = []
    for half, total_attr, play_attr in [
        ("firstHalf",  "TotalTimeFirstHalf",  "PlayingTimeFirstHalf"),
        ("secondHalf", "TotalTimeSecondHalf", "PlayingTimeSecondHalf"),
    ]:
        total_s = _ms_to_s(total_attr)
        npt_s   = _ms_to_s(play_attr)
        rows.append({
            "match_id":         match_id,
            "segment":          half,
            "official_total_s": round(total_s, 3),
            "official_npt_s":   round(npt_s,   3),
            "official_npt_pct": round(npt_s / total_s * 100, 2) if total_s else 0.0,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Comparison · Three methods side by side (one match)
# ---------------------------------------------------------------------------

def compare_npt_methods(
    codes_df: pd.DataFrame,
    match_id: str,
    data_dir: Path,
) -> pd.DataFrame:
    """Side-by-side comparison of M1, M2, and official NPT for one match."""
    sub = codes_df[codes_df["match_id"] == match_id].copy()

    # Compute M1 (frame-counting) and M2 (interval-summation)
    m1 = (
        compute_npt_frame(sub)
        .query("segment != 'total'")
        .set_index("segment")[["npt_s", "npt_pct"]]
        .rename(columns={"npt_s": "m1_npt_s", "npt_pct": "m1_npt_pct"})
    )
    m2 = (
        compute_npt_interval(sub)
        .query("segment != 'total'")
        .set_index("segment")[["npt_s", "npt_pct"]]
        .rename(columns={"npt_s": "m2_npt_s", "npt_pct": "m2_npt_pct"})
    )

    # Load official values from XML
    official = load_official_npt(match_id, data_dir).set_index("segment")[
        ["official_npt_s", "official_npt_pct"]
    ]

    # Join and compute error columns
    cmp = official.join(m1).join(m2)

    for prefix in ("m1", "m2"):
        cmp[f"{prefix}_err_s"] = (
            cmp[f"{prefix}_npt_s"] - cmp["official_npt_s"]
        ).round(3)
        cmp[f"{prefix}_err_pct"] = (
            (cmp[f"{prefix}_npt_s"] - cmp["official_npt_s"])
            / cmp["official_npt_s"] * 100
        ).round(2)

    cols = [
        "official_npt_s", "official_npt_pct",
        "m1_npt_s", "m1_npt_pct", "m1_err_s", "m1_err_pct",
        "m2_npt_s", "m2_npt_pct", "m2_err_s", "m2_err_pct",
    ]

    return cmp[cols]


# ---------------------------------------------------------------------------
# Cross-match summary  (all 7 matches)
# ---------------------------------------------------------------------------

def npt_summary_all(
    codes_all: pd.DataFrame,
    data_dir: Path,
) -> pd.DataFrame:
    """Per-match NPT summary (M1 + M2 + official) for all matches."""
    _require_columns(codes_all, ["match_id", "segment", "gameclock_s", "ballstatus"], "codes_all")
    records = []
    for mid in sorted(codes_all["match_id"].unique()):
        sub = codes_all[codes_all["match_id"] == mid]

        # Load official NPT from XML
        try:
            off_df = load_official_npt(mid, data_dir)
        except (FileNotFoundError, ValueError, KeyError):
            continue

        # Compute M1 and M2
        m1  = compute_npt_frame(sub).query("segment != 'total'")
        m2  = compute_npt_interval(sub).query("segment != 'total'")

        off_npt = off_df["official_npt_s"].sum()
        off_tot = off_df["official_total_s"].sum()
        m1_npt  = m1["npt_s"].sum()
        m2_npt  = m2["npt_s"].sum()

        rec = {
            "match_id":         mid,
            "official_npt_s":   round(off_npt, 1),
            "m1_npt_s":         round(m1_npt,  1),
            "m2_npt_s":         round(m2_npt,  1),
            "m1_err_s":         round(m1_npt - off_npt, 1),
            "m2_err_s":         round(m2_npt - off_npt, 1),
            "official_npt_pct": round(off_npt / off_tot * 100, 1),
            "m1_npt_pct":       round(m1_npt  / off_tot * 100, 1),
            "m2_npt_pct":       round(m2_npt  / off_tot * 100, 1),
        }

        records.append(rec)

    if not records:
        empty = pd.DataFrame(
            columns=[
                "official_npt_s",
                "m1_npt_s",
                "m2_npt_s",
                "m1_err_s",
                "m2_err_s",
                "official_npt_pct",
                "m1_npt_pct",
                "m2_npt_pct",
            ]
        )
        empty.index.name = "match_id"
        return empty

    return pd.DataFrame(records).set_index("match_id")


# ---------------------------------------------------------------------------
# Time-loss episode decomposition
# ---------------------------------------------------------------------------

# Cause eIDs that trigger a stoppage
_CAUSE_EIDS = {
    "Foul",
    "Offside",
    "Caution", "CautionTeamofficial",
    "OutSubstitution",
    "VideoAssistantAction", "GoalDisallowed",
    "RefereeBall",
    "ShotAtGoal_SuccessfulShot", "Penalty_ShotAtGoal_SuccessfulShot",
}

# Restart eIDs and their canonical group name
_RESTART_MAP = {
    "KickOff_Play_Pass":                  "KickOff",
    "ThrowIn":                            "ThrowIn",
    "ThrowIn_Play_Pass":                  "ThrowIn",
    "ThrowIn_Play_Cross":                 "ThrowIn",
    "FreeKick_Play_Pass":                 "FreeKick",
    "FreeKick_Play_Cross":                "FreeKick",
    "FreeKick_ShotAtGoal_SavedShot":      "FreeKick",
    "FreeKick_ShotAtGoal_BlockedShot":    "FreeKick",
    "FreeKick_ShotAtGoal_ShotWide":       "FreeKick",
    "CornerKick_Play_Pass":               "CornerKick",
    "CornerKick_Play_Cross":              "CornerKick",
    "GoalKick_Play_Pass":                 "GoalKick",
    "RefereeBall":                        "RefereeBall",
    "Penalty_ShotAtGoal_SuccessfulShot":  "Penalty",
}

# Cause eIDs grouped for readability
_CAUSE_MAP = {
    "Foul":                               "Foul",
    "Offside":                            "Offside",
    "Caution":                            "Disciplinary",
    "CautionTeamofficial":                "Disciplinary",
    "OutSubstitution":                    "Substitution",
    "VideoAssistantAction":               "VAR",
    "GoalDisallowed":                     "VAR",
    "RefereeBall":                        "Drop Ball",
    "ShotAtGoal_SuccessfulShot":          "Goal",
    "Penalty_ShotAtGoal_SuccessfulShot":  "Goal",
}


def build_episode_table(
    codes_df: pd.DataFrame,
    events_df: pd.DataFrame,
    cause_tol: float = 5.0,
    restart_tol: float = 5.0,
) -> pd.DataFrame:
    """Build a dead-ball episode table with cause/restart labels per interval."""
    _require_columns(codes_df, ["match_id", "segment", "gameclock_s", "ballstatus"], "codes_df")
    _require_columns(events_df, ["match_id", "segment", "eID", "gameclock"], "events_df")

    match_ids = codes_df["match_id"].unique()
    all_rows: list[dict] = []

    # Pre-convert cause/restart keys to lists for np.isin
    _cause_list   = list(_CAUSE_EIDS)
    _restart_list = list(_RESTART_MAP.keys())

    for mid in match_ids:
        c = codes_df[codes_df["match_id"] == mid].sort_values(
            ["segment", "gameclock_s"]
        ).reset_index(drop=True)
        e = events_df[events_df["match_id"] == mid].copy()
        e["gameclock"] = pd.to_numeric(e["gameclock"], errors="coerce")

        # Deduplicate mirrored Home/Away logging
        dedup_keys = [
            c for c in [
                "segment", "eID", "gameclock", "tID", "pID",
                "at_x", "at_y", "to_x", "to_y", "outcome",
                "minute", "second", "timestamp",
            ] if c in e.columns
        ]
        if dedup_keys:
            e = e.drop_duplicates(subset=dedup_keys)

        for seg in c["segment"].unique():
            cs = c[c["segment"] == seg].reset_index(drop=True)
            es = e[e["segment"] == seg].sort_values("gameclock").reset_index(drop=True)

            status = cs["ballstatus"].to_numpy()
            times  = cs["gameclock_s"].to_numpy()

            # Detect dead intervals (1→0 ... 0→1)
            padded  = np.concatenate(([1], status, [1]))
            starts  = np.where((padded[:-1] == 1) & (padded[1:] == 0))[0]
            ends    = np.where((padded[:-1] == 0) & (padded[1:] == 1))[0]

            ev_gc   = es["gameclock"].to_numpy(dtype=float)
            ev_eids = es["eID"].to_numpy(dtype=str)

            for ep_i, (si, ei) in enumerate(zip(starts, ends), start=1):
                t_dead  = float(times[si])
                t_alive = float(times[ei]) if ei < len(times) else float(times[-1])
                dur     = t_alive - t_dead

                # Cause: last cause-eID event within tolerance before dead
                cause_mask = (
                    np.isin(ev_eids, _cause_list) &
                    (ev_gc >= t_dead - cause_tol) &
                    (ev_gc <= t_dead + 1.0)
                )
                cause_eID    = None
                cause_label  = "Unknown"
                cause_gc     = np.nan
                if cause_mask.any():
                    # Pick closest to t_dead
                    cand_idx  = np.where(cause_mask)[0]
                    closest   = cand_idx[np.argmin(np.abs(ev_gc[cand_idx] - t_dead))]
                    cause_eID   = str(ev_eids[closest])
                    cause_label = _CAUSE_MAP.get(cause_eID, cause_eID)
                    cause_gc    = float(ev_gc[closest])

                # Restart: first restart-eID event within tolerance after alive
                restart_mask = (
                    np.isin(ev_eids, _restart_list) &
                    (ev_gc >= t_alive - 1.0) &
                    (ev_gc <= t_alive + restart_tol)
                )
                restart_eID   = None
                restart_label = "Unknown"
                restart_gc    = np.nan
                if restart_mask.any():
                    cand_idx     = np.where(restart_mask)[0]
                    closest      = cand_idx[np.argmin(np.abs(ev_gc[cand_idx] - t_alive))]
                    restart_eID   = str(ev_eids[closest])
                    restart_label = _RESTART_MAP.get(restart_eID, restart_eID)
                    restart_gc    = float(ev_gc[closest])

                all_rows.append({
                    "match_id":        mid,
                    "segment":         seg,
                    "episode":         ep_i,
                    "dead_start_s":    round(t_dead,  2),
                    "dead_end_s":      round(t_alive, 2),
                    "duration_s":      round(dur,     2),
                    "cause_eID":       cause_eID,
                    "cause_label":     cause_label,
                    "cause_gameclock": round(cause_gc,   2) if not np.isnan(cause_gc)   else None,
                    "restart_eID":     restart_eID,
                    "restart_label":   restart_label,
                    "restart_gameclock": round(restart_gc, 2) if not np.isnan(restart_gc) else None,
                })

    return pd.DataFrame(all_rows)


def time_loss_by_restart(episodes: pd.DataFrame) -> pd.DataFrame:
    """Aggregate dead-ball time by restart type across matches."""
    grp = (
        episodes
        .groupby(["match_id", "restart_label"])["duration_s"]
        .agg(n_episodes="count", total_s="sum", mean_s="mean", median_s="median")
        .round(2)
        .reset_index()
    )

    # Overall totals across all matches
    overall = (
        episodes
        .groupby("restart_label")["duration_s"]
        .agg(n_episodes="count", total_s="sum", mean_s="mean", median_s="median")
        .round(2)
        .reset_index()
        .assign(match_id="ALL")
    )
    result = pd.concat([grp, overall], ignore_index=True)

    # Add pct_of_total within each match_id
    totals = result.groupby("match_id")["total_s"].transform("sum")
    result["pct_of_total"] = (result["total_s"] / totals * 100).round(1)

    return result.sort_values(["match_id", "total_s"], ascending=[True, False])


def cause_by_restart(
    episodes: pd.DataFrame,
    metric: str = "duration_s",
) -> pd.DataFrame:
    """Cross-tabulate cause vs restart (by count or total duration)."""
    # Pivot by chosen metric
    if metric == "count":
        pivot = (
            episodes
            .groupby(["cause_label", "restart_label"])
            .size()
            .unstack(fill_value=0)
        )
    elif metric == "duration_s":
        pivot = (
            episodes
            .groupby(["cause_label", "restart_label"])["duration_s"]
            .sum()
            .round(1)
            .unstack(fill_value=0.0)
        )
    else:
        raise ValueError("metric must be either 'count' or 'duration_s'")

    pivot["TOTAL"] = pivot.sum(axis=1)
    pivot.loc["TOTAL"] = pivot.sum(axis=0)
    return pivot.sort_values("TOTAL", ascending=False)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _add_totals(df: pd.DataFrame) -> pd.DataFrame:
    """Append a ``total`` row summing numeric columns."""
    if df.empty:
        return df
    numeric_cols = df.select_dtypes(include="number").columns
    total_row = df[numeric_cols].sum().to_dict()
    if "total_s" in total_row and total_row["total_s"]:
        total_row["npt_pct"] = round(
            total_row.get("npt_s", 0) / total_row["total_s"] * 100, 2
        )
    total_row["match_id"] = df["match_id"].iloc[0] if "match_id" in df.columns else ""
    total_row["segment"]  = "total"
    df.loc[len(df)] = total_row
    return df


# ---------------------------------------------------------------------------
# Restart-based cause inference
# ---------------------------------------------------------------------------

_RESTART_TO_CAUSE: dict[str, str] = {
    "ThrowIn":    "BallOut_Sideline",
    "GoalKick":   "BallOut_GoalLine",
    "CornerKick": "BallOut_GoalLine_Corner",
    "KickOff":    "Goal_KickOff",
}


def infer_cause_from_restart(episodes: pd.DataFrame) -> pd.DataFrame:
    """Stage 1 — resolve Unknown causes by mapping known restart type to cause."""
    eps = episodes.copy()

    # Map restart_label → cause_label for Unknown episodes
    mask = (
        (eps["cause_label"] == "Unknown") &
        (eps["restart_label"].isin(_RESTART_TO_CAUSE))
    )
    eps.loc[mask, "cause_label"] = eps.loc[mask, "restart_label"].map(_RESTART_TO_CAUSE)
    eps["cause_inferred_restart"] = mask
    return eps


def infer_cause(
    episodes: pd.DataFrame,
    ball_dfs: dict,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
    lookback_s: float = 1.5,
) -> tuple[pd.DataFrame, dict]:
    """Two-stage cause inference: restart-based (S1) then ball-coordinate (S2)."""
    # Stage 1: restart → cause mapping
    eps_s1 = infer_cause_from_restart(episodes)
    n_s1       = int(eps_s1["cause_inferred_restart"].sum())
    n_after_s1 = int((eps_s1["cause_label"] == "Unknown").sum())

    # Stage 2: ball-coordinate inference for remaining unknowns
    eps_s2 = infer_cause_from_ball(eps_s1, ball_dfs,
                                   pitch_length=pitch_length,
                                   pitch_width=pitch_width,
                                   lookback_s=lookback_s)
    n_s2    = int(eps_s2["cause_coord_inferred"].sum())
    n_final = int((eps_s2["cause_label"] == "Unknown").sum())
    total   = len(eps_s2)

    # Summary statistics
    stats = {
        "n_s1":       n_s1,
        "n_after_s1": n_after_s1,
        "n_s2":       n_s2,
        "n_final":    n_final,
        "total":      total,
    }
    return eps_s2, stats


# ---------------------------------------------------------------------------
# Ball-coordinate cause inference for Unknown-restart episodes
# ---------------------------------------------------------------------------

def infer_cause_from_ball(
    episodes: pd.DataFrame,
    ball_dfs: dict,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
    lookback_s: float = 1.5,
) -> pd.DataFrame:
    """Stage 2 — resolve Unknown causes using ball's last position before dead."""
    eps = episodes.copy()
    eps["cause_coord_inferred"] = False

    for mid, ball_df in ball_dfs.items():
        match_mask = eps["match_id"] == mid
        if not match_mask.any():
            continue

        for seg, seg_ball in ball_df.groupby("segment"):
            seg_ball = seg_ball.sort_values("gameclock_s").reset_index(drop=True)
            gc = seg_ball["gameclock_s"].to_numpy()
            bx = seg_ball["ball_x"].to_numpy()
            by = seg_ball["ball_y"].to_numpy()

            # Target: episodes with both cause and restart Unknown
            target_mask = (
                match_mask &
                (eps["segment"] == seg) &
                (eps["cause_label"] == "Unknown") &
                (eps["restart_label"] == "Unknown")
            )

            for idx in eps.index[target_mask]:
                t_dead = float(eps.at[idx, "dead_start_s"])

                # Find last ball frame in lookback window
                window = (gc >= t_dead - lookback_s) & (gc <= t_dead + 0.2)
                if not window.any():
                    continue
                last_i = int(np.where(window)[0][-1])
                lx, ly = float(bx[last_i]), float(by[last_i])

                # Check if ball crossed pitch boundary
                oob_goalline = (lx < 0.0) or (lx > pitch_length)
                oob_sideline = (ly < 0.0) or (ly > pitch_width)

                if not (oob_goalline or oob_sideline):
                    continue

                # Infer cause from crossing axis
                restart = str(eps.at[idx, "restart_label"])
                if oob_goalline:
                    inferred = (
                        "BallOut_GoalLine_Corner"
                        if restart == "CornerKick"
                        else "BallOut_GoalLine"
                    )
                else:
                    inferred = "BallOut_Sideline"

                eps.at[idx, "cause_label"]           = inferred
                eps.at[idx, "cause_coord_inferred"]  = True

    return eps
