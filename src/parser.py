"""src/parser.py — Parse IDSSE XML files into flat pandas DataFrames."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Coordinate shift: centre-origin → top-left origin
_X_OFFSET = 52.5
_Y_OFFSET = 34.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_raw(
    match_id: str,
    data_dir: Path,
    events: bool = True,
    positions: bool = True,
):
    """Load raw floodlight objects for *match_id*."""
    import floodlight.io.datasets as fl_datasets
    import floodlight.settings as fl_settings

    # Resolve data directory (strip trailing "idsse_dataset/" if present)
    p = Path(data_dir)
    fl_dir = str(p.parent) if p.name == "idsse_dataset" else str(p)
    fl_settings.DATA_DIR = fl_dir
    fl_datasets.DATA_DIR = fl_dir

    ds = fl_datasets.IDSSEDataset()
    return ds.get(match_id, events=events, positions=positions)


def _validate_match_id(match_id: str) -> None:
    """Raise ``ValueError`` if *match_id* is not in the known set."""
    from src.download import MATCH_IDS as ALL_IDS

    if match_id not in ALL_IDS:
        raise ValueError(f"Unknown match_id '{match_id}'. Valid IDs: {ALL_IDS}")


def _events_to_df(raw_events, match_id: str) -> pd.DataFrame:
    """Convert floodlight event objects to a flat DataFrame (deduped, 0-based gameclock)."""
    # Flatten segment × team into rows
    frames: list[pd.DataFrame] = []
    for segment, teams in raw_events.items():
        for team, ev_obj in teams.items():
            df = ev_obj.events.copy()
            df.insert(0, "team", team)
            df.insert(0, "segment", segment)
            df.insert(0, "match_id", match_id)
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["match_id", "segment", "team"])
    out = pd.concat(frames, ignore_index=True)

    # Normalise gameclock to segment-relative 0-based seconds
    out["gameclock"] = pd.to_numeric(out["gameclock"], errors="coerce")
    for _seg, grp in out.groupby("segment"):
        gc_min = grp["gameclock"].min()
        if gc_min != 0.0 and not np.isnan(gc_min):
            out.loc[grp.index, "gameclock"] = grp["gameclock"] - gc_min

    # Remove exact duplicates (repr() for unhashable qualifier dicts)
    dedup_keys = [
        c for c in [
            "segment", "eID", "gameclock", "tID", "pID",
            "at_x", "at_y", "to_x", "to_y", "outcome", "qualifier",
            "minute", "second", "timestamp",
        ]
        if c in out.columns
    ]
    if dedup_keys:
        dedup_view = out[dedup_keys].copy()
        for col in dedup_keys:
            if dedup_view[col].dtype == "object":
                dedup_view[col] = dedup_view[col].map(
                    lambda v: repr(v) if isinstance(v, (dict, list, set, tuple)) else v
                )
        keep_mask = ~dedup_view.duplicated(keep="first")
        out = out.loc[keep_mask].reset_index(drop=True)

    return out


def _tracking_to_df(
    raw_xy, match_id: str, sample_rate: Optional[int],
) -> pd.DataFrame:
    """Convert floodlight XY objects to a flat tracking DataFrame."""
    if sample_rate is not None and (not isinstance(sample_rate, int) or sample_rate <= 0):
        raise ValueError("sample_rate must be a positive integer or None")

    step = sample_rate or 1
    frames: list[pd.DataFrame] = []
    for segment, teams in raw_xy.items():
        for team, xy_obj in teams.items():
            n_frames, n_cols = xy_obj.xy.shape
            n_players = n_cols // 2
            idx = np.arange(0, n_frames, step)
            arr = xy_obj.xy[idx]

            # Build per-player columns with centre → top-left shift
            rows = {
                "match_id": match_id,
                "segment": segment,
                "team": team,
                "frame": idx,
                "gameclock_s": idx / xy_obj.framerate,
            }
            for p in range(n_players):
                rows[f"p{p}_x"] = arr[:, 2 * p]     + _X_OFFSET
                rows[f"p{p}_y"] = arr[:, 2 * p + 1] + _Y_OFFSET
            frames.append(pd.DataFrame(rows))
    if not frames:
        return pd.DataFrame(columns=["match_id", "segment", "team", "frame", "gameclock_s"])
    return pd.concat(frames, ignore_index=True)


def _codes_to_df(raw_possession, raw_ballstatus, match_id: str) -> pd.DataFrame:
    """Convert ballstatus / possession code objects to a flat DataFrame."""
    # One row per frame per segment
    frames: list[pd.DataFrame] = []
    for segment in raw_ballstatus:
        bs  = raw_ballstatus[segment]
        pos = raw_possession[segment]
        n   = len(bs.code)
        frames.append(pd.DataFrame({
            "match_id":    match_id,
            "segment":     segment,
            "frame":       np.arange(n),
            "gameclock_s": np.arange(n) / bs.framerate,
            "ballstatus":  bs.code,
            "possession":  pos.code,
        }))
    if not frames:
        return pd.DataFrame(
            columns=["match_id", "segment", "frame", "gameclock_s", "ballstatus", "possession"]
        )
    return pd.concat(frames, ignore_index=True)


def _teamsheets_to_df(teamsheets, match_id: str) -> pd.DataFrame:
    """Convert teamsheet objects to a flat DataFrame."""
    frames: list[pd.DataFrame] = []
    for team, ts in teamsheets.items():
        df = ts.teamsheet.copy()
        if "team" not in df.columns:
            df.insert(0, "team", team)
        df.insert(0, "match_id", match_id)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["match_id", "team"])
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Public API — single-match parsers
# ---------------------------------------------------------------------------

def parse_events(match_id: str, data_dir: Path) -> pd.DataFrame:
    """Return event data for *match_id* as a flat DataFrame."""
    _validate_match_id(match_id)
    raw_events, _, _, _, _, _ = _load_raw(match_id, data_dir, positions=False)
    return _events_to_df(raw_events, match_id)



def parse_tracking(
    match_id: str,
    data_dir: Path,
    sample_rate: Optional[int] = None,
) -> pd.DataFrame:
    """Return tracking (XY) data for *match_id*.

    Parameters
    ----------
    sample_rate : int or None
        Keep every *n*-th frame (native rate: 25 Hz).
    """
    _validate_match_id(match_id)
    _, raw_xy, _, _, _, _ = _load_raw(match_id, data_dir, events=False)
    return _tracking_to_df(raw_xy, match_id, sample_rate)


def parse_codes(match_id: str, data_dir: Path) -> pd.DataFrame:
    """Return ballstatus (0=dead, 1=alive) and possession codes."""
    _validate_match_id(match_id)
    _, _, raw_possession, raw_ballstatus, _, _ = _load_raw(
        match_id, data_dir, events=False, positions=True
    )
    return _codes_to_df(raw_possession, raw_ballstatus, match_id)


def parse_teamsheets(match_id: str, data_dir: Path) -> pd.DataFrame:
    """Return player roster for *match_id*."""
    _validate_match_id(match_id)
    _, _, _, _, teamsheets, _ = _load_raw(match_id, data_dir, events=True, positions=False)
    return _teamsheets_to_df(teamsheets, match_id)


def parse_team_names(match_id: str, data_dir: Path) -> dict[str, str]:
    """Return ``{"Home": "<name>", "Away": "<name>"}`` from the matchinfo XML."""
    import xml.etree.ElementTree as ET

    _validate_match_id(match_id)

    # Locate matchinformation XML
    p = Path(data_dir)
    if p.name != "idsse_dataset":
        p = p / "idsse_dataset"
    candidates = list(p.glob(f"DFL_02_01_matchinformation_*{match_id}*.xml"))
    if not candidates:
        raise FileNotFoundError(f"Matchinfo XML not found for {match_id} in {p}")

    # Extract Home/Away team names from <Team> elements
    tree = ET.parse(candidates[0])
    names: dict[str, str] = {}
    for team_elem in tree.getroot().iter():
        tag = team_elem.tag.split("}")[-1] if "}" in team_elem.tag else team_elem.tag
        if tag == "Team" and "TeamName" in team_elem.attrib and "Role" in team_elem.attrib:
            role = team_elem.attrib["Role"]
            label = "Home" if role == "home" else "Away"
            names[label] = team_elem.attrib["TeamName"]
    return names


def parse_ball_tracking(
    match_id: str,
    data_dir: Path,
    sample_rate: Optional[int] = None,
) -> pd.DataFrame:
    """Extract ball XY from positions XML (centre → top-left origin)."""
    _validate_match_id(match_id)
    _, raw_xy, _, _, _, _ = _load_raw(match_id, data_dir, events=False)

    if sample_rate is not None and (not isinstance(sample_rate, int) or sample_rate <= 0):
        raise ValueError("sample_rate must be a positive integer or None")
    step = sample_rate or 1
    frames: list[pd.DataFrame] = []
    for segment, teams in raw_xy.items():
        if "Ball" not in teams:
            continue
        xy_obj = teams["Ball"]
        n_frames = xy_obj.xy.shape[0]
        idx = np.arange(0, n_frames, step)
        arr = xy_obj.xy[idx]

        # Shift centre-origin → top-left origin
        frames.append(pd.DataFrame({
            "match_id":    match_id,
            "segment":     segment,
            "frame":       idx,
            "gameclock_s": idx / xy_obj.framerate,
            "ball_x":      arr[:, 0] + _X_OFFSET,
            "ball_y":      arr[:, 1] + _Y_OFFSET,
        }))

    if not frames:
        return pd.DataFrame(
            columns=["match_id", "segment", "frame", "gameclock_s", "ball_x", "ball_y"]
        )
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Public API — multi-match parsers
# ---------------------------------------------------------------------------

def parse_all_ball_tracking(
    match_ids: list[str],
    data_dir: Path,
    sample_rate: int = 1,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """Parse ball-tracking data for multiple matches → ``{match_id: ball_df}``."""
    if not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")

    result: dict[str, pd.DataFrame] = {}
    for mid in match_ids:
        _validate_match_id(mid)
        if verbose:
            print(f"  {mid} ...", end=" ", flush=True)
        df = parse_ball_tracking(mid, data_dir, sample_rate=sample_rate)
        result[mid] = df
        if verbose:
            print(f"{len(df):,} frames")
    return result


def parse_match(match_id: str, data_dir: Path) -> dict[str, pd.DataFrame]:
    """Parse one match into ``{events, codes, teamsheets}`` DataFrames."""
    _validate_match_id(match_id)
    print(f"Parsing {match_id} ...")

    raw_events, _, raw_possession, raw_ballstatus, teamsheets, _ = _load_raw(
        match_id, data_dir, events=True, positions=True
    )
    result = {
        "events":     _events_to_df(raw_events, match_id),
        "codes":      _codes_to_df(raw_possession, raw_ballstatus, match_id),
        "teamsheets": _teamsheets_to_df(teamsheets, match_id),
    }
    print(f"  events : {len(result['events']):,} rows  |  code frames : {len(result['codes']):,}")
    return result


def parse_all_matches(
    data_dir: Path,
    match_ids: Optional[list[str]] = None,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Parse all (or selected) matches → ``{match_id: {events, codes, teamsheets}}``."""
    from src.download import MATCH_IDS as ALL_IDS
    if match_ids is None:
        match_ids = ALL_IDS
    for mid in match_ids:
        _validate_match_id(mid)
    return {mid: parse_match(mid, data_dir) for mid in match_ids}
