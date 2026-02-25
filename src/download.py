"""src/download.py — Download and validate IDSSE dataset files from Figshare."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional, TypedDict

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------------------------------------------------------------------
# Dataset constants
# ---------------------------------------------------------------------------

ARTICLE_ID = "28196177"
_ARTICLE_API = f"https://api.figshare.com/v2/articles/{ARTICLE_ID}"

MATCH_IDS: list[str] = [
    "J03WMX", "J03WN1",                                    # Competition 1
    "J03WPY", "J03WOH", "J03WQQ", "J03WOY", "J03WR9",     # Competition 2
]

_COMPETITION = {
    "J03WMX": "DFL-COM-000001", "J03WN1": "DFL-COM-000001",
    "J03WPY": "DFL-COM-000002", "J03WOH": "DFL-COM-000002",
    "J03WQQ": "DFL-COM-000002", "J03WOY": "DFL-COM-000002", "J03WR9": "DFL-COM-000002",
}

_FILE_TEMPLATES = {
    "info": "DFL_02_01_matchinformation_{competition}_DFL-MAT-{match_id}.xml",
    "event": "DFL_03_02_events_raw_{competition}_DFL-MAT-{match_id}.xml",
    "position": "DFL_04_03_positions_raw_observed_{competition}_DFL-MAT-{match_id}.xml",
}

_DEFAULT_FILE_TYPES = ("info", "event", "position")


class CatalogueEntry(TypedDict, total=False):
    download_url: str
    md5: str


Catalogue = dict[str, CatalogueEntry]


def _build_retry() -> Retry:
    """Create a ``Retry`` object compatible with urllib3 v1 and v2."""
    retry_kwargs = {
        "total": 3,
        "connect": 3,
        "read": 3,
        "status": 3,
        "backoff_factor": 0.5,
        "status_forcelist": (429, 500, 502, 503, 504),
        "raise_on_status": False,
    }

    # urllib3 v1 uses method_whitelist; v2 uses allowed_methods
    if "allowed_methods" in Retry.__init__.__code__.co_varnames:
        retry_kwargs["allowed_methods"] = frozenset(["GET"])
    else:
        retry_kwargs["method_whitelist"] = frozenset(["GET"])

    return Retry(**retry_kwargs)


def _create_session() -> requests.Session:
    """Return a ``requests.Session`` with automatic retry."""
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=_build_retry())
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _validate_match_ids(match_ids: list[str]) -> None:
    """Raise ``ValueError`` for unknown match IDs."""
    invalid = [mid for mid in match_ids if mid not in _COMPETITION]
    if invalid:
        raise ValueError(f"Unknown match_ids: {invalid}. Valid IDs: {MATCH_IDS}")


def _validate_file_types(file_types: list[str]) -> list[str]:
    """Validate file types and return de-duplicated list."""
    invalid = [ftype for ftype in file_types if ftype not in _FILE_TEMPLATES]
    if invalid:
        valid = sorted(_FILE_TEMPLATES.keys())
        raise ValueError(f"Unknown file_types: {invalid}. Valid types: {valid}")
    # Preserve order and remove duplicates
    return list(dict.fromkeys(file_types))


# ---------------------------------------------------------------------------
# Catalogue & download helpers
# ---------------------------------------------------------------------------

def _fetch_catalogue(session: Optional[requests.Session] = None) -> Catalogue:
    """Fetch file metadata from the Figshare article API → ``{filename: {download_url, md5}}``."""
    owns_session = session is None
    if session is None:
        session = _create_session()

    try:
        # Query Figshare API and build filename → metadata mapping
        r = session.get(_ARTICLE_API, timeout=30)
        r.raise_for_status()
        catalogue: Catalogue = {}
        for file_obj in r.json().get("files", []):
            name = file_obj.get("name")
            download_url = file_obj.get("download_url")
            if not name or not download_url:
                continue
            catalogue[name] = {
                "download_url": download_url,
                "md5": file_obj.get("computed_md5", ""),
            }
        return catalogue
    finally:
        if owns_session:
            session.close()


def _md5(path: Path) -> str:
    """Return hex MD5 digest of *path*."""
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_one(
    session: requests.Session,
    download_url: str,
    output_path: Path,
    expected_md5: Optional[str] = None,
    timeout: int = 300,
) -> bool:
    """Stream-download a single file, optionally verifying MD5."""
    try:
        # Stream download in 1 MB chunks
        with session.get(download_url, timeout=timeout, stream=True) as r:
            r.raise_for_status()

            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1024 * 1024):  # 1 MB chunks
                    if chunk:
                        fh.write(chunk)

        size_mb = output_path.stat().st_size / 1024 ** 2

        # Verify MD5 if expected hash provided
        if expected_md5:
            actual_md5 = _md5(output_path)
            if actual_md5 != expected_md5:
                print(f"  ❌  MD5 mismatch for {output_path.name}")
                print(f"       expected: {expected_md5}")
                print(f"       got:      {actual_md5}")
                try:
                    output_path.unlink()
                    print("       removed corrupted file")
                except OSError:
                    pass
                return False

        print(f"  ✅  {output_path.name}  ({size_mb:.1f} MB)  md5 OK")
        return True

    except (requests.RequestException, OSError) as exc:
        print(f"  ❌  {output_path.name}: {exc}")
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def download_match(
    match_id: str,
    data_dir: Path,
    catalogue: Optional[Catalogue] = None,
    file_types: Optional[list[str]] = None,
    skip_existing: bool = True,
    session: Optional[requests.Session] = None,
) -> dict[str, bool]:
    """Download selected files (info / event / position) for one match."""
    _validate_match_ids([match_id])
    selected_types = _validate_file_types(file_types or list(_DEFAULT_FILE_TYPES))

    owns_session = session is None
    if session is None:
        session = _create_session()

    try:
        if catalogue is None:
            catalogue = _fetch_catalogue(session=session)

        competition = _COMPETITION[match_id]
        dataset_dir = data_dir / "idsse_dataset"
        results: dict[str, bool] = {}

        print(f"\n── {match_id} ──────────────────────────────")
        for ftype in selected_types:
            fname = _FILE_TEMPLATES[ftype].format(competition=competition, match_id=match_id)
            out_path = dataset_dir / fname
            meta = catalogue.get(fname, {})
            expected_md5 = meta.get("md5", "")
            download_url = meta.get("download_url")

            if not download_url:
                print(f"  ❌  {fname}: not found in Figshare catalogue")
                results[ftype] = False
                continue

            # Skip if file already exists with correct MD5
            if skip_existing and out_path.exists() and out_path.stat().st_size > 0:
                if expected_md5 and _md5(out_path) == expected_md5:
                    size_mb = out_path.stat().st_size / 1024 ** 2
                    print(f"  ⏭   {fname}  (already exists, {size_mb:.1f} MB, md5 OK)")
                    results[ftype] = True
                    continue

            results[ftype] = _download_one(
                session=session,
                download_url=download_url,
                output_path=out_path,
                expected_md5=expected_md5,
            )

        return results
    finally:
        if owns_session:
            session.close()


def download_all_matches(
    data_dir: Path,
    match_ids: Optional[list[str]] = None,
    file_types: Optional[list[str]] = None,
    skip_existing: bool = True,
) -> dict[str, dict[str, bool]]:
    """Download files for every match in *match_ids* (default: all 7)."""
    selected_matches = match_ids or list(MATCH_IDS)
    _validate_match_ids(selected_matches)
    selected_types = _validate_file_types(file_types or list(_DEFAULT_FILE_TYPES))

    with _create_session() as session:
        # Fetch catalogue once and reuse for all matches
        print(f"Fetching file catalogue from Figshare article {ARTICLE_ID} ...")
        catalogue = _fetch_catalogue(session=session)
        print(f"  Found {len(catalogue)} files in catalogue.\n")

        print(f"Downloading {len(selected_matches)} match(es) -> {data_dir / 'idsse_dataset'}")
        all_results: dict[str, dict[str, bool]] = {}

        for mid in selected_matches:
            all_results[mid] = download_match(
                match_id=mid,
                data_dir=data_dir,
                catalogue=catalogue,
                file_types=selected_types,
                skip_existing=skip_existing,
                session=session,
            )

    # Print download summary
    print("\n" + "=" * 58)
    print("Download summary")
    print("=" * 58)
    for mid, res in all_results.items():
        ok = all(res.values())
        flag = "OK" if ok else "WARN"
        details = "  ".join(f"{k}={'Y' if v else 'N'}" for k, v in res.items())
        print(f"  [{flag}]  {mid}   {details}")

    total_ok = sum(1 for r in all_results.values() if all(r.values()))
    print(f"\n  {total_ok}/{len(selected_matches)} matches fully downloaded.")
    return all_results


def validate_downloads(
    data_dir: Path,
    match_ids: Optional[list[str]] = None,
    check_md5: bool = False,
) -> bool:
    """Verify expected XML files exist, optionally checking MD5."""
    selected_matches = match_ids or list(MATCH_IDS)
    _validate_match_ids(selected_matches)

    # Fetch catalogue only when MD5 check is requested
    with _create_session() as session:
        catalogue = _fetch_catalogue(session=session) if check_md5 else {}

    dataset_dir = data_dir / "idsse_dataset"
    all_ok = True

    # Check each expected file
    print(f"Validating files (md5_check={check_md5}) ...")
    for mid in selected_matches:
        competition = _COMPETITION[mid]
        for ftype, template in _FILE_TEMPLATES.items():
            fname = template.format(competition=competition, match_id=mid)
            fpath = dataset_dir / fname

            if not fpath.exists() or fpath.stat().st_size == 0:
                print(f"  ❌  MISSING / EMPTY : {fname}")
                all_ok = False
                continue

            size_mb = fpath.stat().st_size / 1024 ** 2

            if check_md5:
                expected = catalogue.get(fname, {}).get("md5")
                if expected and _md5(fpath) != expected:
                    print(f"  ❌  MD5 MISMATCH   : {fname}")
                    all_ok = False
                    continue

            print(f"  ✅  {fname}  ({size_mb:.1f} MB)")

    return all_ok
