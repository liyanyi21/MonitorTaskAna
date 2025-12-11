from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# Allowed input file extensions for tabular logs
ALLOWED_EXTS = {".csv", ".xlsx", ".xls"}

# Canonical column order for block-level tables
CANONICAL_COLS: List[str] = [
    "participant_id",
    "exp_start_time",
    "block_number",
    "block_label",
    "onset_time",
    "offset_time",
    "duration",
]

# Common synonyms mapping to canonical names (best-effort; extend as needed)
SYNONYMS: Dict[str, str] = {
    # participant id
    "participant": "participant_id",
    "participants": "participant_id",
    "participantid": "participant_id",
    "subject": "participant_id",
    "subject_id": "participant_id",
    "subjectid": "participant_id",
    "subj": "participant_id",
    "subjid": "participant_id",
    "id": "participant_id",

    # experiment start time
    "exp_start": "exp_start_time",
    "experiment_start_time": "exp_start_time",
    "start_time_exp": "exp_start_time",
    "expstarttime": "exp_start_time",

    # block number
    "block_no": "block_number",
    "blockindex": "block_number",
    "block_idx": "block_number",
    "blockid": "block_number",
    "block_number": "block_number",

    # block label
    "block": "block_label",
    "blocklabel": "block_label",
    "block_name": "block_label",
    "blocktype": "block_label",

    # onset / offset
    "start_time": "onset_time",
    "onset": "onset_time",
    "stim_onset": "onset_time",
    "offset": "offset_time",
    "end_time": "offset_time",
    "stop_time": "offset_time",

    # duration
    "dur": "duration",
    "block_duration": "duration",

    # misc
    "recording": "recording",
}


def slugify_col(col: str) -> str:
    """Normalize column names: lowercase + underscores, strip special chars.

    Examples
    --------
    >>> slugify_col("Block Name")
    'block_name'
    >>> slugify_col("Start-Time (Exp)")
    'start_time_exp'
    """
    s = col.strip().lower()
    s = re.sub(r"[^\w]+", "_", s)
    s = re.sub(r"__+", "_", s)
    return s.strip("_")


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `df` with normalized and unified column names.

    - Applies `slugify_col` to all columns
    - Renames known variants using `SYNONYMS`
    """
    out = df.copy()
    out.columns = [slugify_col(c) for c in out.columns]
    rename_map = {c: SYNONYMS[c] for c in out.columns if c in SYNONYMS}
    if rename_map:
        out = out.rename(columns=rename_map)
    return out


def _read_csv_with_fallbacks(path: Path) -> pd.DataFrame:
    encodings = ("utf-8", "utf-8-sig", "gbk", "latin-1")
    last_err: Optional[Exception] = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:  # noqa: BLE001
            last_err = e
    # final fallback: let pandas sniff delimiter with python engine
    try:
        return pd.read_csv(path, engine="python")
    except Exception as e:  # noqa: BLE001
        # raise most informative error
        raise last_err or e


def read_any_table(path: Path) -> pd.DataFrame:
    """Read a single table with format/encoding inference.

    Supports CSV (several encodings) and Excel (first sheet).
    """
    ext = path.suffix.lower()
    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=0)
    return _read_csv_with_fallbacks(path)


def combine_tables_in_dir(
    input_dir: Path,
    pattern: Optional[str] = None,
    recursive: bool = False,
    allowed_exts: Optional[Iterable[str]] = None,
    standardize: bool = True,
    canonical_cols: Optional[Sequence[str]] = None,
    output_csv: Optional[Path] = None,
) -> pd.DataFrame:
    """Read and concatenate tables from a directory.

    - Filters by filename `pattern` substring if provided.
    - Recurses into subdirectories if `recursive` is True.
    - Optionally standardizes column names and orders canonical columns first.
    - Optionally saves the combined table to `output_csv`.
    """
    exts = set(ALLOWED_EXTS if allowed_exts is None else allowed_exts)
    files: List[Path] = []
    if recursive:
        for p in input_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                if (pattern is None) or (pattern.lower() in p.name.lower()):
                    files.append(p)
    else:
        for p in input_dir.iterdir():
            if p.is_file() and p.suffix.lower() in exts:
                if (pattern is None) or (pattern.lower() in p.name.lower()):
                    files.append(p)

    parts: List[pd.DataFrame] = []
    for f in sorted(files):
        df = read_any_table(f)
        if standardize:
            df = standardize_columns(df)
        df["_source_file"] = str(f)
        parts.append(df)

    if not parts:
        return pd.DataFrame()

    out = pd.concat(parts, ignore_index=True, sort=False)
    if canonical_cols:
        # move canonical columns to front if present
        cols = list(out.columns)
        front = [c for c in canonical_cols if c in cols]
        rest = [c for c in cols if c not in front]
        out = out[front + rest]

    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_csv, index=False)
    return out


def compute_missed_events(cleaned_events: pd.DataFrame) -> pd.DataFrame:
    """Count missed events per participant/block, including zeros.

    Expects columns: ``participant_id``, ``block_label``, ``is_miss``.
    """
    required = {"participant_id", "block_label", "is_miss"}
    miss = required - set(cleaned_events.columns)
    if miss:
        raise ValueError(f"cleaned_events missing required columns: {sorted(miss)}")

    # Count only is_miss == True
    counts = (
        cleaned_events[cleaned_events["is_miss"]]
        .groupby(["participant_id", "block_label"], dropna=False)
        .size()
        .reset_index(name="missed_count")
    )

    # Ensure all combinations appear with zero
    keys = cleaned_events[["participant_id", "block_label"]].drop_duplicates()
    out = keys.merge(counts, on=["participant_id", "block_label"], how="left")
    out["missed_count"] = out["missed_count"].fillna(0).astype(int)
    return out.sort_values(["participant_id", "block_label"]).reset_index(drop=True)


def compute_event_summary(
    cleaned_events: pd.DataFrame,
    targets_per_block: int = 54,
    rt_offset_seconds: float = 1.3,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute block-level and participant-level summaries from cleaned events.

    Returns
    -------
    event_summary_block : DataFrame
        Columns: participant_id, block_label, is_hit, hit_rate, Reaction_time
    event_summary_participant : DataFrame
        Aggregated mean hit_rate and Reaction_time with simple ranks.
    """
    req = {"participant_id", "block_label", "is_hit", "first_click_ts", "end_time"}
    miss = req - set(cleaned_events.columns)
    if miss:
        raise ValueError(f"cleaned_events missing required columns: {sorted(miss)}")

    # hits per block
    hit_counts = (
        cleaned_events.groupby(["participant_id", "block_label"], dropna=False)["is_hit"]
        .sum()
        .reset_index(name="is_hit")
    )
    hit_counts["hit_rate"] = hit_counts["is_hit"] / float(targets_per_block)

    # Reaction time for hits only
    ce = cleaned_events.copy()
    ce["Reaction_time"] = ce["first_click_ts"] - (ce["end_time"] - rt_offset_seconds)
    rt = (
        ce[ce["is_hit"]]
        .groupby(["participant_id", "block_label"], dropna=False)["Reaction_time"]
        .mean()
        .reset_index()
    )

    event_summary_block = hit_counts.merge(rt, on=["participant_id", "block_label"], how="outer")

    # participant-level means and ranks
    event_summary_participant = (
        event_summary_block.groupby("participant_id", dropna=False)
        .agg({"hit_rate": "mean", "Reaction_time": "mean"})
        .reset_index()
    )
    event_summary_participant["hit_rate_rank"] = event_summary_participant["hit_rate"].rank(ascending=False)
    event_summary_participant["Reaction_time_rank"] = event_summary_participant["Reaction_time"].rank(ascending=True)

    return event_summary_block, event_summary_participant


_BLOCK_MAP = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E", 6: "F", 7: "G", 8: "H"}


def map_block_numbers_to_labels(s: pd.Series) -> pd.Series:
    """Map numeric 1..8 block codes to labels A..H."""
    return s.map(_BLOCK_MAP)


def clean_questionnaire_block(
    df: pd.DataFrame,
    rename_map: Optional[Dict[str, str]] = None,
    keep_columns: Optional[Sequence[str]] = None,
    drop_participants: Optional[Sequence[int]] = None,
    participant_id_fixes: Optional[Dict[int, int]] = None,
    block_label_is_numeric: bool = True,
) -> pd.DataFrame:
    """Standardize and filter block-level questionnaire responses.

    Parameters
    ----------
    rename_map : mapping of original column names to canonical names.
        If not provided, a minimal default is used (extend per dataset as needed).
    keep_columns : final column subset to keep. If None, a sensible default is used.
    drop_participants : participant IDs to exclude entirely.
    participant_id_fixes : mapping for ad-hoc ID fixes, e.g. {0: 4}.
    block_label_is_numeric : if True, maps 1..8 -> A..H.
    """
    default_keep = [
        "participant_id",
        "block_label",
        "mental_demand",
        "physical_demand",
        "temporal_demand",
        "effort",
        "frustration",
        "performance",
        "SSS",
        "perceived_difficulty",
    ]
    if keep_columns is None:
        keep_columns = default_keep

    if rename_map is None:
        # Minimal default mapping (extend in notebook or call-site as needed)
        rename_map = {
            # IDs and block
            "1、被试编号": "participant_id",
            "2、block 编号": "block_label",
            # Scales (subset)
            "3、脑力需求完成这个任务，需要你付出多少的脑力活动（例如思考、决策、计算、记忆、观察、搜索等)？这个任务从脑力方面对你而言是容易还是困难，简单还是复杂，严格还是宽容？": "mental_demand",
            "4、体力需求完成这个任务，需要你付出完成多少体力活动（例如推、拉、转向、控制、激活等)？这个任务是容易还是费力，做起来是轻松还是困难？": "physical_demand",
            "5、时间需求由于这个任务或任务中部分内容的进展节奏或速度，你感到多大的时间压力？这个任务的节奏是缓慢悠闲的还是快速、令人慌乱的？": "temporal_demand",
            "6、努力程度对于你取得的绩效表现，你付出了多大的努力（包括脑力和体力)？": "effort",
            "7、挫败感（受挫程度）在完成任务的过程中，你感到不安全、灰心、烦恼、压力和焦虑的程度如何（与安全、满足、 放松、满意相对)？": "frustration",
            "8、绩效表现对于我们给定的实验任务，你认为自己完成的有多成功？你对自己在完成任务中的表现有多满意？": "performance",
            "9、请按照您完成实验任务时的实际感受选择相应分值": "SSS",
            "10、“我觉得刚刚完成的任务很难。”请选择最符合您感受的选项：": "perceived_difficulty",
        }

    out = df.rename(columns=rename_map).copy()

    # Optional fixes/exclusions
    if participant_id_fixes:
        out["participant_id"] = out["participant_id"].replace(participant_id_fixes)
    if drop_participants:
        out = out[~out["participant_id"].isin(list(drop_participants))]

    # Map numeric block codes to labels
    if block_label_is_numeric:
        out["block_label"] = map_block_numbers_to_labels(out["block_label"])  # type: ignore[arg-type]

    # Filter and drop NAs for perceived_difficulty by default if present
    present = [c for c in keep_columns if c in out.columns]
    out = out[present]
    if "perceived_difficulty" in out.columns:
        out = out[out["perceived_difficulty"].notna()]

    out = out.reset_index(drop=True)
    return out


def deduplicate_clicks(
    clicks: pd.DataFrame,
    R_xy: float = 5.0,
    T_dup1: float = 0.30,
    T_dup2: float = 0.25,
    T_after_hit: float = 0.45,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Tag and filter duplicate click rows using spatial/temporal rules.

    Rules (applied per participant × block):
    - R1 dup_same_event: same event (uid/type/phase) within T_dup1 and <= R_xy
    - R3 dup_after_hit: immediate clicks after a hit within T_after_hit and same vehicle or <= R_xy
    - R2 dup_nearby: double clicks near in time/space within T_dup2 and same vehicle or <= R_xy

    Returns
    -------
    clicks_clean : DataFrame
        Original columns + helper tags, filtered to keep non-duplicates.
    fa_counts : DataFrame
        False-alarm counts per participant/block.
    dedup_summary : DataFrame
        Count of removed rows per reason.
    """
    required = {
        "participant_id",
        "block_label",
        "ts",
        "x",
        "y",
        "event_uid",
        "event_type",
        "phase",
        "vehicle_id",
        "is_hit",
        "is_false_alarm",
    }
    miss = required - set(clicks.columns)
    if miss:
        raise ValueError(f"clicks missing required columns: {sorted(miss)}")

    df = clicks.copy()

    # Light normalization of types
    df["participant_id"] = pd.to_numeric(df["participant_id"], errors="coerce").astype("Int64")
    for c in ["ts", "x", "y"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    def _norm_str(s: pd.Series) -> pd.Series:
        s2 = s.astype("string")
        return s2.where(~s2.fillna("").str.strip().isin(["", "nan", "NaN"]), other=pd.NA)

    for c in ["vehicle_id", "event_uid", "event_type", "phase", "block_label"]:
        df[c] = _norm_str(df[c])

    def _to_bool(s: pd.Series) -> pd.Series:
        if s.dtype == "boolean":
            return s
        return s.astype("string").str.lower().isin(["1", "true", "t", "y", "yes"]).astype("boolean")

    df["is_hit"] = _to_bool(df["is_hit"])  # type: ignore[assignment]
    df["is_false_alarm"] = _to_bool(df["is_false_alarm"])  # type: ignore[assignment]

    def _process_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("ts").copy()

        dt = g["ts"] - g["ts"].shift(1)
        dx = g["x"] - g["x"].shift(1)
        dy = g["y"] - g["y"].shift(1)
        dist = np.hypot(dx, dy)

        same_event = (
            g["event_uid"].eq(g["event_uid"].shift(1))
            & g["event_type"].eq(g["event_type"].shift(1))
            & g["phase"].eq(g["phase"].shift(1))
            & g["event_uid"].notna()
            & g["event_uid"].shift(1).notna()
        ).fillna(False)

        same_vehicle = g["vehicle_id"].eq(g["vehicle_id"].shift(1)).fillna(False)
        prev_hit = g["is_hit"].shift(1).fillna(False)

        near_xy = (dist <= R_xy).fillna(False)
        near_t1 = (dt <= T_dup1).fillna(False)
        near_t2 = (dt <= T_dup2).fillna(False)
        near_after = (dt <= T_after_hit).fillna(False)

        g["dup_reason"] = ""

        # R1: same event duplicate
        m1 = same_event & near_t1 & near_xy
        g.loc[m1, "dup_reason"] = "dup_same_event"

        # R3: trailing clicks after a hit
        m3 = (g["dup_reason"].eq("")) & prev_hit & near_after & (same_vehicle | near_xy)
        g.loc[m3, "dup_reason"] = "dup_after_hit"

        # R2: nearby (double) clicks
        m2 = (g["dup_reason"].eq("")) & near_t2 & (same_vehicle | near_xy)
        g.loc[m2, "dup_reason"] = "dup_nearby"

        g["keep_for_fa"] = g["dup_reason"].eq("")
        return g

    df_tag = (
        df.groupby(["participant_id", "block_label"], group_keys=False)
        .apply(_process_group)
        .reset_index(drop=True)
    )

    clicks_clean = df_tag[df_tag["keep_for_fa"]].reset_index(drop=True)

    fa_counts = (
        clicks_clean[clicks_clean["is_false_alarm"].fillna(False)]
        .groupby(["participant_id", "block_label"], dropna=False)
        .size()
        .reset_index(name="fa_count")
        .sort_values(["participant_id", "block_label"])
        .reset_index(drop=True)
    )

    dedup_summary = (
        df_tag[df_tag["dup_reason"] != ""]
        .groupby("dup_reason")
        .size()
        .reindex(["dup_same_event", "dup_after_hit", "dup_nearby"])
        .rename("removed_rows")
        .reset_index()
        .fillna(0)
    )

    return clicks_clean, fa_counts, dedup_summary


def merge_event_and_questionnaire(
    event_summary_block: pd.DataFrame,
    questionnaire_block: pd.DataFrame,
    how: str = "outer",
) -> pd.DataFrame:
    """Merge event summary with questionnaire block-level data on keys."""
    return pd.merge(
        event_summary_block,
        questionnaire_block,
        on=["participant_id", "block_label"],
        how=how,
    )

