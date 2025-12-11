
import re
import pandas as pd
from typing import Dict, List, Tuple, Optional

# -------- Configuration defaults --------
REVERSE_ITEMS = {1, 2, 3, 6, 7, 8, 11, 12, 15, 16, 20}
FOCUSING_ITEMS = list(range(1, 10))      # Q1–Q9
SHIFTING_ITEMS = list(range(10, 21))     # Q10–Q20

# Common meta-field renames (add more if needed)
META_RENAMES = {
    "序号": "row_id",
    "提交答卷时间": "submitted_at",
    "所用时间": "duration",
    "来源": "source",
    "来源详情": "source_detail",
    "来自IP": "ip",
    "您的被试编号：": "subject_id",
    "您的姓名：": "name",
    "您的性别：": "sex",
    "您的年龄（岁）": "age",
}

def _extract_question_number(col: str) -> Optional[int]:
    """
    Try to extract question number (1..20) from a Chinese item string.
    Supports patterns like:
      '—1. 当周围嘈杂时...' or '1.当周围嘈杂时...' etc.
    """
    # Normalize weird dashes and spaces
    s = col.replace("—", " ").replace("—", " ")
    # Look for a number followed by a dot (ASCII or fullwidth)
    m = re.search(r'(\d{1,2})\s*[\.\uff0e]', s)
    if m:
        try:
            n = int(m.group(1))
            if 1 <= n <= 20:
                return n
        except ValueError:
            return None
    return None

def build_column_mapping(columns: List[str]) -> Dict[str, str]:
    """
    Build a mapping from original column names to cleaned names:
      - Meta fields per META_RENAMES
      - Question items -> Q1...Q20
      - Unrecognized columns are left as-is (you can drop them later)
    """
    mapping: Dict[str, str] = {}
    for c in columns:
        if c in META_RENAMES:
            mapping[c] = META_RENAMES[c]
            continue
        n = _extract_question_number(c)
        if n is not None:
            mapping[c] = f"Q{n}"
        else:
            mapping[c] = c  # keep original if not recognized
    return mapping

def reverse_score(series: pd.Series, scale_max: int) -> pd.Series:
    """
    Reverse scoring for a numeric series given the Likert scale max.
    Works for 4点制 (scale_max=4) or 7点制 (scale_max=7).
    Formula: new = (scale_max + 1) - old
    """
    return (scale_max + 1) - series

def coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors='coerce')
    return out

def score_acs(
    df: pd.DataFrame,
    scale_max: int = 4,
    keep_meta: Optional[List[str]] = None,
    drop_unrecognized: bool = True
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Score the Attentional Control Scale (ACS) from a raw dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Raw data as read from csv/xlsx (Chinese columns ok).
    scale_max : int, default=4
        Likert max (4 or 7 typically). Reverse items use (scale_max+1 - x).
    keep_meta : list[str] | None
        Which meta fields to keep in the output (after renaming). If None,
        keep common ones present in META_RENAMES.
    drop_unrecognized : bool, default=True
        If True, drop columns that aren't meta fields or Q1..Q20.

    Returns
    -------
    (scored_df, mapping):
        scored_df: dataframe with renamed columns, Q1..Q20 numeric, and
        added ACS_total / ACS_focusing / ACS_shifting columns.
        mapping: dict of original_name -> new_name for reference.
    """
    # 1) Build rename mapping
    mapping = build_column_mapping(list(df.columns))

    df1 = df.rename(columns=mapping).copy()

    # 2) Determine which meta fields to retain
    if keep_meta is None:
        keep_meta = [v for v in META_RENAMES.values() if v in df1.columns]

    # 3) Identify question columns
    q_cols = [f"Q{i}" for i in range(1, 21) if f"Q{i}" in df1.columns]

    # 4) Optionally drop unrecognized columns
    if drop_unrecognized:
        cols_to_keep = keep_meta + q_cols
        df1 = df1.loc[:, [c for c in cols_to_keep if c in df1.columns]]

    # 5) Coerce Q columns to numeric
    df1 = coerce_numeric(df1, q_cols)

    # 6) Build adjusted (reverse-handled) columns
    adj_cols = []
    for i in range(1, 21):
        q = f"Q{i}"
        if q not in df1.columns:
            # If any Q is missing, create NaN column to keep shapes consistent
            df1[q] = pd.NA
        if i in REVERSE_ITEMS:
            df1[f"{q}_adj"] = reverse_score(df1[q], scale_max=scale_max)
        else:
            df1[f"{q}_adj"] = df1[q]
        adj_cols.append(f"{q}_adj")

    # 7) Compute total and subscales
    focusing_adj = [f"Q{i}_adj" for i in FOCUSING_ITEMS]
    shifting_adj = [f"Q{i}_adj" for i in SHIFTING_ITEMS]

    df1["ACS_total"] = df1[adj_cols].sum(axis=1, min_count=1)
    df1["ACS_focusing"] = df1[focusing_adj].sum(axis=1, min_count=1)
    df1["ACS_shifting"] = df1[shifting_adj].sum(axis=1, min_count=1)

    # Place score columns at the end (nice ordering)
    score_order = ["ACS_total", "ACS_focusing", "ACS_shifting"]
    meta_then_q_then_scores = keep_meta + [f"Q{i}" for i in range(1,21)] + score_order
    final_cols = [c for c in meta_then_q_then_scores if c in df1.columns]
    df1 = df1.loc[:, final_cols]

    return df1, mapping

# Convenience function to run quickly on a CSV/XLSX
def score_file(
    in_path: str,
    out_path: str,
    scale_max: int = 4,
    keep_meta: Optional[List[str]] = None
) -> None:
    """
    Auto-detect file type (csv/xlsx) and write a scored xlsx to out_path.
    """
    if in_path.lower().endswith(".csv"):
        raw = pd.read_csv(in_path)
    elif in_path.lower().endswith(".xlsx") or in_path.lower().endswith(".xls"):
        raw = pd.read_excel(in_path)
    else:
        raise ValueError("Unsupported file type. Use .csv or .xlsx")

    scored, mapping = score_acs(raw, scale_max=scale_max, keep_meta=keep_meta)
    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        scored.to_excel(writer, index=False, sheet_name="ACS_scored")
        # Also dump mapping for transparency
        pd.DataFrame(
            [{"original": k, "renamed": v} for k, v in mapping.items()]
        ).to_excel(writer, index=False, sheet_name="mapping")

if __name__ == "__main__":
    # Example CLI usage (adjust paths as needed):
    # python acs_scoring.py input.xlsx output_scored.xlsx 4
    import sys
    if len(sys.argv) >= 3:
        in_path = sys.argv[1]
        out_path = sys.argv[2]
        scale_max = int(sys.argv[3]) if len(sys.argv) >= 4 else 4
        score_file(in_path, out_path, scale_max=scale_max)
        print(f"Scored file saved to: {out_path}")
    else:
        print("Usage: python acs_scoring.py <input.csv|xlsx> <output.xlsx> [scale_max]")
