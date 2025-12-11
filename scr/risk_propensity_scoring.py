
import re
import pandas as pd
from typing import Dict, List, Optional, Tuple

# --------------------------
# Risk Propensity Scale (8 items) - Chinese form
# Positive items: 1,4,5,8
# Reverse items:  2,3,6,7
# --------------------------

RPS_REVERSE = {2, 3, 6, 7}
RPS_ITEMS = list(range(1, 9))

# Common meta-field renames
META_RENAMES = {
    "序号": "row_id",
    "提交答卷时间": "submitted_at",
    "所用时间": "duration",
    "来源": "source",
    "来源详情": "source_detail",
    "来自IP": "ip",
    # two variants you showed
    "1、被试编号：": "subject_id",
    "2、姓名：": "name",
    # keep compatibility with previous batch (if present)
    "您的被试编号：": "subject_id",
    "您的姓名：": "name",
    "您的性别：": "sex",
    "您的年龄（岁）": "age",
    # possible simplified headers
    "性别": "sex",
    "年龄": "age",
    "年龄（岁）": "age",
}

def _extract_last_number_before_dot(s: str) -> Optional[int]:
    """Find the LAST number immediately preceding a dot (ASCII '.' or fullwidth variants)."""
    t = s.replace("—", " ").replace("–", " ").replace("—", " ")
    # e.g., "3、2. ..." or "—1. ..." -> take the last number before a dot
    nums = re.findall(r'(\d{1,2})\s*[\.\uff0e\u3002\uFF0E\uFF61]', t)
    if nums:
        try:
            return int(nums[-1])
        except ValueError:
            return None
    # fallback: sometimes "1、我喜欢..." (comma-like '、' after number)
    nums2 = re.findall(r'(\d{1,2})\s*[、,，]', t)
    if nums2:
        try:
            return int(nums2[-1])
        except ValueError:
            return None
    return None

def build_column_mapping(columns: List[str]) -> Dict[str, str]:
    """Map original Chinese headers to meta fields and RP1..RP8 item names."""
    mapping: Dict[str, str] = {}
    for c in columns:
        if c in META_RENAMES:
            mapping[c] = META_RENAMES[c]
            continue
        n = _extract_last_number_before_dot(c)
        if n is not None and 1 <= n <= 8:
            mapping[c] = f"RP{n}"
        else:
            mapping[c] = c  # keep as-is (can be dropped later)
    return mapping

def reverse_series(series: pd.Series, scale_max: int) -> pd.Series:
    """Reverse scoring given Likert max: new = (scale_max + 1) - old."""
    return (scale_max + 1) - series

def coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors='coerce')
    return out

def score_rps(
    df: pd.DataFrame,
    scale_max: int = 4,
    keep_meta: Optional[List[str]] = None,
    drop_unrecognized: bool = True
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Score the 8-item Risk Propensity Scale (Chinese).

    Returns (scored_df, mapping).
    scored_df includes RP1..RP8, RP*_adj, and RPS_total / RPS_mean.
    """
    mapping = build_column_mapping(list(df.columns))
    df1 = df.rename(columns=mapping).copy()

    # meta fields to keep (auto if not provided)
    if keep_meta is None:
        keep_meta = [v for v in META_RENAMES.values() if v in df1.columns]

    # item columns present
    q_cols = [f"RP{i}" for i in RPS_ITEMS if f"RP{i}" in df1.columns]

    if drop_unrecognized:
        cols_to_keep = keep_meta + q_cols
        df1 = df1.loc[:, [c for c in cols_to_keep if c in df1.columns]]

    # numeric
    df1 = coerce_numeric(df1, q_cols)

    # adjusted (reverse) columns
    adj_cols = []
    for i in RPS_ITEMS:
        q = f"RP{i}"
        if q not in df1.columns:
            df1[q] = pd.NA
        if i in RPS_REVERSE:
            df1[f"{q}_adj"] = reverse_series(df1[q], scale_max=scale_max)
        else:
            df1[f"{q}_adj"] = df1[q]
        adj_cols.append(f"{q}_adj")

    # scores
    df1["RPS_total"] = df1[adj_cols].sum(axis=1, min_count=1)
    df1["RPS_mean"]  = df1[adj_cols].mean(axis=1)

    # order columns
    score_cols = ["RPS_total", "RPS_mean"]
    final_order = keep_meta + [f"RP{i}" for i in RPS_ITEMS] + score_cols
    final_order = [c for c in final_order if c in df1.columns]
    df1 = df1.loc[:, final_order]

    return df1, mapping

def standardize_sex_col(series: pd.Series) -> pd.Series:
    """Map variants to 男/女; leave others as-is."""
    sex_map = {
        "男":"男","male":"男","Male":"男","M":"男","m":"男",
        "女":"女","female":"女","Female":"女","F":"女","f":"女"
    }
    return series.astype(str).str.strip().map(lambda x: sex_map.get(x, x))

def demographics_summary(
    df: pd.DataFrame,
    subject_id_col: str = "subject_id",
    sex_col: str = "sex",
    age_col: str = "age",
    exclude_ids: Optional[List[int]] = None
) -> Dict[str, object]:
    """
    Compute male/female counts and percentages, mean and sd of age,
    after optionally excluding subject IDs.
    """
    d = df.copy()
    if subject_id_col in d.columns and exclude_ids:
        sid = pd.to_numeric(d[subject_id_col], errors="coerce")
        d = d[~sid.isin(exclude_ids)].copy()

    # sex
    if sex_col in d.columns:
        d["sex_std"] = standardize_sex_col(d[sex_col])
        n = len(d)
        counts = d["sex_std"].value_counts(dropna=False)
        pct = (counts / n * 100).round(1)
        male_n = int(counts.get("男", 0))
        female_n = int(counts.get("女", 0))
        male_pct = float(pct.get("男", 0.0))
        female_pct = float(pct.get("女", 0.0))
    else:
        n = len(d)
        male_n=female_n=0
        male_pct=female_pct=0.0
        counts = pct = pd.Series(dtype=int)

    # age
    if age_col in d.columns:
        age = pd.to_numeric(d[age_col], errors="coerce")
        age_mean = float(age.mean())
        age_sd = float(age.std(ddof=1))
    else:
        age_mean = float('nan')
        age_sd = float('nan')

    return {
        "N_after_exclusion": int(n),
        "sex_counts": counts.to_dict(),
        "sex_percent": pct.to_dict(),
        "male_n": male_n, "male_pct": male_pct,
        "female_n": female_n, "female_pct": female_pct,
        "age_mean": age_mean, "age_sd": age_sd,
    }
