"""Utility helpers for data preparation and analysis.

This package contains small, focused functions that can be imported and
reused from notebooks to reduce duplication and improve readability.
"""

from .data_prep import (
    ALLOWED_EXTS,
    CANONICAL_COLS,
    SYNONYMS,
    slugify_col,
    standardize_columns,
    read_any_table,
    combine_tables_in_dir,
    compute_missed_events,
    compute_event_summary,
    map_block_numbers_to_labels,
    clean_questionnaire_block,
    deduplicate_clicks,
    merge_event_and_questionnaire,
)

__all__ = [
    "ALLOWED_EXTS",
    "CANONICAL_COLS",
    "SYNONYMS",
    "slugify_col",
    "standardize_columns",
    "read_any_table",
    "combine_tables_in_dir",
    "compute_missed_events",
    "compute_event_summary",
    "map_block_numbers_to_labels",
    "clean_questionnaire_block",
    "deduplicate_clicks",
    "merge_event_and_questionnaire",
]

