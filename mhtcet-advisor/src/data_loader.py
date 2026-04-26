"""
data_loader.py — Loads and normalises cutoff + seat matrix Excel files.

Expected folder layout (relative to project root):
  data/cutoffs/YYYY_CAPX_MH.xlsx   (e.g. 2022_CAP1_MH.xlsx)
  data/seat_matrix/seat_matrix_YYYY.xlsx

The cutoff files use a wide format: one row per college-branch, with each
category's closing merit rank and percentile in separate columns named like
  "GOPENS Merit"  / "GOPENS Percentile"
  "LOBCS Merit"   / "LOBCS Percentile"
  "TFWS Merit"    / "TFWS Percentile"
  …

This loader melts those wide files into a long tidy table:
  college_id | college_name | course_id | course_name | status | seat_type |
  year | round | category | merit | percentile
"""

import os
import re
import glob
import pandas as pd
import streamlit as st
import yaml

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config")


@st.cache_data(show_spinner=False)
def load_config():
    with open(os.path.join(CONFIG_DIR, "categories.yaml")) as f:
        return yaml.safe_load(f)


@st.cache_data(show_spinner=False)
def load_all_cutoffs() -> pd.DataFrame:
    """Load all cutoff files from data/cutoffs/ and return a unified long DataFrame."""
    pattern = os.path.join(DATA_DIR, "cutoffs", "*.xlsx")
    files = glob.glob(pattern)
    if not files:
        return pd.DataFrame()

    frames = []
    for fpath in files:
        fname = os.path.basename(fpath)
        # Parse year and round from filename: 2022_CAP1_MH.xlsx or 2022ENGG_CAP1_CutOff.xlsx
        year, cap_round = _parse_filename(fname)
        if year is None:
            continue
        try:
            df = pd.read_excel(fpath, sheet_name=0)
            df = _normalise_cutoff(df, year, cap_round)
            frames.append(df)
        except Exception as e:
            st.warning(f"Could not load {fname}: {e}")

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


@st.cache_data(show_spinner=False)
def load_seat_matrix() -> pd.DataFrame:
    """Load the latest seat matrix file."""
    pattern = os.path.join(DATA_DIR, "seat_matrix", "*.xlsx")
    files = sorted(glob.glob(pattern), reverse=True)
    if not files:
        return pd.DataFrame()
    try:
        df = pd.read_excel(files[0], sheet_name=0)
        return _normalise_seat_matrix(df)
    except Exception as e:
        st.warning(f"Could not load seat matrix: {e}")
        return pd.DataFrame()


def _parse_filename(fname: str):
    """Extract (year, round_num) from filename."""
    # Matches: 2022_CAP1_MH.xlsx, 2023_CAP2_MH.xlsx, 2022ENGG_CAP1_CutOff.xlsx
    m = re.search(r'(20\d{2}).*?CAP(\d)', fname, re.IGNORECASE)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def _normalise_cutoff(df: pd.DataFrame, year: int, cap_round: int) -> pd.DataFrame:
    """Melt wide cutoff DataFrame into long tidy format."""
    # Standardise column names
    col_map = {
        'College ID': 'college_id',
        'College Name': 'college_name',
        'Course ID': 'course_id',
        'Course Name': 'course_name',
        'Status': 'status',
        'Seat Type': 'seat_type',
        'Stage': 'stage',
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Identify base columns vs category columns
    base_cols = ['college_id', 'college_name', 'course_id', 'course_name', 'status', 'seat_type', 'stage']
    base_cols = [c for c in base_cols if c in df.columns]

    # Find all "X Merit" and "X Percentile" columns
    merit_cols = [c for c in df.columns if c.endswith(' Merit')]
    pct_cols = [c for c in df.columns if c.endswith(' Percentile')]

    # Build category list from merit columns
    categories = [c.replace(' Merit', '') for c in merit_cols]

    rows = []
    for cat in categories:
        merit_col = f"{cat} Merit"
        pct_col = f"{cat} Percentile"
        if merit_col not in df.columns or pct_col not in df.columns:
            continue
        sub = df[base_cols + [merit_col, pct_col]].copy()
        sub = sub.dropna(subset=[pct_col])
        sub = sub[sub[pct_col] > 0]
        sub['category'] = cat
        sub['merit'] = pd.to_numeric(sub[merit_col], errors='coerce')
        sub['percentile'] = pd.to_numeric(sub[pct_col], errors='coerce')
        sub = sub.drop(columns=[merit_col, pct_col])
        rows.append(sub)

    if not rows:
        return pd.DataFrame()

    result = pd.concat(rows, ignore_index=True)
    result['year'] = year
    result['cap_round'] = cap_round
    return result


def _normalise_seat_matrix(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {
        'College ID': 'college_id',
        'College Name': 'college_name',
        'Status': 'status',
        'CAP Seats': 'cap_seats',
        'Choice Code': 'choice_code',
        'Course Name': 'course_name',
        'SI': 'sanctioned_intake',
        'MS Seats': 'ms_seats',
        'TFWS_Seats': 'tfws_seats',
        'EWS_Seats': 'ews_seats',
        'Orphan': 'orphan_seats',
        'SL_Total': 'sl_total',
        'HU_Total': 'hu_total',
        'OHU_Total': 'ohu_total',
    }
    return df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})


def get_available_branches(cutoff_df: pd.DataFrame) -> list:
    if cutoff_df.empty:
        return []
    return sorted(cutoff_df['course_name'].dropna().unique().tolist())


def get_available_colleges(cutoff_df: pd.DataFrame) -> list:
    if cutoff_df.empty:
        return []
    return sorted(cutoff_df['college_name'].dropna().unique().tolist())


def get_college_status_map(cutoff_df: pd.DataFrame) -> dict:
    """Returns {college_name: status}"""
    if cutoff_df.empty:
        return {}
    return dict(zip(cutoff_df['college_name'], cutoff_df['status']))
