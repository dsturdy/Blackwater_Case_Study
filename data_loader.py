# data_loader.py
# -----------------------------------------------------------------------------
# Loads the BW_Test_Pack_Data.xlsx case study file from a relative path.
# Returns:
#   dev_df  → cleaned deviation dataframe
#   bvol_df → cleaned bvol dataframe (merged w/ OHLC if available)
#   ohlc_df → cleaned OHLC dataframe or None
#
# This file preserves *all* logic used in your original script.
# -----------------------------------------------------------------------------

import os
import pandas as pd
import numpy as np


def _clean_columns(df):
    """Clean column whitespace and normalize names."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _ensure_datetime(df, col):
    """Force a column to datetime safely."""
    df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def load_case_study_data(path):
    """
    Load and clean the Excel workbook with the Deviation, BVOL, and OHLC sheets.
    Returns (dev_df, bvol_df, ohlc_df)
    """

    if not os.path.exists(path):
        return None, None, None

    # ------------------------
    # Load Excel
    # ------------------------
    try:
        xls = pd.ExcelFile(path)
    except Exception:
        return None, None, None

    # ------------------------
    # DEVIATION SHEET
    # ------------------------
    if "Deviation" not in xls.sheet_names:
        return None, None, None

    dev_df = pd.read_excel(xls, "Deviation")
    dev_df = _clean_columns(dev_df)

    required_dev = ["Date", "XRT Price", "Deviation"]
    if any(col not in dev_df.columns for col in required_dev):
        return None, None, None

    dev_df = dev_df.rename(columns={
        "Date": "date",
        "XRT Price": "xrt",
        "Deviation": "deviation"
    })

    dev_df = _ensure_datetime(dev_df, "date")
    dev_df = dev_df.dropna(subset=["date"]).sort_values("date")

    # Force numeric
    dev_df["xrt"] = pd.to_numeric(dev_df["xrt"], errors="coerce")
    dev_df["deviation"] = pd.to_numeric(dev_df["deviation"], errors="coerce")

    dev_df = dev_df.set_index("date").sort_index()

    # -------------------------------------------------------
    # Compute forward 20-day returns (same as original file)
    # -------------------------------------------------------
    price_series = dev_df["xrt"].astype(float)
    dev_df["fwd20"] = price_series.shift(-20) / price_series - 1.0

    # ------------------------
    # BVOL SHEET
    # ------------------------
    if "BVOL" in xls.sheet_names:
        bvol_df = pd.read_excel(xls, "BVOL")
    elif "BW Test Pack Data" in xls.sheet_names:
        bvol_df = pd.read_excel(xls, "BW Test Pack Data")
    else:
        return dev_df, None, None  # deviation works even if bvol fails

    bvol_df = _clean_columns(bvol_df)

    required_bvol = ["Date", "XRT Price", "Bvol"]
    if any(col not in bvol_df.columns for col in required_bvol):
        return dev_df, None, None

    bvol_df = bvol_df.rename(columns={
        "Date": "date",
        "XRT Price": "xrt",
        "Bvol": "bvol"
    })

    bvol_df = _ensure_datetime(bvol_df, "date")
    bvol_df = bvol_df.dropna(subset=["date"]).sort_values("date")

    bvol_df["xrt"] = pd.to_numeric(bvol_df["xrt"], errors="coerce")
    bvol_df["bvol"] = pd.to_numeric(bvol_df["bvol"], errors="coerce")

    bvol_df = bvol_df.set_index("date").sort_index()

    # ------------------------
    # OHLC SHEET
    # ------------------------
    ohlc_df = None

    if "XRT_OHLC" in xls.sheet_names:
        ohlc_df = pd.read_excel(xls, "XRT_OHLC")
        ohlc_df = _clean_columns(ohlc_df)

        # Find date column (case-insensitive)
        date_col = None
        for c in ohlc_df.columns:
            if c.lower() == "date":
                date_col = c
                break

        if date_col is None:
            # invalid OHLC sheet
            ohlc_df = None
        else:
            # Normalize names
            ohlc_df = ohlc_df.rename(columns={date_col: "date"})
            ohlc_df.columns = [c.lower() for c in ohlc_df.columns]

            ohlc_df = _ensure_datetime(ohlc_df, "date")
            ohlc_df = ohlc_df.dropna(subset=["date"]).sort_values("date")
            ohlc_df = ohlc_df.set_index("date")

            # Convert OHLC
            for col in ["open", "high", "low", "close"]:
                if col in ohlc_df.columns:
                    ohlc_df[col] = pd.to_numeric(ohlc_df[col], errors="coerce")

    # ------------------------
    # MERGE BVOL WITH OHLC
    # ------------------------
    if ohlc_df is not None:

        # Reindex BVOL to full trading calendar
        bvol_df = bvol_df.reindex(ohlc_df.index)

        # Pull over OHLC columns
        for col in ["open", "high", "low", "close", "volume"]:
            if col in ohlc_df.columns:
                bvol_df[col] = ohlc_df[col]

        # Convert OHLC to numeric again (just to be safe)
        for col in ["open", "high", "low", "close"]:
            if col in bvol_df.columns:
                bvol_df[col] = pd.to_numeric(bvol_df[col], errors="coerce")

        # Remove duplicates
        bvol_df = bvol_df[~bvol_df.index.duplicated(keep="first")]

    return dev_df, bvol_df, ohlc_df
