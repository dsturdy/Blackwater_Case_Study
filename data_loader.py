# data_loader.py
# -----------------------------------------------------------------------------
# Loads the BW_Test_Pack_Data.xlsx case study file from a relative path.
# Returns:
#   dev_df  → cleaned deviation dataframe
#   bvol_df → cleaned bvol dataframe (merged w/ OHLC if available)
#   ohlc_df → cleaned OHLC dataframe or None
# -----------------------------------------------------------------------------

import os
import pandas as pd
import numpy as np


# ============================================================
# 🔍 NEW DEBUGGING BLOCK — ALWAYS PRINTS IN STREAMLIT CLOUD
# ============================================================
def debug_environment(path):
    print("\n================ DEBUG: data_loader.py ================")
    print("Working directory:", os.getcwd())
    print("Root directory contents:", os.listdir("."))

    # Check data folder
    print("\nDoes data/ folder exist?", os.path.isdir("data"))
    if os.path.isdir("data"):
        print("Contents of data/:", os.listdir("data"))

    print("\nRequested Excel path:", path)
    print("Does Excel file exist?", os.path.exists(path))
    print("========================================================\n")


def _clean_columns(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _ensure_datetime(df, col):
    df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def load_case_study_data(path):
    """
    Load and clean the Excel workbook with the Deviation, BVOL, and OHLC sheets.
    Returns (dev_df, bvol_df, ohlc_df)
    """

    # 🔍 PRINT DEBUG INFO
    debug_environment(path)

    if not os.path.exists(path):
        print("❌ DEBUG: Excel file missing. Returning (None, None, None)")
        return None, None, None

    # ------------------------
    # Load Excel
    # ------------------------
    try:
        xls = pd.ExcelFile(path)
        print("✔ DEBUG: Excel workbook loaded successfully")
        print("Sheets:", xls.sheet_names)
    except Exception as e:
        print("❌ DEBUG: Failed to load Excel:", str(e))
        return None, None, None

    # ------------------------
    # DEVIATION SHEET
    # ------------------------
    if "Deviation" not in xls.sheet_names:
        print("❌ DEBUG: No Deviation sheet found")
        return None, None, None

    dev_df = pd.read_excel(xls, "Deviation")
    dev_df = _clean_columns(dev_df)

    required_dev = ["Date", "XRT Price", "Deviation"]
    if any(col not in dev_df.columns for col in required_dev):
        print("❌ DEBUG: Deviation sheet missing required columns")
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

    # Compute forward 20-day returns
    price_series = dev_df["xrt"].astype(float)
    dev_df["fwd20"] = price_series.shift(-20) / price_series - 1.0

    print("✔ DEBUG: Deviation sheet cleaned")

    # ------------------------
    # BVOL SHEET
    # ------------------------
    if "BVOL" in xls.sheet_names:
        bvol_df = pd.read_excel(xls, "BVOL")
    elif "BW Test Pack Data" in xls.sheet_names:
        bvol_df = pd.read_excel(xls, "BW Test Pack Data")
    else:
        print("⚠ DEBUG: No BVOL sheet found — returning deviation only")
        return dev_df, None, None

    bvol_df = _clean_columns(bvol_df)

    required_bvol = ["Date", "XRT Price", "Bvol"]
    if any(col not in bvol_df.columns for col in required_bvol):
        print("❌ DEBUG: BVOL sheet missing required columns")
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

    print("✔ DEBUG: BVOL sheet cleaned")

    # ------------------------
    # OHLC SHEET
    # ------------------------
    ohlc_df = None

    if "XRT_OHLC" in xls.sheet_names:
        ohlc_df = pd.read_excel(xls, "XRT_OHLC")
        ohlc_df = _clean_columns(ohlc_df)

        # Find date col
        date_col = next((c for c in ohlc_df.columns if c.lower() == "date"), None)

        if date_col is None:
            print("⚠ DEBUG: OHLC sheet missing Date column — ignoring OHLC")
            ohlc_df = None
        else:
            ohlc_df = ohlc_df.rename(columns={date_col: "date"})
            ohlc_df.columns = [c.lower() for c in ohlc_df.columns]

            ohlc_df = _ensure_datetime(ohlc_df, "date")
            ohlc_df = ohlc_df.dropna(subset=["date"]).sort_values("date")
            ohlc_df = ohlc_df.set_index("date")

            print("✔ DEBUG: OHLC sheet cleaned")

    # ------------------------
    # MERGE BVOL WITH OHLC
    # ------------------------
    if ohlc_df is not None:
        bvol_df = bvol_df.reindex(ohlc_df.index)

        for col in ["open", "high", "low", "close", "volume"]:
            if col in ohlc_df.columns:
                bvol_df[col] = ohlc_df[col]

        # Remove duplicates
        bvol_df = bvol_df[~bvol_df.index.duplicated(keep="first")]

        print("✔ DEBUG: BVOL merged with OHLC")

    print("✔ DEBUG: Finished loading all data successfully")
    return dev_df, bvol_df, ohlc_df
