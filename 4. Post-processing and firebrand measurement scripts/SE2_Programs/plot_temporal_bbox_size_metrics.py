"""

Updated on Mar 25 2026

@author: Andrew Lawson

This script reads a single sheet of firebrand detection data from an Excel
workbook, converts bounding-box area from px^2 to mm^2 using a specified
calibration, groups detections by time, and plots the mean apparent
firebrand size over time on dual y-axes.

Outputs:
- temporal plot of mean bounding-box area in px^2 and mm^2

Expected sheet columns:
- area_px
- and either:
    - time_s
    - or sample_index
"""

# Import pandas for reading and processing Excel data
import pandas as pd

# Import matplotlib for plotting graphs
import matplotlib.pyplot as plt

# Import Path for robust file path handling
from pathlib import Path

# ---------------- USER SETTINGS ----------------

# Path to the Excel workbook containing the detection data
EXCEL_PATH = r"single_detect.xlsx"

# Name of the sheet to read from the workbook
SHEET_NAME = "detections_by_frame"

# Filename for the saved output plot
OUTPUT_PLOT = r"bbox_area_px_and_mm2_over_time_raw.png"

# Pixel-to-millimetre calibration in the x direction
PX_PER_MM_X = 10.7

# Pixel-to-millimetre calibration in the y direction
PX_PER_MM_Y = 8.0

# Sampling rate used only if time_s is missing and time must be derived
# from sample_index
FPS_SAMPLE = 2.0

# ----------------------------------------------

# Conversion factor from pixel area to mm^2
PX2_PER_MM2 = PX_PER_MM_X * PX_PER_MM_Y


def load_single_sheet(excel_path: str, sheet_name: str) -> pd.DataFrame:
    """
    Load one sheet from the Excel workbook, validate the required columns,
    clean the area data, and convert area from px^2 to mm^2.
    """
    # Read the selected sheet from the workbook
    df = pd.read_excel(excel_path, sheet_name=sheet_name)

    # area_px must be present for the script to work
    required = {"area_px"}
    if not required.issubset(df.columns):
        raise ValueError(f"Sheet '{sheet_name}' must contain at least: {required}")

    # Convert area_px to numeric values, forcing invalid entries to NaN
    df["area_px"] = pd.to_numeric(df["area_px"], errors="coerce")

    # Drop rows where area_px could not be interpreted
    df = df.dropna(subset=["area_px"])

    # Keep only positive bounding-box areas
    df = df[df["area_px"] > 0].copy()

    # Convert bounding-box area from px^2 to mm^2
    df["area_mm2"] = df["area_px"] / PX2_PER_MM2

    return df


def build_time_series(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """
    Build a grouped time series of mean bounding-box area in both px^2 and mm^2.

    Preference is given to the existing time_s column. If time_s is missing,
    time is derived from sample_index using FPS_SAMPLE.
    """
    # Use the time_s column directly if it exists
    if "time_s" in df.columns:
        df["time_s"] = pd.to_numeric(df["time_s"], errors="coerce")
        df = df.dropna(subset=["time_s"]).copy()
        time_col = "time_s"

    # Otherwise derive time from sample_index
    elif "sample_index" in df.columns:
        df["sample_index"] = pd.to_numeric(df["sample_index"], errors="coerce")
        df = df.dropna(subset=["sample_index"]).copy()
        df["time_s"] = df["sample_index"] / FPS_SAMPLE
        time_col = "time_s"

    # Fail if neither time column is available
    else:
        raise ValueError("Need either 'time_s' or 'sample_index' column.")

    # Group detections by time and compute the mean area in both units
    grouped = (
        df.groupby(time_col)
        .agg(
            mean_area_px=("area_px", "mean"),
            mean_area_mm2=("area_mm2", "mean")
        )
        .reset_index()
        .sort_values(time_col)
    )

    return grouped, time_col


def plot_dual_axis_raw(ts: pd.DataFrame, time_col: str, output_plot: str) -> None:
    """
    Plot mean bounding-box area over time on dual y-axes:
    - left axis: px^2
    - right axis: mm^2
    """
    # Create figure and primary axis
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Plot mean area in px^2 on the left axis
    line1 = ax1.plot(
        ts[time_col],
        ts["mean_area_px"],
        label="Mean area (px$^2$)"
    )
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Mean bbox area (px$^2$)")
    ax1.set_title("Mean firebrand size over time")

    # Create secondary y-axis for mm^2
    ax2 = ax1.twinx()

    # Plot mean area in mm^2 on the right axis
    line2 = ax2.plot(
        ts[time_col],
        ts["mean_area_mm2"],
        label="Mean area (mm$^2$)"
    )
    ax2.set_ylabel("Mean bbox area (mm$^2$)")

    # Combine line handles so one legend can describe both axes
    lines = line1 + line2
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="best")

    # Adjust layout to avoid clipping labels
    plt.tight_layout()

    # Save the figure to file
    plt.savefig(output_plot, dpi=300)

    # Display the figure
    plt.show()


def main():
    """
    Main workflow:
    1. Check that the Excel file exists
    2. Load the chosen sheet
    3. Build the grouped time series
    4. Plot the dual-axis graph
    5. Print the saved output filename
    """
    # Convert Excel path string into a Path object
    excel_path = Path(EXCEL_PATH)

    # Check that the workbook exists before trying to read it
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    # Load and clean the selected sheet
    df = load_single_sheet(str(excel_path), SHEET_NAME)

    # Build the time-series data used for plotting
    ts, time_col = build_time_series(df)

    # Generate and save the plot
    plot_dual_axis_raw(ts, time_col, OUTPUT_PLOT)

    # Confirm where the plot was saved
    print(f"Plot saved to: {OUTPUT_PLOT}")


# Run the script only when executed directly
if __name__ == "__main__":
    main()