"""

Updated on Mar 25 2026

@author: Andrew Lawson

"""

# Import pandas for reading and manipulating Excel data
import pandas as pd

# Import matplotlib for plotting graphs
import matplotlib.pyplot as plt

# Import Path for cleaner file path handling
from pathlib import Path


# Constant deposition area used to normalise firebrand counts
# Units: square metres
DEPOSITION_AREA_M2 = 0.03


def plot_5s_rolling_mean_flux_overlay(
    excel_path,
    sheet_names=None,
    time_col="time_s",
    count_col="count_dedup",
    rolling_window_s=5.0,
    area_m2=DEPOSITION_AREA_M2,
    min_periods=1,
    save_path=None,
):
    """
    Read one Excel workbook containing multiple sheets, where each sheet
    represents an individual recording. For each sheet, convert the
    deduplicated firebrand count to firebrand count per m^2, compute a
    5 s rolling mean, and overlay all curves on a single plot.

    Parameters
    ----------
    excel_path : str or Path
        Path to the Excel workbook.
    sheet_names : list[str] or None
        Specific sheet names to plot. If None, all sheets are plotted.
    time_col : str
        Name of the time column in seconds.
    count_col : str
        Name of the deduplicated count column.
    rolling_window_s : float
        Rolling window length in seconds.
    area_m2 : float
        Constant deposition area in m^2 used to normalise firebrand count.
    min_periods : int
        Minimum samples required in the rolling window.
    save_path : str or Path or None
        Optional path to save the output figure.
    """

    # Make sure the normalisation area is physically valid
    if area_m2 <= 0:
        raise ValueError("area_m2 must be greater than zero.")

    # Convert the input path into a Path object
    excel_path = Path(excel_path)

    # Open the Excel workbook so sheet names can be accessed
    xls = pd.ExcelFile(excel_path)

    # If no specific sheets were requested, use every sheet in the workbook
    if sheet_names is None:
        sheet_names = xls.sheet_names

    # Create a new figure for the rolling-mean overlay plot
    plt.figure(figsize=(10, 6))

    # Loop through each selected worksheet
    for sheet in sheet_names:
        # Read the current sheet into a DataFrame
        df = pd.read_excel(excel_path, sheet_name=sheet)

        # Skip the sheet if the required columns are missing
        if time_col not in df.columns or count_col not in df.columns:
            print(f"Skipping '{sheet}': missing '{time_col}' or '{count_col}'")
            continue

        # Keep only the required columns, remove missing values,
        # and create a copy to avoid modifying the original DataFrame
        df = df[[time_col, count_col]].dropna().copy()

        # Sort by time so the rolling mean and plot are in the correct order
        df = df.sort_values(time_col).reset_index(drop=True)

        # At least two points are needed to estimate time spacing properly
        if len(df) < 2:
            print(f"Skipping '{sheet}': not enough data points")
            continue

        # Estimate the time step using the median difference between time values
        dt = df[time_col].diff().median()

        # Skip the sheet if the time spacing is invalid
        if pd.isna(dt) or dt <= 0:
            print(f"Skipping '{sheet}': invalid time spacing")
            continue

        # Convert the requested rolling window in seconds into a number of samples
        window_samples = max(1, int(round(rolling_window_s / dt)))

        # Convert frame-wise deduplicated count into count per m^2
        # by dividing by the constant deposition area
        df["count_per_m2"] = df[count_col] / area_m2

        # Compute the rolling mean of firebrand count per m^2
        # This smooths short-term fluctuations in the signal
        df["rolling_mean_5s_per_m2"] = (
            df["count_per_m2"]
            .rolling(window=window_samples, min_periods=min_periods)
            .mean()
        )

        # Plot the rolling-mean curve for this recording
        plt.plot(
            df[time_col],
            df["rolling_mean_5s_per_m2"],
            linewidth=2,
            label=sheet
        )

    # Label the x-axis
    plt.xlabel("Time (s)")

    # Label the y-axis using LaTeX formatting for m^2
    plt.ylabel(r"Firebrand count per m$^2$ (5 s rolling mean)")

    # Add a title to the plot
    plt.title(r"Deduplicated firebrand count per m$^2$ (5 s rolling mean)")

    # Show a legend using sheet names as labels
    plt.legend()

    # Add a light grid to improve readability
    plt.grid(True, alpha=0.3)

    # Adjust layout to reduce overlapping text
    plt.tight_layout()

    # Save the figure if an output path was provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    # Display the figure
    plt.show()


def plot_cumulative_count_overlay(
    excel_path,
    sheet_names=None,
    time_col="time_s",
    count_col="count_dedup",
    save_path=None,
):
    """
    Read one Excel workbook containing multiple sheets, where each sheet
    represents an individual recording. Compute the cumulative firebrand
    count for each sheet and overlay the curves on one plot.

    Parameters
    ----------
    excel_path : str or Path
        Path to the Excel workbook.
    sheet_names : list[str] or None
        Specific sheet names to plot. If None, all sheets are plotted.
    time_col : str
        Name of the time column in seconds.
    count_col : str
        Name of the deduplicated count column.
    save_path : str or Path or None
        Optional path to save the output figure.
    """

    # Convert the input path into a Path object
    excel_path = Path(excel_path)

    # Open the workbook to access its sheet names
    xls = pd.ExcelFile(excel_path)

    # If no sheets are specified, use all sheets in the workbook
    if sheet_names is None:
        sheet_names = xls.sheet_names

    # Create a new figure for the cumulative-count overlay
    plt.figure(figsize=(10, 6))

    # Loop through each selected worksheet
    for sheet in sheet_names:
        # Read the current sheet into a DataFrame
        df = pd.read_excel(excel_path, sheet_name=sheet)

        # Skip this sheet if the required columns are not present
        if time_col not in df.columns or count_col not in df.columns:
            print(f"Skipping '{sheet}': missing '{time_col}' or '{count_col}'")
            continue

        # Keep only the required columns and remove missing rows
        df = df[[time_col, count_col]].dropna().copy()

        # Sort by time so the cumulative sum progresses correctly
        df = df.sort_values(time_col).reset_index(drop=True)

        # Skip if there are no valid data points
        if len(df) < 1:
            print(f"Skipping '{sheet}': no valid data points")
            continue

        # Compute cumulative firebrand count by summing the counts over time
        df["cumulative_count"] = df[count_col].cumsum()

        # Plot the cumulative count curve for this recording
        plt.plot(
            df[time_col],
            df["cumulative_count"],
            linewidth=2,
            label=sheet
        )

    # Label the x-axis
    plt.xlabel("Time (s)")

    # Label the y-axis
    plt.ylabel("Cumulative firebrand count")

    # Add a title to the plot
    plt.title("Cumulative deduplicated firebrand count")

    # Show a legend using sheet names as labels
    plt.legend()

    # Add a light grid
    plt.grid(True, alpha=0.3)

    # Adjust layout to avoid overlap
    plt.tight_layout()

    # Save the plot if an output path is provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    # Display the figure
    plt.show()


if __name__ == "__main__":
    # Name of the Excel workbook containing all recordings
    excel_file = "Graph_Compilation.xlsx"  # replace with your file name

    # Generate and save the rolling-mean firebrand count per m^2 plot
    plot_5s_rolling_mean_flux_overlay(
        excel_path=excel_file,
        area_m2=DEPOSITION_AREA_M2,
        save_path="rolling_mean_overlay.png"
    )

    # Generate and save the cumulative firebrand count plot
    plot_cumulative_count_overlay(
        excel_path=excel_file,
        save_path="cumulative_count_overlay.png"
    )