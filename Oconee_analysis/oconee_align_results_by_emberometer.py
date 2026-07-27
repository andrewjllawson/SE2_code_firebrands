"""
Updated on July 02 2026

@author: Andrew Lawson

Oconee result-alignment workflow by emberometer ID

Purpose
-------
This script reorganises analysis outputs from multiple burn / confidence result
folders so that results belonging to the same emberometer ID can be inspected
together.

The expected input structure is a root results folder containing subfolders such
as:

    135_0.4CI/
    156_0.4CI/
    180_0.4CI/

Each result folder may contain:
- an Excel workbook with one sheet per analysed video / emberometer,
- count-versus-time PNG plots,
- cumulative-count PNG plots.

The script creates one output folder per emberometer ID and gathers matching
results across all burns / result folders into that location.

Workflow
--------
1. Scan the root results directory for result folders.
2. Find the main Excel workbook in each result folder.
3. Identify the Excel sheet corresponding to each emberometer ID.
4. Find PNG plots whose filenames correspond to that same emberometer ID.
5. Copy matching plots into an emberometer-specific output folder.
6. Read and combine the matching Excel frame-level data.
7. Preserve the summary information from each source result folder.
8. Create one combined Excel workbook per emberometer.
9. Create a presence-summary workbook showing which emberometer IDs have data
   in each source result folder.

Important assumptions
---------------------
- Emberometer IDs are between MIN_ID and MAX_ID.
- Excel sheet names contain the emberometer ID somewhere in the sheet name.
- Plot filenames contain the emberometer ID as their first number.
- Result-folder names represent the source burn / confidence condition and are
  retained as labels in the combined outputs.
- The script does not modify the original result folders; it copies plots and
  creates new combined workbooks in OUTPUT_DIR.
"""

from pathlib import Path
import re
import shutil
import pandas as pd

# ============================================================
# USER SETTINGS
# ============================================================

# Main folder containing your different burn/confidence result folders
# Example:
# ROOT_RESULTS_DIR/
# ├── 135_0.4CI/
# ├── 136_0.4CI/
# └── 137_0.4CI/
ROOT_RESULTS_DIR = Path(r"C:\Users\Andre\ML_Code\Object_Detection\firebrand_yolo\CI_Analysis")

# Output folder where results will be grouped by emberometer ID
OUTPUT_DIR = ROOT_RESULTS_DIR / "aligned_by_emberometer"

# Emberometer IDs to output
MIN_ID = 1
MAX_ID = 24

# Excel files to ignore
IGNORE_EXCEL_KEYWORDS = [
    "presence",
    "combined",
    "aligned"
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def extract_first_number(text):
    """
    Extract the first integer found in a string.

    Examples:
    '3_count_vs_time' -> 3
    'recording_03_trimmed' -> 3
    'Sheet 24' -> 24
    """

    match = re.search(r"\d+", str(text))

    if match:
        return int(match.group())

    return None


def clean_excel_sheet_name(name):
    """
    Make a safe Excel sheet name.
    Excel sheet names:
    - max 31 characters
    - cannot contain: \\ / * ? : [ ]
    """

    bad_chars = ["\\", "/", "*", "?", ":", "[", "]"]

    for char in bad_chars:
        name = name.replace(char, "_")

    return name[:31]


def find_result_folders(root_dir):
    """
    Find folders that appear to contain results.

    A result folder is any folder inside ROOT_RESULTS_DIR that contains:
    - at least one .xlsx file, or
    - at least one .png file

    This skips the output folder.
    """

    result_folders = []

    for item in root_dir.iterdir():

        if not item.is_dir():
            continue

        if item.name == OUTPUT_DIR.name:
            continue

        has_excel = any(item.glob("*.xlsx"))
        has_png = any(item.glob("*.png"))

        if has_excel or has_png:
            result_folders.append(item)

    return sorted(result_folders)


def find_excel_file(result_folder):
    """
    Find the main Excel results file inside a result folder.

    This accepts filenames such as:
    - 135_0.4CI.xlsx
    - 136_0.4CI.xlsx
    - firebrand_results_all_videos.xlsx

    It ignores temporary Excel files and combined output files.
    """

    excel_files = []

    for excel_file in result_folder.glob("*.xlsx"):

        name_lower = excel_file.name.lower()

        # Ignore temporary Excel lock files
        if excel_file.name.startswith("~$"):
            continue

        # Ignore output/combined files
        if any(keyword in name_lower for keyword in IGNORE_EXCEL_KEYWORDS):
            continue

        excel_files.append(excel_file)

    if not excel_files:
        return None

    # If there are multiple, use the largest file as the likely results workbook
    excel_files = sorted(excel_files, key=lambda p: p.stat().st_size, reverse=True)

    return excel_files[0]


def find_matching_pngs(result_folder, emberometer_id):
    """
    Find count and cumulative count plots for a given emberometer ID.

    This assumes plot names look like:
    - 3_count_vs_time.png
    - 3_cumulative_count_vs_time.png
    - recording_03_count_vs_time.png

    Important:
    It only searches PNGs in the result folder itself.
    """

    matching_pngs = []

    for png_file in result_folder.glob("*.png"):

        file_id = extract_first_number(png_file.stem)

        if file_id == emberometer_id:
            matching_pngs.append(png_file)

    return sorted(matching_pngs)


def find_matching_excel_sheet(excel_path, emberometer_id):
    """
    Find the Excel sheet corresponding to one emberometer ID.

    This ignores the Excel filename, because filenames like 135_0.4CI.xlsx
    contain the burn number and confidence threshold, not the emberometer ID.

    It uses the sheet names instead.
    """

    try:
        workbook = pd.ExcelFile(excel_path)
    except Exception:
        return None

    for sheet_name in workbook.sheet_names:

        # Skip overview sheets
        if str(sheet_name).lower() in ["overview", "summary"]:
            continue

        sheet_id = extract_first_number(sheet_name)

        if sheet_id == emberometer_id:
            return sheet_name

    return None


def read_video_sheet(excel_path, sheet_name, result_label, emberometer_id):
    """
    Read one video sheet from the Excel workbook.

    The previous batch analysis workbook usually has:
    - summary table at the top
    - blank row
    - frame-level table starting at the row containing 'sample_index'

    This function separates those two sections.
    """

    try:
        raw_df = pd.read_excel(
            excel_path,
            sheet_name=sheet_name,
            header=None
        )
    except Exception:
        return None, None

    # Locate frame-level data header row
    frame_header_row = None

    for i in range(len(raw_df)):

        row_values = raw_df.iloc[i].astype(str).str.lower().tolist()

        if "sample_index" in row_values:
            frame_header_row = i
            break

    # If no frame table is found, return the raw sheet as summary only
    if frame_header_row is None:

        summary_df = raw_df.copy()
        summary_df.insert(0, "result_folder", result_label)
        summary_df.insert(1, "emberometer_id", emberometer_id)

        return summary_df, None

    # Summary data is above frame table
    summary_df = raw_df.iloc[:frame_header_row].dropna(how="all").copy()
    summary_df.insert(0, "result_folder", result_label)
    summary_df.insert(1, "emberometer_id", emberometer_id)

    # Frame table starts from the sample_index row
    frame_raw = raw_df.iloc[frame_header_row:].reset_index(drop=True)

    # First row becomes header
    frame_raw.columns = frame_raw.iloc[0]

    # Remaining rows are data
    frame_df = frame_raw.iloc[1:].reset_index(drop=True)

    # Remove empty columns
    frame_df = frame_df.dropna(axis=1, how="all")

    # Add identifying columns
    frame_df.insert(0, "result_folder", result_label)
    frame_df.insert(1, "emberometer_id", emberometer_id)

    return summary_df, frame_df


# ============================================================
# MAIN SCRIPT
# ============================================================

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    result_folders = find_result_folders(ROOT_RESULTS_DIR)

    if not result_folders:
        print(f"No result folders found in: {ROOT_RESULTS_DIR}")
        return

    print("Found result folders:")

    for folder in result_folders:
        print(f"  - {folder.name}")

    presence_rows = []

    for emberometer_id in range(MIN_ID, MAX_ID + 1):

        print()
        print(f"Processing emberometer {emberometer_id:02d}")

        ember_output_folder = OUTPUT_DIR / f"emberometer_{emberometer_id:02d}"
        ember_output_folder.mkdir(parents=True, exist_ok=True)

        combined_frame_tables = []
        combined_summary_tables = []

        presence_row = {
            "emberometer_id": emberometer_id
        }

        for result_folder in result_folders:

            result_label = result_folder.name

            print(f"  Checking {result_label}")

            presence_row[result_label] = ""

            # ----------------------------------------------------
            # Copy matching graph images
            # ----------------------------------------------------

            matching_pngs = find_matching_pngs(
                result_folder=result_folder,
                emberometer_id=emberometer_id
            )

            for png_file in matching_pngs:

                new_png_name = f"{result_label}_{png_file.name}"
                destination = ember_output_folder / new_png_name

                shutil.copy2(png_file, destination)

            # ----------------------------------------------------
            # Read matching Excel sheet
            # ----------------------------------------------------

            excel_path = find_excel_file(result_folder)

            if excel_path is None:
                print(f"    No Excel file found in {result_label}")
                continue

            matching_sheet = find_matching_excel_sheet(
                excel_path=excel_path,
                emberometer_id=emberometer_id
            )

            if matching_sheet is None:
                print(f"    No sheet found for emberometer {emberometer_id:02d}")
            else:
                print(f"    Found sheet: {matching_sheet}")

                summary_df, frame_df = read_video_sheet(
                    excel_path=excel_path,
                    sheet_name=matching_sheet,
                    result_label=result_label,
                    emberometer_id=emberometer_id
                )

                if summary_df is not None:
                    combined_summary_tables.append(summary_df)

                if frame_df is not None and not frame_df.empty:
                    combined_frame_tables.append(frame_df)

            # ----------------------------------------------------
            # Mark presence
            # ----------------------------------------------------

            if matching_pngs or matching_sheet is not None:
                presence_row[result_label] = "available"

        presence_rows.append(presence_row)

        # --------------------------------------------------------
        # Write combined Excel for this emberometer
        # --------------------------------------------------------

        if combined_frame_tables or combined_summary_tables:

            output_excel = ember_output_folder / f"emberometer_{emberometer_id:02d}_combined.xlsx"

            with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:

                if combined_frame_tables:

                    all_frames_df = pd.concat(
                        combined_frame_tables,
                        ignore_index=True
                    )

                    all_frames_df.to_excel(
                        writer,
                        sheet_name="all_frame_data",
                        index=False
                    )

                if combined_summary_tables:

                    start_row = 0

                    for summary_df in combined_summary_tables:

                        result_label = str(summary_df["result_folder"].iloc[0])

                        title_df = pd.DataFrame([[f"Summary for {result_label}"]])

                        title_df.to_excel(
                            writer,
                            sheet_name="summaries",
                            index=False,
                            header=False,
                            startrow=start_row
                        )

                        start_row += 2

                        summary_df.to_excel(
                            writer,
                            sheet_name="summaries",
                            index=False,
                            header=False,
                            startrow=start_row
                        )

                        start_row += len(summary_df) + 3

            print(f"  Saved combined Excel: {output_excel.name}")

        else:
            print(f"  No data found for emberometer {emberometer_id:02d}")

    # ------------------------------------------------------------
    # Write presence summary
    # ------------------------------------------------------------

    presence_df = pd.DataFrame(presence_rows)

    presence_output = OUTPUT_DIR / "emberometer_presence_summary.xlsx"

    presence_df.to_excel(presence_output, index=False)

    print()
    print("Done.")
    print(f"Aligned results saved to: {OUTPUT_DIR}")
    print(f"Presence summary saved to: {presence_output}")


if __name__ == "__main__":
    main()