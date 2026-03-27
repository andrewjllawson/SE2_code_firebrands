"""

Updated on Mar 25 2026

@author: Andrew Lawson

"""

# Import pandas for reading/writing Excel files and handling tabular data
import pandas as pd

# Import numpy for conditional calculations and NaN handling
import numpy as np

# ---------------- USER INPUTS ----------------
# Input Excel workbook containing the raw comparison data
input_excel = "results_input.xlsx"

# Name of the worksheet to read from the input workbook
sheet_name = "results_input"

# Output Excel workbook where the processed results will be saved
output_excel = "comparison_results.xlsx"
# --------------------------------------------


# Read the specified worksheet from the input Excel file into a DataFrame
df = pd.read_excel(input_excel, sheet_name=sheet_name)

# List of columns required for the calculations
required_cols = [
    "video_id",
    "method",
    "predicted_detections",
    "false_positives",
    "false_negatives",
]

# Check whether any required columns are missing from the input sheet
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

# Standardise the method column:
# - convert to string
# - remove leading/trailing spaces
# - convert to lowercase
# This avoids issues caused by inconsistent naming such as "ML", " ml ", etc.
df["method"] = df["method"].astype(str).str.strip().str.lower()

# Compute the number of true positives:
# predicted detections minus detections known to be false positives
df["true_positives"] = df["predicted_detections"] - df["false_positives"]

# Compute the ground-truth count:
# true positives plus false negatives
# (i.e. all real objects that should have been detected)
df["ground_truth_count"] = df["true_positives"] + df["false_negatives"]

# Compute precision:
# proportion of predicted detections that were correct
# Avoid division by zero by assigning NaN where predicted_detections is zero
df["precision"] = np.where(
    df["predicted_detections"] > 0,
    df["true_positives"] / df["predicted_detections"],
    np.nan
)

# Compute recall:
# proportion of real objects that were successfully detected
# Avoid division by zero by assigning NaN where ground_truth_count is zero
df["recall"] = np.where(
    df["ground_truth_count"] > 0,
    df["true_positives"] / df["ground_truth_count"],
    np.nan
)

# Compute F1 score:
# harmonic mean of precision and recall
# Only calculate where precision + recall is greater than zero
df["f1_score"] = np.where(
    (df["precision"] + df["recall"]) > 0,
    2 * df["precision"] * df["recall"] / (df["precision"] + df["recall"]),
    np.nan
)

# Compute false discovery rate:
# proportion of predicted detections that were false positives
df["false_discovery_rate"] = np.where(
    df["predicted_detections"] > 0,
    df["false_positives"] / df["predicted_detections"],
    np.nan
)

# Compute false negative rate:
# proportion of real objects missed by the detector
df["false_negative_rate"] = np.where(
    df["ground_truth_count"] > 0,
    df["false_negatives"] / df["ground_truth_count"],
    np.nan
)

# Compute absolute count error:
# difference between predicted detections and actual ground-truth count
df["count_error"] = df["predicted_detections"] - df["ground_truth_count"]

# Compute percentage count error relative to the ground-truth count
df["count_error_percent"] = np.where(
    df["ground_truth_count"] > 0,
    100 * df["count_error"] / df["ground_truth_count"],
    np.nan
)

# Round selected rate-based metrics to 4 decimal places for cleaner output
for col in [
    "precision", "recall", "f1_score",
    "false_discovery_rate", "false_negative_rate",
    "count_error_percent"
]:
    df[col] = df[col].round(4)

# Specify the order of columns for the final per-method output sheet
comparison_cols = [
    "video_id",
    "method",
    "predicted_detections",
    "false_positives",
    "false_negatives",
    "true_positives",
    "ground_truth_count",
    "precision",
    "recall",
    "f1_score",
    "false_discovery_rate",
    "false_negative_rate",
    "count_error",
    "count_error_percent",
]

# Keep only the selected columns in the chosen order
df = df[comparison_cols]

# Create a side-by-side comparison table:
# rows = video IDs
# columns = each metric split by method
pivot = df.pivot(index="video_id", columns="method")

# Flatten the multi-level column names into single strings
# Example: ("precision", "ml") becomes "precision_ml"
pivot.columns = [f"{metric}_{method}" for metric, method in pivot.columns]

# Convert video_id back from index to a normal column
pivot = pivot.reset_index()

# Write both the detailed results and side-by-side comparison to a new Excel workbook
with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
    # Save the full per-method results
    df.to_excel(writer, sheet_name="per_method_results", index=False)

    # Save the side-by-side pivoted comparison
    pivot.to_excel(writer, sheet_name="side_by_side", index=False)

# Print confirmation messages to the console
print("Done.")
print(df)
print(f"Saved to {output_excel}")