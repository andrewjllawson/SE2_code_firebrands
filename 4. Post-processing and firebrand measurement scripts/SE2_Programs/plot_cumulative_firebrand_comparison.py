"""

Updated on Mar 25 2026

@author: Andrew Lawson

"""

# Import pandas for reading CSV files and organising tabular data
import pandas as pd

# Import matplotlib for plotting the cumulative detection curves
import matplotlib.pyplot as plt

# Import Path so the saved file location can be printed clearly
from pathlib import Path

# =========================
# USER INPUTS
# =========================

# Manually reviewed total number of firebrands for comparison
MANUAL_TOTAL = 622

# Input CSV containing the classical computer-vision tracking results
CLASSICAL_CSV = "classical_cv_results.csv"

# Input CSV containing the machine-learning detection results
ML_CSV = "ml_results.csv"

# Name of the output plot image
OUTPUT_PNG = "firebrand_cumulative_comparison.png"

# =========================
# LOAD DATA
# =========================

# Read the classical CV results from CSV
df_classical = pd.read_csv(CLASSICAL_CSV)

# Read the ML results from CSV
df_ml = pd.read_csv(ML_CSV)

# =========================
# CLASSICAL CV:
# Use first appearance time of each tracked object
# =========================

# For the classical CV method, tracked firebrands may appear in multiple rows.
# Group by object_id and take the earliest recorded time for each object so that
# each tracked firebrand is counted only once.
classical_first_times = (
    df_classical.groupby("object_id")["rgb_time"]
    .min()
    .sort_values()
    .reset_index(drop=True)
)

# Create a cumulative count table for the classical CV detections
# The count increases by one each time a new tracked object first appears
classical_cum = pd.DataFrame({
    "time_s": classical_first_times,
    "count": range(1, len(classical_first_times) + 1)
})

# =========================
# ML:
# Assume each row is a detection event at time_s
# Sort by time and accumulate detections
# =========================

# For the ML results, assume each row already represents one detection event.
# Sort the detection times so the cumulative total can be built over time.
ml_times = df_ml["time_s"].sort_values().reset_index(drop=True)

# Create a cumulative count table for ML detections
ml_cum = pd.DataFrame({
    "time_s": ml_times,
    "count": range(1, len(ml_times) + 1)
})

# =========================
# MANUAL TOTAL LINE
# =========================

# Find the largest time value across both methods so the manual reference line
# spans the full width of the plot
max_time = max(
    classical_cum["time_s"].max() if not classical_cum.empty else 0,
    ml_cum["time_s"].max() if not ml_cum.empty else 0
)

# Define x- and y-values for a horizontal line showing the manual total estimate
manual_line_x = [0, max_time]
manual_line_y = [MANUAL_TOTAL, MANUAL_TOTAL]

# =========================
# PLOT
# =========================

# Create the figure
plt.figure(figsize=(9, 5))

# Plot cumulative detections from the classical CV pipeline as a step curve
plt.step(
    classical_cum["time_s"],
    classical_cum["count"],
    where="post",
    label="Classical CV Count"
)

# Plot cumulative detections from the ML pipeline as a step curve
plt.step(
    ml_cum["time_s"],
    ml_cum["count"],
    where="post",
    label="ML Count"
)

# Plot the manually reviewed total as a horizontal reference line
plt.plot(
    manual_line_x,
    manual_line_y,
    label=f"Manual Estimated Total ({MANUAL_TOTAL})"
)

# Label the axes and title
plt.xlabel("Time (s)")
plt.ylabel("Firebrands Detected")
plt.title("Cumulative Firebrand Detections vs Manual Estimate")

# Force axes to start at zero
plt.xlim(left=0)
plt.ylim(bottom=0)

# Show the legend and tidy the layout
plt.legend()
plt.tight_layout()

# Save the figure to file
plt.savefig(OUTPUT_PNG, dpi=300)

# Display the figure in an interactive session
plt.show()

# Print useful summary information
print(f"Saved plot to: {Path(OUTPUT_PNG).resolve()}")
print(f"Classical CV total: {len(classical_first_times)}")
print(f"ML total rows: {len(ml_times)}")
print(f"Manual estimated total: {MANUAL_TOTAL}")