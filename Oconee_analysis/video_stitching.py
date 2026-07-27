"""
Updated on June 14 2026

@author: Andrew Lawson

Oconee RGB video stitching workflow

Purpose
-------
This script finds all RGB AVI video segments in a selected emberometer folder,
orders them using the numeric segment suffix at the end of each filename, checks
for missing or duplicate segments, prints the proposed order for manual review,
and then joins the clips into one final RGB video using FFmpeg.

The script was written to avoid the common sorting problem where filenames are
ordered alphabetically rather than numerically. For example, without numeric
sorting, segment "_19" could incorrectly appear before segment "_2".

Workflow
--------
1. Check that the selected folder exists.
2. Find AVI files whose names begin with "rgb_video".
3. Extract the final integer from each filename.
4. Sort the clips using that integer.
5. Check for missing and duplicate segment numbers.
6. Print the proposed order for manual inspection.
7. Ask the user to confirm before stitching.
8. Create a temporary FFmpeg concat-list file.
9. Join all clips into final_rgb_video.avi.
10. Delete the temporary concat-list file.

Notes
-----
- USE_STREAM_COPY = True performs fast, lossless joining and is preferred where
  all source clips use compatible codecs and stream parameters.
- If stream copy fails, set USE_STREAM_COPY = False. The script will then
  re-encode the clips, which is slower but more tolerant of inconsistencies.
- FFmpeg must be installed and available from Command Prompt / PowerShell.
"""

from pathlib import Path
import re
import subprocess
import sys


# ============================================================
# USER SETTINGS
# ============================================================

# Folder containing the RGB AVI segments for ONE emberometer recording.
#
# Only AVI files beginning with "rgb_video" will be considered, so unrelated
# AVI files in the same folder will not be included in the stitched output.
VIDEO_FOLDER = Path(
    r"path"
)

# Name and location of the final stitched video.
OUTPUT_FILE = VIDEO_FOLDER / "final_rgb_video.avi"

# Stitching mode:
#
# True:
#   - use FFmpeg stream copy
#   - fast
#   - no quality loss
#   - requires all clips to have compatible stream properties
#
# False:
#   - re-encode all clips using MPEG-4
#   - slower
#   - more tolerant of clip inconsistencies
USE_STREAM_COPY = True


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_segment_number(video_path: Path) -> int:
    """
    Extract the final integer from an RGB video filename.

    The final numeric suffix is used as the true sequence number when sorting
    clips. Converting it to an integer prevents alphabetical sorting errors.

    Examples
    --------
    rgb_video_D20260409_T123121_2.avi  -> 2
    rgb_video_D20260409_T123121_19.avi -> 19

    Parameters
    ----------
    video_path : Path
        Path to one RGB video segment.

    Returns
    -------
    int
        Numeric segment suffix.

    Raises
    ------
    ValueError
        If no final numeric suffix can be found.
    """

    # Search the filename stem for an underscore followed by digits at the end.
    match = re.search(r"_(\d+)$", video_path.stem)

    # Reject filenames that do not follow the expected naming convention.
    if match is None:
        raise ValueError(
            f"Could not find a numeric suffix in: {video_path.name}"
        )

    # Convert the matched digits to an integer so that, for example,
    # segment 2 is correctly ordered before segment 19.
    return int(match.group(1))


def escape_for_ffmpeg_concat(path: Path) -> str:
    """
    Convert and escape a filepath for use in an FFmpeg concat-list file.

    FFmpeg's concat demuxer expects file entries in the form:

        file 'C:/path/to/video.avi'

    Single quotes inside paths must therefore be escaped.

    Parameters
    ----------
    path : Path
        Path to a source video segment.

    Returns
    -------
    str
        Escaped absolute path suitable for the concat-list file.
    """

    # Resolve to an absolute path and use forward slashes, which FFmpeg accepts
    # reliably on Windows.
    absolute_path = path.resolve().as_posix()

    # Escape any single quote characters in the path.
    return absolute_path.replace("'", r"'\''")


# ============================================================
# CHECK INPUT FOLDER
# ============================================================

# Stop immediately if the selected folder does not exist.
if not VIDEO_FOLDER.exists():
    sys.exit(f"Folder does not exist:\n{VIDEO_FOLDER}")


# ============================================================
# FIND RGB VIDEO SEGMENTS
# ============================================================

# Find only files that:
#   1. are regular files,
#   2. have the .avi extension,
#   3. begin with "rgb_video".
#
# This avoids accidentally including final_rgb_video.avi or unrelated videos.
videos = [
    path
    for path in VIDEO_FOLDER.iterdir()
    if path.is_file()
    and path.suffix.lower() == ".avi"
    and path.name.lower().startswith("rgb_video")
]

# Stop if no suitable RGB segments were found.
if not videos:
    sys.exit(
        f"No AVI files starting with 'rgb_video' were found in:\n"
        f"{VIDEO_FOLDER}"
    )


# ============================================================
# SORT SEGMENTS NUMERICALLY
# ============================================================

# Sort by the final numeric suffix rather than by the filename text itself.
#
# This specifically prevents ordering problems such as:
#   _1, _10, _11, _2, _20 ...
#
# and instead produces:
#   _1, _2, _3, ... _10, _11 ...
try:
    videos.sort(key=get_segment_number)

except ValueError as error:
    # Stop if any filename does not contain the expected numeric suffix.
    sys.exit(str(error))

# Store the extracted segment numbers for integrity checks.
segment_numbers = [
    get_segment_number(video)
    for video in videos
]


# ============================================================
# CHECK FOR DUPLICATE AND MISSING SEGMENTS
# ============================================================

# Find any segment number that appears more than once.
#
# Duplicate numbers may indicate:
#   - duplicated recordings,
#   - copied files,
#   - incorrect filenames.
duplicates = sorted({
    number
    for number in segment_numbers
    if segment_numbers.count(number) > 1
})

# Determine whether any sequence numbers are absent between the first and last
# detected segments.
#
# Example:
#   available = [1, 2, 3, 5]
#   missing   = [4]
missing = [
    number
    for number in range(
        segment_numbers[0],
        segment_numbers[-1] + 1
    )
    if number not in segment_numbers
]


# ============================================================
# PRINT PROPOSED ORDER FOR MANUAL REVIEW
# ============================================================

# Print the complete order before any output video is created.
#
# This is an important quality-control step because time-series analyses such as
# firebrand count, cumulative count and thermocouple alignment all depend on the
# video clips being stitched in the correct temporal sequence.
print("\n" + "=" * 100)
print("PROPOSED RGB VIDEO ORDER")
print("=" * 100)

for position, video in enumerate(videos, start=1):

    # Extract the actual segment number for display.
    segment_number = get_segment_number(video)

    print(
        f"{position:4d}. "
        f"Segment {segment_number:4d} | "
        f"{video.name}"
    )

print("=" * 100)

# Print a compact summary of the detected sequence.
print(f"Total RGB segments: {len(videos)}")
print(f"First segment:      {segment_numbers[0]}")
print(f"Last segment:       {segment_numbers[-1]}")

# Report duplicate sequence numbers.
if duplicates:
    print(
        f"\nWARNING: duplicate segment numbers: "
        f"{duplicates}"
    )

# Report missing sequence numbers.
if missing:
    print(
        f"\nWARNING: missing segment numbers: "
        f"{missing}"
    )

# Confirm the sequence appears continuous if neither problem was found.
if not duplicates and not missing:
    print(
        "\nNo missing or duplicate segment numbers detected."
    )

# Show where the final stitched video will be written.
print(f"\nOutput file:\n{OUTPUT_FILE}")


# ============================================================
# CONFIRM BEFORE STITCHING
# ============================================================

# Require an explicit manual confirmation after the proposed order has been
# reviewed. This prevents accidental creation of a long incorrectly ordered
# video.
confirmation = input(
    "\nReview the order above.\n"
    "Type YES to stitch the RGB videos, "
    "or anything else to cancel: "
).strip()

# Cancel safely unless the user explicitly enters YES.
if confirmation.upper() != "YES":
    print("\nCancelled. No video was created.")
    sys.exit(0)

# If duplicate segment numbers were detected, require a second confirmation
# before continuing because the final video may contain duplicated footage.
if duplicates:

    second_confirmation = input(
        "\nDuplicate segment numbers were detected.\n"
        "Type CONTINUE to stitch anyway: "
    ).strip()

    if second_confirmation.upper() != "CONTINUE":
        print(
            "\nCancelled because duplicate segments were detected."
        )
        sys.exit(0)


# ============================================================
# CREATE TEMPORARY FFMPEG CONCAT LIST
# ============================================================

# FFmpeg's concat demuxer reads a text file containing one source clip per line.
concat_file = VIDEO_FOLDER / "rgb_ffmpeg_concat_list.txt"

with concat_file.open(
    "w",
    encoding="utf-8",
    newline="\n"
) as file:

    # Write clips in the already validated numeric order.
    for video in videos:

        escaped_path = escape_for_ffmpeg_concat(video)

        file.write(
            f"file '{escaped_path}'\n"
        )


# ============================================================
# BUILD FFMPEG COMMAND
# ============================================================

if USE_STREAM_COPY:

    # Fast, lossless joining.
    #
    # "-f concat" tells FFmpeg to use the concat demuxer.
    # "-safe 0" permits absolute paths in the concat-list file.
    # "-c copy" copies the encoded streams directly without recompression.
    command = [
        "ffmpeg",
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_file),
        "-c", "copy",
        str(OUTPUT_FILE),
    ]

else:

    # Re-encoding mode.
    #
    # This is slower and introduces another encode step, but may succeed when
    # stream copy fails because source clips do not have perfectly matching
    # stream parameters.
    command = [
        "ffmpeg",
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_file),
        "-c:v", "mpeg4",
        "-q:v", "2",
        "-an",
        str(OUTPUT_FILE),
    ]


# ============================================================
# RUN FFMPEG
# ============================================================

print("\nStarting RGB video stitching...")
print(f"Creating:\n{OUTPUT_FILE}\n")

try:

    # Run FFmpeg and raise an exception if it returns an error code.
    subprocess.run(
        command,
        check=True
    )

except FileNotFoundError:

    # This normally means FFmpeg is either not installed or is not included in
    # the system PATH.
    sys.exit(
        "\nFFmpeg was not found.\n"
        "Install FFmpeg and ensure the 'ffmpeg' command is available "
        "in Command Prompt or PowerShell."
    )

except subprocess.CalledProcessError:

    # Stream-copy failures can occur if the source video clips have mismatched
    # codecs or other stream properties.
    sys.exit(
        "\nFFmpeg could not stitch the RGB videos.\n"
        "Try setting USE_STREAM_COPY = False and run the script again."
    )

finally:

    # Always remove the temporary concat-list file, whether FFmpeg succeeds or
    # fails.
    concat_file.unlink(
        missing_ok=True
    )


# ============================================================
# COMPLETE
# ============================================================

print(
    "\nRGB video stitching completed successfully."
)