"""

Updated on Mar 25 2026

@author: Andrew Lawson

"""

# Import os for directory and file handling
import os

# Import OpenCV for reading images and writing tiled outputs
import cv2


# Directory containing the original full-size raw frames
INPUT_DIR = "../firebrand_yolo/dataset_eval/data_raw_frames"

# Output directories for tiled training and validation images
OUT_TRAIN = "../dataset_small/images/train"
OUT_VAL   = "../dataset_small/images/val"

# Tile size used for splitting each frame
TILE = 640

# Horizontal stride between adjacent tiles
# 480 px corresponds to 25% overlap for 640 px tiles
STRIDE_X = 480

# Vertical stride between adjacent tiles
# 440 px is chosen so that y=0 and y=440 cover a 1080 px frame exactly
STRIDE_Y = 440

# Simple train/validation split rule:
# every 5th frame is sent to validation, the rest to training
VAL_EVERY = 5


# Create output directories if they do not already exist
os.makedirs(OUT_TRAIN, exist_ok=True)
os.makedirs(OUT_VAL, exist_ok=True)


def tile_positions_1920x1080():
    """
    Return the fixed tile start positions for a 1920x1080 image.

    This setup gives:
    - 4 tiles across the width
    - 2 tiles down the height

    With TILE = 640, this produces 8 tiles per frame.
    """
    x_starts = [0, 480, 960, 1280]  # 4 tiles across
    y_starts = [0, 440]             # 2 tiles down
    return x_starts, y_starts


# Get the x and y tile start positions
x_starts, y_starts = tile_positions_1920x1080()


# Collect all image files in the input directory
img_files = sorted([
    f for f in os.listdir(INPUT_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])


# Loop through every input image
for idx, fname in enumerate(img_files):
    # Full path to the current image
    path = os.path.join(INPUT_DIR, fname)

    # Read the image using OpenCV
    img = cv2.imread(path)

    # Skip unreadable files
    if img is None:
        print(f"Skipping unreadable: {fname}")
        continue

    # Get image height and width
    h, w = img.shape[:2]

    # This script is designed only for 1920x1080 images
    # Skip anything with a different size
    if (w, h) != (1920, 1080):
        print(f"Warning: {fname} is {w}x{h}, expected 1920x1080. Skipping for now.")
        continue

    # Send every 5th full frame to validation, others to training
    out_dir = OUT_VAL if (idx % VAL_EVERY == 0) else OUT_TRAIN

    # Filename without extension
    stem = os.path.splitext(fname)[0]

    # Counter used to number tiles within the frame
    tile_id = 0

    # Loop over all tile starting positions
    for ys in y_starts:
        for xs in x_starts:
            # Crop one tile from the full image
            tile = img[ys:ys + TILE, xs:xs + TILE]

            # Skip incomplete tiles, though positions are chosen to avoid this
            if tile.shape[0] != TILE or tile.shape[1] != TILE:
                continue

            # Create output tile filename that stores its original frame and location
            out_name = f"{stem}_x{xs}_y{ys}_t{tile_id:02d}.jpg"

            # Save the tile to the chosen train/val folder
            cv2.imwrite(os.path.join(out_dir, out_name), tile)

            # Increment tile counter
            tile_id += 1


# Print final summary once complete
print(f"Done. Created tiles from {len(img_files)} frames.")