import os
import shutil

# Define paths
TEMP_DIR = "data/viton_hd/temp_extracted"
DEST_DIR = "data/viton_hd"

def finish_moving():
    if not os.path.exists(TEMP_DIR):
        print(f"Error: {TEMP_DIR} does not exist. Did you verify the unzip finished?")
        return

    print("Searching for extracted data...")
    # Find where 'train' and 'test' are hiding inside temp_extracted
    source_root = None
    for root, dirs, files in os.walk(TEMP_DIR):
        if "train" in dirs and "test" in dirs:
            source_root = root
            break

    if not source_root:
        print("Could not find train/test folders. Please check data/viton_hd/temp_extracted manually.")
        return

    print(f"Found data in: {source_root}")

    # Move folders
    for folder in ["train", "test"]:
        src = os.path.join(source_root, folder)
        dst = os.path.join(DEST_DIR, folder)

        print(f"Moving {folder} to {dst}...")
        if os.path.exists(src):
            shutil.move(src, dst)
        else:
            print(f"Warning: {folder} not found in source.")

    # Move pairs text files if present
    for txt in ["train_pairs.txt", "test_pairs.txt"]:
        src = os.path.join(source_root, txt)
        dst = os.path.join(DEST_DIR, txt)
        if os.path.exists(src):
            shutil.move(src, dst)

    # Cleanup
    print("Cleaning up temp folder...")
    shutil.rmtree(TEMP_DIR)
    print("Done! Data is ready.")

if __name__ == "__main__":
    finish_moving()
