import os
import zipfile
import shutil
from tqdm import tqdm

# --- CONFIGURATION ---
# Path to your zip file in WSL
ZIP_PATH = "/mnt/g/My Drive/virtual/zalando-hd-resized.zip"

# Destination in your project
DEST_DIR = "data/viton_hd"

def prepare_data():
    # 1. Check if G: Drive is mounted
    if not os.path.exists(ZIP_PATH):
        print(f"Error: Could not find file at: {ZIP_PATH}")
        print("Your G: drive might not be mounted.")
        print("Run this command first: sudo mount -t drvfs G: /mnt/g")
        return

    print(f"Found dataset at: {ZIP_PATH}")
    
    # 2. Extract to a temp folder
    temp_extract_path = os.path.join(DEST_DIR, "temp_extracted")
    if os.path.exists(temp_extract_path):
        shutil.rmtree(temp_extract_path)
    os.makedirs(temp_extract_path, exist_ok=True)

    print("Extracting... (This may take a minute)")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        # TQDM gives us a nice progress bar
        for member in tqdm(zip_ref.infolist(), desc="Unzipping"):
            try:
                zip_ref.extract(member, temp_extract_path)
            except zipfile.error as e:
                print(f"Warning: Could not extract {member.filename}: {e}")

    print("Extraction complete. Moving files...")

    # 3. Locate the 'train' and 'test' folders inside the extracted content
    # (Sometimes zips have an outer folder like "zalando-hd-resized (1)/train")
    source_root = temp_extract_path
    
    # Walk down until we find 'train'
    found_root = False
    for root, dirs, files in os.walk(temp_extract_path):
        if "train" in dirs and "test" in dirs:
            source_root = root
            found_root = True
            break
    
    if not found_root:
        print("Critical Error: Could not find 'train' and 'test' folders inside the zip.")
        print(f"Contents found: {os.listdir(temp_extract_path)}")
        return

    # 4. Move train/test to data/viton_hd/
    for folder in ["train", "test"]:
        src = os.path.join(source_root, folder)
        dst = os.path.join(DEST_DIR, folder)
        
        if os.path.exists(dst):
            print(f"Cleaning old {folder} folder...")
            shutil.rmtree(dst)
            
        print(f"Moving {folder} -> {dst}")
        shutil.move(src, dst)

    # 5. Move the pairs text files if they exist
    for txt_file in ["train_pairs.txt", "test_pairs.txt"]:
        src = os.path.join(source_root, txt_file)
        if os.path.exists(src):
            shutil.move(src, os.path.join(DEST_DIR, txt_file))

    # Cleanup
    shutil.rmtree(temp_extract_path)
    print("\nSuccess! Dataset is ready.")
    print(f"Train Data: {os.path.join(DEST_DIR, 'train')}")
    print(f"Test Data:  {os.path.join(DEST_DIR, 'test')}")

if __name__ == "__main__":
    prepare_data()