import os
import urllib.request
import time

# 6 poses we want
POSES = {
    "warrior_1": "Yoga-82/yoga_dataset_links/Warrior_I_Pose_or_Virabhadrasana_I_.txt",
    "warrior_2": "Yoga-82/yoga_dataset_links/Warrior_II_Pose_or_Virabhadrasana_II_.txt",
    "tree":      "Yoga-82/yoga_dataset_links/Tree_Pose_or_Vrksasana_.txt",
    "downdog":   "Yoga-82/yoga_dataset_links/Downward-Facing_Dog_pose_or_Adho_Mukha_Svanasana_.txt",
    "plank":     "Yoga-82/yoga_dataset_links/Plank_Pose_or_Kumbhakasana_.txt",
    "child":     "Yoga-82/yoga_dataset_links/Child_Pose_or_Balasana_.txt",
}

# download limit per pose (to keep it manageable)
MAX_PER_POSE = 100

for pose_name, txt_file in POSES.items():
    print(f"\nDownloading {pose_name}...")
    
    # create folder
    save_dir = f"dataset/{pose_name}"
    os.makedirs(save_dir, exist_ok=True)
    
    count = 0
    with open(txt_file, "r") as f:
        for line in f:
            if count >= MAX_PER_POSE:
                break
            
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            
            filename, url = parts[0], parts[1]
            save_path = f"{save_dir}/{os.path.basename(filename)}"
            
            # skip if already downloaded
            if os.path.exists(save_path):
                count += 1
                continue
            
            try:
                urllib.request.urlretrieve(url, save_path)
                print(f"  [{count+1}/{MAX_PER_POSE}] {os.path.basename(filename)}")
                count += 1
                time.sleep(0.1)  # be polite to servers
            except Exception as e:
                print(f"  skipped {os.path.basename(filename)}: {e}")

print("\nDone! Check the dataset/ folder.")