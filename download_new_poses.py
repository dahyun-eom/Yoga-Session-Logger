import os
import urllib.request
import time

# 30 most common yoga poses matched to Yoga-82 filenames
POSES = {
    "happy_baby":        "Yoga-82/yoga_dataset_links/Happy_Baby_Pose_or_Ananda_Balasana_.txt",
    "legs_up_wall":      "Yoga-82/yoga_dataset_links/Legs-Up-the-Wall_Pose_or_Viparita_Karani_.txt",
    "handstand":         "Yoga-82/yoga_dataset_links/Handstand_pose_or_Adho_Mukha_Vrksasana_.txt",
    "plow":              "Yoga-82/yoga_dataset_links/Plow_Pose_or_Halasana_.txt",
    "staff":             "Yoga-82/yoga_dataset_links/Staff_Pose_or_Dandasana_.txt",
    "bound_angle":       "Yoga-82/yoga_dataset_links/Bound_Angle_Pose_or_Baddha_Konasana_.txt",
    "reclining_toe":     "Yoga-82/yoga_dataset_links/Reclining_Hand-to-Big-Toe_Pose_or_Supta_Padangusthasana_.txt",
    "wide_forward":      "Yoga-82/yoga_dataset_links/Wide-Legged_Forward_Bend_pose_or_Prasarita_Padottanasana_.txt",
    "gate":              "Yoga-82/yoga_dataset_links/Gate_Pose_or_Parighasana_.txt",
    "upward_plank":      "Yoga-82/yoga_dataset_links/Upward_Plank_Pose_or_Purvottanasana_.txt",
}

# download limit per pose (to keep it manageable)
MAX_PER_POSE = 200

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