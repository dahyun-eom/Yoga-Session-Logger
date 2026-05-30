import os
import urllib.request
import time

# 30 most common yoga poses matched to Yoga-82 filenames
POSES = {
    "reverse_warrior": "Yoga-82/yoga_dataset_links/viparita_virabhadrasana_or_reverse_warrior_pose.txt",
    "downdog":           "Yoga-82/yoga_dataset_links/Downward-Facing_Dog_pose_or_Adho_Mukha_Svanasana_.txt",
    "child":             "Yoga-82/yoga_dataset_links/Child_Pose_or_Balasana_.txt",
    "warrior_1":         "Yoga-82/yoga_dataset_links/Warrior_I_Pose_or_Virabhadrasana_I_.txt",
    "warrior_2":         "Yoga-82/yoga_dataset_links/Warrior_II_Pose_or_Virabhadrasana_II_.txt",
    "warrior_3":         "Yoga-82/yoga_dataset_links/Warrior_III_Pose_or_Virabhadrasana_III_.txt",
    "tree":              "Yoga-82/yoga_dataset_links/Tree_Pose_or_Vrksasana_.txt",
    "plank":             "Yoga-82/yoga_dataset_links/Plank_Pose_or_Kumbhakasana_.txt",
    "cobra":             "Yoga-82/yoga_dataset_links/Cobra_Pose_or_Bhujangasana_.txt",
    "bridge":            "Yoga-82/yoga_dataset_links/Bridge_Pose_or_Setu_Bandha_Sarvangasana_.txt",
    "chair":             "Yoga-82/yoga_dataset_links/Chair_Pose_or_Utkatasana_.txt",
    "triangle":          "Yoga-82/yoga_dataset_links/Extended_Revolved_Triangle_Pose_or_Utthita_Trikonasana_.txt",
    "seated_forward":    "Yoga-82/yoga_dataset_links/Seated_Forward_Bend_pose_or_Paschimottanasana_.txt",
    "low_lunge":         "Yoga-82/yoga_dataset_links/Low_Lunge_pose_or_Anjaneyasana_.txt",
    "pigeon":            "Yoga-82/yoga_dataset_links/Pigeon_Pose_or_Kapotasana_.txt",
    "cat_cow":           "Yoga-82/yoga_dataset_links/Cat_Cow_Pose_or_Marjaryasana_.txt",
    "corpse":            "Yoga-82/yoga_dataset_links/Corpse_Pose_or_Savasana_.txt",
    "standing_forward":  "Yoga-82/yoga_dataset_links/Standing_Forward_Bend_pose_or_Uttanasana_.txt",
    "wheel":             "Yoga-82/yoga_dataset_links/Upward_Bow_(Wheel)_Pose_or_Urdhva_Dhanurasana_.txt",
    "boat":              "Yoga-82/yoga_dataset_links/Boat_Pose_or_Paripurna_Navasana_.txt",
    "camel":             "Yoga-82/yoga_dataset_links/Camel_Pose_or_Ustrasana_.txt",
    "half_moon":         "Yoga-82/yoga_dataset_links/Half_Moon_Pose_or_Ardha_Chandrasana_.txt",
    "eagle":             "Yoga-82/yoga_dataset_links/Eagle_Pose_or_Garudasana_.txt",
    "side_plank":        "Yoga-82/yoga_dataset_links/Side_Plank_Pose_or_Vasisthasana_.txt",
    "locust":            "Yoga-82/yoga_dataset_links/Locust_Pose_or_Salabhasana_.txt",
    "fish":              "Yoga-82/yoga_dataset_links/Fish_Pose_or_Matsyasana_.txt",
    "bow":               "Yoga-82/yoga_dataset_links/Bow_Pose_or_Dhanurasana_.txt",
    "garland":           "Yoga-82/yoga_dataset_links/Garland_Pose_or_Malasana_.txt",
    "dolphin":           "Yoga-82/yoga_dataset_links/Dolphin_Pose_or_Ardha_Pincha_Mayurasana_.txt",
    "extended_side":     "Yoga-82/yoga_dataset_links/Extended_Revolved_Side_Angle_Pose_or_Utthita_Parsvakonasana_.txt",
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