import os
import shutil
import random
from pathlib import Path
import argparse

def sample_vggface2_subset(src_dir, dest_dir, num_ids=250, images_per_id=50, seed=42):
    random.seed(seed)

    src_dir = Path(src_dir)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    all_ids = [d for d in src_dir.iterdir() if d.is_dir()]
    print(f"Found {len(all_ids)} identities.")

    selected_ids = random.sample(all_ids, num_ids)

    for person_dir in selected_ids:
        person_images = list(person_dir.glob("*.jpg"))
        if len(person_images) < images_per_id:
            continue

        selected_images = random.sample(person_images, images_per_id)

        dest_person_dir = dest_dir / person_dir.name
        dest_person_dir.mkdir(parents=True, exist_ok=True)

        for img_path in selected_images:
            shutil.copy(img_path, dest_person_dir / img_path.name)

    print(f"Saved {len(selected_ids)} identities to {dest_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, required=True, help='Path to full VGGFace2 train directory')
    parser.add_argument('--dest', type=str, required=True, help='Output directory for subset')
    parser.add_argument('--num_ids', type=int, default=250, help='Number of identities to sample')
    parser.add_argument('--images_per_id', type=int, default=25, help='Images per identity to sample')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    sample_vggface2_subset(args.src, args.dest, args.num_ids, args.images_per_id, args.seed)



