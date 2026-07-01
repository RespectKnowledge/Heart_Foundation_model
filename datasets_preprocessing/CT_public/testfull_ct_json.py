#python3 -c "
import json
from pathlib import Path

folder = Path('/mnt/all_data/ssl_foundation/data/nnssl_raw/Dataset904_combined/imagesTr')
json_path = '/mnt/all_data/ssl_foundation/data/nnssl_raw/Dataset904_combined/pretrain_data.json'

# Load JSON
with open(json_path) as f:
    d = json.load(f)

key = list(d['datasets'].keys())[0]
subjects = d['datasets'][key]['subjects']

# Count from JSON
json_names = set()
missing_paths = []
for subj in subjects.values():
    for sess in subj['sessions'].values():
        for img in sess['images']:
            json_names.add(img['name'])
            if not Path(img['image_path']).exists():
                missing_paths.append(img['image_path'])

# Count from disk
disk_names = set(f.name for f in folder.iterdir() if f.is_file() or f.is_symlink())

# Broken symlinks
broken_symlinks = [f for f in folder.iterdir() if f.is_symlink() and not f.exists()]

# Compare
in_json_not_disk = json_names - disk_names
in_disk_not_json = disk_names - json_names

total_sessions = sum(len(s['sessions']) for s in subjects.values())
total_images = sum(len(sess['images']) for s in subjects.values() for sess in s['sessions'].values())

print(f'=== Dataset904 Combined CT ===')
print(f'Subjects:              {len(subjects)}')
print(f'Sessions:              {total_sessions}')
print(f'Images (JSON):         {total_images}')
print(f'Files on disk:         {len(disk_names)}')
print(f'Matching:              {len(json_names & disk_names)}')
print()
print(f'Missing image paths:           {len(missing_paths)}')
print(f'In JSON but not on disk:       {len(in_json_not_disk)}')
print(f'On disk but not in JSON:       {len(in_disk_not_json)}')
print(f'Broken symlinks:               {len(broken_symlinks)}')

if missing_paths:
    print('\nExample missing paths:')
    for p in missing_paths[:5]:
        print(f'  ❌ {p}')

if broken_symlinks:
    print('\nExample broken symlinks:')
    for s in broken_symlinks[:5]:
        print(f'  ❌ {s}')

if not in_json_not_disk and not in_disk_not_json and not missing_paths and not broken_symlinks:
    print()
    print('✅ Perfect match — JSON and disk are fully consistent!')
#"