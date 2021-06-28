from subprocess import Popen, PIPE, run
import os
from pathlib import Path

scans_dir = Path("scans")
train_file = Path("scannetv2_train.txt")
val_file = Path("scannetv2_val.txt")
test_file = Path("scannetv2_test.txt")

file_extensions_to_export = [
    "_vh_clean_2.ply",
    "_vh_clean_2.labels.ply",
    "_vh_clean_2.0.010000.segs.json",
    ".aggregation.json"
]

def load_scenes(filepath):
    with filepath.open() as scene_file:
        scenes = scene_file.read().split("\n")
    return scenes

def is_download_complete(scene):
    for ext in file_extensions_to_export:
        filename = scene / (scene.name + ext)
        if not filename.exists():
            return False
    
    return True

train_scenes = load_scenes(train_file)
val_scenes = load_scenes(val_file)
test_scenes = load_scenes(test_file)

if not scans_dir.exists():
    scans_dir.mkdir()

scenes_to_download = train_scenes + val_scenes + test_scenes
scenes_to_download = [
    scans_dir / scene
    for scene in scenes_to_download
]

scenes_to_download = [
    scene
    for scene in current_scenes
    if not is_download_complete(scene)
]

# Download each scene
for scene in scenes_to_download:

    for ext in file_extensions_to_export:
        
        file_to_download = scene / (scene.name + ext)
        if file_to_download.exists():
            file_to_download.unlink()

        p = Popen(["timeout", "240", "python2", "download-scannet.py", "-o", "../scannet", "--id", scene.name, "--type", ext], stdin=PIPE)
        p.communicate(bytes(os.linesep.join(['c', 'n']), 'utf-8'))