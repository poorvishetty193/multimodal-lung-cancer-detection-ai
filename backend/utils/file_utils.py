import shutil
from pathlib import Path

def save_upload(file, save_path: str):
    dest = Path(save_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return str(dest)
