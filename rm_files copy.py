from pathlib import Path

dir_path = Path("/home/wph52/weird/dynamics/rl/runs/Robosuite-Door-Panda_20251016_190434/videos/train")
videos = sorted(dir_path.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
for p in videos[:-10]:
    p.unlink()