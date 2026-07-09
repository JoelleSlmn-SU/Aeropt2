from PIL import Image
from pathlib import Path

GIF_PATH = r"C:\Users\joell\OneDrive - Swansea University\Desktop\PhD Documents\06-Conferences\03-UK Fluids Conf 2026 Modal Basis\figures\sphere_1_xyz.gif"
OUT_DIR = r"C:\Users\joell\OneDrive - Swansea University\Desktop\PhD Documents\06-Conferences\03-UK Fluids Conf 2026 Modal Basis\figures\xyz_animation_frames"
PREFIX = "frame"

Path(OUT_DIR).mkdir(exist_ok=True)

gif = Image.open(GIF_PATH)

frame_count = 0

for i in range(gif.n_frames):
    gif.seek(i)

    frame = gif.convert("RGBA")

    out_path = Path(OUT_DIR) / f"{PREFIX}_{i:04d}.png"
    frame.save(out_path)

    frame_count += 1

print(f"Saved {frame_count} frames to: {OUT_DIR}")
print()
print("Use this in Beamer:")
print(r"\usepackage{animate}")
print()
print(
    rf"\animategraphics[autoplay,loop,width=\linewidth]{{15}}{{{OUT_DIR}/{PREFIX}_}}{{0000}}{{{frame_count-1:04d}}}"
)