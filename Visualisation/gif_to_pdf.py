from pathlib import Path
from PIL import Image, ImageSequence
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader


def gifs_to_pdf(
    gif_folder,
    output_pdf="deformation_cases.pdf",
    frame="middle",   # "first", "middle", or an integer frame index
    page_size=A4,
    add_title=True,
):
    gif_folder = Path(gif_folder)
    gif_paths = sorted(gif_folder.glob("*.gif"))

    if not gif_paths:
        raise FileNotFoundError(f"No GIF files found in: {gif_folder}")

    c = canvas.Canvas(str(output_pdf), pagesize=page_size)
    page_w, page_h = page_size

    margin = 40
    title_space = 30 if add_title else 0
    max_w = page_w - 2 * margin
    max_h = page_h - 2 * margin - title_space

    for i, gif_path in enumerate(gif_paths, start=1):
        gif = Image.open(gif_path)
        frames = [f.copy().convert("RGB") for f in ImageSequence.Iterator(gif)]

        if frame == "first":
            img = frames[0]
        elif frame == "middle":
            img = frames[len(frames) // 2]
        elif isinstance(frame, int):
            img = frames[min(max(frame, 0), len(frames) - 1)]
        else:
            raise ValueError("frame must be 'first', 'middle', or an integer")

        img_w, img_h = img.size
        scale = min(max_w / img_w, max_h / img_h)
        draw_w = img_w * scale
        draw_h = img_h * scale

        x = (page_w - draw_w) / 2
        y = (page_h - draw_h) / 2 - title_space / 2

        if add_title:
            c.setFont("Helvetica", 12)
            c.drawCentredString(
                page_w / 2,
                page_h - margin + 10,
                f"XYZ Displacement Case {i}: {gif_path.stem}"
            )

        c.drawImage(ImageReader(img), x, y, width=draw_w, height=draw_h)
        c.showPage()

    c.save()
    print(f"Saved PDF: {output_pdf}")


if __name__ == "__main__":
    gifs_to_pdf(
        gif_folder=r"C:\Users\joell\OneDrive - Swansea University\Desktop\PhD Documents\01-Codes\Aeropt2\examples\CB Parameterization Comparison 6CNs\param1\surfaces\n_0",
        output_pdf=r"C:\Users\joell\OneDrive - Swansea University\Desktop\PhD Documents\01-Codes\Aeropt2\examples\CB Parameterization Comparison 6CNs\deformation_cases_1.pdf",
        frame="middle",
        add_title=True,
    )