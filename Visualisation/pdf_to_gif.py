"""
pdf_to_gif.py

Convert a PDF (pages full of images or any content) into an animated GIF.

Dependencies:
  pip install pymupdf pillow

Usage:
  python pdf_to_gif.py input.pdf output.gif --dpi 150 --duration 120 --loop 0
"""

import argparse
import os
from typing import List

import fitz  # PyMuPDF
from PIL import Image


def pdf_to_gif(
    pdf_path: str,
    gif_path: str,
    dpi: int = 300,
    duration_ms: int = 500,
    loop: int = 0,
    start_page: int = 1,
    end_page: int | None = None,
    optimize: bool = True,
) -> str:
    """
    Convert PDF pages to an animated GIF.

    Parameters
    ----------
    pdf_path : str
        Input PDF.
    gif_path : str
        Output GIF path.
    dpi : int
        Render resolution. Higher = sharper but bigger and slower.
    duration_ms : int
        Frame duration in milliseconds.
    loop : int
        0 = loop forever, 1 = loop once, etc.
    start_page : int
        1-based first page to include.
    end_page : int | None
        1-based last page to include (inclusive). None = last page of PDF.
    optimize : bool
        Let Pillow try to optimize the GIF size.

    Returns
    -------
    str
        Path to the written GIF.
    """
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    n_pages = doc.page_count

    if start_page < 1 or start_page > n_pages:
        raise ValueError(f"start_page must be in [1, {n_pages}]")

    if end_page is None:
        end_page = n_pages
    if end_page < start_page or end_page > n_pages:
        raise ValueError(f"end_page must be in [{start_page}, {n_pages}]")

    # Convert DPI to PyMuPDF zoom factor (PDF default is 72 DPI)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    frames: List[Image.Image] = []
    try:
        for p in range(start_page - 1, end_page):
            page = doc.load_page(p)
            pix = page.get_pixmap(matrix=mat, alpha=False)

            mode = "RGB"
            img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)

            # GIF likes palette mode; convert later as needed
            frames.append(img)

    finally:
        doc.close()

    if not frames:
        raise RuntimeError("No frames were extracted from the PDF.")

    # Convert frames to palette mode for smaller GIFs (optional, but usually helpful)
    frames_p = [frames[0].convert("P", palette=Image.Palette.ADAPTIVE)]
    for fr in frames[1:]:
        frames_p.append(fr.convert("P", palette=Image.Palette.ADAPTIVE))

    os.makedirs(os.path.dirname(os.path.abspath(gif_path)), exist_ok=True)

    frames_p[0].save(
        gif_path,
        save_all=True,
        append_images=frames_p[1:],
        duration=duration_ms,
        loop=loop,
        optimize=optimize,
        disposal=2,  # helps reduce ghosting in some cases
    )

    return gif_path



pdf_path = r"C:\Users\joell\OneDrive - Swansea University\Desktop\PhD Documents\01-Codes\Aeropt2\examples\CB Opt 09.01\postprocessed\n_0\4\x_sweep_out\eta_sweep.pdf"
gif_path = r"C:\Users\joell\OneDrive - Swansea University\Desktop\PhD Documents\01-Codes\Aeropt2\examples\CB Opt 09.01\postprocessed\n_0\4\x_sweep_out\n0_4_x_sweep_out.gif"
out = pdf_to_gif(
    pdf_path=pdf_path,
    gif_path=gif_path
)