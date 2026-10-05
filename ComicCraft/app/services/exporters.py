from pathlib import Path
from datetime import datetime
import re

from fpdf import FPDF


# ---------------------------------------------------------
# Project directories
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

EXPORT_DIR = BASE_DIR / "static" / "exports"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------

def clean_text(text) -> str:
    """
    Convert generated AI text into safe PDF text.

    This prevents problematic Unicode characters and very long
    unbroken strings from causing FPDF rendering errors.
    """

    if text is None:
        return ""

    text = str(text)

    # Replace common typographic characters with PDF-safe equivalents.
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2026": "...",
        "\u00a0": " ",
        "\u2022": "-",
        "\u2192": "->",
        "\u2190": "<-",
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Remove unsupported control characters except newline/tab.
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Normalize excessive spaces.
    text = re.sub(r"[ \t]+", " ", text)

    # Keep newlines clean.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def safe_filename(value: str) -> str:
    """
    Create a safe filename for Windows/Linux/macOS.
    """

    value = clean_text(value)

    if not value:
        value = "comic"

    value = re.sub(r"[^A-Za-z0-9_-]+", "_", value)

    return value[:80].strip("_") or "comic"


# ---------------------------------------------------------
# PDF class
# ---------------------------------------------------------

class ComicPDF(FPDF):
    """
    Custom PDF class used by ComicCraft.
    """

    def header(self):
        self.set_font("Helvetica", "B", 16)

        self.cell(
            0,
            10,
            "ComicCraft",
            align="C",
        )

        self.ln(8)

    def footer(self):
        self.set_y(-15)

        self.set_font("Helvetica", "", 8)

        self.cell(
            0,
            10,
            f"ComicCraft - Page {self.page_no()}",
            align="C",
        )


# ---------------------------------------------------------
# Safe text rendering
# ---------------------------------------------------------

def write_text(pdf: FPDF, text: str, font_size: int = 11):
    """
    Safely write generated text to the PDF.

    wrapmode='CHAR' is important because AI-generated text can
    sometimes contain a very long unbroken token/string.

    Without character-level wrapping FPDF2 can raise:

        Not enough horizontal space to render a single character
    """

    text = clean_text(text)

    if not text:
        return

    pdf.set_font("Helvetica", "", font_size)

    pdf.multi_cell(
        0,
        7,
        text,
        wrapmode="CHAR",
    )


# ---------------------------------------------------------
# Panel title
# ---------------------------------------------------------

def write_panel_title(pdf: FPDF, panel_number, title):
    """
    Render the panel heading.
    """

    title = clean_text(title)

    if title:
        heading = f"Panel {panel_number}: {title}"
    else:
        heading = f"Panel {panel_number}"

    pdf.set_font("Helvetica", "B", 14)

    pdf.multi_cell(
        0,
        8,
        heading,
        wrapmode="CHAR",
    )

    pdf.ln(3)


# ---------------------------------------------------------
# Panel image
# ---------------------------------------------------------

def add_panel_image(pdf: FPDF, image_path):
    """
    Add a panel image while keeping it inside the printable
    page area.
    """

    if not image_path:
        return

    image_path = Path(str(image_path))

    # Handle paths such as /static/panels/image.png
    if not image_path.is_absolute():
        image_path = BASE_DIR / image_path

    if not image_path.exists():
        return

    try:
        # Printable page dimensions.
        page_width = pdf.w
        page_height = pdf.h

        left_margin = pdf.l_margin
        right_margin = pdf.r_margin

        usable_width = page_width - left_margin - right_margin

        # Reserve space for footer and text.
        max_height = page_height * 0.55

        # Get image dimensions.
        from PIL import Image

        with Image.open(image_path) as img:
            width, height = img.size

        if width <= 0 or height <= 0:
            return

        # Maintain aspect ratio.
        image_width = usable_width
        image_height = image_width * height / width

        # Prevent very tall images.
        if image_height > max_height:
            image_height = max_height
            image_width = image_height * width / height

        # Center image.
        x = (page_width - image_width) / 2

        pdf.image(
            str(image_path),
            x=x,
            y=None,
            w=image_width,
            h=image_height,
        )

        pdf.ln(5)

    except Exception:
        # Do not allow one broken image to crash PDF generation.
        pdf.set_font("Helvetica", "I", 9)

        pdf.multi_cell(
            0,
            6,
            "[Panel illustration could not be embedded.]",
            wrapmode="CHAR",
        )

        pdf.ln(3)


# ---------------------------------------------------------
# Main PDF exporter
# ---------------------------------------------------------

def save_pdf(layout, title="ComicCraft"):
    """
    Generate the final ComicCraft PDF.

    Parameters
    ----------
    layout:
        List of panel dictionaries generated by layout_builder.py.

        Expected structure is approximately:

        [
            {
                "panel_number": 1,
                "title": "...",
                "image_path": "...",
                "text": "..."
            },
            ...
        ]

    title:
        Optional comic title.

    Returns
    -------
    str
        Relative path to the generated PDF.
    """

    if layout is None:
        layout = []

    if not isinstance(layout, list):
        layout = list(layout)

    # Create a unique filename.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = f"{safe_filename(title)}_{timestamp}.pdf"

    output_path = EXPORT_DIR / filename

    # -----------------------------------------------------
    # Create PDF
    # -----------------------------------------------------

    pdf = ComicPDF(
        orientation="P",
        unit="mm",
        format="A4",
    )

    pdf.set_auto_page_break(
        auto=True,
        margin=18,
    )

    pdf.set_margins(
        left=15,
        top=15,
        right=15,
    )

    # -----------------------------------------------------
    # Cover / title page
    # -----------------------------------------------------

    pdf.add_page()

    pdf.ln(25)

    pdf.set_font("Helvetica", "B", 24)

    pdf.multi_cell(
        0,
        12,
        clean_text(title),
        align="C",
        wrapmode="CHAR",
    )

    pdf.ln(10)

    pdf.set_font("Helvetica", "", 11)

    pdf.multi_cell(
        0,
        7,
        "AI-generated comic created with ComicCraft.",
        align="C",
        wrapmode="CHAR",
    )

    # -----------------------------------------------------
    # Panel pages
    # -----------------------------------------------------

    for index, panel in enumerate(layout, start=1):

        if not isinstance(panel, dict):
            continue

        pdf.add_page()

        # Support multiple possible panel-number keys.
        panel_number = (
            panel.get("panel_number")
            or panel.get("panel")
            or panel.get("number")
            or index
        )

        # Support multiple possible title keys.
        panel_title = (
            panel.get("title")
            or panel.get("panel_title")
            or f"Scene {panel_number}"
        )

        # Support multiple possible image keys.
        image_path = (
            panel.get("image_path")
            or panel.get("image")
            or panel.get("image_url")
            or panel.get("path")
        )

        # Support multiple possible text keys.
        panel_text = (
            panel.get("text")
            or panel.get("story")
            or panel.get("narration")
            or panel.get("description")
            or ""
        )

        # -------------------------------------------------
        # Panel heading
        # -------------------------------------------------

        write_panel_title(
            pdf,
            panel_number,
            panel_title,
        )

        # -------------------------------------------------
        # Panel image
        # -------------------------------------------------

        add_panel_image(
            pdf,
            image_path,
        )

        # -------------------------------------------------
        # Panel story
        # -------------------------------------------------

        if panel_text:

            pdf.set_font(
                "Helvetica",
                "B",
                11,
            )

            pdf.multi_cell(
                0,
                7,
                "Story",
                wrapmode="CHAR",
            )

            pdf.ln(1)

            write_text(
                pdf,
                panel_text,
                font_size=10,
            )

    # -----------------------------------------------------
    # Write file
    # -----------------------------------------------------

    pdf.output(str(output_path))

    # Return path relative to project root.
    relative_path = output_path.relative_to(BASE_DIR)

    return str(relative_path).replace("\\", "/")


# ---------------------------------------------------------
# Backward-compatible alias
# ---------------------------------------------------------

def export_pdf(layout, title="ComicCraft"):
    """
    Alias for save_pdf().

    This is useful if another module imports export_pdf()
    instead of save_pdf().
    """

    return save_pdf(
        layout,
        title=title,
    )
