import json
import tempfile
import os
import shutil
import glob
from pdf2image import convert_from_path
import pytesseract


def _detect_winget_poppler_bin():
    """
    Return the Poppler bin directory installed via winget, if available.
    """
    base = os.path.expandvars(
        r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\oschwartz10612.Poppler_Microsoft.Winget.Source_8wekyb3d8bbwe"
    )
    candidates = glob.glob(os.path.join(base, "poppler-*", "Library", "bin"))
    if not candidates:
        return None

    candidates.sort(reverse=True)
    return candidates[0]


def _detect_tesseract_exe():
    """
    Return the Tesseract executable path, if available.
    """
    env_value = os.environ.get("TESSERACT_CMD")
    if env_value and os.path.exists(env_value):
        return env_value

    in_path = shutil.which("tesseract")
    if in_path:
        return in_path

    common_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(common_path):
        return common_path

    return None


def pdf_to_text_json(pdf_path):
    """
    Convert a PDF to a temporary JSON file containing page text.
    Returns the path to the temporary JSON file.
    """
    poppler_path = os.environ.get("POPPLER_PATH")
    tesseract_cmd = _detect_tesseract_exe()

    if not poppler_path:
        poppler_path = _detect_winget_poppler_bin()

    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    else:
        raise RuntimeError(
            "Tesseract not found. Install Tesseract OCR and add it to PATH "
            "or set the TESSERACT_CMD environment variable."
        )

    if not poppler_path and not shutil.which("pdftoppm"):
        raise RuntimeError(
            "Poppler not found (pdftoppm). Install Poppler and add it to PATH "
            "or set the POPPLER_PATH environment variable."
        )

    images = convert_from_path(pdf_path, poppler_path=poppler_path)
    extracted_text = []

    for i, image in enumerate(images):
        text = pytesseract.image_to_string(image)
        extracted_text.append({
            "page": i + 1,
            "text": text,
        })

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w", encoding="utf-8")
    json.dump(extracted_text, temp_file, ensure_ascii=False, indent=2)
    temp_file.close()

    return temp_file.name
