import json
import tempfile
from pdf2image import convert_from_path
import pytesseract

from Text_processing import text_process


def pdf_to_text_json(pdf_path):
    """
    Convertit un PDF en JSON temporaire avec le texte de chaque page.
    Retourne le chemin du fichier JSON temporaire.
    """
    # Convertir PDF en images
    images = convert_from_path(pdf_path)
    extracted_text = []

    for i, image in enumerate(images):
        text = pytesseract.image_to_string(image)
        extracted_text.append({
            "page": i + 1,
            "text": text
        })
    # Créer un fichier JSON temporaire
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json', mode='w', encoding='utf-8')
    json.dump(extracted_text, temp_file, ensure_ascii=False, indent=2)
    temp_file.close()

    return temp_file.name    