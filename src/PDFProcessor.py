import os
import glob
import fitz
import io
from PIL import Image
import pytesseract

from typing import List, Tuple


class PDFProcessor:
    """
    Process PDF files, extracting text and using OCR to analyze images within the PDFs.
    """

    @staticmethod
    def get_pdf_paths(folder_path: str):
        """
        Fetches all PDF file paths from the folder.

        Parameters:
        - path: Path to the folder that contains the PDF files.

        Returns:
        - pdf_paths: List of PDF file paths.
        """
        pdf_paths = glob.glob(os.path.join(folder_path, "*.pdf"))
        print(f"Found {len(pdf_paths)} PDF files")
        return pdf_paths

    def extract_pdf_text(path: str) -> Tuple[List[str], List[int]]:
        """
        Extract text from the input PDF file, with images analyzed using OCR.

        Parameters:
        - path: Path to the PDF file.

        Returns:
        - texts: List of texts extracted from each page
        - page_numbers: List of page numbers
        """
        texts = []
        page_numbers = []
        with fitz.open(path) as doc:
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = ""  # initialize for this page

                # Extract text
                text = page.get_text()
                if text.strip():
                    page_text = text

                # Extract images and perform OCR
                for img_tuple in page.get_images(full=True):
                    xref = img_tuple[0]
                    image_bytes = doc.extract_image(xref)["image"]
                    image = Image.open(io.BytesIO(image_bytes))
                    ocr_text = pytesseract.image_to_string(image)

                    if ocr_text.strip():
                        page_text += "\n" + ocr_text

                page_numbers.append(page_num + 1)
                texts.append(page_text)

        return texts, page_numbers

    def extract_text_from_pdfs_in_folder(folder_path) -> Tuple[List[str], List[str]]:
        """
        Processes all PDFs in the folder and extracts text and OCR content.

        Parameters:
        - path: Path to the folder that contains the PDF files.

        Returns:
        - texts: List of texts extracted from each page
        - info: List of info, i.e., PDF filenames and page numbers 
        where the text has been extracted.
        """
        pdf_paths = PDFProcessor.get_pdf_paths(folder_path)

        texts = []
        info = []
        for path in pdf_paths:
            print(f"Processing {path}")
            texts_cur, page_numbers_cur = PDFProcessor.extract_pdf_text(path)

            texts += texts_cur
            fname_cur = os.path.basename(path)
            info += [f'{fname_cur} {page_num}' 
                    for page_num in page_numbers_cur]

        return texts, info
