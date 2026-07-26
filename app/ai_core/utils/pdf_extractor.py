import pdfplumber
from app.ai_core.utils.validators import validate_pdf_content

MAX_PDF_PAGES = 15

def extract_text_from_pdf(pdf_file_path_or_stream):
    """Extract and validate text from PDF file or stream."""
    try:
        text = ""
        with pdfplumber.open(pdf_file_path_or_stream) as pdf:
            if len(pdf.pages) > MAX_PDF_PAGES:
                return f"PDF exceeds maximum page limit of {MAX_PDF_PAGES}"
                
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        
        if not text:
            return "Could not extract text from PDF. Please ensure it is not a scanned image PDF."

        is_valid, error = validate_pdf_content(text)
        if not is_valid:
            return error
            
        return text
    except Exception as e:
        return f"Error extracting text from PDF: {str(e)}"
