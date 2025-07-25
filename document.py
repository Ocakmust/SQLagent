from pathlib import Path
import PyPDF2
from loggerCenter import LoggerCenter
logger = LoggerCenter().get_logger()

class DocumentProcessor:
    """Processes PDF, DOCX, and TXT documents and extracts text"""
    
    def __init__(self):
        self.text_cache = {}
        self.supported_formats = {
            '.pdf': self._extract_from_pdf,
            #'.docx': self._extract_from_docx,
            '.txt': self._extract_from_txt
        }
    
    def extract_text_from_documents(self, file_path: str) -> str:
        """Extract text from document based on file extension"""
        if file_path in self.text_cache:
            return self.text_cache[file_path]
        
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Document file not found: {file_path}")
            
            file_extension = file_path.suffix.lower()
            
            if file_extension not in self.supported_formats:
                raise ValueError(f"Unsupported file format: {file_extension}. Supported: {list(self.supported_formats.keys())}")
            
            # Extract text using appropriate method
            text = self.supported_formats[file_extension](file_path)
            
            # Cache the result
            self.text_cache[str(file_path)] = text
            logger.info(f"Text extracted from {file_path} ({file_extension})")
            return text
            
        except Exception as e:
            logger.error(f"Document processing failed for {file_path}: {e}")
            raise
    
    def _extract_from_pdf(self, file_path: Path) -> str:
        """Extract text from PDF file"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():  # Only add non-empty pages
                            text += page_text + "\n"
                    except Exception as e:
                        logger.warning(f"Failed to extract text from page {page_num} in {file_path}: {e}")
                        continue
                
                return text.strip()
        except Exception as e:
            logger.error(f"PDF extraction failed for {file_path}: {e}")
            raise
    
    # def _extract_from_docx(self, file_path: Path) -> str:
    #     """Extract text from DOCX file"""
    #     try:
    #         doc = docx.Document(file_path)
    #         paragraphs = []
            
    #         for paragraph in doc.paragraphs:
    #             if paragraph.text.strip():
    #                 paragraphs.append(paragraph.text.strip())
            
    #         return "\n".join(paragraphs)
    #     except Exception as e:
    #         logger.error(f"DOCX extraction failed for {file_path}: {e}")
    #         raise
    
    def _extract_from_txt(self, file_path: Path) -> str:
        """Extract text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read().strip()
        except Exception as e:
            logger.error(f"TXT extraction failed for {file_path}: {e}")
            raise


# def example_usage():
#     """Example usage with different document types"""
    
#     # Example 1: PDF document
#     try:
#         doc=DocumentProcessor()
#         text=doc.extract_text_from_documents("file.pdf")

#         if text:
#             logger.info(f"parsing success")
#             print(f"HERE IS THE TEXT {text}")

#     except Exception as e:
#         print(f" Error: {e}")
    
#     # Example 2: Word document
#     # try:
#     #     docx_context = ContextFind("manual.docx")
#     #     context = docx_context.return_context("How to configure the system?")
#     #     print(f"Word Context: {context}")
#     # except Exception as e:
#     #     print(f"Word Error: {e}")
    
#     # Example 3: Text document
#     try:
#         doc=DocumentProcessor()
#         text=doc.extract_text_from_documents("roadmap.txt")

#         if text:
#             logger.info(f"parsing success")
#             print(f"HERE IS THE TEXT {text}")

#     except Exception as e:
#         print(f" Error: {e}")

# if __name__ == "__main__":
#     example_usage()