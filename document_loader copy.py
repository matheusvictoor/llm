"""
Multi-Format Document Loader for Streamlit App Integration
Handles loading various document types including PDFs, Word docs, Excel files, and text files
"""

import tempfile
import os
from typing import List
from pathlib import Path
import logging

from langchain_core.documents import Document
from multimodal_loader import MultiFormatDocumentLoader as BaseMultiFormatLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StreamlitMultiFormatDocumentLoader:
    """Always loads the fixed local PDF for processing"""
    def __init__(self):
        self.base_loader = BaseMultiFormatLoader()
        self.local_pdf_path = "local_data/geografo_proposta.pdf"

    def load_document(self) -> List[Document]:
        """Load the fixed local PDF document"""
        return self.base_loader.load_document(self.local_pdf_path)

    def get_supported_extensions(self) -> List[str]:
        return self.base_loader.get_supported_extensions()

    def get_supported_extensions_display(self) -> str:
        extensions = self.get_supported_extensions()
        return ", ".join([f".{ext}" for ext in sorted(extensions)])

    def is_supported_file(self, filename: str) -> bool:
        return self.base_loader.is_supported_format(filename)

    def get_local_pdf_info(self) -> dict:
        """Returns info about the fixed local PDF"""
        if os.path.exists(self.local_pdf_path):
            size = os.path.getsize(self.local_pdf_path)
            return {
                "filename": os.path.basename(self.local_pdf_path),
                "size": size,
                "extension": "pdf",
                "is_supported": True,
                "type": "application/pdf"
            }
        else:
            return {
                "filename": os.path.basename(self.local_pdf_path),
                "size": 0,
                "extension": "pdf",
                "is_supported": False,
                "type": "application/pdf"
            }


# Convenience function for always loading the local PDF
def load_document() -> List[Document]:
    loader = StreamlitMultiFormatDocumentLoader()
    return loader.load_document()

# Create default loader instance for easy import
MultiModalDocumentLoader = StreamlitMultiFormatDocumentLoader
