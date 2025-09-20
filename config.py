"""
Configuration settings for the Advanced RAG application
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# UI Configuration
PAGE_TITLE = "Geomimi - IA Geografo (Made by Julio Hsu)"
PAGE_ICON = "🌎"
LAYOUT = "wide"
SIDEBAR_STATE = "expanded"

# File Processing Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
CHROMA_COLLECTION_NAME = "rag-chroma"
CHROMA_PERSIST_DIR = "./.chroma"

# Model Configuration
LLM_TEMPERATURE = 0
TAVILY_SEARCH_RESULTS = 2

# Supported File Types
SUPPORTED_EXTENSIONS = [
    "pdf", "docx", "doc", "csv", "xlsx", "xls", 
    "txt", "md", "py", "js", "html", "xml"
]

# UI Messages
UPLOAD_PLACEHOLDER_TITLE = "📤 Envie um documento para começar"
UPLOAD_PLACEHOLDER_TEXT = "Após enviar um arquivo, você poderá fazer perguntas sobre o conteúdo."
QUESTION_PLACEHOLDER = "Qual o tema principal deste documento?"

# File Categories for UI Display
FILE_CATEGORIES = {
    "📄 Documents": ["PDF (.pdf)", "Word (.docx, .doc)", "Text (.txt, .md)"],
    "📊 Data Files": ["Excel (.xlsx, .xls)", "CSV (.csv)"],
    "💻 Code Files": ["Python (.py)", "JavaScript (.js)", "HTML (.html)", "XML (.xml)"]
}
