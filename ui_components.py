"""
UI components for the Advanced RAG application
"""
import streamlit as st
from config import (
    PAGE_TITLE, PAGE_ICON, LAYOUT, SIDEBAR_STATE, 
    FILE_CATEGORIES, UPLOAD_PLACEHOLDER_TITLE, UPLOAD_PLACEHOLDER_TEXT
)
from utils import format_file_size
import os

def setup_page_config():
    """Sets up Streamlit page settings"""
    st.set_page_config(
        page_title=PAGE_TITLE, 
        page_icon=PAGE_ICON,
        layout=LAYOUT,
        initial_sidebar_state=SIDEBAR_STATE
    )


def render_header():
    """Shows the main header section"""
    st.title(f"{PAGE_ICON} Geomimi - IA Geografo")
    st.subheader("Assistente Inteligente Especializada em Geografia")
    # Destaques de funcionalidades
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("🔍 **Busca Inteligente**\n\nRecuperação avançada com busca embedding automática")
    with col2:
        st.info("📄 **Multi-Formato**\n\nSuporte a PDF, Word, Excel, arquivos de código")
    with col3:
        st.info("🤖 **IA Avançada**\n\nWorkflow LangGraph com detecção smells de alucinação")
    st.divider()


def render_sidebar(document_loader):
    """Shows the sidebar with app info and file types"""
    with st.sidebar:
        # Informações do app
        st.markdown("""
        <div style="background: white; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
            <h4>🔍 Sistema RAG Avançado</h4>
            <p>Envie documentos e faça perguntas inteligentes usando recuperação por IA.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("### 📋 Tipos de Arquivos Suportados")
        # Exibição organizada dos tipos de arquivos
        for category, formats in FILE_CATEGORIES.items():
            with st.expander(category, expanded=False):
                for fmt in formats:
                    st.markdown(f"• {fmt}")


def render_upload_section(document_loader):
    """Shows the document upload section"""
    st.markdown("## 📤 Envio de Documento")
    # Área de upload com estilo simples
    st.info("📁 **Arraste e solte seu documento**\n\nSuportado: PDF, Word, Excel, Texto, Código")
    # Mostrar extensões suportadas
    with st.expander("ℹ️ Ver Todos os Formatos Suportados", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Extensões suportadas:** {document_loader.get_supported_extensions_display()}")
        with col2:
            st.write(f"**Total de formatos:** {len(document_loader.get_supported_extensions())}")
    # File uploader
    user_file = st.file_uploader(
        "Escolha um arquivo", 
        type=document_loader.get_supported_extensions(),
        help="Envie qualquer tipo de documento suportado.",
        label_visibility="collapsed"
    )
    return user_file


def render_file_analysis(file_info):
    """Shows file analysis metrics"""
    st.markdown("### 📊 Análise do Arquivo")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("**📄 Nome do Arquivo**")
        st.write(file_info['filename'])
    with col2:
        st.markdown("**📏 Tamanho**")
        size_display = format_file_size(file_info['size'])
        st.write(size_display)
    with col3:
        st.markdown("**🏷️ Tipo**")
        st.write(f".{file_info['extension'].upper()}")
    with col4:
        st.markdown("**📋 Status**")
        status_icon = "✅" if file_info['is_supported'] else "❌"
        status_text = "Suportado" if file_info['is_supported'] else "Não suportado"
        st.write(f"{status_icon} {status_text}")


def render_upload_placeholder():
    """Shows placeholder when no file is uploaded"""
    st.markdown(f"""
    <div style="text-align: center; padding: 3rem; background: #f8fafc; border-radius: 10px; margin: 2rem 0;">
        <h3>{UPLOAD_PLACEHOLDER_TITLE}</h3>
        <p>{UPLOAD_PLACEHOLDER_TEXT}</p>
    </div>
    """, unsafe_allow_html=True)


def render_question_section(user_file):
    """Shows the question input section"""
    st.markdown("---")
    st.markdown("### 💬 Faça perguntas sobre seu documento")
    
    # Handle both uploaded files and local files
    if user_file == "local_file":
        # For local files, get the path from session state
        local_file_path = st.session_state.get('processed_file', 'local_data/geografo_proposta.pdf')
        file_display = f"📄 **Documento Atual:** {os.path.basename(local_file_path)} (arquivo local)"
    elif hasattr(user_file, 'name'):
        # For uploaded files
        file_display = f"📄 **Documento Atual:** {user_file.name}"
        if hasattr(user_file, 'type') and user_file.type:
            file_display += f" ({user_file.type})"
    else:
        # Fallback for any other case
        file_display = "📄 **Documento Atual:** Documento carregado"
    
    st.markdown(file_display)
    
    # Entrada da pergunta
    col1, col2 = st.columns([4, 1])
    with col1:
        question = st.text_input(
            'Digite sua pergunta:', 
            placeholder="como eh feito calculo de precipitacao? / quem eh o presidente do brasil?",
            disabled=not user_file,
            help="Pergunte qualquer coisa sobre o conteúdo do documento enviado"
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Espaçamento
        ask_button = st.button("Perguntar", use_container_width=True)
    return question, ask_button


def render_answer_section(result):
    """Shows the answer section"""
    st.markdown("### 📝 Resposta")
    st.success(result['solution'])
    st.markdown("---")
