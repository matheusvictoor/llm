
"""
Advanced RAG System with LangGraph

This is the main Streamlit application that demonstrates how to build a RAG system
using LangGraph for workflow management. The application handles document uploads,
processes questions, and generates answers using a LangGraph-orchestrated pipeline.

Key components:
- Document processing and chunking
- LangGraph workflow for RAG operations
- Question answering with fallback to online search
- Evaluation and quality assessment
- Real-time user interface

This implementation shows practical patterns for building RAG applications with
LangGraph, including state management, conditional routing, and error handling.
Good for understanding how LangGraph works with RAG systems.
"""
import streamlit as st

# Local imports
from config import QUESTION_PLACEHOLDER
from utils import clear_chroma_db, initialize_session_state
from ui_components import (
    setup_page_config, render_header, render_sidebar, 
    render_upload_section, render_upload_placeholder,
    render_question_section, render_answer_section,
)
from document_loader import MultiModalDocumentLoader
from document_processor import DocumentProcessor
from rag_workflow import RAGWorkflow

# Initialize components
document_loader = MultiModalDocumentLoader()
document_processor = DocumentProcessor(document_loader)
rag_workflow = RAGWorkflow()


def handle_question_processing(question):
    """Handle the Q&A processing workflow"""
    # Debug info
    print(f"Processing question: {question}")
    
    with st.container():
        with st.spinner('🧠 Analisando sua pergunta e recuperando informações relevantes...'):
            # Process the question - workflow will handle retriever automatically
            result = rag_workflow.process_question(question)
        
        # Render answer section (it will handle its own heading)
        render_answer_section(result)
        
        # Mostrar avaliações e informações do sistema
        if result:
            st.markdown("---")
            st.markdown("### 📊 Informações do Sistema")
            
            # Status do método de busca
            search_method = result.get('search_method', 'Desconhecido')
            online_search = result.get('online_search', False)
            
            if search_method == 'online' or online_search:
                st.info("🌐 Busca Online Utilizada")
            elif search_method == 'documents':
                st.success("📄 Busca em Documentos Utilizada")
            else:
                st.warning("❓ Método de busca não especificado")
            
            # Tabela resumo
            summary_data = []
            
            # Resumo das avaliações dos documentos
            if 'document_evaluations' in result and result['document_evaluations']:
                evaluations = result['document_evaluations']
                relevant_count = sum(1 for eval in evaluations if eval.score.lower() == 'yes')
                total_count = len(evaluations)
                summary_data.append(["📋 Relevância dos Documentos", f"{relevant_count}/{total_count} relevantes"])
                
                # Média de relevância se disponível
                if hasattr(evaluations[0], 'relevance_score'):
                    avg_score = sum(eval.relevance_score for eval in evaluations) / len(evaluations)
                    summary_data.append(["📊 Média de Relevância", f"{avg_score:.2f}"])
            
            # Correspondência Pergunta-Resposta
            if 'question_relevance_score' in result:
                q_relevance = result['question_relevance_score']
                if hasattr(q_relevance, 'binary_score'):
                    match_text = "✅ Bem Correspondido" if q_relevance.binary_score else "❌ Baixa Correspondência"
                    summary_data.append(["❓ Correspondência da Pergunta", match_text])
                if hasattr(q_relevance, 'relevance_score'):
                    summary_data.append(["📈 Pontuação da Pergunta", f"{q_relevance.relevance_score:.2f}"])
                if hasattr(q_relevance, 'completeness'):
                    summary_data.append(["📝 Completude", q_relevance.completeness])
            
            # Avaliação de Relevância do Documento
            if 'document_relevance_score' in result:
                doc_relevance = result['document_relevance_score']
                if hasattr(doc_relevance, 'binary_score'):
                    grounding_text = "✅ Bem Fundamentado" if doc_relevance.binary_score else "❌ Não Fundamentado"
                    summary_data.append(["🎯 Fundamentação da Resposta", grounding_text])
                if hasattr(doc_relevance, 'confidence'):
                    summary_data.append(["🔒 Confiança", f"{doc_relevance.confidence:.2f}"])
            
            # Exibir tabela resumo
            if summary_data:
                import pandas as pd
                df = pd.DataFrame(summary_data, columns=["Métrica", "Valor"])
                st.table(df)
            
            # Mostrar avaliações detalhadas em seção expansível
            with st.expander("🔧 Resultados Detalhados da Avaliação"):
                
                # Tabela de avaliações dos documentos
                if 'document_evaluations' in result and result['document_evaluations']:
                    st.markdown("**📋 Detalhes da Avaliação dos Documentos:**")
                    
                    eval_data = []
                    for i, eval in enumerate(result['document_evaluations']):
                        row = [f"Documento {i+1}", eval.score]
                        
                        if hasattr(eval, 'relevance_score'):
                            row.append(f"{eval.relevance_score:.2f}")
                        else:
                            row.append("N/A")
                        
                        if hasattr(eval, 'coverage_assessment') and eval.coverage_assessment:
                            row.append(eval.coverage_assessment[:50] + "..." if len(eval.coverage_assessment) > 50 else eval.coverage_assessment)
                        else:
                            row.append("N/A")
                        
                        if hasattr(eval, 'missing_information') and eval.missing_information:
                            row.append(eval.missing_information[:50] + "..." if len(eval.missing_information) > 50 else eval.missing_information)
                        else:
                            row.append("N/A")
                        
                        eval_data.append(row)
                    
                    if eval_data:
                        eval_df = pd.DataFrame(eval_data, columns=["Documento", "Pontuação", "Relevância", "Cobertura", "Informação Ausente"])
                        st.dataframe(eval_df, use_container_width=True)
                
                # Tabela de raciocínio
                reasoning_data = []
                if 'question_relevance_score' in result and hasattr(result['question_relevance_score'], 'reasoning'):
                    reasoning_data.append(["Relevância da Pergunta", result['question_relevance_score'].reasoning])
                
                if 'document_relevance_score' in result and hasattr(result['document_relevance_score'], 'reasoning'):
                    reasoning_data.append(["Relevância do Documento", result['document_relevance_score'].reasoning])
                
                if reasoning_data:
                    st.markdown("**🧠 Raciocínio da Avaliação:**")
                    reasoning_df = pd.DataFrame(reasoning_data, columns=["Tipo de Avaliação", "Raciocínio"])
                    st.dataframe(reasoning_df, use_container_width=True)


def handle_user_interaction(user_file):
    """Handle user interactions for Q&A"""
    # if user_file is None:
    #     render_upload_placeholder()
    #     return
    
    # Render question section
    question, ask_button = render_question_section(user_file)
    
    # Process question if submitted
    if ask_button and question.strip():
        handle_question_processing(question)
    elif ask_button and not question.strip():
        st.warning("Por favor, digite uma pergunta antes de clicar em Perguntar.")


def main():
    """Main application function"""
    # Initialize session state and clear DB only once
    initialize_session_state()
    
    # Clear ChromaDB only on first run
    if 'db_cleared' not in st.session_state:
        clear_chroma_db()
        st.session_state.db_cleared = True
        print("ChromaDB cleared on app startup")
    
    # Setup page and render UI
    setup_page_config()
    render_header()
    # render_sidebar(document_loader)
    
    # Auto-load the local PDF file instead of handling file upload
    local_pdf_path = "local_data/geografo_proposta.pdf"

    st.markdown("""
            Documento: **“Proposta de Desenvolvimento de um Geografo Inteligente - Especializado em Calculo Hidrico”**  \n
            Autor: **Felipe Oliveira - Aluno Mestrado em Geografia - UEPB** \n
            Referencia: **Metodologia adaptada de Thornthwaite & Mather (1955) e R. Mainar Medeiros (2018)**
            """
                    )
    
    # Check if we need to process the local file
    if 'local_file_processed' not in st.session_state or not st.session_state.get('local_file_processed', False):
        
        with st.spinner('🔄 Processando documento local...'):
            try:
                # Process the local PDF file directly
                retriever = document_processor.process_local_file(local_pdf_path)
                if retriever:
                    st.session_state.retriever = retriever
                    st.session_state.local_file_processed = True
                    st.session_state.processed_file = local_pdf_path
                    st.success(f"✅ Documento local carregado com sucesso: {local_pdf_path}")
                    print(f"Local file processed, retriever stored in session state")
                else:
                    st.error("❌ Falha ao processar o documento local")
                    print(f"Local file processing failed - no retriever created")
            except Exception as e:
                st.error(f"❌ Erro ao carregar documento local: {str(e)}")
                print(f"Error loading local file: {e}")
    else:
        st.success(f"✅ Documento local carregado: {local_pdf_path}")

    if st.session_state.get('local_file_processed', False):
        handle_user_interaction("local_file")
    else:
        render_upload_placeholder()

if __name__ == "__main__":
    main()
