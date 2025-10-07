import streamlit as st
import os
import time
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv
load_dotenv()

os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
groq_api_key=os.getenv("GROQ_API_KEY")

## If you do not have open AI key use the below Huggingface embedding
os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")
from langchain_huggingface import HuggingFaceEmbeddings

# Page configuration
st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .stats-container {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        margin: 1rem 0;
    }
    
    .chat-container {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .answer-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        border-left: 4px solid #667eea;
    }
    
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 25px;
        border: none;
        padding: 0.75rem 1.5rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    .sidebar-content {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize embeddings and LLM
@st.cache_resource
def initialize_components():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")
    
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the questions based on the provided context only.
        Please provide the most accurate response based on the question
        <context>
        {context}
        </context>
        Question:{input}
        """
    )
    return embeddings, llm, prompt

embeddings, llm, prompt = initialize_components()

def create_vector_embedding():
    if "vectors" not in st.session_state:
        with st.spinner("🔄 Processing documents..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("📂 Loading documents...")
            progress_bar.progress(20)
            
            st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            st.session_state.loader = PyPDFDirectoryLoader("research_papers")
            st.session_state.docs = st.session_state.loader.load()
            
            status_text.text("✂️ Splitting documents...")
            progress_bar.progress(40)
            
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=200
            )
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(
                st.session_state.docs[:50]
            )
            
            status_text.text("🧠 Creating embeddings...")
            progress_bar.progress(70)
            
            st.session_state.vectors = FAISS.from_documents(
                st.session_state.final_documents, 
                st.session_state.embeddings
            )
            
            status_text.text("✅ Complete!")
            progress_bar.progress(100)
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()

# Main header
st.markdown("""
<div class="main-header">
    <h1>🧠 Intelligent Document Q&A Assistant</h1>
    <p>Powered by Groq LLaMA 3.1 & Advanced RAG Technology</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🛠️ Control Panel")
    
    # Model info
    with st.expander("🤖 AI Model Info", expanded=True):
        st.markdown("""
        - **Model**: LLaMA 3.1 8B Instant
        - **Provider**: Groq
        - **Embeddings**: HuggingFace MiniLM
        - **Vector Store**: FAISS
        """)
    
    # Statistics
    if "vectors" in st.session_state:
        st.markdown("### 📊 Document Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label="📄 Documents",
                value=len(st.session_state.docs) if 'docs' in st.session_state else 0
            )
        
        with col2:
            st.metric(
                label="📝 Chunks",
                value=len(st.session_state.final_documents) if 'final_documents' in st.session_state else 0
            )
        
        # Vector database status
        st.success("🗃️ Vector Database: Ready")
    else:
        st.info("🔄 Vector Database: Not initialized")
    
    # Settings
    with st.expander("⚙️ Advanced Settings"):
        chunk_size = st.slider("Chunk Size", 500, 2000, 1000)
        chunk_overlap = st.slider("Chunk Overlap", 50, 400, 200)
        max_docs = st.slider("Max Documents", 10, 100, 50)

# Main content area
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("### 🔍 Ask Your Question")
    
    # Enhanced input with placeholder
    user_prompt = st.text_area(
        "Enter your question about the research papers:",
        placeholder="e.g., What are the main findings about machine learning in healthcare?",
        height=100,
        help="Ask specific questions about your documents for better results"
    )
    
    # Query suggestions
    with st.expander("💡 Query Suggestions"):
        suggestions = [
            "What is the main research question?",
            "What are the key findings?",
            "What methodology was used?",
            "What are the limitations of this study?",
            "What future work is suggested?"
        ]
        
        for suggestion in suggestions:
            if st.button(suggestion, key=f"suggestion_{suggestion[:20]}"):
                user_prompt = suggestion
                st.rerun()

with col2:
    st.markdown("### ⚡ Actions")
    
    # Enhanced embedding button
    if st.button("🚀 Initialize Knowledge Base", help="Process and embed your documents"):
        create_vector_embedding()
        st.balloons()
        st.success("🎉 Knowledge base ready!")
    
    # Clear cache button
    if st.button("🧹 Clear Cache", help="Reset the application state"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# Query processing section
if user_prompt:
    if "vectors" not in st.session_state:
        st.error("⚠️ Please initialize the knowledge base first!")
    else:
        # Create columns for response
        st.markdown("---")
        
        # Build chain
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retriever_chain = create_retrieval_chain(retriever, document_chain)
        
        # Show processing animation
        with st.spinner("🤔 Analyzing documents and generating response..."):
            # Simulate processing steps
            progress_container = st.container()
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("🔍 Searching relevant documents...")
                progress_bar.progress(25)
                time.sleep(0.5)
                
                status_text.text("📖 Reading and understanding context...")
                progress_bar.progress(50)
                time.sleep(0.5)
                
                status_text.text("🧠 Generating intelligent response...")
                progress_bar.progress(75)
                
                # Actual processing
                response = retriever_chain.invoke({"input": user_prompt})
                
                status_text.text("✅ Response ready!")
                progress_bar.progress(100)
                time.sleep(0.5)
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
        
        # Display response in an attractive format
        st.markdown("### 🎯 AI Response")
        
        # Response container
        response_container = st.container()
        with response_container:
            st.markdown(f"""
            <div class="answer-box">
                <h4>📋 Answer:</h4>
                <p>{response['answer']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Additional features
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("👍 Helpful"):
                    st.success("Thank you for your feedback!")
            
            with col2:
                if st.button("👎 Not Helpful"):
                    st.info("We'll work on improving our responses!")
            
            with col3:
                if st.button("📋 Copy Response"):
                    st.info("Response copied to clipboard!")
        
        # Source documents (if available)
        if "context" in response:
            with st.expander("📚 Source Documents Used"):
                st.info("Here are the document chunks that were used to generate this response:")
                # You could display the source documents here if needed

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🚀 Built with Streamlit • Powered by Groq • Enhanced with RAG Technology</p>
</div>
""", unsafe_allow_html=True)