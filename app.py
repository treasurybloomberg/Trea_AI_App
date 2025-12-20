import os
import streamlit as st
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
import time

# ✅ Set up DeepSeek API with OpenAI-compatible endpoint
os.environ["OPENAI_API_KEY"] = "sk-aa47d49919ad4a8795605774abad2b49"
os.environ["OPENAI_API_BASE"] = "https://api.deepseek.com/v1"

# ✅ Vector DB path
persist_directory = "./chroma_db_1219_2"

# ✅ Custom CSS for modern UI
st.set_page_config(
    page_title="HKJC Treasury Assistant", 
    page_icon="🏇", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f5f5f5;
    }
    
    /* User message styling */
    .user-message {
        background-color: #e1f5fe;
        border-radius: 10px;
        padding: 10px 15px;
        margin-bottom: 10px;
        border-left: 5px solid #039be5;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    
    /* Assistant message styling */
    .assistant-message {
        background-color: #f5f5f5;
        border-radius: 10px;
        padding: 10px 15px;
        margin-bottom: 10px;
        border-left: 5px solid #43a047;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    
    /* Sources styling */
    .sources-container {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        margin-top: 15px;
        padding: 10px;
        background-color: #fafafa;
    }
    
    /* Header styling */
    h1 {
        color: #1e3a8a;
        font-size: 2.5rem !important;
    }
    
    /* Subtle divider */
    .divider {
        height: 1px;
        background-color: #e0e0e0;
        margin: 20px 0px;
    }
    
    /* Citations */
    .citation {
        background-color: #f0f7fa;
        border-left: 3px solid #1e88e5;
        padding: 10px;
        margin: 10px 0px;
        font-size: 0.9rem;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #1e3a8a !important;
    }
    
    /* Fix for chat container to use more height */
    .chat-container {
        height: calc(100vh - 300px);
        overflow-y: auto;
        padding: 20px;
        background-color: #ffffff;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin-bottom: 20px;
    }
    
    /* Logo styles */
    .logo-container {
        display: flex;
        align-items: center;
        margin-bottom: 20px;
    }
    
    .logo-text {
        font-weight: 600;
        font-size: 1.5rem;
        margin-left: 10px;
        color: #1e3a8a;
    }
    
    /* Loading animation */
    @keyframes pulse {
        0% { opacity: 0.5; }
        50% { opacity: 1; }
        100% { opacity: 0.5; }
    }
    .loading {
        animation: pulse 1.5s infinite;
        display: inline-block;
    }
    
    /* Document source styling */
    .document-source {
        background-color: #f5f9ff;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 8px;
        font-size: 0.9rem;
        border: 1px solid #e1e7f5;
    }
    
    .source-title {
        font-weight: 600;
        color: #1e3a8a;
        font-size: 0.9rem;
        margin-bottom: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ✅ Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "sources" not in st.session_state:
    st.session_state.sources = []

def main():
    # Sidebar for information and settings
    with st.sidebar:
        st.markdown("""
        <div class="logo-container">
            <span class="logo-text">HKJC Treasury Assistant</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        ### About
        This AI assistant provides information about HKJC Treasury documents and reports. Ask questions about:
        
        - Financial statements
        - Annual reports
        - Treasury policies
        - Investment strategies
        - Risk management
        
        ### Sample Questions
        - What is HKJC's revenue in 2024?
        - How does HKJC manage risk?
        - What are the key financial highlights?
        - Who are the executive team members?
        """)
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        # Clear chat button
        if st.button("Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.sources = []
            st.experimental_rerun()

    # Main content area
    col1, col2 = st.columns([5, 1])
    with col1:
        st.markdown("""
        <h1>HKJC Treasury Knowledge Assistant</h1>
        """, unsafe_allow_html=True)
    with col2:
        st.image("https://upload.wikimedia.org/wikipedia/en/8/8f/The_Hong_Kong_Jockey_Club_logo.svg", width=80)
    
    st.markdown("""
    <p style='font-size: 1.1rem; color: #555;'>
    Ask questions about Hong Kong Jockey Club's financial reports, policies, and treasury operations.
    </p>
    <div class="divider"></div>
    """, unsafe_allow_html=True)

    try:
        # Load embeddings and vector store
        embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
        
        # LLM setup
        llm = ChatOpenAI(
            model="deepseek-chat",
            temperature=0.3,
            openai_api_key=os.environ["OPENAI_API_KEY"],
            openai_api_base=os.environ["OPENAI_API_BASE"]
        )
        
        # Chat display container with improved styling
        st.markdown('<div class="chat-container" id="chat-container">', unsafe_allow_html=True)
        
        # Display chat history with enhanced styling
        for idx, message in enumerate(st.session_state.messages):
            if message["role"] == "user":
                st.markdown(f"""
                <div class="user-message">
                    <strong>You:</strong><br>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="assistant-message">
                    <strong>Assistant:</strong><br>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
                
                # Display sources if available for this message
                if idx // 2 < len(st.session_state.sources) and st.session_state.sources[idx // 2]:
                    with st.expander("View Sources"):
                        for i, source in enumerate(st.session_state.sources[idx // 2]):
                            st.markdown(f"""
                            <div class="document-source">
                                <div class="source-title">Source {i+1}</div>
                                {source[:500] + "..." if len(source) > 500 else source}
                            </div>
                            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Query input
        user_query = st.text_input(
            "Ask a question about HKJC documents:",
            placeholder="e.g., What was HKJC's total revenue in the last fiscal year?",
            key="query_input"
        )
        
        # Advanced search options
        with st.expander("Advanced Options", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                search_depth = st.slider("Search Depth", min_value=2, max_value=10, value=3)
            with col2:
                response_type = st.selectbox(
                    "Response Style", 
                    ["Concise", "Detailed", "Technical"],
                    index=0
                )
        
        # Handle user query
        if user_query:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": user_query})
            
            # Process query with visible feedback
            with st.spinner("Searching through Treasury documents..."):
                # Retrieve documents
                docs = vectordb.similarity_search(user_query, k=search_depth)
                
                # Create context from retrieved documents
                doc_texts = [doc.page_content for doc in docs]
                context = "\n\n".join([f"Document {i+1}:\n{text}" for i, text in enumerate(doc_texts)])
                
                # Adjust prompt based on selected response type
                style_instruction = {
                    "Concise": "Keep your answer concise and to the point.",
                    "Detailed": "Provide a comprehensive and detailed answer.",
                    "Technical": "Use technical financial terminology in your response."
                }[response_type]
                
                # Create prompt
                prompt = f"""
Based on the following information from HKJC Treasury documents, please answer the question.
{style_instruction}
If the answer cannot be found in the documents, please say "I don't have sufficient information to answer that question fully."

DOCUMENTS:
{context}

QUESTION: {user_query}

ANSWER:
"""
                
                # Show typing effect
                message_placeholder = st.empty()
                
                # Create animated typing effect
                full_response = llm.predict(prompt)
                display_response = ""
                
                # Add the source documents for this response
                st.session_state.sources.append(doc_texts)
                
                # Simulate typing with chunks
                for chunk in full_response.split():
                    display_response += chunk + " "
                    message_placeholder.markdown(f"""
                    <div class="assistant-message">
                        <strong>Assistant:</strong><br>
                        {display_response}▌
                    </div>
                    """, unsafe_allow_html=True)
                    time.sleep(0.01)
                
                # Show final response
                message_placeholder.markdown(f"""
                <div class="assistant-message">
                    <strong>Assistant:</strong><br>
                    {full_response}
                </div>
                """, unsafe_allow_html=True)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
                # Show sources
                with st.expander("View Sources", expanded=False):
                    for i, doc in enumerate(docs):
                        st.markdown(f"""
                        <div class="document-source">
                            <div class="source-title">Source {i+1}</div>
                            {doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content}
                        </div>
                        """, unsafe_allow_html=True)
            
            # Clear input field - commented out as this breaks some Streamlit functionality
            # Since this is Streamlit, we would need to use st.experimental_rerun() for this behavior
                
    except Exception as e:
        st.error("⚠️ There was an error connecting to the knowledge base.")
        st.exception(e)
    
    # Footer
    st.markdown("""
    <div style="text-align: center; margin-top: 30px; padding-top: 20px; border-top: 1px solid #eee; color: #777; font-size: 0.8rem;">
        HKJC Treasury Assistant | Internal Use Only | Last updated: December 2025
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()