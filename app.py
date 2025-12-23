import os
import streamlit as st
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import create_qa_with_sources_chain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# ✅ Set up DeepSeek API with OpenAI-compatible endpoint
os.environ["OPENAI_API_KEY"] = "sk-aa47d49919ad4a8795605774abad2b49"
os.environ["OPENAI_API_BASE"] = "https://api.deepseek.com/v1"

# ✅ Vector DB path
persist_directory = "./chroma_db_1223_6"

# ✅ Streamlit UI with enhanced configuration
st.set_page_config(
    page_title="HKJC AI Assistant", 
    page_icon="🏇", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional UI
st.markdown("""
<style>
    /* Main layout and color scheme */
    .main {
        background-color: #f8f9fa;
    }
    
    .main .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Typography */
    h1, h2, h3 {
        color: #19216C;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    h1 {
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    /* Custom Header */
    .header-container {
        display: flex;
        align-items: center;
        padding: 1rem 0;
        border-bottom: 1px solid #e0e0e0;
        margin-bottom: 2rem;
        background-color: white;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    .header-logo {
        margin-right: 15px;
    }
    
    .header-title {
        flex-grow: 1;
    }
    
    /* Chat container */
    .chat-container {
        background-color: white;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        max-height: 60vh;
        overflow-y: auto;
    }
    
    /* Chat messages */
    .stChatMessage {
        padding: 0.8rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* User chat bubble */
    .stChatMessage [data-testid="StChatMessageContent"][data-type="user"] {
        background-color: #e3f2fd;
        border-left: 4px solid #1565c0;
    }
    
    /* Assistant chat bubble */
    .stChatMessage [data-testid="StChatMessageContent"][data-type="assistant"] {
        background-color: #f5f5f5;
        border-left: 4px solid #2e7d32;
    }
    
    /* Chat input area */
    .input-container {
        background-color: white;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    /* Sources expander */
    .sources-expander {
        background-color: #fafafa;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        margin-top: 10px;
    }
    
    .source-item {
        background-color: #f5f5f5;
        border-left: 3px solid #1565c0;
        padding: 10px;
        margin: 10px 0;
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
        border-radius: 4px;
    }
    
    /* Sidebar */
    .css-1d391kg, .css-1v3fvcr {
        background-color: #f1f3f9;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #e0e0e0;
        color: #757575;
        font-size: 0.8rem;
    }
    
    /* Buttons and interactive elements */
    button, .stButton>button, .stDownloadButton>button {
        border-radius: 4px;
        font-weight: 600;
        transition: all 0.2s;
    }
    
    button:hover, .stButton>button:hover, .stDownloadButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 10px rgba(0,0,0,0.1);
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #1565c0 !important;
    }
    
    /* Specific overrides for Streamlit components */
    [data-testid="stForm"] {
        border: none;
        padding: 0;
    }
    
    .element-container {
        margin-bottom: 0.5rem;
    }
    
    /* Category pills for sources */
    .category-pill {
        display: inline-block;
        padding: 3px 10px;
        background-color: #e8f5e9;
        color: #2e7d32;
        border-radius: 15px;
        font-size: 0.8rem;
        margin-right: 5px;
        margin-bottom: 5px;
        border: 1px solid #c8e6c9;
    }
</style>
""", unsafe_allow_html=True)

# ✅ Initialize session state for chat history (unchanged)
if "messages" not in st.session_state:
    st.session_state.messages = []

def main():
    # Sidebar with helpful information
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/en/8/8f/The_Hong_Kong_Jockey_Club_logo.svg", width=80)
        st.markdown("### HKJC AI Assistant")
        st.markdown("---")
        
        st.markdown("""
        ### About
        This AI assistant provides information about HKJC treasury documents. Get instant answers about financial reports, policies, and operations.
        
        ### Sample Questions
        - What were HKJC's key financial highlights last year?
        - How does HKJC manage financial risk?
        - What is the organizational structure of the treasury department?
        - What is HKJC's investment strategy?
        """)
        
        if st.button("Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.experimental_rerun()
            
        st.markdown("---")
        st.markdown("### Settings")
        # Adding some dummy settings that don't affect backend logic
        st.selectbox("Response Style", ["Balanced", "Concise", "Detailed"], index=0)
        st.slider("Search Depth", min_value=3, max_value=10, value=3, disabled=True, 
                 help="Number of documents to retrieve (currently fixed at 3)")

    # Custom header
    st.markdown("""
    <div class="header-container">
        <div class="header-logo">
            <span style="font-size: 2.5rem;">🏇</span>
        </div>
        <div class="header-title">
            <h1>HKJC Treasury Assistant</h1>
            <p style="color: #666; margin: 0;">Your intelligent guide to HKJC information</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    try:
        # ✅ Load embeddings and vector store (unchanged)
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",model_kwargs={"device": "cpu"}, encode_kwargs={"normalize_embeddings": True})
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
        
        # ✅ Use ChatOpenAI (unchanged)
        llm = ChatOpenAI(
            model="deepseek-chat",
            temperature=0.4,
            max_tokens=600,
            openai_api_key=os.environ["OPENAI_API_KEY"],
            openai_api_base=os.environ["OPENAI_API_BASE"]
        )
        
        # Enhanced chat container
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # Display chat history (maintaining the same core functionality)
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Improved input area
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        
        # ✅ User input via chat input (unchanged)
        user_query = st.chat_input("How can I help with HKJC information today?")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        if user_query:
            # Add user message to chat history (unchanged)
            st.session_state.messages.append({"role": "user", "content": user_query})
            
            # Display user message (unchanged)
            with st.chat_message("user"):
                st.markdown(user_query)
            
            # Process query and display response (core logic unchanged)
            with st.chat_message("assistant"):
                with st.spinner("Searching HKJC treasury documents..."):
                    # Simple retrieval (unchanged)
                    docs = docs = vectordb.max_marginal_relevance_search(user_query, k=5, fetch_k=10)
                    
                    # Create context from retrieved documents (unchanged)
                    context = "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])
                    
                    # Create prompt (unchanged)
                    prompt = f"""
Based on the following information from documents, please answer the question.
If the answer cannot be found in the documents, please say "I don't have enough information to answer that question."
When answering:
- **Use clear section headings** to organize your response (e.g., "Financial Overview", "Comparative Analysis", etc.)
- **Use bullet points or numbered lists** where appropriate for clarity
- **Include specific numbers, dates, and financial figures** mentioned in the documents (e.g., HK$ values, percentages)
- **Format all currency (e.g., HK$), numbers, and percentages properly**
- Ensure your answer is **detailed (3–4 paragraphs)** for complex questions
- Do **not** overuse markdown like `*italic*` or `**bold**` inside long sentences
- Always **complete your thoughts and sections**, avoiding cutoff responses
- Avoid combining multiple values into one phrase like **"milliontoHK"**

DOCUMENTS:
{context}

QUESTION: {user_query}

ANSWER:
"""
                    
                    # Get response (unchanged)
                    message_placeholder = st.empty()
                    response = llm.predict(prompt)
                    message_placeholder.markdown(response)
                
                # Add assistant response to chat history (unchanged)
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # Show sources with enhanced styling
                with st.expander("📚 View Source Documents"):
                    for i, doc in enumerate(docs):
                        st.markdown(f"""
                        <div class="source-item">
                            <strong>Source {i+1}</strong><br>
                            {doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content}
                        </div>
                        """, unsafe_allow_html=True)
                
    except Exception as e:
        st.error("⚠️ Unable to connect to the knowledge base. Please try again later.")
        st.exception(e)

    # Professional footer
    st.markdown("""
    <div class="footer">
        <p>HKJC Treasury AI Assistant | For internal use only | Last updated: December 2025</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()