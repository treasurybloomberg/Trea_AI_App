import os
import streamlit as st
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

# ✅ Set up DeepSeek API
os.environ["OPENAI_API_KEY"] = "sk-aa47d49919ad4a8795605774abad2b49"
os.environ["OPENAI_API_BASE"] = "https://api.deepseek.com/v1"

# ✅ Vector DB path
persist_directory = "./chroma_db_1224_1"

# ✅ Streamlit config
st.set_page_config(page_title="HKJC AI Assistant", page_icon="🏇", layout="wide")

# ✅ Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# ✅ Streaming-compatible response container
if "stream_placeholder" not in st.session_state:
    st.session_state.stream_placeholder = None

# ✅ Prompt template that avoids “Document 1 says…”
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a financial analysis assistant. Based on the following excerpts from internal financial documents, answer the user's question as if you're an expert analyst.

DOCUMENT EXCERPTS:
{context}

QUESTION:
{question}

ANSWER:
- Respond naturally and concisely, like a finance analyst.
- DO NOT mention document numbers like "Document 1 says..."
- Focus on the insights, not the source.
- Use bullet points and clear headings if needed.
- Format numbers cleanly (e.g., "HK$534 million")
- Avoid markdown artifacts like asterisks or underscores in numbers
- If information is missing, say "Not enough information available"
"""
)

def main():
    # Sidebar
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/en/8/8f/The_Hong_Kong_Jockey_Club_logo.svg", width=80)
        st.markdown("### HKJC AI Assistant")
        st.markdown("---")
        st.markdown("""
        Ask about:
        - Financial performance
        - Risk management
        - Cash flow details
        """)
        if st.button("Clear Conversation"):
            st.session_state.messages = []
            st.experimental_rerun()
        section_filter = st.selectbox("Filter by Section", ["All", "Income Statement", "Cash Flow Statement"])

    st.markdown("<h1>🏇 HKJC Treasury Assistant</h1>", unsafe_allow_html=True)

    try:
        # Load vector DB
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

        # Configure LLM with streaming
        llm = ChatOpenAI(
            model="deepseek-chat",
            temperature=0.4,
            max_tokens=600,
            streaming=True,
            openai_api_key=os.environ["OPENAI_API_KEY"],
            openai_api_base=os.environ["OPENAI_API_BASE"]
        )

        # Show chat history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat input
        user_query = st.chat_input("Ask something about HKJC treasury...")

        if user_query:
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            with st.chat_message("assistant"):
                with st.spinner("Retrieving documents..."):

                    # Filter logic
                    metadata_filter = {"type": {"$in": ["cashflow_row", "income_row"]}}
                    if section_filter != "All":
                        metadata_filter["section"] = section_filter

                    # Retrieve docs
                    docs = vectordb.max_marginal_relevance_search(
                        user_query,
                        k=6,
                        fetch_k=15,
                        filter=metadata_filter
                    )

                    # Combine context
                    context = "\n\n".join([doc.page_content for doc in docs])

                    # Format prompt
                    prompt = prompt_template.format(context=context, question=user_query)

                # Streaming response display
                full_response = ""
                st.session_state.stream_placeholder = st.empty()

                def stream_handler(chunk):
                    nonlocal full_response
                    full_response += chunk
                    st.session_state.stream_placeholder.markdown(full_response)

                # Run LLM with streaming
                llm.stream(prompt, stream_handler)

                # Save assistant message
                st.session_state.messages.append({"role": "assistant", "content": full_response})

                # Show source expandable
                with st.expander("📚 View Source Excerpts"):
                    for i, doc in enumerate(docs):
                        st.markdown(f"""
                        <div style="background:#f8f9fa;padding:10px;border-left:4px solid #1565c0;margin-bottom:10px">
                        <strong>Excerpt {i+1}</strong><br>
                        {doc.page_content[:500]}{'...' if len(doc.page_content) > 500 else ''}
                        </div>
                        """, unsafe_allow_html=True)

    except Exception as e:
        st.error("⚠️ Something went wrong.")
        st.exception(e)

    st.markdown("""
    <hr>
    <p style='text-align:center;color:#888;font-size:0.8rem'>
    HKJC Treasury Assistant · For internal use · Updated Dec 2025
    </p>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()