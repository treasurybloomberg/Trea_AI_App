import os
import streamlit as st
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import create_qa_with_sources_chain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# ✅ Set up DeepSeek API
os.environ["OPENAI_API_KEY"] = "sk-aa47d49919ad4a8795605774abad2b49"
os.environ["OPENAI_API_BASE"] = "https://api.deepseek.com/v1"

# ✅ Vector DB path
persist_directory = "./chroma_db_1224_1"

# ✅ Streamlit UI
st.set_page_config(page_title="HKJC AI Assistant", page_icon="🏇", layout="wide")

# ✅ Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# ✅ Prompt Template
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a financial analysis assistant. Based on the following document excerpts, answer the user's question.

DOCUMENTS:
{context}

QUESTION:
{question}

ANSWER:
Please provide:
- Clear section headings (e.g., "Summary", "Source", "Analysis", "Conclusion")
- Bullet points or numbered lists where appropriate
- Format financial data clearly (e.g., "HK$534 million")
- Avoid using asterisks, underscores, or markdown formatting inside numbers or currency
- Do not hallucinate — if not found in documents, say "Not enough information"
"""
)

def main():
    # Sidebar
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/en/8/8f/The_Hong_Kong_Jockey_Club_logo.svg", width=80)
        st.markdown("### HKJC AI Assistant")
        st.markdown("---")
        st.markdown("""
        This AI Assistant helps you explore HKJC treasury documents. Ask about:
        - Financial performance
        - Risk management
        - Investment strategy
        """)
        if st.button("Clear Conversation"):
            st.session_state.messages = []
            st.experimental_rerun()
        st.markdown("### Filters")
        section_filter = st.selectbox("Section", ["All", "Income Statement", "Cash Flow Statement"])

    # Header
    st.markdown("<h1>🏇 HKJC Treasury Assistant</h1>", unsafe_allow_html=True)

    try:
        # Embedding and DB
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

        # LLM
        llm = ChatOpenAI(
            model="deepseek-chat",
            temperature=0.4,
            max_tokens=600,
            openai_api_key=os.environ["OPENAI_API_KEY"],
            openai_api_base=os.environ["OPENAI_API_BASE"]
        )

        # Display chat history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Input
        user_query = st.chat_input("Ask something about HKJC treasury...")

        if user_query:
            # Save user message
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            with st.chat_message("assistant"):
                with st.spinner("Retrieving documents..."):

                    # Metadata filter logic
                    metadata_filter = {"type": {"$in": ["cashflow_row", "income_row"]}}
                    if section_filter != "All":
                        metadata_filter["section"] = section_filter

                    # Retrieve documents using metadata filter
                    docs = vectordb.max_marginal_relevance_search(
                        user_query,
                        k=6,
                        fetch_k=15,
                        filter=metadata_filter
                    )

                    # Format context
                    context = "\n\n".join(
                        [f"Document {i + 1}:\n{doc.page_content}" for i, doc in enumerate(docs)]
                    )

                    # Format prompt using template
                    prompt = prompt_template.format(context=context, question=user_query)

                    # Generate response
                    response = llm.predict(prompt)

                    # Display assistant reply
                    st.markdown(response)

                    # Save to history
                    st.session_state.messages.append({"role": "assistant", "content": response})

                # Show sources
                with st.expander("📚 View Source Documents"):
                    for i, doc in enumerate(docs):
                        st.markdown(f"""
                        <div style="background:#f8f9fa;padding:10px;border-left:4px solid #1565c0;margin-bottom:10px">
                        <strong>Source {i+1}</strong><br>
                        {doc.page_content[:500]}{'...' if len(doc.page_content) > 500 else ''}
                        </div>
                        """, unsafe_allow_html=True)

    except Exception as e:
        st.error("⚠️ Failed to retrieve or generate response.")
        st.exception(e)

    # Footer
    st.markdown("""
    <hr>
    <p style='text-align:center;color:#888;font-size:0.8rem'>
    HKJC Treasury Assistant | Internal Use Only | Updated: Dec 2025
    </p>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()