import os
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI

# ✅ Load secrets from Streamlit Cloud or hardcoded for local dev
os.environ["OPENAI_API_KEY"] = "sk-aa47d49919ad4a8795605774abad2b49"
os.environ["OPENAI_API_BASE"] = "https://api.deepseek.com/v1"

# ✅ Path to vector DB
persist_directory = "./chroma_db_combined"

# ✅ Streamlit UI setup
st.set_page_config(page_title="Treasury AI Assistant", layout="wide")
st.markdown("""
<style>
.block-container { padding-top: 2rem; }
.stTextInput > div > div > input { font-size: 1.1rem; }
.stMarkdown h1, h2, h3 { margin-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)

st.title("📄 Treasury AI Assistant")
st.caption("Ask questions about HKJC⭐.")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def main():
    try:
        # ✅ Load vector DB and retriever
        embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
        retriever = vectordb.as_retriever()

        # ✅ Load LLM from DeepSeek
        llm = ChatOpenAI(
            model_name="deepseek-chat",
            temperature=0.3,
            openai_api_key=os.environ["OPENAI_API_KEY"],
            openai_api_base=os.environ["OPENAI_API_BASE"]
        )

        # ✅ Setup ConversationalRetrievalChain
        rag_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )

        # ✅ User input
        query = st.text_input("💬 Ask something about your documents...", placeholder="e.g. What is the conclusion?")
        if query:
            with st.spinner("🤖 Thinking..."):
                result = rag_chain({
                    "question": query,
                    "chat_history": st.session_state.chat_history
                })

            # ✅ Update chat history
            st.session_state.chat_history.append((query, result["answer"]))

            # ✅ Show answer
            st.markdown("### 🧠 Answer")
            st.success(result["answer"])

            # ✅ Show sources
            with st.expander("📄 Source Documents"):
                for i, doc in enumerate(result["source_documents"]):
                    st.markdown(f"**Source {i+1}:**")
                    st.code(doc.page_content[:1000], language="markdown")

    except Exception as e:
        st.error("⚠️ Something went wrong.")
        st.exception(e)

if __name__ == "__main__":
    main()