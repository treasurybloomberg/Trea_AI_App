import os
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# ✅ Load secrets from Streamlit Cloud
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["OPENAI_API_BASE"] = "https://api.deepseek.com/v1"

# ✅ Path to vector DB
persist_directory = "./chroma_db"

# ✅ Streamlit UI setup
st.set_page_config(page_title="PDF-Aware AI Assistant", layout="wide")
st.markdown("""
<style>
.block-container { padding-top: 2rem; }
.stTextInput > div > div > input { font-size: 1.1rem; }
.stMarkdown h1, h2, h3 { margin-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)

st.title("📄 PDF-Aware AI Assistant")
st.caption("Ask questions about your PDF knowledge base (stored in ChromaDB).")

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

        # ✅ Setup RetrievalQA chain
        rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )

        # ✅ User query
        query = st.text_input("💬 Ask something about your documents...", placeholder="e.g. What is the conclusion?")
        if query:
            with st.spinner("🤖 Thinking..."):
                result = rag_chain(query)

            # ✅ Show answer
            st.markdown("### 🧠 Answer")
            st.success(result["result"])

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