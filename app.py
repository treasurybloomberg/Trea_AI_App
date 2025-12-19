import os
import streamlit as st
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

# ✅ Read OpenAI config from Streamlit secrets (recommended)
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["OPENAI_API_BASE"] = st.secrets["OPENAI_API_BASE"]

# ✅ Path to Chroma vector DB
persist_directory = "./chroma_db_combined"

# ✅ Streamlit UI layout
st.set_page_config(page_title="Treasury AI Assistant", layout="wide")
st.title("📄 Treasury AI Assistant")
st.caption("Ask questions about HKJC⭐.")

# ✅ Chat history state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def main():
    try:
        # ✅ Load embeddings (will run on CPU by default)
        embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # ✅ Load persisted Chroma DB
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
        retriever = vectordb.as_retriever()

        # ✅ LLM config (ChatOpenAI from langchain==0.0.319)
        llm = ChatOpenAI(
            temperature=0.3,
            model_name="gpt-3.5-turbo",  # or "deepseek-chat" if supported
            openai_api_key=os.environ["OPENAI_API_KEY"],
            openai_api_base=os.environ["OPENAI_API_BASE"]
        )

        # ✅ Build RAG chain
        rag_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )

        # ✅ User input
        query = st.text_input("💬 Ask a question about your documents...", placeholder="e.g. What is the conclusion?")
        if query:
            with st.spinner("🤖 Thinking..."):
                result = rag_chain({
                    "question": query,
                    "chat_history": st.session_state.chat_history
                })

            # ✅ Store conversation
            st.session_state.chat_history.append((query, result["answer"]))

            # ✅ Show answer
            st.markdown("### 🧠 Answer")
            st.success(result["answer"])

            # ✅ Show source documents
            with st.expander("📄 Source Documents"):
                for i, doc in enumerate(result["source_documents"]):
                    st.markdown(f"**Source {i+1}:**")
                    st.code(doc.page_content[:1000], language="markdown")

    except Exception as e:
        st.error("⚠️ Something went wrong.")
        st.exception(e)

if __name__ == "__main__":
    main()