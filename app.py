import os
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI

# ✅ Load secrets (use st.secrets in production!)
os.environ["OPENAI_API_KEY"] = "sk-aa47d49919ad4a8795605774abad2b49"
os.environ["OPENAI_API_BASE"] = "https://api.deepseek.com/v1"

# ✅ Path to local Chroma vector DB
persist_directory = "./chroma_db_combined"

# ✅ Streamlit UI
st.set_page_config(page_title="Treasury AI Assistant", layout="wide")
st.title("📄 Treasury AI Assistant")
st.caption("Ask questions about HKJC⭐.")

# ✅ Chat history state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def main():
    try:
        # ✅ Embeddings (simplest version — no device config)
        embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
        retriever = vectordb.as_retriever()

        # ✅ LLM from DeepSeek
        llm = ChatOpenAI(
            model_name="deepseek-chat",
            temperature=0.3,
            openai_api_key=os.environ["OPENAI_API_KEY"],
            openai_api_base=os.environ["OPENAI_API_BASE"]
        )

        # ✅ RAG chain
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

            st.session_state.chat_history.append((query, result["answer"]))

            st.markdown("### 🧠 Answer")
            st.success(result["answer"])

            with st.expander("📄 Source Documents"):
                for i, doc in enumerate(result["source_documents"]):
                    st.markdown(f"**Source {i+1}:**")
                    st.code(doc.page_content[:1000], language="markdown")

    except Exception as e:
        st.error("⚠️ Something went wrong.")
        st.exception(e)

if __name__ == "__main__":
    main()