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
persist_directory = "./chroma_db_1219_2"

# ✅ Streamlit UI
st.set_page_config(page_title="Treasury AI Assistant", layout="wide")
st.title("📄 Treasury AI Assistant")
st.caption("Ask questions about HKJC⭐.")

# ✅ Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

def main():
    try:
        # ✅ Load embeddings and vector store
        embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
        
        # ✅ Use ChatOpenAI instead of OpenAI
        llm = ChatOpenAI(
            model="deepseek-chat",  # Adjust model name if needed
            temperature=0.3,
            openai_api_key=os.environ["OPENAI_API_KEY"],
            openai_api_base=os.environ["OPENAI_API_BASE"]
        )
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
        # ✅ User input via chat input
        user_query = st.chat_input("Ask a question about HKJC documents...")
        
        if user_query:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": user_query})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(user_query)
            
            # Process query and display response
            with st.chat_message("assistant"):
                with st.spinner("Searching documents..."):
                    # Simple retrieval
                    docs = vectordb.similarity_search(user_query, k=3)
                    
                    # Create context from retrieved documents
                    context = "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])
                    
                    # Create prompt
                    prompt = f"""
Based on the following information from documents, please answer the question.
If the answer cannot be found in the documents, please say "I don't have enough information to answer that question."

DOCUMENTS:
{context}

QUESTION: {user_query}

ANSWER:
"""
                    
                    # Get response
                    message_placeholder = st.empty()
                    response = llm.predict(prompt)
                    message_placeholder.markdown(response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # Show sources
                with st.expander("View Sources"):
                    for i, doc in enumerate(docs):
                        st.markdown(f"**Source {i+1}:**")
                        st.text(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                
    except Exception as e:
        st.error("⚠️ Something went wrong.")
        st.exception(e)

if __name__ == "__main__":
    main()