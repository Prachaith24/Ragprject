import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from retriever import ensemble_retriever
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="PDF Q&A with RAG",
    page_icon="📚",
    layout="wide"
)

# Title and description
st.title("📚 PDF Question Answering System")
st.markdown("Ask questions about your document using AI-powered retrieval")

# Get API key from environment
groq_api_key = os.getenv("GROQ_API_KEY")

# Sidebar for settings
with st.sidebar:
    st.header("⚙ Configuration")
    
    # Show API key status
    if groq_api_key:
        st.success("✅ Groq API Key loaded from .env")
    else:
        st.error("❌ Groq API Key not found in .env file")
        st.info("Add GROQ_API_KEY=your_key_here to your .env file")
    
    # Model selection
    model_choice = st.selectbox(
        "Select Model:",
        [
            "llama-3.3-70b-versatile",
            "llama-3.1-70b-versatile",
            "mixtral-8x7b-32768",
            "gemma2-9b-it"
        ]
    )
    
    # Temperature slider
    temperature = st.slider(
        "Temperature:",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        help="Higher values make output more random"
    )
    
    # Number of sources
    k_sources = st.slider(
        "Number of sources to retrieve:",
        min_value=1,
        max_value=10,
        value=5,
        help="How many document chunks to use for answering"
    )

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("📄 View Sources"):
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(f"*Source {i}:*")
                    st.text(source.page_content[:300] + "...")
                    st.markdown(f"Page: {source.metadata.get('page', 'N/A')}")
                    st.divider()

# Chat input
if prompt := st.chat_input("Ask a question about your document..."):
    # Check if API key is available
    if not groq_api_key:
        st.error("⚠ Please add GROQ_API_KEY to your .env file!")
        st.stop()
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            try:
                # Initialize Groq LLM
                llm = ChatGroq(
                    model=model_choice,
                    temperature=temperature,
                    groq_api_key=groq_api_key
                )
                
                # Create prompt template
                template = """Answer the question based only on the following context:
{context}

Question: {question}

Answer:"""
                
                prompt_template = ChatPromptTemplate.from_template(template)
                
                # Helper function to format documents
                def format_docs(docs):
                    return "\n\n".join([doc.page_content for doc in docs])
                
                # --- STEP 1: Retrieve Documents First ---
                # We do this outside the chain to avoid the "|" error and 
                # to prevent running the search twice (once for answer, once for display).
                # Note: If .invoke() fails, try .get_relevant_documents() depending on your langchain version
                # Force the use of invoke. If this fails, the installation is definitely broken.
                retrieved_docs = ensemble_retriever.invoke(prompt)
                
                # --- STEP 2: Format Context ---
                formatted_context = format_docs(retrieved_docs)
                
                # --- STEP 3: Define Chain (Simpler now) ---
                # We no longer need the retriever in the chain since we have the context
                chain = (
                    prompt_template
                    | llm
                    | StrOutputParser()
                )
                
                # --- STEP 4: Generate Answer ---
                answer = chain.invoke({
                    "context": formatted_context,
                    "question": prompt
                })
                
                # Display answer
                st.markdown(answer)
                
                # Display sources in expander
                with st.expander("📄 View Sources"):
                    for i, doc in enumerate(retrieved_docs, 1):
                        st.markdown(f"*Source {i}:*")
                        st.text(doc.page_content[:300] + "...")
                        st.markdown(f"Page: {doc.metadata.get('page', 'N/A')}")
                        st.divider()
                
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": retrieved_docs
                })
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                # Optional: print full traceback to terminal for debugging
                # import traceback
                # print(traceback.format_exc())

# Clear chat button in sidebar
with st.sidebar:
    st.divider()
    if st.button("🗑 Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Footer
st.sidebar.divider()
st.sidebar.markdown("---")
st.sidebar.caption("Built with LangChain, Groq, and Streamlit")