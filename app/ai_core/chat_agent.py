import os
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    try:
        
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError:
        class HuggingFaceEmbeddings:
            def __init__(self, model_name="all-MiniLM-L6-v2"):
                pass
            def embed_documents(self, texts):
                return [[0.0] * 384 for _ in texts]
            def embed_query(self, text):
                return [0.0] * 384

from langchain_community.vectorstores import FAISS

from app.core.ai_provider import get_groq_api_key, get_groq_model

class ChatAgent:
    """
    RAG Agent for question answering over medical report contexts.
    Framework-agnostic Python class.
    """
    def __init__(self, api_key: str = None):
        self.api_key = api_key or get_groq_api_key()
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        self.client = Groq(api_key=self.api_key) if self.api_key else None
        self.model_name = get_groq_model()

    def initialize_vector_store(self, text_content: str):
        """Create FAISS vector store from medical report text content."""
        if not text_content or not text_content.strip():
            text_content = "No report context available."

        texts = self.text_splitter.split_text(text_content)
        if not texts:
            texts = [text_content]

        return FAISS.from_texts(texts, self.embeddings)
    
    # this is working part 
    # here api_key is taken from .env file and from backend file 

    def _format_chat_history(self, chat_history):
        """Format chat history list for Groq API format."""
        messages = []
        for msg in chat_history:
            messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
        return messages

    def _contextualize_query(self, query: str, chat_history: list):
        """Reformulate user query considering multi-turn conversation context."""
        if not chat_history or not self.client:
            return query

        recent_history = chat_history[-4:]
        history_text = "\n".join(
            [
                f"{'User' if msg.get('role') == 'user' else 'Assistant'}: {msg.get('content')}"
                for msg in recent_history
            ]
        )

        contextualize_prompt = f"""Given a chat history and the latest user question, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is.

Chat History:
{history_text}

Latest User Question: {query}

Standalone Question:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You reformulate questions to be standalone."},
                    {"role": "user", "content": contextualize_prompt},
                ],
                temperature=0.1,
                max_tokens=200,
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return query

    def get_response(self, query: str, vectorstore=None, chat_history: list = None):
        """Get RAG-contextualized answer for user query."""
        if not self.client:
            return "Error: GROQ_API_KEY is not configured on the server."

        if chat_history is None:
            chat_history = []

        contextualized_query = self._contextualize_query(query, chat_history)

        context = ""
        if vectorstore:
            try:
                retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
                docs = retriever.invoke(contextualized_query)
                context = "\n\n".join([doc.page_content for doc in docs])
                if context.strip() == "No report context available.":
                    context = ""
            except Exception:
                context = ""

        qa_system_prompt = (
            "You are an expert AI clinical assistant. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, state that clearly. "
            "Keep the answer concise and evidence-based."
        )

        messages = [{"role": "system", "content": qa_system_prompt}]

        if chat_history:
            formatted_history = self._format_chat_history(chat_history[-6:])
            messages.extend(formatted_history)

        if context and context.strip():
            user_message = f"Context:\n{context}\n\nQuestion: {query}"
        else:
            user_message = f"Question: {query}\n\nNote: No report context is available. Please answer based on clinical knowledge and chat history."
        
        messages.append({"role": "user", "content": user_message})

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.7,
                max_tokens=500,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating AI response: {str(e)}"
        
        # this is the part where there is working part is begin and the implementation phase is start

