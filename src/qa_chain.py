"""Conversational retrieval chain construction."""

from __future__ import annotations

from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory


def build_qa_chain(
    vector_store,
    memory: ConversationBufferMemory,
    api_key: str,
    chat_model_name: str,
    retrieval_k: int = 4,
) -> ConversationalRetrievalChain:
    """Build the Gemini-backed conversational RAG chain."""
    llm = ChatGoogleGenerativeAI(
        model=chat_model_name,
        google_api_key=api_key,
        temperature=0.2,
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": retrieval_k},
    )

    condense_question_prompt = PromptTemplate.from_template(
        (
            "Given the conversation history and a follow-up question, rewrite the "
            "question so it stands alone without changing its meaning.\n\n"
            "Chat History:\n{chat_history}\n\n"
            "Follow-up Question: {question}\n\n"
            "Standalone Question:"
        )
    )

    qa_prompt = PromptTemplate.from_template(
        (
            "You are a helpful AI assistant answering questions about the provided document.\n"
            "Use only the retrieved context below.\n"
            "If the answer is not present in the context, say you could not find it in the document.\n"
            "Keep the answer clear and concise.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        condense_question_prompt=condense_question_prompt,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        return_source_documents=True,
        output_key="answer",
        verbose=False,
    )
