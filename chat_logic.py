import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from db import p_retriever
from utils import query_refiner
import os

# LangChain setup
llm = ChatOpenAI(
    model="gpt-3.5-turbo-0125",
    api_key=os.getenv('key')
)

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, p_retriever, contextualize_q_prompt
)

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Statefully manage chat history
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state["chat_history"]:
        st.session_state["chat_history"][session_id] = ChatMessageHistory()
    return st.session_state["chat_history"][session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

def initialize_state():
    if "generated" not in st.session_state:
        st.session_state["generated"] = ["Hi!, how can I assist you?"]
    if "requests" not in st.session_state:
        st.session_state["requests"] = []
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = {}
    if "additional_questions" not in st.session_state:
        st.session_state["additional_questions"] = []

def display_additional_questions(additional_questions):
    st.subheader("Additional Questions")
    for idx, question in enumerate(additional_questions):
        if st.button(question, key=f"additional_{idx}"):
            response = get_response(question)
            st.session_state.requests.append(question)
            st.session_state.generated.append(response)
            st.rerun()

def get_response(query):
    answer = conversational_rag_chain.invoke(
        {"input": query},
        config={"configurable": {"session_id": "studymate_0"}},
    )
    response = answer["answer"]

    # for debugging or checking the context
    # print("-----For Debugging-----")
    # print("Question: ",answer["input"])
    # print(answer["context"])

    st.session_state["chat_history"]["studymate_0"].messages = st.session_state["chat_history"]["studymate_0"].messages[-4:]
    additional_questions = query_refiner(st.session_state["chat_history"]["studymate_0"].messages[-2:])
    st.session_state.additional_questions = additional_questions
    return response