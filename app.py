# Streamlit related lib
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space

# For warnings
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# LangChain & OpenAI
from langchain_openai import ChatOpenAI
import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from db import retriever
from utils import query_refiner

st.set_page_config(page_title="StudyMate")
llm = ChatOpenAI(
    model="gpt-3.5-turbo-0125",
    api_key=os.getenv('key')
)

# Sidebar contents
with st.sidebar:
    st.title("🤓 StudyMate")
    st.markdown(
        """
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [Pinecone](https://www.pinecone.io/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
 
    """
    )
    add_vertical_space(2)
    st.write(
        "Made with ❤️ by [Rookie-Parth](https://github.com/that-rookie-parth/StudyMate)"
    )

st.header("🤓 StudyMate")
st.caption("Personalized study help, at your fingertips")

# Bot's response
if "generated" not in st.session_state:
    st.session_state["generated"] = ["Hi!, how can I assist you?"]

# User input
if "requests" not in st.session_state:
    st.session_state["requests"] = []

# Store chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = {}

# Store additional questions
if "additional_questions" not in st.session_state:
    st.session_state["additional_questions"] = []

# LangChain setup
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
    llm, retriever, contextualize_q_prompt
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
store = st.session_state["chat_history"]
session_id = "studymate_0"

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# Containers for the app
response_container = st.container(
    border=True,
    height=450
)
input_container = st.container()

# Function to display additional questions and handle user clicks
def display_additional_questions(additional_questions):
    st.subheader("Additional Questions")
    for idx, question in enumerate(additional_questions):
        if st.button(question, key=f"additional_{idx}"):
            response = get_response(question)
            st.session_state.requests.append(question)
            st.session_state.generated.append(response)
            st.rerun()

# Function to get the response from the conversational chain
def get_response(query):
    response = conversational_rag_chain.invoke(
        {"input": query},
        config={"configurable": {"session_id": session_id}},
    )["answer"]
    store[session_id].messages = store[session_id].messages[-4:]
    additional_questions = query_refiner(store[session_id].messages[-2:])
    st.session_state.additional_questions = additional_questions
    return response

with input_container:
    query = st.chat_input("Got a question? Fire away!")
    if query:
        with st.spinner("Typing...."):
            response = get_response(query)
        st.session_state.requests.append(query)
        st.session_state.generated.append(response)

with response_container:
    if st.session_state["generated"]:
        for i in range(len(st.session_state["generated"])):
            response_container.chat_message("assistant").write(st.session_state["generated"][i])
            if i < len(st.session_state["requests"]):
                response_container.chat_message("user").write(st.session_state["requests"][i])

        # Display additional questions
        if st.session_state["additional_questions"]:
            display_additional_questions(st.session_state["additional_questions"])