import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv
from utils import get_similiar_docs, query_refiner

load_dotenv()
llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="StudyMate - Your Personalised AI Tutor")

# Sidebar contents
with st.sidebar:
    st.title("🤖 Chatbot")
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
        "Made with ❤️ by [Rookie-Parth](https://github.com/that-rookie-parth/Chatbot)"
    )

st.header("StudyMate")

# bot's response
if "generated" not in st.session_state:
    st.session_state["generated"] = ["Hi!, how can i assist you?"]

# user input
if "requests" not in st.session_state:
    st.session_state["requests"] = []

# memory for the bot
if "buffer_memory" not in st.session_state:
    st.session_state["buffer_memory"] = ConversationSummaryBufferMemory(
        llm=llm, max_token_limit=70, return_messages=True
    )

# containers for the app
input_container = st.container()
response_container = st.container()

# conversation buffer window memory
conversation = ConversationChain(
    memory=st.session_state["buffer_memory"],
    llm=llm,
    verbose=True,
)

with input_container:
    query = st.text_input("Query", key="input")
    if query:
        with st.spinner("typing...."):
            refined_query = query_refiner(
                st.session_state.buffer_memory.load_memory_variable({}), query
            )
            st.subheader("Refined Query:")
            st.write(refined_query)
            context = get_similiar_docs(refined_query)

            response = conversation.predict(
                input=f"Context:\n {context} \n\n Query:\n{refined_query}"
            )