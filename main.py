import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

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
    st.write("Made with ❤️ by [Rookie-Parth](https://github.com/that-rookie-parth/Chatbot)")

st.header("StudyMate")

# bot's response
if "generated" not in st.session_state:
    st.session_state['generated'] = ["Hi!, how can i assist you?"]

# user input
if "requests" not in st.session_state:
    st.session_state["requests"] = []

input_container = st.container()
response_container = st.container()

if "buffer_memory" not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(
        k=3, return_messages=True
    )