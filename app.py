import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from chat_logic import initialize_state, get_response, display_additional_questions

st.set_page_config(page_title="StudyMate")

# Initialize the state variables
initialize_state()

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

# Containers for the app
response_container = st.container(
    border=True,
    height=450
)
input_container = st.container()

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