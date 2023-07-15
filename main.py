import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from utils import get_similiar_docs, query_refiner
from streamlit_chat import message


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

# prompt template
template = """As an esteemed expert and experienced teacher, your dedication to fostering students' learning is unparalleled. 
Your role is to guide them with utmost care, answering their questions and preparing high-quality practice 
materials. You understand the importance of thorough and descriptive explanations, ensuring that students grasp 
concepts fully. Before formulating your response, you diligently search for relevant information in the 
similar_docs to provide accurate and well-informed answers. Leveraging the power of GPT, you enhance your 
responses by incorporating the model's expansive knowledge base and linguistic capabilities. 
Additionally, you share the best sources and formulas, empowering students with reliable references 
and tools to deepen their understanding. Your commitment to providing an exceptional learning experience 
shines through in your meticulous preparation and comprehensive guidance, fostering a genuine love for 
learning in each student you engage with."""
system_message_prompt = SystemMessagePromptTemplate.from_template(template)

example_human = HumanMessagePromptTemplate.from_template(
    "Describe the process of neutralization with the help of an example."
)
example_ai = AIMessagePromptTemplate.from_template(
    """The reaction between an acid and a base is known as neutralization reaction. In this 
    reaction, both acid and base cancel each others effect. Neutralisation reaction results 
    in the formation of salt and water. During this reaction, energy in the form of heat is 
    evolved.
    
    Acid + Base → Salt + Water + Heat
    
    For example, when sodium hydroxide (NaOH) is added to hydrochloric acid (HCl), 
    sodium chloride (NaCl) and water (H2O) are obtained.
    
    HCl+NaOH⟶NaCl+H2O"""
)

human_template = "{input}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages(
    [
        system_message_prompt,
        example_human,
        example_ai,
        human_message_prompt,
        MessagesPlaceholder(variable_name="history")
    ]
)   

# conversation buffer window memory
conversation = ConversationChain(
    memory=st.session_state["buffer_memory"],
    llm=llm,
    verbose=True,
    prompt=chat_prompt,
)

# containers for the app
response_container = st.container()
input_container = st.container()

with input_container:
    query = st.text_input("Query", key="input")
    if query:
        with st.spinner("typing...."):
            refined_query = query_refiner(
                st.session_state.buffer_memory.load_memory_variables, query
            )
            st.subheader("Refined Query:")
            st.write(refined_query)
            context = get_similiar_docs(refined_query)

            response = conversation.predict(
                input=f"Context:\n {context} \n\n Query:\n{refined_query}"
            )
        st.session_state.requests.append(query)
        st.session_state.generated.append(response)

with response_container:
    if st.session_state["generated"]:
        for i in range(len(st.session_state["generated"])):
            message(st.session_state["generated"][i], key=str(i))
            if i < len(st.session_state["requests"]):
                message(
                    st.session_state["requests"][i], is_user=True, key=str(i) + "_user"
                )