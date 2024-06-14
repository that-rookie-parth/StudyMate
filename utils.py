from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
import os

chat = ChatOpenAI(
    model="gpt-3.5-turbo-1106",
    temperature=0.2,
    api_key=os.getenv('key')
)

class Ques(BaseModel):
    q1: str = Field(description="based on the given conversation, form a question which is most likely")
    q2: str = Field(description="based on the given conversation, form a question which is less likely")

def query_refiner(conversation):
    # Define the prompt template
    prompt = PromptTemplate(
        template="based on the given conversation, generate two relevant but different queries from the given conversation. Remember that the generated query must not be presnet in the conversation. \n{format_instructions}\n{conversation}\n",
        input_variables=["query", "conversation"],
        partial_variables={"format_instructions": JsonOutputParser(pydantic_object=Ques).get_format_instructions()},
    )

    # Define the chain of operations
    chain = prompt | chat | JsonOutputParser(pydantic_object=Ques)

    # Invoke the chain with both the query and the conversation
    result = chain.invoke({"conversation": conversation})
    
    return list(result.values())