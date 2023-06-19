import streamlit as st
from dotenv import load_dotenv
import os
from langchain import OpenAI, Wikipedia, PromptTemplate
from langchain.agents import initialize_agent, Tool, load_tools
from langchain.agents import AgentType
from langchain.agents.react.base import DocstoreExplorer
from langchain.agents.agent_toolkits import create_python_agent
from langchain.tools.python.tool import PythonREPLTool
from langchain.python import PythonREPL
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.utilities import TextRequestsWrapper

def getSteps(user_input):
    llm = OpenAI(openai_api_key="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", temperature=0)

    template = """
        ### System:
        Give me step by step instruction on how to do that without telling me to read the documentation. Include the necessary packages needed to be installed.

        ### User:
        Analyze through the API documentation in this link provided below.

        ### Link:
        {link}
    """

    prompt = PromptTemplate(
        input_variables=["link"],
        template = template
    )

    final_prompt = prompt.format(link=user_input)

    result = llm(final_prompt)
    st.write(result)
    return result

def generateCode(steps):
    agent_executor = create_python_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613", openai_api_key="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"),
        tool=PythonREPLTool(),
        verbose=True,
        agent="chat-zero-shot-react-description",
        agent_type=AgentType.OPENAI_FUNCTIONS,
        agent_executor_kwargs={"handle_parsing_errors": True},
    )

    template = """
        ### System:
        Response should only be in python code wrapped in a code block and respond as a string output. If you don't know the answer, respond with I don't know the answer.
        Don't write the code in a text editor
        Don't execute the code.
        Don't run any shell or command line.
        Don't install any of the packages.

        ### User:
        Given the steps provided below, produce a python code.
        
        ### Steps:
        {steps}
    """

    prompt = PromptTemplate(
        input_variables=["steps"],
        template = template
    )

    final_prompt = prompt.format(steps=steps)

    code = agent_executor.run(final_prompt)
    
    return code

def main():
    st.set_page_config(page_title = "Some Webpage", page_icon = ":tada:", layout="wide")

    st.subheader("Hello, World :wave:")
    st.title("Enter a url of an API documentation")

    user_input = st.text_input("Enter link here")
    st.write("You entered", user_input)

    steps = getSteps(user_input)
    code = generateCode(steps)
    st.write(code)

if __name__ == "__main__":
    main()