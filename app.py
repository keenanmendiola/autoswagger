import streamlit as st
from dotenv import load_dotenv
import os
from langchain import OpenAI, Wikipedia
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.agents.react.base import DocstoreExplorer
from langchain.agents.agent_toolkits import create_python_agent
from langchain.tools.python.tool import PythonREPLTool
from langchain.python import PythonREPL
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI

def getSteps(user_input):
    llm = OpenAI(openai_api_key=os.environ["OPEN_API_KEY"])
    text = f"Using python, give me steps and/or on how to integrate the API from this link: {user_input}"
    result = llm(text)
    st.write(result)
    return result

def generateCode(steps):
    agent_executor = create_python_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613", openai_api_key=os.environ["OPEN_API_KEY"]),
        tool=PythonREPLTool(),
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        agent_executor_kwargs={"handle_parsing_errors": True},
    )
    
    prompt = f"Given the instructions below, generate code to integrate it with a python written backend service: {steps}"
    code = agent_executor.run(prompt)
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