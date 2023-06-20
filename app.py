import streamlit as st
from dotenv import load_dotenv
import os
from langchain import OpenAI, PromptTemplate
# from langchain.agents import AgentType
# from langchain.agents.agent_toolkits import create_python_agent
# from langchain.tools.python.tool import PythonREPLTool
# from langchain.python import PythonREPL
# from langchain.agents.agent_types import AgentType
# from langchain.chat_models import ChatOpenAI
import openai

load_dotenv()

openai.api_key=os.getenv('OPEN_API_KEY')

def getSteps(context, link):
    llm = OpenAI(openai_api_key=os.getenv('OPEN_API_KEY'), temperature=0)

    template = """
        ### System:
        I want you to act as a programmer that assists other programmers in development.
        Use the context provided below to analyze the website and give me instructions on how to integrate the API from the website.
        Give me step by step pseudocode on how to integrate the API without telling me to read the documentation based on the link provided below. Include the necessary packages needed to be installed.
        Always respond in bullet points.
        Each bullet point should explain what needs to be done if written in code.

        ### User:
        Analyze through the API documentation in this link provided below.

        ### Context:
        {context}

        ### Link:
        {link}
    """

    prompt = PromptTemplate(
        input_variables=["link", "context"],
        template = template
    )

    final_prompt = prompt.format(link=link, context= context)

    result = llm(final_prompt)
    st.write(result)
    return result

def generateCode(steps):

    template = """
        ### System:
        You are a senior back end python developer specialising in API integration.
        If you are not sure about the code then please let us know and build it in to a python class that can be reused
        Generate code based on the steps below

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

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=final_prompt,
        max_tokens=2048,  # Adjust based on desired code length
        n=1,             # Number of code samples to generate
        stop=None,       # Specify an optional stop condition for generated code
        temperature=0  # Controls the randomness of the generated code
    )
    code =  response.choices[0].text.strip()
    
    return code

def generateCodeFromSwagger(URL, language):
    template = """
            ### System:
            You are a senior back end developer specialising in API integration.
            If you are not sure about the code then please let us know and build it in to a class that can be reused.
            Based on the Swagger URL provided below, generate an API call using the language provided below.
            If the request you are making needs a request body or request header, add placeholder values to it.
            If the langauge requested is javascript, write a javascript code using node.js and axios.
            Response should be code wrapped in a function.

            ### User:
            Analyze the API document in the Swagger URL and create an API call based on the what you analyzed written in the language provide below.
            
            ### URL:
            {URL}

            ### language:
            {language}
        """
    
    prompt = PromptTemplate(
        input_variables=["URL", "language"],
        template = template
    )

    final_prompt = prompt.format(URL=URL, language=language)

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=final_prompt,
        max_tokens=2048, 
        n=1,            
        stop=None,       
        temperature=0  
    )
    code =  response.choices[0].text.strip()
    
    return code

def main():
    st.set_page_config(page_title = "Some Webpage", page_icon = ":tada:", layout="wide")

    st.subheader("Hello, World :wave:")

    selected_option = st.radio("Select an option", ("API Documentation", "Swagger Documentation"))

    if selected_option == 'API Documentation':
        context = st.text_input("Context")
        st.write("You entered", context)

    link = st.text_input("Enter link here")
    st.write("You entered", link)

    if selected_option == 'Swagger Documentation':
        language = st.selectbox("Select a language", ("Python", "Javascript"))

    if st.button("Submit"):
        if selected_option == 'API Documentation':
            steps = getSteps(link, context)
            code = generateCode(steps)
            st.write(code)
        elif selected_option == 'Swagger Documentation':
            code = generateCodeFromSwagger(link, language)
            st.write(code)

if __name__ == "__main__":
    main()