import streamlit as st
from dotenv import load_dotenv
import os
from langchain import OpenAI, PromptTemplate, HuggingFaceHub
from langchain.agents import AgentType
from langchain.agents.agent_toolkits import create_python_agent
from langchain.tools.python.tool import PythonREPLTool
from langchain.python import PythonREPL
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
import openai
import json
import ast
import re
import requests
load_dotenv()

openai.api_key=os.getenv('OPEN_API_KEY')

def getSteps(context, link):
    llm = OpenAI(openai_api_key=os.getenv('OPEN_API_KEY'), temperature=0)

    template = """
        ### System:
        I want you to act as a programmer that assists other programmers in development.
        Let us think step-by-step to make sure we get the right answer.
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
        Let us think step-by-step to make sure we get the right answer.
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
            Include any necessary packages needed.
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
    requestDetails = getRequestDetails(URL)

    # json_string = remove_starting_characters(requestDetails)
    # input_dictionary = json_to_dict(json_string)
    
    # return code, input_dictionary
    return code, requestDetails


def remove_starting_characters(text):
    start_index = text.find('{')
    if start_index != -1:
        json_string = text[start_index:]
        return json_string
    else:
        return None
    
def json_to_dict(json_string):
    try:
        json_dict = json.loads(json_string)
        return json_dict
    except json.JSONDecodeError as e:
        print("Invalid JSON string:", e)
        return None

def getRequestDetails(url):
    template = """
            ### System:
            Understand, you are a senior back end developer specialising in API integration.
            A swagger documentation provides endpoint details such as request body, parameters, queries.

            ### User:
            Analyze the API document in the Swagger URL and answer with the following:
             - a list that contains all possible request details for the request parameters, request query string, and request body, and its data types.
             - the link API endpoint that swagger document is trying to call.
            Follow the formoat provided below but respond it as a dictionary and use double quotes for each key and value pair.

            ### URL:
            {url}

            ### Format:
            "link": "find the actual link of the API endpoint being needed to be called, do not use the swagger document.",
            "request":
                "type": "the HTTP verb of the request."
                "body": "the fields required for the request body and their data type. should be an object.",
                "params": "the fields required for the query parameters and their data type. should be an object.",
                "query": "the fields required for the query string and their data type. should be an object."
                "schema": (should be an object.)
                    "fieldName": "the name of the request field. You can leave this empty if there are no options from the swagger document.",
                    "options": "if the type contains a list of options, add it to this list. You can leave this empty if there are no options from the swagger document."
        """
    
    prompt = PromptTemplate(
        input_variables=["url"],
        template = template
    )

    final_prompt = prompt.format(url=url)

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=final_prompt,
        max_tokens=2048, 
        n=1,            
        stop=None,       
        temperature=0  
    )
    request =  response.choices[0].text.strip()
    
    return request 

def generateCodeForRequest(input_dictionary, form_results):
    agent_executor = create_python_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613", openai_api_key=os.getenv('OPEN_API_KEY')),
        tool=PythonREPLTool(),
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        agent_executor_kwargs={"handle_parsing_errors": True},
    )

    prompt = """
        ### System
        You are a senior python developer trying to call an API endpoint that will create a python function to call the API in the url below.

        ### User:
        Below is the url of the API endpoint that needs to be called.
        Also included below are the request params/body possible values for the the request.
        And also, the actual values that are needed to be inputted in the request body/params/query.
        Use the request params/body possible values as reference and match it with the values provided to create your request.
        Given that, create a python function that will call the endpoint and return it.
        Answer with the python code you made.

        ### url
        {url}

        ### values
        {values}

        ## request params/body possible values
        {schema}

    """

    prompt = PromptTemplate(
        input_variables=["url", "values", "schema"],
        template = prompt
    )

    final_prompt = prompt.format(url=input_dictionary["link"], values=form_results, schema=input_dictionary["data"])

    result = agent_executor.run(final_prompt)
    st.write(result)
    runGeneratedCode(result)
    
    return result

# def getRequestDetails():


def runGeneratedCode(code):
    agent_executor = create_python_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-4", openai_api_key=os.getenv('OPEN_API_KEY')),
        tool=PythonREPLTool(),
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        agent_executor_kwargs={"handle_parsing_errors": True},
    )

    prompt = """
        Understand, you are provided with a python function below.
        Based on the import statements in the code, install any necessary packages.
        After installing the packages, run the code and respond to be the result of the function call.
        {code}
    """

    prompt = PromptTemplate(
        input_variables=["code"],
        template = prompt
    )

    final_prompt = prompt.format(code=code)

    result = agent_executor.run(final_prompt)

    st.write(result)

def render_form(json_data):

    form_results = {}

    for field in json_data["data"]:
        field_name = field["fieldName"]
        field_type = field["type"]
        field_options = field["options"]

        st.markdown(f"## {field_name}")

        if field_type == "string":
            if field_options:
                selected_option = st.selectbox("", field_options, key=f"select_{field_name}")
                form_results[field_name] = selected_option
                st.write(f"Selected Option: {selected_option}")
            else:
                user_input = st.text_input("", key=f"text_{field_name}")
                form_results[field_name] = user_input
                st.write(f"Entered Value: {user_input}")
        elif field_type == "number":
            if field_options:
                selected_option = st.selectbox("", field_options, key=f"select_{field_name}")
                form_results[field_name] = selected_option
                st.write(f"Selected Option: {selected_option}")
            else:
                user_input = st.number_input("", key=f"number_{field_name}")
                form_results[field_name] = user_input
                st.write(f"Entered Value: {user_input}")
        elif field_type == "boolean":
            user_input = st.checkbox("", value=False, key=f"checkbox_{field_name}")
            form_results[field_name] = user_input
            st.write(f"Value: {user_input}")
        else:
            st.write("Unsupported field type")

        st.markdown("---")

    if st.button("Call the API"):
        return form_results

def create_input_field(param_name, param_type, options):
    # Helper function to create different types of input fields
    if param_type == "integer":
        return st.number_input(param_name)
    elif param_type == "string":
        return st.text_input(param_name)
    elif param_type == "boolean":
        return st.checkbox(param_name)
    elif param_name in options:
        return st.selectbox(param_name, options[param_name]['options'])
    else:
        st.warning(f"Unknown data type: {param_type}")
        return None

def parse_json_string(json_str):
    # use ast.literal_eval() to parse non-quoted keys of the JSON string
    try:
        fixed_json = ast.literal_eval(json_str)
    except (ValueError, SyntaxError) as e:
        print(e)
        return None

    # use json.dumps() and json.loads() to normalize the JSON string
    try:
        fixed_json = json.loads(json.dumps(fixed_json))
        return fixed_json
    except ValueError as e:
        print(e)
        return None
    
def url_builder(url: str, params: dict) -> str:
    if '{' not in url or '}' not in url:
        return url
    for key, value in params.items():
        url = url.replace('{' + key + '}', str(value))
    return url

def call_api(url, body=None, params=None, query_string=None, http_verb='GET'):
    """
    Calls an API endpoint using the specified HTTP verb and passes any necessary data.

    :param url: The URL of the API endpoint.
    :param body: (optional) A dictionary containing the request body data.
    :param params: (optional) A dictionary containing any query parameters.
    :param query_string: (optional) A dictionary containing any query string parameters.
    :param http_verb: (optional) The HTTP verb to use. Defaults to 'GET'.

    :returns: The response object returned by the server.
    """
    # Define the headers for the request
    headers = {'Content-Type': 'application/json'}

    # Determine which HTTP verb to use
    if http_verb.upper() == 'GET':
        response = requests.get(url, headers=headers, json=body, params=params)
    elif http_verb.upper() == 'POST':
        response = requests.post(url, headers=headers, json=body, params=params)
    elif http_verb.upper() == 'PUT':
        response = requests.put(url, headers=headers, json=body, params=params)
    elif http_verb.upper() == 'DELETE':
        response = requests.delete(url, headers=headers, json=body, params=params)
    else:
        raise ValueError('Invalid HTTP verb specified.')

    # Check for errors in the response
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as err:
        print(err)
        return None

    # Return the response object
    return response

def main():
    st.set_page_config(page_title = "Some Webpage", page_icon = ":tada:", layout="wide")

    st.subheader("Hello, World :wave:")

    # selected_option = st.radio("Select an option", ("API Documentation", "Swagger Documentation"))

    # if selected_option == 'API Documentation':
    #     context = st.text_input("Context")
    #     st.write("You entered", context)

    # link = st.text_input("Enter link here")
    # st.write("You entered", link)

    # if selected_option == 'Swagger Documentation':
    #     language = st.selectbox("Select a language", ("Python", "Javascript"))

    # if st.button("Submit"):
    #     if selected_option == 'API Documentation':
    #         steps = getSteps(link, context)
    #         # code = generateCode(steps)
    #         # st.write(code)
    #     elif selected_option == 'Swagger Documentation':
    link = st.text_input("Enter link here")
    code, requestDetails = generateCodeFromSwagger(link, "Python")
    st.write(code)
    parsed_json = requestDetails.replace(" ", "")[11:]
    json_dict = json.loads(parsed_json)
    st.write(json_dict)
    request = json_dict["request"]


    with st.form("request_form"):

        # request_form = st.form(key="request_form")
        st.subheader(json_dict["link"])
        st.subheader(json_dict["request"]["type"])
        st.subheader("Body")
        body_values = {}
        for key, val in request["body"].items():
            if val == "integer":
                body_values[key] = st.number_input(f"Enter {key}", value=1)
            elif val == "string":
                if key in request["schema"]:
                    if request["schema"][key]["options"] != []:
                        body_values[key] = st.selectbox(f"Select {key}", options=request["schema"][key]["options"], key=f"query_{key}")
                    else:
                        body_values[key] = st.text_input(f"Enter {key}", key=f"query_{key}")
                else:
                    body_values[key] = st.text_input(f"Enter {key}", key=f"query_{key}")
            elif val == "date-time":
                body_values[key] = st.date_input(f"Enter {key}")
            elif val == "boolean":
                body_values[key] = st.checkbox(f"Select {key}", value=True)

        # Create input fields for each key-value pair in the params object
        st.subheader("Params")
        params_values = {}
        for key, val in request["params"].items():
            if val == "integer":
                params_values[key] = st.number_input(f"Enter {key}", key=f"params_{key}", value=1)
            elif val == "string":
                if key in request["schema"]:
                    if request["schema"][key]["options"] != []:
                        params_values[key] = st.selectbox(f"Select {key}", options=request["schema"][key]["options"], key=f"query_{key}")
                    else:
                        params_values[key] = st.text_input(f"Enter {key}", key=f"query_{key}")
                else:
                    params_values[key] = st.text_input(f"Enter {key}", key=f"query_{key}")
            elif val == "date-time":
                params_values[key] = st.date_input(f"Enter {key}", key=f"params_{key}")
            elif val == "boolean":
                params_values[key] = st.checkbox(f"Select {key}", key=f"params_{key}", value=True)

        # Create input fields and dropdown menus for each key-value pair in the query object
        st.subheader("Query")
        query_values = {}
        for key, val in request["query"].items():
            if val == "integer":
                query_values[key] = st.number_input(f"Enter {key}", key=f"query_{key}", value=1)
            elif val == "string":
                if key in request["schema"]:
                    if request["schema"][key]["options"] != []:
                        query_values[key] = st.selectbox(f"Select {key}", options=request["schema"][key]["options"], key=f"query_{key}")
                    else:
                        query_values[key] = st.text_input(f"Enter {key}", key=f"query_{key}")
                else:
                    query_values[key] = st.text_input(f"Enter {key}", key=f"query_{key}")
            elif val == "date-time":
                query_values[key] = st.date_input(f"Enter {key}", key=f"query_{key}")
            elif val == "boolean":
                query_values[key] = st.checkbox(f"Select {key}", key=f"query_{key}", value=True)

        submitted = st.form_submit_button("Call the API")
        
        if submitted:
            merged_dict = {}
            merged_dict.update(body_values)
            merged_dict.update(params_values)
            merged_dict.update(query_values)
            url = url_builder(json_dict["link"], merged_dict)
            st.write(url)
            response = call_api(url, body=body_values, params= params_values, query_string= query_values, http_verb=json_dict["request"]["type"])
            st.write(response.json())

if __name__ == "__main__":
    main()