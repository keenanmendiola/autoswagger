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
import pandas as pd

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

def generateCode(URL, body, params, query_string, security_header):
    template = """
            ### System:
            You are a senior back end developer specialising in API integration.
            
            ### User:
            Create a python function to perform a network request based on the URL, body, params, query_string, and security_header provided below.
            For the security header line, add a comment saying 'Add this to your environment variables.'
            
            ### URL:
            {URL}

            ### body:
            {body}

            ### params:
            {params}

            ### query_string:
            {query_string}

            ### security_header:
            {security_header}
        """
    
    prompt = PromptTemplate(
        input_variables=["URL", "body", "params", "query_string", "security_header"],
        template = template
    )

    final_prompt = prompt.format(URL=URL, body=body, params=params, query_string=query_string, security_header=security_header)


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

def readSwaggerDoc(swaggerLink):
    template = """
        ### System:
        Understand, you are a senior back end developer specialising in API integration.
        A swagger documentation provides a list of endpoints and its details such as request body, parameters, queries.

        ### User:
        Analyze the API document in the Swagger URL and answer with the following:
            - the link to the JSON file of the Swagger URL, only return the link of the JSON file.
        It is possible that the JSON file link is written in a different format.
        
        ### URL:
        {URL}
    """
    
    prompt = PromptTemplate(
        input_variables=["URL"],
        template = template
    )

    final_prompt = prompt.format(URL=swaggerLink)

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=final_prompt,
        max_tokens=2048, 
        n=1,            
        stop=None,       
        temperature=0  
    )
    swaggerJsonLink =  response.choices[0].text.strip()
    
    return swaggerJsonLink 
    
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

def getBaseApiURL(url):
    template = """
            ### System:
            Understand, you are a senior back end developer specialising in API integration.
            A swagger documentation provides endpoint details such as request body, parameters, queries.

            ### User:
            Analyze the API document in the Swagger URL and answer with the following:
            - The base url for the API, only answer with the base url.

            ### URL:
            {url}
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

def find_url(string):
    regex = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    url = re.search(regex, string)
    
    if url:
        return url.group()
    else:
        return None

def extract_sub_dicts_with_key_value(dictionary, key, value):
    results = []
    for k, v in dictionary.items():
        if isinstance(v, dict):
            # Recursively search through the sub-dictionary
            sub_results = extract_sub_dicts_with_key_value(v, key, value)
            results.extend(sub_results)
        elif k == key and v == value:
            # Found a match, add the sub-dictionary to the results list
            results.append(dictionary)
    return results

def replace_placeholders(url, data):
    has_placeholder = False
    
    for item in data:
        if item["name"] in url:
            has_placeholder = True
            url = url.replace("{" + item["name"] + "}", str(item["value"]))
    
    if not has_placeholder:
        return url
    
    return url

def generateCodeForPath(baseUrl, path):
    template = """
            ### System:
            You are a senior back end developer specialising in API integration.
            
            ### User:
            Create a python function to perform a network request based on the contents of the dictionary provided below.
            The first key is the path of the endpoint which you will need to concatenate with the baseUrl provided below.
            The second key is the http verb such as get, post, put, delete. the value of the http verb key are the request details.
            The request details may include a parameters key that contains needed details for the request body, params, query strings.
            The request details also describes what the endpoint will consume and produce, and also the security needed, this can be added to the request headers.
            Do not use the dictionary in the function, extract only the needed values.
            For the security header, add a comment saying 'you can get this value from your environment variables.'
            Answer with the properly formatted python code only.

            ### baseUrl:
            {baseUrl}
            ### Request Dictionary:
            {path}
        """
    
    prompt = PromptTemplate(
        input_variables=["path", "baseUrl"],
        template = template
    )

    final_prompt = prompt.format(path=path, baseUrl=baseUrl)


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

    # if st.button("Submit"):
    output = readSwaggerDoc(link)
    url = find_url(output)
    baseUrl = getBaseApiURL(url)
    st.write(baseUrl)
    response = requests.get(url)
    
    try:
        swagger_data = response.json()
        grouped_paths = {}
        for path, path_details in swagger_data['paths'].items():
            for verb, verb_details in path_details.items():
                tag = verb_details.get('tags', ['Other'])[0]  # Default tag 'Other' if not specified
                row = {
                    'Path': path,
                    'HTTP Verb': verb.upper(),
                    'Description': verb_details.get('description', '')
                }
                if tag in grouped_paths:
                    grouped_paths[tag].append(row)
                else:
                    grouped_paths[tag] = [row]

        # Display tables for each tag
        for tag, paths in grouped_paths.items():
            st.subheader(f"Tag: {tag}")
            st.table(pd.DataFrame(paths))

        paths = swagger_data['paths']

        security = ['basic','apiKey', 'oauth2']
        selected_security = st.selectbox('Select an option:', security)

        if selected_security == "apiKey":
            key = st.text_input("Enter API Key")
        elif selected_security == "apiKey":
            key = st.text_input("Enter Client Id")
        elif selected_security == 'basic':
            for path in paths:
                if "login" in path:
                    input_values = {}
                    loginVerb = next(iter(paths[path]))
                    loginParams = paths[path][loginVerb]["parameters"]
                    for param in loginParams:
                        value = st.text_input(param["name"], key=param["name"])
                        # Store input value in the dictionary
                        input_values[param["name"]] = value
                    if st.button("Get Basic Authentication"):
                        st.write(input_values)
        
        apiCallOptions = st.radio("", ("Generate Code for calling endpoint/s", "Call and Generate code for an endpoint"))

        if apiCallOptions == "Generate Code for calling endpoint/s":
            selected_options = st.multiselect('Select paths', list(paths.keys()))
            for path in selected_options:
                generatedCode = generateCodeForPath(baseUrl, paths[path])
                st.write(generatedCode)
        elif apiCallOptions == "Call and Generate code for an endpoint":
        # Create a dropdown list with the paths
            selected_path = st.selectbox('Select a path', list(paths.keys()))
            
            # Get the request details for the selected path
            request_details = paths[selected_path]
            # st.write(request_details)

            # Check if there are any parameters defined for the selected path
            verb = next(iter(request_details))
            if 'parameters' in request_details[verb]:
                parameters = request_details[verb]['parameters']
            else:
                st.write("Empty")
                parameters = []

            request_dict = {}
            # Create a dynamic form based on the request details
            # with st.form(key='dynamic_form'):
            # Iterate over the parameters and create the corresponding input fields



            for parameter in parameters:
                param_name = parameter['name']
                param_type = parameter['type']
                param_options = parameter.get('items', [])
                param_in = parameter["in"]

                # Determine the appropriate input field based on the parameter type
                if param_type == 'string':
                    value = st.text_input(param_name)
                elif param_type == 'integer':
                    value = st.number_input(param_name, step=1, value=0)
                elif param_type == 'boolean':
                    value = st.checkbox(param_name)
                elif param_type == 'array':
                    if param_options:
                        value = st.selectbox('Select a value', list(param_options['enum']))
                
                # Store the parameter value in a dictionary, using the parameter name as the key
                request_dict[param_name] = {"in": param_in,"value": value, "name": param_name}
                    
            submit_button = st.button(label='Call the API')

            # Process the form submission if the submit button is clicked
            if submit_button:
                body = extract_sub_dicts_with_key_value(request_dict, "in", "body")
                form_data = extract_sub_dicts_with_key_value(request_dict, "in", "form_data")
                params = extract_sub_dicts_with_key_value(request_dict, "in", "path")
                query_string = extract_sub_dicts_with_key_value(request_dict, "in", "query")

                requestUrl = baseUrl + selected_path
                pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+/\S+'
                matches = re.findall(pattern, requestUrl)

                body.extend(form_data)
                
                newUrl = replace_placeholders(matches[0], params)

                new_body = {d["name"]: d["value"] for d in body}
                new_params = {d["name"]: d["value"] for d in params}
                new_query = {d["name"]: d["value"] for d in query_string}

                response = call_api(newUrl, body=new_body, params= new_query, query_string= new_query, http_verb=verb)
                st.write("Response", response.json())
                code = generateCode(URL=newUrl, body=new_body, params=new_params, query_string=new_query, security_header=f"api_key={key}")
                st.write(code)

    except Exception as e:
        st.write(e)

if __name__ == "__main__":
    main()