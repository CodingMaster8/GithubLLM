
import streamlit as st
from llama_index.vector_stores import TimescaleVectorStore
from llama_index import ServiceContext, StorageContext
from llama_index.indices.vector_store import VectorStoreIndex
from llama_index.llms import OpenAI
from llama_index import set_global_service_context


from datetime import datetime, timedelta

import psycopg2
from dateparser.search import search_dates
from llama_index.vector_stores.types import MetadataInfo, VectorStoreInfo
import requests
from urllib.parse import urlparse


common_file_extensions = {
    '.py': 'Python script file',
    '.pyc': 'Compiled Python file',
    '.pyo': 'Optimized Python file',
    '.pyw': 'Python script for Windows',
    '.ipynb': 'Jupyter Notebook file',
    '.txt': 'Plain text file',
    '.md': 'Markdown file',
    '.json': 'JSON file',
    '.csv': 'Comma-separated values file',
    '.xml': 'XML file',
    '.html': 'HTML file',
    '.css': 'Cascading Style Sheets file',
    '.js': 'JavaScript file',
    '.yaml': 'YAML file',
    '.yml': 'YAML file',
    '.ini': 'Configuration file',
    '.cfg': 'Configuration file',
    '.toml': 'TOML configuration file',
    '.log': 'Log file',
    '.sql': 'SQL file',
    '.db': 'Database file',
    '.sqlite': 'SQLite database file',
    '.sqlite3': 'SQLite 3 database file',
    '.h5': 'HDF5 file',
    '.hdf5': 'HDF5 file',
    '.pickle': 'Pickle file',
    '.pkl': 'Pickle file',
    '.tar': 'Tape Archive file',
    '.gz': 'Gzip compressed file',
    '.zip': 'ZIP compressed file',
    '.rar': 'RAR compressed file',
    '.7z': '7-Zip compressed file',
    '.exe': 'Executable file',
    '.dll': 'Dynamic Link Library file',
    '.so': 'Shared object file',
    '.o': 'Object file',
    '.a': 'Static library file',
    '.whl': 'Wheel file (Python package)',
    '.egg': 'Egg file (Python package)',
    '.sh': 'Shell script file',
    '.bat': 'Batch file',
    '.ps1': 'PowerShell script file',
    '.vbs': 'VBScript file',
    '.pl': 'Perl script file',
    '.rb': 'Ruby script file',
    '.java': 'Java source file',
    '.class': 'Java class file',
    '.jar': 'Java Archive file',
    '.c': 'C source file',
    '.h': 'C header file',
    '.cpp': 'C++ source file',
    '.hpp': 'C++ header file',
    '.cs': 'C# source file',
    '.go': 'Go source file',
    '.rs': 'Rust source file',
    '.swift': 'Swift source file',
    '.kt': 'Kotlin source file',
    '.m': 'Objective-C source file',
    '.mm': 'Objective-C++ source file',
    '.dart': 'Dart source file',
    '.php': 'PHP source file',
    '.asp': 'Active Server Pages file',
    '.aspx': 'Active Server Pages Extended file',
    '.jsp': 'JavaServer Pages file',
    '.ts': 'TypeScript file',
    '.tsx': 'TypeScript JSX file',
    '.jsx': 'JavaScript JSX file',
}


def extract_filename(query):
    # Split the query into words
    words = query.split()

    # Iterate over each word to find the one with a dot character
    for word in words:
        if '.' in word:
            # Check if the word ends with any of the known extensions
            for ext in common_file_extensions.keys():
                if word.endswith(ext):
                    return word
    return None


def preprocess_query(query):
    # Parse specific phrases like "last week", "last month", etc.
    now = datetime.now()
    date_format = "%Y-%m-%d"

    if "last week" in query:
        start_date = (now - timedelta(days=now.weekday() + 7)).strftime(date_format)
        end_date = (now - timedelta(days=now.weekday() + 1)).strftime(date_format)
        query = query.replace("last week", f"{start_date} to {end_date}")
    elif "last month" in query:
        first_day_last_month = now.replace(day=1) - timedelta(days=1)
        start_date = first_day_last_month.replace(day=1).strftime(date_format)
        end_date = first_day_last_month.strftime(date_format)
        query = query.replace("last month", f"{start_date} to {end_date}")

    elif "to" in query:
        # Parse month names and other date phrases using search_dates
        date_search_result = search_dates(query, settings={'PREFER_DATES_FROM': 'past'})
        if date_search_result and len(date_search_result) >= 2:
            start_date = date_search_result[0][1].isoformat()  # Convert to ISO format
            end_date = date_search_result[1][1].isoformat()  # Convert to ISO format
            query = query.replace(f"{date_search_result[0][0]} to {date_search_result[1][0]}",
                                  f"{start_date} to {end_date}")

    else:
        # Parse month names and other date phrases using search_dates
        date_search_result = search_dates(query, settings={'PREFER_DATES_FROM': 'past'})
        if date_search_result:
            for date_str, date_obj in date_search_result:
                if isinstance(date_obj, datetime):
                    if date_obj.day == now.day or date_obj.day == now.day+1:  # Only the month is specified
                        # Set start date to the first day of the specified month and end date to the last day
                        start_date = date_obj.replace(day=1).isoformat()  # Convert to ISO format
                        end_date = (date_obj.replace(day=1) + timedelta(days=32)).replace(day=1) - timedelta(days=1)
                        end_date = end_date.isoformat()  # Convert to ISO format
                        query = query.replace(date_str, f"{start_date} to {end_date}")
                    else:
                        start_date = date_obj.isoformat()  # Convert to ISO format
                        end_date = (date_obj + timedelta(hours=23)).isoformat()
                        query = query.replace(date_str, f"{start_date} to {end_date}")

                elif isinstance(date_obj, str) and date_str.lower() == date_obj.lower():
                    # Handle cases where the month name is lowercase in the query but uppercase in the result
                    query = query.replace(date_str, f"{start_date} to {end_date}")
    return query


def get_repos():
    with psycopg2.connect(dsn=st.secrets["TIMESCALE_SERVICE_URL"]) as connection:
        # Create a cursor within the context manager
        with connection.cursor() as cursor:
            try:
                select_data_sql = "SELECT * FROM time_machine_catalog;"
                cursor.execute(select_data_sql)
            except psycopg2.errors.UndefinedTable as e:
                return {}

            catalog_entries = cursor.fetchall()

            catalog_dict = {}
            for entry in catalog_entries:
                repo_url, table_name = entry
                catalog_dict[repo_url] = table_name

            return catalog_dict


def get_auto_retriever(index, retriever_args, code, content, file_name):
    if code == True:
        vector_store_info = VectorStoreInfo(
            content_info=f"Description of the commits to a repository. Describes changes made to the repository, also the file {file_name} is included, which "
                         f"implement the following code {content}",
            metadata_info=[
                MetadataInfo(
                    name="commit_hash",
                    type="str",
                    description="Commit Hash",
                ),
                MetadataInfo(
                    name="author",
                    type="str",
                    description="Author of the commit",
                ),
                MetadataInfo(
                    name="__start_date",
                    type="datetime in iso format",
                    description="All results will be after this datetime",

                ),
                MetadataInfo(
                    name="__end_date",
                    type="datetime in iso format",
                    description="All results will be before this datetime",

                )
            ],
        )
    else:
        vector_store_info = VectorStoreInfo(
            content_info="Description of the commits to a repository. Describes changes made to the repository",
            metadata_info=[
                MetadataInfo(
                    name="commit_hash",
                    type="str",
                    description="Commit Hash",
                ),
                MetadataInfo(
                    name="author",
                    type="str",
                    description="Author of the commit",
                ),
                MetadataInfo(
                    name="__start_date",
                    type="datetime in iso format",
                    description="All results will be after this datetime",

                ),
                MetadataInfo(
                    name="__end_date",
                    type="datetime in iso format",
                    description="All results will be before this datetime",

                )
            ],
        )
    from llama_index.indices.vector_store.retrievers import VectorIndexAutoRetriever
    retriever = VectorIndexAutoRetriever(index,
                                         vector_store_info=vector_store_info,
                                         service_context=index.service_context,
                                         **retriever_args)

    # build query engine
    from llama_index.query_engine.retriever_query_engine import RetrieverQueryEngine
    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever, service_context=index.service_context
    )

    from llama_index.tools.query_engine import QueryEngineTool
    # convert query engine to tool
    query_engine_tool = QueryEngineTool.from_defaults(query_engine=query_engine)

    from llama_index.agent import OpenAIAgent
    chat_engine = OpenAIAgent.from_tools(
        tools=[query_engine_tool],
        llm=index.service_context.llm,
        verbose=True
        # service_context=index.service_context
    )
    return chat_engine


def extract_repo_info(repo_url):
    parsed_url = urlparse(repo_url)
    path_parts = parsed_url.path.strip('/').split('/')
    if len(path_parts) >= 2:
        owner = path_parts[0]
        repo = path_parts[1]
        return owner, repo
    else:
        raise ValueError("Invalid repository URL")


def get_repo_info(repo_url, branch='main', token=None, display_files=False):
    owner, repo = extract_repo_info(repo_url)

    def list_files(owner, repo, path='', branch='main', token=None):
        url = f'https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={branch}'
        headers = {'Accept': 'application/vnd.github.v3+json'}
        if token:
            headers['Authorization'] = f'token {token}'
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            items = response.json()
            all_files = []
            for item in items:
                if item['type'] == 'file':
                    all_files.append(item['path'])
                elif item['type'] == 'dir':
                    all_files.extend(list_files(owner, repo, item['path'], branch, token))
            return all_files
        else:
            raise Exception(f'Error: {response.status_code}, {response.json()}')

    all_files = list_files(owner, repo, branch=branch, token=token)
    code_extensions = {'.py', '.c', '.cpp', '.java', '.js', '.ts', '.rb', '.go', '.rs'}
    code_files = [file for file in all_files if any(file.endswith(ext) for ext in code_extensions)]

    if display_files:
        print("Code files in the repository:")
        for idx, file in enumerate(code_files):
            print(f"{idx}: {file}")

    return owner, repo, code_files


def get_file_contents(repo_url, file_name, branch='main', token=None, display_content=False):
    owner, repo, code_files = get_repo_info(repo_url, branch=branch, token=token, display_files=False)

    # Se busca el path de un archivo específico a partir de su nombre
    file_path = next((file for file in code_files if file.endswith(f'/{file_name}')), None)
    if not file_path:
        raise ValueError(f"File {file_name} not found in the repository.")

    url = f'https://api.github.com/repos/{owner}/{repo}/contents/{file_path}?ref={branch}'
    headers = {'Accept': 'application/vnd.github.v3.raw'}
    if token:
        headers['Authorization'] = f'token {token}'
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        content = response.text
        if display_content:
            print(f"\nContents of {file_path}:")
            print(content)
        return content
    else:
        raise Exception(f'Error: {response.status_code}, {response.json()}')


def tm_demo():
    repos = get_repos()

    months = st.sidebar.slider('How many months back to search (0=no limit)?', 0, 130, 0)

    if "config_months" not in st.session_state.keys() or months != st.session_state.config_months:
        st.session_state.clear()

    topk = st.sidebar.slider('How many commits to retrieve', 1, 150, 20)
    if "config_topk" not in st.session_state.keys() or topk != st.session_state.config_topk:
        st.session_state.clear()

    if len(repos) > 0:
        repo = st.sidebar.selectbox("Choose a repo", repos.keys())
    else:
        st.error("No repositories found, please [load some data first](/LoadData)")
        return

    if "config_repo" not in st.session_state.keys() or repo != st.session_state.config_repo:
        st.session_state.clear()

    st.session_state.config_months = months
    st.session_state.config_topk = topk
    st.session_state.config_repo = repo

    if "messages" not in st.session_state.keys():  # Initialize the chat messages history
        st.session_state.messages = [
            {"role": "assistant",
             "content": "Please choose a repo and time filter on the sidebar and then ask me a question about the git history"}
        ]

    vector_store = TimescaleVectorStore.from_params(
        service_url=st.secrets["TIMESCALE_SERVICE_URL"],
        table_name=repos[repo],
        time_partition_interval=timedelta(days=7),
    )

    service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.3))
    set_global_service_context(service_context)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store, service_context=service_context)

    # Check if there is a prompt from the user
    if prompt := st.chat_input("Your question"):  # Prompt for user input and save to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

    # Initialize code-related variables
    code = False
    content = ""
    file_name = None

    # Extract the filename from the prompt if it exists
    if prompt:
        file_name = extract_filename(prompt)
        print(file_name)
        print(repo)

    # If a filename was found in the prompt, retrieve its contents
    if file_name:
        try:
            token = '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'  # Replace with your actual token
            content = get_file_contents(repo, file_name, branch="main", token=token, display_content=True)
            code = True
            print("Success Getting File Content")
        except ValueError as e:
            st.error(str(e))
        except Exception as e:
            print("Exeption Error")
            st.error(f"Failed to retrieve file contents: {str(e)}")
            #import traceback
            #traceback.print_exc()

    # Chat engine goes into the session to retain history
    if "chat_engine" not in st.session_state.keys():  # Initialize the chat engine
        retriever_args = {"similarity_top_k": int(topk)}
        if months > 0:
            end_dt = datetime.now()
            start_dt = end_dt - timedelta(weeks=4 * months)
            retriever_args["vector_store_kwargs"] = ({"start_date": start_dt, "end_date": end_dt})
        st.session_state.chat_engine = get_auto_retriever(index, retriever_args, code, content, file_name)

    for message in st.session_state.messages:  # Display the prior chat messages
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # If last message is not from assistant, generate a new response
    if st.session_state.messages and st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                modified_prompt = preprocess_query(prompt)
                response = st.session_state.chat_engine.chat(modified_prompt, function_call="query_engine_tool")
                st.write(response.response)
                message = {"role": "assistant", "content": response.response}
                st.session_state.messages.append(message)  # Add response to message history

# Configure the Streamlit page
st.set_page_config(page_title="Github LLM Beta", page_icon="🧑‍💼")
st.markdown("# Github LLM Beta 0.1")
st.sidebar.header("Welcome to the Github LLM Beta")

# Set up logging for debugging if necessary
debug_llamaindex = False
if debug_llamaindex:
    import logging
    import sys

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Run the demo function
tm_demo()
print("Success")
