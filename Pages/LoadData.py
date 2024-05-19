from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
import time
import subprocess
import shutil
import psycopg2

import numpy as np

import streamlit as st
from streamlit.hello.utils import show_code

from llama_index.vector_stores import TimescaleVectorStore
from llama_index import StorageContext
from llama_index.indices.vector_store import VectorStoreIndex

import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from timescale_vector import client

from typing import List, Tuple

from llama_index.schema import TextNode
from llama_index.embeddings import OpenAIEmbedding
from git import Repo

from llama_index.text_splitter import SentenceSplitter



def create_uuid(date_string: str):
    """ Takes a string date_string representing a date and time in ISO format (YYYY-MM-DDTHH:MM:SS) as input.
    It converts this string into a datetime object using datetime.fromisoformat(date_string)
    and then generates a UUID (Universally Unique Identifier) based on this datetime object
    using a hypothetical client.uuid_from_time(datetime_obj) method.
    The UUID is then converted to a string using str(uuid) and returned."""

    datetime_obj = datetime.fromisoformat(date_string)
    uuid = client.uuid_from_time(datetime_obj)
    return str(uuid)


# Create a Node object from a single row of data // This could be changed for LangChain
def create_nodes(row):
    """takes a pandas DataFrame row (row) as input and creates a list of TextNode objects based on the information in the row.

    Here's a breakdown of what the function does:

    It creates a SentenceSplitter object (text_splitter from llama.index) with a chunk_size of 1024. This object is used to split the text content into chunks.

    It converts the row into a dictionary (record) and then extracts relevant information (record_content) from the dictionary, such as "Date", "Author", "Subject", and "Body", concatenating them into a single string.

    It uses the text_splitter to split the record_content into chunks (text_chunks) based on the specified chunk_size.

    It creates a list of TextNode objects (nodes) using a list comprehension. Each TextNode object has an id_ generated from the "Date" field using the create_uuid function, a text chunk from text_chunks, and metadata containing additional information such as "Commit Hash", "Author", and "Date" from the record dictionary.

    Finally, the function returns the list of TextNode objects (nodes) created based on the input row."""

    text_splitter = SentenceSplitter(chunk_size=1024)

    record = row.to_dict()
    record_content = (
            "Date: " + str(record["Date"])
            + " "
            + "Author: " + record['Author']
            + " "
            + "Changes: " + str(record["Subject"])
            + " "
            + str(record["Body"])
            + " "
            + "Files Changed: " + str(record["Files Changed"])
    )

    text_chunks = text_splitter.split_text(record_content)
    nodes = [TextNode(
        id_=create_uuid(record["Date"]),
        text=chunk,
        metadata={
            "commit_hash": record["Commit Hash"],
            "author": record['Author'],
            "date": record["Date"],
            "files_changes": record["Files Changed"]
        },
    ) for chunk in text_chunks]

    st.write(nodes[0].get_content(metadata_mode="all"))

    return nodes


def github_url_to_table_name(github_url):
    """ This function replaces the / for _ in the url and adds a li_ at the start, returns the string as table_name"""
    repository_path = github_url.replace("https://github.com/", "")
    table_name = "li_" + repository_path.replace("/", "_")
    return table_name


def record_catalog_info(repo):
    """handle recording information about a repository into a PostgreSQL database using the psycopg2 library,
    which is a PostgreSQL adapter for Python.

    repo: link of a Github repository.

    - It establishes a connection to a PostgreSQL database.
    - Creates a cursor -> A cursor allows executing SQL commands in the database.
    - It then executes SQL commands within the cursor:
         - Creates a table named time_machine_catalog if it doesn't already exist. This table has two columns: repo_url and table_name.
         - Deletes any existing record in the time_machine_catalog table with the same repo_url as the one passed to the function.
         - Prepares an SQL statement to insert data into the time_machine_catalog table. This statement inserts the repo_url and a table_name into the table.
    """

    with psycopg2.connect(dsn=st.secrets["TIMESCALE_SERVICE_URL"]) as connection:
        # Create a cursor within the context manager
        with connection.cursor() as cursor:
            # Define the Git catalog table creation SQL command
            create_table_sql = """CREATE TABLE IF NOT EXISTS time_machine_catalog (
            repo_url TEXT PRIMARY KEY, 
            table_name TEXT);"""
            cursor.execute(create_table_sql)

            delete_sql = "DELETE FROM time_machine_catalog WHERE repo_url = %s"
            cursor.execute(delete_sql, (repo,))

            insert_data_sql = """
            INSERT INTO time_machine_catalog (repo_url, table_name)
            VALUES (%s, %s);
            """

            table_name = github_url_to_table_name(repo)
            cursor.execute(insert_data_sql, (repo, table_name))
            return table_name


def load_into_db(table_name, df_combined):
    """efficiently loads data into a TimescaleDB database table.
    leveraging parallel processing for embedding generation and database insertion while providing progress
    updates to the user. It also creates an index to optimize query performance after data insertion."""

    # Setting up the Embedding Model
    embedding_model = OpenAIEmbedding()
    embedding_model.api_key = st.secrets["OPENAI_API_KEY"]

    # Setting up the VectorStore of TimeScale
    ts_vector_store = TimescaleVectorStore.from_params(
        service_url=st.secrets["TIMESCALE_SERVICE_URL"],
        table_name=table_name,
        time_partition_interval=timedelta(days=7),
    )

    # Full clean of the existing tables on the database // Should be changed in future
    ts_vector_store._sync_client.drop_table()
    ts_vector_store._sync_client.create_tables()

    # This is done to divide the data into manageable chunks for parallel processing.
    cpus = cpu_count()
    st.write(cpus)
    st.write(df_combined.index)
    min_splits = len(df_combined.index) / 1000  # no more than 1000 rows/split
    num_splits = int(max(cpus, min_splits))

    # It displays a spinner indicating processing is ongoing and initializes a progress bar.
    st.spinner("Processing...")
    progress = st.progress(0, f"Processing, with {num_splits} splits")
    start = time.time()

    # Iterates over the rows of the dataframe, creating nodes using the create_nodes function, and flattens the resulting list into nodes_combined.
    nodes_combined = [item for sublist in [create_nodes(row) for _, row in df_combined.iterrows()] for item in sublist]

    # Splits nodes into num_splits smaller tasks (node_tasks) for parallel processing.
    node_tasks = np.array_split(nodes_combined, num_splits)

    def worker(nodes):
        """
        Defines a worker function (worker) that processes a chunk of nodes.

        This function retrieves content for each node, generates embeddings using the embedding model,
        assigns the embeddings to the corresponding nodes, and adds the nodes to the Timescale Vector Store.
        """

        start = time.time()
        texts = [n.get_content(metadata_mode="all") for n in nodes]
        embeddings = embedding_model.get_text_embedding_batch(texts)
        for i, node in enumerate(nodes):
            node.embedding = embeddings[i]
        duration_embedding = time.time() - start
        start = time.time()
        ts_vector_store.add(nodes)
        duration_db = time.time() - start
        return (duration_embedding, duration_db)

    embedding_durations = []
    db_durations = []

    # Uses a ThreadPoolExecutor to execute the worker function in parallel over the node tasks.
    with ThreadPoolExecutor() as executor:
        times = executor.map(worker, node_tasks)

        for index, worker_times in enumerate(times):
            duration_embedding, duration_db = worker_times
            embedding_durations.append(duration_embedding)
            db_durations.append(duration_db)
            progress.progress((index + 1) / num_splits, f"Processing, with {num_splits} splits")

    progress.progress(100,
                      f"Processing embeddings took {sum(embedding_durations)}s. Db took {sum(db_durations)}s. Using {num_splits} splits")

    st.spinner("Creating the index...")
    progress = st.progress(0, "Creating the index")
    start = time.time()
    ts_vector_store.create_index()  #This can be updated!!!
    duration = time.time() - start
    progress.progress(100, f"Creating the index took {duration} seconds")
    st.success("Done")


def get_history(repo, branch, limit):
    """Retrieves the commit history from a specified branch of a Git repository,
    processes it, and returns it as a pandas DataFrame."""

    st.spinner("Fetching git history...")
    start = time.time()
    progress = st.progress(0, "Fetching git history")
    # Clean up any existing "tmprepo" directory
    shutil.rmtree("tmprepo", ignore_errors=True)

    # Clone the Git repository with the specified branch
    res = subprocess.run(
        [
            "git",
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            "--single-branch",
            "--branch=" + branch,
            repo + ".git",
            "tmprepo",
        ],
        capture_output=True,
        text=True,
        cwd=".",  # Set the working directory here
    )

    if res.returncode != 0:
        st.error("Error running Git \n\n" + str(res.stderr))
        raise ValueError(f"Git failed: {res.returncode}")

    repo = Repo('tmprepo')

    # Create lists to store data
    commit_hashes = []
    authors = []
    dates = []
    subjects = []
    bodies = []
    files_changed = []  # New list to store files changed in each commit

    # Iterate through commits and collect data
    for commit in repo.iter_commits():
        commit_hash = commit.hexsha
        author = commit.author.name
        date = commit.committed_datetime.isoformat()
        message_lines = commit.message.splitlines()
        subject = message_lines[0]
        body = "\n".join(message_lines[1:]) if len(message_lines) > 1 else ""

        commit_hashes.append(commit_hash)
        authors.append(author)
        dates.append(date)
        subjects.append(subject)
        bodies.append(body)

        # Get the list of files changed in this commit
        changed_files = []
        for file in commit.stats.files.keys():
            changed_files.append(file)
        files_changed.append(", ".join(changed_files))

    # Create a DataFrame from the collected data
    data = {
        "Commit Hash": commit_hashes,
        "Author": authors,
        "Date": dates,
        "Subject": subjects,
        "Body": bodies,
        "Files Changed": files_changed
    }

    df = pd.DataFrame(data)

    # Light data cleaning on DataFrame
    df = df.astype(str)
    if limit > 0:
        df = df[:limit]

    duration = time.time() - start
    progress.progress(100, f"Fetching git history took {duration} seconds")
    return df


def load_git_history():
    """provides an interactive interface for users to load Git commit history data
    from a specified repository and branch into a database"""

    repo = st.text_input("Repo", "https://github.com/postgres/postgres")
    branch = st.text_input("Branch", "main")
    limit = int(st.text_input("Limit number commits (0 for no limit)", "1000"))

    if st.button("Load data into the database"):
        df = get_history(repo, branch, limit)
        table_name = record_catalog_info(repo)
        load_into_db(table_name, df)


st.set_page_config(page_title="Load git history", page_icon="💿")
st.markdown("# Load git history for analysis")
st.sidebar.header("Load git history")
st.write(
    """Load Git history!"""
)
if st.secrets.get("ENABLE_LOAD") == 1:
    load_git_history()
else:
    st.warning(
        "Loading is disabled on the demo site. Please follow the instructions in the [README](https://github.com/cevian-streamlit/tsv-timemachine/tree/main) to enable loading.")
# show_code(tm_demo)
