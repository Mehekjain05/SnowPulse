from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.messages import SystemMessage
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from prompts import QUALIFIED_TABLE_NAME
import streamlit as st
import os
import ast
import re

# Streamlit page config
st.set_page_config(page_title="SnowPulse", layout="wide")
st.title("SnowPulse")

st.write("""👋 Welcome to SnowPulse! ❄️

I'm here to help you explore School Infrastructure and Facilities Data. Whether you're looking for information on water supply, school management, or any other facility, feel free to ask me!

Just type your question, and I'll do my best to provide accurate and insightful answers. Let's get started! 🚀""")

# Snowflake DB connection parameters
DATABASE = "TEST"
SCHEMA = "PUBLIC"
WAREHOUSE = "COMPUTE_WH"
ROLE = "ACCOUNTADMIN"

# Helper function to extract query results as a list
def query_as_list(db, query):
    res = db.run(query)
    res = [el for sub in ast.literal_eval(res) for el in sub if el]
    res = [re.sub(r"\b\d+\b", "", string).strip() for string in res]
    return list(set(res))

# Check if connection is already established
if 'db' not in st.session_state:
    with st.spinner("Connecting to Snowflake..."):
        snowflake_url = f"snowflake://{os.getenv('SNOWFLAKE_USER')}:{os.getenv('SNOWFLAKE_PASSWORD')}@{os.getenv('SNOWFLAKE_ACCOUNT')}/{DATABASE}/{SCHEMA}?warehouse={WAREHOUSE}&role={ROLE}"

        # Initialize the database connection
        db = SQLDatabase.from_uri(snowflake_url, sample_rows_in_table_info=2)

        # Initialize the OpenAI LLM and embeddings model
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        embedding = OpenAIEmbeddings(model="text-embedding-3-small")
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        tools = toolkit.get_tools()

        # Store connection and tools in session state
        st.session_state.db = db
        st.session_state.llm = llm
        st.session_state.tools = tools

        # Fetch data for vector store
        years = query_as_list(st.session_state.db, f"SELECT ACADEMIC_YEAR FROM {QUALIFIED_TABLE_NAME}")
        state_names = query_as_list(st.session_state.db, f"SELECT STATE_NAME FROM {QUALIFIED_TABLE_NAME}")
        district_names = query_as_list(st.session_state.db, f"SELECT DISTRICT_NAME FROM {QUALIFIED_TABLE_NAME}")
        block_name = query_as_list(st.session_state.db, f"SELECT UDISE_BLOCK_NAME FROM {QUALIFIED_TABLE_NAME}")
        school_category_name = query_as_list(st.session_state.db, f"SELECT SCHOOL_CATEGORY_NAME FROM {QUALIFIED_TABLE_NAME}")
        school_management_name = query_as_list(st.session_state.db, f"SELECT SCHOOL_MANAGEMENT_NAME FROM {QUALIFIED_TABLE_NAME}")
        location = query_as_list(st.session_state.db, f"SELECT LOCATION FROM {QUALIFIED_TABLE_NAME}")
        school_type = query_as_list(st.session_state.db, f"SELECT SCHOOL_TYPE FROM {QUALIFIED_TABLE_NAME}")

        # FAISS vector store and retriever
        vector_db = FAISS.from_texts(years + state_names + district_names + block_name + school_category_name + school_management_name + school_type, embedding)
        vector_db.save_local("faiss_index")
        old_db = FAISS.load_local("faiss_index", embedding, allow_dangerous_deserialization=True)
        st.session_state.retriever = old_db.as_retriever(search_kwargs={"k": 5})
        st.success("Snowflake connection established successfully!")

# Use the database connection from the session state
db = st.session_state.db
retriever = st.session_state.retriever

# Retriever tool for proper noun
description = """Use to look up values to filter on. Input is an approximate spelling of the proper noun, output is \
valid proper nouns. Use the noun most similar to the search."""
retriever_tool = create_retriever_tool(
    retriever,
    name="search_proper_nouns",
    description=description,
)
st.session_state.tools.append(retriever_tool)

# Define the system message for the agent
system = """You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct SQLite query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the given tools. Only use the information returned by the tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

You have access to the following tables: {table_names}

If you need to filter on a proper noun, you must ALWAYS first look up the filter value using the "search_proper_nouns" tool!
Do not try to guess at the proper name - use this function to find similar ones.""".format(
    table_names=db.get_usable_table_names()
)

system_message = SystemMessage(content=system)

# Create the agent
agent = create_react_agent(st.session_state.llm, st.session_state.tools, state_modifier=system_message)



# Input query from the user
user_query = st.text_input("Enter your question:", value="What are the total number of schools that have functional drinking water?")

if st.button("Run Query"):
    with st.spinner("Thinking..."):
        # Create a chat message container to stream responses
        reasoning_steps = []
        full_reasoning = ""
        
        # Display the reasoning in an expander
        reasoning_expander = st.expander("🔄 Thinking... Click to view reasoning steps", expanded=False)
        
        # Stream the agent's reasoning steps and results
        for response in agent.stream({"messages": [HumanMessage(content=user_query)]}):
            reasoning_steps.append(response)
            
            # Format and append reasoning steps
            if 'agent' in response:
                if 'tool_calls' in response['agent']['messages'][0].additional_kwargs:
                    if 'function' in response['agent']['messages'][0].additional_kwargs['tool_calls'][0]:
                        function_name = response['agent']['messages'][0].additional_kwargs['tool_calls'][0]['function']['name']
                        function_args = response['agent']['messages'][0].additional_kwargs['tool_calls'][0]['function']['arguments']
                        step = f"**Function called**: `{function_name}` with arguments: `{function_args}`"
                else:
                    step = response['agent']['messages'][0].content
            else:
                step = response['tools']['messages'][0].content

            # Display each step as the agent processes
            full_reasoning += f"{step}\n\n"
            with reasoning_expander:
                st.markdown(f"{full_reasoning}")
        

        st.success("Query completed!")

        # Optionally show the final result (since it will be streamed above as well)
        st.write("### Final Result:")
        st.write(response['agent']['messages'][0].content)
