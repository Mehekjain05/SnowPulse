import streamlit as st
import snowflake
import os
import pandas as pd
import plotly.express as px
from snowflake_connection import connect_to_snowflake
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

SCHEMA_PATH = "TEST.PUBLIC"
QUALIFIED_TABLE_NAME = f"{SCHEMA_PATH}.INFRASTRUCTURE_AVAILABILITY_AND_FACILITIES"
TABLE_DESCRIPTION = """
The "INFRASTRUCTURE_AVAILABILITY_AND_FACILITIES" table contains structured data on school infrastructure across districts, with a variety of data types for each column:

Geographic and School Information: Includes columns like ACADEMIC_YEAR (Date), STATE_CODE(Integer), DISTRICT_CODE(Integer), BLOCK_CODE (Integer), and SCHOOL_CATEGORY_NAME, SCHOOL_MANAGEMENT_NAME(VARCHAR), DISTRICT_NAME (VARCHAR), which identify the location and type of school.

Facilities Availability: Columns such as FUNCTIONAL_DRINKING_WATER, FUNCTIONAL_GIRL_TOILET, COMPUTER_AVAILABLE, and LIBRARY_OR_READING_CORNER_OR_BOOK_BANK store binary values (1 for available, 0 for not available) or booleans indicating the presence or absence of these key facilities.

Health and Safety Features: COMPLETE_MEDICAL_CHECKUP, INCINERATOR, and HANDWASH are also represented as binary/boolean values, showing whether these health-related features are available.

Numeric Counts: Columns like TOTAL_NUMBER_OF_SCHOOLS store integers, representing the total number of schools in each district.

This table supports a granular analysis of school infrastructure, enabling insights into how resources and facilities are distributed across different regions and school management types.
"""
METADATA_QUERY = f"SELECT COLUMN_NAME, DATA_TYPE FROM {SCHEMA_PATH.split('.')[0]}.INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = '{SCHEMA_PATH.split('.')[1]}' AND TABLE_NAME = 'INFRASTRUCTURE_AVAILABILITY_AND_FACILITIES';"

GEN_SQL = """
You will be acting as an AI Snowflake SQL Expert named Frosty.
Your goal is to give correct, executable sql query to users.
You will be replying to users who will be confused if you don't respond in the character of Frosty.
You are given one table, the table name is in <tableName> tag, the columns are in <columns> tag.
The user will ask questions, for each question you should respond and include a sql query based on the question and the table. 

{context}

Here are 6 critical rules for the interaction you must abide:
<rules>
1. You MUST MUST wrap the generated sql code within ``` sql code markdown in this format e.g
```sql
(select 1) union (select 2)
```
2. If I don't tell you to find a limited set of results in the sql query or question, you MUST limit the number of responses to 10.
3. Text / string where clauses must be fuzzy match e.g ilike %keyword%
4. Make sure to generate a single snowflake sql code, not multiple. 
5. You should only use the table columns given in <columns>, and the table given in <tableName>, you MUST NOT hallucinate about the table names
6. DO NOT put numerical at the very front of sql variable.
</rules>

Don't forget to use "ilike %keyword%" for fuzzy match queries (especially for variable_name column)
and wrap the generated sql code with ``` sql code markdown in this format e.g:
```sql
(select 1) union (select 2)
```

For each question from the user, make sure to include a query in your response.

Now to get started, please briefly introduce yourself, describe the table at a high level, and share the available metrics in 2-3 sentences.
Then provide 3 example questions using bullet points.
"""
connector, cursor = connect_to_snowflake()
@st.cache_data(show_spinner="Loading context for SnowPulse...")
def get_table_context(table_name: str, table_description: str,  connector, cursor, metadata_query: str = None,):
    table = table_name.split(".")

    cursor.execute(f"""
        SELECT COLUMN_NAME, DATA_TYPE FROM {table[0].upper()}.INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = '{table[1].upper()}' AND TABLE_NAME = '{table[2].upper()}'
    """)
    columns_data = cursor.fetchall()

    columns = "\n".join(
        [f"- **{col[0]}**: {col[1]}" for col in columns_data]
    )
    
    context = f"""
Here is the table name <tableName> {'.'.join(table)} </tableName>

<tableDescription>{table_description}</tableDescription>

Here are the columns of the {'.'.join(table)}

<columns>\n\n{columns}\n\n</columns>
    """
    
    if metadata_query:
        cursor.execute(metadata_query)
        metadata_data = cursor.fetchall()
        
        metadata = "\n".join(
            [f"- **{row[0]}**: {row[1]}" for row in metadata_data]
        )
        context = context + f"\n\nAvailable variables by VARIABLE_NAME:\n\n{metadata}"
    
    connector.close()
    
    return context

def get_system_prompt(CONN, CUR):
    table_context = get_table_context(
        table_name=QUALIFIED_TABLE_NAME,
        table_description=TABLE_DESCRIPTION,
        connector=CONN,
        cursor=CUR,
        metadata_query=METADATA_QUERY
    )
    return GEN_SQL.format(context=table_context)
