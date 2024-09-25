import snowflake.connector
import os

def connect_to_snowflake():
    connector = snowflake.connector.connect(
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        warehouse='COMPUTE_WH',
        database='TEST',
        schema='PUBLIC',
        role='ACCOUNTADMIN'
    )
    cursor = connector.cursor()
    return connector, cursor