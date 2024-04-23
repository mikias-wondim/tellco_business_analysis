import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
import psycopg2

def connect_to_database(username, password, host, port, db_name):
    """
    Connects to a PostgreSQL database using SQLAlchemy.

    Args:
        username (str): The username for the database.
        password (str): The password for the database user.
        host (str): The host where the database is running.
        port (str): The port number for the database.
        db_name (str): The name of the database to connect to.

    Returns:
        engine: SQLAlchemy engine object for the database connection.
    """
    try:
        # Create the database connection URL
        db_url = f'postgresql://{username}:{password}@{host}:{port}/{db_name}'

        # Create a SQLAlchemy engine
        engine = create_engine(db_url)

        print("Successfully connected to the database!")
        return engine

    except SQLAlchemyError as e:
        print(f"Error connecting to the database: {e}")
        return None

def import_data_to_dataframe(engine, table_name):
    """
    Imports data from a PostgreSQL table into a pandas DataFrame.

    Args:
        engine: SQLAlchemy engine object for the database connection.
        table_name (str): The name of the table to import data from.

    Returns:
        pd.DataFrame: Pandas DataFrame containing the imported data.
    """
    try:
        # Define the SQL query to select all data from the table
        sql_query = f'SELECT * FROM {table_name}'

        # Execute the query and import the results into a pandas DataFrame
        df = pd.read_sql(sql_query, engine)

        return df

    except SQLAlchemyError as e:
        print(f"Error importing data from the database: {e}")
        return None

