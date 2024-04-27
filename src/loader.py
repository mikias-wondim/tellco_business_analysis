import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, Float, String
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

def create_table(engine, table_name, df):
    """
    Create a table in a PostgreSQL database based on the columns of a DataFrame.

    Args:
        engine (sqlalchemy.engine.base.Engine): SQLAlchemy engine object representing the database connection.
        table_name (str): Name of the table to be created.
        df (pandas.DataFrame): DataFrame containing the columns to be used for creating the table.

    Returns:
        bool: True if table creation is successful, False otherwise.
    """
    try:
        # Create a MetaData object
        metadata = MetaData(bind=engine)

        # Get column names and data types from the DataFrame
        columns = [(col, String) for col in df.columns]

        # Define the table structure
        table = Table(
            table_name,
            metadata,
            Column('id', Integer, primary_key=True),
            *[
                Column(name, dtype) for name, dtype in columns
            ]
        )

        # Check if the table already exists
        if not table.exists():
            # Create the table in the database
            metadata.create_all(engine)
            print(f"Table '{table_name}' created successfully.")
            return True
        else:
            print(f"Table '{table_name}' already exists.")
            return False

    except SQLAlchemyError as e:
        print(f"Error creating table '{table_name}': {e}")
        return False


def drop_table(engine, table_name):
    """
    Drop a table with a specific name from the PostgreSQL database.

    Args:
        engine (sqlalchemy.engine.base.Engine): SQLAlchemy engine object representing the database connection.
        table_name (str): Name of the table to be dropped.

    Returns:
        bool: True if table dropping is successful, False otherwise.
    """
    try:
        # Create a MetaData object
        metadata = MetaData(bind=engine)

        # Reflect the existing table
        existing_table = Table(table_name, metadata, autoload=True, autoload_with=engine)

        # Drop the table from the database
        existing_table.drop(engine)

        print(f"Table '{table_name}' dropped successfully.")
        return True

    except SQLAlchemyError as e:
        print(f"Error dropping table '{table_name}': {e}")
        return False

def save_to_db(df, engine, table_name):
    """
    Save a DataFrame to a PostgreSQL table.

    Args:
        df (pandas.DataFrame): DataFrame to be saved to the database.
        engine (sqlalchemy.engine.base.Engine): SQLAlchemy engine object representing the database connection.
        table_name (str): Name of the table in the database.

    Returns:
        bool: True if data is successfully saved to the table, False otherwise.
    """
    try:
        # Save the DataFrame to the PostgreSQL database
        df.to_sql(table_name, engine, index=False, if_exists='replace')
        print(f"Data saved to table '{table_name}' successfully.")
        return True

    except SQLAlchemyError as e:
        print(f"Error saving data to table '{table_name}': {e}")
        return False

def alter_column_type(table_name, column_name, new_data_type, connection_params):
  # Database connection details (replace with your credentials)
  DATABASE_URI = f"postgresql://{connection_params['user']}:{connection_params['password']}@{connection_params['host']}:{connection_params['port']}/{connection_params['dbname']}"

  # Connect to PostgreSQL database
  conn = psycopg2.connect(DATABASE_URI)
  cur = conn.cursor()

  try:
    # Alter column data type with backticks around column name (ensure single set)
    cur.execute(f"""
    ALTER TABLE {table_name}
    ALTER COLUMN "{column_name}" SET DATA TYPE {new_data_type}
    USING "{column_name}"::{new_data_type.lower()};

    """)

    conn.commit()
    print(f"Column '{column_name}' in table '{table_name}' altered to data type '{new_data_type}'.")

  except (Exception, psycopg2.Error) as error:
    print(f"Error altering column: {error}")
    conn.rollback()  # Rollback on error

  finally:
    if conn:
      cur.close()
      conn.close()