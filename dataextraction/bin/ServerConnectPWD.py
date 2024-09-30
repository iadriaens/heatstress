# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 2021

@author: u0141520
"""

from sshtunnel import SSHTunnelForwarder
from sqlalchemy import create_engine
from sqlalchemy import inspect
from sqlalchemy_utils import database_exists, create_database
import pandas as pd
# import getpass


class LT_connect(object):

    '''
    The class 'LT_connect' can be used to connect to the LT postgres database 
    via a ssh connection.\n

    Functions:
    ----------

    tables(self, db, schema, psql_user, psql_pass)\n
    print_columns(self, db, table_name, psql_user, psql_pass)\n
    query(self, db, query, psql_user, psql_pass)\n
    execute(self, db, query, psql_user, psql_pass)\n
    insert(self, db, query, data, psql_user, psql_pass)\n
    ret_con(self, db, psql_user, psql_pass)\n
    create_db(self, db, psql_user, psql_pass)

    Parameters
    ----------

    db : name of a postgres database that should be accessed \n
    p_host : address of the database of the system 
    (usually localhost - 127.0.0.1) \n
    p_port : port for postgresql (usually 5432) \n
    ssh : if a ssh connection is necessary insert 'True' \n
    ssh_user : account name of the ssh user \n
    ssh_host : ip address of the server to which to connect \n
    ssh_pkey : filepath to the ssh key for faster access \n
    psql_user : account name of the postgres user \n
    psql_pass : password for the postgres account \n

    Return
    ------

    None


    '''

    def __init__(self, pgres_host, pgres_port, db, ssh, ssh_user, ssh_host, ssh_pwd, psql_user, psql_pass):
        '''
        __init__(self, pgres_host, pgres_port, db, ssh, ssh_user, ssh_host, ssh_pkey, psql_user, psql_pass): 
        -----------------------------------------------
        defines global class parameters for ssh connection\n

        Parameters
        ----------
        db : name of a postgres database that should be accessed \n
        pgres_host : address of the database of the system 
        (usually localhost - 127.0.0.1) \n
        pgres_port : port for postgresql (usually 5432) \n
        ssh : if a ssh connection is necessary insert 'True' \n
        ssh_user : account name of the ssh user \n
        ssh_host : ip address of the server to which to connect \n
        ssh_pkey : filepath to the ssh key for faster access \n
        psql_user : account name of the postgres user \n
        psql_pass : password for the postgres account \n

        Returns:
        --------
        None
        '''
        # SSH Tunnel Variables
        self.pgres_host = pgres_host
        self.pgres_port = pgres_port
        self.psql_user = psql_user
        self.psql_pass = psql_pass

        if ssh == True:
            self.server = SSHTunnelForwarder(
                (ssh_host, 53001),
                ssh_username=ssh_user,
                ssh_password=ssh_pwd,
                remote_bind_address=(pgres_host, pgres_port),
            )
            server = self.server
            server.start()  # start ssh server
            self.local_port = server.local_bind_port
            print(f'Server connected via SSH ...')
        elif ssh == False:
            self.local_port = pgres_port

    def tables(self, db, schema, psql_user, psql_pass):
        '''
        tables(self, db, schema, psql_user, psql_pass): 
        -----------------------------------------------
        returns all table names in a given 'schema' of a database 'db'\n

        Parameters:
        ----------

        db : name of a postgres database that should be accessed \n
        schema : name of the schema that should be analyzed\n
        psql_user : account name of the postgres user \n
        psql_pass : password for the postgres account \n

        Returns:
        --------
        tables_df (pandas dataframe of table names)

        '''

        # create an engine to connect to the postgreSQL database, documentation
        # of functions can be found at https://docs.sqlalchemy.org/en/14/

        engine = create_engine(
            f'postgresql://{psql_user}:{psql_pass}@{self.pgres_host}:{self.local_port}/{db}')
        inspector = inspect(engine)
        tables = inspector.get_table_names(schema=schema)
        self.tables_df = pd.DataFrame(tables, columns=['table name'])
        engine.dispose()
        return self.tables_df

    def print_columns(self, db, table_name, psql_user, psql_pass):
        '''
        print_columns(self, db, table_name, psql_user, psql_pass)
        -----------------------------------------------
        returns all table names in a given 'schema' of a database 'db'\n

        Parameters:
        ----------

        db : name of a postgres database that should be accessed \n
        table_name : name of the table for which the columns schould be checked \n
        psql_user : account name of the postgres user \n
        psql_pass : password for the postgres account \n

        Returns:
        --------
        tables_df (pandas dataframe of column names)

        '''

        # create an engine to connect to the postgreSQL database, documentation
        # of functions can be found at https://docs.sqlalchemy.org/en/14/

        engine = create_engine(
            f'postgresql://{psql_user}:{psql_pass}@{self.pgres_host}:{self.local_port}/{db}')
        if ' ' in table_name:
            if '"' in table_name:
                pass
            else:
                table_name = "'" + table_name + "'"
        query = """
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = %s
        ;
        """ % table_name
        self.table_df = pd.read_sql(query, engine)
        engine.dispose()
        return self.table_df

    def query(self, db, query, psql_user, psql_pass):
        '''
        query(self, db, query, psql_user, psql_pass)
        -----------------------------------------------
        executes a postgreSQL query in the database 'db' (return = true)\n

        Parameters:
        ----------

        db : name of a postgres database that should be accessed \n
        query : insert char string of postgreSQL code that should be queried \n
        psql_user : account name of the postgres user \n
        psql_pass : password for the postgres account \n

        Returns:
        --------
        query_df (pandas dataframe of query result)

        '''
        # create an engine to connect to the postgreSQL database, documentation
        # of functions can be found at https://docs.sqlalchemy.org/en/14/
        engine = create_engine(
            f'postgresql://{psql_user}:{psql_pass}@{self.pgres_host}:{self.local_port}/{db}')
        self.query_df = pd.read_sql(query, engine)
        engine.dispose()
        return self.query_df

    def execute(self, db, query, psql_user, psql_pass):
        '''
        execute(self, db, query, psql_user, psql_pass)
        -----------------------------------------------
        executes a postgreSQL query in the database 'db' (return = false)\n

        Parameters:
        ----------

        db : name of a postgres database that should be accessed \n
        query : insert char string of postgreSQL code that should be queried \n
        psql_user : account name of the postgres user \n
        psql_pass : password for the postgres account \n

        Returns:
        --------
        None

        '''
        # create an engine to connect to the postgreSQL database, documentation
        # of functions can be found at https://docs.sqlalchemy.org/en/14/
        engine = create_engine(
            f'postgresql://{psql_user}:{psql_pass}@{self.pgres_host}:{self.local_port}/{db}')
        engine.execute(query)
        engine.dispose()

    def insert(self, db, query, data, psql_user, psql_pass):
        '''
        insert(self, db, query, data, psql_user, psql_pass)
        -----------------------------------------------
        executes a postgreSQL query in the database 'db' (return = false),
        used to insert data with parameter data, use '%(name)s' in the query text
        and a dictionary ({name : value}) for data \n

        Parameters:
        ----------

        db : name of a postgres database that should be accessed \n
        query : insert char string of postgreSQL code that should be queried \n
        data : dictionary of data that should be used in the query \n
        psql_user : account name of the postgres user \n
        psql_pass : password for the postgres account \n

        Returns:
        --------
        None

        '''
        # create an engine to connect to the postgreSQL database, documentation
        # of functions can be found at https://docs.sqlalchemy.org/en/14/
        engine = create_engine(
            f'postgresql://{psql_user}:{psql_pass}@{self.pgres_host}:{self.local_port}/{db}')
        engine.execute(query, data)
        engine.dispose()

    def ret_con(self, db, psql_user, psql_pass):
        '''
        ret_con(self, db, psql_user, psql_pass)
        -----------------------------------------------
        returns the engine to connect to the database 'db'\n

        Parameters:
        ----------

        db : name of a postgres database that should be accessed \n
        psql_user : account name of the postgres user \n
        psql_pass : password for the postgres account \n

        Returns:
        --------
        engine

        '''
        # create an engine to connect to the postgreSQL database, documentation
        # of functions can be found at https://docs.sqlalchemy.org/en/14/
        engine = create_engine(
            f'postgresql://{psql_user}:{psql_pass}@{self.pgres_host}:{self.local_port}/{db}')
        return engine

    def create_db(self, db, psql_user, psql_pass):
        '''
        create_db(self, db, psql_user, psql_pass)
        -----------------------------------------------
        creates the database 'db'\n

        Parameters:
        ----------

        db : name of a postgres database that should be accessed \n
        psql_user : account name of the postgres user \n
        psql_pass : password for the postgres account \n

        Returns:
        --------
        None

        '''
        # create an engine to connect to the postgreSQL database, documentation
        # of functions can be found at https://docs.sqlalchemy.org/en/14/
        engine = create_engine(
            f'postgresql://{psql_user}:{psql_pass}@{self.pgres_host}:{self.local_port}/{db}')
        if not database_exists(engine.url):
            create_database(engine.url)
        else:
            print('A database with the name "' + db + '" already exists')
