'''
pip install influxdb
pip install pymysql
pip install SQLAlchemy
'''

import sys
import os

import requests
from influxdb import InfluxDBClient 
import pymysql
from sqlalchemy import create_engine
import pandas as pd


def mariadb_query(host, port, database, user, password, query, 
                  charset='utf8mb4', if_exists='append', autocommit=True, mode='get'):
    conn = pymysql.connect(
        host=host, 
        port=port, 
        user=user, 
        password=password, 
        db=database,
        charset=charset,
        autocommit=autocommit
    )
    
    cursor = conn.cursor()
    cursor.execute(query)
    
    if mode == 'get':
        # column
        column_info = cursor.description
        column_name_list = [column[0] for column in column_info]
        
        # data
        data = cursor.fetchall()
        cursor.close()
        whole_df = pd.DataFrame(data, columns=column_name_list)
        return whole_df
    elif mode == 'update':
        cursor.close()


def influxdb_query(host, port, version, database, query,
                   user=None, password=None,  # ver 1
                   token=None # ver 2
                   ): 
    
    if version == 1:
        client = InfluxDBClient(
            host=host, 
            port=port, 
            username=user, 
            password=password,
            database=database, 
            timeout=30,
            retries=1,
        )
        result = client.query(query)
        
    if version == 2:
        headers = {}
        headers['Authorization'] = f'Token {token}'
        
        client = InfluxDBClient(
            host=host, 
            port=port, 
            username=None, 
            password=None,
            database=database, 
            timeout=30,
            retries=1,
            headers=headers
        )
        result = client.query(query)
    
    return result



def query2df(result, query_type):
    if len(result) == 0:
        return pd.DataFrame()
    if query_type == 'influxdb2':
        column_name_list = vars(result)['_raw']['series'][0]['columns']
        value_list = vars(result)['_raw']['series'][0]['values']
        whole_df = pd.DataFrame(value_list, columns=column_name_list)
    return whole_df
   
    
    
def influxdb_request(host, port, database, query, token):
    # InfluxDB settings
    # influxdb_url = f""
    # database = "your_database"
    # query = "SELECT * FROM your_measurement LIMIT 10"  # Replace with your actual query
    # token = "your_auth_token"  # Replace with your actual token
    
    url = f"http://{host}:{port}/query?db={database}&q={query}"
    headers = {"Authorization": f"Token {token}"}
    
    # Send the GET request to the InfluxDB API with headers
    response = requests.get(url, headers=headers)
    
    # Check if the request was successful (status code 200)
    dataa_list = 0
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()
        
        # Extract and print the results
        results = data.get("results", [])
        for result in results:
            series = result.get("series", [])
            for s in series:
                print(s.get("values", []))
                sys.exit()
    else:
        print(f"Error: {response.status_code} - {response.text}")
        
    return response
    

def df2db(db_type, df, host, port, password, user, db_name, table, charset='utf8mb4', if_exists='append', index=False):
    if db_type == 'maria':
        url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{db_name}?charset={charset}"
        engine = create_engine(url)
        conn = engine.connect()
        df.to_sql(name=table, con=engine, if_exists=if_exists, index=index)
        conn.close() 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
