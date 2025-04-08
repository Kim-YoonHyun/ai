# DB
import pandas as pd
import pymysql
from sqlalchemy import create_engine


# def get_info():
#     db_host = '192.168.0.85'
#     db_port = 3306
#     db_user = 'theimc'
#     db_passward = 'theimc#10!'
#     db_name = 'BUSMONITORING'
#     charset = 'utf8mb4'
#     if_exists = 'append'
#     autocommit = True
#     return (db_host, db_port, db_user, db_passward, db_name, charset, if_exists, autocommit)


# def get_info_p5000():
#     db_host = "59.25.131.135"
#     db_port = 3306
#     db_user = "ai_m"
#     db_password = "temp"
#     db_name = "bus"
#     charset = 'utf8mb4'
#     if_exists = 'append'
#     autocommit = True
#     return (db_host, db_port, db_user, db_password, db_name, charset, if_exists ,autocommit)


def db_connect(host, user, port, passward, name, charset='utf8mb4', if_exists='append', autocommit=True):
    # db_host = db_info_dict['host']
    # db_port = db_info_dict['port']
    # db_user = db_info_dict['user']
    # db_passward = db_info_dict['passward']
    # db_name = db_info_dict['name']
    # charset = db_info_dict['charset']
    # if_exists = db_info_dict['if_exists']
    # autocommit = db_info_dict['autocommit']
    conn = pymysql.connect(
        host=host, 
        user=user, 
        port=port, 
        password=passward, 
        db=name,
        charset=charset,
        autocommit=autocommit
    )
    return conn


def select_db(conn, query, where=None):
    cursor = conn.cursor()
    query = query
    cursor.execute(query)
    info = cursor.fetchall()
    cursor.close()
    return info


def delete_db(db_info_dict, table, where=None):
    cursor = db_connect(db_info_dict).cursor()
    if where:
        query = f"""
            DELETE FROM {table}
            where {where}
        """
    else:
        query = f"""
            DELETE FROM {table}
        """
    cursor.execute(query)
    cursor.close()


def update_db(db_info_dict, table, set_, where):
    cursor = db_connect(db_info_dict).cursor()
    query = f"""
        update {table}
        set {set_}
        where {where}
    """
    cursor.execute(query)
    cursor.close()



def pd2db(db_host, db_port, db_user, db_passward, 
          db_name, charset, if_exists, autocommit,
          df, table, encoding='utf-8-sig', index=False):
    
    url = f"mysql+pymysql://{db_user}:{db_passward}@{db_host}:{db_port}/{db_name}?charset={charset}"
    # engine = create_engine(url, encoding=encoding)
    engine = create_engine(url)
    conn = engine.connect()
    df.to_sql(name=table, con=engine, if_exists=if_exists, index=index)
    conn.close()
    

# def pd2db(db_info_dict, df, table, encoding='utf-8-sig', index=False):
#     db_host = db_info_dict['host']
#     db_port = db_info_dict['port']
#     db_user = db_info_dict['user']
#     db_passward = db_info_dict['passward']
#     db_name = db_info_dict['name']
#     charset = db_info_dict['charset']
#     if_exists = db_info_dict['if_exists']
#     autocommit = db_info_dict['autocommit']
#     url = f"mysql+pymysql://{db_user}:{db_passward}@{db_host}:{db_port}/{db_name}?charset={charset}"
#     # engine = create_engine(url, encoding=encoding)
#     engine = create_engine(url)
#     conn = engine.connect()
#     df.to_sql(name=table, con=engine, if_exists=if_exists, index=index)
#     conn.close()
    

"""
def main():
    db_host = '211.195.9.226'
    db_port = 3306
    db_user = 'root'
    db_passward = 'theimc#10!'
    db_name = 'flagship'
    charset = 'utf8mb4'
    table = 'test'
    if_exists = 'append'
    autocommit = True
    
    df = pd.read_csv('D:/python/project/temp/example.csv', encoding='utf-8-sig')
    
    pd2db(
        db_user=db_user, 
        db_passward=db_passward, 
        db_host=db_host, 
        db_port=db_port, 
        db_name=db_name, 
        charset=charset, 
        df=df, 
        table=table, 
        if_exists=if_exists, 
        encoding='utf-8-sig', 
        index=False
    )
    

if __name__ == '__main__':
    main()

"""









            
