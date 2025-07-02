import requests
import pandas as pd
import time
#import pyodbc
import os
import snowflake.connector
import boto3
import configparser
import sys


sys.path.insert(0,"D:\\Users\\nikalas\\Documents\\Cricket data pipline")

conf = configparser.ConfigParser()



confg = conf.read('C:\\Users\\nikal\\config.ini',encoding='utf-8')

confg.get('s3')


print("Current working directory:", os.getcwd())
print("Script location:", os.path.abspath(('config.ini')))



def s3_upload():
    
    my_bucket = 'nikalas-sf'
    
    s3_client = boto3.client('s3')

    buckets = s3_client.list_buckets()
    
    for i in buckets['Buckets']:
        print(i['Name'])
    
    # Location of .csv file that needs to be uploaded
    csv_loc = r'D:\Users\nikalas\Documents\Cricket data pipline\out'
    
    for i in os.listdir(csv_loc):
        if i.endswith('.csv'):
            print(i)
            file_name = csv_loc + "\\" + i 
            s3_client.upload_file(Filename= file_name, Bucket= my_bucket, Key= i)
            




def get_data():
    
    global url
    global headers
    global querystring
    
    
    
    formats = [ 'test','odi', 't20']
    ranking = ['batsmen' , 'bowlers' ,'allrounders']
    
    for r in ranking:
        
        
        for i in formats:
            
            querystring = {'formatType':i}
            
            response = requests.get(url, headers=headers, params=querystring)
            
            if response.status_code == 200:
                
                print(response.status_code)
                
                df = pd.DataFrame(response.json()["rank"])
            
                df.to_csv(f'D:\\Users\\nikalas\\Documents\\Cricket data pipline\\out\\{r}_{i}.csv',index= False)
            
                time.sleep(1)
                
            else:
                print(response.status_code)
                break
    

    print(time.ctime())        
    response.close()
                    

                    
## This function loads data from .csv file to dbo tables on sql server        
def load_sql_tables():
    '''
   

    ## SF connection

    cursor=conn.cursor()
    
    
    
    sql_loc = r'D:\Users\nikalas\Documents\Cricket data pipline\SF_SQL Scripts'
    
        
    sql = []
        
    for i in os.listdir(sql_loc):
        if i.endswith('.sql'):
            print(i)
    
            with open(f'{sql_loc}\{i}','r') as f:
                sql.append(f.read())
               
    
    for i in sql:
        print(i)
        cursor.execute(i)
        time.sleep(5)
    
    cursor.close()          
    print('Data load is completed')                


  
    
get_data()
load_sql_tables() 
    
    
    
    
    
    