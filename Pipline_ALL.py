import requests
import pandas as pd
import time
import pyodbc
import os


headers = {
	"X-RapidAPI-Key": "9338e1242fmsh920dc963a304d2cp11bab5jsnd42b392c1c00",
	"X-RapidAPI-Host": "cricbuzz-cricket.p.rapidapi.com"
}



def get_data():
    
    global url
    global headers
    global querystring
    
    formats = [ 'test','odi', 't20']
    ranking = ['batsmen' , 'bowlers' ,'allrounders']
    
    for r in ranking:
        url= f'https://cricbuzz-cricket.p.rapidapi.com/stats/v1/rankings/{r}'
        
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
    
    ## pyodbc connection
    server_name = 'localhost\SQLEXPRESS'
    db_name = 'API_data'
    
   
        
    cnxn = pyodbc.connect(Driver='{SQL Server}',
                          Server=server_name,
                          Database=db_name,
                          Trusted_Connection='yes',
                          autocommit= True)
                         
    cursor = cnxn.cursor()
        
    sql_loc = 'D:\\Users\\nikalas\\Documents\\Cricket data pipline\\SQL Scripts'
        
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
    
    
    
    
    
    