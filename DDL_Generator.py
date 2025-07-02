# -*- coding: utf-8 -*-


import pandas as pd
import os


def ddl_generator():
        
    file_path = str(input('plaese give the input file location : - '))
    ddl_loc = str(input('plaese give the DDL output file location : - '))
    
    
    # Load Excel file into a pandas DataFrame
    
    if len(os.listdir(file_path)) >= 0:
            
        
        for i in os.listdir(file_path):
            
            
            if "csv" in i:
                df = pd.read_csv(file_path + f'\\{i}')
        
            elif "xlsx" in i:
                df = pd.read_excel(file_path + f'\\{i}')
                
            elif "json" in i:
                df = pd.read_json(file_path + f'\\{i}')
                
            else:
                print('file format out of scope, Please download the file in desired (csv or xlsx or json) format and check the given file location')
                continue
            
            # Generate DDL statement
            table_name = str(i.split('.')[0])
            print(table_name)
            
            ddl_statement = f"CREATE OR REPLACE TABLE {table_name} (\n"
            
            for column_name, data_type in zip(df.columns, df.dtypes):
                
                if "int" in str(data_type):
                    ddl_statement += f"    {column_name} INT,\n"
                
                elif "float" in str(data_type):
                    ddl_statement += f"    {column_name} FLOAT,\n"
                
                elif "object" in str(data_type):
                    
                    result = len(str(df[column_name].max()))
                    rounded_result = (result+100) // 100*100 # Round up to the nearest multiple of 100
                    ddl_statement += f"    {column_name} VARCHAR({rounded_result}),\n"
                
                elif "datetime" in str(data_type):
                    ddl_statement += f"    {column_name} DATETIME,\n"
                    
                elif "bool" in str(data_type):
                    ddl_statement += f"    {column_name} BOOLEAN,\n"
            
            ddl_statement = ddl_statement.rstrip(',\n') + "\n);"
            
            print('DDL statement is ready, creating the output sql')
            
            
            with open(f'{ddl_loc}\\{table_name}_DDL.sql','w') as ddl:
                ddl.write(ddl_statement)
            
            print(f'DDL for {table_name} is saved at {ddl_loc}')
        
    else:
        print(f'No file found at {file_path} location')

ddl_generator()


