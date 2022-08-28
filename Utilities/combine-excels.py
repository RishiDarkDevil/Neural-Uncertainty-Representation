# Code to combine multiple excel files, into different sheets of one excel file

# pip install openpyxl
# pip install xlrd
import numpy as np
import pandas as pd
import os

# Path to the folder containing only the excel files to be combined
input_path = input("Enter directory path containing all the excel files(Add a / at the end): ")
file_names = list(os.listdir(input_path))

print("Files that are being combined:")
for name in file_names:
	print(name)

# Loading the excel files
excel_files = list()
if file_names[0].endswith(".xls"):
	excel_files = [pd.read_excel(input_path+name, engine=None) for name in file_names]
elif file_names[0].endswith(".xlsx"):
	excel_files = [pd.read_excel(input_path+name, engine='openpyxl') for name in file_names]
else:
	print("Bad Extension(.xls or .xlsx allowed)")
	exit()

output_path = input("Enter the path to the output excel file(add .xlsx extension): ")

# Combine and generate output file
for file, name in zip(excel_files, file_names):
	if os.path.isfile(output_path):		
		with pd.ExcelWriter(output_path, mode='a') as writer:
			file.to_excel(writer, sheet_name= name.replace(".xls", "").replace(".xlsx", ""))
	else:
		with pd.ExcelWriter(output_path) as writer:
			file.to_excel(writer, sheet_name= name.replace(".xls", "").replace(".xlsx", ""))