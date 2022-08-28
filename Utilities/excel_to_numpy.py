import numpy as np
import pandas as pd

# Input Path for the Excel files
path = input("Enter the path of the input Excel File:")

print("Reading Excel File...")

# Data Orginally in the form of Excel Sheets with each sheet as a subject, with rows as TRs and columns as Voxels of the ROI
data_xl = pd.ExcelFile(path, engine="openpyxl")

# Initialization of Tensors of Dimensions (TR, Voxels, Subjects)
first_sheet = data_xl.parse(data_xl.sheet_names[0], header=None)
data = np.zeros((first_sheet.shape[0], first_sheet.shape[1], len(data_xl.sheet_names)))

# Print Details
print("Number of TRs: ", data.shape[0])
print("Number of Voxels: ", data.shape[1])
print("Number of Subjects: ", data.shape[2])

print("Converting Excel File to Numpy Tensor...")

# Converts the Excel Sheets corresponding to the ROIs to Tensor
for i in range(len(data_xl.sheet_names)): # Takes long duration to complete, prefer saving the output tensor
  data[:,:,i] = data_xl.parse(data_xl.sheet_names[i], header=None).to_numpy()

# Saving the Tensor data
output_path = input("Enter the path(including the output file name with extension):")

np.save(output_path, data)
