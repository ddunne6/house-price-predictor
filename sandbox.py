import numpy as np             # importing numpy library as np                     
pre_one_array = np.array([10, 20, 30, 40, 50])   # defining a 1D array
print(pre_one_array)                  # printing the array
norm = np.linalg.norm(pre_one_array)     # To find the norm of the array
print(norm)                        # Printing the value of the norm
normalized_array = pre_one_array/norm  # Formula used to perform array normalization
print(normalized_array)  