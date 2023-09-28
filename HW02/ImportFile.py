import numpy as np

# Text file data converted to integer data type
File_data = np.loadtxt('HomeworkData/Druns.txt', dtype=float)
print(File_data.shape)
