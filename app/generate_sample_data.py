import numpy as np

# Example input data (5 samples, 3 features)
np.random.seed(22)  # For reproducibility

width = 10
height = 10
weight = np.random.random((width, height, 3))

input_data = np.random.random((10, 3))

# Save the array to a .npy file
np.save("data/input_data.npy", input_data)
print("Data saved to input_data.npy")
