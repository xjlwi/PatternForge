import numpy as np

# Example input data (5 samples, 3 features)

width = 100
height=3
input_data = np.random.random((width, height, 3))

# input_data = np.array(
#     [
#         [0.1, 0.5, 0.9],
#         [0.7, 0.2, 0.3],
#         [0.4, 0.8, 0.6],
#         # [0.9, 0.1, 0.5],
#         # [0.3, 0.7, 0.2],
#     ]
# )

# Save the array to a .npy file
np.save("data/input_data.npy", input_data)
print("Data saved to input_data.npy")
