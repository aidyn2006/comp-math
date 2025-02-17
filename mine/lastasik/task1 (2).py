import numpy as np

# Full time and velocity data (excluding t=20 which is unknown)
time_full = np.array([0, 5, 10, 15])  # Time in seconds (excluding t=20)
velocity_full = np.array([3, 14, 69, 228])  # Velocity at known times

# Step size (h) between each time point
h = 5  # Time difference is 5 seconds


# Forward differences function to calculate first and second differences
def forward_differences(x, y):
    n = len(x)
    first_diff = np.zeros(n - 1)
    second_diff = np.zeros(n - 2)

    # First forward difference
    for i in range(n - 1):
        first_diff[i] = y[i + 1] - y[i]

    # Second forward difference
    for i in range(n - 2):
        second_diff[i] = first_diff[i + 1] - first_diff[i]

    return first_diff, second_diff


# Calculate the first and second forward differences for velocity data
first_diff, second_diff = forward_differences(time_full, velocity_full)

# Now, we can extrapolate to find the velocity at t=20
# Extrapolate using first and second differences
v_20 = velocity_full[-1] + first_diff[-1] + second_diff[-1]  # Using forward differences for prediction

print("Extrapolated velocity at t=20:", v_20)
