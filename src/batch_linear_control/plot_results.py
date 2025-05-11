import h5py
import numpy as np
import matplotlib.pyplot as plt

file_path = "/home/cisko90/batch_control/sim_data/simulation_results.h5"

# Load the HDF5 file
with h5py.File(file_path, "r") as h5f:
    # Read the dataset
    all_outputs_debug = h5f["all_outputs_debug"][:]
    plant_state_dim = h5f.attrs["plant_state_dim"]
    controller_state_dim = h5f.attrs["controller_state_dim"]
    output_dim = h5f.attrs["output_dim"]
    batch_size = h5f.attrs["batch_size"]
    simulation_steps = h5f.attrs["simulation_steps"]


# Plot all batches in the same plot
plt.figure(figsize=(10, 6))
for batch_id in range(batch_size):
    output_to_plot = all_outputs_debug[batch_id, :].squeeze()
    plt.plot(output_to_plot, label=f"Batch {batch_id}")

plt.xlabel("Time Step")
plt.ylabel("Output")
plt.title("Output Over Time for All Batches")
plt.legend()
plt.show()