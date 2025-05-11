# Minimal example of steppig a linear controler system with a batch of controller at every time step.

import jax
import jax.numpy as jnp
from functools import partial
import h5py


# 1. System Dynamics (Discrete-Time)
def system_step(state, system_input, A, B, C, D):
    """
    Discrete-time state-space update.
    """
    next_state = A @ state + B @ system_input
    output = C @ state + D @ system_input
    return next_state, output


# Define controller stepping function
def controller_step(controller_state, error, A_c, B_c, C_c, D_c):
    """
    Discrete-time controller update.
    """
    # Compute next controller state and control output
    next_controller_state = A_c @ controller_state + B_c * error
    control_output = C_c @ controller_state + D_c * error
    return next_controller_state, control_output


# 3. Actuator Saturation
def saturate(u, u_min, u_max):
    """
    Applies actuator saturation limits.
    """
    return jnp.clip(u, u_min, u_max)

# 4. Batched Simulation Step
def sim_step(carry, t, A, B, C, D, u_min, u_max):
    """
    Performs one simulation step for an entire batch of systems in parallel.

    Args:
        carry: A tuple containing (plant_states, controller_states, controller_params).
        t: The current time step.
        A, B, C, D: System matrices (same for all in batch).
        u_min, u_max: Actuator limits.

    Returns:
        A tuple containing (next_plant_states, next_controller_states, controller_params), outputs.
    """
    plant_states, controller_states, controller_params = carry
    references_t = references[t] # Get reference for this time step

    A_c, B_c, C_c, D_c = controller_params

    # Vectorize controller update
    next_controller_states, control_outputs = jax.vmap(controller_step)(
        controller_states,
        references_t - (plant_states @ C.T),  # Error calculation
        A_c, B_c, C_c, D_c, # Controller matrices
    )

    # Apply saturation
    control_outputs_saturated = jax.vmap(
        saturate,
        in_axes=(0, None, None),
        )(control_outputs, u_min, u_max)

    # Vectorize system update
    next_plant_states, outputs = jax.vmap(
        system_step,
        in_axes=(0, 0, None, None, None, None),
        )(
        plant_states, control_outputs_saturated, A, B, C, D
    )

    return (next_plant_states, next_controller_states, controller_params), outputs


# 5. Simulation Setup
batch_size = 500  # Number of different controller parameter sets
simulation_steps = 100

# # Example system matrices (on GPU)
A = jnp.array([[0.1]], device=jax.devices('gpu')[0])  # (plant_state_dim, plant_state_dim)
B = jnp.array([[1.]], device=jax.devices('gpu')[0])  # (plant_state_dim, control_dim)
C = jnp.array([[1.]], device=jax.devices('gpu')[0])  # (output_dim, plant_state_dim)
D = jnp.array([[0.]], device=jax.devices('gpu')[0])  # (output_dim, control_dim)
u_min = jnp.array([-jnp.inf])  # Actuator lower limit
u_max = jnp.array([jnp.inf])  # Actuator upper limit

# # Definition of multiple controllers
# A_c1 = jnp.array([[0.]]) 
# B_c1 = jnp.array([[0.]])          
# C_c1 = jnp.array([[0.]])             
# D_c1 = jnp.array([[1.]]) 

# A_c2 = jnp.array([[0.]]) 
# B_c2 = jnp.array([[0.]])          
# C_c2 = jnp.array([[0.]])             
# D_c2 = jnp.array([[0.1]])

# controller_params = (
#     jnp.array([A_c1, A_c2]),  # Shape: (batch_size, state_dim, state_dim)
#     jnp.array([B_c1, B_c2]),  # Shape: (batch_size, state_dim, 1)
#     jnp.array([C_c1, C_c2]),  # Shape: (batch_size, output_dim, state_dim)
#     jnp.array([D_c1, D_c2]),  # Shape: (batch_size, output_dim, 1)
# )

# Allocate controller matrices with scan of proportional gain
A_c_arr = jnp.zeros((batch_size, 1, 1))
B_c_arr = jnp.zeros((batch_size, 1, 1))
C_c_arr = jnp.zeros((batch_size, 1, 1))
D_c_arr = jnp.zeros((batch_size, 1, 1))
p_gain = jnp.linspace(0.1, 1.0, num=batch_size)  # Proportional gain for each controller

for i in range(batch_size):
    D_c_arr = D_c_arr.at[i].set(jnp.array([[p_gain[i]]]))  # Set proportional gain for each controller

controller_params = (
    A_c_arr,  # Shape: (batch_size, state_dim, state_dim)
    B_c_arr,  # Shape: (batch_size, state_dim, 1)
    C_c_arr,  # Shape: (batch_size, output_dim, state_dim)
    D_c_arr,  # Shape: (batch_size, output_dim, 1)
)

# Move to GPU
controller_params = jax.device_put(controller_params, device=jax.devices('gpu')[0])

plant_state_dim = A.shape # Get state dimension from A
controller_state_dim = (1,1) #Get controller state dimension

# Example initial states (on GPU)
initial_plant_states = jnp.zeros((batch_size,) + (plant_state_dim), device=jax.devices('gpu')[0])
initial_controller_states = jnp.zeros((batch_size,) + controller_state_dim, device=jax.devices('gpu')[0])

# Generate a reference of zeros
references = jnp.ones((simulation_steps, batch_size) + C.shape, device=jax.devices('gpu')[0])

# 6. Run Simulation
carry_init = (initial_plant_states, initial_controller_states, controller_params)
# Disable gradient tracking. Not needed.
carry_init = jax.lax.stop_gradient(carry_init)
time_steps = jax.lax.stop_gradient(jnp.arange(simulation_steps))

# JIT-compile the batched step function
compiled_step = jax.jit(
    partial(sim_step, A=A, B=B, C=C, D=D, u_min=u_min, u_max=u_max),
    )


# Use jax.lax.scan to iterate over time steps
(final_states, _, _), all_outputs = jax.lax.scan(
    compiled_step,
    carry_init,
    jnp.arange(simulation_steps),
)

# 7. Get Results (Copy only once)
final_outputs_cpu = all_outputs.block_until_ready()  # Transfer from GPU to CPU

# Flip the first and second dimensions, leaving others untouched
all_outputs_debug = jnp.swapaxes(final_outputs_cpu, 0, 1)

batch_id = 1
import matplotlib.pyplot as plt

# Extract the output for the specified batch
output_to_plot = all_outputs_debug[batch_id, :].squeeze()

# Plot the output over time
plt.figure(figsize=(8, 4))
plt.plot(output_to_plot, label=f"Batch {batch_id}")
plt.xlabel("Time Step")
plt.ylabel("Output")
plt.title("Output Over Time")
plt.legend()
plt.grid(True)
plt.show()


# Serialize the results into an HDF5 file
output_file = "/home/cisko90/batch_control/sim_data/simulation_results.h5"
with h5py.File(output_file, "w") as h5f:
    h5f.create_dataset("all_outputs_debug", data=all_outputs_debug)
    h5f.attrs["plant_state_dim"] = plant_state_dim
    h5f.attrs["controller_state_dim"] = controller_state_dim
    h5f.attrs["output_dim"] = C.shape[0]
    h5f.attrs["batch_size"] = batch_size
    h5f.attrs["simulation_steps"] = simulation_steps


print(f"Results saved to {output_file}")