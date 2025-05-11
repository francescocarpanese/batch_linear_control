To efficiently leveraging GPU parallelism when performing parameter scan of closed loop simulation of LTI (linear time-invariant) systems, one option is, at every time step, to compute the closed loop system response for all parameter sets in parallel. This can be done using a batch processing approach, where the system response is computed for all parameter sets at once, rather than iterating through each parameter set one at a time.

In this prototype project, jax is used for vectorisation and GPU compilation. 

To test the package.

- Install `uv` package manager
- Pull the repo
- Create venv and dependencies `uv sync`
- Activate virtual evn `source .venv/bin/activate`
- Run example `python src/batch_linear_control/linear_control.py`