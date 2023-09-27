import submitit
from transfer_plots import get_transfer_plot
import os

log_folder = f"/checkpoint/alupu/explainable_policies/plotting/submitit_logs/"
if not os.path.exists(log_folder):
    os.makedirs(log_folder)

executor = submitit.AutoExecutor(folder=log_folder)
executor.update_parameters(
    slurm_partition="learnlab",
    gpus_per_node=1,
    cpus_per_task=10,
    timeout_min=60,
    slurm_job_name="violin",
)

envs = ["hopper", "walker2d", "reacher", "inverted_double_pendulum",
        "ant", "halfcheetah", "humanoid", "humanoidstandup"]

jobs = []
with executor.batch():
    for env in envs:
        job = executor.submit(get_transfer_plot, env)
        jobs.append(job)

print(f"Launched {len(jobs)} jobs. Plotting on cluster, how exciting!")



