import submitit
from distill_brax import train_from_arg_string
import os

log_folder = f"/checkpoint/alupu/explainable_policies/brax/submitit_logs/"
if not os.path.exists(log_folder):
    os.mkdir(log_folder)
executor = submitit.AutoExecutor(folder=log_folder)
executor.update_parameters(
    slurm_partition="learnfair",
    gpus_per_node=8,
    cpus_per_task=80,
    timeout_min=1080,
)

jobs = []
with executor.batch():
    for D in [4]:
        for env in ["reacher", "pusher"]:
            for epochs in [200]:
                for seed in [0, 1, 2]:
                    print(env, D, epochs, seed)
                    folder = f"/private/home/alupu/explainable-policies/results/brax_individual_runs/{env}/D{D}_E{epochs}/seed{seed}/"
                    if not os.path.exists(folder):
                        os.mkdir(folder)
                    argstring = \
                        f"--env {env} " \
                        f"--epochs {epochs} " \
                        f"--dataset_size {D} " \
                        f"--generations 1000 " \
                        f"--popsize 2048 " \
                        f"--rollouts 2 " \
                        f"--eval_envs 4 " \
                        f"--width 512 " \
                        f"--seed {seed} " \
                        f"--folder {folder} " \
                        f"--normalize_obs 1 "

                    job = executor.submit(train_from_arg_string, argstring)
                    jobs.append(job)

