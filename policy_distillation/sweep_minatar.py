import submitit
from distill_minatar import train_from_arg_string
import os

log_folder = f"/checkpoint/alupu/explainable_policies/minatar/submitit_logs/"
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
    for D in [64]:
        for env in ["SpaceInvaders-MinAtar"]:
            for epochs in [200]:
                for seed in [0:
                    print(env, D, epochs, seed)
                    folder = f"/private/home/alupu/explainable-policies/results/minatar/test/{env}/D{D}_E{epochs}/seed{seed}/"
                    if not os.path.exists(folder):
                        os.mkdir(folder)
                    argstring = \
                        f"--env {env} " \
                        f"--epochs {epochs} " \
                        f"--dataset_size {D} " \
                        f"--generations 100 " \
                        f"--popsize 1024 " \
                        f"--rollouts 2 " \
                        f"--eval_envs 4 " \
                        f"--width 64 " \
                        f"--seed {seed} " \
                        f"--folder {folder} " \
                        f"--normalize_obs 0 "

                    job = executor.submit(train_from_arg_string, argstring)
                    jobs.append(job)

