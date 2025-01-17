import submitit
from overfit_brax import train_from_arg_string
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
        "--dry",
        action="store_true",
        help="If dry, do not submit to slurm",
        default=False
    )
args = parser.parse_args()

log_folder = f"/checkpoint/alupu/explainable_policies/overfit_brax/submitit_logs/"
if not os.path.exists(log_folder):
    os.makedirs(log_folder)

executor = submitit.AutoExecutor(folder=log_folder)
executor.update_parameters(
    slurm_partition="learnfair",
    gpus_per_node=8,
    cpus_per_task=80,
    timeout_min=2000,
    slurm_job_name="gymnax",
)

jobs = []
with executor.batch():
    for env in ["ant", "halfcheetah", "walker2d"]:
        for D in [128]:
            for seed in [0, 1]:
                for epochs in [400, 700, 1000]:
                    print(env, D, epochs, seed)
                    folder = f"/private/home/alupu/explainable-policies/results/overfit_brax/{env}/p2048_r2_e4/D{D}_E{epochs}/seed{seed}/"
                    if not os.path.exists(folder):
                        os.makedirs(folder)
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
                        f"--normalize_obs 1 " \

                    if not args.dry:
                        job = executor.submit(train_from_arg_string, argstring)
                        jobs.append(job)
                    else:
                        print(argstring)
                                
print(f"Launched {len(jobs)} jobs. May the gradients be ever in your favour!")

executor = submitit.AutoExecutor(folder=log_folder)
executor.update_parameters(
    slurm_partition="learnfair",
    gpus_per_node=8,
    cpus_per_task=80,
    timeout_min=3600,
    slurm_job_name="humanoid",
)

jobs = []
with executor.batch():
    for D in [128]:
        for env in ["humanoid", "humanoidstandup"]:
            for seed in [0, 1]:
                for epochs in [400, 700, 1000]:
                    print(env, D, epochs, seed)
                    folder = f"/private/home/alupu/explainable-policies/results/overfit_brax/p2048_r2_e4/{env}/D{D}_E{epochs}/seed{seed}/"
                    if not os.path.exists(folder):
                        os.makedirs(folder)
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
                        f"--normalize_obs 1 " \

                    if not args.dry:
                        job = executor.submit(train_from_arg_string, argstring)
                        jobs.append(job)
                    else:
                        print(argstring)

print(f"Launched {len(jobs)} jobs. May the gradients be ever in your favour!")
