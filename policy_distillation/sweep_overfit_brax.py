import submitit
from distill_brax import train_from_arg_string
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
    timeout_min=1000,
    slurm_job_name="ant",
)

jobs = []
with executor.batch():
    for env in ["ant", "halfcheetah", "walker2d"]:
        for D in [4, 64]:
            for seed in [0, 1]:
                for epochs in [100, 200, 400]:
                    for sigma_init in [0.03]:
                            print(env, D, epochs, sigma_init, seed)
                            folder = f"/private/home/alupu/explainable-policies/results/overfit_brax/{env}/D{D}_E{epochs}/seed{seed}/"
                            if not os.path.exists(folder):
                                os.makedirs(folder)
                            argstring = \
                                f"--env {env} " \
                                f"--epochs {epochs} " \
                                f"--dataset_size {D} " \
                                f"--generations 1000 " \
                                f"--popsize 1024 " \
                                f"--rollouts 1 " \
                                f"--eval_envs 4 " \
                                f"--width 512 " \
                                f"--seed {seed} " \
                                f"--folder {folder} " \
                                f"--normalize_obs 1 " \
                                f"--sigma_init {sigma_init} " \

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
    for D in [4, 64]:
        for env in ["humanoid", "humanoidstandup"]:
            for seed in [0, 1]:
                for epochs in [100, 200, 400]:
                    for sigma_init in [0.03]:
                        print(env, D, epochs, sigma_init, seed)
                        folder = f"/private/home/alupu/explainable-policies/results/overfit_brax/{env}/D{D}_E{epochs}/seed{seed}/"
                        if not os.path.exists(folder):
                            os.makedirs(folder)
                        argstring = \
                            f"--env {env} " \
                            f"--epochs {epochs} " \
                            f"--dataset_size {D} " \
                            f"--generations 1000 " \
                            f"--popsize 1024 " \
                            f"--rollouts 1 " \
                            f"--eval_envs 4 " \
                            f"--width 512 " \
                            f"--seed {seed} " \
                            f"--folder {folder} " \
                            f"--normalize_obs 1 " \
                            f"--sigma_init {sigma_init} " \
 \
                                if not args.dry:
                                    job = executor.submit(train_from_arg_string, argstring)
                                    jobs.append(job)
                                else:
                                    print(argstring)

print(f"Launched {len(jobs)} jobs. May the gradients be ever in your favour!")
