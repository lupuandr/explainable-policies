import submitit
from distill_minatar import train_from_arg_string
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

log_folder = f"/checkpoint/alupu/explainable_policies/minatar/submitit_logs/"
if not os.path.exists(log_folder):
    os.makedirs(log_folder)
executor = submitit.AutoExecutor(folder=log_folder)
executor.update_parameters(
    slurm_partition="learnfair",
    gpus_per_node=8,
    cpus_per_task=80,
    timeout_min=900,
    slurm_job_name="MinAtar"
)

jobs = []
with executor.batch():
    for D in [64]:
        for env in ["SpaceInvaders-MinAtar", "Breakout-MinAtar", "Freeway-MinAtar", "Asterix-MinAtar"]:
            for epochs in [200, 400]:
                for seed in [0,1]:
                    print(env, D, epochs, seed)
                    folder = f"/private/home/alupu/explainable-policies/results/minatar/{env}/D{D}_E{epochs}/seed{seed}/"
                    if not os.path.exists(folder):
                        os.makedirs(folder)
                    argstring = \
                        f"--env {env} " \
                        f"--epochs {epochs} " \
                        f"--dataset_size {D} " \
                        f"--generations 20000 " \
                        f"--popsize 1024 " \
                        f"--rollouts 2 " \
                        f"--eval_envs 4 " \
                        f"--width 64 " \
                        f"--seed {seed} " \
                        f"--folder {folder} " \

                    if not args.dry:
                        job = executor.submit(train_from_arg_string, argstring)
                        jobs.append(job)
                    else:
                        print("python distill_minatar.py ", argstring)

                    
if not args.dry:
    print(f"Launched {len(jobs)} jobs. May the gradients be ever in your favour!")
    
