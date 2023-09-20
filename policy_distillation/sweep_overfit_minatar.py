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
    timeout_min=120,
    slurm_job_name="MinAtar"
)

jobs = []
with executor.batch():
    job_idx = 0
    for D in [32, 128]:
        for env in ["SpaceInvaders-MinAtar"]:
            for epochs in [16, 32, 64]:
                for seed in [0]:#,1]:
                    for sigma_init in [1.5, 1.0, 0.5]:
                        for sigma_decay in [1.0, 0.999]:
                            for lrate_init in [0.1, 0.05]:  # Outer loop
                                for lrate_decay in [1.0, 0.999]: # Outer loop
                                    for lr in [0.01, 0.003, 0.001]: # Inner loop
                                        for net in ["MLP"]:
                                            for width in [512]: #, 256]:
                                                print(env, D, epochs, seed)
                                                folder = f"/private/home/alupu/explainable-policies/results/minatar_overfit/{env}/{net}_SNES/D{D}_learn_labels{int(False)}_norm{int(False)}/si{sigma_init}_sd{sigma_decay}_li{lrate_init}_ld{lrate_decay}/lr{lr}_E{epochs}_W{width}/seed{seed}_job_idx{job_idx}/"
                                                if not os.path.exists(folder):
                                                    os.makedirs(folder)
                                                argstring = \
                                                    f"--env {env} " \
                                                    f"--dataset_size {D} " \
                                                    f"--sigma_init {sigma_init} " \
                                                    f"--sigma_decay {sigma_decay} " \
                                                    f"--sigma_limit {0.001} " \
                                                    f"--lrate_init {lrate_init} " \
                                                    f"--lrate_decay {lrate_decay} " \
                                                    f"--lr {lr} " \
                                                    f"--epochs {epochs} " \
                                                    f"--width {width} " \
                                                    f"--generations 1000 " \
                                                    f"--popsize 2048 " \
                                                    f"--rollouts 1 " \
                                                    f"--eval_envs 32 " \
                                                    f"--seed {seed} " \
                                                    f"--folder {folder} " \
                                                    f"--overfit_seed 420 " \
                                                    f"--const_normalize_obs 1 " \
                                                    f"--net {net} " \
                                                    f"--es_strategy SNES " \
                                                
                                                argstring = argstring + "--overfit "

                                                if not args.dry:
                                                    job = executor.submit(train_from_arg_string, argstring)
                                                    jobs.append(job)
                                                else:
                                                    print(f"{job_idx}: python distill_minatar.py ", argstring)
                                                    
                                                job_idx += 1

                    
if not args.dry:
    print(f"Launched {len(jobs)} jobs. May the gradients be ever in your favour!")
    
