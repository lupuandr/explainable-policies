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
    for D in [4]:#, 16, 64]:
        for env in ["SpaceInvaders-MinAtar", "Breakout-MinAtar", "Freeway-MinAtar", "Asterix-MinAtar"]:
            for epochs in [25, 50]:#, 200, 400]:
                for seed in [0]:#,1]:
                    for sigma_init in [0.02]:#, 0.01]: #[0.02, 0.01, 0.005, 0.0025]:
                        for sigma_decay in [1.0]:#, 0.999]: #, 0.995]:
                            for lrate_init in [0.01, 0.005]: #[0.04, 0.02, 0.01, 0.005, 0.0025]: # Outer loop
                                for lrate_decay in [1.0]:#, 0.999]: #, 0.995]: # Outer loop
                                    for lr in [0.2, 0.1, 0.05]: #[0.2, 0.1, 0.05, 0.025]: # Inner loop
                                            for width in [64, 128]: #, 256]:
                                                print(env, D, epochs, seed)
                                                folder = f"/private/home/alupu/explainable-policies/results/minatar/{env}/MLP/D{D}_learn_labels{int(False)}_norm{int(False)}/si{sigma_init}_sd{sigma_decay}_li{lrate_init}_ld{lrate_decay}/lr{lr}_E{epochs}_W{width}/seed{seed}_job_idx{job_idx}/"
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
                                                    f"--popsize 1024 " \
                                                    f"--rollouts 2 " \
                                                    f"--eval_envs 4 " \
                                                    f"--seed {seed} " \
                                                    f"--folder {folder} " \
                                                
                                                argstring = argstring + "--greedy_act "

                                                if not args.dry:
                                                    job = executor.submit(train_from_arg_string, argstring)
                                                    jobs.append(job)
                                                else:
                                                    print(f"{job_idx}: python distill_minatar.py ", argstring)
                                                    
                                                job_idx += 1

                    
if not args.dry:
    print(f"Launched {len(jobs)} jobs. May the gradients be ever in your favour!")
    
