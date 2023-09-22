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
    timeout_min=240,
    slurm_job_name="CNN_MinAtar2"
)


# CNN sweep
jobs = []
with executor.batch():
    job_idx = 0
    for D in [16, 32]:
        for env in ["SpaceInvaders-MinAtar", "Breakout-MinAtar"]:
            for epochs in [32, 64, 128, 256]:
                for seed in [0]:
                    for sigma_init in [1.0]:
                        for temperature in [20]:
                            for lr in [0.1, 0.05, 0.01, 0.003]: # Inner loop
                                for net in ["CNN"]:
                                    for width in [32, 64, 256]: #, 256]:
                                        for ffwd_layers in [1]:
                                            print(env, D, epochs, seed)
                                            folder = f"/private/home/alupu/explainable-policies/results/overfit_minatar/{env}/{net}_SNES/D{D}_learn_labels{int(False)}_greedy{int(True)}_norm{int(False)}/si{sigma_init}_T{temperature}/lr{lr}_E{epochs}_W{width}_L{ffwd_layers}/seed{seed}_job_idx{job_idx}/"
                                            if not os.path.exists(folder):
                                                os.makedirs(folder)
                                            argstring = \
                                                f"--env {env} " \
                                                f"--dataset_size {D} " \
                                                f"--sigma_init {sigma_init} " \
                                                f"--temperature {temperature} " \
                                                f"--sigma_limit {0.001} " \
                                                f"--lr {lr} " \
                                                f"--epochs {epochs} " \
                                                f"--width {width} " \
                                                f"--ffwd_layers {ffwd_layers} " \
                                                f"--generations 2000 " \
                                                f"--popsize 2048 " \
                                                f"--rollouts 1 " \
                                                f"--eval_envs 32 " \
                                                f"--seed {seed} " \
                                                f"--folder {folder} " \
                                                f"--overfit_seed {420 + seed} " \
                                                f"--const_normalize_obs 1 " \
                                                f"--net {net} " \
                                                f"--es_strategy SNES " \

                                            argstring = argstring + "--overfit "
                                            argstring = argstring + "--greedy_act "

                                            if not args.dry:
                                                job = executor.submit(train_from_arg_string, argstring)
                                                jobs.append(job)
                                            else:
                                                print(f"{job_idx}: python distill_minatar.py ", argstring)

                                            job_idx += 1

 # ----------------------------------------------

# MLP Sweep
# jobs = []
# with executor.batch():
#     job_idx = 0
#     for D in [16, 32]:
#         for env in ["SpaceInvaders-MinAtar", "Asterix-MinAtar", "Breakout-MinAtar"]:
#             for epochs in [32, 64]:
#                 for seed in [0, 1]:
#                     for sigma_init in [1.5, 1.0, 0.5]:
#                         for temperature in [10, 15, 20, 25, 30]:
#                             for lr in [0.003]: # Inner loop
#                                 for net in ["MLP"]:
#                                     for width in [512]: #, 256]:
#                                         print(env, D, epochs, seed)
#                                         folder = f"/private/home/alupu/explainable-policies/results/overfit_minatar/{env}/{net}_SNES/D{D}_learn_labels{int(False)}_greedy{int(True)}_norm{int(False)}/si{sigma_init}_T{temperature}/lr{lr}_E{epochs}_W{width}/seed{seed}_job_idx{job_idx}/"
#                                         if not os.path.exists(folder):
#                                             os.makedirs(folder)
#                                         argstring = \
#                                             f"--env {env} " \
#                                             f"--dataset_size {D} " \
#                                             f"--sigma_init {sigma_init} " \
#                                             f"--temperature {temperature} " \
#                                             f"--sigma_limit {0.001} " \
#                                             f"--lr {lr} " \
#                                             f"--epochs {epochs} " \
#                                             f"--width {width} " \
#                                             f"--generations 2000 " \
#                                             f"--popsize 2048 " \
#                                             f"--rollouts 1 " \
#                                             f"--eval_envs 32 " \
#                                             f"--seed {seed} " \
#                                             f"--folder {folder} " \
#                                             f"--overfit_seed {420 + seed} " \
#                                             f"--const_normalize_obs 1 " \
#                                             f"--net {net} " \
#                                             f"--es_strategy SNES " \

#                                         argstring = argstring + "--overfit "
#                                         argstring = argstring + "--greedy_act "

#                                         if not args.dry:
#                                             job = executor.submit(train_from_arg_string, argstring)
#                                             jobs.append(job)
#                                         else:
#                                             print(f"{job_idx}: python distill_minatar.py ", argstring)

#                                         job_idx += 1
                                           
                                            

# ----------------------------------------------

# Learned labels Sweep
jobs = []
with executor.batch():
    job_idx = 0
    for D in [16, 32]:
        for env in ["SpaceInvaders-MinAtar", "Asterix-MinAtar", "Breakout-MinAtar"]:
            for epochs in [32, 64]:
                for seed in [0]:
                    for sigma_init in [0.5]:
                        for temperature in [20]:
                            for lr in [0.003, 0.01, 0.05]: # Inner loop
                                for net in ["MLP"]:
                                    for width in [256]: #, 256]:
                                        
                                        if net == "MLP":
                                            width = width * 2
                                        # Affects CNN only
                                        ffwd_layers = 1
                                        
                                        print(env, D, epochs, seed)
                                        folder = f"/private/home/alupu/explainable-policies/results/overfit_minatar/{env}/{net}_SNES/D{D}_learn_labels{int(True)}_greedy{int(True)}_norm{int(False)}/si{sigma_init}_T{temperature}/lr{lr}_E{epochs}_W{width}/seed{seed}_job_idx{job_idx}/"
                                        if not os.path.exists(folder):
                                            os.makedirs(folder)
                                        argstring = \
                                            f"--env {env} " \
                                            f"--dataset_size {D} " \
                                            f"--sigma_init {sigma_init} " \
                                            f"--temperature {temperature} " \
                                            f"--sigma_limit {0.001} " \
                                            f"--lr {lr} " \
                                            f"--epochs {epochs} " \
                                            f"--width {width} " \
                                            f"--ffwd_layers {ffwd_layers} " \
                                            f"--generations 2000 " \
                                            f"--popsize 2048 " \
                                            f"--rollouts 1 " \
                                            f"--eval_envs 32 " \
                                            f"--seed {seed} " \
                                            f"--folder {folder} " \
                                            f"--overfit_seed {420 + seed} " \
                                            f"--const_normalize_obs 1 " \
                                            f"--net {net} " \
                                            f"--es_strategy SNES " \

                                        argstring = argstring + "--overfit "
                                        argstring = argstring + "--greedy_act "
                                        argstring = argstring + "--learn_labels "

                                        if not args.dry:
                                            job = executor.submit(train_from_arg_string, argstring)
                                            jobs.append(job)
                                        else:
                                            print(f"{job_idx}: python distill_minatar.py ", argstring)

                                        job_idx += 1


# --------------------------------------------------

                    
if not args.dry:
    print(f"Launched {len(jobs)} jobs. May the gradients be ever in your favour!")
    
