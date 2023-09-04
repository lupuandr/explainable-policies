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

log_folder = f"/checkpoint/alupu/explainable_policies/brax/submitit_logs/"
if not os.path.exists(log_folder):
    os.makedirs(log_folder)
# executor = submitit.AutoExecutor(folder=log_folder)
# executor.update_parameters(
#     slurm_partition="learnfair",
#     gpus_per_node=8,
#     cpus_per_task=80,
#     timeout_min=240,
#     slurm_job_name="xs_sweep",
# )

# jobs = []
# with executor.batch():
#     for D in [4, 64, 1024]:
#         for env in ["reacher", "inverted_pendulum", "inverted_double_pendulum"]:
#             for epochs in [50, 100, 200, 400]:
#                 for seed in [0, 1, 2]:
#                     print(env, D, epochs, seed)
#                     folder = f"/private/home/alupu/explainable-policies/results/brax_individual_runs/{env}/D{D}_E{epochs}/seed{seed}/"
#                     if not os.path.exists(folder):
#                         os.makedirs(folder)
#                     argstring = \
#                         f"--env {env} " \
#                         f"--epochs {epochs} " \
#                         f"--dataset_size {D} " \
#                         f"--generations 1000 " \
#                         f"--popsize 2048 " \
#                         f"--rollouts 2 " \
#                         f"--eval_envs 4 " \
#                         f"--width 512 " \
#                         f"--seed {seed} " \
#                         f"--folder {folder} " \
#                         f"--normalize_obs 1 "
#                     if not args.dry:
#                         job = executor.submit(train_from_arg_string, argstring)
#                         jobs.append(job)
#                     else:
#                         print(argstring)
  
# print("-------------------------------")
        
# executor = submitit.AutoExecutor(folder=log_folder)
# executor.update_parameters(
#     slurm_partition="learnfair",
#     gpus_per_node=8,
#     cpus_per_task=80,
#     timeout_min=1080,
#     slurm_job_name="pusher",
# )

# jobs = []
# with executor.batch():
#     for D in [4, 64, 1024]:
#         for env in ["pusher"]:
#             for epochs in [50, 100, 200, 400]:
#                 for seed in [0, 1, 2]:
#                     print(env, D, epochs, seed)
#                     folder = f"/private/home/alupu/explainable-policies/results/brax_individual_runs/{env}/D{D}_E{epochs}/seed{seed}/"
#                     if not os.path.exists(folder):
#                         os.makedirs(folder)
#                     argstring = \
#                         f"--env {env} " \
#                         f"--epochs {epochs} " \
#                         f"--dataset_size {D} " \
#                         f"--generations 1000 " \
#                         f"--popsize 2048 " \
#                         f"--rollouts 2 " \
#                         f"--eval_envs 4 " \
#                         f"--width 512 " \
#                         f"--seed {seed} " \
#                         f"--folder {folder} " \
#                         f"--normalize_obs 1 "

#                     if not args.dry:
#                         job = executor.submit(train_from_arg_string, argstring)
#                         jobs.append(job)
#                     else:
#                         print(argstring)

# print("-------------------------------")
        
executor = submitit.AutoExecutor(folder=log_folder)
executor.update_parameters(
    slurm_partition="learnfair",
    gpus_per_node=8,
    cpus_per_task=80,
    timeout_min=3600,
    slurm_job_name="ant",
)

jobs = []
with executor.batch():
    for D in [64]:
        for env in ["ant"]:
            for seed in [0, 1]:
                for epochs in [400]:
                    for sigma_init in [0.03]:
                        for lrate_init in [0.005, 0.01, 0.02, 0.04]:
                            print(env, D, epochs, lrate_init, seed)
                            folder = f"/private/home/alupu/explainable-policies/results/brax_individual_runs/{env}/D{D}_E{epochs}/si{sigma_init}_lri{lrate_init}/seed{seed}/"
                            if not os.path.exists(folder):
                                os.makedirs(folder)
                            argstring = \
                                f"--env {env} " \
                                f"--epochs {epochs} " \
                                f"--dataset_size {D} " \
                                f"--generations 2000 " \
                                f"--popsize 1024 " \
                                f"--rollouts 4 " \
                                f"--eval_envs 4 " \
                                f"--width 512 " \
                                f"--seed {seed} " \
                                f"--folder {folder} " \
                                f"--normalize_obs 1 " \
                                f"--sigma_init {sigma_init} " \
                                f"--lrate_init {lrate_init} " \

                            if not args.dry:
                                job = executor.submit(train_from_arg_string, argstring)
                                jobs.append(job)
                            else:
                                print(argstring)
                                
print(f"Launched {len(jobs)} jobs. May the gradients be ever in your favour!")


# for env in "ant"
# do
#     for D in 4 64
#     do
#         for epochs in 400
#         do
#             for seed in 0 1 2
#             do 
#                 echo $env ${D} ${epochs} ${seed}
#                 folder="/private/home/alupu/explainable-policies/results/brax_individual_runs/${env}/rollouts16_envs16/D${D}_E${epochs}/seed${seed}/"
#                 mkdir -p ${folder}
#                 sbatch --job-name "hr_${env}_${D}" \
#                        --partition learnfair \
#                        --gpus-per-node 8 \
#                        --cpus-per-task 80 \
#                        --output ${folder}/std.out \
#                        --error ${folder}/std.err \
#                        --time 3600 \
#                        --wrap "
#                 #!/bin/bash
#                 python /private/home/alupu/explainable-policies/policy_distillation/distill_brax.py \
#                        --env ${env} \
#                        --epochs ${epochs} \
#                        --dataset_size ${D} \
#                        --generations 1000 \
#                        --popsize 2048 \
#                        --rollouts 16 \
#                        --eval_envs 16 \
#                        --width 512 \
#                        --seed ${seed} \
#                        --folder ${folder} \
#                 "
#             done
#         done
#     done
# done


