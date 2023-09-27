import submitit
from baseline_brax import train_from_arg_string as train_es_brax
from baseline_minatar import train_from_arg_string as train_es_minatar
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

# # BRAX
# # Common args:
# root = "/private/home/alupu/explainable-policies/results/ICLR/baselines/brax/no_virtual_batch_norm/"

# log_folder = f"/checkpoint/alupu/explainable_policies/brax/submitit_logs/"
# if not os.path.exists(log_folder):
#     os.makedirs(log_folder)

# def schedule_job_brax(env, seed, folder):
#     if not os.path.exists(folder):
#         os.makedirs(folder)

#     argstring = \
#         f"--env {env} " \
#         f"--seed {seed} " \
#         f"--folder {folder} " \

#     argstring = argstring + "--normalize_obs 1 --const_normalize_obs 0 "

#     if not args.dry:
#         job = executor.submit(train_es_brax, argstring)
#         return job
#     else:
#         print(argstring)
#         return -1


# executor = submitit.AutoExecutor(folder=log_folder)
# executor.update_parameters(
#     slurm_partition="learnlab",
#     gpus_per_node=8,
#     cpus_per_task=80,
#     timeout_min=300,
#     slurm_job_name="es_small_brax",
#     slurm_exclude="learnfair2058"
# )

# jobs = []
# with executor.batch():
#     for env in ["hopper", "walker2d", "reacher", "inverted_double_pendulum"]:
#         for seed in [0, 1, 2]:
#             folder = f"{root}/{env}/seed{seed}/"
#             job = schedule_job_brax(env, seed, folder)
#             jobs.append(job)

# if not args.dry:
#     print(f"Launched {len(jobs)} jobs. May the gradients be ever in your favour!")

# executor = submitit.AutoExecutor(folder=log_folder)
# executor.update_parameters(
#     slurm_partition="learnlab",
#     gpus_per_node=8,
#     cpus_per_task=80,
#     timeout_min=720,
#     slurm_job_name="es_big_brax",
#     slurm_exclude="learnfair2058"
# )

# jobs = []
# with executor.batch():
#     for env in ["humanoid", "humanoidstandup", "halfcheetah", "ant"]:
#         for seed in [0, 1, 2]:
#             folder = f"{root}/{env}/seed{seed}/"
#             job = schedule_job_brax(env, seed, folder)
#             jobs.append(job)

# if not args.dry:
#     print(f"Launched {len(jobs)} jobs. May the gradients be ever in your favour!")


# # ==================================================================================================


# Common args:
root = "/private/home/alupu/explainable-policies/results/ICLR/baselines/minatar/virtual_batch_norm/"

log_folder = f"/checkpoint/alupu/explainable_policies/minatar/submitit_logs/"
if not os.path.exists(log_folder):
    os.makedirs(log_folder)

def schedule_job_minatar(env, seed, folder, mode):
    if not os.path.exists(folder):
        os.makedirs(folder)

    argstring = \
        f"--env {env} " \
        f"--seed {seed} " \
        f"--folder {folder} " \

    argstring = argstring + "--normalize_obs 0 --const_normalize_obs 1 "
    
    if mode == "half_popsize":
        argstring += "--popsize 1024 "
    elif mode == "half_width":
        argstring += "--width 256 "

    if not args.dry:
        job = executor.submit(train_es_minatar, argstring)
        return job
    else:
        print(argstring)
        return -1

# MinAtar
executor = submitit.AutoExecutor(folder=log_folder)
executor.update_parameters(
    slurm_partition="learnlab",
    gpus_per_node=8,
    cpus_per_task=80,
    timeout_min=300,
    slurm_job_name="rerun_es_minatar",
    slurm_exclude="learnfair2058"
)

jobs = []
with executor.batch():
    for env in ["SpaceInvaders-MinAtar", "Breakout-MinAtar", "Asterix-MinAtar", "Freeway-MinAtar"]:
        for seed in [0, 1, 2]:
            for mode in ["half_popsize", "half_width"]:
                folder = f"{root}/{env}/{mode}/seed{seed}/"
                job = schedule_job_minatar(env, seed, folder, mode)
                jobs.append(job)

if not args.dry:
    print(f"Launched {len(jobs)} jobs. May the gradients be ever in your favour!")
