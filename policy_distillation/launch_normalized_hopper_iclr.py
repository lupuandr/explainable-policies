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

# Common args:
root = "/private/home/alupu/explainable-policies/results/ICLR/brax/"

log_folder = f"/checkpoint/alupu/explainable_policies/brax/submitit_logs/"
if not os.path.exists(log_folder):
    os.makedirs(log_folder)


def schedule_job(env, D, seed, epochs, folder, overfit, args):
    if not os.path.exists(folder):
        os.makedirs(folder)
        
    if overfit:
        rollouts = 1
    else:
        rollouts = 2

    argstring = \
        f"--env {env} " \
        f"--epochs {epochs} " \
        f"--dataset_size {D} " \
        f"--seed {seed} " \
        f"--project Behaviour-Distillation-ICLR " \
        f"--folder {folder} " \
        f"--rollouts {rollouts} " \

    argstring = argstring + "--normalize_obs 0 --const_normalize_obs 1 "
    
    if overfit:
        argstring = argstring + f"--overfit --overfit_seed {420+seed} "

    if not args.dry:
        job = executor.submit(train_from_arg_string, argstring)
        return job
    else:
        print(argstring)
        return -1


# SMALL ENVS
executor = submitit.AutoExecutor(folder=log_folder)
executor.update_parameters(
    slurm_partition="learnlab",
    gpus_per_node=8,
    cpus_per_task=80,
    timeout_min=300,
    slurm_job_name="hopper",
    slurm_exclude="learnfair2058"
)

jobs = []
with executor.batch():
    for env in ["hopper"]:
        for D in [4]:
            for seed in [0, 1, 2]:
                for overfit in [False]:

                    epochs = 200 if env in ["hopper", "inverted_double_pendulum"] else 400

                    if overfit:
                        folder = f"{root}/overfit/{env}/D{D}/seed{seed}/"
                        job = schedule_job(env, D, seed, epochs, folder, overfit=True, args=args)
                    else:
                        folder = f"{root}/distill/{env}/D{D}/seed{seed}/"
                        job = schedule_job(env, D, seed, epochs, folder, overfit=False, args=args)
                    jobs.append(job)

if not args.dry:
    print(f"Launched {len(jobs)} jobs. May the gradients be ever in your favour!")
