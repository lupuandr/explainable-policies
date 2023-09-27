import submitit
from distill_dataset import train_from_arg_string
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

log_folder = f"/checkpoint/alupu/explainable_policies/mnist/submitit_logs/"
if not os.path.exists(log_folder):
    os.makedirs(log_folder)
executor = submitit.AutoExecutor(folder=log_folder)
executor.update_parameters(
    slurm_partition="learnlab",
    gpus_per_node=8,
    cpus_per_task=80,
    timeout_min=1440,
    slurm_job_name="MNIST"
)

# ConvNet Sweep
jobs = []
with executor.batch():
    job_idx = 0
    for D in [10]:
        for init_mode in ["mean"]:
            for dataset in ["MNIST", "FashionMNIST"]:
                for seed in [0, 1, 2]:

                    folder = f"/private/home/alupu/explainable-policies/results/ICLR/mnist/{dataset}/D{D}/seed{seed}/"
                    if not os.path.exists(folder):
                        os.makedirs(folder)
                    argstring = \
                        f"--net CNN " \
                        f"--dataset_size {D} " \
                        f"--dataset {dataset} " \
                        f"--seed {seed} " \
                        f"--project Behaviour-Distillation-ICLR " \
                        f"--folder {folder} " \


                    # Log dataset to wandb
                    argstring = argstring + "--log_dataset "
                    argstring = argstring + "--normalize "

                    if not args.dry:
                        job = executor.submit(train_from_arg_string, argstring)
                        jobs.append(job)
                    else:
                        print(f"{job_idx}: python distill_dataset.py ", argstring)

                    job_idx += 1

if not args.dry:
    print(f"Launched {len(jobs)} jobs. May the gradients be ever in your favour!")