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
    slurm_partition="learnfair",
    gpus_per_node=8,
    cpus_per_task=80,
    timeout_min=180,
    slurm_job_name="MNIST"
)

# ConvNet Sweep
jobs = []
with executor.batch():
    job_idx = 0
    for D in [10]: #, 50]:
        for dataset in ["MNIST", "FashionMNIST"]:
            for init_mode in ["zero", "mean"]:
                for learn_labels in [False]:
                    for sigma_init in [0.02, 0.01]: #[0.02, 0.01, 0.005, 0.0025]:
                        for sigma_decay in [1.0, 0.999]: #, 0.995]:
                            for lrate_init in [0.04, 0.01]: #[0.04, 0.02, 0.01, 0.005, 0.0025]: # Outer loop
                                for lrate_decay in [1.0, 0.999]: #, 0.995]: # Outer loop
                                    for lr in [0.1, 0.05]: #[0.2, 0.1, 0.05, 0.025]: # Inner loop
                                        for epochs in [1000, 2000]: #[500, 1000, 2000, 4000]:
                                            for width in [128]: #, 256]:
                                                for seed in [0]: #, 1, 2]:

                                                    folder = f"/private/home/alupu/explainable-policies/results/mnist/{dataset}/CNN/D{D}_M{init_mode}_learn_labels{int(learn_labels)}_norm{int(True)}/si{sigma_init}_sd{sigma_decay}_li{lrate_init}_ld{lrate_decay}/lr{lr}_E{epochs}_W{width}/seed{seed}_job_idx{job_idx}/"
                                                    if not os.path.exists(folder):
                                                        os.makedirs(folder)
                                                    argstring = \
                                                        f"--net CNN " \
                                                        f"--dataset_size {D} " \
                                                        f"--dataset {dataset} " \
                                                        f"--init_mode {init_mode} " \
                                                        f"--sigma_init {sigma_init} " \
                                                        f"--sigma_decay {sigma_decay} " \
                                                        f"--sigma_limit {0.001} " \
                                                        f"--lrate_init {lrate_init} " \
                                                        f"--lrate_decay {lrate_decay} " \
                                                        f"--lr {lr} " \
                                                        f"--epochs {epochs} " \
                                                        f"--width {width} " \
                                                        f"--generations 3000 " \
                                                        f"--popsize 512 " \
                                                        f"--rollouts 2 " \
                                                        f"--seed {seed} " \
                                                        f"--folder {folder} " \

                                                    if learn_labels:
                                                        argstring = argstring + "--learn_labels "
                                                    
                                                    # Log dataset to wandb
                                                    argstring = argstring + "--log_dataset "
                                                    argstring = argstring + "--normalize "

                                                    if not args.dry:
                                                        job = executor.submit(train_from_arg_string, argstring)
                                                        jobs.append(job)
                                                    else:
                                                        print(f"{job_idx}: python distill_minatar.py ", argstring)
                                                        
                                                    job_idx += 1




# MLP SWEEP
# jobs = []
# with executor.batch():
#     job_idx = 0
#     for D in [10]: #, 50]:
#         for dataset in ["MNIST", "FashionMNIST"]:
#             for init_mode in ["zero", "sample", "mean"]:
#                 for learn_labels in [False]:
#                     for sigma_init in [0.02, 0.01]: #[0.02, 0.01, 0.005, 0.0025]:
#                         for sigma_decay in [1.0, 0.999]: #, 0.995]:
#                             for lrate_init in [0.04, 0.01]: #[0.04, 0.02, 0.01, 0.005, 0.0025]: # Outer loop
#                                 for lrate_decay in [1.0, 0.999]: #, 0.995]: # Outer loop
#                                     for lr in [0.1, 0.05]: #[0.2, 0.1, 0.05, 0.025]: # Inner loop
#                                         for epochs in [2000]: #[500, 1000, 2000, 4000]:
#                                             for width in [128]: #, 256]:
#                                                 for seed in [0]: #, 1, 2]:

#                                                     folder = f"/private/home/alupu/explainable-policies/results/mnist/{dataset}/D{D}_M{init_mode}_learn_labels{int(learn_labels)}_norm{int(True)}/si{sigma_init}_sd{sigma_decay}_li{lrate_init}_ld{lrate_decay}/lr{lr}_E{epochs}_W{width}/seed{seed}_job_idx{job_idx}/"
#                                                     if not os.path.exists(folder):
#                                                         os.makedirs(folder)
#                                                     argstring = \
#                                                         f"--dataset_size {D} " \
#                                                         f"--dataset {dataset} " \
#                                                         f"--init_mode {init_mode} " \
#                                                         f"--sigma_init {sigma_init} " \
#                                                         f"--sigma_decay {sigma_decay} " \
#                                                         f"--sigma_limit {0.001} " \
#                                                         f"--lrate_init {lrate_init} " \
#                                                         f"--lrate_decay {lrate_decay} " \
#                                                         f"--lr {lr} " \
#                                                         f"--epochs {epochs} " \
#                                                         f"--width {width} " \
#                                                         f"--generations 3000 " \
#                                                         f"--popsize 512 " \
#                                                         f"--rollouts 2 " \
#                                                         f"--seed {seed} " \
#                                                         f"--folder {folder} " \

#                                                     if learn_labels:
#                                                         argstring = argstring + "--learn_labels "
                                                    
#                                                     # Log dataset to wandb
#                                                     argstring = argstring + "--log_dataset "
#                                                     argstring = argstring + "--normalize "

#                                                     if not args.dry:
#                                                         job = executor.submit(train_from_arg_string, argstring)
#                                                         jobs.append(job)
#                                                     else:
#                                                         print(f"{job_idx}: python distill_minatar.py ", argstring)
                                                        
#                                                     job_idx += 1

if not args.dry:
    print(f"Launched {len(jobs)} jobs. May the gradients be ever in your favour!")