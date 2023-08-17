
#!/bin/bash


# for env in "ant"
# do
#     for D in 4 64
#     do
#         for epochs in 400
#         do
#             for seed in 0 1
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
#                        --time 2880 \
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

# Split in two only to have different Slurm runtimes
for env in "hopper" "reacher" "pusher" "inverted_pendulum" "inverted_double_pendulum"
do
    for D in 2 4 64
    do
        for epochs in 50 100 200 400
        do
            for seed in 0 1 2
            do 
                echo $env ${D} ${epochs} ${seed}
                folder="/private/home/alupu/explainable-policies/results/brax_individual_runs/${env}/D${D}_E${epochs}/seed${seed}/"
                mkdir -p ${folder}
                sbatch --job-name "${env}_${D}" \
                       --partition learnfair \
                       --gpus-per-node 8 \
                       --cpus-per-task 80 \
                       --output ${folder}/std.out \
                       --error ${folder}/std.err \
                       --time 1080 \
                       --wrap "
                #!/bin/bash
                python /private/home/alupu/explainable-policies/policy_distillation/distill_brax.py \
                       --env ${env} \
                       --epochs ${epochs} \
                       --dataset_size ${D} \
                       --generations 1000 \
                       --popsize 2048 \
                       --rollouts 2 \
                       --eval_envs 4 \
                       --width 512 \
                       --seed ${seed} \
                       --folder ${folder} \
                "
            done
        done
    done
done

# for env in "humanoid" "humanoidstandup"
# do
#     for D in 4 64 1024
#     do
#         for epochs in 50 100 200 400
#         do
#             for seed in 0 1 2
#             do 
#                 echo $env ${D} ${epochs} ${seed}
#                 folder="/private/home/alupu/explainable-policies/results/brax_individual_runs/${env}/D${D}_E${epochs}/seed${seed}/"
#                 mkdir -p ${folder}
#                 sbatch --job-name "${env}_${D}" \
#                        --partition learnfair \
#                        --gpus-per-node 8 \
#                        --cpus-per-task 80 \
#                        --output ${folder}/std.out \
#                        --error ${folder}/std.err \
#                        --time 2880 \
#                        --wrap "
#                 #!/bin/bash
#                 python /private/home/alupu/explainable-policies/policy_distillation/distill_brax.py \
#                        --env ${env} \
#                        --epochs ${epochs} \
#                        --dataset_size ${D} \
#                        --generations 1000 \
#                        --popsize 2048 \
#                        --rollouts 2 \
#                        --eval_envs 4 \
#                        --width 512 \
#                        --seed ${seed} \
#                        --folder ${folder} \
#                 "
#             done
#         done
#     done
# done

# # ----------------------------------------------------------------------------------------------

# env="humanoid"

# for D in 1024 2048
# do
# echo $env ${D}
# mkdir -p /private/home/alupu/explainable-policies/results/brax_${env}/D${D}/
# sbatch --job-name "${env}_${D}" \
#        --partition learnfair \
#        --gpus-per-node 8 \
#        --cpus-per-task 80 \
#        --output /private/home/alupu/explainable-policies/results/brax_${env}/D${D}/std.out \
#        --error /private/home/alupu/explainable-policies/results/brax_${env}/D${D}/std.err \
#        --time 2880 \
#        --wrap "
# #!/bin/bash
# python /private/home/alupu/explainable-policies/policy_distillation/distill_brax.py \
#        --env ${env} \
#        --epochs 500 \
#        --dataset_size ${D} \
#        --generations 500 \
#        --popsize 1024 \
# "
# done

# # ----------------------------------------------------------------------------------------------

# env="hopper"

# for D in 4
# do
# for epochs in 10 20 40
# do
# for popsize in 512 1024 2048
# do
# for activation in tanh relu
# do
# for width in 64 512
# do
# echo $env ${D}
# mkdir -p /private/home/alupu/explainable-policies/results/brax_${env}/D${D}/${activation}E${epochs}P${popsize}/
# sbatch --job-name "${env}_${D}" \
#        --partition learnfair \
#        --gpus-per-node 8 \
#        --cpus-per-task 80 \
#        --output /private/home/alupu/explainable-policies/results/brax_${env}/D${D}/${activation}E${epochs}P${popsize}/std.out \
#        --error /private/home/alupu/explainable-policies/results/brax_${env}/D${D}/${activation}E${epochs}P${popsize}/std.err \
#        --time 500 \
#        --wrap "
# #!/bin/bash
# python /private/home/alupu/explainable-policies/policy_distillation/distill_brax.py \
#        --env ${env} \
#        --epochs ${epochs} \
#        --dataset_size ${D} \
#        --generations 200 \
#        --popsize ${popsize} \
#        --activation ${activation} \
#        --width ${width} \
# "
# done
# done
# done
# done
# done

# # ----------------------------------------------------------------------------------------------

# env="reacher"

# for D in 128 1024
# do
# echo $env ${D}
# mkdir -p /private/home/alupu/explainable-policies/results/brax_${env}/D${D}/
# sbatch --job-name "${env}_${D}" \
#        --partition learnfair \
#        --gpus-per-node 8 \
#        --cpus-per-task 80 \
#        --output /private/home/alupu/explainable-policies/results/brax_${env}/D${D}/std.out \
#        --error /private/home/alupu/explainable-policies/results/brax_${env}/D${D}/std.err \
#        --time 1440 \
#        --wrap "
# #!/bin/bash
# python /private/home/alupu/explainable-policies/policy_distillation/distill_brax.py \
#        --env ${env} \
#        --epochs 200 \
#        --dataset_size ${D} \
#        --generations 500 \
#        --popsize 1024 \
# "
# done

# # ----------------------------------------------------------------------------------------------

# env="halfcheetah"

# for D in 128 1024
# do
# echo $env ${D}
# mkdir -p /private/home/alupu/explainable-policies/results/brax_${env}/D${D}/
# sbatch --job-name "${env}_${D}" \
#        --partition learnfair \
#        --gpus-per-node 8 \
#        --cpus-per-task 80 \
#        --output /private/home/alupu/explainable-policies/results/brax_${env}/D${D}/std.out \
#        --error /private/home/alupu/explainable-policies/results/brax_${env}/D${D}/std.err \
#        --time 1440 \
#        --wrap "
# #!/bin/bash
# python /private/home/alupu/explainable-policies/policy_distillation/distill_brax.py \
#        --env ${env} \
#        --epochs 200 \
#        --dataset_size ${D} \
#        --generations 500 \
#        --popsize 1024 \
# "
# done

# ----------------------------------------------------------------------------------------------

# env="humanoidstandup"

# for D in 128 # 1024
# do
# echo $env ${D}
# mkdir -p /private/home/alupu/explainable-policies/results/brax_${env}/D${D}/
# sbatch --job-name "${env}_${D}" \
#        --partition learnfair \
#        --gpus-per-node 8 \
#        --cpus-per-task 80 \
#        --output /private/home/alupu/explainable-policies/results/brax_${env}/D${D}/std.out \
#        --error /private/home/alupu/explainable-policies/results/brax_${env}/D${D}/std.err \
#        --time 1680 \
#        --wrap "
# #!/bin/bash
# python /private/home/alupu/explainable-policies/policy_distillation/distill_brax.py \
#        --env ${env} \
#        --epochs 200 \
#        --dataset_size ${D} \
#        --generations 500 \
#        --popsize 1024 \
#        --seed 0
# "
# done

# ----------------------------------------------------------------------------------------------

# env="walker2d"

# for D in 16 128 1024
# do
# echo $env ${D}
# mkdir -p /private/home/alupu/explainable-policies/results/brax_${env}/D${D}/
# sbatch --job-name "${env}_${D}" \
#        --partition learnfair \
#        --gpus-per-node 8 \
#        --cpus-per-task 80 \
#        --output /private/home/alupu/explainable-policies/results/brax_${env}/D${D}/std.out \
#        --error /private/home/alupu/explainable-policies/results/brax_${env}/D${D}/std.err \
#        --time 1440 \
#        --wrap "
# #!/bin/bash
# python /private/home/alupu/explainable-policies/policy_distillation/distill_brax.py \
#        --env ${env} \
#        --epochs 200 \
#        --dataset_size ${D} \
#        --generations 500 \
#        --popsize 1024 \
# "
# done