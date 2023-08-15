
#!/bin/bash


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

env="hopper"

for D in 4
do
for epochs in 10 20 40
do
for popsize in 512 1024 2048
do
for activation in tanh relu
do
for width in 64 512
do
echo $env ${D}
mkdir -p /private/home/alupu/explainable-policies/results/brax_${env}/D${D}/${activation}E${epochs}P${popsize}/
sbatch --job-name "${env}_${D}" \
       --partition learnfair \
       --gpus-per-node 8 \
       --cpus-per-task 80 \
       --output /private/home/alupu/explainable-policies/results/brax_${env}/D${D}/${activation}E${epochs}P${popsize}/std.out \
       --error /private/home/alupu/explainable-policies/results/brax_${env}/D${D}/${activation}E${epochs}P${popsize}/std.err \
       --time 500 \
       --wrap "
#!/bin/bash
python /private/home/alupu/explainable-policies/policy_distillation/distill_brax.py \
       --env ${env} \
       --epochs ${epochs} \
       --dataset_size ${D} \
       --generations 200 \
       --popsize ${popsize} \
       --activation ${activation} \
       --width ${width} \
"
done
done
done
done
done

# # ----------------------------------------------------------------------------------------------

# env="ant"

# for D in 64 256 1024
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