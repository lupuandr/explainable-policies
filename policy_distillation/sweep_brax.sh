#!/bin/bash

env="ant"

# Creating a temporary file to hold all combinations of parameters
parameters_file=$(mktemp)
for D in 4 16 64
do
  for activation in tanh
  do
    for width in 512
    do
      for popsize in 512 #1024 2048
      do
        for epochs in 20 40 80
        do
          for sigma_init in 0.03 0.06
          do
            for sigma_decay in 1.0 0.999 1.001
            do
              for seed in 0 1
              do
                folder="/private/home/alupu/explainable-policies/results/brax/${env}/D${D}/${activation}${width}/p${popsize}_e${epochs}_si${sigma_init}_sd${sigma_decay}/seed${seed}/"
                echo "${env} ${D} ${activation} ${width} ${popsize} ${epochs} ${sigma_init} ${sigma_decay} ${seed} ${folder}" >> $parameters_file
              done
            done
          done
        done
      done
    done
  done
done

# Submitting the job array
sbatch --job-name "${env}_sweep" \
       --partition learnfair \
       --gpus-per-node 8 \
       --cpus-per-task 80 \
       --time 500 \
       --array=1-$(cat $parameters_file | wc -l) \
       --output /private/home/alupu/explainable-policies/results/brax/${env}/sweep_logs/%a.out \
       --error /private/home/alupu/explainable-policies/results/brax/${env}/sweep_logs/%a.err \
       --wrap "
#!/bin/bash

# Read the corresponding line from the parameters file
read env D activation width popsize epochs sigma_init sigma_decay seed folder < <(sed -n \${SLURM_ARRAY_TASK_ID}p $parameters_file)

# Make the directory
mkdir -p \$folder

# Execute the script
python /private/home/alupu/explainable-policies/policy_distillation/distill_brax.py \\
       --env \$env \\
       --epochs \$epochs \\
       --dataset_size \$D \\
       --generations 200 \\
       --popsize \$popsize \\
       --activation \$activation \\
       --width \$width \\
       --sigma_init \$sigma_init \\
       --sigma_decay \$sigma_decay \\
       --seed \$seed \\
       --folder \$folder \\
"
#### rm $parameters_file