#!/bin/bash 

#SBATCH --job-name=glow_vlog
#SBATCH --mem-per-cpu=1024
#SBATCH --partition long
#SBATCH --account cvit_bhaasha
#SBATCH --gres=gpu:1
#SBATCH --mincpus=10
#SBATCH --nodes=1
#SBATCH --time 4-01:00:00
#SBATCH --signal=B:HUP@600
#SBATCH -w gnode70

source /home2/aditya1/miniconda3/bin/activate nvae
cd /ssd_scratch/cvit/aditya1/glow-tts/
sh train_ddi.sh configs/base.json base