#!/bin/bash 

#SBATCH --job-name=taco_vlog
#SBATCH --mem-per-cpu=1024
#SBATCH --partition long
#SBATCH --account cvit_bhaasha
#SBATCH --gres=gpu:1
#SBATCH --mincpus=10
#SBATCH --nodes=1
#SBATCH --time 4-01:00:00
#SBATCH --signal=B:HUP@600
#SBATCH -w gnode74

source /home2/aditya1/miniconda3/bin/activate fastspeech2
cd /ssd_scratch/cvit/aditya1/tacotron2/
python train.py --output_directory=outdir_vlog --log_directory=logdir_vlog