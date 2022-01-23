#!/bin/bash 

#SBATCH --job-name=fadec2
#SBATCH --mem-per-cpu=1024
#SBATCH --partition long
#SBATCH --account cvit_bhaasha
#SBATCH --gres=gpu:4
#SBATCH --mincpus=40
#SBATCH --nodes=1
#SBATCH --time 4-00:00:00
#SBATCH --signal=B:HUP@600
#SBATCH -w gnode37

source /home2/aditya1/miniconda3/bin/activate fastspeech2
cd /ssd_scratch/cvit/aditya1
python landmark_generation.py -s AnfisaNava,BestDressed,SejalKumar,MKBHD,JohnnyHarris