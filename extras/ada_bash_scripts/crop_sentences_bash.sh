#!/bin/bash 

#SBATCH --job-name=crop5
#SBATCH --mem-per-cpu=1024
#SBATCH --partition long
#SBATCH --account cvit_bhaasha
#SBATCH --gres=gpu:0
#SBATCH --mincpus=40
#SBATCH --nodes=1
#SBATCH --time 4-01:00:00
#SBATCH --signal=B:HUP@600
#SBATCH -w gnode68

source /home2/aditya1/miniconda3/bin/activate fastspeech2
cd /ssd_scratch/cvit/aditya1/
python download_yt_transcripts.py -s AnfisaNava,BestDressed,JackEdwards,JohnnyHarris,SejalKumar
