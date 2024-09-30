#!/bin/bash
#SBATCH --job-name=bias_identification
#SBATCH --account=rrg-zhu2048
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=150G
#SBATCH --output=job_%x_%j.out
#SBATCH --error=job_%x_%j.err
#SBATCH --time=1-12:00:00

 


#srun --exclusive --nodes=1 --mem=150G bash -c 'python answer_gen.py --model_name "./Llama-2-70b-chat-hf" --output_dir "outputs/llama2-70b" --max_length 128 --dataset_path "../../dataset/annotations/dataset.csv" --prompt_name basic'


#srun --nodes=1 --mem=150G bash -c 'python answer_gen.py --model_name "/home/chufluo/scratch/Llama-2-70b-chat-hf" --output_dir "outputs/llama2-70b" --max_length 128 --dataset_path "../../dataset/annotations/dataset.csv" --prompt_name basic'

srun --nodes=1 --mem=150G bash -c 'source ~/ENV/bin/activate; module load python/3.10.13; python answer_gen.py --constrained --model_name "/home/chufluo/scratch/Meta-Llama-3-70B-Instruct" --output_dir "outputs/llama3-70b" --max_length 128 --dataset_path "../../dataset/annotations/dataset.csv" --prompt_name basic_llama3'
srun --nodes=1 --mem=150G bash -c 'source ~/ENV/bin/activate; module load python/3.10.13; python answer_gen.py --constrained --model_name "/home/chufluo/scratch/Llama-2-70b-chat-hf" --output_dir "outputs/llama2-70b" --max_length 128 --dataset_path "../../dataset/annotations/dataset.csv" --prompt_name basic_w_inst'


#python3 answer_gen.py --model_name "meta-llama/llama-2-13b-chat-hf" --output_dir "outputs/llama-13b" --max_length 128 --dataset_path "../../dataset/annotations/dataset.csv" --prompt_name basic

#python3 answer_gen.py --model_name "upstage/SOLAR-10.7B-Instruct-v1.0" --output_dir "outputs/solar-10b" --max_length 128 --dataset_path "../../dataset/annotations/dataset.csv" --prompt_name basic

#python3 answer_gen.py --model_name "mistralai/Mistral-7B-Instruct-v0.3" --output_dir "outputs/mistral-7b" --max_length 128 --dataset_path "../../dataset/annotations/dataset.csv" --prompt_name basic


