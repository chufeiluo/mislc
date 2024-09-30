#!/bin/bash -l
# SLURM SUBMIT SCRIPT
#SBATCH --account=general
#SBATCH --partition=Combined
#SBATCH --nodes=1
#SBATCH --time=300:00:00
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --ntasks-per-node=1


export OPENAI_API_KEY=sk-proj-SkfAcVPF3BUChO2Ybgp0T3BlbkFJRQ4zXl3KUc748JrQikMb 

python gen_outputs.py \
	    --dataset_path ../dataset/annotations/dataset.csv \
	    --docs_path ../dataset/definitions/split_docs/ \
	        --index_path ../dataset/definitions/index/ \
		--output_dir 'outputs/gpt-3.5' --model_source openai --model_name "gpt-3.5-turbo" --model_parallelism --max_new_tokens 256 --prompt_name basic_inst_retrieval
python gen_outputs.py \
	    --dataset_path ../dataset/annotations/dataset.csv \
	    --docs_path ../dataset/definitions/split_docs/ \
	        --index_path ../dataset/definitions/index/ \
		--output_dir 'outputs/gpt-4o' --model_source openai --model_name "gpt-4o" --model_parallelism --max_new_tokens 256 --prompt_name basic_inst_retrieval

