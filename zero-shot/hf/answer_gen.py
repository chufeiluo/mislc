import os
import argparse
import pandas as pd
from transformers import pipeline
from tqdm import tqdm
from vllm import LLM, SamplingParams

#from utils.model_utils import load_model_and_tokenizer
from utils.file_utils import print_args
os.environ['HF_TOKEN'] = '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'
instruction = 'Classify the claim as either "factual" or "misinformation". Do not refuse to answer. Do not engage in explanations and politeness. Only respond with the words "misinformation", "factual", or "unsure". Do not add further context.'
instruction_un = 'Classify the claim as either "factual" or "misinformation".'
parser = argparse.ArgumentParser()

parser.add_argument("--output_dir", type=str)

# Model params
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--max_length", type=int, default=None)
parser.add_argument("--cache_dir", type=str, default=None)
# parser.add_argument("--model_parallelism", action="store_true")
parser.add_argument("--local_only", action="store_true")
parser.add_argument("--constrained", action="store_true")
parser.add_argument("--auth_token", type=str, default=None)

# Dataset params
parser.add_argument("--dataset_path", type=str, required=True)

# Generation params
parser.add_argument("--prompt_name", type=str, required=True)

args = parser.parse_args()

args.output_dir = os.path.join(args.output_dir, f"prompt={args.prompt_name}")
if args.output_dir is not None:
    os.makedirs(args.output_dir, exist_ok=True)
print_args(args, output_dir=args.output_dir)

df = pd.read_csv(args.dataset_path)

#model, tokenizer, config, device = load_model_and_tokenizer(
    #args.model_name,
    #model_parallelism=True,
    #cache_dir=args.cache_dir,
    #local_only=args.local_only
#)

#model_max_length = config.n_positions if hasattr(config, "n_positions") else config.max_position_embeddings
sampling_params = SamplingParams(temperature=0.3, top_p=1)
llm = LLM(model=args.model_name, download_dir='/home/chufluo/scratch', tensor_parallel_size=4, max_num_seqs=2,gpu_memory_utilization=0.95, max_model_len=1024)
#pipe = pipeline('text-generation', model=model, tokenizer=tokenizer)


def apply_prompt(s, pbar=None):
    #gen = pipe(s, max_new_tokens=1024)

    gen = llm.generate(s, sampling_params)
    
    if pbar is not None:
        pbar.update(1)
    
    return gen[0].outputs[0].text


with open(f"../../prompts/{args.prompt_name}.txt", 'r') as f:
    prompt_template = f.read()

if args.constrained:
    prompt_template = prompt_template.replace('{inst}', instruction)
else:
    prompt_template = prompt_template.replace('{inst}', instruction_un)

df = df.assign(prompt=lambda x: x.claim.apply(lambda y: prompt_template.format(claim=y.strip())))  # create prompts
with tqdm(total=df.shape[0]) as pbar:
    df = df.assign(response=lambda x: x.prompt.apply(lambda y: apply_prompt(y, pbar)))


df.to_csv(os.path.join(args.output_dir, 'output.csv'), index=False)
