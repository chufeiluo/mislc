import os
import argparse

import json
import numpy as np
import pandas as pd
import torch
import transformers
from tqdm import tqdm
from pyserini.search.lucene import LuceneSearcher
import openai
import logging
from dotenv import load_dotenv

from vllm import SamplingParams

from utils.file_utils import print_args
from utils.model_utils import load_model_and_tokenizer


load_dotenv()

logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


def get_doc(retriever, query):
    hits = retriever.search(
        query,
        k=1,
    )

    for hit in hits:  # if nothing hit
        doc_id = hit.docid
        with open(os.path.join(args.docs_path, f"doc_{doc_id}.json"), 'r') as f:
            return json.load(f)['contents']

instruction = """Classify the claim as either "factual" or "misinformation". Do not refuse to answer. Do not engage in explanations and politeness. Only respond with the words "misinformation", "factual", or "unsure". Do not add further context.
"""

instruction_un = """Classify the claim as either "factual" or "misinformation".
"""

def ralm_generation(prompt, stride, max_query_length, retriever, model_source, model=None, tokenizer=None, device=None):
    sampling_params = SamplingParams(max_tokens=stride, temperature=0.3)

    generated_text = ""
    generated_tokens = []
    generation_length = 0
    while True:
        query_len = min(len(prompt + generated_text), max_query_length * 4 // 3 * 5)
        query = (prompt + generated_text)[-(query_len):]
        doc_str = get_doc(retriever, query)
        if doc_str is not None:
            doc_str = "Here is some relevant legal context on \"misinformation\":\n" + doc_str.strip()
        # doc_str_len = len(doc_str) if doc_str is not None else 0

        if doc_str is not None:
            # print("retrieved")
            input_str = prompt.format(retrieved_doc=doc_str)
        else:
            input_str = prompt.format(retrieved_doc="")
        
        if model_source == 'hf':
            input_str += generated_text
            # print(input_str)
            # inputs = tokenizer(input_str, return_tensors="pt").to(device)
            # input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]

            outputs = model.generate(
                input_str,
                sampling_params=sampling_params,
                use_tqdm=False,
            )
            # generated_tokens.append(new_tokens)
            # generated_text += tokenizer.decode(new_tokens[0], skip_special_tokens=True)
            generated_text += outputs[0].outputs[0].text

            # print(prompt)

            new_tokens = outputs[0].outputs[0].token_ids
            generation_length += len(new_tokens)
            if len(new_tokens) == 0:
                break
            if new_tokens[-1] == tokenizer.eos_token_id:
                break
        else:
            try:
                response = openai.chat.completions.create(
                    model=args.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": 'You will receive a claim and maybe some relevant legal literature on misinformation. Classify the claim as either "factual" or "misinformation"',
                        },
                        {
                            "role": "user",
                            "content": input_str,
                        },
                        {
                            "role": "assistant",
                            "content": generated_text,
                        },
                    ],
                    max_tokens=stride,
                    frequency_penalty=1,
                    presence_penalty=1,
                ).model_dump()
            except openai.BadRequestError:
                print("Bad Request Error")
                break

            finish_reason = response['choices'][0]['finish_reason']
            # print("==== gen text:", generated_text)
            generated_text += " " + response['choices'][0]['message']['content']
            # print(response['choices'][0]['message'])

            # print("==== prompt:", prompt, "====")
            # print("==== gen text:", generated_text)
            # print("=" * 20)
            
            generation_length += stride

            # prompt += " " + generated_text
            if finish_reason != 'length':
                break
        
        if generation_length >= args.max_new_tokens:
                break
        
    # generated_tokens = torch.cat(generated_tokens, dim=1)
    # return generated_tokens[0]
    return prompt + generated_text


def generate_outputs(
        df,
        prompt_template,
        retriever,
        stride=4,
        query_length=32,
        model=None,
        tokenizer=None,
        device=None,
        output_dir=None,
        # retrieval_max_length=256,
):


    def apply_prompt(s, pbar=None):
        gen = ralm_generation(
            s,
            stride,
            query_length,
            retriever,
            args.model_source,
            model,
            tokenizer,
            device,
        )
        
        if pbar is not None:
            pbar.update(1)
        
        return gen

    # prompt_template = 'Claim:\n{claim}\n\nClassify the claim as either "factual" or "misinformation".'

    df = df.assign(prompt=lambda x: x.claim.apply(lambda y: prompt_template.format(claim=y.strip())))  # create prompts
    with tqdm(total=df.shape[0]) as pbar:
        df = df.assign(response=lambda x: x.prompt.apply(lambda y: apply_prompt(y, pbar)))

    df.to_csv(os.path.join(output_dir, "output.csv"), index=False)


def main(args):
    args.output_dir = os.path.join(args.output_dir, f"prompt={args.prompt_name}", 'constrained')
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    print_args(args, output_dir=args.output_dir)

    if args.model_source == 'hf':
        model, tokenizer, config, device = load_model_and_tokenizer(
            args.model_name, model_parallelism=args.model_parallelism, cache_dir=args.cache_dir, auth_token=args.auth_token
        )

        # Model context size (e.g., 1024 for GPT-2)
        max_length = args.max_length
        model_max_length = config.n_positions if hasattr(config, "n_positions") else config.max_position_embeddings
        if max_length is None or max_length > model_max_length:
            max_length = model_max_length
        print("Max context length:", max_length)

        transformers.logging.set_verbosity_error()

    searcher = LuceneSearcher(args.index_path)

    dataset_df = pd.read_csv(args.dataset_path)

    with open(f"../prompts/{args.prompt_name}.txt", 'r') as f:
        prompt_template = f.read()

    prompt_template = prompt_template.replace('{inst}', instruction)

    if args.model_source == 'hf':
        generate_outputs(
            dataset_df,
            prompt_template,
            searcher,
            args.stride,
            args.query_length,
            model,
            tokenizer,
            device,
            args.output_dir,
        )
    else:
        generate_outputs(
            dataset_df,
            prompt_template,
            searcher,
            args.stride,
            output_dir=args.output_dir,
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", type=str)

    # Model params
    parser.add_argument("--model_source", choices=['hf', 'openai'], type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--model_parallelism", action="store_true")
    parser.add_argument("--auth_token", type=str, default=None)

    # Dataset params
    parser.add_argument("--dataset_path", type=str, required=True)

    # retrieval params
    parser.add_argument("--index_path", type=str, required=True)
    parser.add_argument("--docs_path", type=str, required=True)
    parser.add_argument("--query_length", type=int, default=32)

    # generation params
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--prompt_name", type=str, required=True)

    args = parser.parse_args()

    main(args)
