import multiprocessing
import os
import argparse
import json
import random
import math
import re

import numpy as np
import pandas as pd
import torch
import transformers
from transformers import pipeline
from tqdm import tqdm
import openai
import logging
from dotenv import load_dotenv

from nltk.tokenize import sent_tokenize
from pyserini.search.lucene import LuceneSearcher

from vllm import SamplingParams
from copy import deepcopy


load_dotenv()


from ast import literal_eval

from utils.file_utils import print_args
from utils.model_utils import load_model_and_tokenizer
from utils.google import link_to_text, GoogleSearcher


logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


def openai_pipeline(prompt, max_new_tokens=256):
    try:
        response = openai.chat.completions.create(
            model=args.query_model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            max_tokens=max_new_tokens,
        ).model_dump()

        return response['choices'][0]['message']['content']
    except openai.BadRequestError:
        print("Bad Request Error")
        return ""

def calculate_transition_scores(sequences):
    logprobs = []
    
    for s in sequences:
        tokens = s.token_ids
        beams = s.logprobs
        temp = []
        for i in range(len(tokens)):
            if tokens[i] in beams[i]: # each beam is a dictionary of tokens
                if type(beams[i][tokens[i]]) is float:
                    temp.append(beams[i][tokens[i]])
                else:
                    temp.append(beams[i][tokens[i]].logprob)
            else:
                print("error")
                temp.append(1)
        logprobs.append(temp)

    return logprobs


def generate_with_scores_vllm(prompt, model, tokenizer, device):
    #inputs = tokenizer(prompt, return_tensors="pt").to(device)
    #input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]
    prompt_token_ids = model.get_tokenizer().encode(prompt)[-4096:]
    prompt = model.get_tokenizer().decode(prompt_token_ids)
 
    sampling_params= SamplingParams(temperature=0.3, top_p=1, max_tokens=64, logprobs=1)
    outputs = model.generate(
        prompt,
        sampling_params=sampling_params,
        use_tqdm=False,
    )
    generated_str = outputs[0].outputs[0].text
    generated_tokens = np.array(outputs[0].outputs[0].token_ids)
    probs = np.exp(calculate_transition_scores(outputs[0].outputs)[0])
 
    return generated_str, generated_tokens, probs, outputs[0].outputs[0].finished



def one_sentence_generation(prompt, model, tokenizer, device, max_new_tokens):
    generated_text = ""
    generated_tokens = []
    new_tokens_length = 0
    probs = []
    generation_length = 0
    prompt = deepcopy(prompt)
    while True:
        prompt += generated_text
        new_tokens_str, new_tokens, new_scores, finish_reason = generate_with_scores_vllm(prompt, model, tokenizer, device)
        generated_tokens.append(new_tokens)
        new_tokens_length += len(new_tokens)
        probs.append(new_scores)

        # print(new_tokens[0])

        generation_length += new_tokens.shape[0]
        if generation_length >= args.max_sentence_tokens:
            break

        sentences = sent_tokenize(new_tokens_str)
        # print("new_tokens_str", new_tokens_str)
        # print(sentences)
        if len(sentences) > 1 or (len(sentences) > 0 and sentences[0][-1] == '.'):
            period_count = sentences[0].count('.')
            for idx, tok in enumerate(new_tokens):
                if tok == '.':
                    period_count -= 1
                if period_count == 0:
                    generated_tokens[-1] = new_tokens[:idx+1]
                    probs[-1] = new_scores[:idx+1]
                    new_tokens_str = sentences[0]
                    generated_text += new_tokens_str
                    finish_reason = False
                    break
            break

        generated_text += new_tokens_str

        if finish_reason == True:
            break
        if new_tokens_length > max_new_tokens:
            break

    generated_tokens = np.concatenate(generated_tokens)
    probs = np.concatenate(probs)
    # generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    return generated_tokens, probs, generated_text, finish_reason


def is_retrieval_triggered(probs, theta):
    return (probs < theta).any()


def masked_sentence_query(tokens, mask, tokenizer):
    return tokenizer.decode(tokens[mask], skip_special_tokens=True)


def get_spans(mask, value):
    start = None
    stop = None
    spans = []
    for i in range(len(mask) - 1):
        if mask[i] == value and start is None:
            start = i
            stop = i + 1
        if start is not None:
            if mask[stop] == value:
                stop += 1
            else:  # span ends
                spans.append((start, stop))
                start = None
    if start is not None:
        spans.append((start, len(mask)))

    return spans


INST_TMPL = "[INST] {prompt} [/INST]"

POUND_TMPL = """### User:
{prompt}

### Assistant:
"""
instruction = """Classify the claim as either "factual" or "misinformation". Do not refuse to answer. Do not engage in explanations and politeness. Only respond with the words "misinformation", "factual", or "unsure". Do not add further context.
"""

instruction_un = """Classify the claim as either "factual" or "misinformation".
"""

def gen_questions_query(gen_text, sentenece_tokens, mask, tokenizer):
    spans = get_spans(mask, False)

    gen_text = gen_text.replace(instruction_un, "")
    gen_text = gen_text.replace(instruction, "")
    gen_text = re.sub(r'<<SYS>>.*<</SYS>>', '', gen_text)
    gen_text = re.sub(r'<\|start_header_id\|>system<\|end_header_id\|>(.|\n)*?<\|eot_id\|>', '', gen_text)

    prompt_template = '{gen_text}\nGiven the above passage, ask a question to which the answer is the term/entity/phrase "{span_text}"'
    if args.query_chat_template == 'inst':
        prompt_template = INST_TMPL.format(prompt=prompt_template)
    elif args.query_chat_template == 'pound':
        prompt_template = POUND_TMPL.format(prompt=prompt_template)

    queries = []
    for span in spans:
        start, stop = span
        span_tokens = sentenece_tokens[start:stop]
        span_text = tokenizer.decode(span_tokens, skip_special_tokens=True)

        prompt = prompt_template.format(gen_text=gen_text, span_text=span_text)
        # TODO: change 128
        if args.query_model_source != 'openai':
            question = query_gen_pipeline(prompt, max_new_tokens=128)[0]["generated_text"][len(prompt):]
        else:
            question = query_gen_pipeline(prompt, max_new_tokens=128)
        queries.append(question)
    
    return queries


def get_google_results(queries):
    contents = ""
    all_res = GoogleSearcher().batch_search(queries)
    web_pages_len = 2048 // len(queries)
    for res in all_res:
        for hit in (res or []):  # to do nothing if it's empty
            if args.web_doc_type == 'snippet' and 'snippet' in hit:
                contents += hit['snippet'] + "\n"
            elif args.web_doc_type == 'page_text':
                text = link_to_text(hit['link']) or ""
                contents += text[:min(len(text), web_pages_len)] + "\n"
            break
    
    return contents


def get_pyserini_results(retriever, queries):
    if args.random_doc:
        rand_doc_id = random.randrange(index_docs_num)
        with open(os.path.join(args.docs_path, f"doc_{rand_doc_id}.json"), 'r') as f:
            return json.load(f)['contents']
    
    all_res = retriever.batch_search(
        queries,
        qids=[str(i) for i in range(len(queries))],
        k=args.num_docs,
        threads=min(4, multiprocessing.cpu_count()),
    )

    max_score = -1
    best_doc = None
    for res in all_res.values():
        for hit in res:
            if hit.score > max_score:
                max_score = hit.score
                with open(os.path.join(args.docs_path, f"doc_{hit.docid}.json"), 'r') as f:
                    best_doc = json.load(f)['contents']
    
    return best_doc


def get_retrieved_doc(retriever, queries, id=None, legal_issues=None):
    contents = ""
    if args.retrieval in ['google', 'mix']:
        if oracle and id is not None and os.path.exists(os.path.join(args.url_dir, str(id))):
            print(len(list(os.listdir(os.path.join(args.url_dir, str(id))))))
            for file in os.listdir(os.path.join(args.url_dir, str(id))):
                with open(os.path.join(args.url_dir, str(id), file), 'r') as f:
                    contents += ''.join(f.readlines())[:2048]
        if len(contents) == 0:
            contents = get_google_results(queries)
        print(len(contents))
        contents = contents[:1000]
        if len(contents) > 0:
            contents = "\n\nHere are some reliable, real-time web search results for the claim:\n" + contents

    if args.retrieval == 'google' or oracle:
        return "", contents
    
    best_doc = ""
    if oracle and legal_issues is not None:
        best_doc = ''.join([(defs[defs['term'] == x] if x in defs['term'] else '') for x in legal_issues])
    else:
        best_doc = get_pyserini_results(retriever, queries)
    
        if len(contents) > 0:
            best_doc = (best_doc or "") + contents


    print(len(best_doc), len(contents))
    return best_doc, contents


def flare_sentence_gen(prompt, model, tokenizer, config, device, retriever, theta, beta, claim=None, row_id=None, legal_issues=None, max_new_tokens=512):
    # temporary sentence generation
    generated_tokens, probs, generated_text, finish_reason = one_sentence_generation(prompt.format(retrieved_doc="", search_results=""), model, tokenizer, device, max_new_tokens)
    # print("generated sentence:", tokenizer.decode(generated_tokens))

    queries_record = []
    if is_retrieval_triggered(probs, theta):
        mask = probs >= beta

        generated_text = tokenizer.decode(generated_tokens)

        queries = []
        # implicit query masking
        if claim is not None:
            queries.append(claim)
        # explicit query questions
        queries += gen_questions_query(prompt + generated_text, generated_tokens, mask, tokenizer)
        queries_record.append({
            "texts": queries,
            "temp_sentence": generated_text,
            "prompt": prompt
        })

        # retrieval
        retrieved_doc, contents = get_retrieved_doc(retriever, queries, row_id, legal_issues)
        


        # print("=== retrieval trigerred")
        # regeneration
        if retrieved_doc is not None:  # in case nothing matches
            retrieved_doc = "Here is some relevant legal context on \"misinformation\":\n" + retrieved_doc
            generated_tokens, _, _, _ = one_sentence_generation(prompt.format(retrieved_doc=retrieved_doc, search_results=contents), model, tokenizer, device, max_new_tokens)
            # print("regenerated sentence:", tokenizer.decode(generated_tokens))
    
    # print("*"*40)
    return generated_tokens, queries_record


def flare_generation(prompt, model, tokenizer, config, device, retriever, claim=None, row_id=None, legal_issues=None):
    generation_str = ""
    generation_length = 0
    queries = []
    while True:
        if generation_length == 0:
            gen_tokens, new_queries = flare_sentence_gen(prompt + generation_str, model, tokenizer, config, device, retriever, 1.0, args.beta, claim, row_id, legal_issues)
        else:
            gen_tokens, new_queries = flare_sentence_gen(prompt + generation_str, model, tokenizer, config, device, retriever, args.theta, args.beta, claim, row_id, legal_issues)
        generation_str += tokenizer.decode(gen_tokens, skip_special_tokens=True)
        queries += new_queries

        # print("gen str:", generation_str)

        generation_length += gen_tokens.shape[0]

        if generation_length >= args.max_new_tokens:
            break
        if gen_tokens[-1] == tokenizer.eos_token_id:
            break

    return generation_str, queries


def generate_outputs(df, model, tokenizer, config, device, retriever, prompt_template):
    # prompt = "World Cup 2022 was the last with 32 teams before the increase to"
    # gen = flare_generation(prompt, model, tokenizer, config, device, retriever)
    # print(gen)

    # def apply_prompt(s, pbar=None):
    #     gen = flare_generation(
    #         s,
    #         model,
    #         tokenizer,
    #         config,
    #         device,
    #         retriever,
    #     )
        
    #     if pbar is not None:
    #         pbar.update(1)
        
    #     return gen

    # prompt_template = 'Claim:\n{claim}\n\nClassify the claim as either "factual" or "misinformation".'

    df = df.assign(prompt=lambda x: x.claim.apply(lambda y: prompt_template.format(claim=y.strip())))  # create prompts

    responses = []
    if os.path.exists(os.path.join(args.output_dir, 'dump.txt')):
        with open(os.path.join(args.output_dir, 'dump.txt'), 'r') as f:
            data = ''.join(f.readlines()).split('\n--------------&&&&&&*--------\n')

        print(len(data))
        data = {x.split('\n********************------------\n')[0]: x.split('\n********************------------\n')[1] for x in data if '********************------------' in x}
        for i in range(len(df)):
            d = str(df.iloc[i]['id'])
            if d in data:
                responses.append(literal_eval(data[str(df.iloc[i]['id'])]))
            else:
                responses.append('')
    print(sum([len(x) for x in responses]))
    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        if idx >= len(responses) or responses[idx] == '':
            gen = flare_generation(
                row['prompt'],
                model,
                tokenizer,
                config,
                device,
                retriever,
                row['claim'],
                row['id'],
                (row['legal_issues'] if not pd.isna(row['legal_issues']) else [])
            )
            with open(os.path.join(args.output_dir, 'dump.txt'), 'a') as f:
                f.write(str(row['id']) + '\n********************------------\n')
                f.write(str(gen))
                f.write('\n--------------&&&&&&*--------\n')
            if idx >= len(responses):

                responses.append(gen)
            else:
                responses[idx] = gen

    gen_df = pd.DataFrame(responses, columns=['response', 'queries'])
    df = pd.concat([df, gen_df], axis=1)
    df.to_csv(os.path.join(args.output_dir, "output.csv"), index=False)


def main(args):
    args.output_dir = os.path.join(args.output_dir, f"prompt={args.prompt_name}/constrained={args.constrained}_theta={args.theta}_beta={args.beta}")
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    print_args(args, output_dir=args.output_dir)

    model, tokenizer, config, device = load_model_and_tokenizer(
        args.model_name, model_parallelism=args.model_parallelism, cache_dir=args.cache_dir, auth_token=args.auth_token, num_gpus=args.num_gpus
    )
    # using the same model for question generation for now
    global query_gen_pipeline
    if args.query_model_source == 'same':
        query_gen_pipeline = model
    elif args.query_model_source == 'hf':
        query_gen_pipeline = model
    elif args.query_model_source == 'openai':
        query_gen_pipeline = openai_pipeline

    # Model context size (e.g., 1024 for GPT-2)
    max_length = args.max_length
    model_max_length = config.n_positions if hasattr(config, "n_positions") else config.max_position_embeddings
    if max_length is None or max_length > model_max_length:
        max_length = model_max_length
    print("Max context length:", max_length)

    global defs
    defs = pd.read_csv(args.defs_path)

    global oracle
    oracle = args.oracle

    searcher = None
    global index_docs_num
    if args.retrieval in ['bm25', 'mix']:
        searcher = LuceneSearcher(args.index_path)
        index_docs_num = len([name for name in os.listdir(args.docs_path) if os.path.isfile(os.path.join(args.docs_path, name))])

    dataset_df = pd.read_csv(args.dataset_path)
    if args.chunk >= 0:# splittig the dataset into three parts for longer experiments
        start = int(math.ceil(len(dataset_df)/3))*args.chunk
        end = min(int(math.ceil(len(dataset_df)/3))*(args.chunk+1), len(dataset_df))
        dataset_df = dataset_df.iloc[range(start,end)]

    with open(f"../prompts/{args.prompt_name}.txt", 'r') as f:
        prompt_template = f.read()

    if args.constrained:
        prompt_template = prompt_template.replace('{inst}',instruction)
    else:
        prompt_template = prompt_template.replace('{inst}', instruction_un)

    transformers.logging.set_verbosity_error()

    generate_outputs(dataset_df, model, tokenizer, config, device, searcher, prompt_template)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", type=str)

    parser.add_argument("--url_dir", type=str, default='../dataset/annotations/downloaded_urls/')
    # Model params
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--model_parallelism", action="store_true")
    parser.add_argument("--auth_token", type=str, default=None)
    parser.add_argument("--query_model_source", type=str, default='same', choices=['same', 'hf', 'openai'])
    parser.add_argument("--query_model_name", type=str, required=False)
    parser.add_argument("--query_chat_template", choices=['inst', 'pound'], default=None)

    # Dataset params
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--defs_path", type=str, default='../dataset/definitions.csv')
    parser.add_argument("--chunk", type=int, default=-1)
    parser.add_argument("--constrained", action="store_true")

    # retrieval params
    parser.add_argument("--retrieval", choices=['bm25', 'google', 'mix'], default='bm25')
    parser.add_argument("--web_doc_type", choices=['snippet', 'page_text'], default='snippet')
    parser.add_argument("--index_path", type=str, required=False)
    parser.add_argument("--docs_path", type=str, required=False)
    parser.add_argument("--random_doc", action="store_true", help="picking random doc in retrieval step for ablations")
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--num_docs", type=int, default=1)
    parser.add_argument("--theta", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.4)

    # Generation params
    parser.add_argument("--max_sentence_tokens", default=256)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--prompt_name", type=str, required=True)

    args = parser.parse_args()

    main(args)
