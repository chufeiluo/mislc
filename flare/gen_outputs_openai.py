import multiprocessing
import os
import argparse
import json
import random
import math
from copy import deepcopy

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
import re

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
            model=args.model_name,
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


def generate_with_scores(prompt):
    outputs = openai.chat.completions.create(
        model=args.model_name,
        messages=prompt,
        logprobs=True,
        # temperature=0.1,
        max_tokens=64,
    ).model_dump()

    generated_str = outputs['choices'][0]['message']['content']
    generated_tokens = np.array([o['token'] for o in outputs['choices'][0]['logprobs']['content']])
    probs = np.exp([o['logprob'] for o in outputs['choices'][0]['logprobs']['content']])

    return generated_str, generated_tokens, probs, outputs['choices'][0]['finish_reason']


def one_sentence_generation(prompt):
    generated_text = ""
    generated_tokens = []
    probs = []
    generation_length = 0
    prompt = deepcopy(prompt)
    while True:
        prompt[-1]['content'] += generated_text
        new_tokens_str, new_tokens, new_scores, finish_reason = generate_with_scores(prompt)
        generated_tokens.append(new_tokens)
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
                    finish_reason = 'sentence'
                    break
            break

        generated_text += new_tokens_str

        if finish_reason == 'stop':
            break

    generated_tokens = np.concatenate(generated_tokens)
    probs = np.concatenate(probs)
    # generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    return generated_tokens, probs, generated_text, finish_reason


def is_retrieval_triggered(probs, theta):
    return (probs < theta).any()


def masked_sentence_query(tokens, mask):
    return "".join(tokens[mask])


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


instruction = """Classify the claim as either "factual" or "misinformation". Do not refuse to answer. Do not engage in explanations and politeness. Only respond with the words "misinformation", "factual", or "unsure". Do not add further context.
"""
def gen_questions_query(gen_text, sentenece_tokens, mask):
    # print(mask)
    spans = get_spans(mask, False)
    # print("spans", spans)

    gen_text = gen_text.replace(instruction, "")
    gen_text = re.sub(r'<<SYS>>.*<</SYS>>', '', gen_text)

    prompt_template = '{gen_text}\nGiven the above passage, ask a question to which the answer is the term/entity/phrase "{span_text}"'

    queries = []
    for span in spans:
        start, stop = span
        span_tokens = sentenece_tokens[start:stop]
        span_text = "".join(span_tokens)

        prompt = prompt_template.format(gen_text=gen_text, span_text=span_text)
        # print("query prompt", prompt)
        # TODO: change 256
        question = query_gen_pipeline(prompt, max_new_tokens=256)
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


def get_retrieved_doc(retriever, queries):
    contents = ""
    if args.retrieval in ['google', 'mix']:
        contents = get_google_results(queries)
        
        if len(contents) > 0:
            contents = "\n\nWeb search results for the claim:\n" + contents
    if args.retrieval == 'google':
        return contents
    
    best_doc = get_pyserini_results(retriever, queries)
    
    if len(contents) > 0:
        best_doc = (best_doc or "") + contents
    return best_doc


def flare_sentence_gen(prompt, retriever, theta, beta):
    # print(prompt)
    # temporary sentence generation
    tmp_prompt = deepcopy(prompt)
    tmp_prompt[0]['content'] = prompt[0]['content'].format(retrieved_doc="")
    generated_tokens, probs, generated_text, finish_reason = one_sentence_generation(tmp_prompt)
    # print("generated sentence:", generated_text)
    # print("generated tokens:", generated_tokens)
    # print("probs", probs)

    queries_record = []
    if is_retrieval_triggered(probs, theta):
        mask = probs >= beta

        # generated_text = tokenizer.decode(generated_tokens)

        queries = []
        # implicit query masking
        queries.append(masked_sentence_query(generated_tokens, mask))
        # explicit query questions
        passage_text = tmp_prompt[0]['content'] + "\n\n" + tmp_prompt[-1]['content'] + " " + generated_text
        # print("passage text", passage_text)
        queries += gen_questions_query(passage_text, generated_tokens, mask)
        queries_record.append({
            "texts": queries,
            "temp_sentence": generated_text,
            "prompt": prompt
        })
        # print("=== retrieval trigerred", flush=True)
        # print("queries", queries)

        # retrieval
        retrieved_doc = get_retrieved_doc(retriever, queries)

        # regeneration
        if retrieved_doc is not None:  # in case nothing matches
            retrieved_doc = "Here is some relevant legal context on \"misinformation\":\n" + retrieved_doc
            tmp_prompt = deepcopy(prompt)
            tmp_prompt[0]['content'] = prompt[0]['content'].format(retrieved_doc=retrieved_doc)
            generated_tokens, ـ, generated_text, finish_reason = one_sentence_generation(tmp_prompt)
            # print("regenerated sentence:", generated_text)
    
    # print("*"*40)
    # print("~~~", generated_text)
    return generated_tokens, queries_record, generated_text, finish_reason


def flare_generation(prompt, retriever):
    gen_str = ""
    generation_length = 0
    queries = []
    prompt = [
        {
            "role": "system",
            "content": 'You will receive a claim and maybe some relevant legal literature on misinformation. Classify the claim as either "factual" or "misinformation"',
        },
        {
            "role": "user",
            "content": prompt,
        },
        {
            "role": "assistant",
            "content": "",
        },
    ]
    while True:
        gen_tokens, new_queries, gen_str, finish_reason = flare_sentence_gen(prompt, retriever, args.theta, args.beta)
        prompt[-1]['content'] += " " + gen_str
        prompt[-1]['content'] = prompt[-1]['content'].lstrip()
        queries += new_queries

        # print("gen str:", generation_str)

        # print(generation_length)
        # print(finish_reason)
        generation_length += gen_tokens.shape[0]

        if generation_length >= args.max_new_tokens:
            break
        if finish_reason == 'stop':
            break

    
    gen_response = prompt[-1]['content']
    print(gen_response, flush=True)
    return gen_response, queries


def generate_outputs(df, retriever, prompt_template):
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
                retriever,
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
    args.output_dir = os.path.join(args.output_dir, f"prompt={args.prompt_name}_theta={args.theta}_beta={args.beta}")
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    print_args(args, output_dir=args.output_dir)

    global query_gen_pipeline
    query_gen_pipeline = openai_pipeline

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

    transformers.logging.set_verbosity_error()

    generate_outputs(dataset_df, searcher, prompt_template)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", type=str)

    # Model params
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--model_parallelism", action="store_true")
    parser.add_argument("--auth_token", type=str, default=None)

    # Dataset params
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--chunk", type=int, default=-1)
    # retrieval params
    parser.add_argument("--retrieval", choices=['bm25', 'google', 'mix'], default='bm25')
    parser.add_argument("--web_doc_type", choices=['snippet', 'page_text'], default='snippet')
    parser.add_argument("--index_path", type=str, required=False)
    parser.add_argument("--docs_path", type=str, required=False)
    parser.add_argument("--random_doc", action="store_true", help="picking random doc in retrieval step for ablations")
    parser.add_argument("--num_docs", type=int, default=1)
    parser.add_argument("--theta", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.4)

    # Generation params
    parser.add_argument("--max_sentence_tokens", default=256)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--prompt_name", type=str, required=True)

    args = parser.parse_args()

    main(args)
