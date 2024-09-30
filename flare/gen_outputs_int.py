import multiprocessing
import os
import argparse
import json

import json
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

from ralm.file_utils import print_args
from ralm.model_utils import load_model_and_tokenizer


load_dotenv()

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


def generate_with_scores(prompt, model, tokenizer, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]

    outputs = model.generate(
        **inputs,
        max_new_tokens=64,
        num_beams=1,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True,
    )
    transition_scores = model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)
    generated_tokens = outputs.sequences[:, input_length:]

    return generated_tokens, transition_scores


def one_sentence_generation(prompt, model, tokenizer, device):
    generated_text = ""
    generated_tokens = []
    transition_scores = []
    generation_length = 0
    while True:
        new_tokens, new_scores = generate_with_scores(prompt + generated_text, model, tokenizer, device)
        new_tokens_str = tokenizer.decode(new_tokens[0], skip_special_tokens=True)
        generated_tokens.append(new_tokens)
        transition_scores.append(new_scores)
        generated_text += new_tokens_str

        # print(new_tokens[0])
        # print(generated_text)

        generation_length += new_tokens.shape[1]
        if generation_length >= args.max_sentence_tokens:
            break

        sentences = sent_tokenize(new_tokens_str)
        # print(new_tokens_str)
        # print(sentences)
        if len(sentences) > 1 or (len(sentences) > 0 and sentences[0][-1] == '.'):
            period_count = sentences[0].count('.')
            for idx, tok in enumerate(new_tokens[0]):
                if tokenizer.decode(tok) == '.':
                    period_count -= 1
                if period_count == 0:
                    generated_tokens[-1] = new_tokens[:, :idx+1]
                    transition_scores[-1] = new_scores[:, :idx+1]
                    break
            break

            # sent_end_idx = new_tokens_str.find(sentences[0]) + len(sentences[0])
            # num_sent_tokens = tokenizer(new_tokens_str[:sent_end_idx], return_tensors="pt").input_ids.shape[1] - 1  # no idea why -1
            # print(new_tokens_str)
            # print(new_tokens)
            # print("="*10)
            # print(new_tokens_str[:sent_end_idx])
            # print(tokenizer(new_tokens_str[:sent_end_idx], return_tensors="pt").input_ids)
            # print("="*10)
            # generated_tokens[-1] = new_tokens[:, :num_sent_tokens]
            # transition_scores[-1] = new_scores[:, :num_sent_tokens]
            # break

        for idx, tok in enumerate(new_tokens[0]):
            if tok == tokenizer.eos_token_id:
                generated_tokens[-1] = new_tokens[:, :idx+1]
                transition_scores[-1] = new_scores[:, :idx+1]
                break
        else:
            continue  # only executed if the inner loop did NOT break
        break  # only executed if the inner loop DID break

        # for idx, (tok, score) in enumerate(zip(new_tokens[0], new_scores[0])):
        #     # print(f"| {tok:5d} | {tokenizer.decode(tok):8s} | {score.cpu().numpy():.3f} | {np.exp(score.cpu().numpy())}")
        #     if tokenizer.decode(tok) == '.' or tok == tokenizer.eos_token_id:
        #         generated_tokens[-1] = new_tokens[:, :idx+1]
        #         transition_scores[-1] = new_scores[:, :idx+1]
        #         break
        # else:
        #     continue  # only executed if the inner loop did NOT break
        # break  # only executed if the inner loop DID break
        

    generated_tokens = torch.cat(generated_tokens, dim=1)
    transition_scores = torch.cat(transition_scores, dim=1)
    probs = torch.exp(transition_scores)
    # generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    return generated_tokens[0], probs[0]


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


def gen_questions_query(gen_text, sentenece_tokens, mask, tokenizer):
    spans = get_spans(mask, False)

    prompt_template = '{gen_text}\nGiven the above passage, ask a question to which the answer is the term/entity/phrase "{span_text}"'
    if args.query_inst:
        prompt_template = f"[INST] {prompt_template} [/INST]"

    queries = []
    for span in spans:
        start, stop = span
        span_tokens = sentenece_tokens[start:stop]
        span_text = tokenizer.decode(span_tokens, skip_special_tokens=True)

        prompt = prompt_template.format(gen_text=gen_text, span_text=span_text)
        # TODO: change 256
        if args.query_model_source != 'openai':
            question = query_gen_pipeline(prompt, max_new_tokens=256)[0]["generated_text"][len(prompt):]
        else:
            question = query_gen_pipeline(prompt, max_new_tokens=256)
        queries.append(question)
    
    return queries

def intermediate_query(gen_text):
    # spans = get_spans(mask, False)
    global defs
    prompt_template = ['Here are some definitions of problematic behaviours.\n']
    prompt_template.extend(['{0}: {1}'.format(x[1].type_name, x[1].definition) for x in defs.iterrows()])
    prompt_template.append('\n*********\nText: {gen_text}\n**********\nIdentify the most relevant problematic behaviour related to the above text, or state "n/a" if none apply. Explain your answer.')
    
    prompt_template = '\n'.join(prompt_template)
    if args.query_inst:
        prompt_template = f"[INST] {prompt_template} [/INST]"

    prompt = prompt_template.format(gen_text=gen_text)
    # TODO: change 256

    # print(prompt)
    
    if args.query_model_source != 'openai':
        question = query_gen_pipeline(prompt, max_new_tokens=256)[0]["generated_text"][len(prompt):]
    else:
        question = query_gen_pipeline(prompt, max_new_tokens=256)
    
    print(question)
    return question

def get_retrieved_doc(retriever, queries):
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


def flare_sentence_gen(prompt, model, tokenizer, config, device, retriever, theta, beta, int_pred=None):
    # temporary sentence generation
    generated_tokens, probs = one_sentence_generation(prompt.format(retrieved_doc=""), model, tokenizer, device)
    # print("generated sentence:", tokenizer.decode(generated_tokens))

    queries_record = []
    if is_retrieval_triggered(probs, theta):
        mask = probs >= beta

        generated_text = tokenizer.decode(generated_tokens)

        queries = []
        # replacing query masking with intermediate query
        queries.append(int_pred)
        # explicit query questions
        #queries += gen_questions_query(prompt + generated_text, generated_tokens, mask, tokenizer)
        queries_record.append({
            "texts": queries,
            "temp_sentence": generated_text,
            "prompt": prompt
        })

        # retrieval
        retrieved_doc = get_retrieved_doc(retriever, queries)

        # print("=== retrieval trigerred")
        # regeneration
        if retrieved_doc is not None:  # in case nothing matches
            retrieved_doc = "Here is some relevant legal context on \"misinformation\":\n" + retrieved_doc
            generated_tokens, ـ = one_sentence_generation(prompt.format(retrieved_doc=retrieved_doc), model, tokenizer, device)
            # print("regenerated sentence:", tokenizer.decode(generated_tokens))
    
    # print("*"*40)
    return generated_tokens, queries_record


def flare_generation(prompt, model, tokenizer, config, device, retriever, int_pred=None):
    generation_str = ""
    generation_length = 0
    queries = []
    while True:
        gen_tokens, new_queries = flare_sentence_gen(prompt + generation_str, model, tokenizer, config, device, retriever, args.theta, args.beta, int_pred)
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
    if 'int_pred' not in df.columns:
        df = df.assign(int_pred= lambda x: x.claim.apply(intermediate_query))

        df.to_csv(args.dataset_path)

    responses = []
    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        gen = flare_generation(
            row['prompt'],
            model,
            tokenizer,
            config,
            device,
            retriever,
            row['int_pred'],
        )
        responses.append(gen)

    gen_df = pd.DataFrame(responses, columns=['response', 'queries'])
    df = pd.concat([df, gen_df], axis=1)
    df.to_csv(os.path.join(args.output_dir, "output.csv"), index=False)


def main(args):
    args.output_dir = os.path.join(args.output_dir, f"prompt={args.prompt_name}_theta={args.theta}_beta={args.beta}")
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    print_args(args, output_dir=args.output_dir)

    model, tokenizer, config, device = load_model_and_tokenizer(
        args.model_name, model_parallelism=args.model_parallelism, cache_dir=args.cache_dir, auth_token=args.auth_token
    )
    # using the same model for question generation for now
    global query_gen_pipeline
    if args.query_model_source == 'same':
        query_gen_pipeline = pipeline('text-generation', model=model, tokenizer=tokenizer)
    elif args.query_model_source == 'hf':
        query_gen_pipeline = pipeline('text-generation', args.query_model_name)
    elif args.query_model_source == 'openai':
        query_gen_pipeline = openai_pipeline

    global defs
    defs = pd.read_csv(args.defs_path)
    # print(defs)
    
    # Model context size (e.g., 1024 for GPT-2)
    max_length = args.max_length
    model_max_length = config.n_positions if hasattr(config, "n_positions") else config.max_position_embeddings
    if max_length is None or max_length > model_max_length:
        max_length = model_max_length
    print("Max context length:", max_length)

    searcher = LuceneSearcher(args.index_path)

    dataset_df = pd.read_csv(args.dataset_path)

    with open(f"../prompts/{args.prompt_name}.txt", 'r') as f:
        prompt_template = f.read()

    transformers.logging.set_verbosity_error()

    generate_outputs(dataset_df, model, tokenizer, config, device, searcher, prompt_template)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", type=str)

    # Model params
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--model_parallelism", action="store_true")
    parser.add_argument("--auth_token", type=str, default=None)
    parser.add_argument("--query_model_source", type=str, default='same', choices=['same', 'hf', 'openai'])
    parser.add_argument("--query_model_name", type=str, required=False)
    parser.add_argument("--query_inst", action="store_true", help="add [INST] [/INST] for query generation")

    # Dataset params
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--defs_path", type=str, required=True)

    # retrieval params
    parser.add_argument("--index_path", type=str, required=True)
    parser.add_argument("--num_docs", type=int, default=1)
    parser.add_argument("--docs_path", type=str, required=True)
    parser.add_argument("--theta", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.4)

    # Generation params
    parser.add_argument("--max_sentence_tokens", default=256)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--prompt_name", type=str, required=True)

    args = parser.parse_args()

    main(args)
