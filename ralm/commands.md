# mkl

```bash
conda install -c intel mkl
```

fix `.so` problem: https://stackoverflow.com/a/67479054

# wikipedia data

## prepare data

```bash
python prepare_retrieval_data.py \
--retrieval_type sparse \
--tokenizer_name gpt2 \
--max_length 1024 \
--dataset_path wikitext \
--dataset_name wikitext-103-v1 \
--dataset_split validation \
--index_name wikipedia-dpr \
--forbidden_titles_path ralm/retrievers/wikitext103_forbidden_titles.txt \
--stride 4 \
--output_file retrieve \
--num_tokens_for_query 32 \
--num_docs 16 
```

with slurn:
```bash
s_srun $(which python) prepare_retrieval_data.py --retrieval_type sparse --tokenizer_name gpt2 --max_length 1024 --dataset_path wikitext --dataset_name wikitext-103-v1 --dataset_split validation --index_name wikipedia-dpr --forbidden_titles_path ralm/retrievers/wikitext103_forbidden_titles.txt --stride 4 --output_file retrieval --num_tokens_for_query 32 --num_docs 16
```

## eval w/o retreival

```bash
python eval_lm.py \
--model_name gpt2 \
--dataset_path wikitext \
--dataset_name wikitext-103-v1 \
--dataset_split validation \
--output_dir test \
--stride 4 \
--max_length 1024 
```

with slurn:
```bash
a_srun $(which python) eval_lm.py --model_name gpt2 --dataset_path wikitext --dataset_name wikitext-103-v1 --dataset_split validation --output_dir test --stride 4 --max_length 1024
```

## eval w/ retreival

```bash
python eval_lm.py \
--model_name gpt2 \
--dataset_path wikitext \
--dataset_name wikitext-103-v1 \
--dataset_split validation \
--output_dir test \
--stride 4 \
--max_length 1024 \
--model_parallelism \
--retrieved_file retrieval
```

# mumin data

## building index

```bash
python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input collection_json \
  --index indexes/mumin \
  --generator DefaultLuceneDocumentGenerator \
  --threads 16 \
  --storePositions --storeDocvectors --storeRaw
```

## prepare data

```bash
python prepare_retrieval_data.py \
--retrieval_type sparse \
--tokenizer_name gpt2 \
--max_length 1024 \
--load_from file \
--dataset_path "../mumin/mumin-small.txt" \
--index_name "../mumin/indexes/mumin" \
--stride 4 \
--output_file "retrieval_data/mumin-small.json" \
--num_tokens_for_query 32 \
--num_docs 16 
```

## eval w/o retreival

```bash
python eval_lm.py \
--model_name gpt2 \
--load_from file \
--dataset_path "../mumin/mumin-small.txt" \
--output_dir experiments/mumin_wo_retrival \
--stride 4 \
--max_length 1024 
```

## eval w/ retreival

```bash
python eval_lm.py \
--model_name gpt2 \
--load_from file \
--dataset_path "../mumin/mumin-small.txt" \
--output_dir experiments/mumin_w_retrival_default \
--stride 4 \
--max_length 1024 \
--model_parallelism \
--retrieved_file retrieval_data/mumin-small.json
```

## eval classification

```bash
huggingface-cli login
```

```bash
python eval_classification.py \
--model_name meta-llama/Llama-2-13b-chat-hf \
--output_dir experiments/mumin/llama2-13b-chat \
--stride 4 --model_parallelism \
--retrieved_file ../mumin/mumin-small.json
```