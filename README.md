# mislc
Misinformation with Legal Consequences (MisLC): A New Task Towards Harnessing Societal Harm of Misinformation

Accepted to the Findings of EMNLP 2024

[paper link](https://arxiv.org/abs/2410.03829)

Requirements:
- install python 3.10.x
- run `pip install -e requirements.txt`
- request the `dataset/` directory for the legal database and gold labels
For the dataset, please contact me at chufei.luo@queensu.ca! TBD: release on huggingface datasets

---

## Generating the Pyserini retrieval index
1. unzip split_docs.zip (contact at chufei.luo@queensu.ca for access) to `split_docs/`
2. Run the following command:
```
python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input split_docs \
  --index index \
  --generator DefaultLuceneDocumentGenerator \
  --threads 2 \
  --storePositions --storeDocvectors --storeRaw
```

---

## Important options

run the `gen_outputs.py` script in the respective directory for each method.

`retrieval` - `bm25` is the legal dataset, `google` is web search, `mix` is both

`query_model_source` (FLARE only) - `openai` for openAI models (you need to also specify `--query_model_name`), `same` for using the base model to generate queries

`prompt_name` - corresponds to the name of a .txt file in `prompts/`

`constrained` - include the system prompt to supress refusals

`oracle` - use the annotator-specified source when available

If you have any questions about the other commands in our script, please let me know!

---

## Example commands

Generate results using FLARE retrieval method on only the legal database
```
python gen_outputs.py \
    --dataset_path ../dataset/annotations/dataset.csv \
    --model_name meta-llama/Llama-2-70b-chat-hf \
    --output_dir experiments/Llama-2-70b-chat-hf/query=gpt3.5 \
    --query_model_source openai \
    --query_model_name gpt-3.5-turbo \
    --model_parallelism \
    --retrieval bm25 \
    --web_doc_type snippet \
    --docs_path ../dataset/definitions/split_docs/ \
    --index_path ../dataset/definitions/index/ \
    --prompt_name basic_inst_retrieval \
    --max_new_tokens 128 \
    --max_length 4096 \
    --theta 0.5 \
```

