
export OPENAI_API_KEY=sk-proj-SkfAcVPF3BUChO2Ybgp0T3BlbkFJRQ4zXl3KUc748JrQikMb;


python gen_outputs_openai.py     --dataset_path ../dataset/annotations/dataset.csv     --model_name gpt-4o   --output_dir experiments/gpt4o-google   --model_parallelism   --docs_path ../dataset/definitions/split_docs/   --index_path ../dataset/definitions/index/    --prompt_name basic_retrieval_openai  --max_new_tokens 128   --theta 0.5 --retrieval google  --web_doc_type snippet

python gen_outputs_openai.py --dataset_path ../dataset/annotations/dataset.csv --model_name gpt-4o --output_dir experiments/gpt4o-google-mix --model_parallelism --docs_path ../dataset/definitions/split_docs/ --index_path ../dataset/definitions/index/ --prompt_name basic_retrieval_openai --max_new_tokens 128  --theta 0.5  --retrieval mix --web_doc_type snippet

python gen_outputs_openai.py     --dataset_path ../dataset/annotations/dataset.csv     --model_name gpt-4o   --output_dir experiments/gpt4o   --model_parallelism   --docs_path ../dataset/definitions/split_docs/   --index_path ../dataset/definitions/index/    --prompt_name basic_retrieval_openai  --max_new_tokens 128   --theta 0.5 --retrieval bm25  --web_doc_type snippet

