


python3 gen_outputs_openai.py --dataset_path ../dataset/annotations/dataset.csv --model_name gpt-4o --output_dir experiments/gpt4o-inst-removed-google-3 --model_parallelism  --docs_path ../dataset/definitions/split_docs/   --index_path ../dataset/definitions/index/ --prompt_name basic_retrieval_longanswer_openai  --max_new_tokens 128     	--theta 0.6 	--retrieval google 	--web_doc_type snippet --num_docs 3


python gen_outputs_openai.py     --dataset_path ../dataset/annotations/dataset.csv     --model_name gpt-4o   --output_dir experiments/gpt4o-inst-removed-google   --model_parallelism   --docs_path ../dataset/definitions/split_docs/   --index_path ../dataset/definitions/index/    --prompt_name basic_retrieval_longanswer_openai  --max_new_tokens 128   --theta 0.6 --retrieval google  --web_doc_type snippet 
											     



python gen_outputs_openai.py   --dataset_path ../dataset/annotations/dataset.csv   --model_name gpt-4o   --output_dir experiments/gpt4o-inst-removed-google-mix-3  --model_parallelism   --docs_path ../dataset/definitions/split_docs/  --index_path ../dataset/definitions/index/   --prompt_name basic_retrieval_longanswer_openai   --max_new_tokens 128    --theta 0.6 --retrieval mix --web_doc_type snippet --num_docs 3

python gen_outputs_openai.py --dataset_path ../dataset/annotations/dataset.csv --model_name gpt-4o --output_dir experiments/gpt4o-inst-removed-google-mix --model_parallelism --docs_path ../dataset/definitions/split_docs/ --index_path ../dataset/definitions/index/ --prompt_name basic_retrieval_longanswer_openai --max_new_tokens 128  --theta 0.6  --retrieval mix --web_doc_type snippet

