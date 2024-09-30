import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
from huggingface_hub import login

from vllm import LLM

def load_tokenizer(model_name):
    if "Llama-2" in model_name:
        return LlamaTokenizer.from_pretrained(model_name)
    return AutoTokenizer.from_pretrained(model_name)


def load_model_and_tokenizer(model_name, model_parallelism=False, cache_dir=None, auth_token=None, max_len=4096, num_gpus=1):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_count = torch.cuda.device_count()

    config = AutoConfig.from_pretrained(model_name)
    model_args = {}
    if cache_dir is not None:
        model_args["cache_dir"] = cache_dir
    if model_parallelism:
        model_args["device_map"] = "auto"
        model_args["low_cpu_mem_usage"] = True
    if hasattr(config, "torch_dtype") and config.torch_dtype is not None:
        model_args["torch_dtype"] = config.torch_dtype
    if auth_token is not None:
        model_args["use_auth_token"] = auth_token

    model = LLM(model=model_name, tensor_parallel_size=num_gpus, download_dir=cache_dir, dtype='half', max_num_seqs=1, gpu_memory_utilization=0.95, max_model_len=max_len)

    # model = AutoModelForCausalLM.from_pretrained(model_name, **model_args).eval()
    # if not model_parallelism:
    #     model = model.to(device)
    tokenizer = load_tokenizer(model_name)

    # if device_count > 1 and not model_parallelism:
    #     model = torch.nn.DataParallel(model)

    return model, tokenizer, config, device
