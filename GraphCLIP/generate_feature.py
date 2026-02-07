import sys
sys.path.append("/home/shenghua/Graph-Foundation-Library")

import os
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from root import ROOT_DIR
from data_provider import *
from data_provider.data_generator import *
from exp.exp_pretrain_tag import ExpPretrain


def mean_pooling(model_output, attention_mask):
    """
    Mean pooling that takes the attention mask into account.
    """
    token_embeddings = model_output  # [B, L, D]
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask_expanded, 1) / torch.clamp(mask_expanded.sum(1), min=1e-9)


def encode_text_graph(
    data,
    name,
    lm_type="tiny",
    chunk_size=1024,
    device=None,
):
    """
    Input:
        data.raw_texts: list of strings

    Output:
        data.x: encoded text embeddings
        Saves data.x to save_path
    """
    dir_path = f"{ROOT_DIR}/datasets/{name}/preprocess/graphclip/"
    os.makedirs(dir_path, exist_ok=True)
    feature_path = dir_path + "node_feature.pt"
    if os.path.exists(feature_path):
        print(f"> Loading existing embeddings from: {feature_path}")
        data.x = torch.load(feature_path)
        return data

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Model choices
    text_ids = {
        "qwen2.5-0.5b": "../Qwen2.5-0.5B-Instruct",
        "qwen2.5-1.5b": "../Qwen2.5-1.5B-Instruct",
        "sbert": "sentence-transformers/multi-qa-distilbert-cos-v1",
        "tiny": "sentence-transformers/all-MiniLM-L6-v2",
        "e5": "intfloat/e5-base-v2",
    }

    model_id = text_ids[lm_type]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device)
    model.eval()

    texts = data.raw_texts
    num_nodes = len(texts)
    print(f"> Encoding {num_nodes} nodes using: {model_id}")

    all_embs = []

    # Encode in chunks
    for i in tqdm(range(0, num_nodes, chunk_size)):
        batch = texts[i: i + chunk_size] # automatically clamps the slice

        batch_tok = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**batch_tok)
            embeddings = mean_pooling(outputs.last_hidden_state, batch_tok["attention_mask"])

        all_embs.append(embeddings.cpu())

    # Concatenate all embeddings
    data.x = torch.cat(all_embs, dim=0)

    # Save to local file
    torch.save(data.x, feature_path)
    print(f"> Saved embeddings to: {feature_path}")

    return data

class ExpPretrainGraphCLIP(ExpPretrain):
    def __init__(self, args, pretrain_dict):
        super(ExpPretrainGraphCLIP, self).__init__(args, pretrain_dict)
        if args.compress_function == "none" :
            self.pretrain_dict = self._build_encoded_data_dict()
        else:
            self.pretrain_dict = self._get_compressed_data_dict()
        self.save_path = self._create_save_path() + f"_{args.compress_function}"

    def _build_encoded_data_dict(self):
        data_dict = {}
        for name, data in self._get_data_dict(is_text=True, need_y=False).items():
            data = encode_text_graph(
                data=data,
                name=name,
                lm_type="tiny",
                chunk_size=2048,
                device=self.device,
            )
            data_dict[name] = data
        self._print_main("Finished generating node features for all datasets.")
        return data_dict


if __name__ == "__main__":
    # your can run this part in tools.ipynb
    class Args:
        def __init__(self, task_name):
            self.model = "graphclip"
            self.model_id = "exp3"
            self.task_name = task_name
            self.pattern = "simple"
            self.compress_function = "none"
            self.cache_compress = False
            self.seed = 0
            self.is_logging = False
            self.num_workers = 0
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    args = Args(task_name="pretrain")
    pretrain_exps = {
        "exp1": pretrain,
        "exp2": pretrain,
        "exp3": pretrain_exp3,
        "exp4": pretrain_exp4,
    }
    pretrain_dict = pretrain_exps[args.model_id]
    ExpPretrainGraphCLIP(args, pretrain_dict)
