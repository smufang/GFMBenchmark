import sys
sys.path.append("/home/shenghua/Graph-Foundation-Library")
from GraphCLIP.llm_prompts import pretrain_categories, pretrain_prompts
from GraphCLIP.utils.args import Arguments
import torch
from xml.etree.ElementTree import Element, SubElement, tostring
import xml.dom.minidom
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from collections import defaultdict
import json
from tqdm import tqdm
import time
import re


def parse_text_to_attrs(text):
    """
    Parse a single raw text string into a list of (key, value) attributes.
    Keys are converted to lowercase. Empty or missing values are kept as empty strings.
    """

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    attrs = []

    for line in lines:
        match = re.match(r"([^:]+):\s*(.*)", line)  # allow empty values after colon
        if match:
            key = match.group(1).strip().rstrip(".").lower()
            value = match.group(2).strip()
        else:
            key = "type"  # default key for lines without colon
            value = line
        attrs.append((key, value))
    return attrs


def build_graphml_from_data(
    edge_index, edge_type, raw_texts, relation_texts, is_undirected=True
):
    """
    Convert a PyG data object into a GraphML string.

    Parameters:
    - data: PyG Data object with raw_texts (nodes) and optional relation_texts (edges)
    - include_edge_text: bool, whether to use relation_texts for edge labels

    Returns:
    - str: formatted GraphML string with UTF-8 declaration
    """
    edge_list = edge_index.t().tolist()

    # ---------------- Build XML root ----------------
    root = Element("graphml")

    # ---------------- Node keys ----------------
    # Collect all unique keys across all nodes
    all_node_keys = set()
    node_attrs_list = []
    for text in raw_texts:
        attrs = parse_text_to_attrs(text)
        node_attrs_list.append(attrs)
        all_node_keys.update(k for k, _ in attrs)

    node_keys = list(all_node_keys)  # preserve order
    node_key_id_map = {k: f"d{i}" for i, k in enumerate(node_keys)}

    for key in node_keys:
        SubElement(
            root,
            "key",
            {
                "id": node_key_id_map[key],
                "for": "node",
                "attr.name": key,
                "attr.type": "string",
            },
        )

    # ---------------- Edge key ----------------
    edge_key_id = f"d{len(node_keys)}"
    SubElement(
        root,
        "key",
        {"id": edge_key_id, "for": "edge", "attr.name": "type", "attr.type": "string"},
    )

    # ---------------- Graph ----------------
    graph = SubElement(
        root,
        "graph",
        {"id": "G", "edgedefault": "undirected" if is_undirected else "directed"},
    )

    # ---------------- Add nodes ----------------
    for i, attrs in enumerate(node_attrs_list):
        node = SubElement(graph, "node", {"id": f"n{i}"})
        for key, value in attrs:
            data_elem = SubElement(node, "data", {"key": node_key_id_map[key]})
            data_elem.text = value

    # ---------------- Add edges ----------------
    for j, (src, tgt) in enumerate(edge_list):
        edge = SubElement(
            graph, "edge", {"id": f"e{j}", "source": f"n{src}", "target": f"n{tgt}"}
        )
        data_elem = SubElement(edge, "data", {"key": edge_key_id})
        if edge_type[j]:
            data_elem.text = ", ".join([relation_texts[et] for et in edge_type[j]])
        else:
            data_elem.text = "to"

    # ---------------- Output XML string ----------------
    xml_bytes = xml.dom.minidom.parseString(
        tostring(root, encoding="utf-8")
    ).toprettyxml(indent="  ", encoding="utf-8")
    return xml_bytes.decode("utf-8")


def generate_summary_vllm(prompts, model, max_tokens=512, temperature=0.8, top_p=0.95):
    stop_token_ids = [151329, 151336, 151338]
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop_token_ids=stop_token_ids,
    )
    outputs = model.generate(prompts, sampling_params)
    return outputs


def main(args):
    config = Arguments().parse_args()

    model_id = "Qwen/Qwen2-72B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = LLM(
        model=model_id,
        max_model_len=15000,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        tensor_parallel_size=4,
        gpu_memory_utilization=0.91,
    )

    # this five imports must be placed after the LLM initialization to avoid CUDA error, especially 'is_undirected'
    from GraphCLIP.data.sampling import pyg_random_walk
    from torch_geometric.utils import is_undirected
    from exp.exp_pretrain_tag import ExpPretrain
    from data_provider import pretrain
    from data_provider.data_generator import ogbn_arxiv

    pretrain = {"ogbn-arxiv": ogbn_arxiv}

    exp = ExpPretrain(args, pretrain_dict=pretrain)

    for name, data in exp._get_data_dict(is_text=True, need_y=False).items():
        exp._print_main(f"===== Start summarizing {name} =====")
        exp._write_log(f"===== Start summarizing {name} =====")
        start_time = time.time()

        isUndirected = is_undirected(data.edge_index)
        if hasattr(data, "batch") and data.batch is not None:
            # For multi-graph datasets
            if not data.is_sorted(sort_by_row=True):
                raise ValueError(f"{name} is not sorted by row.")
            node_ids = data.batch.unique()
            all_n_ids = [
                (data.batch == i).nonzero(as_tuple=True)[0] for i in data.batch.unique()
            ]
            starts = torch.tensor([n_ids[0].item() for n_ids in all_n_ids])
            ends = torch.tensor([n_ids[-1].item() for n_ids in all_n_ids])
            src = data.edge_index[0]
            start_pos = torch.searchsorted(src, starts)
            end_pos = torch.searchsorted(src, ends, right=True)
            all_edges = [
                data.edge_index[:, s:e]
                for s, e in zip(start_pos.tolist(), end_pos.tolist())
            ]
        else:
            node_ids = torch.arange(data.x.shape[0])
            all_n_ids, all_edges = pyg_random_walk(
                node_ids, data, length=64, restart_prob=0.5, is_undirected=isUndirected
            )  # all_n_ids, all_edges are the original indices

        if data.edge_type.max() > 0:
            row, col = data.edge_index
            edge_map = defaultdict(list)
            for s, t, et in zip(row.tolist(), col.tolist(), data.edge_type.tolist()):
                edge_map[(s, t)].append(et)  # for duplicate edges

        item_list = []
        messages = []
        for i, seed in tqdm(
            enumerate(node_ids.tolist()), desc=f"Building prompts for {name}"
        ):
            item = {}
            item["id"] = seed
            item["graph"] = all_edges[i].tolist()
            sub_n_id = torch.unique(all_n_ids[i]).tolist()
            sub_e_id = all_edges[i]
            sub_raw_texts = [data.raw_texts[nid] for nid in sub_n_id]

            if data.edge_type.max() == 0:
                sub_edge_type = [[0]] * sub_e_id.size(1)
            else:
                sub_edge_type = []
                for s, t in zip(sub_e_id[0].tolist(), sub_e_id[1].tolist()):
                    et_list = edge_map[(s, t)]
                    sub_edge_type.append(et_list)

            prompt_str = pretrain_prompts[name].format(
                seed=seed, categories=pretrain_categories[name]
            )
            graphml = build_graphml_from_data(
                edge_index=sub_e_id,
                edge_type=sub_edge_type,
                raw_texts=sub_raw_texts,
                relation_texts=data.relation_texts,
                is_undirected=isUndirected,
            )

            instrcution = prompt_str + graphml
            message = [
                {
                    "role": "system",
                    "content": "You are a helpful language and graph assistant. You are able to understand the graph content that the user provides, and assist the user with a variety of tasks using natural language.",
                },
                {"role": "user", "content": instrcution},
            ]
            messages.append(message)
            item_list.append(item)

        batch = 8
        exp._print_main(f"Start generating summaries with vLLM for {name}")
        exp._write_log(f"Start generating summaries with vLLM for {name}")
        for i in tqdm(range(0, len(messages), batch)):
            if i + batch >= len(messages):
                end = len(messages)
            else:
                end = i + batch
            cur_messages = messages[i:end]
            cur_messages = tokenizer.apply_chat_template(
                cur_messages, tokenize=False, add_generation_prompt=True
            )
            responses = generate_summary_vllm(cur_messages, model, max_tokens=512)

            for j, output in enumerate(responses):
                item_list[i + j]["summary"] = output.outputs[0].text

        output_path = f"GraphCLIP/summary/summary-{name}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(item_list, f, indent=4, ensure_ascii=False)

        elapsed = time.time() - start_time
        exp._write_log(f"Finished summarizing {name} in {elapsed:.2f}s.")
        exp._write_log(f"Summary saved to {output_path}\n")


if __name__ == "__main__":
    class Args:
        def __init__(self, task_name):
            self.model = "graphclip"
            self.model_id = "exp1"
            self.task_name = task_name
            self.input_dim = 128
            self.compress_function = "none"
            self.cache_compress = False
            self.seed = 0
            self.is_logging = True
            self.num_workers = 0
            self.device = None
    main(Args(task_name="other-simple"))
