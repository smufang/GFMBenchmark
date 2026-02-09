import sys
sys.path.append("/GFM") # your project root directory

from GraphCLIP.llm_prompts import pretrain_categories, pretrain_prompts
from GraphCLIP.utils.args import Arguments
import torch
from xml.etree.ElementTree import Element, SubElement, tostring
import xml.dom.minidom
from collections import defaultdict
import json
from tqdm import tqdm
import time
import re
import os

from GraphCLIP.data.sampling import pyg_random_walk
from torch_geometric.utils import is_undirected
from exp.exp_pretrain_tag import ExpPretrain
from data_provider import *
from data_provider.data_generator import *
from root import ROOT_DIR
import tiktoken


def count_tokens_messages(messages):
    """
    Count tokens for a list of message lists.
    Args:
        messages: List of message lists, where each message list contains
                 [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
    """
    encoding = tiktoken.get_encoding("o200k_base")
    total_tokens = 0
    
    for message_list in messages:
        # Each message_list is [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
        for message in message_list:
            total_tokens += len(encoding.encode(message["content"]))
    
    return total_tokens


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
    node_index, edge_index, edge_type, raw_texts, relation_texts, is_undirected=True
):
    """
    Convert a PyG data object into a GraphML string.
    Args:
        node_index (List[int]): List of node indices in the subgraph.
        edge_index (Tensor): Edge index tensor of shape [2, num_edges].
        edge_type (List[Set[int]]): Set of edge type lists for each edge.
        raw_texts (List[str]): List of raw text strings for each node.
        relation_texts (Dict[int, str]): Mapping from edge type IDs to their textual descriptions.
        is_undirected (bool): Whether the graph is undirected.
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
    for idx, attrs in zip(node_index, node_attrs_list):
        node = SubElement(graph, "node", {"id": f"n{idx}"})
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


def upload_with_retry(client, path, max_retry=5):
    for i in range(max_retry):
        try:
            return client.files.create(
                file=open(path, "rb"),
                purpose="batch"
            )
        except Exception as e:
            print(f"[Retry {i+1}/{max_retry}] upload failed")
            time.sleep(5 * (i + 1))
    raise RuntimeError("Upload failed after retries")


def create_batch_job(jsonl_path, client, endpoint="/v1/chat/completions"):
    print(f"[Batch] Uploading JSONL file: {jsonl_path}")
    # file_obj = client.files.create(file=open(jsonl_path, "rb"), purpose="batch")
    file_obj = upload_with_retry(client, jsonl_path)

    print(f"[Batch] File uploaded: file_id={file_obj.id}")

    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint=endpoint,
        completion_window="24h",  # cost-effective long window
    )

    print(f"[Batch] Batch created: batch_id={batch.id}")
    return batch.id


def write_jsonl_in_chunks(base_path, requests, chunk_size=8192):
    file_paths = []
    file_index = 0
    line_count = 0
    current_path = f"{base_path}_{file_index}.jsonl"
    f = open(current_path, "w", encoding="utf-8")
    file_paths.append(current_path)

    for req in requests:
        f.write(json.dumps(req, ensure_ascii=False) + "\n")
        line_count += 1

        # If current file hits chunk size → switch to a new file
        if line_count >= chunk_size:
            f.close()
            file_index += 1
            line_count = 0
            current_path = f"{base_path}_{file_index}.jsonl"
            f = open(current_path, "w", encoding="utf-8")
            file_paths.append(current_path)

    f.close()
    return file_paths


def merge_batch_summary(item_list_path, batch_id, summary_path, client):
    with open(item_list_path, "r", encoding="utf-8") as f:
        item_list = json.load(f)

    batch_info = client.batches.retrieve(batch_id)
    if batch_info.status != "completed":
        raise ValueError(
            f"Batch job {batch_id} is not finished, status: {batch_info.status}"
        )

    output_file_id = batch_info.output_file_id
    result_bytes = client.files.content(output_file_id)
    result_lines = result_bytes.text.splitlines()
    print(f"Loaded {len(result_lines)} results from batch {batch_id}\n"
            f"input_tokens={batch_info.usage.input_tokens}, "
            f"avg={batch_info.usage.input_tokens / batch_info.request_counts.completed:.2f}\n"
            f"output_tokens={batch_info.usage.output_tokens}, "
            f"avg={batch_info.usage.output_tokens / batch_info.request_counts.completed:.2f}\n")

    for line in result_lines:
        res = json.loads(line)
        custom_id = res.get("custom_id")
        idx = int(custom_id.split("-")[-1])
        summary_text = res["response"]["body"]["choices"][0]["message"]["content"]
        item_list[idx]["summary"] = summary_text

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(item_list, f, ensure_ascii=False, indent=4)

    print(f"Merged summaries saved to {summary_path}")


def merge_batch_summaries(item_list_path, batch_ids, summary_path, client):
    # Load item list
    with open(item_list_path, "r", encoding="utf-8") as f:
        item_list = json.load(f)

    # Collect results from each batch
    for batch_id in batch_ids:
        batch_info = client.batches.retrieve(batch_id)
        if batch_info.status != "completed":
            raise ValueError(
                f"Batch job {batch_id} is not finished (status: {batch_info.status})"
            )

        output_file_id = batch_info.output_file_id
        result_bytes = client.files.content(output_file_id)
        result_lines = result_bytes.text.splitlines()
        print(f"Loaded {len(result_lines)} results from batch {batch_id}\n"
                f"input_tokens={batch_info.usage.input_tokens}, "
                f"avg={batch_info.usage.input_tokens / batch_info.request_counts.completed:.2f}\n"
                f"output_tokens={batch_info.usage.output_tokens}, "
                f"avg={batch_info.usage.output_tokens / batch_info.request_counts.completed:.2f}")

        for line in result_lines:
            res = json.loads(line)
            custom_id = res.get("custom_id")
            idx = int(custom_id.split("-")[-1])
            summary_text = res["response"]["body"]["choices"][0]["message"]["content"]
            item_list[idx]["summary"] = summary_text

    # Save merged output
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(item_list, f, ensure_ascii=False, indent=4)
    
    print(f"Merged summaries saved to {summary_path}")


def main_request(args, pretrain_dict, client):
    exp = ExpPretrain(args, pretrain_dict=pretrain_dict)

    output_dir = f"{ROOT_DIR}/GraphCLIP/api/"
    os.makedirs(output_dir, exist_ok=True)
    save_path = output_dir + "output_dict.json"

    if os.path.exists(save_path):
        with open(save_path, "r", encoding="utf-8") as f:
            output_dict = json.load(f)
    else:
        output_dict = {}
    start_time = time.time()
    for name, data in exp._get_data_dict(is_text=True, need_y=False).items():
        exp._print_main(f"===== Start summarizing {name} =====")
        exp._write_log(f"===== Start summarizing {name} =====")
        dir_path = f"{ROOT_DIR}/datasets/{name}/preprocess/graphclip/"
        os.makedirs(dir_path, exist_ok=True)

        summary_path = dir_path + "summary.json"
        if os.path.exists(summary_path):
            exp._print_main(f"Summary file ({name}) already exists at {summary_path}, skipping...")
            continue
        
        item_path = dir_path + "item_list.json"
        requests_path_base = dir_path + "request"
        # Check if request JSONL files already exist
        existing_jsonl_files = sorted(
            [f for f in os.listdir(dir_path) if f.startswith("request_") and f.endswith(".jsonl")],
            key=lambda x: int(x.split("_")[1].split(".")[0])  # sort by index
        )
        if existing_jsonl_files:
            exp._print_main(f"Found {len(existing_jsonl_files)} existing request files for {name}, skipping generation phase...")
            requests_paths = [os.path.join(dir_path, f) for f in existing_jsonl_files]
            # Count tokens from existing files
            messages = []
            for req_path in requests_paths:
                with open(req_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        req = json.loads(line)
                        messages.append(req['body']['messages'])
            input_tokens = count_tokens_messages(messages)
            exp._print_main(f"Total input tokens: {input_tokens:,}")
        
        else:
            isUndirected = is_undirected(data.edge_index)
            if hasattr(data, "batch") and data.batch is not None:
                # For multi-graph datasets
                if not data.is_sorted(sort_by_row=True):
                    raise ValueError(f"{name} is not sorted by row.")
                subgraph_ids = data.batch.unique()
                all_n_ids = [
                    (data.batch == i).nonzero(as_tuple=True)[0] for i in data.batch.unique()
                ] # [num_graphs, num_nodes_in_graph]
                starts = torch.tensor([n_ids[0].item() for n_ids in all_n_ids])
                ends = torch.tensor([n_ids[-1].item() for n_ids in all_n_ids])
                src = data.edge_index[0]
                start_pos = torch.searchsorted(src, starts)
                end_pos = torch.searchsorted(src, ends, right=True)
                all_edges = [
                    data.edge_index[:, s:e]
                    for s, e in zip(start_pos.tolist(), end_pos.tolist())
                ] # [num_graphs, 2, num_edges_in_graph]
            else:
                subgraph_ids = torch.arange(data.x.shape[0])
                all_n_ids, all_edges = pyg_random_walk(
                    subgraph_ids, data, length=64, restart_prob=0.5, is_undirected=isUndirected
                )  # all_n_ids, all_edges are the original indices

            if data.edge_type.max() > 0:
                row, col = data.edge_index
                edge_map = defaultdict(set)
                for s, t, et in zip(row.tolist(), col.tolist(), data.edge_type.tolist()):
                    edge_map[(s, t)].add(et)  # for duplicate edges

            item_list = []
            messages = []
            requests = []
            for i, seed in tqdm(
                enumerate(subgraph_ids.tolist()), desc=f"Building prompts for {name}"
            ):
                item = {}
                item["id"] = seed
                item["graph"] = all_edges[i].tolist()
                sub_n_id = torch.unique(all_n_ids[i]).tolist() # [num_nodes_in_subgraph]
                sub_e_id = all_edges[i] # [2, num_edges_in_subgraph]
                sub_raw_texts = [data.raw_texts[nid] for nid in sub_n_id]

                if data.edge_type.max() == 0:
                    sub_edge_type = [[0]] * sub_e_id.size(1)
                else:
                    sub_edge_type = []
                    for s, t in zip(sub_e_id[0].tolist(), sub_e_id[1].tolist()):
                        et_set = edge_map[(s, t)]
                        sub_edge_type.append(et_set)

                prompt_str = pretrain_prompts[name].format(
                    seed=seed, categories=pretrain_categories[name]
                )
                graphml = build_graphml_from_data(
                    node_index=sub_n_id,
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
                req = {
                    "custom_id": f"request-{i}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "gpt-5-mini",
                        "messages": message,
                        "verbosity": "low",
                        "reasoning_effort": "minimal",  # "minimal" / "low" / "medium" / "high"
                    },
                }

                item_list.append(item)
                messages.append(message)
                requests.append(req)

            # Save item list to JSON
            with open(item_path, "w", encoding="utf-8") as f:
                json.dump(item_list, f, indent=4, ensure_ascii=False)
                exp._print_main(f"Saved items to {item_path}")
            
            # Write requests to JSONL files in chunks
            requests_paths = write_jsonl_in_chunks(requests_path_base, requests, chunk_size=6000)
            input_tokens = count_tokens_messages(messages)
            exp._print_main(f"Total input tokens: {input_tokens:,}")
            
        batch_ids = []
        for requests_path in requests_paths:
            batch_id = create_batch_job(requests_path, client)
            batch_ids.append(batch_id)
            time.sleep(5)

        output_dict[name] = {
            "batch_ids": batch_ids,
            "item_list_path": item_path,
            "summary_path": summary_path,
            "input_tokens": input_tokens
        }
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(output_dict, f, ensure_ascii=False, indent=2)
        exp._print_main(f"Updated output_dict.json for {name}")

    elapsed = time.time() - start_time
    exp._print_main(f"Elapsed time: {elapsed:.2f} seconds")
    exp._print_main(f"Saved output directory info to {save_path}")
    exp._print_main("All batch jobs created. Please monitor their status and merge summaries once completed.")


def get_fail_requests(client, name):
    save_path = f"{ROOT_DIR}/GraphCLIP/api/output_dict.json"
    with open(save_path, "r", encoding="utf-8") as f:
        output_dir = json.load(f)

    retry_requests = []
    for batch_id in output_dir[name]["batch_ids"]:
        error_file_id = client.batches.retrieve(batch_id).error_file_id
        input_file_id = client.batches.retrieve(batch_id).input_file_id

        if error_file_id is None:
            continue
        fail_ids = set()
        error_bytes = client.files.content(error_file_id)
        for line in error_bytes.text.splitlines():
            error_info = json.loads(line)
            #if error_info['response']['status_code'] >= 500: # server error, need retry
            fail_ids.add(error_info["custom_id"])

        input_bytes = client.files.content(input_file_id)
        for line in input_bytes.text.splitlines():
            req = json.loads(line)
            if req["custom_id"] in fail_ids:
                retry_requests.append(req)
    
    print(f"[INFO] Total retry requests: {len(retry_requests)}")
    retry_path_base = f"{ROOT_DIR}/datasets/{name}/preprocess/graphclip/retry"
    retry_paths = write_jsonl_in_chunks(retry_path_base, retry_requests, chunk_size=6000)
    return retry_paths


def retry_failed_requests(client, name):
    dir_path = f"{ROOT_DIR}/datasets/{name}/preprocess/graphclip/"
    save_path = f"{ROOT_DIR}/GraphCLIP/api/output_dict.json"
    with open(save_path, "r", encoding="utf-8") as f:
        output_dir = json.load(f)

    retry_jsonl_files = sorted(
        [f for f in os.listdir(dir_path) if f.startswith("retry_") and f.endswith(".jsonl")],
        key=lambda x: int(x.split("_")[1].split(".")[0])  # sort by index
    )
    retry_paths = [os.path.join(dir_path, f) for f in retry_jsonl_files]
    batch_ids = []
    for retry_path in retry_paths:
        batch_id = create_batch_job(retry_path, client)
        batch_ids.append(batch_id)
        time.sleep(5)

    output_dir[name]["batch_ids"].extend(batch_ids)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(output_dir, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Updated output_dict.json for {name}, added {len(batch_ids)} batches")
    return batch_ids


def check_summary(name):
    summary_path = f"{ROOT_DIR}/datasets/{name}/preprocess/graphclip/summary.json"
    with open(summary_path, "r", encoding="utf-8") as f:
        item_list = json.load(f)
    missing_list = []
    for idx, item in enumerate(item_list):
        if "summary" not in item:
            missing_list.append(idx)
    print(f"[INFO] {name} Missing summaries: {len(missing_list)}, indices: {missing_list}")


if __name__ == "__main__":
    # your can run this part in tools.ipynb
    class Args:
        def __init__(self, task_name):
            self.model = "graphclip"
            self.model_id = "exp1"
            self.task_name = task_name
            self.pattern = "simple"
            self.compress_function = "none"
            self.cache_compress = False
            self.seed = 0
            self.is_logging = True
            self.num_workers = 0
            self.device = None

    from openai import OpenAI

    api_key = 'your_api_key'
    client = OpenAI(api_key=api_key)

    args = Args(task_name="pretrain")
    pretrain_exps = {
        "exp1": pretrain,
        "exp2": pretrain,
        "exp3": pretrain_exp3,
        "exp4": pretrain_exp4,
    }
    pretrain_dict = pretrain_exps[args.model_id]

    main_request(args, pretrain_dict, client)

    save_path = f"{ROOT_DIR}/GraphCLIP/api/output_dict.json"
    with open(save_path, "r", encoding="utf-8") as f:
        output_dict = json.load(f)

    for name in output_dict.keys():
        merge_batch_summary(
            item_list_path=output_dict[name]["item_list_path"],
            batch_id=output_dict[name]["batch_id"],
            summary_path=output_dict[name]["summary_path"],
            client=client,
        )
