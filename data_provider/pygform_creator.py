import os
import gzip
import requests
import scipy.io
from collections import defaultdict
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset, HeteroData, Data
from torch_geometric.utils import from_scipy_sparse_matrix, to_undirected, remove_self_loops


class Cora(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['cora.pt']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        url = "https://raw.githubusercontent.com/LechengKong/OneForAll/main/data/single_graph/Cora/cora.pt"
        save_path = os.path.join(self.raw_dir, self.raw_file_names[0])
        print(f"Downloading {url}")
        os.makedirs(self.raw_dir, exist_ok=True)
        r = requests.get(url)
        r.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(r.content)
        print(f"Extracting {save_path}")

    def process(self):
        raw_path = os.path.join(self.raw_dir, self.raw_file_names[0])
        data = torch.load(raw_path, weights_only=False)
        new_texts = []
        for text in data.raw_text:
            parts = text.split(':', 1)
            modified = f'Title:{parts[0]}\nAbstract:{parts[1]}'
            new_texts.append(modified)
        
        data.raw_texts = new_texts
        del data.raw_text
        data_list = [data]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class PubMed(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['pubmed.pt']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        url = "https://raw.githubusercontent.com/LechengKong/OneForAll/main/data/single_graph/Pubmed/pubmed.pt"
        save_path = os.path.join(self.raw_dir, self.raw_file_names[0])
        print(f"Downloading {url}")
        os.makedirs(self.raw_dir, exist_ok=True)
        r = requests.get(url)
        r.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(r.content)
        print(f"Extracting {save_path}")

    def process(self):
        raw_path = os.path.join(self.raw_dir, self.raw_file_names[0])
        data = torch.load(raw_path, weights_only=False)
        data_list = [data]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class Arxiv(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['titleabs.tsv.gz']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        url = "https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv.gz"
        save_path = os.path.join(self.raw_dir, self.raw_file_names[0])
        print(f"Downloading {url}")
        os.makedirs(self.raw_dir, exist_ok=True)
        r = requests.get(url)
        r.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(r.content)
        print(f"Extracting {save_path}")

    def process(self):
        from ogb.nodeproppred import PygNodePropPredDataset
        with patch('builtins.input', return_value='y'):
            dataset = PygNodePropPredDataset(name='ogbn-arxiv', root=self.raw_dir)
        data = dataset[0]

        # Load the mapping file from node index to paper ID
        map_path = os.path.join(self.raw_dir, 'ogbn_arxiv/mapping/nodeidx2paperid.csv.gz')
        df_map = pd.read_csv(map_path, compression='gzip', dtype={'paper id': str})

        # Load the compressed text file (title and abstract) using gzip
        raw_path = os.path.join(self.raw_dir, self.raw_file_names[0])
        with gzip.open(raw_path, 'rt', encoding='utf-8') as f:
            df_text = pd.read_csv(f, sep='\t', names=['MAG_ID', 'title', 'abstract'], dtype={'MAG_ID': str})

        # Merge the mapping dataframe and the text dataframe on paper ID to align node indices with texts
        df = df_map.merge(df_text, how='left', left_on='paper id', right_on='MAG_ID')

        df['title'] = df['title'].fillna('')
        df['abstract'] = df['abstract'].fillna('')
        df['raw_texts'] = 'Title: ' + df['title'].astype(str) + '. \nAbstract: ' + df['abstract'].astype(str)

        data.raw_texts = df['raw_texts'].tolist()
        data.x = data.x.contiguous()
        data_list = [data]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class Genre(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        from tgb.nodeproppred.dataset_pyg import PyGNodePropPredDataset
        with patch('builtins.input', return_value='y'):
            dataset = PyGNodePropPredDataset(name="tgbn-genre", root="datasets/Genre")
        data_list = [dataset.get_TemporalData()]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class Actor(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['Actor.npz']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        url = "https://media.githubusercontent.com/media/honey0219/LLM4HeG/refs/heads/main/dataset/Actor.npz"
        save_path = os.path.join(self.raw_dir, self.raw_file_names[0])
        os.makedirs(self.raw_dir, exist_ok=True)

        print(f"Downloading {url}")
        r = requests.get(url)
        r.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(r.content)
        print(f"Extracting {save_path}")

    def process(self):
        npz_path = os.path.join(self.raw_dir, self.raw_file_names[0])
        data_npz = np.load(npz_path, allow_pickle=True)

        edges = data_npz['edges']  # shape [2, num_edges]
        node_labels = data_npz['node_labels']  # shape [num_nodes]
        node_features = data_npz['node_features']  # shape [num_nodes, num_features]
        node_texts = data_npz['node_texts']  # shape [num_nodes], dtype=object
        label_texts = data_npz['label_texts']  # shape [num_labels], dtype=object
        train_mask = data_npz['train_masks']  # shape [num_nodes], bool
        val_mask = data_npz['val_masks']  # shape [num_nodes], bool
        test_mask = data_npz['test_masks']  # shape [num_nodes], bool

        # to tensor
        edge_index = torch.tensor(edges, dtype=torch.long).contiguous()
        x = torch.tensor(node_features, dtype=torch.float)
        y = torch.tensor(node_labels, dtype=torch.long)
        train_mask = torch.tensor(train_mask, dtype=torch.bool)
        val_mask = torch.tensor(val_mask, dtype=torch.bool)
        test_mask = torch.tensor(test_mask, dtype=torch.bool)

        # to list
        node_texts_list = [text.replace(";", ". \n") for text in node_texts.tolist()]
        label_texts_list = label_texts.tolist()

        data = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            raw_texts=node_texts_list,
            label_names=label_texts_list
        )

        data_list = [data]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class Texas(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['Texas.npz']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        url = "https://media.githubusercontent.com/media/honey0219/LLM4HeG/refs/heads/main/dataset/Texas.npz"
        save_path = os.path.join(self.raw_dir, self.raw_file_names[0])
        os.makedirs(self.raw_dir, exist_ok=True)

        print(f"Downloading {url}")
        r = requests.get(url)
        r.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(r.content)
        print(f"Extracting {save_path}")

    def process(self):
        npz_path = os.path.join(self.raw_dir, self.raw_file_names[0])
        data_npz = np.load(npz_path, allow_pickle=True)

        edges = data_npz['edges']  # shape [2, num_edges]
        node_labels = data_npz['node_labels']  # shape [num_nodes]
        node_features = data_npz['node_features']  # shape [num_nodes, num_features]
        node_texts = data_npz['node_texts']  # shape [num_nodes], dtype=object
        label_texts = data_npz['label_texts']  # shape [num_labels], dtype=object
        train_mask = data_npz['train_masks']  # shape [num_nodes], bool
        val_mask = data_npz['val_masks']  # shape [num_nodes], bool
        test_mask = data_npz['test_masks']  # shape [num_nodes], bool

        # to tensor
        edge_index = torch.tensor(edges, dtype=torch.long).contiguous()
        x = torch.tensor(node_features, dtype=torch.float)
        y = torch.tensor(node_labels, dtype=torch.long)
        train_mask = torch.tensor(train_mask, dtype=torch.bool)
        val_mask = torch.tensor(val_mask, dtype=torch.bool)
        test_mask = torch.tensor(test_mask, dtype=torch.bool)

        # to list
        node_texts_list = [f'Content: {text}' for text in node_texts.tolist()]
        label_texts_list = label_texts.tolist()

        data = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            raw_texts=node_texts_list,
            label_names=label_texts_list
        )

        data_list = [data]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class Wisconsin(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['Wisconsin.npz']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        url = "https://media.githubusercontent.com/media/honey0219/LLM4HeG/refs/heads/main/dataset/Wisconsin.npz"
        save_path = os.path.join(self.raw_dir, self.raw_file_names[0])
        os.makedirs(self.raw_dir, exist_ok=True)

        print(f"Downloading {url}")
        r = requests.get(url)
        r.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(r.content)
        print(f"Extracting {save_path}")

    def process(self):
        npz_path = os.path.join(self.raw_dir, self.raw_file_names[0])
        data_npz = np.load(npz_path, allow_pickle=True)

        edges = data_npz['edges']  # shape [2, num_edges]
        node_labels = data_npz['node_labels']  # shape [num_nodes]
        node_features = data_npz['node_features']  # shape [num_nodes, num_features]
        node_texts = data_npz['node_texts']  # shape [num_nodes], dtype=object
        label_texts = data_npz['label_texts']  # shape [num_labels], dtype=object
        train_mask = data_npz['train_masks']  # shape [num_nodes], bool
        val_mask = data_npz['val_masks']  # shape [num_nodes], bool
        test_mask = data_npz['test_masks']  # shape [num_nodes], bool

        # to tensor
        edge_index = torch.tensor(edges, dtype=torch.long).contiguous()
        x = torch.tensor(node_features, dtype=torch.float)
        y = torch.tensor(node_labels, dtype=torch.long)
        train_mask = torch.tensor(train_mask, dtype=torch.bool)
        val_mask = torch.tensor(val_mask, dtype=torch.bool)
        test_mask = torch.tensor(test_mask, dtype=torch.bool)

        # to list
        node_texts_list = [f'Content: {text}' for text in node_texts.tolist()]
        label_texts_list = label_texts.tolist()

        data = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            raw_texts=node_texts_list,
            label_names=label_texts_list
        )

        data_list = [data]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class Cornell(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['Cornell.npz']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        url = "https://media.githubusercontent.com/media/honey0219/LLM4HeG/refs/heads/main/dataset/Cornell.npz"
        save_path = os.path.join(self.raw_dir, self.raw_file_names[0])
        os.makedirs(self.raw_dir, exist_ok=True)

        print(f"Downloading {url}")
        r = requests.get(url)
        r.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(r.content)
        print(f"Extracting {save_path}")

    def process(self):
        npz_path = os.path.join(self.raw_dir, self.raw_file_names[0])
        data_npz = np.load(npz_path, allow_pickle=True)

        edges = data_npz['edges']  # shape [2, num_edges]
        node_labels = data_npz['node_labels']  # shape [num_nodes]
        node_features = data_npz['node_features']  # shape [num_nodes, num_features]
        node_texts = data_npz['node_texts']  # shape [num_nodes], dtype=object
        label_texts = data_npz['label_texts']  # shape [num_labels], dtype=object
        train_mask = data_npz['train_masks']  # shape [num_nodes], bool
        val_mask = data_npz['val_masks']  # shape [num_nodes], bool
        test_mask = data_npz['test_masks']  # shape [num_nodes], bool

        # to tensor
        edge_index = torch.tensor(edges, dtype=torch.long).contiguous()
        x = torch.tensor(node_features, dtype=torch.float)
        y = torch.tensor(node_labels, dtype=torch.long)
        train_mask = torch.tensor(train_mask, dtype=torch.bool)
        val_mask = torch.tensor(val_mask, dtype=torch.bool)
        test_mask = torch.tensor(test_mask, dtype=torch.bool)

        # to list
        node_texts_list = [f'Content: {text}' for text in node_texts.tolist()]
        label_texts_list = label_texts.tolist()

        data = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            raw_texts=node_texts_list,
            label_names=label_texts_list
        )

        data_list = [data]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class YelpSemRec(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['yelp.mat']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        url = 'https://raw.githubusercontent.com/zzqsmall/SemRec/refs/heads/master/data/yelp.mat'
        save_path = os.path.join(self.raw_dir, 'yelp.mat')
        print(f"Downloading {url}")
        r = requests.get(url)
        r.raise_for_status()
        with open(save_path, 'wb') as f:
            f.write(r.content)
        print(f"Extracting {save_path}")

    def process(self):
        raw_path = os.path.join(self.raw_dir, 'yelp.mat')
        mat = scipy.io.loadmat(raw_path)

        entity = mat['entity'][0]
        entity_names = [str(e[0]) for e in mat['entityName'][0]]
        relation_pairs = mat['relationIdx'] - 1
        relations = mat['relation'][0]

        data = HeteroData()

        for i, name in enumerate(entity_names):
            ids = [row[0].item() for row in entity[i]]
            data[name] = ids

        for i, (src, dst) in enumerate(relation_pairs):
            src_name = entity_names[src]
            dst_name = entity_names[dst]
            relation = from_scipy_sparse_matrix(relations[i])
            data[src_name, dst_name].edge_index = relation[0]
            data[src_name, dst_name].edge_attr = relation[1]

        torch.save(self.collate([data]), self.processed_paths[0])


class Amazon(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        self._raw_folder_path = root
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['feature.txt', 'train.txt', 'valid.txt', 'test.txt']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        base_url = "https://raw.githubusercontent.com/SY1706203/GATNE/master/dataset/amazon/"
        raw_dir = os.path.join(self.root, 'raw')
        os.makedirs(raw_dir, exist_ok=True)
        print(f"Downloading {base_url}")

        for filename in self.raw_file_names:
            filepath = os.path.join(raw_dir, filename)
            if not os.path.exists(filepath):
                print(f"Downloading {filename}")
                url = base_url + filename
                r = requests.get(url)
                r.raise_for_status()
                with open(filepath, 'wb') as f:
                    f.write(r.content)
            else:
                print(f"{filename} already exists, skipping download.")
            print(f"Extracting {filepath}")

    def process(self):
        raw_dir = os.path.join(self._raw_folder_path, 'raw')
        paths = {name: os.path.join(raw_dir, name) for name in self.raw_file_names}

        nodes = self.load_nodes(paths['feature.txt'])
        train_edges = self.load_edges(paths['train.txt'], filter_positive=False)
        valid_edges = self.load_edges(paths['valid.txt'], filter_positive=True)
        test_edges = self.load_edges(paths['test.txt'], filter_positive=True)

        data = self.Amazon_HeteroDataCreate(nodes, train_edges, valid_edges, test_edges)
        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])

    def load_nodes(self, path):
        """Load node features from feature.txt, skipping header line"""
        with open(path, 'r') as f:
            return {
                line.split()[0]: line.split()[1:]
                for i, line in enumerate(f) if i > 0
            }

    def load_edges(self, path, filter_positive=False):
        """Load edges from train/valid/test files with optional positive filtering"""
        edges = defaultdict(list)
        with open(path, 'r') as f:
            for line in f:
                parts = line[:-1].split(' ')
                # If not filtering (train) or label is 1 (positive edges in valid/test), add the edge
                if not filter_positive or int(parts[3]) == 1:
                    edges[parts[0]].append((parts[1], parts[2]))
        return edges

    def Amazon_HeteroDataCreate(self, nodes, train_edges, valid_edges, test_edges):
        """Create heterogeneous graph data"""
        node_ids = list(nodes.keys())
        node_map = {nid: i for i, nid in enumerate(node_ids)}
        x = torch.tensor([[float(val) for val in nodes[nid]] for nid in node_ids], dtype=torch.float)

        data = HeteroData()
        data['product'].x = x

        all_edges = defaultdict(list)
        edge_splits = defaultdict(list)

        for edge_dict, label in [(train_edges, 'train'), (valid_edges, 'val'), (test_edges, 'test')]:
            for etype, edges in edge_dict.items():
                all_edges[etype].extend(edges)
                edge_splits[etype].extend([label] * len(edges))

        for etype, edges in all_edges.items():
            if not edges:
                continue

            src = [node_map[s] for s, _ in edges]
            dst = [node_map[t] for _, t in edges]
            edge_index = torch.tensor([src, dst], dtype=torch.long)

            splits = edge_splits[etype]
            masks = {
                'train_mask': torch.tensor([s == 'train' for s in splits], dtype=torch.bool),
                'val_mask': torch.tensor([s == 'val' for s in splits], dtype=torch.bool),
                'test_mask': torch.tensor([s == 'test' for s in splits], dtype=torch.bool)
            }
            etype_dict = {'1':'also_bought', '2':'also_viewed'}
            edge_type = ('product', etype_dict[etype], 'product')
            data[edge_type].edge_index = to_undirected(edge_index) # Make edges undirected
            for mask_name, mask_tensor in masks.items():
                data[edge_type][mask_name] = mask_tensor

        return data


class AmazonH(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['Amazon.npz']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        url = "https://media.githubusercontent.com/media/honey0219/LLM4HeG/refs/heads/main/dataset/Amazon.npz"
        save_path = os.path.join(self.raw_dir, self.raw_file_names[0])
        os.makedirs(self.raw_dir, exist_ok=True)

        print(f"Downloading {url}")
        r = requests.get(url)
        r.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(r.content)
        print(f"Extracting {save_path}")

    def process(self):
        npz_path = os.path.join(self.raw_dir, self.raw_file_names[0])
        data_npz = np.load(npz_path, allow_pickle=True)

        edges = data_npz['edges']  # shape [2, num_edges]
        node_labels = data_npz['node_labels']  # shape [num_nodes]
        node_features = data_npz['node_features']  # shape [num_nodes, num_features]
        node_texts = data_npz['node_texts']  # shape [num_nodes], dtype=object
        label_texts = data_npz['label_texts']  # shape [num_labels], dtype=object
        train_mask = data_npz['train_masks']  # shape [num_nodes], bool
        val_mask = data_npz['val_masks']  # shape [num_nodes], bool
        test_mask = data_npz['test_masks']  # shape [num_nodes], bool

        # to tensor
        edge_index = torch.tensor(edges, dtype=torch.long).contiguous()
        x = torch.tensor(node_features, dtype=torch.float)
        y = torch.tensor(node_labels, dtype=torch.long)
        train_mask = torch.tensor(train_mask, dtype=torch.bool)
        val_mask = torch.tensor(val_mask, dtype=torch.bool)
        test_mask = torch.tensor(test_mask, dtype=torch.bool)

        # to list
        node_texts_list = [text.replace(";", ". \n") for text in node_texts.tolist()]
        label_texts_list = label_texts.tolist()

        data = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            raw_texts=node_texts_list,
            label_names=label_texts_list
        )

        data_list = [data]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class Products(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['processed_data.pt']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        raise FileNotFoundError(
            "Failed to load Products: Dataset not found."
            "Please download 'processed_data.pt' from 'https://utexas.app.box.com/s/i7y03rzm40xt9bjbaj0dfdgxeyjx77gb/file/1441818321356' "
            "and move it to 'datasets/Products/raw"
        )

    def process(self):
        raw_path = os.path.join(self.raw_dir, self.raw_file_names[0])
        print(f"Extracting {raw_path}")
        data = torch.load(raw_path)
        data.raw_texts = [f'Content: {desc}' for desc in data.raw_texts]    
        data_list = [data]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class FB15k237(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        data_list = [self.FB15k237_DataCreate()]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def FB15k237_DataCreate(self):
        '''
        Load the FB15K-237 knowledge graph dataset using PyKEEN and converts it into PyG Data
        with original triple text retained.
        '''
        from pykeen.datasets import FB15k237 as FB15k237_pykeen

        dataset = FB15k237_pykeen()
        train_triples = dataset.training.mapped_triples
        valid_triples = dataset.validation.mapped_triples
        test_triples = dataset.testing.mapped_triples
        all_triples = torch.cat([train_triples, valid_triples, test_triples], dim=0)

        num_entities = dataset.num_entities
        num_edges = all_triples.size(0)

        # Map entity/relation ids to names
        # id_to_entity = {v: k for k, v in dataset.entity_to_id.items()}
        # id_to_relation = {v: k for k, v in dataset.relation_to_id.items()}

        # Generate human-readable triples (text)
        # triple = [
        #     (
        #         id_to_entity[int(h.item())],
        #         id_to_relation[int(r.item())],
        #         id_to_entity[int(t.item())]
        #     )
        #     for h, r, t in all_triples
        # ]

        # Prepare PyG format
        edge_index = all_triples[:, [0, 2]].T.contiguous()
        edge_type = all_triples[:, 1]
        entity = [f'Entity: {text}' for text in dataset.entity_to_id.keys()]
        relation = list(dataset.relation_to_id.keys())
        train_mask = torch.zeros(num_edges, dtype=torch.bool)
        val_mask = torch.zeros(num_edges, dtype=torch.bool)
        test_mask = torch.zeros(num_edges, dtype=torch.bool)
        train_mask[:train_triples.size(0)] = True
        val_mask[train_triples.size(0):train_triples.size(0) + valid_triples.size(0)] = True
        test_mask[-test_triples.size(0):] = True

        data = Data(
            num_nodes=num_entities,
            edge_index=edge_index,
            edge_type=edge_type,
            raw_texts=entity,
            relation_texts=relation,
            #triple_texts=triple,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
        )
        return data


class WN18RR(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        data_list = [self.WN18RR_DataCreate()]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def WN18RR_DataCreate(self):
        '''
        Load the WN18RR knowledge graph dataset using PyKEEN and converts it into PyG Data
        '''
        from pykeen.datasets import WN18RR as WN18RR_pykeen
        dataset = WN18RR_pykeen()
        train_triples = dataset.training.mapped_triples
        valid_triples = dataset.validation.mapped_triples
        test_triples = dataset.testing.mapped_triples
        all_triples = torch.cat([train_triples, valid_triples, test_triples], dim=0)

        num_entities = dataset.num_entities
        num_edges = all_triples.size(0)

        # Map entity/relation ids to names
        # id_to_entity = {v: k for k, v in dataset.entity_to_id.items()}
        # id_to_relation = {v: k for k, v in dataset.relation_to_id.items()}

        # Generate human-readable triples (text)
        # triple = [
        #     (
        #         id_to_entity[int(h.item())],
        #         id_to_relation[int(r.item())],
        #         id_to_entity[int(t.item())]
        #     )
        #     for h, r, t in all_triples
        # ]

        # Prepare PyG format
        edge_index = all_triples[:, [0, 2]].T.contiguous()
        edge_type = all_triples[:, 1]
        entity = [f'Entity: {text}' for text in dataset.entity_to_id.keys()]
        relation = list(dataset.relation_to_id.keys())
        train_mask = torch.zeros(num_edges, dtype=torch.bool)
        val_mask = torch.zeros(num_edges, dtype=torch.bool)
        test_mask = torch.zeros(num_edges, dtype=torch.bool)
        train_mask[:train_triples.size(0)] = True
        val_mask[train_triples.size(0):train_triples.size(0) + valid_triples.size(0)] = True
        test_mask[-test_triples.size(0):] = True

        data = Data(
            num_nodes=num_entities,
            edge_index=edge_index,
            edge_type=edge_type,
            raw_texts=entity,
            relation_texts=relation,
            # triple_texts=triple,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
        )
        return data


class TFinance(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['tfinance']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        raise FileNotFoundError(
            "Failed to load T-Finance: Dataset not found."
            "Please download 'tfinance' from 'https://dgraph.xinye.com/dataset#DGraph-Fin' "
            "and move it to 'datasets/T-Finance/raw"
        )

    def process(self):
        raw_path = os.path.join(self.raw_dir, self.raw_file_names[0])
        print(f"Extracting {raw_path}")
        data = self.TFinance_DataCreate(raw_path)
        data_list = [data]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def TFinance_DataCreate(self, root):
        '''
        DGL Package:
        !pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121
        !pip install dgl==2.1.0+cu121 -f https://data.dgl.ai/wheels/cu121/repo.html
        !pip install torchdata==0.7.1

        Transfer DGL graph to PyG Data
        '''
        from dgl.data.utils import load_graphs
        path = os.path.join(root, 'raw/tfinance')
        graph, _ = load_graphs(path)
        graph = graph[0]
        graph.ndata['label'] = graph.ndata['label'].argmax(1)
        src, dst = graph.edges()
        edge_index = torch.stack([src, dst], dim=0)  # shape = [2, num_edges]

        data = Data(
            x=graph.ndata['feature'].to(torch.float64),
            y=graph.ndata['label'],
            edge_index=edge_index
        )

        def to_unique_undirected(edge_index):
            edge_list = edge_index.t().tolist()
            unique_edges = list({tuple(sorted(e)) for e in edge_list})
            return torch.tensor(unique_edges, dtype=torch.long).t()

        data.edge_index = to_unique_undirected(data.edge_index)
        return data


def get_mag_metapath2vec_transform(root,is_reverse_added=True):
    """
    Returns a transform function that computes or loads MetaPath2Vec embeddings for a heterogeneous graph.
    If embeddings exist, they are loaded and assigned to node features. Otherwise, embeddings are trained and saved.
    """
    def mag_metapath2vec_transform(data):
        from torch_sparse import transpose
        from torch_geometric.nn import MetaPath2Vec
        nodes_embedding_path = os.path.join(root, "nodes_embedding.pt")
        # Add reverse edges to the heterogeneous graph
        if os.path.exists(nodes_embedding_path) and not is_reverse_added:
            # without adding reverse edges
            embedding_dict = torch.load(nodes_embedding_path,weights_only=False)
            for node_type, emb in embedding_dict.items():
                if hasattr(data[node_type], 'x') and data[node_type].x is not None:
                    data[node_type].x = torch.cat([emb, data[node_type].x], dim=1)
                else:
                    data[node_type].x = emb
            return data
        
        data[('institution', 'rev_affiliated_with', 'author')].edge_index = transpose(
                    data.edge_index_dict[('author', 'affiliated_with', 'institution')],None, 
                    m=data.num_nodes_dict['author'],
                    n=data.num_nodes_dict['institution'])[0]

        data[('paper', 'rev_writes', 'author')].edge_index = transpose(
            data.edge_index_dict[('author', 'writes', 'paper')], None,
            m=data.num_nodes_dict['author'], 
            n=data.num_nodes_dict['paper'])[0]

        data[('paper', 'rev_cites', 'paper')].edge_index = transpose(
            data.edge_index_dict[('paper', 'cites', 'paper')], None,
            m=data.num_nodes_dict['paper'],
            n=data.num_nodes_dict['paper'])[0]

        data[('field_of_study', 'rev_has_topic', 'paper')].edge_index = transpose(
            data.edge_index_dict[('paper', 'has_topic', 'field_of_study')], None,
            m=data.num_nodes_dict['paper'],
            n=data.num_nodes_dict['field_of_study'])[0]
        
        if os.path.exists(nodes_embedding_path) and is_reverse_added:
            # adding reverse edges
            embedding_dict = torch.load(nodes_embedding_path,weights_only=False)
            for node_type, emb in embedding_dict.items():
                if hasattr(data[node_type], 'x') and data[node_type].x is not None:
                    data[node_type].x = torch.cat([emb, data[node_type].x], dim=1)
                else:
                    data[node_type].x = emb
            return data

        args = {
            'cuda': 0,
            'embedding_dim': 128,
            'walk_length': 64,
            'context_size': 7,
            'walks_per_node': 5,
            'num_negative_samples': 5,
            'batch_size': 128,
            'learning_rate': 0.01,
            'epochs': 5,
            'log_steps': 500
        }
        args['cuda'] = f'cuda:{args["cuda"]}' if torch.cuda.is_available() and args["cuda"] >= 0 else 'cpu'

        metapath = [
            ('author', 'writes', 'paper'),
            ('paper', 'has_topic', 'field_of_study'),
            ('field_of_study', 'rev_has_topic', 'paper'),
            ('paper', 'rev_cites', 'paper'),
            ('paper', 'rev_writes', 'author'),
            ('author', 'affiliated_with', 'institution'),
            ('institution', 'rev_affiliated_with', 'author'),
            ('author', 'writes', 'paper'),
            ('paper', 'cites', 'paper'),
            ('paper', 'rev_writes', 'author')
        ]
        print("Training MetaPath2Vec embeddings...")
        metapath2vec_model = MetaPath2Vec(
            data.edge_index_dict,
            embedding_dim=args['embedding_dim'],
            metapath=metapath,
            walk_length=args['walk_length'],
            context_size=args['context_size'],
            walks_per_node=args['walks_per_node'],
            num_negative_samples=args['num_negative_samples']
        ).to(args['cuda'])

        loader = metapath2vec_model.loader(
            batch_size=args['batch_size'], shuffle=True, num_workers=4
        )
        optimizer = torch.optim.Adam(metapath2vec_model.parameters(), lr=args['learning_rate'])

        # Train
        metapath2vec_model.train()
        for epoch in range(1, args['epochs'] + 1):
            for i, (pos_rw, neg_rw) in enumerate(loader):
                optimizer.zero_grad()
                loss = metapath2vec_model.loss(
                    pos_rw.to(args['cuda']),
                    neg_rw.to(args['cuda'])
                )
                loss.backward()
                optimizer.step()

                if (i + 1) % args['log_steps'] == 0:
                    print(f'Epoch: {epoch:02d}, Step: {i + 1:03d}/{len(loader)}, Loss: {loss:.4f}')
        
        # Save the learned embeddings and assign them to node features
        embedding_dict = {}
        for node_type in metapath2vec_model.num_nodes_dict:
            embedding_dict[node_type] = metapath2vec_model(node_type).detach().cpu()
            if hasattr(data[node_type], 'x') and data[node_type].x is not None:
                data[node_type].x = torch.cat([embedding_dict[node_type], data[node_type].x], dim=1)
            else:
                data[node_type].x = embedding_dict[node_type]

        torch.save(embedding_dict, nodes_embedding_path)
        return data
    return mag_metapath2vec_transform


def HeTGB_undirected_transform(data):
    edge_index = to_undirected(data.edge_index)
    edge_index, _ = remove_self_loops(edge_index)
    data.edge_index = edge_index
    return data


def undirected_transform(data):
    edge_index = to_undirected(data.edge_index)
    data.edge_index = edge_index
    return data