import os
import random
import numpy as np
import torch
import torch.distributed as dist
from torch_geometric.data import Data
from models import MDGPT, SAMGPT, GraphCLIP, GCOPE, MDGFM, \
                    GCN, GAT, Simple_HGN, HeCo, TGN, DDGCL
from data_provider import dict2id
from datetime import datetime
from root import ROOT_DIR


class ExpBasic(object):
    def __init__(self, args, pretrain_dict=None):
        self.args = args
        self.model_dict = {
            "mdgpt": MDGPT,
            "samgpt": SAMGPT,
            "gcope": GCOPE,
            "graphclip": GraphCLIP,
            "mdgfm": MDGFM,
            "gcn": GCN,
            "gat": GAT,
            "simple_hgn": Simple_HGN,
            "heco": HeCo,
            "tgn": TGN,
            "ddgcl": DDGCL,
            "none": None, # used to check data
        }
        self.pretrain_dict = pretrain_dict
        if self.pretrain_dict is not None:
            self.args.pretrain_domain_id = dict2id(
                pretrain_dict
            )  # used in pretrain model built
        self.compress_func = self._select_dimension_alignment()

        if self.args.pattern in ["cross", "single", "none"]:
            self.device = self._acquire_device() # acquired in exp
            if self.args.task_name == "pretrain":
                self.setting = self._pretrain_setting()
            else:
                self.setting = self._setting()
        elif self.args.pattern == "simple":
            self.device = self.args.device # acquired outside
            if self.args.task_name == "pretrain":
                self.setting = self._simple_pretrain_setting()
            else:
                self.setting = self._simple_setting()
        else:
            raise ValueError(f"Unknown pattern: {self.args.pattern}")
        self._set_seed(args.seed)
        self._setup_logger()

    @staticmethod
    def _is_main_process():
        """Check if current process is the main process (rank 0)"""
        return not dist.is_initialized() or dist.get_rank() == 0

    def _print_main(self, *args, **kwargs):
        """Print only in main process"""
        if self._is_main_process():
            print(*args, **kwargs)

    def _setting(self):
        setting = "{}({})_{}({})_{}_{}_{}_{}_id{}_hd{}_nl{}_nh{}_{}({})_s{}_{}shots".format(
            self.args.model,
            self.args.model_id,
            self.args.task_name,  # petrain
            self.args.pattern,  # cross or single
            self.args.mode,  # lp or gcl
            self.args.backbone,
            self.args.compress_function,
            self.args.combinetype,
            self.args.input_dim,
            self.args.hidden_dim,
            self.args.num_layers,
            self.args.num_heads,
            self.args.target_data,
            self.args.exp_id,
            self.args.seed,
            self.args.num_shots,
        )
        return setting

    def _pretrain_setting(self):
        setting = "{}({})_{}({})_{}_{}_{}_{}_id{}_hd{}_nl{}_nh{}".format(
            self.args.model,
            self.args.model_id,
            "pretrain",
            self.args.pattern,  # cross or single
            self.args.mode,  # lp or gcl
            self.args.backbone,
            self.args.compress_function,
            self.args.combinetype,
            self.args.input_dim,
            self.args.hidden_dim,
            self.args.num_layers,
            self.args.num_heads,
        )
        return setting

    def _simple_setting(self):
        setting = "{}({})_{}({})_{}({})_s{}_{}shots".format(
            self.args.model,
            self.args.model_id,
            self.args.task_name,
            self.args.pattern,
            self.args.target_data, 
            self.args.exp_id,
            self.args.seed,
            self.args.num_shots,
        )
        return setting
    
    def _simple_pretrain_setting(self):
        setting = "{}({})_{}({})".format(
            self.args.model,
            self.args.model_id,
            "pretrain",
            self.args.pattern,
        )
        return setting

    def _setup_logger(self):
        """Setup simple logger - creates log file in logs directory"""
        if not self._is_main_process() or self.args.is_logging is False:
            self.log_file = None
            return

        # Create logs directory
        log_dir = os.path.join(ROOT_DIR, "logs", self.args.model)
        os.makedirs(log_dir, exist_ok=True)

        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"{self.setting}_{timestamp}.log"
        log_path = os.path.join(log_dir, log_filename)
        self.log_file = open(log_path, "w", encoding="utf-8")
        self._write_log(f"Experiment started: {self.args.model}")
        self._write_log(f"Experiment setting: {self.setting}")
        print(f"Log file created: {log_path}")

    def _write_log(self, message: str):
        """Write a message to the log file if it exists"""
        if (
            hasattr(self, "log_file")
            and self._is_main_process()
            and self.args.is_logging is True
        ):
            self.log_file.write(f"[{datetime.now()}] {message}\n")
            self.log_file.flush()

    def __del__(self):
        if hasattr(self, "log_file") and self.log_file is not None:
            self.log_file.close()

    def _set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
        self._print_main(f"Set random seed to {seed}")

    def _acquire_device(self):
        if self.args.use_gpu:
            if self.args.gpu_type == "cuda" and torch.cuda.is_available():
                if self.args.use_multi_gpu:
                    os.environ["CUDA_VISIBLE_DEVICES"] = self.args.devices
                    if "LOCAL_RANK" in os.environ:
                        local_rank = int(os.environ["LOCAL_RANK"])
                    else:
                        raise ValueError(
                            "LOCAL_RANK environment variable is not set for multi-GPU training."
                        )
                    torch.cuda.set_device(local_rank)
                    if not dist.is_initialized():
                        dist.init_process_group(backend="nccl")
                    self._print_main(f"Use multi-GPU: {self.args.devices}")
                    return torch.device(local_rank)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
                    print(f"Use GPU: cuda {self.args.gpu}")
                    return torch.device(f"cuda:{self.args.gpu}")
            elif self.args.gpu_type == "mps" and torch.backends.mps.is_available():
                print("Use GPU: mps")
                return torch.device("mps")

        print("Use CPU")
        return torch.device("cpu")

    def _load_checkpoint(self, model_path, model, strict=True):
        """Load checkpoint with proper handling of DDP wrapper"""

        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)

            # Handle DDP wrapper key mismatch
            if hasattr(model, "module"):
                # Current model is DDP wrapped, but checkpoint might not have 'module.' prefix
                if not any(key.startswith("module.") for key in checkpoint.keys()):
                    # Add 'module.' prefix to checkpoint keys
                    checkpoint = {f"module.{k}": v for k, v in checkpoint.items()}
            else:
                # Current model is not DDP wrapped, but checkpoint might have 'module.' prefix
                if any(key.startswith("module.") for key in checkpoint.keys()):
                    # Remove 'module.' prefix from checkpoint keys
                    checkpoint = {
                        k.replace("module.", ""): v for k, v in checkpoint.items()
                    }

            model.load_state_dict(checkpoint, strict=strict)
            self._print_main(f"Loaded checkpoint from {model_path}")
        else:
            raise FileNotFoundError(f"Checkpoint not found at {model_path}")

    def _monitor_resources(self, shm_threshold=75, gpu_threshold=80, auto_gc=True, verbose=True):
        """
        Monitor /dev/shm, CPU memory, and GPU memory usage.
        Supports single GPU, DataParallel (DP), and DistributedDataParallel (DDP).
        Automatically triggers cleanup if usage exceeds thresholds.
        """
        import shutil
        import psutil
        # --- /dev/shm ---
        total, used, free = shutil.disk_usage("/dev/shm")
        shm_percent = used / total * 100
        # --- CPU memory ---
        mem = psutil.virtual_memory()
        if verbose:
            self._print_main(
                f"\n[Monitor] /dev/shm: {shm_percent:.2f}% ({used // (1024**3)}G / {total // (1024**3)}G)"
            )
            self._print_main(
                f"[Monitor] CPU memory: {mem.percent:.2f}% ({mem.used // (1024**3)}G / {mem.total // (1024**3)}G)"
            )

        # --- GPU memory ---
        gpu_percent_max = 0.0
        if torch.cuda.is_available() and hasattr(self, "device") and self.device.type == "cuda":
            num_devices = torch.cuda.device_count()

            for d in range(num_devices):
                mem_alloc = torch.cuda.memory_allocated(d) / 1024**2
                mem_total = torch.cuda.get_device_properties(d).total_memory / 1024**2
                percent = (mem_alloc / mem_total * 100) if mem_total > 0 else 0.0
                gpu_percent_max = max(gpu_percent_max, percent)
                if verbose:
                    self._print_main(
                        f"[Monitor] GPU {d} memory: {percent:.2f}% ({mem_alloc:.0f}MB / {mem_total:.0f}MB)"
                    )

        # --- Trigger cleanup if thresholds exceeded ---
        if auto_gc and (shm_percent > shm_threshold or mem.percent > 90):
            self._print_main(
                "/dev/shm or CPU memory usage is high, triggering CPU memory cleanup"
            )
            self._clear_cpu_memory()

        if auto_gc and gpu_percent_max > gpu_threshold:
            self._print_main("GPU memory usage is high, triggering GPU memory cleanup")
            self._clear_gpu_memory()

    def _clear_cpu_memory(self):
        """Force Python CPU memory cleanup using garbage collector"""
        import gc

        gc.collect()
        self._print_main("CPU memory cleared")

    def _clear_gpu_memory(self):
        """Force GPU memory cleanup by releasing unused cached memory, safe for DP/DDP"""
        torch.cuda.empty_cache()
        self._print_main("GPU memory cleared")
    
    def _select_dimension_alignment(self):
        """Select dimension alignment function"""
        alignment_funcs = {
            "pca": "compress_pca",
            "svd": "compress_svd",
            "svd_gcope": "gcope_svd",
            "none": None,
        }
        if self.args.compress_function in alignment_funcs:
            if alignment_funcs[self.args.compress_function] is None:
                self._print_main("Compression function is None.")
                return None
            from utils import compress_func

            return getattr(compress_func, alignment_funcs[self.args.compress_function])
        else:
            raise ValueError(
                f"Unknown compress function: {self.args.compress_function}"
            )
    
    def _get_compressed_data(self, data):
        data.x = self.compress_func(data.x, k=self.args.input_dim)
        return data

    def _pure_data_dict(self, data_dict):
        """
        used in downstream with single dataset - 
        source core and data attribute you can find in data_provider/data_loader.py - complete_data() function
        """
        if self.args.model == "none":
            return data_dict
        for name, data in data_dict.items():
            new_data = Data(
                x=data.x.contiguous(),
                edge_index=data.edge_index.contiguous(),
                batch=data.batch if hasattr(data, "batch") else None
            )
            if self.args.model in ["mdgpt", "samgpt", "mdgfm"]:
                new_data.name=data.name
            elif self.args.model in ["gcope"]:
                new_data.raw_texts = data.raw_texts
            elif self.args.model in ["g2p2"]:
                new_data.raw_texts = data.raw_texts
                new_data.token_cache = data.token_cache if hasattr(data, "token_cache") else None
            elif self.args.model in ["graphclip"]:
                new_data.edge_type = data.edge_type
                new_data.raw_texts = data.raw_texts
                new_data.relation_texts = data.relation_texts
            else:
                pass
            data_dict[name] = new_data
        return data_dict
        
    def _pure_data(self, data):
        """
        used in downstream with single dataset - 
        source core and data attribute you can find in data_provider/data_loader.py - complete_data() function
        """
        if self.args.model == "none":
            return data
        new_data = Data(
            x=data.x.contiguous(),
            edge_index=data.edge_index.contiguous(),
            batch=data.batch if hasattr(data, "batch") else None,
        )
        if self.args.model in ["mdgpt", "samgpt", "mdgfm"]:
            pass
        elif self.args.model in ["gcope"]:
            new_data.raw_texts = data.raw_texts
        elif self.args.model in ["g2p2", "graphclip"]:
            if self.args.task_name == "edge":
                new_data.relation_texts = data.relation_texts
            else:
                assert hasattr(data, "label_names"), f"Data don't have label_names field"
                assert hasattr(data, "label_descs"), f"Data don't have label_descs field"
                new_data.label_names = data.label_names
                new_data.label_descs = data.label_descs
        else:
            pass
        return new_data
    
    def _simple_preprocess(self, dataset):
        """Get original dataset with simple onehot feature preprocessing"""
        from torch_geometric.data import Data, HeteroData, TemporalData
        from utils.format_trans import simple_process_hetero, simple_process_temporal
        from data_provider.data_loader import create_x

        if len(dataset) > 1:
            self.data_type = "multi"
            data = dataset
        else:
            data = dataset[0]
            if isinstance(data, TemporalData):
                data = simple_process_temporal(data, need_y=False)
                self.data_type = "temporal"
            elif isinstance(data, HeteroData):
                data = simple_process_hetero(data, need_y=False, fill_onehot=True)
                self.data_type = "hetero"
            elif isinstance(data, Data):
                data = create_x(data)
                self.data_type = "data"
            else:
                raise ValueError(f"Unknown data type: {type(data)}")
        return data
    
    def _basic_preprocess(self, dataset):
        """Get original dataset without compression"""
        from torch_geometric.data import Data, HeteroData, TemporalData
        from utils.format_trans import temporal_to_data, hetero_to_data, multi_to_one
        from data_provider.data_loader import create_x, complete_data

        if len(dataset) > 1:
            data = multi_to_one(dataset, need_y=self.need_y)
            self.data_type = "multi"
        else:
            data = dataset[0]
            if isinstance(data, TemporalData):
                data = temporal_to_data(data, need_y=self.need_y)
                self.data_type = "temporal"
            elif isinstance(data, HeteroData):
                data = hetero_to_data(data, need_y=self.need_y)
                self.data_type = "hetero"
            elif isinstance(data, Data):
                data = create_x(data)
                self.data_type = "data"
            else:
                raise ValueError(f"Unknown data type: {type(data)}")
        data = complete_data(data, self.args.target_data, need_y=self.need_y)
        return data

    def get_loader(self):
        raise NotImplementedError

    def _build_model(self):
        raise NotImplementedError

    def _get_pretrain_model(self):
        raise NotImplementedError

    def train(self):
        pass

    def vali(self):
        pass

    def test(self):
        pass
