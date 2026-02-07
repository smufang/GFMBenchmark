from exp.exp_basic import ExpBasic
import torch
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.tools import EarlyStopping
import os
import time
from tqdm import tqdm
import traceback


class ExpPretrain(ExpBasic):
    def __init__(self, args, pretrain_dict):
        super(ExpPretrain, self).__init__(args, pretrain_dict)
        self._print_main(f'Pretrain datasets: {", ".join(self.pretrain_dict.keys())}')

        self.need_input_nodes = False

        if self.args.cache_compress and self.compress_func is not None:
            self._print_main("Creating compressed data cache...")
            self.compressed_data = self._pure_data_dict(self._get_compressed_data_dict())
        else:
            self.compressed_data = None

        if self.args.model != "none":
            self.model = self._build_model().to(self.device)


    def _build_model(self):
        model = self.model_dict[self.args.model].Pretrain(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            local_rank = int(os.environ["LOCAL_RANK"])
            model = model.to(local_rank)
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True,
            )
        return model
    
    def _get_data_dict(self, is_text=True, need_y=False):
        """Get original muti-datasets without compression"""
        from data_provider.data_loader import get_original_data

        return get_original_data(
            self.pretrain_dict, is_text=is_text, need_y=need_y
        )
    
    def _get_compressed_data_dict(self):
        """Get feature dimension aligned data by specified compression function"""
        from data_provider.data_loader import get_compressed_data

        return get_compressed_data(
            self.pretrain_dict, compress_fc=self.compress_func, k=self.args.input_dim
        )

    def _get_loader(self):
        """Get data loader"""
        from data_provider.data_loader import pretrain_loader

        return pretrain_loader(
            self.pretrain_dict,
            num_neighbors=self.args.num_neighbors,
            max_nodes=self.args.max_nodes,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            compress_fc=self.compress_func,
            k=self.args.input_dim,
            compressed_data=self.compressed_data,
            need_input_nodes=self.need_input_nodes,
        )
        
    def _select_optimizer(self):
        model_optim = optim.Adam(
            self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay
        )
        return model_optim

    def _create_save_path(self):
        return self.args.checkpoints + "/" + self.setting

    def _create_progress_bar(self, train_loader, epoch):
        if self._is_main_process():
            return tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{self.args.epochs}",
                position=0,
                leave=True,
                dynamic_ncols=False,
            )
        else:
            return train_loader

    def train_single_epoch(self, train_loader, optimizer, scaler, epoch):
        """Train model for a single epoch"""
        train_loss = 0.0
        num_tasks = 0
        epoch_time = time.time()
        avg_loss = float("inf")

        try:
            pbar = self._create_progress_bar(train_loader, epoch)

            for batch_idx, batch_data in enumerate(pbar, start=1):
                batch_data = batch_data.to(self.device)
                optimizer.zero_grad()
                # Forward and backward pass with optional mixed precision
                if self.args.use_amp:
                    with torch.amp.autocast("cuda"):
                        loss = self.model(batch_data)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss = self.model(batch_data)
                    loss.backward()
                    optimizer.step()

                # Update loss statistics
                num_batch_tasks = batch_data.num_nodes
                train_loss += loss.item() * num_batch_tasks
                num_tasks += num_batch_tasks
                avg_loss = train_loss / num_tasks
                torch.cuda.empty_cache()

                # Update progress bar and log (only in main process)
                if self._is_main_process() and hasattr(pbar, "set_postfix"):
                    pbar.set_postfix(
                        {"loss": f"{loss.item():.6f}", 
                         "avg_loss": f"{avg_loss:.6f}"}
                    )
            self._write_log(
                f"Epoch {epoch + 1} finished. "
                f"Average Loss: {avg_loss:.6f}, "
                f"Time: {time.time() - epoch_time:.2f}s"
            )

        except Exception as e:
            self._write_log(
                f"ERROR in Epoch {epoch + 1}\n"
                f"Error type: {type(e).__name__}\n"
                f"Error message: {str(e)}\n"
                f"Traceback:\n{traceback.format_exc()}\n"
            )
            raise e
        return avg_loss

    def train(self):
        optimizer = self._select_optimizer()
        path = self._create_save_path()

        os.makedirs(path, exist_ok=True)
        if self.args.continue_train is True:
            model_path = path + "/" + "checkpoint.pth"
            self._load_checkpoint(model_path, self.model)

        self._print_main(f"Checkpoints path: {path}")

        time_start = time.time()
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        scaler = torch.amp.GradScaler("cuda") if self.args.use_amp else None

        # Training loop
        for epoch in range(self.args.epochs):
            train_loader = self._get_loader()
            self.model.train()

            avg_loss = self.train_single_epoch(train_loader, optimizer, scaler, epoch)

            # Early stopping check
            early_stopping(avg_loss, self.model, path)
            if early_stopping.early_stop:
                self._print_main(f"Early stopping at epoch: {epoch + 1}")
                self._write_log(f"Early stopping at epoch: {epoch + 1}")
                break

        # Load best model and print results
        best_model_path = os.path.join(path, "checkpoint.pth")
        self._load_checkpoint(best_model_path, self.model)

        training_info = {
            "best_loss": early_stopping.val_loss_min,
            "training_time": time.time() - time_start,
            "model_path": best_model_path,
        }

        self._print_main(
            f"\nTraining finished:",
            f'Best loss: {training_info["best_loss"]:.6f} ',
            f'Training time: {training_info["training_time"]:.2f}s',
            f'\nSave path: {training_info["model_path"]}',
        )
        self._write_log(
            f"Training finished:"
            f'Best loss: {training_info["best_loss"]:.6f} '
            f'Training time: {training_info["training_time"]:.2f}s\n'
            f'Save path: {training_info["model_path"]}'
        )

        return self.model

    def vali(self):
        path = self._create_save_path()
        model_path = path + "/" + "checkpoint.pth"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint path {model_path} does not exist.")
        self._load_checkpoint(model_path, self.model)

        vali_loader = self._get_loader()
        self.model.eval()
        total_loss = 0.0
        num_nodes = 0
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(vali_loader, start=1):
                batch_data = batch_data.to(self.device)
                loss = self.model(batch_data)
                self._print_main(f"Batch: {batch_idx}, Loss: {loss.item():.6f}")
                total_loss += loss.item() * batch_data.num_nodes
                num_nodes += batch_data.num_nodes

        avg_loss = total_loss / num_nodes
        self._print_main(f"Validation Loss: {avg_loss:.6f}")
