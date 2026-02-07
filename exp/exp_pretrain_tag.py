from exp.exp_basic import ExpBasic
from tqdm import tqdm


class ExpPretrain(ExpBasic):
    def __init__(self, args, pretrain_dict):
        super(ExpPretrain, self).__init__(args, pretrain_dict)
        self._print_main(f'Pretrain datasets: {", ".join(self.pretrain_dict.keys())}')

        if self.args.model in ["g2p2"]:
            self.need_input_nodes = True
            self.need_token_cache = True
        elif self.args.model in ["graphclip"]:
            self.need_input_nodes = False
            self.need_token_cache = False
        else:
            self.need_input_nodes = False
            self.need_token_cache = False

        if self.args.cache_compress and self.compress_func is not None:
            self._print_main("Creating compressed data cache...")
            self.compressed_data = self._pure_data_dict(self._get_compressed_data_dict())
        else:
            self.compressed_data = None

    def _get_data_dict(self, is_text=True, need_y=False):
        """Get original muti-datasets without compression"""
        from data_provider.data_loader import get_original_data

        return get_original_data(
            self.pretrain_dict, is_text=is_text, need_y=need_y, need_token_cache=self.need_token_cache
        )
    
    def _get_compressed_data_dict(self):
        """Get feature dimension aligned data by specified compression function"""
        from data_provider.data_loader import get_compressed_data

        return get_compressed_data(
            self.pretrain_dict, compress_fc=self.compress_func, k=self.args.input_dim, need_token_cache=self.need_token_cache
        )

    def _get_loader(self):
        """Get data loader"""
        if self.args.preprocess == "basic":
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
        elif self.args.pattern == "single" and self.args.preprocess == "simple":
            if len(self.pretrain_dict) != 1:
                raise ValueError(
                    f"Single-pattern pretrain expects exactly one dataset, got {len(self.pretrain_dict)}"
                )
            name, data = next(iter(self.pretrain_dict.items()))
            self._print_main(f"Using single pretrain dataset: {name}")
            return [self._simple_preprocess(data())]
        else:
            raise ValueError("GFM pretrain only supports basic preprocess methods.")

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
