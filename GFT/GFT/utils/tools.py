import numpy as np
import torch
import torch.distributed as dist


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0.0000001):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.best_model = None

    def __call__(self, val_loss, model, path, is_save=True):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_model = model
            if is_save:
                self.save_checkpoint(val_loss, self.best_model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model = model
            if is_save:
                self.save_checkpoint(val_loss, self.best_model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if not dist.is_initialized() or dist.get_rank() == 0:
            if self.verbose:
                print(f'Loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
            if hasattr(model, 'module'):
                torch.save(model.module.state_dict(), path + '/' + 'checkpoint.pth')
            else:
                torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
            self.val_loss_min = val_loss
