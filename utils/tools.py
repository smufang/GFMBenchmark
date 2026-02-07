import numpy as np
import torch
import torch.distributed as dist


# class EarlyStopping:
#     def __init__(self, patience=7, verbose=False, delta=0.0000001):
#         self.patience = patience
#         self.verbose = verbose
#         self.counter = 0
#         self.best_score = None
#         self.early_stop = False
#         self.val_loss_min = np.inf
#         self.delta = delta
#         self.best_model = None

#     def __call__(self, val_loss, model, path, is_save=True):
#         score = -val_loss
#         if self.best_score is None:
#             self.best_score = score
#             self.best_model = model
#             if is_save:
#                 self.save_checkpoint(val_loss, self.best_model, path)
#         elif score < self.best_score + self.delta:
#             self.counter += 1
#             if (not dist.is_initialized() or dist.get_rank() == 0) and self.verbose:
#                 print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
#             if self.counter >= self.patience:
#                 self.early_stop = True
#         else:
#             self.best_score = score
#             self.best_model = model
#             if is_save:
#                 self.save_checkpoint(val_loss, self.best_model, path)
#             self.counter = 0

#     def save_checkpoint(self, val_loss, model, path):
#         if not dist.is_initialized() or dist.get_rank() == 0:
#             if self.verbose:
#                 print(f'Loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
#             if hasattr(model, 'module'):
#                 torch.save(model.module.state_dict(), path + '/' + 'checkpoint.pth')
#             else:
#                 torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
#             self.val_loss_min = val_loss


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=1e-7):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.best_state_dict = None

    def __call__(self, val_loss, model, path=None, is_save=True):
        score = -val_loss

        if self.best_score is None:
            self._update_best(score, val_loss, model, path, is_save)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self._is_main_process() and self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.counter = 0
            self._update_best(score, val_loss, model, path, is_save)

    def _update_best(self, score, val_loss, model, path, is_save):
        self.best_score = score
        self._save_best_state(model)
        if is_save and path is not None:
            self._save_checkpoint(val_loss, model, path)

    def _save_best_state(self, model):
        state_dict = self._get_state_dict(model)
        self.best_state_dict = {
            k: v.detach().cpu().clone()
            for k, v in state_dict.items()
        }

    def _get_state_dict(self, model):
        return model.module.state_dict() if hasattr(model, 'module') else model.state_dict()

    def _is_main_process(self):
        return (not dist.is_initialized()) or dist.get_rank() == 0

    def _save_checkpoint(self, val_loss, model, path):
        if not self._is_main_process():
            return

        if self.verbose:
            print(f'Loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). '
                f'Saving model ...')

        torch.save(self._get_state_dict(model), path + '/checkpoint.pth')
        self.val_loss_min = val_loss

    def load_best_model(self, model, device=None):
        if self.best_state_dict is None:
            return
        model_state = model.module if hasattr(model, 'module') else model
        model_state.load_state_dict(self.best_state_dict)
        if device is not None:
            model.to(device)