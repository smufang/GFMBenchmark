from copy import deepcopy
import copy
import numpy as np

class EarlyStopping:
    def __init__(self, patience=50):
        self.patience = patience
        self.counter = 0
        self.best_val = -np.inf
        self.best_dict = None
        self.early_stop = False

    def __call__(self, result):
        if result['val'] > self.best_val:
            self.best_val = result['val']
            self.best_dict = result
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

class TrainLossEarlyStopping:
    def __init__(self, patience=50, delta=0.0, verbose=False):
        self.patience = int(patience)
        self.delta = float(delta)
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.best_model = None
        self.best_model_state = None
        self.best_result = None
        self.early_stop = False

    def __call__(self, train_loss, model, result_for_record=None):
        try:
            cur = float(train_loss)
        except Exception:
            return False
        if self.best_loss is None:
            self.best_loss = cur
            self.best_model = deepcopy(model)
            self.best_model_state = deepcopy(model.state_dict())
            self.best_result = deepcopy(result_for_record) if result_for_record is not None else None
            self.counter = 0
            if self.verbose:
                print(f"[EarlyStopping] init best_loss = {self.best_loss:.6f}")
            return False
        # smaller is better
        if cur < self.best_loss - self.delta:
            self.best_loss = cur
            self.best_model = deepcopy(model)
            self.best_model_state = deepcopy(model.state_dict())
            self.best_result = deepcopy(result_for_record) if result_for_record is not None else None
            self.counter = 0
            if self.verbose:
                print(f"[EarlyStopping] improvement: best_loss = {self.best_loss:.6f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"[EarlyStopping] no improv ({self.counter}/{self.patience}); current={cur:.6f}, best={self.best_loss:.6f}")
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop