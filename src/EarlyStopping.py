import torch

class EarlyStopping:
    def __init__(self, store_path, patience=30, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.epochs_without_improvement = 0
        self.store_path = store_path

    def should_stop(self, model, current_loss):
        if self.best_loss is None or (self.best_loss - current_loss > self.min_delta):
            self.best_loss = current_loss
            self.epochs_without_improvement = 0
            torch.save(model.state_dict(), self.store_path)  # Save the best model
        else:
            self.epochs_without_improvement += 1

        if self.epochs_without_improvement >= self.patience:
            return True
        return False