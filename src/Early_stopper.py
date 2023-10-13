import torch
import numpy as np


class EarlyStopper:
    def __init__(self,
                 patients: int
                 ):
        self.patients = patients
        self.model_states = {}  # dict {epoch : model.state_dict()}
        self.idx_best_model = 0
        self.loss_best_model = np.inf
        self.steps_last_best = 0   # counts epochs of the last best model
        self.do_stop = False

    def save_model_state(self, epoch: int, state_dict, current_loss: float):
        self.model_states[epoch] = state_dict
        # lower loss achived
        if current_loss < self.loss_best_model:
            self.idx_best_model = epoch
            self.loss_best_model = current_loss
            self.steps_last_best = 0
        # current loss is higher or equal
        else:
            self.steps_last_best += 1
            if self.steps_last_best == self.patients:
                self.do_stop = True


    def save_best_model(self, model_path: str):
        print(f'loss of best model={self.loss_best_model} at epoch={self.idx_best_model}')
        best_model = self.model_states[self.idx_best_model]
        torch.save(best_model, model_path)


