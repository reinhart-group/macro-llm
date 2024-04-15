import numpy as np
from sklearn import ensemble, model_selection
import torch


class EnsembleRFModel(object):

    def __init__(self, n_models=3, constructor=ensemble.RandomForestRegressor, model_params={}):
        self.n_models = n_models
        self.model_params = model_params
        self.models = [constructor(**self.model_params) for _ in range(self.n_models)]

    def fit(self, x, y):
        cv = model_selection.KFold(n_splits=self.n_models, shuffle=True, random_state=0)
        for i, (fold_train_idx, fold_val_idx) in enumerate(cv.split(x)):
            _ = self.models[i].fit(x[fold_train_idx], y[fold_train_idx])
        return self

    def predict(self, x, return_std=False):
        all_y = np.vstack([it.predict(x).reshape(1, -1) for it in self.models])

        if return_std:
            return np.mean(all_y, axis=0), np.std(all_y, axis=0)
        else:
            return np.mean(all_y, axis=0)


def predict_from_model(model, this_data):

    if 'MLP' in model.original_name:  # for MLP
        testX = torch.tensor(this_data, dtype=torch.float)
    elif 'CNN' in model.original_name:  # for CNN:
        testX = torch.tensor(this_data, dtype=torch.float).unsqueeze(1)
    elif 'GRU' in model.original_name:  # for RNN:
        testX = torch.tensor(this_data, dtype=torch.float).unsqueeze(2)

    pred_z_fwd = model(testX).detach().numpy()
    pred_z_rev = model(torch.fliplr(testX)).detach().numpy()
    pred_z = 0.5*(pred_z_fwd + pred_z_rev)

    return pred_z


def run_sequences(sequences, model_ensemble):
    AB = {'A': 0, 'B': 1}
    int_seq = np.array([[int(AB[x]) for x in s] for s in sequences])

    predictions = []
    for model in model_ensemble:
        out = predict_from_model(model, int_seq)
        predictions.append(out)
    z_pred = np.array(predictions).mean(axis=0)

    result = []
    for i, s in enumerate(sequences):
        result += [f'{sequences[i]}: {z_pred[i, 0]:.3f}, {z_pred[i, 1]:.3f}']

    return '\n'.join(result)


def evaluate_sequences(sequences, target, model_ensemble, sort=True):
    AB = {'A': 0, 'B': 1}
    int_seq = np.array([[int(AB[x]) for x in s] for s in sequences])

    predictions = []
    for model in model_ensemble:
        out = predict_from_model(model, int_seq)
        predictions.append(out)
    z_pred = np.array(predictions).mean(axis=0)
    z_dist = np.linalg.norm(z_pred - target, axis=1)

    result = []
    if sort:
        ordering = np.argsort(z_dist)
    else:
        ordering = np.arange(len(z_dist))
    for i in ordering:
        result += [f'{sequences[i]}: {z_dist[i]:.3f}']

    return '\n'.join(result)