"""CPU perceptron for the subdaily signal experiments; imported only by the signals pipeline.

The network sees the same rows, features, calibrator and threshold rule as the tree candidates, so a
difference measures the architecture rather than the information or the operating point.
"""
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler

HIDDEN, BLOCKS, DROPOUT = 64, 2, .2
BATCH, EPOCHS, PATIENCE, LEARNING_RATE, WEIGHT_DECAY, SEED = 2048, 80, 8, 3e-3, 1e-4, 42


class Perceptron(nn.Module):
    """Input projection plus additive residual blocks, the architecture used by the MLP cohorts."""

    def __init__(self, features, hidden=HIDDEN, blocks=BLOCKS, dropout=DROPOUT):
        super().__init__()
        self.input = nn.Linear(features, hidden)
        self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(hidden, hidden)) for _ in range(blocks)])
        self.activation = nn.SiLU()
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        hidden = self.input(x)
        for block in self.blocks:
            hidden = self.activation(hidden + block(hidden))
        return self.head(hidden).squeeze(1)


def matrix(frame, features):
    """Numeric matrix with missing entries filled at the standardized origin after scaling.

    Tree candidates split on missing values natively, so filling at the training mean keeps the two
    families comparable on information rather than giving the network a different feature contract.
    """
    return np.nan_to_num(frame[features].apply(pd.to_numeric, errors='coerce').to_numpy(float), nan=0.)


def fit_network(features, train, validation, seed=SEED, epochs=EPOCHS, patience=PATIENCE):
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    if train.empty or validation.empty or train.target.nunique() < 2:
        raise ValueError('Require nonempty two-class training and validation rows')
    scaler = StandardScaler().fit(matrix(train, features))
    def tensors(part):
        return torch.from_numpy(scaler.transform(matrix(part, features)).astype('float32'))
    x, y = tensors(train), torch.from_numpy(train.target.to_numpy('float32'))
    vx, vy = tensors(validation), torch.from_numpy(validation.target.to_numpy('float32'))
    model = Perceptron(len(features))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    generator = torch.Generator().manual_seed(seed)
    best_loss, best_state, stale = np.inf, None, 0
    epochs_run = 0
    for _ in range(epochs):
        model.train()
        for batch in torch.randperm(len(train), generator=generator).split(BATCH):
            optimizer.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(model(x[batch]), y[batch])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite network training loss')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            value = float(nn.functional.binary_cross_entropy_with_logits(model(vx), vy))
        if value < best_loss - 1e-5:
            best_loss, stale = value, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
        epochs_run += 1
    if best_state is None:
        raise ValueError('Network training produced no finite validation loss')
    return dict(network=best_state, scaler=scaler, features=list(features), validation_log_loss=best_loss,
        parameters=sum(p.numel() for p in model.parameters()), epochs_run=epochs_run)


def network_logits(bundle, frame):
    torch.set_num_threads(4)
    model = Perceptron(len(bundle['features']))
    model.load_state_dict(bundle['network'])
    model.eval()
    x = torch.from_numpy(bundle['scaler'].transform(matrix(frame, bundle['features'])).astype('float32'))
    outputs = []
    with torch.no_grad():
        for batch in torch.arange(len(frame)).split(8192):
            outputs.append(model(x[batch]).numpy())
    return np.concatenate(outputs) if outputs else np.empty(0)
