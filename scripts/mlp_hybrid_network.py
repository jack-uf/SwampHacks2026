"""Hybrid CPU perceptrons for the MLP hybrid onset experiment; import only inside the neural worker."""
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler

HIDDEN = 64
DROPOUT = .2
EPOCHS = 200
PATIENCE = 20
BATCH = 2048
LEARNING_RATE = 3e-3
WEIGHT_DECAY = 1e-4
SEED = 42
OFFSET_COLUMN = 'damped_offset'


class Perceptron(nn.Module):
    """The two-layer SiLU network from the earlier experiment, with one scalar output."""

    def __init__(self, features, hidden=HIDDEN, dropout=DROPOUT):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(features, hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(dropout))
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        return self.head(self.trunk(x)).squeeze(1)


def _encoded(scaler, frame, features):
    return torch.from_numpy(scaler.transform(frame[features]).astype('float32'))


def _offset(offset_scaler, frame):
    if offset_scaler is None:
        return torch.zeros(len(frame))
    values = offset_scaler.transform(frame[[OFFSET_COLUMN]]).astype('float32')
    return torch.from_numpy(values).squeeze(1)


def fit_network(name, features, train, validation, offset=False, balanced=False):
    """Fit one perceptron; `offset` adds the standardized damped projection to the logit."""
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(SEED)
    if train.empty or validation.empty:
        raise ValueError('Require nonempty train and validation rows')
    if offset and OFFSET_COLUMN not in train:
        raise ValueError('An offset model needs the ' + OFFSET_COLUMN + ' column')
    scaler = StandardScaler().fit(train[features])
    offset_scaler = StandardScaler().fit(train[[OFFSET_COLUMN]]) if offset else None
    x = _encoded(scaler, train, features)
    y = torch.from_numpy(train['target'].to_numpy(dtype='float32'))
    shift = _offset(offset_scaler, train)
    vx = _encoded(scaler, validation, features)
    vy = torch.from_numpy(validation['target'].to_numpy(dtype='float32'))
    vshift = _offset(offset_scaler, validation)
    model = Perceptron(len(features))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    weight = None
    if balanced:
        positives = float(train['target'].sum())
        weight = torch.tensor([(len(train) - positives) / max(positives, 1.0)])
    generator = torch.Generator().manual_seed(SEED)
    best_loss, best_state, stale, epochs_run = np.inf, None, 0, 0

    def logits(inputs, shifts):
        return model(inputs) + shifts

    for _ in range(EPOCHS):
        model.train()
        for batch in torch.randperm(len(train), generator=generator).split(BATCH):
            optimizer.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits(x[batch], shift[batch]), y[batch], pos_weight=weight)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite hybrid training loss')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
        epochs_run += 1
        model.eval()
        with torch.no_grad():
            value = float(nn.functional.binary_cross_entropy_with_logits(
                logits(vx, vshift), vy))
        if value < best_loss - 1e-5:
            best_loss, stale = value, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    if best_state is None:
        raise ValueError('Hybrid training produced no finite validation loss')
    model.load_state_dict(best_state)
    return dict(architecture=name, features=list(features), scaler=scaler, offset=bool(offset),
        balanced=bool(balanced), offset_scaler=offset_scaler,
        network={k: v.detach().cpu() for k, v in model.state_dict().items()},
        epochs_run=epochs_run, validation_log_loss=best_loss,
        parameters=sum(p.numel() for p in model.parameters()))


def network_raw(bundle, frame):
    """Logits before calibration, including the physical offset when the model uses one."""
    torch.set_num_threads(2)
    model = Perceptron(len(bundle['features']))
    model.load_state_dict(bundle['network'])
    model.eval()
    x = _encoded(bundle['scaler'], frame, bundle['features'])
    shift = _offset(bundle['offset_scaler'], frame)
    outputs = []
    with torch.no_grad():
        for batch in torch.arange(len(frame)).split(4096):
            outputs.append((model(x[batch]) + shift[batch]).numpy())
    return np.concatenate(outputs) if outputs else np.zeros(0)
