"""Small CPU perceptrons for the MLP onset experiment; import only inside the neural worker."""
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler

HIDDEN = 64
DROPOUT = .2
BLOCKS = 2
EPOCHS = 200
PATIENCE = 20
BATCH = 2048
LEARNING_RATE = 3e-3
WEIGHT_DECAY = 1e-4
SEED = 42
HORIZONS = [6, 12, 24]


class Direct(nn.Module):
    """Two hidden layers with a single output head per horizon."""

    def __init__(self, features, heads=1, hidden=HIDDEN, dropout=DROPOUT):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(features, hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(dropout))
        self.head = nn.Linear(hidden, heads)

    def forward(self, x):
        return self.head(self.trunk(x))


class Residual(nn.Module):
    """Input projection followed by additive residual blocks."""

    def __init__(self, features, heads=1, hidden=HIDDEN, dropout=DROPOUT, blocks=BLOCKS):
        super().__init__()
        self.input = nn.Linear(features, hidden)
        self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(hidden, hidden)) for _ in range(blocks)])
        self.activation = nn.SiLU()
        self.head = nn.Linear(hidden, heads)

    def forward(self, x):
        hidden = self.input(x)
        for block in self.blocks:
            hidden = self.activation(hidden + block(hidden))
        return self.head(hidden)


ARCHITECTURES = {'mlp_direct': Direct, 'mlp_residual': Residual, 'mlp_multitask': Direct}


def build(name, features):
    if name not in ARCHITECTURES:
        raise ValueError('Unknown MLP architecture')
    return ARCHITECTURES[name](features, heads=3 if name == 'mlp_multitask' else 1)


def _tensors(scaler, frame, features):
    return torch.from_numpy(scaler.transform(frame[features]).astype('float32'))


def _log_loss(logits, target):
    return nn.functional.binary_cross_entropy_with_logits(logits, target)


def fit_network(name, features, train, validation):
    """Train one architecture with early stopping on the held-out validation year."""
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(SEED)
    if train.empty or validation.empty or not np.isin(train['head'], [0, 1, 2]).all():
        raise ValueError('Require nonempty train and validation rows with a head column')
    scaler = StandardScaler().fit(train[features])
    x = _tensors(scaler, train, features)
    head = torch.from_numpy(train['head'].to_numpy()).long()
    y = torch.from_numpy(train['target'].to_numpy(dtype='float32'))
    vx = _tensors(scaler, validation, features)
    vhead = torch.from_numpy(validation['head'].to_numpy()).long()
    vy = torch.from_numpy(validation['target'].to_numpy(dtype='float32'))
    model = build(name, len(features))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    generator = torch.Generator().manual_seed(SEED)
    best_loss, best_state, stale, epochs_run = np.inf, None, 0, 0
    for _ in range(EPOCHS):
        model.train()
        for batch in torch.randperm(len(train), generator=generator).split(BATCH):
            optimizer.zero_grad()
            logits = model(x[batch])
            picked = logits.gather(1, head[batch][:, None]).squeeze(1)
            loss = _log_loss(picked, y[batch])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite MLP training loss')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
        epochs_run += 1
        model.eval()
        with torch.no_grad():
            logits = model(vx)
            value = float(_log_loss(logits.gather(1, vhead[:, None]).squeeze(1), vy))
        if value < best_loss - 1e-5:
            best_loss, stale = value, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    if best_state is None:
        raise ValueError('MLP training produced no finite validation loss')
    model.load_state_dict(best_state)
    return dict(architecture=name, features=list(features), scaler=scaler,
        network={k: v.detach().cpu() for k, v in model.state_dict().items()},
        heads=3 if name == 'mlp_multitask' else 1, epochs_run=epochs_run,
        validation_log_loss=best_loss, parameters=sum(p.numel() for p in model.parameters()))


def network_probability(bundle, frame, horizon):
    """Calibrator-ready probabilities for rows that all share one horizon."""
    torch.set_num_threads(2)
    model = build(bundle['architecture'], len(bundle['features']))
    model.load_state_dict(bundle['network'])
    model.eval()
    x = _tensors(bundle['scaler'], frame, bundle['features'])
    heads = bundle['heads']
    head = 0 if heads == 1 else HORIZONS.index(horizon)
    outputs = []
    with torch.no_grad():
        for batch in torch.arange(len(frame)).split(4096):
            logits = model(x[batch])
            picked = logits[:, head]
            outputs.append(torch.sigmoid(picked).numpy())
    if not outputs:
        return np.zeros(0)
    return np.concatenate(outputs)


def horizon_heads(frame, horizons=HORIZONS):
    """Stack per-horizon rows with the head index each row trains against."""
    parts = []
    for horizon in horizons:
        part = frame.loc[frame.horizon.eq(horizon)].copy()
        part['head'] = HORIZONS.index(horizon)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)
