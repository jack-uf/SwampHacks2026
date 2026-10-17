"""FT-Transformer used by the transformer experiment.

Imported only by torch-enabled workers; importing this module imports torch.
The class lives at module level so fitted models round-trip through joblib.
"""
import copy

import numpy as np
import torch
from torch import nn

THREADS = 8


class FTTransformer(nn.Module):
    def __init__(self, n_features, d_token=32, n_blocks=2, n_heads=4, dropout=.1):
        super().__init__()
        torch.set_num_threads(THREADS)
        self.weight = nn.Parameter(torch.randn(n_features, d_token) * .02)
        self.bias = nn.Parameter(torch.zeros(n_features, d_token))
        self.cls = nn.Parameter(torch.zeros(1, 1, d_token))
        layer = nn.TransformerEncoderLayer(d_model=d_token, nhead=n_heads, dim_feedforward=2 * d_token,
                                           dropout=dropout, activation='relu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_blocks)
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, 1)

    def forward(self, x):
        tokens = x.unsqueeze(-1) * self.weight + self.bias
        tokens = torch.cat([self.cls.expand(x.shape[0], -1, -1), tokens], dim=1)
        return self.head(self.norm(self.encoder(tokens)[:, 0])).squeeze(-1)


class FTTransformerClassifier:
    """Frozen recipe from the protocol: no search, seed 42, early stop on the calibration rows."""

    def __init__(self, d_token=32, n_blocks=2, n_heads=4, dropout=.1, epochs=60, patience=10,
                 batch=256, lr=1e-3, weight_decay=1e-5, seed=42):
        self.kwargs = dict(d_token=d_token, n_blocks=n_blocks, n_heads=n_heads, dropout=dropout, epochs=epochs,
                           patience=patience, batch=batch, lr=lr, weight_decay=weight_decay, seed=seed)

    def fit(self, X, y, X_eval, y_eval):
        torch.set_num_threads(THREADS)
        torch.manual_seed(self.kwargs['seed'])
        X = np.asarray(X, dtype=float)
        self.mean_, self.std_ = X.mean(axis=0), X.std(axis=0) + 1e-9
        scaled, eval_scaled = self._scale(X), self._scale(np.asarray(X_eval, dtype=float))
        y = np.asarray(y, dtype=float)
        pos_weight = torch.tensor((len(y) - y.sum()) / max(y.sum(), 1.0))
        net = FTTransformer(X.shape[1], self.kwargs['d_token'], self.kwargs['n_blocks'],
                            self.kwargs['n_heads'], self.kwargs['dropout'])
        optimizer = torch.optim.AdamW(net.parameters(), lr=self.kwargs['lr'], weight_decay=self.kwargs['weight_decay'])
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        X_tensor, y_tensor = torch.tensor(scaled, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        eval_x = torch.tensor(eval_scaled, dtype=torch.float32)
        eval_y = torch.tensor(np.asarray(y_eval, dtype=float), dtype=torch.float32)
        rng = np.random.default_rng(self.kwargs['seed'])
        best_loss, best_state, stale, epochs_run = np.inf, None, 0, 0
        for _ in range(self.kwargs['epochs']):
            epochs_run += 1
            net.train()
            order = rng.permutation(len(X_tensor))
            for start in range(0, len(X_tensor), self.kwargs['batch']):
                rows = torch.tensor(order[start:start + self.kwargs['batch']], dtype=torch.long)
                optimizer.zero_grad()
                loss = criterion(net(X_tensor[rows]), y_tensor[rows])
                loss.backward()
                optimizer.step()
            net.eval()
            with torch.no_grad():
                eval_loss = float(nn.functional.binary_cross_entropy_with_logits(net(eval_x), eval_y))
            if eval_loss < best_loss - 1e-6:
                best_loss, best_state, stale = eval_loss, copy.deepcopy(net.state_dict()), 0
            else:
                stale += 1
                if stale >= self.kwargs['patience']:
                    break
        net.load_state_dict(best_state)
        self.net_, self.epochs_trained_, self.best_eval_loss_ = net, epochs_run, best_loss
        return self

    def _scale(self, X):
        return (X - self.mean_) / self.std_

    def predict_proba(self, X):
        torch.set_num_threads(THREADS)
        self.net_.eval()
        with torch.no_grad():
            logits = self.net_(torch.tensor(self._scale(np.asarray(X, dtype=float)), dtype=torch.float32))
            positive = torch.sigmoid(logits).numpy()
        return np.column_stack([1 - positive, positive])

