"""Matched binary and onset-day neural fits; import only in isolated workers."""
import resource
import time

import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler

from scripts.response_network import ResponseMixture, DYNAMIC, CONTEXT
from scripts.response_research import onset_categories


def supervision_loss(probabilities, categories, name, reduction='mean'):
    if name == 'day':
        return nn.functional.nll_loss(probabilities.clamp_min(1e-7).log(), categories, reduction=reduction)
    if name == 'binary':
        return nn.functional.binary_cross_entropy((1 - probabilities[:, 0]).clamp(1e-7, 1 - 1e-7),
            categories.gt(0).float(), reduction=reduction)
    raise ValueError('Unknown supervision')


def fit_matched(name, frame, seed):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    started = time.monotonic()
    scalers = [StandardScaler().fit(frame[columns]) for columns in [DYNAMIC, CONTEXT]]
    x, context = [torch.from_numpy(scaler.transform(frame[columns]).astype('float32'))
                  for scaler, columns in zip(scalers, [DYNAMIC, CONTEXT])]
    y = torch.from_numpy(onset_categories(frame)).long()
    model = ResponseMixture(len(DYNAMIC), len(CONTEXT), 1, False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.001)
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(60):
        model.train()
        for batch in torch.randperm(len(frame), generator=generator).split(2048):
            optimizer.zero_grad()
            probabilities, _ = model(x[batch], context[batch])
            loss = supervision_loss(probabilities, y[batch], name)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
        if (epoch + 1) % 20 == 0:
            print(name, seed, 'epoch', epoch + 1, 'seconds', round(time.monotonic() - started, 1), flush=True)
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 6_000_000_000 or time.monotonic() - started > 900:
            raise RuntimeError('Frozen neural resource budget exceeded')
    return dict(network={k: value.detach().cpu() for k, value in model.state_dict().items()},
        experts=1, conditioned=False, scalers=scalers, seed=seed,
        parameters=sum(p.numel() for p in model.parameters()))
