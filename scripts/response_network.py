"""Small CPU response experts; import only inside the neural worker."""
import resource
import time

import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler

from scripts.response_research import SIGNATURE, onset_categories
from scripts.hydrology_research import COMBINED, SNOW

DYNAMIC = [f for f in COMBINED + SNOW if f not in ['latitude', 'longitude', 'elevation_m', 'log_area']]
CONTEXT = SIGNATURE + ['log_area', 'elevation_m']


class ResponseMixture(nn.Module):
    def __init__(self, dynamic_size, context_size, experts, conditioned):
        super().__init__()
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(dynamic_size, 32), nn.SiLU(),
            nn.Linear(32, 32), nn.SiLU(), nn.Linear(32, 7)) for _ in range(experts)])
        self.gate = nn.Linear(context_size, experts) if conditioned else None

    def forward(self, dynamic, context):
        outputs = torch.stack([expert(dynamic) for expert in self.experts], dim=1)
        weights = self.gate(context).softmax(1) if self.gate is not None else torch.full(
            outputs.shape[:2], 1 / len(self.experts), device=dynamic.device)
        probabilities = (outputs[:, :, :4].softmax(2) * weights[:, :, None]).sum(1)
        residual = (outputs[:, :, 4:] * weights[:, :, None]).sum(1)
        return probabilities, residual


def fit_network(name, frame):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(42)
    started = time.monotonic()
    scalers = [StandardScaler().fit(frame[columns]) for columns in [DYNAMIC, CONTEXT]]
    x, context = [torch.from_numpy(scaler.transform(frame[columns]).astype('float32'))
                  for scaler, columns in zip(scalers, [DYNAMIC, CONTEXT])]
    y = torch.from_numpy(onset_categories(frame)).long()
    targets = torch.from_numpy((frame[['flow_1', 'flow_2', 'flow_3']].to_numpy() - frame.q_3.to_numpy()[:, None]).astype('float32'))
    experts = 1 if name == 'pooled' else 4
    conditioned = name in ['conditioned', 'multitask']
    auxiliary = name in ['multitask', 'uniform']
    model = ResponseMixture(len(DYNAMIC), len(CONTEXT), experts, conditioned)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.001)
    generator = torch.Generator().manual_seed(42)
    for epoch in range(60):
        model.train()
        for batch in torch.randperm(len(frame), generator=generator).split(2048):
            optimizer.zero_grad()
            probabilities, residual = model(x[batch], context[batch])
            loss = nn.functional.nll_loss(probabilities.clamp_min(1e-7).log(), y[batch])
            if auxiliary:
                loss = loss + .5 * nn.functional.huber_loss(residual, targets[batch], delta=.2)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite neural training loss')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
        if (epoch + 1) % 20 == 0:
            print(name, 'epoch', epoch + 1, 'seconds', round(time.monotonic() - started, 1), flush=True)
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 6_000_000_000 or time.monotonic() - started > 900:
            raise RuntimeError('Frozen neural resource budget exceeded')
    return dict(network={k: value.detach().cpu() for k, value in model.state_dict().items()},
        experts=experts, conditioned=conditioned, scalers=scalers,
        parameters=sum(p.numel() for p in model.parameters()))


def network_output(bundle, frame):
    torch.set_num_threads(2)
    model = ResponseMixture(len(DYNAMIC), len(CONTEXT), bundle['experts'], bundle['conditioned'])
    model.load_state_dict(bundle['network']); model.eval()
    x, context = [torch.from_numpy(scaler.transform(frame[columns]).astype('float32'))
                  for scaler, columns in zip(bundle['scalers'], [DYNAMIC, CONTEXT])]
    probabilities, trajectories = [], []
    with torch.no_grad():
        for batch in torch.arange(len(frame)).split(4096):
            p, residual = model(x[batch], context[batch])
            probabilities.append(p.numpy()); trajectories.append(residual.numpy())
    return np.concatenate(probabilities), np.concatenate(trajectories) + frame.q_3.to_numpy()[:, None]
