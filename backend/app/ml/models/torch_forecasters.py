"""Modelos neuronales: LSTM, ConvLSTM, CNN-LSTM y ConvLSTM-Attention.

Implementados en PyTorch sobre CPU. El objetivo es incidencia semanal (conteo), por lo
que la pérdida es Poisson NLL y la salida se activa con softplus para garantizar
predicciones no negativas.

Las arquitecturas se declaran a nivel de módulo para que puedan serializarse con
``joblib`` (una clase definida dentro de ``fit`` no es importable y rompe el guardado).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.ml.models.base import BaseForecaster, ModelData

POISSON_EPS = 1e-6

try:  # pragma: no cover - depende de la instalación
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


def _require_torch():
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch no está instalado: los modelos neuronales no pueden entrenarse en este entorno."
        )
    return torch


# --------------------------------------------------------------------------------------
# Arquitecturas
# --------------------------------------------------------------------------------------
if TORCH_AVAILABLE:

    class LSTMNet(nn.Module):
        def __init__(self, n_features: int, hidden: int, layers: int, dropout: float) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                n_features, hidden, num_layers=layers, batch_first=True,
                dropout=dropout if layers > 1 else 0.0,
            )
            self.norm = nn.LayerNorm(hidden)
            self.head = nn.Sequential(
                nn.Linear(hidden, max(8, hidden // 2)), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(max(8, hidden // 2), 1),
            )

        def forward(self, seq):
            out, _ = self.lstm(seq)
            last = self.norm(out[:, -1])
            return F.softplus(self.head(last)).squeeze(-1)

    class ConvLSTMCell(nn.Module):
        """Celda ConvLSTM2D (Shi et al., 2015) construida con convoluciones 2D."""

        def __init__(self, in_ch: int, hidden_ch: int, kernel: int) -> None:
            super().__init__()
            self.conv = nn.Conv2d(in_ch + hidden_ch, 4 * hidden_ch, kernel, padding=kernel // 2)
            self.hidden_ch = hidden_ch

        def forward(self, x, state):
            h, c = state
            gates = self.conv(torch.cat([x, h], dim=1))
            i, f, o, g = gates.chunk(4, dim=1)
            c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
            h = torch.sigmoid(o) * torch.tanh(c)
            return h, c

        def init_state(self, batch: int, height: int, width: int, device: Optional[Any] = None):
            shape = (batch, self.hidden_ch, height, width)
            return (
                torch.zeros(shape, dtype=torch.float32, device=device),
                torch.zeros(shape, dtype=torch.float32, device=device),
            )

    class ConvLSTMStack(nn.Module):
        def __init__(self, in_ch: int, hidden_ch: int, layers: int, kernel: int) -> None:
            super().__init__()
            self.cells = nn.ModuleList(
                [ConvLSTMCell(in_ch if index == 0 else hidden_ch, hidden_ch, kernel) for index in range(layers)]
            )

        def forward(self, seq):
            batch, steps, _, height, width = seq.shape
            device = seq.device
            states = [cell.init_state(batch, height, width, device=device) for cell in self.cells]
            outputs: List[Any] = []
            for step in range(steps):
                x = seq[:, step]
                for index, cell in enumerate(self.cells):
                    hidden, cell_state = cell(x, states[index])
                    states[index] = (hidden, cell_state)
                    x = hidden
                outputs.append(x)
            return outputs
    class ConvLSTMNet(nn.Module):
        def __init__(self, in_ch: int, height: int, width: int, hidden: int, layers: int, kernel: int) -> None:
            super().__init__()
            self.stack = ConvLSTMStack(in_ch, hidden, layers, kernel)
            self.head = nn.Conv2d(hidden, 1, kernel_size=1)
            self.cell_embed = nn.Embedding(height * width, 8)
            self.bias = nn.Linear(8, 1)

        def forward(self, grid, cell):
            last = self.stack(grid)[-1]
            raw = self.head(last).flatten(1)
            emb = self.bias(self.cell_embed(cell))
            return F.softplus(raw.gather(1, cell.unsqueeze(1)).squeeze(1) + emb.squeeze(1))

    class CNNLSTMNet(nn.Module):
        def __init__(
            self, in_ch: int, height: int, width: int, filters: int, hidden: int,
            layers: int, dropout: float,
        ) -> None:
            super().__init__()
            pool_h, pool_w = max(1, height // 2), max(1, width // 2)
            self.encoder = nn.Sequential(
                nn.Conv2d(in_ch, filters, 3, padding=1), nn.BatchNorm2d(filters), nn.GELU(),
                nn.Conv2d(filters, filters, 3, padding=1), nn.GELU(),
                nn.AdaptiveAvgPool2d((pool_h, pool_w)),
            )
            feat_dim = filters * pool_h * pool_w
            self.lstm = nn.LSTM(
                feat_dim, hidden, num_layers=layers, batch_first=True,
                dropout=dropout if layers > 1 else 0.0,
            )
            self.cell_embed = nn.Embedding(height * width, 8)
            self.head = nn.Sequential(
                nn.LayerNorm(hidden + 8), nn.Linear(hidden + 8, max(8, hidden // 2)),
                nn.GELU(), nn.Dropout(dropout), nn.Linear(max(8, hidden // 2), 1),
            )

        def forward(self, grid, cell):
            batch, steps, channels, height_, width_ = grid.shape
            frames = grid.reshape(batch * steps, channels, height_, width_)
            features = self.encoder(frames).reshape(batch, steps, -1)
            out, _ = self.lstm(features)
            combined = torch.cat([out[:, -1], self.cell_embed(cell)], dim=1)
            return F.softplus(self.head(combined)).squeeze(-1)

    class ConvLSTMAttentionNet(nn.Module):
        def __init__(
            self, in_ch: int, height: int, width: int, hidden: int, layers: int,
            kernel: int, heads: int, dropout: float,
        ) -> None:
            super().__init__()
            self.stack = ConvLSTMStack(in_ch, hidden, layers, kernel)
            self.temporal_score = nn.Sequential(
                nn.Conv2d(hidden, hidden, 1), nn.Tanh(), nn.Conv2d(hidden, heads, 1)
            )
            self.spatial_score = nn.Conv2d(hidden, 1, kernel_size=1)
            self.dropout = nn.Dropout(dropout)
            self.head = nn.Conv2d(hidden, 1, kernel_size=1)
            self.cell_embed = nn.Embedding(height * width, 8)
            self.bias = nn.Linear(8, 1)

        def forward(self, grid, cell):
            outputs = self.stack(grid)                    # lista de (B, C, H, W)
            stack = torch.stack(outputs, dim=1)           # (B, T, C, H, W)
            batch, steps, channels, height, width = stack.shape
            # Atención temporal: puntaje por paso y por cabeza, promediado espacialmente.
            scores = self.temporal_score(stack.reshape(batch * steps, channels, height, width))
            scores = scores.reshape(batch, steps, -1, height * width).mean(-1)  # (B, T, heads)
            weights = torch.softmax(scores.mean(-1), dim=1)                     # (B, T)
            weighted = (stack * weights[:, :, None, None, None]).sum(1)         # (B, C, H, W)
            # Atención espacial: resalta las celdas relevantes para el pronóstico.
            spatial = torch.softmax(self.spatial_score(weighted).flatten(1), dim=1)
            raw = self.head(weighted).flatten(1)
            attended = (raw * spatial).sum(1)
            emb = self.bias(self.cell_embed(cell)).squeeze(1)
            return F.softplus(attended + emb)


# --------------------------------------------------------------------------------------
# Entrenamiento compartido
# --------------------------------------------------------------------------------------
class _TorchTrainerMixin:
    net = None

    def _train_loop(
        self,
        net,
        train_tensors: Dict[str, Any],
        val_tensors: Optional[Dict[str, Any]],
        params: Dict[str, Any],
        trial,
        max_epochs: int,
    ) -> List[Dict[str, float]]:
        torch = _require_torch()
        import torch.nn as nn

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net.to(device)
        optimizer = torch.optim.AdamW(
            net.parameters(),
            lr=float(params.get("lr", 1e-3)),
            weight_decay=float(params.get("weight_decay", 1e-4)),
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
        criterion = nn.PoissonNLLLoss(log_input=False, full=False, eps=POISSON_EPS)

        batch_size = int(params.get("batch_size", 64))
        total = int(len(train_tensors["y"]))
        indices = np.arange(total)
        rng = np.random.default_rng(int(params.get("seed", 42)))
        best_state = {key: value.detach().clone() for key, value in net.state_dict().items()}
        best_val = float("inf")
        patience = int(params.get("patience", 8))
        stale = 0
        history: List[Dict[str, float]] = []

        for epoch in range(1, max_epochs + 1):
            net.train()
            rng.shuffle(indices)
            epoch_loss = 0.0
            batches = 0
            for start in range(0, total, batch_size):
                batch_idx = indices[start : start + batch_size]
                inputs = {key: value[batch_idx].to(device) for key, value in train_tensors.items() if key != "y"}
                target = train_tensors["y"][batch_idx].to(device)
                optimizer.zero_grad(set_to_none=True)
                prediction = net(**inputs)
                loss = criterion(torch.clamp(prediction, min=POISSON_EPS), target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                optimizer.step()
                epoch_loss += float(loss.item())
                batches += 1

            record = {"epoch": float(epoch), "train_loss": round(epoch_loss / max(1, batches), 5)}
            monitored = record["train_loss"]
            if val_tensors is not None and len(val_tensors["y"]) > 0:
                net.eval()
                with torch.no_grad():
                    val_inputs = {key: value.to(device) for key, value in val_tensors.items() if key != "y"}
                    prediction = net(**val_inputs)
                from sklearn.metrics import mean_absolute_error

                val_mae = float(
                    mean_absolute_error(val_tensors["y"].numpy(), torch.clamp(prediction, min=0).detach().cpu().numpy())
                )
                record["val_mae"] = round(val_mae, 5)
                monitored = val_mae
                scheduler.step(val_mae)
            history.append(record)

            if monitored < best_val - 1e-6:
                best_val = monitored
                stale = 0
                best_state = {key: value.detach().clone() for key, value in net.state_dict().items()}
            else:
                stale += 1
                if stale >= patience:
                    break

            if trial is not None:
                from optuna import TrialPruned

                trial.report(monitored, step=epoch)
                if trial.should_prune():
                    raise TrialPruned()

        net.load_state_dict(best_state)
        self.net = net
        return history


def _to_tensors(data: ModelData, *, grid: bool) -> Dict[str, Any]:
    """Convierte las matrices NumPy a tensores en el orden esperado por PyTorch.

    Las series se almacenan como (N, T, H, W, C) porque es el orden natural de NumPy;
    las convoluciones esperan (N, T, C, H, W), por lo que se permutan aquí.
    """
    torch = _require_torch()
    out: Dict[str, Any] = {"y": torch.as_tensor(data.y, dtype=torch.float32)}
    if grid:
        tensor = torch.as_tensor(data.grid, dtype=torch.float32)
        out["grid"] = tensor.permute(0, 1, 4, 2, 3).contiguous()
        out["cell"] = torch.as_tensor(data.cell, dtype=torch.long)
    else:
        out["seq"] = torch.as_tensor(data.seq, dtype=torch.float32)
    return out


def _predict_grid(model, net, data: ModelData) -> np.ndarray:
    torch = _require_torch()
    device = next(net.parameters()).device if list(net.parameters()) else torch.device("cpu")

    net.eval()
    grid = torch.as_tensor(data.grid, dtype=torch.float32).permute(0, 1, 4, 2, 3).contiguous()
    cell = torch.as_tensor(data.cell, dtype=torch.long)
    outputs = []
    with torch.no_grad():
        for start in range(0, len(cell), 128):
            g = grid[start : start + 128].to(device)
            c = cell[start : start + 128].to(device)
            outputs.append(net(grid=g, cell=c).detach().cpu())
    return np.clip(torch.cat(outputs).numpy(), 0, None)


class LSTMForecaster(_TorchTrainerMixin, BaseForecaster):
    key = "lstm"
    name = "LSTM"
    family = "recurrent"
    requires = "sequence"

    def fit(self, train, val=None, *, params=None, trial=None, epochs=60):
        params = params or {}
        net = LSTMNet(
            int(train.seq.shape[2]),
            int(params.get("hidden_size", 64)),
            int(params.get("num_layers", 2)),
            float(params.get("dropout", 0.15)),
        )
        self._history = self._train_loop(
            net, _to_tensors(train, grid=False),
            _to_tensors(val, grid=False) if val is not None and len(val) else None,
            params, trial, int(params.get("max_epochs", epochs)),
        )
        return self

    def predict(self, data: ModelData) -> np.ndarray:
        torch = _require_torch()
        device = next(self.net.parameters()).device if list(self.net.parameters()) else torch.device("cpu")

        self.net.eval()
        with torch.no_grad():
            s = torch.as_tensor(data.seq, dtype=torch.float32).to(device)
            prediction = self.net(seq=s)
        return np.clip(prediction.detach().cpu().numpy(), 0, None)

    def training_history(self):
        return getattr(self, "_history", [])


class ConvLSTMForecaster(_TorchTrainerMixin, BaseForecaster):
    key = "convlstm"
    name = "ConvLSTM"
    family = "convolutional_recurrent"
    requires = "grid"

    def fit(self, train, val=None, *, params=None, trial=None, epochs=60):
        params = params or {}
        net = ConvLSTMNet(
            int(train.grid.shape[-1]), int(train.grid.shape[2]), int(train.grid.shape[3]),
            int(params.get("hidden_channels", 16)), int(params.get("num_layers", 2)),
            int(params.get("kernel_size", 3)),
        )
        self._history = self._train_loop(
            net, _to_tensors(train, grid=True),
            _to_tensors(val, grid=True) if val is not None and len(val) else None,
            params, trial, int(params.get("max_epochs", epochs)),
        )
        return self

    def predict(self, data: ModelData) -> np.ndarray:
        return _predict_grid(self, self.net, data)

    def training_history(self):
        return getattr(self, "_history", [])


class CNNLSTMForecaster(_TorchTrainerMixin, BaseForecaster):
    key = "cnn_lstm"
    name = "CNN-LSTM"
    family = "hybrid"
    requires = "grid"

    def fit(self, train, val=None, *, params=None, trial=None, epochs=60):
        params = params or {}
        net = CNNLSTMNet(
            int(train.grid.shape[-1]), int(train.grid.shape[2]), int(train.grid.shape[3]),
            int(params.get("cnn_filters", 16)), int(params.get("hidden_size", 64)),
            int(params.get("num_layers", 1)), float(params.get("dropout", 0.15)),
        )
        self._history = self._train_loop(
            net, _to_tensors(train, grid=True),
            _to_tensors(val, grid=True) if val is not None and len(val) else None,
            params, trial, int(params.get("max_epochs", epochs)),
        )
        return self

    def predict(self, data: ModelData) -> np.ndarray:
        return _predict_grid(self, self.net, data)

    def training_history(self):
        return getattr(self, "_history", [])


class ConvLSTMAttentionForecaster(_TorchTrainerMixin, BaseForecaster):
    key = "convlstm_attention"
    name = "ConvLSTM-Attention"
    family = "hybrid_attention"
    requires = "grid"

    def fit(self, train, val=None, *, params=None, trial=None, epochs=60):
        params = params or {}
        net = ConvLSTMAttentionNet(
            int(train.grid.shape[-1]), int(train.grid.shape[2]), int(train.grid.shape[3]),
            int(params.get("hidden_channels", 16)), int(params.get("num_layers", 2)),
            int(params.get("kernel_size", 3)), int(params.get("attention_heads", 2)),
            float(params.get("dropout", 0.1)),
        )
        self._history = self._train_loop(
            net, _to_tensors(train, grid=True),
            _to_tensors(val, grid=True) if val is not None and len(val) else None,
            params, trial, int(params.get("max_epochs", epochs)),
        )
        return self

    def predict(self, data: ModelData) -> np.ndarray:
        return _predict_grid(self, self.net, data)

    def training_history(self):
        return getattr(self, "_history", [])
