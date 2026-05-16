"""Shared LSTM model and loss — single source of truth for train / app / learn / CLI."""
import torch
import torch.nn as nn

# Default hyperparameters (aligned with learn.py pre-training)
INPUT_DIM = 4
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.2
LOSS_ALPHA = 1.0
LOSS_BETA = 3.0
RANDOM_SEED = 42


class FastDelayPredictLSTM(nn.Module):
    """
    Per-stage delay in log-space (mu, logvar).
    Features: [W_um, CL_fF, VDD_V, stage_pos].
    """

    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.layernorm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dim, 2)

    def forward(self, x, lengths=None):
        proj = self.input_proj(x)
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                proj, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True, total_length=proj.size(1)
            )
        else:
            lstm_out, _ = self.lstm(proj)
        out = self.layernorm(lstm_out + proj)
        out = self.dropout(out)
        out = self.output_layer(out)
        return out[:, :, 0], out[:, :, 1]


def loss_function(mu, logvar, y_stage_log, y_total, lengths, alpha=LOSS_ALPHA, beta=LOSS_BETA):
    """
    lengths: (B,) actual stage counts per chain.
    y_total: (B,) linear-domain total delay targets.
    """
    device = mu.device
    b, seq_len = mu.size()
    mask = torch.arange(seq_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)

    logvar_clamped = torch.clamp(logvar, min=-10.0, max=10.0)
    var_full = torch.exp(logvar_clamped)

    mu_m = mu[mask]
    logvar_m = logvar_clamped[mask]
    precision_m = (1.0 / (var_full + 1e-12))[mask]
    y_m = y_stage_log[mask]

    nll = 0.5 * (logvar_m + precision_m * (mu_m - y_m) ** 2)
    stage_nll = torch.sum(nll) / (mask.sum().float() + 1e-12)

    pred_stage = torch.exp(mu + 0.5 * var_full)
    pred_total = torch.sum(pred_stage * mask.float(), dim=1)
    mse_total = nn.functional.mse_loss(pred_total, y_total)

    loss = alpha * stage_nll + beta * mse_total
    return loss, float(stage_nll.detach().cpu()), float(mse_total.detach().cpu())
