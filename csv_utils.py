"""CSV column normalization for training data."""
import pandas as pd

# Aliases -> canonical names used by RealInverterChainDataset
_COLUMN_ALIASES = {
    "Weight_um": "W_um",
    "weight_um": "W_um",
    "CL_ff": "CL_fF",
    "Vdd_V": "VDD_V",
    "delay_ps": "delay_ps",
    "total_delay": "total_delay_ps",
    "stage_delay": "stage_delay_ps",
}


def normalize_training_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Rename known column aliases; does not copy the frame unless needed."""
    rename = {c: _COLUMN_ALIASES[c] for c in df.columns if c in _COLUMN_ALIASES}
    if rename:
        df = df.rename(columns=rename)
    return df
