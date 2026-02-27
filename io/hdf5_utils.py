import os
import typing
import pandas as pd
from pathlib import Path

def save_to_hdf5(df: pd.DataFrame, output_path: typing.Union[str, bytes, os.PathLike], key: str = 'data'):
    """
    Save a pandas DataFrame to an HDF5 file.
    Requires the 'h5py' and 'tables' packages.
    """
    try:
        import tables # pandas to_hdf requires PyTables
    except ImportError:
        raise ImportError("Saving to HDF5 requires 'tables'. Please install it using 'pip install tables' or 'poetry install --extras io'")

    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_hdf(str(path), key=key, mode='w', format='table', data_columns=True)
    return str(path)

def load_from_hdf5(file_path: typing.Union[str, bytes, os.PathLike], key: str = 'data') -> pd.DataFrame:
    """
    Load a pandas DataFrame from an HDF5 file.
    """
    try:
        import tables
    except ImportError:
        raise ImportError("Loading from HDF5 requires 'tables'. Please install it using 'pip install tables' or 'poetry install --extras io'")

    path = Path(file_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {path}")
    return pd.read_hdf(str(path), key=key)

def handle_output(df: pd.DataFrame, return_type: str, output_path: typing.Optional[typing.Union[str, bytes, os.PathLike]], key: str = 'data') -> typing.Union[pd.DataFrame, str]:
    """
    Utility function to handle the standard I/O return formats.
    """
    if return_type.lower() == 'dataframe':
        return df
    elif return_type.lower() == 'hdf5':
        if not output_path:
            raise ValueError("output_path must be provided when return_type is 'hdf5'")
        return save_to_hdf5(df, output_path, key=key)
    else:
        raise ValueError(f"Unsupported return_type: {return_type}. Use 'dataframe' or 'hdf5'")
