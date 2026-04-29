import os
import scipy.io
import h5py
import torch
import numpy as np
from scipy.io import loadmat

def load_dataset(data_path: str):
    ext = os.path.splitext(data_path)[-1].lower()

    if ext == ".mat":
        data = loadmat(data_path)
        X = torch.from_numpy(data["inputs"].astype(np.float32))
        Y = torch.from_numpy(data["outputs"].astype(np.float32))
        uin = data["uin"].squeeze() if "uin" in data else None

    elif ext in (".h5", ".hdf5"):
        with h5py.File(data_path, "r") as hf:
            X = torch.from_numpy(hf["inputs"][:].astype(np.float32))
            Y = torch.from_numpy(hf["outputs"][:].astype(np.float32))
            uin = hf["uin"][:] if "uin" in hf else None

    return X, Y, uin

def load_data(path):
    data = scipy.io.loadmat(path)
    return (torch.from_numpy(data["inputs"].astype("float32")),
            torch.from_numpy(data["outputs"].astype("float32")))