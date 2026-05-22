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
        amp   = data["amp"].squeeze()   if "amp"   in data else None
        lam   = data["lam"].squeeze()   if "lam"   in data else None
        phase = data["phase"].squeeze() if "phase" in data else None

    elif ext in (".h5", ".hdf5"):
        with h5py.File(data_path, "r") as hf:
            X = torch.from_numpy(hf["inputs"][:].astype(np.float32))
            Y = torch.from_numpy(hf["outputs"][:].astype(np.float32))
            uin = hf["uin"][:] if "uin" in hf else None
            amp   = hf["amp"][:]   if "amp"   in hf else None
            lam   = hf["lam"][:]   if "lam"   in hf else None
            phase = hf["phase"][:] if "phase" in hf else None

    return X, Y, uin, amp, lam, phase

def load_data(path):
    data = scipy.io.loadmat(path)
    return (torch.from_numpy(data["inputs"].astype("float32")),
            torch.from_numpy(data["outputs"].astype("float32")))