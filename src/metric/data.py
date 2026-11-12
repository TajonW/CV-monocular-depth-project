"""NYU labeled MATLAB data with the official 795/654 split."""

import h5py
import numpy as np
from scipy.io import loadmat


def load_splits(path, total=1449):
    values = loadmat(path)
    train, test = (np.asarray(values[key]).reshape(-1).astype(np.int64) - 1
                   for key in ("trainNdxs", "testNdxs"))
    if len(train) != 795 or len(test) != 654:
        raise ValueError("Expected official 795 development / 654 test images")
    joined = np.concatenate((train, test))
    if len(np.unique(joined)) != total or not np.array_equal(np.sort(joined), np.arange(total)):
        raise ValueError("Invalid, duplicate, overlapping or incomplete MATLAB split indices")
    return train, test


class NYU:
    def __init__(self, path):
        self.file = h5py.File(path, "r")
        if self.file["images"].shape != (1449, 3, 640, 480) or self.file["depths"].shape != (1449, 640, 480):
            self.file.close()
            raise ValueError("Unexpected NYU labeled HDF5 layout")

    def get(self, index):
        rgb = np.asarray(self.file["images"][int(index)]).transpose(2, 1, 0).copy()
        depth = np.asarray(self.file["depths"][int(index)]).T.copy()
        if rgb.dtype != np.uint8 or depth.dtype != np.float32:
            raise ValueError("Unexpected NYU data types")
        return rgb, depth

    def close(self):
        self.file.close()
