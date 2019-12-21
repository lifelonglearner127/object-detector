import numpy as np
import h5py


def dump_dataset(data, labels, path, dataset_name, method="w"):
    db = h5py.File(path, method)
    dataset = db.create_dataset(dataset_name, (len(data), len(data[0]) + 1), dtype="float")
    dataset[0:len(data)] = np.c_[labels, data]
    db.close()


def load_dataset(path, dataset_name):
    db = h5py.File(path, "r")
    (labels, data) = (db[dataset_name][:, 0], db[dataset_name][:, 1:])
    db.close()
    return (data, labels)
