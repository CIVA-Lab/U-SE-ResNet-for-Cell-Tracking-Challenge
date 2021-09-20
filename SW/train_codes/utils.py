def is_ctc_dataset_3d(dataset_name):
    """
    returns True if the CTC dataset is 3D (e.g. Fluo-C3DH-H157)
    otherwise False (e.g. BF-C2DL-MuSC)
    """
    return dataset_name.split('-')[1].find('3D') != -1
