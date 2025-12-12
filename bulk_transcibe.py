import numpy as np
from multiprocessing import Pool
import pandas as pd
import os
import glob
from pathlib import Path
from collections import defaultdict
import gc


BASE_DEST = r'C:\local_data\futures_compressed'
CHUNK_SIZE_JSON = 1_000_000      # tune based on memory
CHUNK_SIZE_NPY  = 1_000_000    # rows per chunk for .npy


def convert_json_to_npz(path: str):
    p = Path(path)
    use_cols = ["b", "B", "a", "A", "T"]

    # keep T as int64 during reading so we can do safe int arithmetic
    dtypes = {
        "b": "float32",
        "B": "float32",
        "a": "float32",
        "A": "float32",
        "T": "int64",
    }

    # -------- First pass: count rows and get T0 --------
    n_rows = 0
    t0 = None

    reader1 = pd.read_json(
        path,
        lines=True,
        dtype=dtypes,
        chunksize=CHUNK_SIZE_JSON
    )

    for chunk in reader1:
        if len(chunk) == 0:
            continue

        if t0 is None:
            # file is already time-ordered; first row is min T
            t0 = int(chunk["T"].iloc[0])

        n_rows += len(chunk)

    if n_rows == 0:
        return  # empty file, nothing to do

    # -------- Allocate final arrays --------
    timestamps = np.empty(n_rows, dtype=np.int32)       # relative ms
    features   = np.empty((n_rows, 4), dtype=np.float32)  # b, B, a, A

    # -------- Second pass: fill arrays --------
    reader2 = pd.read_json(
        path,
        lines=True,
        dtype=dtypes,
        chunksize=CHUNK_SIZE_JSON
    )

    start = 0
    for chunk in reader2:
        if len(chunk) == 0:
            continue

        chunk = chunk[use_cols]

        # timestamps
        t64 = chunk["T"].to_numpy(dtype=np.int64)
        rel = t64 - t0

        if rel.max() > np.iinfo(np.int32).max or rel.min() < 0:
            raise ValueError(f"Relative time out of int32 range for file {path}")

        end = start + len(chunk)
        timestamps[start:end] = rel.astype(np.int32)

        # features in [b, B, a, A] order
        features[start:end, 0] = chunk["b"].to_numpy(dtype=np.float32)
        features[start:end, 1] = chunk["B"].to_numpy(dtype=np.float32)
        features[start:end, 2] = chunk["a"].to_numpy(dtype=np.float32)
        features[start:end, 3] = chunk["A"].to_numpy(dtype=np.float32)

        start = end

        del chunk, t64, rel
        gc.collect()

    coin_path = path.split('\\')
    coin_name = coin_path[-2] # Gets the name of the coin
    dest = os.path.join(BASE_DEST, coin_name)
    if not os.path.exists(dest):
        os.mkdir(dest)
    out_path = os.path.join(BASE_DEST, coin_name, p.stem + ".npz")
    np.savez(out_path, t=timestamps, x=features)

    del timestamps, features
    gc.collect()


def convert_npy_to_npz(path: str):
    p = Path(path)

    # mmap: we don't pull the whole array into RAM at once
    arr = np.load(path, mmap_mode='r')

    if arr.ndim != 2 or arr.shape[1] != 5:
        raise ValueError(f".npy file {path} has unexpected shape {arr.shape}, expected (N, 5)")

    n_rows = arr.shape[0]

    # first row's T is the min timestamp, by your guarantee
    # assuming T is in column 4 (0-based index)
    # if stored as float64, round to nearest ms integer
    t0 = int(round(float(arr[0, 4])))

    timestamps = np.empty(n_rows, dtype=np.int32)
    features   = np.empty((n_rows, 4), dtype=np.float32)

    # Process in row chunks to control peak memory
    for start in range(0, n_rows, CHUNK_SIZE_NPY):
        end = min(start + CHUNK_SIZE_NPY, n_rows)

        # sub is a view into the memmapped array (cheap)
        sub = arr[start:end]

        # T column: convert to int64 ms and make relative
        t64 = np.rint(sub[:, 4]).astype(np.int64)  # rint for safety if stored as float
        rel = t64 - t0

        if rel.max() > np.iinfo(np.int32).max or rel.min() < 0:
            raise ValueError(f"Relative time out of int32 range for file {path}")

        timestamps[start:end] = rel.astype(np.int32)

        # features: columns [0:4] -> float32
        features[start:end, :] = sub[:, 0:4].astype(np.float32)

        del sub, t64, rel
        gc.collect()

    coin_path = path.split('\\')
    coin_name = coin_path[-2] # Gets the name of the coin
    dest = os.path.join(BASE_DEST, coin_name)
    if not os.path.exists(dest):
        os.mkdir(dest)
    out_path = os.path.join(BASE_DEST, coin_name, p.stem + ".npz")
    np.savez(out_path, t=timestamps, x=features)

    del timestamps, features, arr
    gc.collect()


def file_processor(path: str):
    try:
        p = Path(path)

        if p.suffix == ".npy":
            convert_npy_to_npz(path)
        else:
            # assume JSON / NDJSON
            convert_json_to_npz(path)
    except Exception as e:
        print("error in processing", path)
        print("error : ", e)


if __name__ == "__main__":
    # find all source files
    folders = glob.glob(r'\\desktop-ditjgjb\E\data\futures\*')
    print(folders)
    records = []
    for folder in folders:
        records.append(glob.glob(folder + "\\*"))
        
    flat_records = []
    for sublist in records:
        for item in sublist:
            flat_records.append(item)
    
    print(flat_records)
    print("Number of records", len(flat_records))

    
    # de-dup logic: keep one file per base name
    groups = defaultdict(list)
    for f in flat_records:
        p = Path(f)
        base = p.stem
        groups[base].append(p)

    kept = []
    for base, paths in groups.items():
        npy_files = [p for p in paths if p.suffix == ".npy"]
        if npy_files:
            # if any .npy exists for that base, treat those as the source
            kept.extend(npy_files)
        else:
            kept.extend(paths)

    kept = [str(p) for p in kept]
    print("Files to process:", len(kept))

    # Important: keep processes low until you're sure memory usage is stable
    with Pool(processes=20) as pool:
        pool.map(file_processor, kept)

    print("done")
    