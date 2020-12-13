import numpy as np
from tqdm import tqdm
import time
import multiprocessing
from pathlib import Path
import argparse
import collections
import sys

from create_data_common import (
    render_depth_maps_mesh,
    load_depth_maps,
    imread,
    write_im_scaled,
    combine_counts,
)

sys.path.append("../")
import co
import ext


def compute_and_write_count_pw(count_path, idx, dms, Ks, Rs, ts):
    print(f"compute pw count {count_path}")
    count = ext.preprocess.count_nbs(
        dms[idx],
        Ks[idx],
        Rs[idx],
        ts[idx],
        dms,
        Ks,
        Rs,
        ts,
        bwd_depth_thresh=0.1,
    )
    count[idx] = 0
    np.save(count_path, count)


def run(dense_dir, scale, dm_write_vis=True):
    run_tic = time.time()
    dense_dir = Path(dense_dir)

    pw_dir = dense_dir / f"ibr3d_pw_{scale:.2f}"
    pw_dir.mkdir(parents=True, exist_ok=True)

    im_paths = []
    im_paths += sorted((dense_dir / "images").glob("*.png"))
    im_paths += sorted((dense_dir / "images").glob("*.jpg"))
    im_paths += sorted((dense_dir / "images").glob("*.jpeg"))
    print(f"found {len(im_paths)} images")

    print(f"write scaled input images if needed to {pw_dir}")
    write_im_scaled(im_paths, scale, pw_dir)

    im0 = imread(im_paths[0], scale)
    height, width = im0.shape[:2]

    Ks, Rs, ts = co.colmap.load_cameras(dense_dir / "sparse", im_paths, scale)

    print(f"write camera params to")
    np.save(pw_dir / "Ks.npy", Ks)
    np.save(pw_dir / "Rs.npy", Rs)
    np.save(pw_dir / "ts.npy", ts)

    mesh_path = dense_dir / "delaunay_photometric.ply"
    dm_paths = render_depth_maps_mesh(
        pw_dir, mesh_path, Ks, Rs, ts, height, width, write_vis=dm_write_vis
    )
    dms = load_depth_maps(dm_paths)

    print("preprocess")

    dms[dms <= 0] = 1e9

    # for idx in range(len(im_paths)):
    #     count_path = pw_dir / f"count_{idx:08d}.npy"
    #     if not count_path.exists():
    #         print(f"add {count_path} to list to process")
    #         compute_and_write_count_pw(count_path, idx, dms, Ks, Rs, ts)
    with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
        args = []
        for idx in range(len(im_paths)):
            count_path = pw_dir / f"count_{idx:08d}.npy"
            if not count_path.exists():
                print(f"add {count_path} to list to process")
                args.append((count_path, idx, dms, Ks, Rs, ts))
        p.starmap(compute_and_write_count_pw, args)

    print("combine counts")
    combine_counts(pw_dir)

    print(f"took {time.time() - run_tic}[s]")
