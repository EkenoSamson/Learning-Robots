import numpy as np
import h5py
import os
from tqdm import tqdm

def rotate_xy(vecs, theta):
    """
    Apply a 2D SO(2) rotation (around z-axis) to the x-y components of a batch of 3D vectors.
    """
    rot = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    vecs_xy = vecs[..., :2]  # only x, y
    vecs_rest = vecs[..., 2:]  # keep z or rest
    rotated_xy = np.dot(vecs_xy, rot.T)
    return np.concatenate([rotated_xy, vecs_rest], axis=-1)

def process_demo(hdf_grp, theta):
    """
    Apply rotation to specific keys within a single demo group.
    """
    for key in ["obs", "next_obs"]:
        # keys to rotate (we rotate only vectors with XY components)
        for obs_key in ["robot0_eef_pos", "robot0_eef_vel_lin", "robot0_eef_vel_ang"]:
            if obs_key in hdf_grp[key]:
                vec = hdf_grp[key][obs_key][:]
                rotated = rotate_xy(vec, theta)
                hdf_grp[key][obs_key][...] = rotated

    return hdf_grp

def make_equivariant_dataset(src_path, out_path):
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Source dataset not found: {src_path}")

    # Copy the original dataset to a new file
    with h5py.File(src_path, "r") as src_file, h5py.File(out_path, "w") as out_file:
        src_file.copy("/", out_file)
        print(f"Copied dataset to: {out_path}")

    # Re-open in write mode
    with h5py.File(out_path, "r+") as data_file:
        demos = [key for key in data_file.keys() if key.startswith("demo_")]
        print(f"Processing {len(demos)} demos for SO(2)-equivariance...")

        for demo_name in tqdm(demos):
            demo_grp = data_file[demo_name]
            theta = np.random.uniform(-np.pi, np.pi)  # random rotation angle
            process_demo(demo_grp, theta)

        print("✅ Done preprocessing. Dataset saved at:")
        print(f"    {out_path}")

# Example usage
if __name__ == "__main__":
    SRC = "~/robomimic/datasets/lift/ph/low_dim_v15.hdf5"
    DEST = "~/robomimic/datasets/lift/ph/low_dim_v15_so2.hdf5"
    make_equivariant_dataset(SRC, DEST)
