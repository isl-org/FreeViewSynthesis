import numpy as np
import open3d as o3d
import cv2
import itertools
from tqdm import tqdm
from pathlib import Path
import sys

sys.path.append("../")
import co

import pyrender


def rotm_from_quat(q):
    q = q.reshape(-1, 4)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.array(
        [
            [
                1 - 2 * y * y - 2 * z * z,
                2 * x * y - 2 * z * w,
                2 * x * z + 2 * y * w,
            ],
            [
                2 * x * y + 2 * z * w,
                1 - 2 * x * x - 2 * z * z,
                2 * y * z - 2 * x * w,
            ],
            [
                2 * x * z - 2 * y * w,
                2 * y * z + 2 * x * w,
                1 - 2 * x * x - 2 * y * y,
            ],
        ],
        dtype=q.dtype,
    )
    R = R.transpose((2, 0, 1))
    return R.squeeze()


def translation_from_cameracenter(R, C):
    C = C.reshape(-1, 3, 1)
    R = R.reshape(-1, 3, 3)
    t = -R @ C
    return t.squeeze()


def imread(p, scale=1):
    im = cv2.imread(str(p))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    if scale != 1:
        im = cv2.resize(im, (0, 0), fx=scale, fy=scale)
    return im


def imwrite(p, im, scale255=True):
    if scale255:
        im = 255 * im
    im = np.clip(im, 0, 255).astype(np.uint8)
    cv2.imwrite(str(p), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))


def write_im_scaled(im_paths, scale, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, im_path in enumerate(im_paths):
        out_path = out_dir / f"im_{idx:08d}{im_path.suffix}"
        if not out_path.exists():
            im = imread(im_path, scale)
            imwrite(str(out_path), im, False)


def render_depth_maps_mesh(
    dm_dir,
    mesh_path,
    Ks,
    Rs,
    ts,
    height,
    width,
    znear=0.05,
    zfar=1500,
    write_vis=True,
):
    print(f"render depth maps to {dm_dir}")
    dm_dir.mkdir(parents=True, exist_ok=True)

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color((0.7, 0.7, 0.7))

    verts = np.asarray(mesh.vertices).astype(np.float32)
    faces = np.asarray(mesh.triangles).astype(np.int32)
    colors = np.asarray(mesh.vertex_colors).astype(np.float32)
    normals = np.asarray(mesh.vertex_normals).astype(np.float32)

    dm_paths = []
    for view_idx, K, R, t in zip(itertools.count(), tqdm(Ks), Rs, ts):
        dm_path = dm_dir / f"dm_{view_idx:08d}.npy"
        dm_paths.append(dm_path)
        if dm_path.exists():
            continue

        scene = pyrender.Scene()
        mesh = pyrender.Mesh(
            primitives=[
                pyrender.Primitive(
                    positions=verts,
                    normals=normals,
                    color_0=colors,
                    indices=faces,
                    mode=pyrender.GLTF.TRIANGLES,
                )
            ],
            is_visible=True,
        )
        mesh_node = pyrender.Node(mesh=mesh, matrix=np.eye(4))
        scene.add_node(mesh_node)

        cam = pyrender.IntrinsicsCamera(
            fx=K[0, 0],
            fy=K[1, 1],
            cx=K[0, 2],
            cy=K[1, 2],
            znear=znear,
            zfar=zfar,
        )
        T = np.eye(4)
        T[:3, :3] = R.T
        T[:3, 3] = (-R.T @ t.reshape(3, 1)).ravel()
        cv2gl = np.array(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        )
        T = T @ cv2gl
        cam_node = pyrender.Node(camera=cam, matrix=T)
        scene.add_node(cam_node)

        light = pyrender.DirectionalLight(color=np.ones(3), intensity=3)
        light_node = pyrender.Node(light=light, matrix=np.eye(4))
        scene.add_node(light_node, parent_node=cam_node)

        render = pyrender.OffscreenRenderer(width, height)
        color, depth = render.render(scene)
        np.save(dm_path, depth)

        if write_vis:
            depth[depth <= 0] = np.NaN
            depth = co.plt.image_colorcode(depth)
            imwrite(dm_path.with_suffix(".jpg"), depth)

    return dm_paths


def render_depth_maps_pcd(
    dm_dir,
    pcd_path,
    Ks,
    Rs,
    ts,
    height,
    width,
    znear=0.05,
    zfar=1500,
    write_vis=True,
):
    print(f"render depth maps to {dm_dir}")
    dm_dir.mkdir(parents=True, exist_ok=True)

    pcd = o3d.io.read_point_cloud(str(pcd_path))
    pcd.paint_uniform_color((0.7, 0.7, 0.7))

    verts = np.asarray(pcd.points).astype(np.float32)
    colors = np.asarray(pcd.colors).astype(np.float32)

    dm_paths = []
    for view_idx, K, R, t in zip(itertools.count(), tqdm(Ks), Rs, ts):
        dm_path = dm_dir / f"dm_{view_idx:08d}.npy"
        dm_paths.append(dm_path)
        if dm_path.exists():
            continue

        scene = pyrender.Scene()
        mesh = pyrender.Mesh.from_points(verts, colors=colors)
        mesh_node = pyrender.Node(mesh=mesh, matrix=np.eye(4))
        scene.add_node(mesh_node)

        cam = pyrender.IntrinsicsCamera(
            fx=K[0, 0],
            fy=K[1, 1],
            cx=K[0, 2],
            cy=K[1, 2],
            znear=znear,
            zfar=zfar,
        )
        T = np.eye(4)
        T[:3, :3] = R.T
        T[:3, 3] = (-R.T @ t.reshape(3, 1)).ravel()
        cv2gl = np.array(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        )
        T = T @ cv2gl
        cam_node = pyrender.Node(camera=cam, matrix=T)
        scene.add_node(cam_node)

        light = pyrender.DirectionalLight(color=np.ones(3), intensity=3)
        light_node = pyrender.Node(light=light, matrix=np.eye(4))
        scene.add_node(light_node, parent_node=cam_node)

        render = pyrender.OffscreenRenderer(width, height)
        color, depth = render.render(scene)
        np.save(dm_path, depth)

        if write_vis:
            depth[depth <= 0] = np.NaN
            depth = co.plt.image_colorcode(depth)
            imwrite(dm_path.with_suffix(".jpg"), depth)

    return dm_paths


def load_depth_maps(dm_paths):
    dms = []
    for dm_path in dm_paths:
        dms.append(np.load(dm_path))
    return np.array(dms)


def unproject_depths(dms, Ks, Rs, ts):
    Ks = np.array(Ks)
    Rs = np.array(Rs)
    ts = np.array(ts)
    height, width = dms.shape[1:]
    uu, vv = np.meshgrid(
        np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
    )
    pxc = 0.5
    uvh = np.stack((uu + pxc, vv + pxc, np.ones_like(uu)), axis=2)
    dm_mask = dms <= 0

    uvd = (dms[..., None] * uvh[None])[..., None]
    xyz = np.linalg.inv(Ks[:, None, None]) @ uvd
    xyz -= ts[:, None, None, :, None]
    xyz = Rs.transpose(0, 2, 1)[:, None, None] @ xyz

    xyz = xyz[~dm_mask]

    return xyz.reshape(-1, 3)


def combine_counts(root_dir):
    count_paths = sorted(Path(root_dir).glob("count_*.npy"))
    counts = []
    for count_path in count_paths:
        count = np.load(count_path)
        counts.append(count)
    counts = np.array(counts)
    np.save(root_dir / "counts.npy", counts)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    np.random.seed(42)

    mesh = o3d.geometry.TriangleMesh.create_box()
    mesh = mesh.translate((-0.5, -0.5, -0.5))
    mesh.paint_uniform_color((0.5, 0.5, 0.5))
    mesh.compute_vertex_normals()

    height, width = 100, 120
    focal = height
    K = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])
    Ks, Rs, ts = [], [], []
    for _ in range(10):
        Ks.append(K)
        Rs.append(co.geometry.rotm_random())
        ts.append(np.array([0, 0, 2.0]))

    dms = render_depth_maps(mesh, Ks, Rs, ts, height, width)
    # for dm in dms:
    #     plt.figure()
    #     plt.imshow(dm)
    # plt.show()

    xyz = unproject_depths(dms, Ks, Rs, ts)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.paint_uniform_color((1, 0, 0))
    o3d.visualization.draw_geometries([mesh, pcd])
