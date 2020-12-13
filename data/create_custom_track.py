import numpy as np
import scipy.interpolate
import scipy.spatial.transform
import open3d as o3d
import time
from pathlib import Path
import itertools
import multiprocessing
import argparse
import sys

from create_data_common import (
    render_depth_maps_mesh,
    render_depth_maps_pcd,
    load_depth_maps,
    imread,
    write_im_scaled,
    rotm_from_quat,
    translation_from_cameracenter,
    combine_counts,
)

sys.path.append("../")
import co
import ext


class TrackCreator(object):
    def __init__(self, geoms, srcRs=None, srcts=None):
        self.geoms = geoms
        self.srcRs = srcRs
        self.srcts = srcts

    def playback_waypoints(self, wpRs, wpts, ims=None, nbs=None):
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Playback", visible=True)
        for geom in self.geoms:
            vis.add_geometry(geom)
        ctr = vis.get_view_control()
        ctr.set_constant_z_near(0.1)
        cam = ctr.convert_to_pinhole_camera_parameters()
        for view_idx, R, t in zip(itertools.count(), wpRs, wpts):
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t
            cam.extrinsic = T
            ctr.convert_from_pinhole_camera_parameters(cam)
            vis.poll_events()
            vis.update_renderer()
            if ims is None or nbs is None:
                time.sleep(0.1)
            else:
                print(f"  playback frame {view_idx+1}/{len(wpRs)}")
                nb_ims = [ims[nbs[view_idx, nb_idx]] for nb_idx in range(8)]
                nb_ims = co.plt2d.image_matrix(nb_ims)
                plt.figure()
                plt.imshow(nb_ims)
                co.plt.tight_no_ticks()
                plt.show()
        vis.destroy_window()

    def create_waypoints(self, Rs=None, ts=None):
        if Rs is None:
            Rs = []
        else:
            Rs = [R for R in Rs]
        if ts is None:
            ts = []
        else:
            ts = [t for t in ts]
        callback_data = {"run": True, "Rs": Rs, "ts": ts, "play": False}

        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name="Create Waypoints", visible=True)
        for geom in self.geoms:
            vis.add_geometry(geom)

        ctr = vis.get_view_control()
        ctr.set_constant_z_near(0.1)
        # cam = ctr.convert_to_pinhole_camera_parameters()

        def vis_quit(vis):
            callback_data["run"] = False
            callback_data["play"] = False
            return False

        vis.register_key_callback(ord("Q"), vis_quit)
        print("print [q] to quit waypoint selection")

        def vis_quit_play(vis):
            callback_data["run"] = False
            callback_data["play"] = True
            return False

        vis.register_key_callback(ord("P"), vis_quit_play)
        print("print [p] to playback waypoints")

        def vis_record(vis):
            ctr = vis.get_view_control()
            cam = ctr.convert_to_pinhole_camera_parameters()
            T = cam.extrinsic
            R = T[:3, :3]
            t = T[:3, 3]
            callback_data["Rs"].append(R)
            callback_data["ts"].append(t)
            print(f'recorded waypoint #{len(callback_data["Rs"])}')

        vis.register_key_callback(ord("R"), vis_record)
        print("print [r] to record waypoint")

        def vis_drop_last(vis):
            callback_data["Rs"].pop()
            callback_data["ts"].pop()
            print(f'droped last waypoint #{len(callback_data["Rs"]) + 1}')

        vis.register_key_callback(ord("D"), vis_drop_last)
        print("print [d] to drop last recorded waypoint")

        def vis_goto_last_camera(vis):
            Rs = callback_data["Rs"]
            ts = callback_data["ts"]
            if len(Rs) == 0 or len(ts) == 0:
                return
            T = np.eye(4)
            T[:3, :3] = Rs[-1]
            T[:3, 3] = ts[-1]
            ctr = vis.get_view_control()
            cam = ctr.convert_to_pinhole_camera_parameters()
            cam.extrinsic = T
            ctr.convert_from_pinhole_camera_parameters(cam)

        vis.register_key_callback(ord("G"), vis_goto_last_camera)
        print("print [g] to return to last waypoint")

        src_idx = [0]

        def _vis_goto_next_src_camera(vis, step):
            src_idx[0] = src_idx[0] + step
            if src_idx[0] < 0:
                src_idx[0] = self.srcRs.shape[0] - 1
            elif src_idx[0] >= self.srcRs.shape[0]:
                src_idx[0] = 0
            print(f"gone to src camera {src_idx[0]}")
            R = self.srcRs[src_idx[0]]
            t = self.srcts[src_idx[0]]
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t
            ctr = vis.get_view_control()
            cam = ctr.convert_to_pinhole_camera_parameters()
            cam.extrinsic = T
            ctr.convert_from_pinhole_camera_parameters(cam)

        def vis_goto_next_src_camera_pstep1(vis):
            _vis_goto_next_src_camera(vis, 1)

        def vis_goto_next_src_camera_pstep10(vis):
            _vis_goto_next_src_camera(vis, 10)

        def vis_goto_next_src_camera_mstep1(vis):
            _vis_goto_next_src_camera(vis, -1)

        def vis_goto_next_src_camera_mstep10(vis):
            _vis_goto_next_src_camera(vis, -10)

        if self.srcRs is not None and self.srcts is not None:
            vis.register_key_callback(ord("J"), vis_goto_next_src_camera_pstep1)
            vis.register_key_callback(ord("K"), vis_goto_next_src_camera_mstep1)
            vis.register_key_callback(
                ord("N"), vis_goto_next_src_camera_pstep10
            )
            vis.register_key_callback(
                ord("M"), vis_goto_next_src_camera_mstep10
            )
            print("print [j], [k], [n], [m] to go to src camera")

        # camT = ctr.convert_to_pinhole_camera_parameters().extrinsic
        # cams = cams_ls(camT[:3, :3], camT[:3, 3], self.srcRs, self.srcts)
        # vis_cams.add_geometry(cams)
        while callback_data["run"]:
            vis.poll_events()
            vis.update_renderer()

        #     vis_cams.remove_geometry(cams)
        #     camT = ctr.convert_to_pinhole_camera_parameters().extrinsic
        #     cams = cams_ls(camT[:3, :3], camT[:3, 3], self.srcRs, self.srcts)
        #     vis_cams.add_geometry(cams)
        #     # vis_cams.update_geometry()
        #     vis_cams.update_renderer()
        vis.destroy_window()
        # vis_cams.destroy_window()

        return callback_data["play"], callback_data["Rs"], callback_data["ts"]

    def run(self, wpRs=None, wpts=None):
        run = True
        while run:
            run, wpRs, wpts = self.create_waypoints(wpRs, wpts)
            if run:
                self.playback_waypoints(*interpolate(wpRs, wpts))
        return wpRs, wpts


def cameracenter_from_translation(R, t):
    t = t.reshape(-1, 3, 1)
    R = R.reshape(-1, 3, 3)
    C = -R.transpose(0, 2, 1) @ t
    return C.squeeze()


def translation_from_cameracenter(R, C):
    C = C.reshape(-1, 3, 1)
    R = R.reshape(-1, 3, 3)
    t = -R @ C
    return t.squeeze()


def rot_x(x, dtype=np.float32):
    x = np.array(x, copy=False)
    x = x.reshape(-1, 1)
    R = np.zeros((x.shape[0], 3, 3), dtype=dtype)
    R[:, 0, 0] = 1
    R[:, 1, 1] = np.cos(x).ravel()
    R[:, 1, 2] = -np.sin(x).ravel()
    R[:, 2, 1] = np.sin(x).ravel()
    R[:, 2, 2] = np.cos(x).ravel()
    return R.squeeze()


def rot_y(y, dtype=np.float32):
    y = np.array(y, copy=False)
    y = y.reshape(-1, 1)
    R = np.zeros((y.shape[0], 3, 3), dtype=dtype)
    R[:, 0, 0] = np.cos(y).ravel()
    R[:, 0, 2] = np.sin(y).ravel()
    R[:, 1, 1] = 1
    R[:, 2, 0] = -np.sin(y).ravel()
    R[:, 2, 2] = np.cos(y).ravel()
    return R.squeeze()


def rot_z(z, dtype=np.float32):
    z = np.array(z, copy=False)
    z = z.reshape(-1, 1)
    R = np.zeros((z.shape[0], 3, 3), dtype=dtype)
    R[:, 0, 0] = np.cos(z).ravel()
    R[:, 0, 1] = -np.sin(z).ravel()
    R[:, 1, 0] = np.sin(z).ravel()
    R[:, 1, 1] = np.cos(z).ravel()
    R[:, 2, 2] = 1
    return R.squeeze()


def rotm_from_xyz(xyz):
    xyz = np.array(xyz, copy=False).reshape(-1, 3)
    return (rot_x(xyz[:, 0]) @ rot_y(xyz[:, 1]) @ rot_z(xyz[:, 2])).squeeze()


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


def quat_from_rotm(R):
    R = R.reshape(-1, 3, 3)
    w = np.sqrt(np.maximum(0, 1 + R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]))
    x = np.sqrt(np.maximum(0, 1 + R[:, 0, 0] - R[:, 1, 1] - R[:, 2, 2]))
    y = np.sqrt(np.maximum(0, 1 - R[:, 0, 0] + R[:, 1, 1] - R[:, 2, 2]))
    z = np.sqrt(np.maximum(0, 1 - R[:, 0, 0] - R[:, 1, 1] + R[:, 2, 2]))
    q0 = np.empty((R.shape[0], 4), dtype=R.dtype)
    q0[:, 0] = w
    q0[:, 1] = x * np.sign(x * (R[:, 2, 1] - R[:, 1, 2]))
    q0[:, 2] = y * np.sign(y * (R[:, 0, 2] - R[:, 2, 0]))
    q0[:, 3] = z * np.sign(z * (R[:, 1, 0] - R[:, 0, 1]))
    q1 = np.empty((R.shape[0], 4), dtype=R.dtype)
    q1[:, 0] = w * np.sign(w * (R[:, 2, 1] - R[:, 1, 2]))
    q1[:, 1] = x
    q1[:, 2] = y * np.sign(y * (R[:, 1, 0] + R[:, 0, 1]))
    q1[:, 3] = z * np.sign(z * (R[:, 0, 2] + R[:, 2, 0]))
    q2 = np.empty((R.shape[0], 4), dtype=R.dtype)
    q2[:, 0] = w * np.sign(w * (R[:, 0, 2] - R[:, 2, 0]))
    q2[:, 1] = x * np.sign(x * (R[:, 0, 1] + R[:, 1, 0]))
    q2[:, 2] = y
    q2[:, 3] = z * np.sign(z * (R[:, 1, 2] + R[:, 2, 1]))
    q3 = np.empty((R.shape[0], 4), dtype=R.dtype)
    q3[:, 0] = w * np.sign(w * (R[:, 1, 0] - R[:, 0, 1]))
    q3[:, 1] = x * np.sign(x * (R[:, 0, 2] + R[:, 2, 0]))
    q3[:, 2] = y * np.sign(y * (R[:, 1, 2] + R[:, 2, 1]))
    q3[:, 3] = z
    q = q0 * (w[:, None] > 0) + (w[:, None] == 0) * (
        q1 * (x[:, None] > 0)
        + (x[:, None] == 0) * (q2 * (y[:, None] > 0) + (y[:, None] == 0) * (q3))
    )
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return q.squeeze()


def quat_slerp_space(q0, q1, t=None, num=100, endpoint=True):
    q0 = q0.reshape(-1, 4)
    q1 = q1.reshape(-1, 4)
    dot = (q0 * q1).sum(axis=1)

    ma = dot < 0
    q1[ma] *= -1
    dot[ma] *= -1

    if t is None:
        t = np.linspace(0, 1, num=num, endpoint=endpoint, dtype=q0.dtype)
    t = t.reshape((-1, 1))
    num = t.shape[0]

    res = np.empty((q0.shape[0], num, 4), dtype=q0.dtype)

    ma = dot > 0.9995
    if np.any(ma):
        res[ma] = (q0[ma] + t[..., None] * (q1[ma] - q0[ma])).transpose(1, 0, 2)

    ma = ~ma
    if np.any(ma):
        q0 = q0[ma]
        q1 = q1[ma]
        dot = dot[ma]

        dot = np.clip(dot, -1, 1)
        theta0 = np.arccos(dot)
        theta = theta0 * t
        s0 = np.cos(theta) - dot * np.sin(theta) / np.sin(theta0)
        s1 = np.sin(theta) / np.sin(theta0)
        res[ma] = ((s0[..., None] * q0) + (s1[..., None] * q1)).transpose(
            1, 0, 2
        )
    return res.squeeze()


def cameras_lineset(Rs, ts, size=10, color=(0.5, 0.5, 0)):
    points = []
    lines = []
    colors = []

    for R, t in zip(Rs, ts):
        C0 = cameracenter_from_translation(R, t).reshape(3, 1)
        cam_points = []
        cam_points.append(C0)
        cam_points.append(
            C0 + R.T @ np.array([-size, -size, 3.0 * size]).reshape(3, 1)
        )
        cam_points.append(
            C0 + R.T @ np.array([+size, -size, 3.0 * size]).reshape(3, 1)
        )
        cam_points.append(
            C0 + R.T @ np.array([+size, +size, 3.0 * size]).reshape(3, 1)
        )
        cam_points.append(
            C0 + R.T @ np.array([-size, +size, 3.0 * size]).reshape(3, 1)
        )
        cam_points = np.concatenate(
            [pt.reshape(1, 3) for pt in cam_points], axis=0
        )
        cam_lines = np.array(
            [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]
        )

        points.append(cam_points)
        lines.append(len(lines) * 5 + cam_lines)
        colors.extend([color for _ in range(8)])

    points = np.concatenate(points, axis=0)
    lines = np.concatenate(lines, axis=0)
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    ls.colors = o3d.utility.Vector3dVector(np.array(colors, dtype=np.float64))
    return ls


def interpolate_waypoints(wpRs, wpts, steps=25):
    wpRs = np.array(wpRs)
    wpts = np.array(wpts)
    wpqs = quat_from_rotm(wpRs)
    wpCs = cameracenter_from_translation(wpRs, wpts)
    qs, Cs = [], []
    for idx in range(wpRs.shape[0] - 1):
        q0, q1 = wpqs[idx], wpqs[idx + 1]
        C0, C1 = wpCs[idx], wpCs[idx + 1]
        alphas = np.linspace(0, 1, num=steps, endpoint=False)
        Cs.append((1 - alphas[:, None]) * C0 + alphas[:, None] * C1)
        qs.append(quat_slerp_space(q0, q1, t=alphas))
    Rs = rotm_from_quat(np.vstack(qs))
    ts = translation_from_cameracenter(Rs, np.vstack(Cs))
    return Rs, ts


def interpolate_waypoints_spline(wpRs, wpts, steps=25, k=None, s=0):
    wpRs = np.array(wpRs)
    wpts = np.array(wpts)
    wpqs = quat_from_rotm(wpRs)
    wpCs = cameracenter_from_translation(wpRs, wpts)

    # uu = np.linalg.norm(wpCs[1:] - wpCs[:-1], axis=1)
    # uu = np.cumsum(uu) / np.sum(uu)

    if k is None:
        k = 2 if wpCs.shape[0] <= 3 else 3
    tck, u = scipy.interpolate.splprep(wpCs.T, k=k, s=s)
    Cs = scipy.interpolate.splev(np.linspace(0, 1, num=steps), tck)
    Cs = np.array(Cs).T

    dists = [0] + [v for v in np.linalg.norm(Cs[1:] - Cs[:-1], axis=1)]
    dists = np.cumsum(dists)
    dists_inv = np.interp(
        np.linspace(dists[0], dists[-1], len(dists)),
        dists,
        np.linspace(0, 1, len(dists)),
    )
    Cs = scipy.interpolate.splev(dists_inv, tck)
    Cs = np.array(Cs).T

    rotations = scipy.spatial.transform.Rotation.from_matrix(wpRs)
    rot_spline = scipy.spatial.transform.RotationSpline(u, rotations)
    Rs = rot_spline(dists_inv).as_matrix()

    ts = translation_from_cameracenter(Rs, np.vstack(Cs))
    return Rs, ts


def process_count(
    out_path, tgt_dm, tgt_K, tgt_R, tgt_t, src_dms, src_K, src_R, src_t
):
    print(f"process {out_path}")
    count = ext.preprocess.count_nbs(
        tgt_dm,
        tgt_K,
        tgt_R,
        tgt_t,
        src_dms,
        src_Ks,
        src_Rs,
        src_ts,
        bwd_depth_thresh=0.1,
    )
    np.save(out_path, count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", type=str, required=True)
    parser.add_argument("-d", "--pw-dir", type=str, required=True)
    parser.add_argument(
        "-v",
        "--vis-cams",
        type=float,
        default=0.1,
        help="use 10 for DTU, or 0.1 for T&T",
    )
    parser.add_argument("-s", "--steps", type=int, default=200)
    parser.add_argument("--create-track", type=co.utils.str2bool, default=False)
    args = parser.parse_args()

    pw_dir = Path(args.pw_dir)

    src_Ks = np.load(pw_dir / "Ks.npy")
    src_Rs = np.load(pw_dir / "Rs.npy")
    src_ts = np.load(pw_dir / "ts.npy")

    src_dm_paths = sorted(pw_dir.glob("dm_*.npy"))
    src_dms = load_depth_maps(src_dm_paths)
    height, width = src_dms.shape[-2:]

    mesh_path = pw_dir.parent / "delaunay_photometric.ply"
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color((0.7, 0.7, 0.7))

    out_dir = pw_dir.parent / f"ibr3d_{args.name}"
    out_dir.mkdir(exist_ok=True, parents=True)
    print(f"create out_dir {out_dir}")

    waypoint_Rs_path = out_dir / "waypoint_Rs.npy"
    waypoint_ts_path = out_dir / "waypoint_ts.npy"
    waypoint_Rs, waypoint_ts = None, None
    if waypoint_Rs_path.exists() and waypoint_ts_path.exists():
        print("load existing waypoints")
        waypoint_Rs = np.load(waypoint_Rs_path)
        waypoint_ts = np.load(waypoint_ts_path)
    if waypoint_Rs is None or args.create_track:
        print("create waypoints")
        track = TrackCreator([mesh], src_Rs, src_ts)
        waypoint_Rs, waypoint_ts = track.run(None, None)
        np.save(waypoint_Rs_path, waypoint_Rs)
        np.save(waypoint_ts_path, waypoint_ts)

    waypoint_Rs, waypoint_ts = interpolate_waypoints_spline(
        waypoint_Rs, waypoint_ts, steps=args.steps
    )
    waypoint_Ks = np.array([src_Ks[0] for _ in waypoint_Rs])
    np.save(out_dir / "Rs.npy", waypoint_Rs)
    np.save(out_dir / "Ks.npy", waypoint_Ks)
    np.save(out_dir / "ts.npy", waypoint_ts)

    if args.vis_cams > 0:
        cam_size = args.vis_cams
        o3d.visualization.draw_geometries(
            [
                mesh,
                cameras_lineset(src_Rs, src_ts, cam_size, (0.0, 0.85, 0.5)),
                cameras_lineset(
                    waypoint_Rs, waypoint_ts, cam_size, (0.85, 0.5, 0)
                ),
            ]
        )

    print("render waypoint depth maps")
    tgt_dm_paths = render_depth_maps_mesh(
        out_dir,
        mesh_path,
        waypoint_Ks,
        waypoint_Rs,
        waypoint_ts,
        height,
        width,
        write_vis=True,
    )
    tgt_dms = load_depth_maps(tgt_dm_paths)

    print("compute counts")
    run_tic = time.time()
    with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
        args = []
        for idx, tgt_dm, tgt_K, tgt_R, tgt_t in zip(
            itertools.count(), tgt_dms, waypoint_Ks, waypoint_Rs, waypoint_ts
        ):
            count_path = out_dir / f"count_{idx:08d}.npy"
            if not count_path.exists():
                args.append(
                    (
                        count_path,
                        tgt_dm,
                        tgt_K,
                        tgt_R,
                        tgt_t,
                        src_dms,
                        src_Ks,
                        src_Rs,
                        src_ts,
                    )
                )
        p.starmap(process_count, args)
    combine_counts(out_dir)
    print(f"processing count took {time.time() - run_tic}[s]")
