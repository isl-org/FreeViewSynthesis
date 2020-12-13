# Based on https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
# New BSD license

import numpy as np
import itertools
import subprocess
import shutil
import struct
from pathlib import Path
import argparse
import collections
import sqlite3
import multiprocessing
import sys

try:
    from . import plt
    from . import sty
except:
    import plt
    import sty


CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"]
)
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"]
)
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"]
)
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"]
)


class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12),
}
CAMERA_MODEL_IDS = dict(
    [(camera_model.model_id, camera_model) for camera_model in CAMERA_MODELS]
)
CAMERA_MODEL_NAMES = dict(
    [(camera_model.model_name, camera_model) for camera_model in CAMERA_MODELS]
)


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_cameras_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)
    """
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(
                    id=camera_id,
                    model=model,
                    width=width,
                    height=height,
                    params=params,
                )
    return cameras


def read_cameras_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for camera_line_index in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ"
            )
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(
                fid,
                num_bytes=8 * num_params,
                format_char_sequence="d" * num_params,
            )
            cameras[camera_id] = Camera(
                id=camera_id,
                model=model_name,
                width=width,
                height=height,
                params=np.array(params),
            )
        assert len(cameras) == num_cameras
    return cameras


def read_images_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack(
                    [
                        tuple(map(float, elems[0::3])),
                        tuple(map(float, elems[1::3])),
                    ]
                )
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = Image(
                    id=image_id,
                    qvec=qvec,
                    tvec=tvec,
                    camera_id=camera_id,
                    name=image_name,
                    xys=xys,
                    point3D_ids=point3D_ids,
                )
    return images


def read_images_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for image_index in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi"
            )
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":  # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q"
            )[0]
            x_y_id_s = read_next_bytes(
                fid,
                num_bytes=24 * num_points2D,
                format_char_sequence="ddq" * num_points2D,
            )
            xys = np.column_stack(
                [
                    tuple(map(float, x_y_id_s[0::3])),
                    tuple(map(float, x_y_id_s[1::3])),
                ]
            )
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=image_name,
                xys=xys,
                point3D_ids=point3D_ids,
            )
    return images


def read_points3D_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    points3D = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                point3D_id = int(elems[0])
                xyz = np.array(tuple(map(float, elems[1:4])))
                rgb = np.array(tuple(map(int, elems[4:7])))
                error = float(elems[7])
                image_ids = np.array(tuple(map(int, elems[8::2])))
                point2D_idxs = np.array(tuple(map(int, elems[9::2])))
                points3D[point3D_id] = Point3D(
                    id=point3D_id,
                    xyz=xyz,
                    rgb=rgb,
                    error=error,
                    image_ids=image_ids,
                    point2D_idxs=point2D_idxs,
                )
    return points3D


def read_points3d_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for point_line_index in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd"
            )
            point3D_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q"
            )[0]
            track_elems = read_next_bytes(
                fid,
                num_bytes=8 * track_length,
                format_char_sequence="ii" * track_length,
            )
            image_ids = np.array(tuple(map(int, track_elems[0::2])))
            point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
            points3D[point3D_id] = Point3D(
                id=point3D_id,
                xyz=xyz,
                rgb=rgb,
                error=error,
                image_ids=image_ids,
                point2D_idxs=point2D_idxs,
            )
    return points3D


def array_to_blob(array):
    if sys.version_info[0] >= 3:
        return array.tostring()
    else:
        return np.getbuffer(array)


def blob_to_array(blob, dtype, shape=(-1,)):
    if sys.version_info[0] >= 3:
        return np.fromstring(blob, dtype=dtype).reshape(*shape)
    else:
        return np.frombuffer(blob, dtype=dtype).reshape(*shape)


def read_array(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(
            fid, delimiter="&", max_rows=1, usecols=(0, 1, 2), dtype=int
        )
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()


def write_array(p, arr):
    arr
    shape_str = [f"{s}&" for s in [arr.shape[1], arr.shape[0], *arr.shape[2:]]]
    while len(shape_str) < 3:
        shape_str.append("1&")
    shape_str = "".join(shape_str)
    with open(p, "wb") as fp:
        fp.write(shape_str.encode("latin1"))
        arr.tofile(fp)


def write_next_bytes(fid, data, format_char_sequence, endian_character="<"):
    if isinstance(data, (list, tuple)):
        bytes = struct.pack(endian_character + format_char_sequence, *data)
    else:
        bytes = struct.pack(endian_character + format_char_sequence, data)
    fid.write(bytes)


def write_cameras_text(out_path, cameras):
    with open(out_path, "w") as fp:
        for id, camera in cameras.items():
            params = [str(v) for v in camera.params]
            fp.write(
                f'{camera.id} {camera.model} {camera.width} {camera.height} {" ".join(params)}\n'
            )


def write_cameras_binary(out_path, cameras):
    with open(out_path, "wb") as fid:
        write_next_bytes(fid, len(cameras), "Q")
        for _, cam in cameras.items():
            model_id = CAMERA_MODEL_NAMES[cam.model].model_id
            camera_properties = [cam.id, model_id, cam.width, cam.height]
            write_next_bytes(fid, camera_properties, "iiQQ")
            for p in cam.params:
                write_next_bytes(fid, float(p), "d")
    return cameras


def write_images_text(out_path, images, image_callback=None):
    with open(out_path, "w") as fp:
        for id, image in images.items():
            if image_callback is not None:
                image = image_callback(image)
            fp.write(
                f"{image.id} {image.qvec[0]} {image.qvec[1]} {image.qvec[2]} {image.qvec[3]} {image.tvec[0]} {image.tvec[1]} {image.tvec[2]} {image.camera_id} {image.name}\n"
            )
            for xy, pid in zip(image.xys, image.point3D_ids):
                fp.write(f"{xy[0]} {xy[1]} {pid} ")
            fp.write("\n")


def write_images_binary(out_path, images):
    with open(out_path, "wb") as fid:
        write_next_bytes(fid, len(images), "Q")
        for _, img in images.items():
            write_next_bytes(fid, img.id, "i")
            write_next_bytes(fid, img.qvec.tolist(), "dddd")
            write_next_bytes(fid, img.tvec.tolist(), "ddd")
            write_next_bytes(fid, img.camera_id, "i")
            for char in img.name:
                write_next_bytes(fid, char.encode("utf-8"), "c")
            write_next_bytes(fid, b"\x00", "c")
            write_next_bytes(fid, len(img.point3D_ids), "Q")
            for xy, p3d_id in zip(img.xys, img.point3D_ids):
                write_next_bytes(fid, [*xy, p3d_id], "ddq")


def write_points3D_text(out_path, points3D):
    with open(out_path, "w") as fp:
        for id, point in points3D.items():
            fp.write(
                f"{point.id} {point.xyz[0]} {point.xyz[1]} {point.xyz[2]} {point.rgb[0]} {point.rgb[1]} {point.rgb[2]} {point.error}"
            )
            for image_id, point2d_idx in zip(
                point.image_ids, point.point2D_idxs
            ):
                fp.write(f" {image_id} {point2d_idx}")
            fp.write("\n")


def write_points3D_binary(out_path, points3D):
    with open(out_path, "wb") as fid:
        write_next_bytes(fid, len(points3D), "Q")
        for _, pt in points3D.items():
            write_next_bytes(fid, pt.id, "Q")
            write_next_bytes(fid, pt.xyz.tolist(), "ddd")
            write_next_bytes(fid, pt.rgb.tolist(), "BBB")
            write_next_bytes(fid, pt.error, "d")
            track_length = pt.image_ids.shape[0]
            write_next_bytes(fid, track_length, "Q")
            for image_id, point2D_id in zip(pt.image_ids, pt.point2D_idxs):
                write_next_bytes(fid, [image_id, point2D_id], "ii")


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
    # # http://muri.materials.cmu.edu/wp-content/uploads/2015/06/RotationPaperRevised.pdf
    # https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
    # http://www.iri.upc.edu/files/scidoc/2083-A-Survey-on-the-Computation-of-Quaternions-from-Rotation-Matrices.pdf
    # https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2015/01/matrix-to-quat.pdf
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


class Colmap(object):
    def __init__(self, colmap_path, working_dir, im_paths):
        self.colmap_path = colmap_path

        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)

        if len(im_paths) > 0:
            self.im_dir = Path(im_paths[0]).parent
            for im_path in im_paths:
                if Path(im_path).parent != self.im_dir:
                    raise Exception(
                        "all images in im_path must be in the same folder"
                    )
        self.im_paths = im_paths

        self.db_path = self.working_dir / "database.db"
        self.sparse_dir = self.working_dir / "sparse"
        self.sparse_dir_0 = self.working_dir / "sparse" / "0"
        self.dense_dir = self.working_dir / "dense"
        self.camera_model = "RADIAL"
        self.single_camera = False
        self.mapper_min_num_matches = 15
        self.max_im_size = -1
        self.mapper_min_num_matches = 15
        self.patch_match_min_triangulation_angle = 1
        self.patch_match_geometric = False
        self.patch_match_window_radius = 5
        self.patch_match_filter = 1
        self.patch_match_filter_min_ncc = 0.1
        self.patch_match_depth_min = 0
        self.patch_match_depth_max = 20
        self.fusion_min_num_pixels = 5
        self.fusion_max_reproj_error = 2
        self.fusion_max_depth_error = 0.1

        self.openmvs_bin_dir = Path("")
        self.openmvs_working_dir = self.working_dir / "openmvs"
        self.openmvs_densify_resolution_level = 0
        self.openmvs_densify_max_resolution = 3200
        self.openmvs_densify_min_resolution = 640
        self.openmvs_densify_number_views = 12
        self.openmvs_densify_optimize = 1

    def _exec_cmd(self, cmd):
        print(f'{sty.bold}{sty.fg002}{" ".join(cmd)}{sty.rs}')
        subprocess.run(cmd)

    def _fusion_output_path(self):
        if self.patch_match_geometric:
            return self.dense_dir / "fused_geometric.ply"
        else:
            return self.dense_dir / "fused_photometric.ply"

    def _poisson_output_path(self):
        if self.patch_match_geometric:
            return self.dense_dir / "poisson_geometric.ply"
        else:
            return self.dense_dir / "poisson_photometric.ply"

    def _delaunay_output_path(self):
        if self.patch_match_geometric:
            return self.dense_dir / "delaunay_geometric.ply"
        else:
            return self.dense_dir / "delaunay_photometric.ply"

    def feature_extractor(self):
        cmd = [
            str(self.colmap_path),
            "feature_extractor",
            "--database_path",
            str(self.db_path),
            "--image_path",
            str(self.im_dir),
            # "--image_list_path",
            # str(self.image_list_path),
            "--ImageReader.camera_model",
            self.camera_model,
            "--ImageReader.single_camera",
            "1" if self.single_camera else "0",
            # "--ImageReader.existing_camera_id",
            # "1",
            "--SiftExtraction.use_gpu",
            "1",
        ]
        self._exec_cmd(cmd)

    def feature_matcher(self):
        cmd = [
            str(self.colmap_path),
            "exhaustive_matcher",
            "--database_path",
            str(self.db_path),
            "--SiftMatching.use_gpu",
            "1",
        ]
        self._exec_cmd(cmd)

    def mapper(self):
        self.sparse_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(self.colmap_path),
            "mapper",
            "--database_path",
            str(self.db_path),
            "--image_path",
            str(self.im_dir),
            # "--image_list_path",
            # str(self.image_list_path),
            "--output_path",
            str(self.sparse_dir),
        ]
        self._exec_cmd(cmd)

    def point_triangulator(self):
        self.sparse_dir_0.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(self.colmap_path),
            "point_triangulator",
            "--database_path",
            str(self.db_path),
            "--image_path",
            str(self.im_dir),
            "--input_path",
            str(self.sparse_dir),
            "--output_path",
            str(self.sparse_dir_0),
            "--Mapper.min_num_matches",
            str(self.mapper_min_num_matches),
        ]
        self._exec_cmd(cmd)

    def image_undistorter(self):
        self.dense_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(self.colmap_path),
            "image_undistorter",
            "--image_path",
            str(self.im_dir),
            "--input_path",
            str(self.sparse_dir_0),
            "--output_path",
            str(self.dense_dir),
            "--output_type",
            "COLMAP",
            "--max_image_size",
            str(self.max_im_size),
        ]
        self._exec_cmd(cmd)

    def write_patch_match_cfg(self, patch_match_cfg_path, im_associations={}):
        # im_associations .. dict of im_name => list of im_names
        # patch_match_cfg_path = self.dense_dir / 'stereo' / 'patch-match.cfg'
        with open(patch_match_cfg_path, "w") as fp:
            for k, v in im_associations.items():
                fp.write(f"{k}\n")
                fp.write(f'{", ".join(v)}\n')

    def patch_match_stereo_manual(self, im_associations={}):
        self.write_patch_match_cfg(
            self.dense_dir / "stereo" / "patch-match.cfg", im_associations
        )
        cmd = [
            str(self.colmap_path),
            "patch_match_stereo",
            "--workspace_path",
            str(self.dense_dir),
            "--workspace_format",
            "COLMAP",
            "--PatchMatchStereo.depth_min",
            str(self.patch_match_depth_min),
            "--PatchMatchStereo.depth_max",
            str(self.patch_match_depth_max),
            "--PatchMatchStereo.min_triangulation_angle",
            str(self.patch_match_min_triangulation_angle),
            "--PatchMatchStereo.max_image_size",
            str(self.max_im_size),
            "--PatchMatchStereo.geom_consistency",
            "true" if self.patch_match_geometric else "false",
            "--PatchMatchStereo.window_radius",
            str(self.patch_match_window_radius),
            "--PatchMatchStereo.filter",
            str(self.patch_match_filter),
            "--PatchMatchStereo.filter_min_ncc",
            str(self.patch_match_filter_min_ncc),
        ]
        self._exec_cmd(cmd)

    def patch_match_stereo_auto(self):
        cmd = [
            str(self.colmap_path),
            "patch_match_stereo",
            "--workspace_path",
            str(self.dense_dir),
            "--workspace_format",
            "COLMAP",
            "--PatchMatchStereo.max_image_size",
            str(self.max_im_size),
            "--PatchMatchStereo.min_triangulation_angle",
            str(self.patch_match_min_triangulation_angle),
            "--PatchMatchStereo.geom_consistency",
            "true" if self.patch_match_geometric else "false",
            "--PatchMatchStereo.window_radius",
            str(self.patch_match_window_radius),
            "--PatchMatchStereo.filter",
            str(self.patch_match_filter),
            "--PatchMatchStereo.filter_min_ncc",
            str(self.patch_match_filter_min_ncc),
        ]
        self._exec_cmd(cmd)

    def stereo_fusion(self):
        cmd = [
            str(self.colmap_path),
            "stereo_fusion",
            "--workspace_path",
            str(self.dense_dir),
            "--workspace_format",
            "COLMAP",
            "--input_type",
            "geometric" if self.patch_match_geometric else "photometric",
            "--output_path",
            str(self._fusion_output_path()),
            "--StereoFusion.min_num_pixels",
            str(self.fusion_min_num_pixels),
            "--StereoFusion.max_reproj_error",
            str(self.fusion_max_reproj_error),
            "--StereoFusion.max_depth_error",
            str(self.fusion_max_depth_error),
        ]
        self._exec_cmd(cmd)

    def poisson_meshing(self):
        cmd = [
            str(self.colmap_path),
            "poisson_mesher",
            "--input_path",
            str(self._fusion_output_path()),
            "--output_path",
            str(self._poisson_output_path()),
        ]
        self._exec_cmd(cmd)

    def delaunay_meshing(self):
        fused_path = self.dense_dir / "fused.ply"
        fused_vis_path = self.dense_dir / "fused.ply.vis"
        fusion_out_path = self._fusion_output_path()
        if not fused_path.exists():
            shutil.copy(fusion_out_path, fused_path)
        if not fused_vis_path.exists():
            shutil.copy(fusion_out_path.with_suffix(".ply.vis"), fused_vis_path)
        cmd = [
            str(self.colmap_path),
            "delaunay_mesher",
            "--input_path",
            str(self.dense_dir),
            "--output_path",
            str(self._delaunay_output_path()),
        ]
        self._exec_cmd(cmd)
        fused_path.unlink()
        fused_vis_path.unlink()

    def select_images(self):
        conn = sqlite3.Connection(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT image_id, name, camera_id FROM images")
        rows = cursor.fetchall()
        images = {}
        for row in rows:
            images[row[1]] = {
                "image_id": row[0],
                "image": row[1],
                "camera_id": row[2],
            }
        conn.close()
        return images

    def write_calib(self, Ks, Ts, imdims):
        # dims ... N x 2 (width, height)
        if self.camera_model != "PINHOLE":
            raise Exception(
                f"camera_model={self.camera_model} has to be PINHOLE"
            )
        self.sparse_dir.mkdir(parents=True, exist_ok=True)

        images = self.select_images()

        conn = sqlite3.Connection(str(self.db_path))
        cursor = conn.cursor()
        # cursor.execute('SELECT * FROM cameras')
        # rows = cursor.fetchall()
        cursor.execute("DELETE FROM cameras")
        conn.commit()
        with open(self.sparse_dir / "cameras.txt", "w") as fp:
            for idx, im_path, K, dim in zip(
                itertools.count(), self.im_paths, Ks, imdims
            ):
                image = images[im_path.name]
                cam_id = image["camera_id"]
                camera_model = 1  # PINHOLE
                prior_focal_length = False
                fp.write(
                    f"{cam_id} PINHOLE {int(dim[0])} {int(dim[1])} {K[0,0]} {K[1,1]} {K[0,2]} {K[1,2]}\n"
                )
                cursor.execute(
                    "INSERT OR REPLACE INTO cameras VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        cam_id,
                        camera_model,
                        int(dim[0]),
                        int(dim[1]),
                        array_to_blob(
                            np.array(
                                (K[0, 0], K[1, 1], K[0, 2], K[1, 2]),
                                dtype=np.float64,
                            )
                        ),
                        prior_focal_length,
                    ),
                )
                conn.commit()
        conn.close()

        with open(self.sparse_dir / "images.txt", "w") as fp:
            for idx, im_path, T in zip(itertools.count(), self.im_paths, Ts):
                R, t = T[:3, :3], T[:3, 3]
                q = quat_from_rotm(R)
                image = images[im_path.name]
                fp.write(
                    f"{image['image_id']} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} {image['camera_id']} {im_path.name}\n\n"
                )
        with open(self.sparse_dir / "points3D.txt", "w") as fp:
            pass

    def sparse_reconstruction_known_calib(self, Ks, Ts, imdims):
        # Ts     ... [Rs | ts]
        # imdims ... N x 2 (width, height)
        self.feature_extractor()
        self.write_calib(Ks, Ts, imdims)
        self.feature_matcher()
        self.point_triangulator()

    def sparse_reconstruction_unknown_calib(self):
        self.feature_extractor()
        self.feature_matcher()
        self.mapper()

    def dense_reconstruction(self):
        self.image_undistorter()
        self.patch_match_stereo_auto()
        self.stereo_fusion()

    def write_sparse_points_ply(self):
        import open3d as o3d
        points3D_path = self.sparse_dir_0 / "points3D.bin"

        pts = read_points3d_binary(str(points3D_path))
        n = len(pts)
        xyz = np.empty((n, 3))
        colors = np.empty((n, 3))
        for idx, key in enumerate(pts.keys()):
            xyz[idx] = pts[key].xyz
            colors[idx] = pts[key].rgb.astype(colors.dtype) / 255

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(str(points3D_path.with_suffix(".ply")), pcd)

    def _openmvs_mvs_project_path(self):
        return self.openmvs_working_dir / "open.mvs"

    def openmvs_init(self):
        cameras = read_cameras_binary(self.dense_dir / "sparse" / "cameras.bin")
        images = read_images_binary(self.dense_dir / "sparse" / "images.bin")
        points3D = read_points3d_binary(
            self.dense_dir / "sparse" / "points3D.bin"
        )

        def image_callback(image):
            return BaseImage(
                image.id,
                image.qvec,
                image.tvec,
                image.camera_id,
                image.name.replace(".jpg", ".png"),
                image.xys,
                image.point3D_ids,
            )

        write_cameras_text(self.dense_dir / "sparse" / "cameras.txt", cameras)
        write_images_text(
            self.dense_dir / "sparse" / "images.txt",
            images,
            image_callback=image_callback,
        )
        write_points3D_text(
            self.dense_dir / "sparse" / "points3D.txt", points3D
        )

        cmd = [
            str(self.openmvs_bin_dir / "InterfaceCOLMAP"),
            "--working-folder",
            str(self.openmvs_working_dir),
            "--input-file",
            str(self.dense_dir),
            # '--image-folder', str(self.dense_dir / 'images'),
            "--output-file",
            str(self._openmvs_mvs_project_path()),
        ]
        self._exec_cmd(cmd)

    def openmvs_densify_point_cloud(self):
        cmd = [
            str(self.openmvs_bin_dir / "DensifyPointCloud"),
            "--working-folder",
            str(self.openmvs_working_dir),
            "--input-file",
            str(self._openmvs_mvs_project_path()),
            "--output-file",
            str(self.openmvs_working_dir / "tmp"),
            # '--archive-type', str(0), # text based output
            "--resolution-level",
            str(self.openmvs_densify_resolution_level),
            # '--max-resolution', str(self.openmvs_densify_max_resolution),
            "--min-resolution",
            str(self.openmvs_densify_min_resolution),
            "--number-views",
            str(self.openmvs_densify_number_views),
            "--optimize",
            str(self.openmvs_densify_optimize),
        ]
        self._exec_cmd(cmd)

    def mvsnet_parameters(
        self, max_d=0, interval_scale=1, theta0=5, sigma1=1, sigma2=10
    ):
        # code adapted from https://github.com/YoYo000/MVSNet
        cameras = read_cameras_binary(self.dense_dir / "sparse" / "cameras.bin")
        images = read_images_binary(self.dense_dir / "sparse" / "images.bin")
        points3d = read_points3d_binary(
            self.dense_dir / "sparse" / "points3D.bin"
        )

        # fmt: off
        param_type = {
            'SIMPLE_PINHOLE': ['f', 'cx', 'cy'],
            'PINHOLE': ['fx', 'fy', 'cx', 'cy'],
            'SIMPLE_RADIAL': ['f', 'cx', 'cy', 'k'],
            'SIMPLE_RADIAL_FISHEYE': ['f', 'cx', 'cy', 'k'],
            'RADIAL': ['f', 'cx', 'cy', 'k1', 'k2'],
            'RADIAL_FISHEYE': ['f', 'cx', 'cy', 'k1', 'k2'],
            'OPENCV': ['fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2'],
            'OPENCV_FISHEYE': ['fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'k3', 'k4'],
            'FULL_OPENCV': ['fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6'],
            'FOV': ['fx', 'fy', 'cx', 'cy', 'omega'],
            'THIN_PRISM_FISHEYE': ['fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'sx1', 'sy1']
        }
        # fmt: on

        # intrinsic
        intrinsic = {}
        for camera_id, cam in cameras.items():
            params_dict = {
                key: value
                for key, value in zip(param_type[cam.model], cam.params)
            }
            if "f" in param_type[cam.model]:
                params_dict["fx"] = params_dict["f"]
                params_dict["fy"] = params_dict["f"]
            intrinsic[camera_id] = np.array(
                [
                    [params_dict["fx"], 0, params_dict["cx"]],
                    [0, params_dict["fy"], params_dict["cy"]],
                    [0, 0, 1],
                ]
            )

        # extrinsic
        extrinsic = {}
        for image_id, image in images.items():
            e = np.zeros((4, 4))
            e[:3, :3] = qvec2rotmat(image.qvec)
            e[:3, 3] = image.tvec
            e[3, 3] = 1
            extrinsic[image_id] = e

        # depth range and interval
        depth_ranges = {}
        for image_id, image in images.items():
            zs = []
            for p3d_id in image.point3D_ids:
                if p3d_id == -1:
                    continue
                transformed = np.matmul(
                    extrinsic[image_id],
                    [
                        points3d[p3d_id].xyz[0],
                        points3d[p3d_id].xyz[1],
                        points3d[p3d_id].xyz[2],
                        1,
                    ],
                )
                zs.append(np.asscalar(transformed[2]))
            zs_sorted = sorted(zs)
            # relaxed depth range
            depth_min = zs_sorted[int(len(zs) * 0.01)]
            depth_max = zs_sorted[int(len(zs) * 0.99)]
            # determine depth number by inverse depth setting, see supplementary material
            if max_d == 0:
                image_int = intrinsic[image.camera_id]
                image_ext = extrinsic[image_id]
                image_r = image_ext[0:3, 0:3]
                image_t = image_ext[0:3, 3]
                p1 = [image_int[0, 2], image_int[1, 2], 1]
                p2 = [image_int[0, 2] + 1, image_int[1, 2], 1]
                P1 = np.matmul(np.linalg.inv(image_int), p1) * depth_min
                P1 = np.matmul(np.linalg.inv(image_r), (P1 - image_t))
                P2 = np.matmul(np.linalg.inv(image_int), p2) * depth_min
                P2 = np.matmul(np.linalg.inv(image_r), (P2 - image_t))
                depth_num = (1 / depth_min - 1 / depth_max) / (
                    1 / depth_min - 1 / (depth_min + np.linalg.norm(P2 - P1))
                )
            else:
                depth_num = max_d
            depth_interval = (
                (depth_max - depth_min) / (depth_num - 1) / interval_scale
            )
            depth_ranges[image.name] = (
                depth_min,
                depth_interval,
                depth_num,
                depth_max,
            )

        # view selection
        queue = []
        pairs = {}
        for image_id0 in images.keys():
            for image_id1 in images.keys():
                key = (min(image_id0, image_id1), max(image_id0, image_id1))
                if image_id0 != image_id1 and key not in pairs:
                    queue.append(
                        (
                            image_id0,
                            image_id1,
                            images,
                            points3d,
                            extrinsic,
                            theta0,
                            sigma1,
                            sigma2,
                        )
                    )
                    pairs[key] = True
        processes = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=processes)
        result = pool.map(_mvsnet_calc_score, queue)
        # result = map(calc_score, queue)
        scores = collections.defaultdict(dict)
        for image_id0, image_id1, s in result:
            name0 = images[image_id0].name
            name1 = images[image_id1].name
            scores[name0][name1] = s
            scores[name1][name0] = s
        return depth_ranges, scores


def _mvsnet_calc_score(inputs):
    image_id0, image_id1, images, points3d, extrinsic, theta0, sigma1, sigma2 = (
        inputs
    )
    id_i = images[image_id0].point3D_ids
    id_j = images[image_id1].point3D_ids
    id_intersect = [it for it in id_i if it in id_j]
    cam_center_i = -np.matmul(
        extrinsic[image_id0][:3, :3].transpose(), extrinsic[image_id0][:3, 3:4]
    )[:, 0]
    cam_center_j = -np.matmul(
        extrinsic[image_id1][:3, :3].transpose(), extrinsic[image_id1][:3, 3:4]
    )[:, 0]
    score = 0
    for pid in id_intersect:
        if pid == -1:
            continue
        p = points3d[pid].xyz
        theta = (180 / np.pi) * np.arccos(
            np.dot(cam_center_i - p, cam_center_j - p)
            / np.linalg.norm(cam_center_i - p)
            / np.linalg.norm(cam_center_j - p)
        )
        score += np.exp(
            -(theta - theta0)
            * (theta - theta0)
            / (2 * (sigma1 if theta <= theta0 else sigma2) ** 2)
        )
    return image_id0, image_id1, score


def mvsnet_from_colmap_dense(
    dense_dir,
    out_dir,
    max_d=0,
    interval_scale=1,
    theta0=5,
    sigma1=1,
    sigma2=10,
    im_suffix=".jpg",
):
    dense_dir = Path(dense_dir)
    out_dir = Path(out_dir)

    print("copy images")
    im_paths = sorted((dense_dir / "images").glob(f"*{im_suffix}"))
    im_out_dir = out_dir / "images"
    im_out_dir.mkdir(exist_ok=True, parents=True)
    im_name_to_idx = {}
    for idx, im_path in enumerate(im_paths):
        im_name_to_idx[im_path.name] = idx
        shutil.copyfile(im_path, im_out_dir / f"{idx:08d}{im_path.suffix}")

    print("compute params")
    colmap = Colmap("", dense_dir.parent, [])
    depth_ranges, scores = colmap.mvsnet_parameters(
        max_d=max_d,
        interval_scale=interval_scale,
        theta0=theta0,
        sigma1=sigma1,
        sigma2=sigma2,
    )

    print("write cams")
    Ks, Rs, ts = load_cameras(dense_dir / "sparse", im_paths)
    cam_out_dir = out_dir / "cams"
    cam_out_dir.mkdir(exist_ok=True, parents=True)
    for idx, im_path in enumerate(im_paths):
        cam_path = cam_out_dir / f"{idx:08}_cam.txt"
        T = np.eye(4)
        T[:3, :3] = Rs[idx]
        T[:3, 3] = ts[idx]
        with open(cam_path, "w") as fp:
            fp.write("extrinsic\n")
            fp.write(" ".join(map(str, T[0])) + "\n")
            fp.write(" ".join(map(str, T[1])) + "\n")
            fp.write(" ".join(map(str, T[2])) + "\n")
            fp.write(" ".join(map(str, T[3])) + "\n")
            fp.write("\n")
            fp.write("intrinsic\n")
            fp.write(" ".join(map(str, Ks[idx][0])) + "\n")
            fp.write(" ".join(map(str, Ks[idx][1])) + "\n")
            fp.write(" ".join(map(str, Ks[idx][2])) + "\n")
            fp.write("\n")
            fp.write(" ".join(map(str, depth_ranges[im_path.name])))

    print("write pairs")
    with open(out_dir / "pair.txt", "w") as fp:
        fp.write(f"{len(im_paths)}\n")
        for idx, im_path in enumerate(im_paths):
            fp.write(f"{idx}\n")
            nbs = {
                k: v
                for k, v in sorted(
                    scores[im_path.name].items(), key=lambda item: -item[1]
                )
            }
            fp.write(f"10")
            n_nbs = 0
            for k, v in nbs.items():
                fp.write(f" {im_name_to_idx[k]} {v}")
                n_nbs += 1
                if n_nbs == 10:
                    break
            fp.write("\n")


def write_depthmap_vis(
    dense_dir, im_name, geometric=True, vmin=None, vmax=None, ext="png"
):
    import cv2

    dense_dir = Path(dense_dir)

    im_path = dense_dir / "images" / im_name
    tmp = "geometric" if geometric else "photometric"
    bin_path = dense_dir / "stereo" / "depth_maps" / f"{im_name}.{tmp}.bin"
    out_path = (
        dense_dir / "stereo" / "depth_maps" / f"dmvis_{im_name}.{tmp}.{ext}"
    )

    if not (im_path.exists() and bin_path.exists()):
        return None

    dm = read_array(str(bin_path))
    dm[dm <= 0] = np.NaN
    if vmin is None:
        vmin = np.nanpercentile(dm, 5)
    if vmax is None:
        vmax = np.nanpercentile(dm, 95)
    dm3 = plt.image_colorcode(dm, vmin=vmin, vmax=vmax)
    im = cv2.cvtColor(cv2.imread(str(im_path)), cv2.COLOR_BGR2RGB) / 255

    alpha = 0.25
    dmim = alpha * im + (1 - alpha) * dm3

    out = np.hstack((im, dm3, dmim))
    out = (255 * out).astype(np.uint8)
    cv2.imwrite(str(out_path), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    return out_path


def load_cameras(sparse_dir, im_paths, scale=1, dtype=None):
    sparse_dir = Path(sparse_dir)
    if (sparse_dir / "images.bin").exists():
        ims = read_images_binary(sparse_dir / "images.bin")
    else:
        ims = read_images_text(sparse_dir / "images.txt")
    if (sparse_dir / "cameras.bin").exists():
        cams = read_cameras_binary(sparse_dir / "cameras.bin")
    else:
        cams = read_cameras_text(sparse_dir / "cameras.txt")

    ims = {im.name: im for key, im in ims.items()}

    Ks = np.empty((len(im_paths), 3, 3), dtype=np.float64)
    Rs = np.empty((len(im_paths), 3, 3), dtype=np.float64)
    ts = np.empty((len(im_paths), 3), dtype=np.float64)
    for idx, im_path in enumerate(im_paths):
        im = ims[im_path.name]
        camera_id = im.camera_id
        K = np.eye(3)
        K[0, 0], K[1, 1], K[0, 2], K[1, 2] = cams[camera_id].params
        Ks[idx] = K
        Rs[idx] = rotm_from_quat(im.qvec)
        ts[idx] = im.tvec
    if scale != 1:
        Ks[:, :2] *= scale

    if dtype is not None:
        Ks = Ks.astype(np.float32)
        Rs = Rs.astype(np.float32)
        ts = ts.astype(np.float32)

    return Ks, Rs, ts

def load_cameras_all(sparse_dir):
    sparse_dir = Path(sparse_dir)
    if (sparse_dir / "images.bin").exists():
        ims = read_images_binary(sparse_dir / "images.bin")
    else:
        ims = read_images_text(sparse_dir / "images.txt")
    if (sparse_dir / "cameras.bin").exists():
        cams = read_cameras_binary(sparse_dir / "cameras.bin")
    else:
        cams = read_cameras_text(sparse_dir / "cameras.txt")

    ims = {im.name: im for key, im in ims.items()}

    Ks = []
    Rs = []
    ts = []
    heights, widths = [], []
    for im_name in ims.keys():
        im = ims[im_name]
        camera_id = im.camera_id
        K = np.eye(3)
        K[0, 0], K[1, 1], K[0, 2], K[1, 2] = cams[camera_id].params
        Ks.append(K)
        Rs.append(rotm_from_quat(im.qvec))
        ts.append(im.tvec)
        heights.append(cams[camera_id].height)
        widths.append(cams[camera_id].width)

    Ks = np.array(Ks)
    Rs = np.array(Rs)
    ts = np.array(ts)
    heights = np.array(heights)
    widths = np.array(widths)

    return Ks, Rs, ts, heights, widths

def main_bin_to_txt(args):
    print("bin-to-txt")
    sparse_dir = Path(args.sparse_dir)
    print(sparse_dir)

    cam_bin = sparse_dir / "cameras.bin"
    cam_txt = sparse_dir / "cameras.txt"
    if cam_bin.exists():
        print(f"convert {cam_bin} to {cam_txt}")
        cameras = read_cameras_binary(str(cam_bin))
        write_cameras_text(cam_txt, cameras)

    img_bin = sparse_dir / "images.bin"
    img_txt = sparse_dir / "images.txt"
    if img_bin.exists():
        print(f"convert {img_bin} to {img_txt}")
        images = read_images_binary(str(img_bin))
        write_images_text(img_txt, images)

    pts_bin = sparse_dir / "points3D.bin"
    pts_txt = sparse_dir / "points3D.txt"
    if pts_bin.exists():
        print(f"convert {pts_bin} to {pts_txt}")
        points3d = read_points3d_binary(str(pts_bin))
        write_points3D_text(pts_txt, points3d)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_bin_to_txt = subparsers.add_parser("bin-to-txt")
    parser_bin_to_txt.add_argument("--sparse-dir", type=str)
    parser_bin_to_txt.set_defaults(func=main_bin_to_txt)

    args = parser.parse_args()
    args.func(args)
