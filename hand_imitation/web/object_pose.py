import numpy as np
import cv2
import os
from collections import OrderedDict
import csv
from PIL import Image

VERTEX_COLORS = [
    (0, 0, 0),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 255),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 0),
]

_CUR_DIR = os.path.dirname(__file__)


def parse_csv(filename):
    """Parse the CSV file to acquire the information of objects used in TOC.

    Args:
        filename (str): a CSV file containing object information.

    Returns:
        dict: {str: dict}, object_name -> object_info
    """
    object_db = OrderedDict()
    with open(filename, 'r') as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            object_db[row['object']] = row
    return object_db


OBJECT_DB = parse_csv(os.path.join(_CUR_DIR, "objects.csv"))
OBJECT_NAMES = tuple(OBJECT_DB.keys())
NUM_OBJECTS = len(OBJECT_NAMES)


def get_corners():
    """Get 8 corners of a cuboid. (The order follows OrientedBoundingBox in open3d)
        (y)
        2 -------- 7
       /|         /|
      5 -------- 4 .
      | |        | |
      . 0 -------- 1 (x)
      |/         |/
      3 -------- 6
      (z)
    """
    corners = np.array([[0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                        [1.0, 1.0, 1.0],
                        [0.0, 1.0, 1.0],
                        [1.0, 0.0, 1.0],
                        [1.0, 1.0, 0.0],
                        ])
    return corners - [0.5, 0.5, 0.5]


def get_edges(corners):
    assert len(corners) == 8
    edges = []
    for i in range(8):
        for j in range(i + 1, 8):
            if np.sum(corners[i] == corners[j]) == 2:
                edges.append((i, j))
    assert len(edges) == 12
    return edges


def draw_projected_box3d(image, center, size, rotation,
                         extrinsic, intrinsic,
                         color=(0, 255, 0), thickness=1):
    """Draw a projected 3D bounding box on the image.
​
    Args:
        image (np.ndarray): [H, W, 3] array.
        center: [3]
        size: [3]
        rotation (np.ndarray): [3, 3]
        extrinsic (np.ndarray): [4, 4]
        intrinsic (np.ndarray): [3, 3]
        color: [3]
        thickness (int): thickness of lines
    Returns:
        np.ndarray: updated image.
    """
    corners = get_corners()  # [8, 3]
    edges = get_edges(corners)  # [12, 2]
    corners = corners * size
    corners_world = corners @ rotation.T + center
    corners_camera = corners_world @ extrinsic[:3, :3].T + extrinsic[:3, 3]
    corners_image = corners_camera @ intrinsic.T
    uv = corners_image[:, 0:2] / corners_image[:, 2:]
    uv = uv.astype(int)
    z = corners_image[:, 2]

    for (i, j) in edges:
        if z[i] <= 0.0 or z[j] <= 0.0:
            import warnings
            warnings.warn('Some corners are behind the camera.')
            continue
        cv2.line(
            image,
            (uv[i, 0], uv[i, 1]),
            (uv[j, 0], uv[j, 1]),
            tuple(color),
            thickness,
            cv2.LINE_AA,
        )

    for i, (u, v) in enumerate(uv):
        cv2.circle(image, (u, v), radius=1, color=VERTEX_COLORS[i], thickness=1)
    return image


def draw_pose_bbox_image(image, object_name, pose, intrinsic):
    if object_name not in OBJECT_NAMES:
        raise ValueError(f"{object_name} not in known object lists")
    object_property = OBJECT_DB[object_name]
    object_size = np.array([object_property[key] for key in ['width', 'length', 'height']])
    rotation = pose[:3, :3]
    center = pose[:3, 3]
    return draw_projected_box3d(image, center, object_size.astype(np.float), rotation, np.eye(4), intrinsic)


def recover_flip_pose(pose: np.ndarray):
    fx, fy = 612.99, 613.113
    w, h = 640, 480
    cx, cy = 317.723, 239.658
    flip_matrix = np.array([[-1, 0, (w - 2 * cx) / fx, 0], [0, -1, (h - 2 * cy) / fy, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    return flip_matrix @ pose


def generate_pose_images(pose_files, image_files, cache_files, object_name, intrinsic):
    for pose_file, image_file, cache_file in zip(pose_files, image_files, cache_files):
        print(f"Finish object pose image at {cache_file}")
        pose = np.load(pose_file, allow_pickle=True)
        pose = np.concatenate([pose, np.array([[0, 0, 0, 1]])], axis=0)
        pose = recover_flip_pose(pose)

        image = np.asarray(Image.open(image_file))
        pose_image = draw_pose_bbox_image(image, object_name, pose, intrinsic)
        Image.fromarray(pose_image).save(cache_file)
    return
