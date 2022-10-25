import cv2
import numpy as np


def get_checkerboard_pose(image, board_size=(4, 5), square_size=0.03, intrinsic=np.eye(3), visualize=False):
    # Note that board_size = (a, b) and board_size = (b, a) will output different pose due to different origin
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(grey, board_size, None)
    if not found:
        import warnings
        warnings.warn("Checkerboard not found!")
        return None

    criteria_subpixel = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(grey, corners, (5, 5), (-1, -1), criteria_subpixel)
    obj_point = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    obj_point[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2) * square_size

    # Only for visualization
    if visualize:
        img = cv2.drawChessboardCorners(image, board_size, corners, True)
        cv2.imshow("checker_board", img)
        cv2.waitKey()

    _, rot_vector, trans_vector = cv2.solvePnP(obj_point, corners, intrinsic, None)
    rotation_matrix, _ = cv2.Rodrigues(rot_vector)
    homo_matrix = np.eye(4)
    homo_matrix[0:3, 0:3] = rotation_matrix
    homo_matrix[0:3, 3:4] = trans_vector
    return homo_matrix


def get_point_cloud_from_depth(depth_image, intrinsic, extrinsic=None):
    v, u = np.indices(depth_image.shape)  # [H, W], [H, W]
    z = depth_image  # [H, W]
    uv1 = np.stack([u + 0.5, v + 0.5, np.ones_like(z)], axis=-1)
    points_camera = uv1 @ np.linalg.inv(intrinsic).T * z[..., None]  # [H, W, 3]
    if extrinsic is None:
        return points_camera
    else:
        extrinsic_inv = np.linalg.inv(extrinsic)
        points_world = points_camera @ extrinsic_inv[:3, :3].T + extrinsic_inv[:3, 3]
        return points_world


def np2pcd(points, colors=None, normals=None):
    import open3d as o3d
    """Convert numpy array to open3d PointCloud."""
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        colors = np.array(colors)
        if colors.ndim == 2:
            assert len(colors) == len(points)
        elif colors.ndim == 1:
            colors = np.tile(colors, (len(points), 1))
        else:
            raise RuntimeError(colors.shape)
        pc.colors = o3d.utility.Vector3dVector(colors)
    if normals is not None:
        assert len(points) == len(normals)
        pc.normals = o3d.utility.Vector3dVector(normals)
    return pc


def __main():
    import glob
    import os
    from hand_imitation.misc.pose_utils import inverse_pose
    import open3d as o3d
    np.set_printoptions(precision=4)
    data_root = "/home/sim/data/wanglab_dataset/dataset_v3/raw_image/placement/banana/seq_30"
    cam0_files = sorted(glob.glob(os.path.join(data_root, "0_color", "*.png")))
    cam1_files = sorted(glob.glob(os.path.join(data_root, "1_color", "*.png")))

    intrinsic1 = np.array([[612.99, 0, 317.723], [0, 613.113, 239.658], [0, 0, 1]])
    intrinsic0 = np.array([[612.76, 0, 322.287], [0, 613.31, 234.676], [0, 0, 1]])

    # Check two cameras have the same sequence
    for cam0_file in cam0_files:
        cam1_file = cam0_file.replace("0_color", "1_color")
        if cam1_file not in cam1_files:
            raise FileNotFoundError(f"File {cam1_file} not exist! Check your calibration data")

    # Merge Point Cloud
    cam_pose = np.array(
        [[0.0454, 0.8483, 0.5276, -0.4907],
         [-0.8307, 0.3255, -0.4518, 0.3915],
         [-0.5549, -0.4177, 0.7194, 0.3348],
         [0., 0., 0., 1.]])

    for cam0_file in cam0_files:
        depth0_file = cam0_file.replace("0_color", "0_depth")
        depth1_file = cam0_file.replace("0_color", "1_depth")
        depth0_image = cv2.imread(depth0_file, cv2.IMREAD_UNCHANGED) / 1000.0
        depth1_image = cv2.imread(depth1_file, cv2.IMREAD_UNCHANGED) / 1000.0
        point0 = get_point_cloud_from_depth(depth0_image, intrinsic0, extrinsic=cam_pose)
        point1 = get_point_cloud_from_depth(depth1_image, intrinsic1)

        cam1_file = cam0_file.replace("0_color", "1_color")
        img0 = cv2.imread(cam0_file, cv2.IMREAD_COLOR)
        img1 = cv2.imread(cam1_file, cv2.IMREAD_COLOR)
        img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        points = np.reshape(np.concatenate([point0, point1], axis=0), [-1, 3])
        colors = np.reshape(np.concatenate([img0, img1], axis=0), [-1, 3]) / 255.0
        pcd = np2pcd(points, colors)
        o3d.visualization.draw_geometries([pcd])

        # o3d.io.write_point_cloud("merge.pcd", pcd)
        # exit()

    for cam0_file in cam0_files:
        cam1_file = cam0_file.replace("0_color", "1_color")

        img0 = cv2.imread(cam0_file, cv2.IMREAD_COLOR)
        board2cam0 = get_checkerboard_pose(img0, visualize=True, intrinsic=intrinsic0)
        print(board2cam0)

        img1 = cv2.imread(cam1_file, cv2.IMREAD_COLOR)
        board2cam1 = get_checkerboard_pose(img1, visualize=True, intrinsic=intrinsic1)
        # print(board2cam1)

        if board2cam0 is not None and board2cam1 is not None:
            cam12cam0 = board2cam0 @ inverse_pose(board2cam1)
            print(np.array2string(cam12cam0, separator=","))


if __name__ == '__main__':
    __main()
