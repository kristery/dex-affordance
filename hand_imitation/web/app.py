from flask import Flask, render_template, request, redirect, url_for
import requests
import os
from PIL import Image
import numpy as np
from object_pose import generate_pose_images

DATASET_ROOT = "/home/sim/data/wanglab_dataset/dataset_v2/"
APP_NAME = f"Hand Imitation Project Visualizer:{os.path.basename(DATASET_ROOT)}"
CACHE_DIR = "cache"
CAMERA_MAT = np.array([[612.99, 0, 317.723], [0, 613.113, 239.658], [0, 0, 1]])

app = Flask(__name__, static_folder=DATASET_ROOT)
app.config.update(
    APPNAME=APP_NAME,
)
CURRENT_DIR = "/"


class DataCell:
    def __init__(self, image, name, task_name, object_name, sequence_name, camera_name):
        self.image = image
        self.name = name
        self.task = task_name
        self.object = object_name
        self.sequence = sequence_name
        self.camera = camera_name


class ObjectPoseDataCell(DataCell):
    def __init__(self, pose_image, image, name, task_name, object_name, sequence_name, camera_name):
        super().__init__(image, name, task_name, object_name, sequence_name, camera_name)
        self.pose_image = pose_image


@app.route('/')
def index():
    file_dirs = sorted(os.listdir(DATASET_ROOT))
    return render_template(
        "gallery.html",
        dir_name="/",
        parent_dir="/",
        data_cells=[],
        page_name="files",
        file_dirs=file_dirs,
    )


@app.route('/goto', methods=['POST', 'GET'])
def goto():
    return redirect('/' + request.form['index'])


@app.route("/<path:file_path>")
def main(file_path: str):
    file_path = file_path.strip('/')
    level_of_path = file_path.count('/')
    print(file_path, level_of_path)
    if level_of_path == 4:
        if 'object_pose' in file_path:
            return render_object_pose(file_path)
        else:
            return render_img_directory(file_path)
    elif level_of_path == 5:
        file_path = os.path.dirname(file_path)
        return render_img_directory(file_path)
    elif level_of_path < 4:
        return render_file_directory(file_path, level_of_path)


@app.route("/refresh", methods=['POST'])
def refresh():
    dir_path = CURRENT_DIR
    abs_path = os.path.join(DATASET_ROOT, dir_path)
    task_name, object_name, sequence_name, camera_name = dir_path.split("/")[-4:]
    files = os.listdir(abs_path)

    pose_files = []
    image_files = []
    cache_files = []
    cache_dir = pose_file_to_image_file(os.path.join(DATASET_ROOT, CACHE_DIR, dir_path))
    os.makedirs(cache_dir, exist_ok=True)
    for file in reorder_data_file(files):
        if file.endswith(".npy"):
            pose_file = os.path.join(DATASET_ROOT, dir_path, file)
            image_file = pose_file_to_image_file(pose_file)
            pose_files.append(pose_file)
            image_files.append(image_file)
            cache_files.append(pose_file_to_image_file(os.path.join(DATASET_ROOT, CACHE_DIR, dir_path, file)))

    generate_pose_images(pose_files, image_files, cache_files, object_name, CAMERA_MAT)
    return redirect('/' + CURRENT_DIR)


def pose_file_to_image_file(pose_file):
    return pose_file.replace("object_pose", "raw_image").replace(".npy", ".png").replace("objpose", "color")


def reorder_data_file(files):
    numbers = []
    for file in files:
        res = [int(i) for i in file if i.isdigit()]
        numbers.append(int("".join([str(e) for e in res])))
    new_order = np.argsort(numbers)
    files = [files[i] for i in new_order]
    return files


def render_img_directory(dir_path):
    abs_path = os.path.join(DATASET_ROOT, dir_path)
    task_name, object_name, sequence_name, camera_name = dir_path.split("/")[-4:]
    files = os.listdir(abs_path)
    parent_dir = os.path.join(dir_path, "../")
    data_cells = []
    for file in reorder_data_file(files):
        if file.endswith(".png"):
            image_file = os.path.join(dir_path, file)
            data_cell = DataCell(image_file, os.path.basename(image_file), task_name, object_name, sequence_name,
                                 camera_name)
            data_cells.append(data_cell)
    return render_template(
        "gallery.html",
        dir_name=dir_path,
        parent_dir=parent_dir,
        data_cells=data_cells,
        page_name="raw_image",
        file_dirs=[],
    )


def render_file_directory(dir_path, level_of_path):
    abs_path = os.path.join(DATASET_ROOT, dir_path)
    file_dirs = sorted(os.listdir(abs_path))
    file_dirs = [os.path.join(dir_path, file) for file in file_dirs]
    parent_dir = os.path.join(dir_path, "../") if level_of_path >= 0 else dir_path
    return render_template(
        "gallery.html",
        dir_name=dir_path,
        parent_dir=parent_dir,
        data_cells=[],
        page_name="files",
        file_dirs=file_dirs,
    )


def render_object_pose(dir_path):
    abs_path = os.path.join(DATASET_ROOT, dir_path)
    task_name, object_name, sequence_name, camera_name = dir_path.split("/")[-4:]
    files = os.listdir(abs_path)
    parent_dir = os.path.join(dir_path, "../")
    global CURRENT_DIR
    CURRENT_DIR = dir_path

    data_cells = []
    for file in reorder_data_file(files):
        if file.endswith(".npy"):
            pose_file = os.path.join(dir_path, file)
            image_file = pose_file_to_image_file(pose_file)
            pose_image_file = os.path.join(CACHE_DIR, image_file)
            data_cell = ObjectPoseDataCell(pose_image_file, image_file, os.path.basename(image_file), task_name,
                                           object_name, sequence_name, camera_name)
            data_cells.append(data_cell)

    return render_template(
        "gallery_object_pose.html",
        dir_name=dir_path,
        parent_dir=parent_dir,
        data_cells=data_cells,
        page_name="object_pose",
    )


if __name__ == '__main__':
    app.run(debug=True)
