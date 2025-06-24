import os
from joblib import Parallel, delayed
from pathlib import Path


import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from datasets import load_dataset
from datasets import IterableDataset
from tqdm import tqdm


def create_yolov6_dataset_yaml(
    yolo_data_dir: str
) -> None:
    """
    yolo_X_dir args should be relative to the yolov6
    repo root, not relative to prepare_dataset.py
    """
    
    yaml_file = "./YOLOv6/data/wider_face.yaml"
    train_images_dir = os.path.abspath(os.path.join(yolo_data_dir, "images", "train"))
    val_images_dir = os.path.abspath(os.path.join(yolo_data_dir, "images", "val"))
    test_images_dir = os.path.abspath(os.path.join(yolo_data_dir, "images", "test"))

    classes = ['Face']
    names_str = ""
    for item in classes:
        names_str = names_str + ", \'%s\'" % item
    names_str = "names: [" + names_str[1:] + "]"

    with open(yaml_file, "w") as wobj:
        wobj.write("train: %s\n" % train_images_dir)
        wobj.write("val: %s\n" % val_images_dir)
        wobj.write("test: %s\n" % test_images_dir)
        wobj.write("nc: %d\n" % len(classes))
        wobj.write(names_str + "\n")


def download_dataset(show_example: bool = False):
    wider_face_train = load_dataset('wider_face', split='train')
    wider_face_val = load_dataset('wider_face', split='validation')
    wider_face_test = load_dataset('wider_face', split='test')

    print("Num images in wider_face training set: %i" % (len(wider_face_train)))
    print("Num images in wider_face val set: %i" % (len(wider_face_val)))
    print("Num images in wider_face test set: %i" % (len(wider_face_test)))

    img = np.array(wider_face_train[110]['image'], dtype=np.uint8)
    faces = wider_face_train[110]['faces']
    bboxes = faces['bbox']

    fig, ax = plt.subplots()
    ax.imshow(img)

    for bbox in bboxes:
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                                 linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    if show_example:
        plt.show()

    return wider_face_train, wider_face_val, wider_face_test


def _write_files(data_point, data_type, dst_dir: Path, i: int):
    pil_img = data_point['image']
    label = data_point['faces']
    img_filename = data_type + str(i) + ".png"
    # img_filename = str(i) + ".jpg"

    dst_image_file = dst_dir / "images" / data_type /  f"{img_filename}"
    #print(f"image path: {dst_image_file}")
    dst_label_file = dst_dir / "labels" / data_type /  f"{img_filename}"
    #print(f"label path: {dst_label_file}")
    dst_label_file = dst_label_file.with_suffix(".txt")
    if dst_label_file.exists():
        return

    # we're only detecting faces, so class_id is constant
    class_id = 0
    img_width, img_height = pil_img.size
    with dst_label_file.open("w") as wobj:
        for bbox in label['bbox']:
            cx = (bbox[0] + (bbox[2]/2.0)) / img_width
            cy = (bbox[1] + (bbox[3]/2.0)) / img_height

            # output annotation is:
            #   class_id, center_x, center_y, box_width, box_height
            # image width and height normalized to (0, 1)
            box_width = bbox[2]/img_width
            box_height = bbox[3]/img_height
            output_line = f"{class_id} {cx} {cy} {box_width} {box_height}\n"
            wobj.write(output_line)
    pil_img.save(str(dst_image_file))


def convert_to_yolov6_format(
    dataset: IterableDataset,
    dst_dir: Path,
    data_type: str
) -> None:
    (dst_dir / Path(f"images/{data_type}")).mkdir(parents=True, exist_ok=True)
    (dst_dir / Path(f"labels/{data_type}")).mkdir(parents=True, exist_ok=True)

    Parallel(n_jobs=32)(delayed(_write_files)(dataset[i], data_type, dst_dir, i)
                        for i in tqdm(range(len(dataset))))


if __name__ == "__main__":
    widerface_dataset_path = "./YOLOv6/widerface_data"

    (wider_face_train, wider_face_val, wider_face_test) = download_dataset()

    convert_to_yolov6_format(wider_face_val, data_type="train", dst_dir=Path("./widerface_data"))
    convert_to_yolov6_format(wider_face_val, data_type="val", dst_dir=Path("./widerface_data"))
    convert_to_yolov6_format(wider_face_val, data_type="test", dst_dir=Path("./widerface_data"))

    create_yolov6_dataset_yaml(widerface_dataset_path)
