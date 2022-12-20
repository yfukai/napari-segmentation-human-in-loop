"""
Widget for human-in-loop training of segmentation models.
"""
import logging
import os
from enum import Enum
from glob import glob
from os import path
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import napari.types
import numpy as np
from magicgui import magic_factory
from napari.utils.notifications import show_error, show_info
from skimage.io import imread

from napari_segmentation_human_in_loop._trainer import CellposeTrainer

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if TYPE_CHECKING:
    pass

IMAGES_EXT = [".tiff", ".tif", ".png"]
IMAGES_LAYER_NAME = "segmentation_hil_image"
LABELS_LAYER_NAME = "segmentation_hil_labels"


class Trainers(Enum):
    cellpose = CellposeTrainer


def _check_input_folder(folder):
    if folder == "" or not path.isdir(folder):
        logger.warning(f"path {folder} is not a valid directory")
        show_error(f"path {folder} is not a valid directory")
        return False
    else:
        return True


def _load_images_and_label_paths(folder):
    logger.debug("load images and labels started")
    image_paths = sum(
        [glob(path.join(folder, "*" + ext)) for ext in IMAGES_EXT], []
    )
    if len(image_paths) == 0:
        show_error(f"no image was found in path {folder}")
        return [], [], []
    logger.debug(f"found {len(image_paths)} images")
    label_paths = [f + ".label.npy" for f in image_paths]
    res = [
        (f, lf) for f, lf in zip(image_paths, label_paths) if path.isfile(lf)
    ]
    image_with_label_paths, new_label_paths = (
        zip(*res) if len(res) > 0 else ([], [])
    )
    image_without_label_paths = [
        f for f, lf in zip(image_paths, label_paths) if not path.isfile(lf)
    ]
    logger.debug(f"images with labels {image_with_label_paths}")
    logger.debug(f"labels {new_label_paths}")
    logger.debug("load images and labels finished")
    return image_with_label_paths, image_without_label_paths, new_label_paths


@magic_factory(
    input_folder=dict(
        widget_type="FileEdit",
        mode="d",
        label="input directory: ",
        tooltip="directory containing input images and segmentations",
    ),
    call_button="Save segmentation and train",
)
def wizard_widget(
    viewer: napari.Viewer,
    input_folder: Path,
    model_name: str,
    training: bool,
    trainer_cls: Trainers = Trainers.cellpose,
    cyto_channel: int = 0,
    nuclear_channel: int = 0,
) -> Optional[List[napari.types.LayerDataTuple]]:
    logger.debug("training called")
    if not _check_input_folder(input_folder):
        return
    if model_name == "":
        show_error("model_name must not be empty")
        return

    model_path = input_folder / "models"
    os.makedirs(model_path, exist_ok=True)
    trainer = trainer_cls.value(model_path, model_name)
    trainer.channels = (cyto_channel, nuclear_channel)

    if (
        IMAGES_LAYER_NAME in viewer.layers
        and LABELS_LAYER_NAME in viewer.layers
    ):
        logger.debug("saving images")
        image_path = viewer.layers[IMAGES_LAYER_NAME].metadata["filename"]
        logger.debug(image_path)
        label_data = viewer.layers[LABELS_LAYER_NAME].data
        np.save(image_path + ".label.npy", label_data)

    logger.debug("loading images")
    (
        image_with_label_paths,
        image_without_label_paths,
        label_paths,
    ) = _load_images_and_label_paths(input_folder)

    train_images = [imread(f) for f in image_with_label_paths]
    train_labels = [np.load(f) for f in label_paths]
    logger.debug("loading images finished")

    if len(train_images) > 0 and training:
        logger.debug("training started")
        trainer.train(train_images, train_labels)
        logger.debug("training finished")

    if len(image_without_label_paths) > 0:
        new_image_path = image_without_label_paths[0]
    else:
        show_info("All images were used for training.")
        new_image_path = image_with_label_paths[0]
    new_image = imread(new_image_path)
    logger.debug("prediction started")
    new_label = trainer.predict([new_image])[0]
    logger.debug("prediction finished")
    image_props = {
        "name": IMAGES_LAYER_NAME,
        "metadata": {"filename": new_image_path},
    }
    if new_image.shape[-1] == 3:
        image_props.update(
            {
                "channel_axis": -1,
                "rgb": False,
                "colormap": ["red", "green", "blue"],
            }
        )

    return [
        (
            new_image,
            image_props,
            "image",
        ),
        (new_label, {"name": LABELS_LAYER_NAME}, "labels"),
    ]
