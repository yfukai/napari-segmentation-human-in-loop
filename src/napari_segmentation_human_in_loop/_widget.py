# %%
"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
import logging
import os
from enum import Enum
from glob import glob
from os import path
from pathlib import Path
from typing import TYPE_CHECKING, List

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

IMAGES_EXT = [".tiff", ".png"]
SELECTED_IMAGES_LAYER_NAME = "segmentation_hil_image"
SELECTED_LABELS_LAYER_NAME = "segmentation_hil_labels"


class Trainers(Enum):
    cellpose = CellposeTrainer


def _check_input_folder(folder):
    if folder == "" or not path.isdir(folder):
        logger.warning(f"path {folder} is not a valid directory")
        show_error(f"path {folder} is not a valid directory")
        return False
    else:
        return True


# %%


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
        zip(res) if len(res) > 0 else [],
        [],
    )
    image_without_label_paths = [
        f for f, lf in zip(image_paths, label_paths) if not path.isfile(lf)
    ]
    logger.debug("load images and labels finished")
    return image_with_label_paths, image_without_label_paths, new_label_paths


# %%


@magic_factory(
    input_folder=dict(
        widget_type="FileEdit",
        mode="d",
        label="custom model path: ",
        tooltip="if model type is custom, specify file path to it here",
    ),
    call_button="Save segmentation and train",
)
def wizard_widget(
    viewer: napari.Viewer,
    input_folder: Path,
    model_name: str,
    trainer_cls: Trainers = Trainers.cellpose,
) -> List[napari.types.LayerDataTuple]:
    logger.debug("training called")
    if not _check_input_folder(input_folder):
        return
    model_path = input_folder / "models" / model_name
    os.makedirs(model_path.parent, exist_ok=True)
    trainer = trainer_cls.value(model_path)

    if (
        SELECTED_IMAGES_LAYER_NAME in viewer.layers
        and SELECTED_LABELS_LAYER_NAME in viewer.layers
    ):
        logger.debug("saving images")
        image_path = viewer.layers[SELECTED_IMAGES_LAYER_NAME].metadata[
            "filename"
        ]
        label_data = viewer.layers[SELECTED_LABELS_LAYER_NAME].data
        np.save(image_path + ".label.npy", label_data)

    logger.debug("loading images")
    (
        image_with_label_paths,
        image_without_label_paths,
        label_paths,
    ) = _load_images_and_label_paths(input_folder)

    train_images = [imread(f) for f in image_with_label_paths]
    train_labels = [imread(f) for f in label_paths]

    logger.debug("loading images")
    if len(train_images) > 0:
        trainer.train(train_images, train_labels)

    if len(image_without_label_paths) > 0:
        new_image_path = image_without_label_paths[0]
    else:
        show_info("All images were used for training.")
        new_image_path = image_with_label_paths[0]
    new_image = imread(new_image_path)
    new_label = trainer.predict(new_image)

    return [
        (
            new_image,
            {
                "name": SELECTED_IMAGES_LAYER_NAME,
                "metadata": {"filename": new_image_path},
            },
            "image",
        ),
        (new_label, {"name": SELECTED_LABELS_LAYER_NAME}, "labels"),
    ]


# %%
