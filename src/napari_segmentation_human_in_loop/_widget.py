"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
import logging
from enum import Enum
from glob import glob
from os import path
from typing import TYPE_CHECKING

from magicgui import magic_factory
from napari.utils.notifications import show_error
from skimage.io import imread

from ._trainer import CellposeTrainer

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if TYPE_CHECKING:
    pass

IMAGES_EXT = [".tiff", ".png"]


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
    image_with_label_paths, new_label_paths = zip(
        [(f, lf) for f, lf in zip(image_paths, label_paths) if path.isfile(lf)]
    )
    image_without_label_paths = [
        f for f, lf in zip(image_paths, label_paths) if not path.isfile(lf)
    ]
    logger.debug("load images and labels finished")
    return image_with_label_paths, image_without_label_paths, new_label_paths


@magic_factory(
    input_folder=dict(
        widget_type="FileEdit",
        mode="d",
        label="custom model path: ",
        tooltip="if model type is custom, specify file path to it here",
    ),
    call_button="Training",
)
def wizard_widget(
    input_folder, train_all: bool, trainer_cls: Trainers = Trainers.cellpose
):
    if not _check_input_folder(input_folder):
        return
    logger.debug("training called")
    (
        image_with_label_paths,
        image_without_label_paths,
        label_paths,
    ) = _load_images_and_label_paths(input_folder)

    if train_all:
        images = [imread(f) for f in image_with_label_paths]
        labels = [imread(f) for f in label_paths]
    else:
        image = [...]
        label = [...]

    trainer = trainer_cls.value()
    trainer.train(image, label)

    if len(image_without_label_paths) > 0:
        new_image = image_without_label_paths[0]
        new_label = trainer.predict(new_image)
