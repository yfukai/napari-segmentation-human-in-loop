"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
from typing import TYPE_CHECKING

from magicgui import magic_factory

if TYPE_CHECKING:
    pass


@magic_factory(
    train_partial=dict(
        widget_type="PushButton",
        text="Train partial",
        tooltip="partially train the model using the current image / label",
    ),
    train_all=dict(
        widget_type="PushButton",
        text="Train all",
        tooltip="train the model using the all images / labels in directory",
    ),
    call_button=False,
)
def wizard_widget(train_partial, train_all):
    pass
