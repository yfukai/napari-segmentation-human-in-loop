import pytest

# https://github.com/MouseLand/cellpose-napari/pull/35 thanks @psobolewskiPhD!
import torch  # noqa: F401

from napari_segmentation_human_in_loop._widget import (
    IMAGES_LAYER_NAME,
    LABELS_LAYER_NAME,
    Trainers,
    wizard_widget,
)


@pytest.mark.parametrize(
    "training",
    [
        False,
    ],
)
def test_example_magic_widget(make_napari_viewer, datadir, training):
    viewer = make_napari_viewer()
    # this time, our widget will be a MagicFactory or FunctionGui instance

    my_widget = wizard_widget()

    # if we "call" this object, it'll execute our function
    my_widget(
        viewer,
        datadir,
        "test2",
        training=training,
        trainer_cls=Trainers.cellpose,
    )

    assert IMAGES_LAYER_NAME in viewer.layers
    assert LABELS_LAYER_NAME in viewer.layers


#    # read captured output and check that it's as we expected
#    # captured = capsys.readouterr()
#
#
##    assert captured.out == f"you have selected {layer}\n"
