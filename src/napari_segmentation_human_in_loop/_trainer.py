from os import path
from pathlib import Path
from typing import Tuple

from napari.types import LabelsData


class CellposeTrainer:
    channels: Tuple[int, int] = (0, 0)

    def __init__(self, model_path: Path, model_name: str) -> None:
        self.model_path = model_path
        self.model_name = model_name

    def _get_model(self):
        model_exists = path.exists(self.model_path / self.model_name)
        from cellpose import models

        return models.CellposeModel(
            gpu=True,
            pretrained_model=str(self.model_path / self.model_name)
            if model_exists
            else False,
            model_type="cyto" if not model_exists else None,
        )

    def train(self, images, labels) -> None:
        model = self._get_model()
        model.train(
            images,
            labels,
            channels=self.channels,
            save_path=str(self.model_path.parent),
            model_name=self.model_name,
        )

    def predict(self, images) -> LabelsData:
        model = self._get_model()
        return model.eval(images, channels=self.channels, rescale=1.0)
