from typing import Iterable

from PIL import Image
from pydantic import BaseModel, ConfigDict


class DetectionResult(BaseModel):
    image: Image.Image
    bbox: Iterable[int]

    model_config = ConfigDict(arbitrary_types_allowed=True)
