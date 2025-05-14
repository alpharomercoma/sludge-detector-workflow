from importlib import metadata as _metadata

__version__ = "0.1.0"

from .models import VideoQFormerModel
from .data import VideoCaptionDataset

__all__ = ["VideoQFormerModel", "VideoCaptionDataset", "__version__"]