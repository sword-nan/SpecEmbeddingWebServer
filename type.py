from typing import TypedDict, Sequence, Callable, Optional

import torch
from torch import nn
from torch import device
import numpy as np
import numpy.typing as npt

BatchType = Sequence[torch.Tensor]
StepTrain = Callable[[nn.Module, nn.Module, device,
                      BatchType, Optional[Callable[..., int]]], Sequence[torch.Tensor]]
StepVal = Callable[[nn.Module, nn.Module, device,
                    BatchType, Optional[Callable[..., int]]], Sequence[torch.Tensor]]

class Peak(TypedDict):
    mz: str
    intensity: npt.NDArray


class MetaData(TypedDict):
    peaks: Sequence[Peak]
    smiles: str


class TokenSequence(TypedDict):
    mz: npt.NDArray[np.int32]
    intensity: npt.NDArray[np.float32]
    mask: npt.NDArray[np.bool_]
    smiles: str


class TokenizerConfig(TypedDict):
    max_len: int
    show_progress_bar: bool