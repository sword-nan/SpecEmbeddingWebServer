import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

class ModelTester:
    def __init__(
        self,
        model: Module,
        device: torch.device,
        show_prgress_bar: bool = True
    ) -> None:
        self.model = model
        self.device = device
        self.show_prgress_bar = show_prgress_bar

    def test(self, dataloader: DataLoader):
        self.model.eval()
        result = []
        with torch.no_grad():
            pbar = dataloader
            if self.show_prgress_bar:
                pbar = tqdm(dataloader, total=len(
                    dataloader), desc="embedding")
            for x in pbar:
                x = [d.to(self.device) for d in x]
                pred: torch.Tensor = self.model(*x)
                result.append(pred.cpu().numpy())
        return np.concatenate(result, axis=0)
