from torch.utils.data import Dataset
import torch


class LegalDataset(Dataset):

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        sample = self.samples[idx]

        return {
            "tort_id": sample["tort_id"],
            "U": sample["U"],
            "P": sample["P"],
            "D": sample["D"],
            "R_P": torch.tensor(sample["R_P"], dtype=torch.float),
            "R_D": torch.tensor(sample["R_D"], dtype=torch.float),
            "T": torch.tensor(sample["T"], dtype=torch.float)
        }