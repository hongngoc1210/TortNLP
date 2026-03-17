from torch.utils.data import DataLoader
from functools import partial

from data_utils import collate_fn

def build_dataloader(dataset, tokenizer, batch_size=4):

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=partial(collate_fn, tokenizer=tokenizer)
    )

    return loader