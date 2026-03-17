import random


def train_dev_test_split(samples, seed=42):

    random.seed(seed)

    random.shuffle(samples)

    n = len(samples)

    train_end = int(0.8 * n)
    dev_end = int(0.9 * n)

    train_samples = samples[:train_end]
    dev_samples = samples[train_end:dev_end]
    test_samples = samples[dev_end:]

    return train_samples, dev_samples, test_samples
