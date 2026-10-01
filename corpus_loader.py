"""Minimal reader for the training corpus: every series from all four sources,
shuffled uniformly, yielded in batches.

    unzip corpus.zip -d corpus
    for batch in batches("corpus", batch_size=64):
        ...  # list of 1-D float32 arrays; lengths differ (up to 1024, except
             # audio clips, which are whole recordings at 4 kHz: 0.3 s to minutes)
"""

from pathlib import Path

import numpy as np


def load(root):
    """Return a function mapping a global index to one series, and the count."""
    root = Path(root)
    values = np.load(root / "lotsa/values.npy", mmap_mode="r")
    offsets = np.load(root / "lotsa/offsets.npy")
    market = np.load(root / "market/series.npy", mmap_mode="r")  # (n, 780)
    pfn = np.load(root / "forecastpfn/series.npy", mmap_mode="r")  # (n, 1024)
    audio = np.load(root / "audio/values.npy", mmap_mode="r")
    audio_offsets = np.load(root / "audio/offsets.npy")
    sizes = np.cumsum([len(offsets) - 1, len(market), len(pfn), len(audio_offsets) - 1])

    def get(i):
        if i < sizes[0]:
            return np.asarray(values[offsets[i] : offsets[i + 1]])
        if i < sizes[1]:
            return np.asarray(market[i - sizes[0]])
        if i < sizes[2]:
            return np.asarray(pfn[i - sizes[1]])
        j = i - sizes[2]
        return np.asarray(audio[audio_offsets[j] : audio_offsets[j + 1]])

    return get, int(sizes[-1])


def batches(root, batch_size=64, seed=0):
    """One epoch: every series exactly once, in a uniformly random order."""
    get, n = load(root)
    order = np.random.default_rng(seed).permutation(n)
    for start in range(0, n, batch_size):
        yield [get(i) for i in order[start : start + batch_size]]
