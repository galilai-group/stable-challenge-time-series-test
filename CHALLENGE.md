Pretrain a time series encoder, self-supervised, on the training corpus. It is scored on held-out time series
tasks from several domains, which stay private, by how well a linear probe on its frozen embeddings solves them.

## Quick start

The environment is managed with [uv](https://docs.astral.sh/uv/getting-started/installation/). `uv sync` creates
`.venv` from [pyproject.toml](pyproject.toml), with the exact versions pinned in `uv.lock`.

```bash
uv sync                                                    # create .venv with all dependencies
unzip corpus.zip -d corpus                                 # the training data (see below)
uv run example_train.py --corpus corpus --out model.onnx   # train and export a small example encoder
uv run test_submission.py model.onnx                       # check a model before submitting it
```

Use `uv run` in front of any command (e.g. `uv run python my_training.py`), or activate the environment with
`source .venv/bin/activate`. Add packages you need for training with `uv add <package>`. On Linux, PyTorch and
onnxruntime are installed with CUDA 12 support.

## Training data

**Download: `corpus.zip` (8.9 GB) from CORPUS_URL.** Unzip it to get four parts, each with a `manifest.json`:

| Part | Size | Content |
| --- | --- | --- |
| `lotsa/` | 2.7 GB | univariate series from 92 public datasets (energy, traffic, weather, sales, ...), up to 1,024 points each |
| `market/` | 0.7 GB | US equities, one full trading day per series: 780 volume-weighted prices at 30 s intervals, 2019-07 to 2020-12 |
| `forecastpfn/` | 2.5 GB | synthetic series with trend, seasonality and noise, 1,024 points each |
| `audio/` | 3.0 GB | 25,624 environmental sound recordings at 4 kHz, whole clips (0.3 s to minutes) |

Series are raw: unnormalized, unpadded, of different lengths, and not shuffled within a part.
[corpus_loader.py](corpus_loader.py) reads all four parts as one shuffled stream of 1-D float32 arrays:

```python
import corpus_loader
for batch in corpus_loader.batches("corpus", batch_size=64):
    ...  # a list of 1-D float32 arrays of different lengths
```

**Rules.** The corpus is the only training data allowed: no other datasets and no pretrained weights. It carries
no labels.

## Example

[example_train.py](example_train.py) trains a small encoder with [stable-pretraining](https://github.com/galilai-group/stable-pretraining)
and exports it. The encoder (`SeriesViT`) is stable-pretraining's `ViT` run on each series as a 1 x 1024 image
with 1 x 32 patches, after per-series normalization; the objective (predict whether a series' next step goes up,
stays or goes down) is deliberately simple. It is a starting point that shows what a submission must get right,
not a strong baseline.

## Model format

An ONNX file (opset 17 or newer), an encoder only:

- **input**: float32, shape exactly `[64, 1, 1024]`: a batch of 64 univariate windows of 1,024 values
- **output**: float32, shape `[64, D]` with `D <= 2048`, all finite

Exactly one input and one output; their names do not matter. Export in eval mode with a static batch of 64.

**Normalize inside your model.** Windows reach the model as stored, at very different scales and offsets across
tasks, and the evaluator does not rescale them. Series shorter than 1,024 values are padded; longer ones are cut
into windows whose embeddings are averaged.

## Evaluation

[evaluate.py](evaluate.py) runs on a GPU with onnxruntime:

```bash
python evaluate.py model.onnx path/to/eval_data      # prints {"score": ..., "score_task1": ..., ...}
```

The evaluation data is private. For each task, every window is embedded by your frozen model, the embeddings of
each example are averaged, and a ridge probe is fitted on part of the examples and scored on the rest by ROC AUC
(one-vs-rest per class, averaged over classes). Each task's AUC is mapped to a score that is **0 for an untrained
encoder and 1 for a strong reference model**, unbounded both ways. The leaderboard **score** is a weighted
mean of the task scores; `score_task1`, `score_task2`, ... are shown alongside without saying which task is which.

The full evaluation embeds about 350,000 windows and must finish within an hour; the reference model takes about
4 minutes on one GPU.

## Validate submission

Check your model before uploading it:

```bash
uv run test_submission.py path/to/model.onnx
```

It loads and runs the model with the evaluator's own code and explains how to fix any problem it finds. The checks
are listed at the top of [test_submission.py](test_submission.py). Only submit once validation passes.
[example_submission/model.onnx](example_submission/model.onnx) is a model that passes.
