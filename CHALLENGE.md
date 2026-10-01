
# Setup

Installation instructions

# Training Data

# Evaluation

Provide an evaluate.py that takes two arguments:
- [onnx-model-path] that produces a json "score" used to generate the leaderboard.
- [evaluation-data-path]

It should output a json score.


# Submission 

Model Format: input and output dimensions expected.

## Validating submission

Include a test_submission.py [onnx-model-path] that checks submissions match needed format and any other constraints for the challenge.
