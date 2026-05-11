# Contributing

KeyFlow Studio is currently maintained as a private development repository. These guidelines prepare the project for a future public workflow.

## Development Flow

1. Create a focused branch for each change.
2. Keep changes scoped to one feature, fix, or documentation update.
3. Run relevant tests before opening a pull request.
4. Do not commit secrets, model weights, generated media, or local environment files.

## Local Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Validation

Use focused tests while developing:

```bash
pytest tests/test_node_graph_dialog_smoke.py -q
```

Run broader checks when touching shared runtime, graph contracts, or worker behavior.

## Pull Requests

Pull requests should include:

- What changed.
- Why it changed.
- How it was tested.
- Known risks or follow-up work.

## Security And Private Assets

Never include access keys, AWS credentials, private footage, model checkpoints, or downloaded weights in commits. If a secret is committed accidentally, rotate it immediately and remove it from Git history before sharing the repository.