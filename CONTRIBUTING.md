# Contributing

Thank you for taking an interest in KeyFlow Studio. This repository is public, but the project is still in preview: keep contributions focused, tested, and clear about model or platform assumptions.

## Development Flow

1. Create a focused branch for each change.
2. Keep changes scoped to one feature, fix, or documentation update.
3. Run relevant tests before opening a pull request.
4. Keep credentials, model weights, generated media, and local environment files out of commits.

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

Never include access keys, AWS credentials, private footage, model checkpoints, or downloaded weights in commits. If a secret is committed accidentally, rotate it immediately and contact the maintainer so the exposure can be handled before more work lands on top of it.