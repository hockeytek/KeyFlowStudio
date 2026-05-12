# Security Policy

## Reporting

Report security issues directly to the repository owner through a private channel. Do not open public issues for credentials, vulnerabilities, private infrastructure details, or media that should not be shared publicly.

## Secrets

Do not commit:

- AWS access keys or credential CSV files.
- `.env` files.
- SSH private keys or `.pem` files.
- Model provider tokens.
- Private media or customer footage.

If a secret is exposed:

1. Revoke or rotate the secret immediately.
2. Remove it from the working tree.
3. Notify the maintainer so Git history and release assets can be reviewed.
4. Audit recent access logs where applicable.

## Model Assets

Model weights and checkpoints may have separate licenses. Keep them outside Git and document download/setup steps instead of redistributing them.