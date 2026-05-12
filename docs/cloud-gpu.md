# Cloud GPU Setup

KeyFlow Studio includes an optional EC2 worker for GPU inference. Use cloud GPU for workflows that need more VRAM or throughput than local hardware can provide.

## Recommended Workflow

1. Develop and run smoke tests locally.
2. Validate model loading on a local GPU if available.
3. Use a known commit or release tag.
4. Pull the code on EC2.
5. Run a short cloud smoke test.
6. Run production-sized jobs only after the smoke test passes.

## AWS Quotas

For `g6e.xlarge` in `eu-central-1`, request quota for `Running On-Demand G and VT instances`.

Useful sizing:

- `1 x g6e.xlarge` = 4 vCPU quota.
- `2 x g6e.xlarge` = 8 vCPU quota.

Spot quota is separate. If `All G and VT Spot Instance Requests` is `0`, launch On-Demand, not Spot.

## Worker

The EC2 worker is located in [../ec2_worker/](../ec2_worker/).

Typical launch shape:

```bash
cd ec2_worker
uvicorn worker:app --host 0.0.0.0 --port 8080 --workers 1
```

Keep one job per GPU unless the model and VRAM profile are known to support concurrency.

## Credentials

Use AWS profiles, instance roles, or environment-specific secret stores for credentials. Do not paste credentials into issue reports or logs.

Common local credential file names include:

- `keyflow-cli_accessKeys.csv`
- `.env`
- `.env.*`
- `*.pem`