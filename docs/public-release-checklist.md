# Public Release Checklist

Use this checklist before changing repository visibility from private to public.

## Secrets

- [ ] Run a secret scan against the working tree.
- [ ] Run a secret scan against Git history.
- [ ] Confirm no AWS access keys, `.env` files, SSH keys, or token files are tracked.
- [ ] Rotate any credential that was ever committed accidentally.

## Large And Private Assets

- [ ] Confirm no model weights are tracked.
- [ ] Confirm no generated outputs are tracked.
- [ ] Confirm no private footage, customer media, or local test clips are tracked.
- [ ] Confirm `.gitignore` excludes local caches, weights, checkpoints, and outputs.

## Legal And Licensing

- [ ] Decide whether the project is open-source or source-available.
- [ ] Review third-party model licenses and redistribution terms.
- [ ] Review bundled fonts, icons, sounds, and UI assets.
- [ ] Update `LICENSE` if public reuse is intended.

## Documentation

- [ ] Add screenshots or a short demo that can be redistributed.
- [ ] Verify installation docs on a clean machine.
- [ ] Verify cloud GPU docs with a non-production AWS account.
- [ ] Make limitations clear in the README.

## CI And Quality

- [ ] Confirm GitHub Actions passes on a clean checkout.
- [ ] Keep CPU-only smoke tests available for CI.
- [ ] Keep GPU-heavy tests opt-in.