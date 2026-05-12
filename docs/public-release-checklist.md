# Release Hygiene Checklist

Use this checklist before publishing a new release asset, changing packaged resources, or updating a public preview tag.

## Secrets

- [ ] Run a secret scan against the working tree.
- [ ] Run a secret scan against Git history.
- [ ] Confirm no AWS access keys, `.env` files, SSH keys, or token files are tracked.
- [ ] Rotate any credential that was ever committed accidentally.

## Large And Restricted Assets

- [ ] Confirm no model weights are tracked.
- [ ] Confirm no generated outputs are tracked.
- [ ] Confirm no private footage, customer media, or local test clips are tracked or attached to releases.
- [ ] Confirm `.gitignore` excludes local caches, weights, checkpoints, and outputs.

## Legal And Licensing

- [ ] Review third-party model licenses and redistribution terms.
- [ ] Review bundled fonts, icons, sounds, and UI assets.
- [ ] Confirm the published license text matches the intended reuse model.

## Documentation

- [ ] Add screenshots or a short demo that can be redistributed.
- [ ] Verify installation docs on a clean machine.
- [ ] Verify cloud GPU docs with a non-production AWS account.
- [ ] Make limitations clear in the README.

## CI And Quality

- [ ] Confirm GitHub Actions passes on a clean checkout.
- [ ] Keep CPU-only smoke tests available for CI.
- [ ] Keep GPU-heavy tests opt-in.