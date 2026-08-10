# ProtEnc extraction refactor (code review follow-up)

**Origin Stamp**
- Timestamp: 2026-08-10T17:36:51Z
- Git Commit: `98bd110` (`master` after CLI–encoder parity merge)
- MLflow Run ID: none

## What changed

### Integrated on `master` (`456e726`)

Parallel workstreams from the code review were merged via `cursor/refactor-review-fixes-a24e`:

- **CLI bugs:** output-format inference uses `--output_format`; `--no_gpu` forces CPU; output choices limited to `lmdb`/`hdf5`; CSV default columns aligned.
- **DataParallel:** ESMC models skip `nn.DataParallel` with a warning; DDP is not implemented.
- **Encoder performance:** prepare→encode streams per batch (no full in-memory prepare); `empty_cache_on_batch` defaults false; autocast uses device type; OOM retry clears cache once.
- **Tests:** list `encode` yields `(index, embedding)`; `skip_large_models` logic fixed; `test_dataparallel.py` added.
- **Docs:** README aligned with CLI paths, HDF5 support, encode API tuple behavior, `repr_layers`, pooling defaults.

### CLI–encoder parity (`cursor/cli-encoder-parity-a24e`)

Bulk CLI (`protenc` / `protenc.console.extract`) now routes through `get_encoder` / `ProteinEncoder` instead of duplicating `get_model`, autocast, and DataParallel setup.

- `_create_encoder()` maps CLI flags to encoder constructor args.
- `encode_sequences_batch()` and `show_progress=False` avoid nested tqdm in file-driven pipelines.
- Pooling uses `_maybe_pool_embedding()` (mean only for `PER_RESIDUE` when `--compute_mean`).
- I/O remains in the CLI layer: readers, LMDB skip-keys, `max_prot_len`, wildcards, `cast_to`, `dry_run`.

## What still needs doing

| Area | Gap |
|------|-----|
| **Distributed inference** | Only `nn.DataParallel`; no DDP / multi-node. |
| **Preprocessor workers** | `preprocess_workers` / `--num_workers` accepted but encoding path does not parallelize preparation. |
| **LMDB throughput** | Writes still on the inference thread (GPU stalls noted in prior CLI TODO). |
| **MCP parity** | No bulk I/O, multi-layer, or data-parallel tools for agents. |
| **Tests** | `test_get_encoder` compares two separate loads; MPNN weights and ESMC HF checkpoints missing in CI; 7 env-dependent failures remain. |
| **Formats** | Parquet/pickle output not implemented; README todo items (model offloading, sharding, evals) still open. |

## Merge / verification note

Target test baseline after full integration: `pytest tests/` — expect ~18 passed, ~15 skipped, ~7 failed (environment), unless weights/checkpoints are provisioned.
