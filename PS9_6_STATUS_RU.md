# PYTHIA PS9.6 — Compression Cost Search Engine

PS9.6 adds an experimental Python reference for bounded transform-chain search.

## Features
- Reversible modular Delta and XOR predictors for lags 1,2,3,4,8,16,32.
- Reversible byte-lane transposition for widths 4,8,16,32,64.
- Reversible bit-plane conversion (standalone; not yet included in chain search because framing/padding cost is not implemented).
- Bounded beam search over transform chains, with duplicate transforms prevented.
- Search uses a prefix capped at 262,144 bytes by default; the winning chain is then re-evaluated on the full input.
- Score uses measured zlib compressed bytes plus transform descriptor overhead. zlib is a proxy, not the PYTHIA production coder.

## Validation
- Python compile and self-test passed.
- Transform and chained round-trip tests passed on empty, patterned, repeated, and randomized data.
- On `mixed_10mb_compression_test.bin` (10,485,760 bytes), depth 2, beam 5: identity selected. Full-input zlib payload was 5,249,847 bytes; descriptor-adjusted score identical. No gain was found by this candidate family on this sample.

## Limits / next work
- Search does not yet invoke PYTHIA's Rust archive pipeline or measure final archive bytes.
- Bit-plane candidate needs a framing format including original length and overhead before integration into chain search.
- Prefix scoring can miss patterns occurring later; future version should sample multiple windows and then compare finalists on full data.
- Rust compilation was not performed (toolchain unavailable in environment).
