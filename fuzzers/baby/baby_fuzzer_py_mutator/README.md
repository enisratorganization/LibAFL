# Baby fuzzer with a Python mutator

This is the minimalistic [baby fuzzer](../baby_fuzzer), but all mutations run a Python function
in an **embedded** CPython interpreter (via PyO3), using LibAFL's `PyMutator` (`libafl/src/mutators/python.rs`).
In contrast to the [external mutator fuzzer](../baby_fuzzer_external_mutator), there is no process and no
protocol: the mutation is a direct function call, so it is much faster — but a blocking `mutate` blocks the fuzzer.

It runs on a single core until a crash occurs and then exits.

## The `PyMutator`

The Python side is always a single function, `mutate(b: bytes) -> bytes`:
it receives the input as `bytes` and returns the mutated input as a bytes-like object
(`bytes`, `bytearray`, or a list of `int`s). The code can be passed inline or imported as a module:

```rust
// Inline code (`mutate` has to be defined at the top level)
let mutator = PyMutator::from_code(&mut state, "def mutate(b): return b + b'!'")?;
// A module providing the same function, imported from a directory / from `sys.path`
let mutator = PyMutator::from_module_in(&mut state, "./py_mutators", "my_mutator")?;
let mutator = PyMutator::from_module(&mut state, "my_mutator")?; // e.g., via PYTHONPATH
```

Semantics (see the module docs in `libafl/src/mutators/python.rs` for the full list):

- Python's global `random` module is seeded from the fuzzer's state RNG at construction,
  so every fuzzer instance mutates differently (and reproducibly, given a fixed seed).
- The mutator is test-run once during construction: broken code (syntax error, no `mutate`, exception,
  wrong return type) makes the constructor fail instead of silently never mutating.
- Later exceptions or non-bytes results are logged with `warn` and the mutation is `Skipped`
  (the interpreter is not reset). Returning the unchanged input or more than `max_size` bytes
  is `Skipped` as well (logged with `debug`).
- There is **no timeout** and only `bytes` inputs are supported (no multipart).

## The Python mutator

[`mutator.py`](./mutator.py) is imported as a module by default. Its `mutate` applies a stack of
1..8 random byte-level operations (bit flip, set/arithmetic byte, insert, delete, duplicate, swap),
using AFL-style "interesting" values, mostly printable.

It also runs stand-alone, to test and debug the mutation strategy in isolation:

```sh
python3 mutator.py --seed 1 -n 10 "hello world"   # print 10 mutations of the input
printf 'abc' | python3 mutator.py -n 5 --chain    # read from stdin, mutate repeatedly
```

## Building and Running

Building requires a CPython installation that PyO3 supports (found via `python3` on the `PATH`,
or the `PYO3_PYTHON` env var; the resulting binary links `libpython`, so that interpreter has to
stay available at runtime).

```sh
cargo build --release
./target/release/baby_fuzzer_py_mutator                                  # imports `mutator.py` from this folder
RUST_LOG=warn  ./target/release/baby_fuzzer_py_mutator                   # show skipped mutations that raised
RUST_LOG=debug ./target/release/baby_fuzzer_py_mutator --iters 10        # ... plus unchanged/too-big results
./target/release/baby_fuzzer_py_mutator --module my_mutator --module-dir /path/to/dir
./target/release/baby_fuzzer_py_mutator --module some.pkg.on.python.path --module-dir ""
```

Options: `--iters <n>` (stop after `n` iterations instead of running until a crash),
`--module <name>` (default `mutator`), `--module-dir <path>` (prepended to `sys.path`,
default: this folder; empty to rely on `PYTHONPATH`/cwd).

To see the crash being found, run it in a scratch directory: it exits with status 134 (the harness
panics on `abc...`) and leaves the crashing input in `./crashes/`.

Unlike the other baby fuzzers, `log` is used without `release_max_level_info`, so debug logs are available in release builds.
