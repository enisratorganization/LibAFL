# Baby fuzzer with a Python mutator

This is the minimalistic [baby fuzzer](../baby_fuzzer), but all mutations run a Python function
in an **embedded** CPython interpreter (via PyO3), using LibAFL's `PyMutator` (`libafl/src/mutators/python.rs`).
In contrast to the [external mutator fuzzer](../baby_fuzzer_external_mutator), there is no process and no
protocol: the mutation is a direct function call, so it is much faster — but a blocking `mutate` blocks the fuzzer.

It runs on a single core until a crash occurs and then exits.

## The `PyMutator`

The Python side consists of up to two functions (at least one is required):

- `mutate(b: bytes) -> bytes` for bytes inputs: it receives the input as `bytes` and returns the mutated
  input as a bytes-like object (`bytes`, `bytearray`, or a list of `int`s).
- `mutate_multi(parts: list[tuple[str, bytes]]) -> list[tuple[str, bytes]]` for `MultipartInput`s
  (feature `multipart_inputs`, see below).

The function that matches the input type is called (if it is missing, mutating fails with an error).
The code can be passed inline or imported as a module:

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
- There is **no timeout**: a blocking function blocks the fuzzer.

### Multipart inputs

`mutate_multi` gets one `(key, value)` tuple per part of a `MultipartInput<I, K>` and returns a list of such tuples.
The key type `K` here is always `str`
 Python can change the values and reorder, remove,
  or duplicate parts (a pair with an existing key clones that key), but a result with a new key is `Skipped`
  (with a `warn` log), like exceptions, wrong types, no parts, unchanged parts, and a part larger than `max_size`.
- The constructor test-runs `mutate_multi` with the single part `("probe", <8 bytes>)`.

```python
def mutate_multi(parts):  # [("header", b"..."), ("payload", b"...")]
    key, value = random.choice(parts)
    return parts + [(key, value + b"!")]  # duplicates an (existing) key
```

## The Python mutator

[`mutator.py`](./mutator.py) is imported as a module by default. Its `mutate` applies a stack of
1..8 random byte-level operations (bit flip, set/arithmetic byte, insert, delete, duplicate, swap),
using AFL-style "interesting" values, mostly printable.

Its `mutate_multi` usually mutates the value of one part with `mutate`, and sometimes duplicates
(with a mutated value), removes, or swaps parts.

It also runs stand-alone, to test and debug the mutation strategy in isolation:

```sh
python3 mutator.py --seed 1 -n 10 "hello world"   # print 10 mutations of the input
printf 'abc' | python3 mutator.py -n 5 --chain    # read from stdin, mutate repeatedly
python3 mutator.py --part a=hello --part b=world -n 5   # mutate_multi on two parts
```

## Building and Running

Building requires a CPython installation that PyO3 supports (found via `python3` on the `PATH`,
or the `PYO3_PYTHON` env var; the resulting binary links `libpython`, so that interpreter has to
stay available at runtime).

```sh
cargo build --release
cargo build --release --features multipart --target-dir target/multipart   # MultipartInput (`header` + `payload`)
./target/release/baby_fuzzer_py_mutator                                  # imports `mutator.py` from this folder
RUST_LOG=warn  ./target/release/baby_fuzzer_py_mutator                   # show skipped mutations that raised
RUST_LOG=debug ./target/release/baby_fuzzer_py_mutator --iters 10        # ... plus unchanged/too-big results
./target/release/baby_fuzzer_py_mutator --module my_mutator --module-dir /path/to/dir
./target/release/baby_fuzzer_py_mutator --module some.pkg.on.python.path --module-dir ""
```

Options: `--iters <n>` (stop after `n` iterations instead of running until a crash),
`--module <name>` (default `mutator`), `--module-dir <path>` (prepended to `sys.path`,
default: this folder; empty to rely on `PYTHONPATH`/cwd).

With the feature `multipart`, the fuzzer works on a `MultipartInput<BytesInput, String>` with the parts `header`
and `payload` (generated randomly), and `mutator.py`'s `mutate_multi` is used. The harness only looks at the
concatenated `payload` parts (the crash needs them to start with `abc`), the `header` is ignored. The module has to match the input type: a module with only
`mutate` fails (error in the fuzzing loop) on multipart inputs, and vice versa.

To see the crash being found, run it in a scratch directory: it exits with status 134 (the harness
panics on `abc...`) and leaves the crashing input in `./crashes/`.

Unlike the other baby fuzzers, `log` is used without `release_max_level_info`, so debug logs are available in release builds.
