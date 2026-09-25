# Baby fuzzer with an external mutator

This is the minimalistic [baby fuzzer](../baby_fuzzer), but all mutations are delegated to an
external program via LibAFL's `ExternalProcessMutator` (`libafl/src/mutators/external.rs`).

It runs on a single core until a crash occurs and then exits.

## Protocol

The external program is spawned once, with the switch `--mutator` appended to its arguments,
and stdin/stdout/stderr connected to the fuzzer. Every line consists of fields separated by
*exactly one* space; the first field is an id (`u64` as hex, without `0x`), all others are hex-encoded data:

- For each mutation, the fuzzer writes a request line to the program's stdin.
- The program answers with exactly one reply line on stdout, starting with the id of the request.
  A reply with a different id makes the fuzzer respawn the program (and the mutation is `Skipped`).
  A reply consisting of the id only means "no mutation" (`Skipped`); returning the unchanged input is `Skipped` as well.
- Anything on stderr is logged with `warn` severity. With `--kill-on-stderr`, the program is also killed
  and respawned (and the mutation is `Skipped`).
- If the program does not answer within the timeout (or exits/crashes), it is killed and respawned (`Skipped`).
  The first request to a freshly spawned process gets an additional startup timeout (default: 5s).

The data fields depend on the input type (a mutator program is built for one of them, the fuzzer doesn't tell):

| Input                                       | Line                                               | Example                                   |
|---------------------------------------------|----------------------------------------------------|-------------------------------------------|
| `BytesInput` (anything `HasMutatorBytes`)   | `<id> <bytes>`                                     | `1f 616263` (`abc`)                       |
| `MultipartInput<I, K>`                      | `<id> <key1> <value1> <key2> <value2> ...`         | `2 226122 7879 226222 ` (`"a": xy`, `"b": ""`) |

Multipart keys are the `Debug` representation of `K` (e.g., `"a"` *with quotes* for `String` keys).
The fuzzer maps the keys of the reply back to the input's keys, so parts may be changed, reordered,
removed, or duplicated, but every key in the reply must be one of the request's keys.
Empty values are empty fields (note the trailing space in the example).

## The Python mutator

[`mutator.py`](./mutator.py) has two modes:

- `--mutator`: serve the protocol (the fuzzer runs `python3 mutator.py [args...] --mutator`).
- otherwise: stand-alone mode, to test and debug the mutation strategy in isolation:

```sh
python3 mutator.py --seed 1 -n 10 "hello world"                 # print 10 mutations of the input
printf 'abc' | python3 mutator.py -n 5 --chain                  # read input from stdin, mutate repeatedly
python3 mutator.py --multipart --part a=hello --part b=world    # mutate a multipart input
python3 mutator.py --mutator                                    # speak the protocol by hand (type e.g. `1 616263`)
```

Without `--multipart`, the script mutates bytes inputs, with it, multipart inputs.
Requests of the other kind are answered with "no mutation" and a warning on stderr.

For testing the fuzzer's error handling, the script can inject faults (counters restart with every respawn):
`--hang-every N [--hang-seconds S]`, `--stderr-every N`, `--crash-every N`, `--garbage-every N` (invalid hex),
`--wrong-id-every N`.

Since the switch is a plain argument, the script can also be spawned directly via its shebang (`-- ./mutator.py`).

## Running

```sh
cargo build --release
./target/release/baby_fuzzer_external_mutator                        # uses `python3 ./mutator.py`
RUST_LOG=warn  ./target/release/baby_fuzzer_external_mutator --timeout-ms 100 -- python3 mutator.py --hang-every 300
RUST_LOG=warn  ./target/release/baby_fuzzer_external_mutator --kill-on-stderr -- python3 mutator.py --stderr-every 100
RUST_LOG=libafl::mutators=debug ./target/release/baby_fuzzer_external_mutator --iters 1  # every roundtrip
RUST_LOG=libafl::mutators=trace ./target/release/baby_fuzzer_external_mutator --iters 1  # ... including the raw lines
```

With the feature `multipart`, the fuzzer uses a `MultipartInput<BytesInput, String>` with a `header` and a `payload`
part instead (the harness only looks at the `payload` parts), and passes `--multipart` to the default mutator:

```sh
cargo build --release --features multipart
./target/release/baby_fuzzer_external_mutator
RUST_LOG=warn ./target/release/baby_fuzzer_external_mutator -- python3 mutator.py --multipart --wrong-id-every 90
```

Options: `--timeout-ms <ms>` (default 1000), `--kill-on-stderr`, `--iters <n>` (stop after `n` iterations
instead of running until a crash), and `-- <program> [args...]` to use any other external mutator
(remember `--multipart` for `mutator.py` with the `multipart` feature).

Unlike the other baby fuzzers, `log` is used without `release_max_level_info`, so debug logs are available in release builds.

