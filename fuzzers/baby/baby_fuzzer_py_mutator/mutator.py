#!/usr/bin/env python3
"""Python mutator module for LibAFL's `PyMutator`.

The fuzzer imports this module (`PyMutator::from_module_in`) and calls `mutate` for every mutation.
The embedded interpreter seeds Python's global `random` module from the fuzzer's state RNG,
so each fuzzer run gets its own (reproducible) mutation stream.

`mutate(b: bytes) -> bytes` applies a stack of random byte-level operations (AFL-style):

>>> mutate(b"hello world")  # doctest: +SKIP
b'he\x00lo worlD'

Stand-alone mode, to test and debug the mutation strategy in isolation::

    python3 mutator.py --seed 1 -n 10 "hello world"
    printf 'abc' | python3 mutator.py -n 5 --chain
"""

import argparse
import random
import sys

# Some "interesting" values, AFL-style
INTERESTING = [0x00, 0x01, 0x7F, 0x80, 0xFF, ord("a"), ord("b"), ord("c"), ord(" "), ord("\n")]

# Max length of a mutated input (the fuzzer's max size is enforced on the Rust side, too)
MAX_LEN = 4096
# Max number of operations applied per mutation
MAX_OPS = 8


def random_byte(rng: random.Random) -> int:
    """Mostly printable bytes (like the baby fuzzer's generator), sometimes anything."""
    roll = rng.random()
    if roll < 0.7:
        return rng.randint(0x20, 0x7E)
    if roll < 0.85:
        return rng.choice(INTERESTING)
    return rng.randint(0, 255)


# ----- single mutation operations, each works in-place on a bytearray -----


def op_bit_flip(data: bytearray, rng: random.Random) -> None:
    if data:
        data[rng.randrange(len(data))] ^= 1 << rng.randrange(8)


def op_set_byte(data: bytearray, rng: random.Random) -> None:
    if data:
        data[rng.randrange(len(data))] = random_byte(rng)


def op_arith(data: bytearray, rng: random.Random) -> None:
    if data:
        pos = rng.randrange(len(data))
        data[pos] = (data[pos] + rng.randint(-8, 8)) & 0xFF


def op_insert(data: bytearray, rng: random.Random) -> None:
    if len(data) < MAX_LEN:
        data.insert(rng.randint(0, len(data)), random_byte(rng))


def op_delete(data: bytearray, rng: random.Random) -> None:
    if len(data) > 1:
        # delete 1..4 bytes, but never everything (an unchanged/empty input means "skip")
        start = rng.randrange(len(data))
        end = min(len(data), start + rng.randint(1, min(4, len(data) - 1)))
        del data[start:end]


def op_duplicate(data: bytearray, rng: random.Random) -> None:
    if data and len(data) < MAX_LEN:
        start = rng.randrange(len(data))
        chunk = data[start : start + rng.randint(1, 8)]
        pos = rng.randint(0, len(data))
        data[pos:pos] = chunk[: MAX_LEN - len(data)]


def op_swap(data: bytearray, rng: random.Random) -> None:
    if len(data) > 1:
        a, b = rng.randrange(len(data)), rng.randrange(len(data))
        data[a], data[b] = data[b], data[a]


OPERATIONS = [op_bit_flip, op_set_byte, op_arith, op_insert, op_delete, op_duplicate, op_swap]


def mutate(b: bytes) -> bytes:
    """The function called by the `PyMutator`: applies 1..`MAX_OPS` random operations."""
    rng = random  # the module proxies the global RNG, seeded by the fuzzer
    buf = bytearray(b[:MAX_LEN])
    if not buf:
        buf.append(random_byte(rng))
    for _ in range(rng.randint(1, MAX_OPS)):
        rng.choice(OPERATIONS)(buf, rng)
    return bytes(buf)


def main() -> None:
    """Stand-alone mode: print mutations of an input, for testing the strategy in isolation."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input", nargs="?", help="input (default: read stdin)")
    parser.add_argument("-n", type=int, default=5, help="number of mutations to print")
    parser.add_argument("--seed", type=int, help="seed the global random module")
    parser.add_argument(
        "--chain", action="store_true", help="mutate the previous result instead of the input"
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
    data = (
        sys.stdin.buffer.read() if args.input is None else args.input.encode()
    )
    for _ in range(args.n):
        data = mutate(data)
        print(repr(data))
        if not args.chain:
            data = args.input.encode()


if __name__ == "__main__":
    main()
