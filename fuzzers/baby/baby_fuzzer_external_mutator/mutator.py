#!/usr/bin/env python3
"""External mutator for LibAFL's `ExternalProcessMutator`.

The script has two modes, selected by the ``--mutator`` switch:

* ``--mutator`` (`ExternalProcessMutator` appends it to the arguments, i.e., it runs
  ``python3 mutator.py [args...] --mutator``): serve the line-based protocol.
  Every line consists of fields, separated by single spaces. The first field is
  the request id (hex), the others are hex-encoded data. For every request line on
  stdin, answer with exactly one reply line with the same id on stdout.
  A reply consisting of the id only means "no mutation" (the fuzzer counts it as `Skipped`).

  - bytes inputs (default):   ``<id> <bytes>``
  - multipart inputs (``--multipart``): ``<id> <key1> <value1> <key2> <value2> ...``,
    where the keys are the `Debug` representation of the Rust keys (e.g., ``"a"``
    with quotes for `String` keys). Keys are opaque: reuse, but never invent keys.

  The fuzzer does not tell which kind of input it sends: a mutator is built for one kind.

* otherwise: stand-alone mode to test and debug the mutation strategy in isolation, e.g.::

      python3 mutator.py --seed 1 -n 10 "hello world"
      printf 'abc' | python3 mutator.py -n 5 --chain
      python3 mutator.py --multipart --part a=hello --part b=world -n 5
      python3 mutator.py --mutator      # speak the protocol by hand (type e.g. "1 616263")

For testing the fuzzer's error handling, faults can be injected in mutator mode
(counters are per process, so they start over after every respawn), e.g.::

    --hang-every 100      sleep instead of answering every 100th request (timeout)
    --stderr-every 50     write a warning to stderr every 50th request
    --crash-every 200     exit without answering every 200th request
    --garbage-every 70    answer with invalid hex every 70th request
    --wrong-id-every 90   answer with a wrong id every 90th request
"""

import argparse
import os
import random
import sys
import time


# Some "interesting" values, AFL-style
INTERESTING = [0x00, 0x01, 0x7F, 0x80, 0xFF, ord("a"), ord("b"), ord("c"), ord(" "), ord("\n")]


def random_byte(rng: random.Random) -> int:
    """Mostly printable bytes (like the baby fuzzer's generator), sometimes anything."""
    roll = rng.random()
    if roll < 0.7:
        return rng.randint(0x20, 0x7E)
    if roll < 0.85:
        return rng.choice(INTERESTING)
    return rng.randint(0, 255)


# ----- single mutation operations, each works in-place on a bytearray -----


def op_bit_flip(data: bytearray, rng: random.Random, max_len: int) -> None:
    if data:
        pos = rng.randrange(len(data))
        data[pos] ^= 1 << rng.randrange(8)


def op_set_byte(data: bytearray, rng: random.Random, max_len: int) -> None:
    if data:
        data[rng.randrange(len(data))] = random_byte(rng)


def op_arith(data: bytearray, rng: random.Random, max_len: int) -> None:
    if data:
        pos = rng.randrange(len(data))
        data[pos] = (data[pos] + rng.randint(-8, 8)) & 0xFF


def op_insert(data: bytearray, rng: random.Random, max_len: int) -> None:
    if len(data) < max_len:
        data.insert(rng.randint(0, len(data)), random_byte(rng))


def op_delete(data: bytearray, rng: random.Random, max_len: int) -> None:
    if len(data) > 1:
        # delete 1..4 bytes, but never everything (an empty reply means "no mutation")
        start = rng.randrange(len(data))
        end = min(len(data), start + rng.randint(1, min(4, len(data) - 1)))
        del data[start:end]


def op_duplicate(data: bytearray, rng: random.Random, max_len: int) -> None:
    if data and len(data) < max_len:
        start = rng.randrange(len(data))
        chunk = data[start : start + rng.randint(1, 8)]
        pos = rng.randint(0, len(data))
        data[pos:pos] = chunk[: max_len - len(data)]


def op_swap(data: bytearray, rng: random.Random, max_len: int) -> None:
    if len(data) > 1:
        a, b = rng.randrange(len(data)), rng.randrange(len(data))
        data[a], data[b] = data[b], data[a]


OPERATIONS = [op_bit_flip, op_set_byte, op_arith, op_insert, op_delete, op_duplicate, op_swap]


def mutate(data: bytes, rng: random.Random, args: argparse.Namespace) -> bytes:
    """Applies a stack of 1..max_ops random operations."""
    buf = bytearray(data[: args.max_len])
    if not buf:
        buf.append(random_byte(rng))
    for _ in range(rng.randint(1, args.max_ops)):
        rng.choice(OPERATIONS)(buf, rng, args.max_len)
    return bytes(buf)


Part = tuple[bytes, bytes]  # (key, i.e., its Debug representation, value)


def mutate_parts(parts: list[Part], rng: random.Random, args: argparse.Namespace) -> list[Part]:
    """Mutates a multipart input: usually the value of one part, sometimes the list of parts.

    Keys are opaque: we can only reuse existing ones (the fuzzer can't create new keys).
    """
    parts = list(parts)
    if not parts:
        return parts
    roll = rng.random()
    if roll < 0.1 and len(parts) < args.max_parts:
        # duplicate a (mutated) part
        key, value = rng.choice(parts)
        parts.insert(rng.randint(0, len(parts)), (key, mutate(value, rng, args)))
    elif roll < 0.15 and len(parts) > 1:
        del parts[rng.randrange(len(parts))]
    elif roll < 0.2 and len(parts) > 1:
        a, b = rng.randrange(len(parts)), rng.randrange(len(parts))
        parts[a], parts[b] = parts[b], parts[a]
    else:
        idx = rng.randrange(len(parts))
        key, value = parts[idx]
        parts[idx] = (key, mutate(value, rng, args))
    return parts


def every(n: int, count: int) -> bool:
    return n > 0 and count % n == 0


def format_line(req_id: str, fields: list[bytes]) -> bytes:
    """A protocol line: the id, followed by the hex-encoded fields, separated by single spaces."""
    return " ".join([req_id, *(field.hex() for field in fields)]).encode("ascii") + b"\n"


def serve(args: argparse.Namespace, rng: random.Random) -> int:
    """The protocol loop: one request line in, one reply line out."""
    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer

    def reply(line: bytes) -> None:
        stdout.write(line)
        stdout.flush()

    def warn(msg: str) -> None:
        print(f"mutator: {msg}", file=sys.stderr, flush=True)

    count = 0
    while True:
        line = stdin.readline()
        if not line:  # EOF: the fuzzer is gone
            return 0
        count += 1

        # Split on single spaces only: empty fields (empty values) are valid.
        raw_fields = line.rstrip(b"\r\n").split(b" ")
        req_id = raw_fields[0].decode("ascii", errors="replace")
        try:
            fields = [bytes.fromhex(field.decode("ascii")) for field in raw_fields[1:]]
        except ValueError:
            warn(f"received invalid hex: {line!r}")
            reply(format_line(req_id, []))  # only the id == no mutation
            continue
        if (args.multipart and len(fields) % 2 != 0) or (not args.multipart and len(fields) != 1):
            kind = "key/value pairs (--multipart)" if args.multipart else "a single field (no --multipart)"
            warn(f"expected {kind}, got {len(fields)} fields: {line!r}")
            reply(format_line(req_id, []))
            continue

        # --- fault injection, for testing the fuzzer's error handling ---
        if every(args.crash_every, count):
            warn(f"injected crash at request #{count}")
            os._exit(3)
        if every(args.hang_every, count):
            time.sleep(args.hang_seconds)
        if every(args.stderr_every, count):
            warn(f"injected warning at request #{count}")
        if every(args.garbage_every, count):
            reply(f"{req_id} this-is-not-hex\n".encode())
            continue
        if every(args.wrong_id_every, count):
            reply(format_line(f"{int(req_id, 16) + 1:x}", fields))
            continue

        if args.multipart:
            parts = mutate_parts(list(zip(fields[0::2], fields[1::2])), rng, args)
            reply(format_line(req_id, [field for part in parts for field in part]))
        else:
            reply(format_line(req_id, [mutate(fields[0], rng, args)]))


def show_parts(parts: list[Part]) -> str:
    return "[" + ", ".join(f"{key.decode(errors='replace')}: {value!r}" for key, value in parts) + "]"


def show_bytes(data: bytes) -> str:
    return f"{data!r} ({data.hex()})"


def standalone(args: argparse.Namespace, rng: random.Random) -> int:
    """Prints mutations of a given input, to test the mutation strategy in isolation."""
    if args.multipart:
        # Keys are formatted like Rust's `Debug` for `String` keys: with quotes
        data = [(f'"{key}"'.encode(), value.encode()) for key, _, value in (p.partition("=") for p in args.part)]
    elif args.hex is not None:
        data = bytes.fromhex(args.hex)
    elif args.input is not None:
        data = args.input.encode()
    else:
        data = sys.stdin.buffer.read()
    mutate_fn, show = (mutate_parts, show_parts) if args.multipart else (mutate, show_bytes)
    print(f"input       : {show(data)}")
    for i in range(args.count):
        mutated = mutate_fn(data, rng, args)
        if args.chain:
            data = mutated
        print(f"mutation {i:3}: {show(mutated)}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seed", type=int, default=None, help="RNG seed (default: random)")
    parser.add_argument("--max-ops", type=int, default=4, help="max. stacked operations per mutation")
    parser.add_argument("--max-len", type=int, default=64, help="max. length of mutated inputs (or parts)")
    parser.add_argument(
        "--multipart", action="store_true", help="mutate multipart inputs (key/value pairs) instead of bytes"
    )
    parser.add_argument("--max-parts", type=int, default=8, help="max. number of parts (with --multipart)")

    serve_group = parser.add_argument_group("mutator mode (--mutator)")
    serve_group.add_argument(
        "--mutator", action="store_true", help="serve the protocol (passed by the fuzzer)"
    )
    serve_group.add_argument("--hang-every", type=int, default=0, metavar="N")
    serve_group.add_argument("--hang-seconds", type=float, default=10.0, metavar="S")
    serve_group.add_argument("--stderr-every", type=int, default=0, metavar="N")
    serve_group.add_argument("--crash-every", type=int, default=0, metavar="N")
    serve_group.add_argument("--garbage-every", type=int, default=0, metavar="N")
    serve_group.add_argument("--wrong-id-every", type=int, default=0, metavar="N")

    alone_group = parser.add_argument_group("stand-alone mode")
    alone_group.add_argument("input", nargs="?", help="input string (default: read raw bytes from stdin)")
    alone_group.add_argument("--hex", help="input as hex instead")
    alone_group.add_argument(
        "--part", action="append", default=[], metavar="KEY=VALUE", help="a part (with --multipart), repeatable"
    )
    alone_group.add_argument("-n", "--count", type=int, default=10, help="number of mutations to print")
    alone_group.add_argument("--chain", action="store_true", help="mutate the previous mutation, not the input")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    if args.mutator:
        return serve(args, rng)
    return standalone(args, rng)


if __name__ == "__main__":
    sys.exit(main())

