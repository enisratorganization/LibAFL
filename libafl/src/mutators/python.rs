//! This module implements the [`PyMutator`], where each mutation runs a Python function
//! in an embedded `CPython` interpreter (via `PyO3`) to mutate bytes in a target-specific way.
//!
//! The Python side consists of up to two functions (at least one is required):
//!
//! - `mutate(b: bytes) -> bytes` mutates a bytes input: it receives the input as `bytes` and
//!   returns the mutated input as a bytes-like object (`bytes`, `bytearray`, or a list of `int`s).
//! - `mutate_multi(parts: list[tuple[str, bytes]]) -> list[tuple[str, bytes]]` mutates a
//!   [`MultipartInput`](crate::inputs::MultipartInput) (feature `multipart_inputs`),
//!   see [below](#multipart-inputs).
//!
//! The function matching the input type is used; mutating an input for which the function is missing
//! is an error. The code can be given inline ([`PyMutator::from_code`]), or as a module that is imported
//! ([`PyMutator::from_module`] / [`PyMutator::from_module_in`]).
//!
//! The interpreter is initialized once per process (with signal handling disabled, so Python
//! never installs handlers nor raises `KeyboardInterrupt`) and never finalized.
//! Python's global `random` module is seeded from the state's RNG ([`HasRand`]),
//! so different fuzzers get different mutation streams (and reproducible ones with a fixed seed).
//!
//! If a function raises an exception, or returns something that is not bytes-like,
//! the error is logged with `warn` severity and the mutation is [`MutationResult::Skipped`]
//! (the interpreter is *not* reset). Returning the unchanged input, or more than
//! [`HasMaxSize::max_size`] bytes, is `Skipped` as well (logged with `debug`).
//!
//! # Multipart inputs
//!
//! `mutate_multi` gets a list of `(key, value)` tuples, one per part, and returns a list of tuples.
//! The key type `K` of a `MultipartInput<I, K>` is generic, but Python needs a `str`. As a quirk
//! (shared with the [`ExternalProcessMutator`](super::ExternalProcessMutator)), the key is
//! formatted with its [`Debug`](core::fmt::Debug) implementation, and then
//! **one leading and one trailing `"` are removed, if present**. So for `K = String`, the key `a`
//! is passed as `a` (its `Debug` string is `"a"`). The tests only cover `K = String`.
//!
//! Keys can't be created from strings, so each key of the result has to match one of the input's keys:
//! the `Debug` string of the key has to be equal to the returned `str` with the removed `"` re-added
//! (so a returned `"a"` with quotes does *not* match the `String` key `a`).
//! Python can change the parts' bytes and reorder, remove, or duplicate parts (a new pair with an
//! existing key clones that key), but it can't introduce new keys: a result with an unknown key is
//! logged with `warn` and `Skipped`. As for bytes, exceptions, wrongly typed results, empty results (no parts),
//! unchanged results, and parts larger than [`HasMaxSize::max_size`] are `Skipped`, too.
//!
//! Unlike [`ExternalProcessMutator`](super::ExternalProcessMutator), there is **no timeout**:
//! a blocking `mutate` (e.g., an endless loop) blocks the fuzzer.
use alloc::{
    borrow::Cow,
    ffi::CString,
    string::{String, ToString},
    vec::Vec,
};
use core::fmt::Debug;
use std::path::Path;

use libafl_bolts::{Error, Named, rands::Rand};
use pyo3::{
    PyErr,
    prelude::{Bound, Py, PyAny, Python},
    types::{PyAnyMethods, PyBytes, PyDict, PyList},
};

use super::{MutationResult, Mutator};
#[cfg(feature = "multipart_inputs")]
use crate::inputs::MultipartInput;
use crate::{
    corpus::CorpusId,
    inputs::{HasMutatorBytes, ResizableMutator},
    state::{HasMaxSize, HasRand},
};

/// The name of the Python function that mutates bytes inputs: `mutate(b: bytes) -> bytes`
pub const MUTATE_FN: &str = "mutate";
/// The name of the Python function that mutates multipart inputs:
/// `mutate_multi(parts: list[tuple[str, bytes]]) -> list[tuple[str, bytes]]`
pub const MUTATE_MULTI_FN: &str = "mutate_multi";

/// Converts a [`PyErr`] into a libafl-native [`Error`]
#[allow(clippy::needless_pass_by_value)] // We need this signature for `.map_err`
fn convert_error(err: PyErr) -> Error {
    Error::illegal_state(format!("Python error: {err}"))
}

/// A [`Mutator`] that runs a Python function `mutate(b: bytes) -> bytes` (for bytes inputs) or
/// `mutate_multi(parts: list[tuple[str, bytes]]) -> list[tuple[str, bytes]]` (for
/// [`MultipartInput`](crate::inputs::MultipartInput)s) in an embedded `CPython` interpreter.
///
/// See the [module-level documentation](self) for the details.
///
/// # Example
///
/// ```rust,ignore
/// // Inline code (mind the newlines: `mutate` has to be defined at the top level)
/// let mutator = PyMutator::from_code(&mut state, "def mutate(b: bytes) -> bytes:\n    return b + b'!'\n")?;
/// // Or a module `my_mutator` (from `./py_mutators`), providing the same function
/// let mutator = PyMutator::from_module_in(&mut state, "./py_mutators", "my_mutator")?;
/// let mut stages = tuple_list!(StdMutationalStage::new(mutator));
/// ```
pub struct PyMutator {
    name: Cow<'static, str>,
    /// Where the `mutate` function came from, for logs and [`Debug`]
    source: String,
    /// The Python `mutate` function, if defined
    mutate: Option<Py<PyAny>>,
    /// The Python `mutate_multi` function, if defined
    mutate_multi: Option<Py<PyAny>>,
}

impl Debug for PyMutator {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PyMutator")
            .field("name", &self.name)
            .field("source", &self.source)
            .finish_non_exhaustive()
    }
}

impl PyMutator {
    /// Creates a [`PyMutator`] from inline Python code, which has to define a top-level
    /// [`MUTATE_FN`] and/or [`MUTATE_MULTI_FN`] function (module-level imports in the code are fine).
    ///
    /// Returns an error if the code cannot be compiled, defines neither function,
    /// or a function fails on a test input (each is called once during construction).
    pub fn from_code<S: HasRand>(state: &mut S, code: &str) -> Result<Self, Error> {
        Self::create(state, format!("inline code ({} bytes)", code.len()), |py| {
            let code = CString::new(code)
                .map_err(|_| Error::illegal_argument("Python code must not contain a NUL"))?;
            let globals = PyDict::new(py);
            py.run(&code, Some(&globals), None).map_err(convert_error)?;
            // The functions are looked up as attributes of the namespace, like for modules
            py.import("types")
                .and_then(|types| types.call_method("SimpleNamespace", (), Some(&globals)))
                .map_err(convert_error)
        })
    }

    /// Creates a [`PyMutator`] by importing the module `module` (which has to be on `sys.path`,
    /// e.g., via `PYTHONPATH`) and using its [`MUTATE_FN`] and/or [`MUTATE_MULTI_FN`] function.
    ///
    /// Returns an error if the module cannot be imported, defines neither function,
    /// or a function fails on a test input.
    pub fn from_module<S: HasRand>(state: &mut S, module: &str) -> Result<Self, Error> {
        Self::create(state, format!("module {module:?}"), |py| {
            py.import(module)
                .map(Bound::into_any)
                .map_err(convert_error)
        })
    }

    /// Creates a [`PyMutator`] from a module in the given directory:
    /// `dir` is prepended to `sys.path`, then the module is imported as in [`Self::from_module`].
    pub fn from_module_in<S: HasRand, P: AsRef<Path>>(
        state: &mut S,
        dir: P,
        module: &str,
    ) -> Result<Self, Error> {
        let dir = dir.as_ref();
        Self::create(
            state,
            format!("module {module:?} in {}", dir.display()),
            |py| {
                py.import("sys")
                    .and_then(|sys| sys.getattr("path"))
                    .and_then(|path| path.call_method1("insert", (0, dir.display().to_string())))
                    .map_err(convert_error)?;
                py.import(module)
                    .map(Bound::into_any)
                    .map_err(convert_error)
            },
        )
    }

    /// Sets a custom name for this mutator.
    #[must_use]
    pub fn with_name<N>(mut self, name: N) -> Self
    where
        N: Into<Cow<'static, str>>,
    {
        self.name = name.into();
        self
    }

    /// The common constructor: initializes the interpreter, seeds Python's `random` from the
    /// state's RNG, gets the namespace with the functions via `get_namespace`, and calls each
    /// function it defines once to test it.
    fn create<S, F>(state: &mut S, source: String, get_namespace: F) -> Result<Self, Error>
    where
        S: HasRand,
        F: FnOnce(Python<'_>) -> Result<Bound<'_, PyAny>, Error>,
    {
        Python::initialize();
        let mutator = Python::attach(|py| -> Result<Self, Error> {
            py.import("random")
                .and_then(|random| random.call_method1("seed", (state.rand_mut().next(),)))
                .map_err(convert_error)?;
            let namespace = get_namespace(py)?;
            let find = |name| -> Result<Option<Py<PyAny>>, Error> {
                let function = namespace.getattr_opt(name).map_err(convert_error)?;
                Ok(function.map(Bound::unbind))
            };
            Ok(Self {
                name: Cow::Borrowed("PyMutator"),
                source,
                mutate: find(MUTATE_FN)?,
                mutate_multi: find(MUTATE_MULTI_FN)?,
            })
        })?;
        if mutator.mutate.is_none() && mutator.mutate_multi.is_none() {
            return Err(Error::illegal_argument(format!(
                "{} does not define a `{MUTATE_FN}` or `{MUTATE_MULTI_FN}` function",
                mutator.source
            )));
        }
        // Test-run the functions once (like the `LuaMutator` does) to catch broken mutators early.
        let probe = state.rand_mut().next().to_le_bytes().to_vec();
        let test_runs = [
            mutator
                .mutate
                .as_ref()
                .map(|f| Self::call_bytes(f, &probe).map(drop)),
            mutator
                .mutate_multi
                .as_ref()
                .map(|f| Self::call_multi(f, &[("probe", &probe)]).map(drop)),
        ];
        if let Some(err) = test_runs.into_iter().flatten().find_map(Result::err) {
            return Err(Error::illegal_state(format!(
                "{} from {} failed the test run ({err}), rejecting mutator",
                mutator.name, mutator.source
            )));
        }
        Ok(mutator)
    }

    /// Calls `mutate(bytes)` and extracts the bytes-like result.
    fn call_bytes(function: &Py<PyAny>, bytes: &[u8]) -> Result<Vec<u8>, PyErr> {
        Python::attach(|py| function.call1(py, (PyBytes::new(py, bytes),))?.extract(py))
    }

    /// Calls `mutate_multi(parts)` with a list of `(key, bytes)` tuples and extracts the
    /// resulting list of `(key, bytes)` tuples.
    fn call_multi(
        function: &Py<PyAny>,
        parts: &[(&str, &[u8])],
    ) -> Result<Vec<(String, Vec<u8>)>, PyErr> {
        Python::attach(|py| {
            let parts = parts
                .iter()
                .map(|&(key, bytes)| (key, PyBytes::new(py, bytes)));
            function.call1(py, (PyList::new(py, parts)?,))?.extract(py)
        })
    }
}

impl<I, S> Mutator<I, S> for PyMutator
where
    I: HasMutatorBytes + ResizableMutator<u8>,
    S: HasMaxSize,
{
    fn mutate(&mut self, state: &mut S, input: &mut I) -> Result<MutationResult, Error> {
        let Some(function) = &self.mutate else {
            return Err(Error::illegal_state(format!(
                "{} from {} does not define `{MUTATE_FN}`, cannot mutate bytes inputs",
                self.name, self.source
            )));
        };
        let mutated = match Self::call_bytes(function, input.mutator_bytes()) {
            Ok(mutated) => mutated,
            Err(err) => {
                log::warn!(
                    "{}: python mutate from {} raised {err}, skipping",
                    self.name,
                    self.source
                );
                return Ok(MutationResult::Skipped);
            }
        };
        if mutated.as_slice() == input.mutator_bytes() {
            log::debug!("{}: python mutate returned the unchanged input", self.name);
            return Ok(MutationResult::Skipped);
        }
        if mutated.len() > state.max_size() {
            log::debug!(
                "{}: python mutate returned {} bytes, exceeding max size {}, skipping",
                self.name,
                mutated.len(),
                state.max_size()
            );
            return Ok(MutationResult::Skipped);
        }
        input.resize(mutated.len(), 0);
        input.mutator_bytes_mut().copy_from_slice(&mutated);
        Ok(MutationResult::Mutated)
    }

    #[inline]
    fn post_exec(&mut self, _state: &mut S, _new_corpus_id: Option<CorpusId>) -> Result<(), Error> {
        Ok(())
    }
}

/// The key as Python sees it: its [`Debug`] string, without the surrounding `"` (if any).
/// This is a quirk to get a `str` from any `K`, see the [module docs](self#multipart-inputs).
#[cfg(feature = "multipart_inputs")]
fn py_key(debug: &str) -> &str {
    let key = debug.strip_prefix('"').unwrap_or(debug);
    key.strip_suffix('"').unwrap_or(key)
}

/// Mutates a [`MultipartInput`] with the Python function `mutate_multi`, which gets the parts as
/// a list of `(key, bytes)` (keys formatted as described at [`py_key`]) and returns such a list.
///
/// A key that Python returns has to match the [`Debug`] string of an existing key (with `"`
/// re-added), since keys can't be created from strings: the matching key is cloned.
/// This allows to change, reorder, remove, and duplicate parts, but not to add new keys.
#[cfg(feature = "multipart_inputs")]
impl<I, K, S> Mutator<MultipartInput<I, K>, S> for PyMutator
where
    I: HasMutatorBytes + ResizableMutator<u8> + Clone,
    K: Debug + Clone,
    S: HasMaxSize,
{
    fn mutate(
        &mut self,
        state: &mut S,
        input: &mut MultipartInput<I, K>,
    ) -> Result<MutationResult, Error> {
        let Some(function) = &self.mutate_multi else {
            return Err(Error::illegal_state(format!(
                "{} from {} does not define `{MUTATE_MULTI_FN}`, cannot mutate multipart inputs",
                self.name, self.source
            )));
        };
        // `Debug` strings of the keys, to match the keys of the result against
        let debug_keys: Vec<String> = input
            .parts()
            .iter()
            .map(|(key, _)| format!("{key:?}"))
            .collect();
        let request: Vec<(&str, &[u8])> = debug_keys
            .iter()
            .zip(input.parts())
            .map(|(key, (_, part))| (py_key(key), part.mutator_bytes()))
            .collect();
        let reply = match Self::call_multi(function, &request) {
            Ok(reply) => reply,
            Err(err) => {
                log::warn!(
                    "{}: python mutate_multi from {} raised {err}, skipping",
                    self.name,
                    self.source
                );
                return Ok(MutationResult::Skipped);
            }
        };
        if reply.is_empty()
            || reply
                .iter()
                .map(|(k, v)| (k.as_str(), v.as_slice()))
                .eq(request)
        {
            log::debug!(
                "{}: python mutate_multi returned no parts or the unchanged input",
                self.name
            );
            return Ok(MutationResult::Skipped);
        }

        let max_size = state.max_size();
        let mut parts = Vec::with_capacity(reply.len());
        for (key, bytes) in &reply {
            // Matching `py_key(debug) == key` is the same as re-adding the `"` that Python does not
            // see and comparing with the `Debug` string (only if that has quotes in the first place)
            let Some(idx) = debug_keys.iter().position(|k| py_key(k) == key) else {
                log::warn!(
                    "{}: python mutate_multi returned unknown key {key:?}, skipping",
                    self.name
                );
                return Ok(MutationResult::Skipped);
            };
            if bytes.len() > max_size {
                log::debug!(
                    "{}: python mutate_multi returned a part of {} bytes, exceeding max size {max_size}, skipping",
                    self.name,
                    bytes.len()
                );
                return Ok(MutationResult::Skipped);
            }
            // Clone the key and the part (as a template for the new bytes) of the matching pair
            let (key, template) = &input.parts()[idx];
            let mut part = template.clone();
            part.resize(bytes.len(), 0);
            part.mutator_bytes_mut().copy_from_slice(bytes);
            parts.push((key.clone(), part));
        }
        *input = MultipartInput::new(parts);
        Ok(MutationResult::Mutated)
    }

    #[inline]
    fn post_exec(&mut self, _state: &mut S, _new_corpus_id: Option<CorpusId>) -> Result<(), Error> {
        Ok(())
    }
}

impl Named for PyMutator {
    fn name(&self) -> &Cow<'static, str> {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;
    use std::{fs, path::PathBuf};

    use libafl_bolts::{rands::StdRand, serdeany::SerdeAnyMap};

    use super::PyMutator;
    use crate::{
        HasMetadata,
        inputs::{BytesInput, HasMutatorBytes},
        mutators::{MutationResult, Mutator},
        state::{HasMaxSize, HasRand},
    };

    /// A state with a seeded rng and a settable max size
    struct TestState {
        rand: StdRand,
        max_size: usize,
    }

    impl TestState {
        fn new(seed: u64) -> Self {
            Self {
                rand: StdRand::with_seed(seed),
                max_size: 1337,
            }
        }
    }
    impl HasRand for TestState {
        type Rand = StdRand;
        fn rand(&self) -> &Self::Rand {
            &self.rand
        }
        fn rand_mut(&mut self) -> &mut Self::Rand {
            &mut self.rand
        }
    }
    impl HasMaxSize for TestState {
        fn max_size(&self) -> usize {
            self.max_size
        }
        fn set_max_size(&mut self, max_size: usize) {
            self.max_size = max_size;
        }
    }
    impl HasMetadata for TestState {
        fn metadata_map(&self) -> &SerdeAnyMap {
            unimplemented!()
        }
        fn metadata_map_mut(&mut self) -> &mut SerdeAnyMap {
            unimplemented!()
        }
    }

    /// Mutates `input` once with `mutator`, returning the result and the mutated bytes
    fn run(
        state: &mut TestState,
        mutator: &mut PyMutator,
        input: &[u8],
    ) -> (MutationResult, Vec<u8>) {
        let mut input = BytesInput::new(input.to_vec());
        let result = mutator.mutate(state, &mut input).unwrap();
        (result, input.mutator_bytes().to_vec())
    }

    #[test]
    fn test_inline_code() {
        let mut state = TestState::new(1337);
        let mut mutator =
            PyMutator::from_code(&mut state, "def mutate(b): return b + b'!'").unwrap();
        assert_eq!(
            run(&mut state, &mut mutator, b"abc"),
            (MutationResult::Mutated, b"abc!".to_vec())
        );
    }

    #[test]
    fn test_unchanged_and_return_types() {
        let mut state = TestState::new(1);
        let mut unchanged = PyMutator::from_code(&mut state, "def mutate(b): return b").unwrap();
        assert_eq!(
            run(&mut state, &mut unchanged, b"abc"),
            (MutationResult::Skipped, b"abc".to_vec())
        );
        // bytearray and list-of-int results are accepted, too
        for code in [
            "def mutate(b): return bytearray(b) + b'!'",
            "def mutate(b): return list(b) + [33]",
        ] {
            let mut mutator = PyMutator::from_code(&mut state, code).unwrap();
            assert_eq!(
                run(&mut state, &mut mutator, b"x"),
                (MutationResult::Mutated, b"x!".to_vec()),
                "code: {code}"
            );
        }
    }

    #[test]
    fn test_errors_skipped() {
        let mut state = TestState::new(2);
        // Only fail on empty inputs, so the constructor's test run succeeds
        let mut raising =
            PyMutator::from_code(&mut state, "def mutate(b): return b'x' if b else 1/0").unwrap();
        assert_eq!(
            run(&mut state, &mut raising, b""),
            (MutationResult::Skipped, Vec::new())
        );
        // Not bytes-like
        let mut garbage =
            PyMutator::from_code(&mut state, "def mutate(b): return b'x' if b else 42").unwrap();
        assert_eq!(
            run(&mut state, &mut garbage, b""),
            (MutationResult::Skipped, Vec::new())
        );
        // The interpreter recovers: the next mutation works
        assert_eq!(
            run(&mut state, &mut garbage, b"a").0,
            MutationResult::Mutated
        );
    }

    #[test]
    fn test_max_size() {
        let mut state = TestState::new(3);
        let mut mutator = PyMutator::from_code(&mut state, "def mutate(b): return b * 3").unwrap();
        state.set_max_size(4);
        assert_eq!(
            run(&mut state, &mut mutator, b"ab"),
            (MutationResult::Skipped, b"ab".to_vec())
        );
        state.set_max_size(10);
        assert_eq!(
            run(&mut state, &mut mutator, b"ab"),
            (MutationResult::Mutated, b"ababab".to_vec())
        );
    }

    #[test]
    fn test_python_random() {
        // The global `random` module (seeded from the state's rng) has to be usable and advance.
        let mut state = TestState::new(7);
        let mut mutator = PyMutator::from_code(
            &mut state,
            "import random\ndef mutate(b):\n    return bytes([random.randrange(256) for _ in range(16)])\n",
        )
        .unwrap();
        let mut seen: Vec<Vec<u8>> = Vec::new();
        for _ in 0..5 {
            let out = run(&mut state, &mut mutator, b"x").1;
            assert_eq!(out.len(), 16);
            assert!(
                !seen.contains(&out),
                "random produced the same 16 bytes twice"
            );
            seen.push(out);
        }
    }

    #[test]
    fn test_module() {
        let dir: PathBuf =
            std::env::temp_dir().join(format!("libafl_py_mutator_test_{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let module = "libafl_py_mutator_test_mod";
        fs::write(
            dir.join(format!("{module}.py")),
            "def mutate(b): return b + b'-from-module'\n",
        )
        .unwrap();

        let mut state = TestState::new(4);
        let mut mutator = PyMutator::from_module_in(&mut state, &dir, module).unwrap();
        assert_eq!(
            run(&mut state, &mut mutator, b"x"),
            (MutationResult::Mutated, b"x-from-module".to_vec())
        );
        // After the import, the module is in `sys.modules`, so plain `from_module` works now, too
        let mut mutator = PyMutator::from_module(&mut state, module).unwrap();
        assert_eq!(
            run(&mut state, &mut mutator, b"y"),
            (MutationResult::Mutated, b"y-from-module".to_vec())
        );
        fs::remove_dir_all(&dir).unwrap();
    }

    /// Tests for [`crate::inputs::MultipartInput`], with `String` keys only (see the module docs).
    /// The `Debug` string of the key `a` is `"a"`, but Python sees `a`.
    #[cfg(feature = "multipart_inputs")]
    mod multipart {
        use alloc::{
            string::{String, ToString},
            vec::Vec,
        };

        use core::sync::atomic::{AtomicU64, Ordering};

        use super::TestState;
        use crate::{
            inputs::{BytesInput, HasMutatorBytes, MultipartInput},
            mutators::{MutationResult, Mutator, PyMutator},
            state::HasMaxSize,
        };

        type Input = MultipartInput<BytesInput, String>;
        type Parts = Vec<(String, Vec<u8>)>;

        fn parts(parts: &[(&str, &[u8])]) -> Parts {
            parts
                .iter()
                .map(|(k, v)| ((*k).to_string(), v.to_vec()))
                .collect()
        }

        /// Mutates the parts `[("a", "x"), ("b", "y")]` once with the `mutate_multi` function in `code`
        /// (which also gets the constructor's test run with the key `probe`).
        fn mutate_ab(code: &str) -> (MutationResult, Parts) {
            // A new seed for each call: equal seeds re-seed Python's global `random` (shared by all
            // tests running in parallel) to equal states, which breaks `test_python_random`.
            static SEED: AtomicU64 = AtomicU64::new(1000);
            let mut state = TestState::new(SEED.fetch_add(1, Ordering::Relaxed));
            let mut mutator = PyMutator::from_code(&mut state, code).unwrap();
            run_multi(&mut state, &mut mutator, &[("a", b"x"), ("b", b"y")])
        }

        fn run_multi(
            state: &mut TestState,
            mutator: &mut PyMutator,
            input: &[(&str, &[u8])],
        ) -> (MutationResult, Parts) {
            let mut input: Input = MultipartInput::new(
                parts(input)
                    .into_iter()
                    .map(|(k, v)| (k, BytesInput::new(v)))
                    .collect(),
            );
            let result = mutator.mutate(state, &mut input).unwrap();
            let parts = input
                .parts()
                .iter()
                .map(|(k, v)| (k.clone(), v.mutator_bytes().to_vec()))
                .collect();
            (result, parts)
        }

        #[test]
        fn test_keys_without_quotes() {
            // Python sees the keys without `"`, and the result is mapped back to the existing keys
            let (result, out) = mutate_ab(
                "def mutate_multi(parts): return [(k, v + b'|' + k.encode()) for k, v in parts]",
            );
            assert_eq!(result, MutationResult::Mutated);
            assert_eq!(out, parts(&[("a", b"x|a"), ("b", b"y|b")]));
        }

        #[test]
        fn test_reorder_remove_duplicate() {
            // Reorder, and a new pair with an existing key (clones the key)
            let (result, out) = mutate_ab(
                "def mutate_multi(parts):\n    return [parts[-1], parts[0], ('a', b'new')]",
            );
            assert_eq!(result, MutationResult::Mutated);
            assert_eq!(out, parts(&[("b", b"y"), ("a", b"x"), ("a", b"new")]));
            // Remove, and an empty part
            let (result, out) = mutate_ab("def mutate_multi(parts): return [('b', b'')]");
            assert_eq!(result, MutationResult::Mutated);
            assert_eq!(out, parts(&[("b", b"")]));
        }

        #[test]
        fn test_quote_in_key() {
            // The `Debug` string of the key `q"x` is `"q\"x"`: only the outer quotes are removed
            let mut state = TestState::new(12);
            let mut mutator = PyMutator::from_code(
                &mut state,
                "def mutate_multi(parts): return [(k, k.encode()) for k, _ in parts]",
            )
            .unwrap();
            let (result, out) = run_multi(&mut state, &mut mutator, &[("q\"x", b"v")]);
            assert_eq!(result, MutationResult::Mutated);
            assert_eq!(out, parts(&[("q\"x", b"q\\\"x")]));
        }

        #[test]
        fn test_skipped() {
            let unchanged = parts(&[("a", b"x"), ("b", b"y")]);
            // Results that are fine for the constructor's test run (key `probe`), but get skipped
            for code in [
                "def mutate_multi(parts): return parts",       // unchanged
                "def mutate_multi(parts): return list(parts)", // unchanged
                "def mutate_multi(parts): return []",          // no parts
                "def mutate_multi(parts): return [('zzz', b'x')]", // unknown key
                "def mutate_multi(parts): return [('\"a\"', b'x')]", // quotes are not Python's
                "def mutate_multi(parts): return [('a', b'x'), ('c', b'')]", // one unknown key
            ] {
                assert_eq!(
                    mutate_ab(code),
                    (MutationResult::Skipped, unchanged.clone()),
                    "code: {code}"
                );
            }
            // Exceptions and wrongly typed results: only after the constructor's test run (it rejects those)
            for result in [
                "1/0",              // raises
                "[(1, b'x')]",      // key is not a str
                "[('a', 'x')]",     // value is not bytes
                "b'a'",             // not a list
                "[('a', b'x', 1)]", // not a pair
            ] {
                let code = format!(
                    "def mutate_multi(parts):\n    if parts[0][0] == 'probe': return parts\n    return {result}"
                );
                assert_eq!(
                    mutate_ab(&code),
                    (MutationResult::Skipped, unchanged.clone()),
                    "result: {result}"
                );
            }
        }

        #[test]
        fn test_max_size() {
            let mut state = TestState::new(13);
            let mut mutator = PyMutator::from_code(
                &mut state,
                "def mutate_multi(parts): return [(k, v * 3) for k, v in parts]",
            )
            .unwrap();
            // The limit applies to each part
            state.set_max_size(2);
            let input: &[(&str, &[u8])] = &[("a", b"xy"), ("b", b"z")];
            assert_eq!(
                run_multi(&mut state, &mut mutator, input),
                (MutationResult::Skipped, parts(input))
            );
            state.set_max_size(6);
            assert_eq!(
                run_multi(&mut state, &mut mutator, input),
                (
                    MutationResult::Mutated,
                    parts(&[("a", b"xyxyxy"), ("b", b"zzz")])
                )
            );
        }

        #[test]
        fn test_functions_have_to_exist() {
            let mut state = TestState::new(14);
            // `mutate_multi` only: fine to construct, but bytes inputs are an error
            let mut multi_only = PyMutator::from_code(
                &mut state,
                "def mutate_multi(parts): return [(k, v + b'!') for k, v in parts]",
            )
            .unwrap();
            let mut bytes = BytesInput::new(b"x".to_vec());
            assert!(multi_only.mutate(&mut state, &mut bytes).is_err());
            // `mutate` only: multipart inputs are an error
            let mut bytes_only =
                PyMutator::from_code(&mut state, "def mutate(b): return b + b'!'").unwrap();
            let mut input: Input = MultipartInput::new(Vec::new());
            assert!(bytes_only.mutate(&mut state, &mut input).is_err());
            // Both functions can live side by side
            let mut both = PyMutator::from_code(
                &mut state,
                "def mutate(b): return b + b'!'\ndef mutate_multi(parts): return [(k, v + b'?') for k, v in parts]",
            )
            .unwrap();
            assert_eq!(
                both.mutate(&mut state, &mut bytes).unwrap(),
                MutationResult::Mutated
            );
            assert_eq!(bytes.mutator_bytes(), b"x!");
            assert_eq!(
                run_multi(&mut state, &mut both, &[("k", b"v")]),
                (MutationResult::Mutated, parts(&[("k", b"v?")]))
            );
        }

        #[test]
        fn test_ctor_test_run() {
            let mut state = TestState::new(15);
            // A broken `mutate_multi` is rejected at construction, even if `mutate` works
            assert!(
                PyMutator::from_code(
                    &mut state,
                    "def mutate(b): return b\ndef mutate_multi(parts): raise RuntimeError('nope')"
                )
                .is_err()
            );
            // ... and so is a result that is not a list of pairs
            assert!(
                PyMutator::from_code(&mut state, "def mutate_multi(parts): return 42").is_err()
            );
        }
    }

    #[test]
    fn test_ctor_errors() {
        let mut state = TestState::new(5);
        // Syntax error
        assert!(PyMutator::from_code(&mut state, "def mutate(b):").is_err());
        // Neither `mutate` nor `mutate_multi`
        assert!(PyMutator::from_code(&mut state, "x = 1").is_err());
        // NUL byte in the code
        assert!(PyMutator::from_code(&mut state, "def mutate(b): return b\x00").is_err());
        // Broken `mutate`: fails during the constructor's test run
        assert!(
            PyMutator::from_code(&mut state, "def mutate(b): raise RuntimeError('nope')").is_err()
        );
        // Unknown module
        assert!(PyMutator::from_module(&mut state, "no_such_module_xyz123").is_err());
    }
}
