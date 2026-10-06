//! This module implements the [`PyMutator`], where each mutation runs a Python function
//! in an embedded `CPython` interpreter (via `PyO3`) to mutate bytes in a target-specific way.
//!
//! The Python side is always a single function, `mutate(b: bytes) -> bytes`:
//! it receives the input as `bytes` and returns the mutated input as a bytes-like object
//! (`bytes`, `bytearray`, or a list of `int`s).
//! The code can be given inline ([`PyMutator::from_code`]), or as a module that is imported
//! ([`PyMutator::from_module`] / [`PyMutator::from_module_in`]).
//!
//! The interpreter is initialized once per process (with signal handling disabled, so Python
//! never installs handlers nor raises `KeyboardInterrupt`) and never finalized.
//! Python's global `random` module is seeded from the state's RNG ([`HasRand`]),
//! so different fuzzers get different mutation streams (and reproducible ones with a fixed seed).
//!
//! If `mutate` raises an exception, or returns something that is not bytes-like,
//! the error is logged with `warn` severity and the mutation is [`MutationResult::Skipped`]
//! (the interpreter is *not* reset). Returning the unchanged input, or more than
//! [`HasMaxSize::max_size`] bytes, is `Skipped` as well (logged with `debug`).
//!
//! Unlike [`ExternalProcessMutator`](super::ExternalProcessMutator), there is **no timeout**:
//! a blocking `mutate` (e.g., an endless loop) blocks the fuzzer.
use alloc::{
    borrow::Cow,
    ffi::CString,
    string::{String, ToString},
    vec::Vec,
};
use std::path::Path;

use libafl_bolts::{Error, Named, rands::Rand};
use pyo3::{
    PyErr,
    prelude::{Bound, Py, PyAny, Python},
    types::{PyAnyMethods, PyBytes, PyDict, PyDictMethods},
};

use super::{MutationResult, Mutator};
use crate::{
    corpus::CorpusId,
    inputs::{HasMutatorBytes, ResizableMutator},
    state::{HasMaxSize, HasRand},
};

/// The name of the Python function that performs the mutation: `mutate(b: bytes) -> bytes`
pub const MUTATE_FN: &str = "mutate";

/// Converts a [`PyErr`] into a libafl-native [`Error`]
#[allow(clippy::needless_pass_by_value)] // We need this signature for `.map_err`
fn convert_error(err: PyErr) -> Error {
    Error::illegal_state(format!("Python error: {err}"))
}

/// A [`Mutator`] that runs a Python function `mutate(b: bytes) -> bytes` in an embedded
/// `CPython` interpreter, on inputs consisting of bytes.
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
    /// The Python `mutate` function
    mutate: Py<PyAny>,
}

impl core::fmt::Debug for PyMutator {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PyMutator")
            .field("name", &self.name)
            .field("source", &self.source)
            .finish_non_exhaustive()
    }
}

impl PyMutator {
    /// Creates a [`PyMutator`] from inline Python code, which has to define a top-level
    /// [`MUTATE_FN`] function (module-level imports in the code are fine).
    ///
    /// Returns an error if the code cannot be compiled, does not define the function,
    /// or the function fails on a test input (it is called once during construction).
    pub fn from_code<S: HasRand>(state: &mut S, code: &str) -> Result<Self, Error> {
        Self::create(state, format!("inline code ({} bytes)", code.len()), |py| {
            let code = CString::new(code)
                .map_err(|_| Error::illegal_argument("Python code must not contain a NUL"))?;
            let globals = PyDict::new(py);
            py.run(&code, Some(&globals), None).map_err(convert_error)?;
            PyDictMethods::get_item(&globals, MUTATE_FN)
                .map_err(convert_error)?
                .ok_or_else(|| {
                    Error::illegal_argument(format!(
                        "Python code does not define a `{MUTATE_FN}` function"
                    ))
                })
                .map(Bound::unbind)
        })
    }

    /// Creates a [`PyMutator`] by importing the module `module` (which has to be on `sys.path`,
    /// e.g., via `PYTHONPATH`) and using its [`MUTATE_FN`] function.
    ///
    /// Returns an error if the module cannot be imported or the function fails on a test input.
    pub fn from_module<S: HasRand>(state: &mut S, module: &str) -> Result<Self, Error> {
        Self::create(state, format!("module {module:?}"), |py| {
            import_mutate(py, module)
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
                import_mutate(py, module)
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
    /// state's RNG, gets the function via `get_mutate`, and calls it once to test it.
    fn create<S, F>(state: &mut S, source: String, get_mutate: F) -> Result<Self, Error>
    where
        S: HasRand,
        F: FnOnce(Python<'_>) -> Result<Py<PyAny>, Error>,
    {
        Python::initialize();
        let mutator = Self {
            name: Cow::Borrowed("PyMutator"),
            source,
            mutate: Python::attach(|py| -> Result<Py<PyAny>, Error> {
                py.import("random")
                    .and_then(|random| random.call_method1("seed", (state.rand_mut().next(),)))
                    .map_err(convert_error)?;
                get_mutate(py)
            })?,
        };
        // Test-run the function once (like the `LuaMutator` does) to catch broken mutators early.
        let probe = state.rand_mut().next().to_le_bytes().to_vec();
        mutator.call(&probe).map_err(|err| {
            Error::illegal_state(format!(
                "{} from {} failed the test run ({err}), rejecting mutator",
                mutator.name, mutator.source
            ))
        })?;
        Ok(mutator)
    }

    /// Calls the Python function with `bytes` and extracts the bytes-like result.
    fn call(&self, bytes: &[u8]) -> Result<Vec<u8>, PyErr> {
        Python::attach(|py| {
            let arg = PyBytes::new(py, bytes);
            self.mutate.call1(py, (arg,))?.extract(py)
        })
    }
}

/// Imports `module` and returns its [`MUTATE_FN`] function.
fn import_mutate(py: Python<'_>, module: &str) -> Result<Py<PyAny>, Error> {
    py.import(module)
        .and_then(|m| m.getattr(MUTATE_FN))
        .map(Bound::unbind)
        .map_err(convert_error)
}

impl<I, S> Mutator<I, S> for PyMutator
where
    I: HasMutatorBytes + ResizableMutator<u8>,
    S: HasMaxSize,
{
    fn mutate(&mut self, state: &mut S, input: &mut I) -> Result<MutationResult, Error> {
        let mutated = match self.call(input.mutator_bytes()) {
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

    #[test]
    fn test_ctor_errors() {
        let mut state = TestState::new(5);
        // Syntax error
        assert!(PyMutator::from_code(&mut state, "def mutate(b):").is_err());
        // No `mutate` function
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
