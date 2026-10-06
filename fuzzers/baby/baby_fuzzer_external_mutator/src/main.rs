//! A baby fuzzer that uses the [`ExternalProcessMutator`]:
//! every mutation is delegated to an external program (by default, `python3 mutator.py`),
//! which is started with the `--mutator` switch appended to its arguments.
//!
//! Usage:
//!
//! ```text
//! baby_fuzzer_external_mutator [--timeout-ms <ms>] [--kill-on-stderr] [--iters <n>] [-- <program> [args...]]
//! ```
//!
//! Use `RUST_LOG=warn` to see the external mutator's stderr and respawns,
//! `RUST_LOG=debug` for a detailed trace of every mutation roundtrip.
//!
//! With the feature `multipart`, the fuzzer uses a `MultipartInput` (with a `header` and a `payload` part)
//! instead of a `BytesInput`, and passes `--multipart` to the default external mutator.
use std::{path::PathBuf, process, ptr::write, time::Duration};

#[cfg(feature = "tui")]
use libafl::monitors::tui::TuiMonitor;
#[cfg(not(feature = "tui"))]
use libafl::monitors::SimpleMonitor;
use libafl::{
    corpus::{InMemoryCorpus, OnDiskCorpus},
    events::SimpleEventManager,
    executors::{ExitKind, InProcessExecutor},
    feedbacks::{CrashFeedback, MaxMapFeedback},
    fuzzer::{Fuzzer, StdFuzzer},
    generators::RandPrintablesGenerator,
    inputs::{BytesInput, HasTargetBytes},
    mutators::ExternalProcessMutator,
    observers::StdMapObserver,
    schedulers::QueueScheduler,
    stages::mutational::{MutationalStage, StdMutationalStage},
    state::StdState,
};
#[cfg(feature = "multipart")]
use libafl::{generators::Generator, inputs::MultipartInput, state::HasRand, Error};
use libafl_bolts::{current_nanos, nonzero, rands::StdRand, tuples::tuple_list, AsSlice};

/// Coverage map with explicit assignments due to the lack of instrumentation
static mut SIGNALS: [u8; 16] = [0; 16];
// TODO: This will break soon, fix me! See https://github.com/AFLplusplus/LibAFL/issues/2786
#[allow(static_mut_refs)] // only a problem in nightly
static mut SIGNALS_PTR: *mut u8 = unsafe { SIGNALS.as_mut_ptr() };

/// Assign a signal to the signals map
fn signals_set(idx: usize) {
    unsafe { write(SIGNALS_PTR.add(idx), 1) };
}

/// The input type: plain bytes
#[cfg(not(feature = "multipart"))]
type FuzzInput = BytesInput;

/// The bytes the harness looks at
#[cfg(not(feature = "multipart"))]
fn harness_bytes(input: &FuzzInput) -> Vec<u8> {
    input.target_bytes().as_slice().to_vec()
}

/// The input type: multiple parts, identified by `String` keys
#[cfg(feature = "multipart")]
type FuzzInput = MultipartInput<BytesInput, String>;

/// The key of the parts the harness looks at
#[cfg(feature = "multipart")]
const PAYLOAD_KEY: &str = "payload";

/// The bytes the harness looks at: all `payload` parts, concatenated (other parts are ignored)
#[cfg(feature = "multipart")]
fn harness_bytes(input: &FuzzInput) -> Vec<u8> {
    input
        .parts()
        .iter()
        .filter(|(key, _)| key == PAYLOAD_KEY)
        .flat_map(|(_, part)| part.target_bytes().as_slice().to_vec())
        .collect()
}

/// Generates multipart inputs with a random `header` and `payload` part
#[cfg(feature = "multipart")]
struct MultipartGenerator(RandPrintablesGenerator);

#[cfg(feature = "multipart")]
impl<S: HasRand> Generator<FuzzInput, S> for MultipartGenerator {
    fn generate(&mut self, state: &mut S) -> Result<FuzzInput, Error> {
        Ok(MultipartInput::new(vec![
            ("header".into(), self.0.generate(state)?),
            (PAYLOAD_KEY.into(), self.0.generate(state)?),
        ]))
    }
}

/// The arguments for the default external mutator (`python3 mutator.py`)
fn default_mutator_args() -> Vec<String> {
    let mut args = vec![format!("{}/mutator.py", env!("CARGO_MANIFEST_DIR"))];
    if cfg!(feature = "multipart") {
        args.push("--multipart".into());
    }
    args
}

/// Command line options
#[derive(Debug)]
struct Options {
    /// Timeout for a single roundtrip to the external mutator
    timeout: Duration,
    /// Kill and respawn the external mutator whenever it writes to stderr
    kill_on_stderr: bool,
    /// Stop after this many fuzzing iterations (run forever / until a crash if `None`)
    iters: Option<u64>,
    /// The external mutator program
    program: String,
    /// Arguments for the external mutator program
    args: Vec<String>,
}

fn usage(prog: &str) -> ! {
    eprintln!(
        "Usage: {prog} [--timeout-ms <ms>] [--kill-on-stderr] [--iters <n>] [-- <program> [args...]]\n\
         \n\
         Without `-- <program>`, `python3 {}` is used as external mutator.",
        default_mutator_args().join(" ")
    );
    process::exit(1);
}

fn parse_args() -> Options {
    let mut argv = std::env::args();
    let prog = argv
        .next()
        .unwrap_or_else(|| "baby_fuzzer_external_mutator".into());
    let mut options = Options {
        timeout: Duration::from_millis(1000),
        kill_on_stderr: false,
        iters: None,
        program: "python3".into(),
        args: default_mutator_args(),
    };

    let parse_num = |value: Option<String>| -> u64 {
        value
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| usage(&prog))
    };

    while let Some(arg) = argv.next() {
        match arg.as_str() {
            "--timeout-ms" => options.timeout = Duration::from_millis(parse_num(argv.next())),
            "--kill-on-stderr" => options.kill_on_stderr = true,
            "--iters" => options.iters = Some(parse_num(argv.next())),
            "--" => {
                options.program = argv.next().unwrap_or_else(|| usage(&prog));
                options.args = argv.by_ref().collect();
            }
            _ => usage(&prog),
        }
    }
    options
}

#[expect(clippy::manual_assert)]
pub fn main() {
    env_logger::init();
    let options = parse_args();
    println!("External mutator setup: {options:?}");

    // The closure that we want to fuzz
    let mut harness = |input: &FuzzInput| {
        let buf = harness_bytes(input);
        signals_set(0);
        if !buf.is_empty() && buf[0] == b'a' {
            signals_set(1);
            if buf.len() > 1 && buf[1] == b'b' {
                signals_set(2);
                if buf.len() > 2 && buf[2] == b'c' {
                    panic!("Artificial bug triggered =)");
                }
            }
        }
        ExitKind::Ok
    };

    // Create an observation channel using the signals map
    // TODO: This will break soon, fix me! See https://github.com/AFLplusplus/LibAFL/issues/2786
    #[allow(static_mut_refs)] // only a problem in nightly
    let observer = unsafe { StdMapObserver::from_mut_ptr("signals", SIGNALS_PTR, SIGNALS.len()) };

    // Feedback to rate the interestingness of an input
    let mut feedback = MaxMapFeedback::new(&observer);

    // A feedback to choose if an input is a solution or not
    let mut objective = CrashFeedback::new();

    // create a State from scratch
    let mut state = StdState::new(
        // RNG
        StdRand::with_seed(current_nanos()),
        // Corpus that will be evolved, we keep it in memory for performance
        InMemoryCorpus::new(),
        // Corpus in which we store solutions (crashes in this example),
        // on disk so the user can get them after stopping the fuzzer
        OnDiskCorpus::new(PathBuf::from("./crashes")).unwrap(),
        // States of the feedbacks.
        // The feedbacks can report the data that should persist in the State.
        &mut feedback,
        // Same for objective feedbacks
        &mut objective,
    )
    .unwrap();

    // The Monitor trait define how the fuzzer stats are displayed to the user
    #[cfg(not(feature = "tui"))]
    let mon = SimpleMonitor::new(|s| println!("{s}"));
    #[cfg(feature = "tui")]
    let mon = TuiMonitor::builder()
        .title("Baby Fuzzer (external mutator)")
        .enhanced_graphics(false)
        .build();

    // The event manager handle the various events generated during the fuzzing loop
    // such as the notification of the addition of a new item to the corpus
    let mut mgr = SimpleEventManager::new(mon);

    // A queue policy to get testcasess from the corpus
    let scheduler = QueueScheduler::new();

    // A fuzzer with feedbacks and a corpus scheduler
    let mut fuzzer = StdFuzzer::new(scheduler, feedback, objective);

    // Create the executor for an in-process function with just one observer
    let mut executor = InProcessExecutor::new(
        &mut harness,
        tuple_list!(observer),
        &mut fuzzer,
        &mut state,
        &mut mgr,
    )
    .expect("Failed to create the Executor");

    // Generator of printable bytearrays of max size 32 (for each part of multipart inputs)
    #[cfg(not(feature = "multipart"))]
    let mut generator = RandPrintablesGenerator::new(nonzero!(32));
    #[cfg(feature = "multipart")]
    let mut generator = MultipartGenerator(RandPrintablesGenerator::new(nonzero!(32)));

    // Generate 8 initial inputs
    state
        .generate_initial_inputs(&mut fuzzer, &mut executor, &mut generator, &mut mgr, 8)
        .expect("Failed to generate the initial corpus");

    // Setup a mutational stage that delegates all mutations to the external program
    let mutator = ExternalProcessMutator::new(&options.program, &options.args)
        .expect("Failed to spawn the external mutator")
        .with_timeout(options.timeout)
        .with_kill_on_stderr(options.kill_on_stderr);
    let mut stages = tuple_list!(StdMutationalStage::new(mutator));

    if let Some(iters) = options.iters {
        fuzzer
            .fuzz_loop_for(&mut stages, &mut executor, &mut state, &mut mgr, iters)
            .expect("Error in the fuzzing loop");
        let mutator = stages.0.mutator();
        println!(
            "Done after {iters} iterations, the external mutator was spawned {} time(s)",
            mutator.spawn_count()
        );
    } else {
        fuzzer
            .fuzz_loop(&mut stages, &mut executor, &mut state, &mut mgr)
            .expect("Error in the fuzzing loop");
    }
}
