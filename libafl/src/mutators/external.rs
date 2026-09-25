//! The [`ExternalProcessMutator`] delegates mutations to an external, long-running program.
//!
//! The external program is spawned once, with the [`EXTERNAL_MUTATOR_SWITCH`] (i.e., `--mutator`)
//! appended to its arguments, and then talks to the fuzzer through a simple line-based protocol.
//!
//! Each line consists of fields, separated by *exactly one* space (`' '`), and is terminated by `\n`:
//!
//! 1. For each call to [`Mutator::mutate`], the fuzzer writes a request line to the program's `stdin`:
//!    - the first field is the request id, a `u64` in (lowercase) hex, without `0x` prefix,
//!    - the following fields describe the input, encoded as hex (see below).
//! 2. The program answers with exactly one reply line on its `stdout`, in the same format:
//!    - the first field must be the id of the request, otherwise the program is respawned,
//!    - the following fields describe the mutated input.
//!      A reply consisting only of the id means "no mutation" and results in [`MutationResult::Skipped`].
//! 3. Anything written to `stderr` is logged with `warn` severity.
//!    If [`ExternalProcessMutator::kill_on_stderr`] is set, the program is also killed and respawned
//!    (and the mutation is [`MutationResult::Skipped`]).
//!
//! The fields describing the input depend on the input type:
//! - Bytes inputs ([`HasMutatorBytes`], e.g., [`crate::inputs::BytesInput`]): a single field, the bytes.
//!   For example, input `abc` with id `1f` is sent as `1f 616263`.
//! - `MultipartInput<I, K>` (feature `multipart_inputs`): a pair of fields for each part,
//!   the key (formatted with [`core::fmt::Debug`], as hex) and the part's bytes (as hex).
//!   For example, the parts `[("a", "xy"), ("b", "")]` with id `2` are sent as `2 226122 7879 226222 `
//!   (`"a"`, with quotes, is the `Debug` representation of the `String` key `a`; note the empty last field).
//!   Every key in the reply has to be one of the input's keys, but parts may be reordered, removed, or duplicated.
//!
//! An external program is built for one of these input types, there is no negotiation.
//!
//! If the program does not answer within the configured timeout, crashes, or exits,
//! it will be killed (if needed) and respawned, and the mutation is [`MutationResult::Skipped`].
//! The first request to a freshly (re)spawned process gets an additional startup timeout,
//! see [`ExternalProcessMutator::with_startup_timeout`].
//! Programs that answer exactly one request and exit afterwards are supported as well (but slow),
//! they will be transparently respawned before the next mutation.
//!
//! Thanks to the `--mutator` switch, a single binary/script can detect whether it runs as a mutator
//! for the fuzzer, or stand-alone (for example, to test and debug its mutation strategy in isolation).
//! The switch is appended *after* all user-supplied arguments, so it also reaches scripts started via
//! an interpreter (e.g., program `python3` with arguments `["mutator.py"]` runs `python3 mutator.py --mutator`).
//!
//! This mutator is only available on unix, since it relies on `poll`.
use alloc::{
    borrow::Cow,
    format,
    string::{String, ToString},
    vec::Vec,
};
#[cfg(feature = "multipart_inputs")]
use core::fmt::Debug;
use core::time::Duration;
use std::{
    ffi::{OsStr, OsString},
    io::{self, ErrorKind, Read, Write},
    os::fd::AsFd,
    process::{Child, ChildStderr, ChildStdin, ChildStdout, Command, ExitStatus, Stdio},
    sync::mpsc::{self, Receiver, RecvTimeoutError, SendError, Sender, TryRecvError},
    thread,
    time::Instant,
};

use libafl_bolts::{Error, Named};
use nix::{
    errno::Errno,
    poll::{PollFd, PollFlags, PollTimeout, poll},
    sys::signal::{SigSet, SigmaskHow, Signal, pthread_sigmask},
};

use super::{MutationResult, Mutator};
#[cfg(feature = "multipart_inputs")]
use crate::inputs::MultipartInput;
use crate::{
    corpus::CorpusId,
    inputs::{HasMutatorBytes, ResizableMutator},
    state::HasMaxSize,
};

/// The command line switch that is appended to the arguments of the external mutator process,
/// so it can tell that it's running as mutator for the fuzzer.
pub const EXTERNAL_MUTATOR_SWITCH: &str = "--mutator";

/// The default timeout for a single mutation roundtrip to the external process.
pub const DEFAULT_EXTERNAL_MUTATOR_TIMEOUT: Duration = Duration::from_secs(1);

/// The default additional time a freshly spawned process gets to answer its first request
/// (for example, for interpreter startup).
pub const DEFAULT_EXTERNAL_MUTATOR_STARTUP_TIMEOUT: Duration = Duration::from_secs(5);

/// How long we wait for the process to exit on its own (after closing its stdin)
/// when the mutator is dropped, before killing it.
const SHUTDOWN_GRACE_PERIOD: Duration = Duration::from_millis(200);

/// How long we wait for the exit status of a process that closed its stdout.
const EXIT_STATUS_WAIT: Duration = Duration::from_millis(100);

/// Events sent from the pipe reader thread to the mutator
#[derive(Debug)]
enum ProcessEvent {
    /// A line (without the line terminator) was received on `stdout`
    Stdout(Vec<u8>),
    /// A line (without the line terminator) was received on `stderr`
    Stderr(Vec<u8>),
    /// `stdout` got closed (usually, the process exited)
    StdoutClosed,
}

/// The result of waiting for the reply of the external process
#[derive(Debug)]
enum Reply {
    /// We received a reply with the expected id, and (if there was any) the rest of the line
    Payload(Option<Vec<u8>>),
    /// We received a line on stdout without the expected id
    WrongId(Vec<u8>),
    /// The process did not answer in time
    Timeout,
    /// The process exited or closed its stdout
    Exited,
    /// The process wrote to stderr, and we are configured to kill it in this case
    Stderr,
}

/// The things we found when draining all pending events
#[derive(Debug, Default, Clone, Copy)]
struct Pending {
    /// Anything was written to stderr, and [`ExternalProcessMutator::kill_on_stderr`] is set
    kill_for_stderr: bool,
    /// The process exited or closed its stdout
    exited: bool,
}

/// A running instance of the external mutator process, including its I/O threads.
#[derive(Debug)]
struct ExternalProcess {
    child: Child,
    /// Sends requests to the stdin writer thread. Dropping it closes the child's stdin.
    stdin_tx: Option<Sender<Vec<u8>>>,
    /// Receives lines and state changes from the stdout/stderr reader thread.
    events: Receiver<ProcessEvent>,
    /// The number of replies this process instance sent so far
    replies: usize,
    /// If the child has already been reaped
    reaped: bool,
}

impl ExternalProcess {
    /// Kills (if it's still running) and reaps the child. Can safely be called multiple times.
    fn kill(&mut self) {
        // Close stdin first, so the writer thread terminates.
        drop(self.stdin_tx.take());
        if self.reaped {
            return;
        }
        self.reaped = true;
        let pid = self.child.id();
        if let Ok(Some(status)) = self.child.try_wait() {
            log::debug!("External mutator process (pid {pid}) already exited with {status}");
            return;
        }
        log::debug!("Killing external mutator process (pid {pid})");
        if let Err(err) = self.child.kill() {
            log::debug!("Failed to kill external mutator process (pid {pid}): {err}");
        }
        match self.child.wait() {
            Ok(status) => {
                log::debug!("External mutator process (pid {pid}) terminated with {status}");
            }
            Err(err) => log::debug!("Failed to reap external mutator process (pid {pid}): {err}"),
        }
    }

    /// Waits up to `timeout` for the child to exit, and returns its exit status.
    fn wait_timeout(&mut self, timeout: Duration) -> Option<ExitStatus> {
        let deadline = Instant::now() + timeout;
        loop {
            match self.child.try_wait() {
                Ok(Some(status)) => return Some(status),
                Ok(None) if Instant::now() < deadline => thread::sleep(Duration::from_millis(1)),
                _ => return None,
            }
        }
    }

    /// Closes the child's stdin, waits up to `grace` for it to exit on its own, and kills it otherwise.
    fn shutdown(&mut self, grace: Duration) {
        drop(self.stdin_tx.take());
        self.wait_timeout(grace);
        self.kill();
    }
}

impl Drop for ExternalProcess {
    fn drop(&mut self) {
        self.kill();
    }
}

/// Encodes a request line: the `id`, followed by the hex-encoded `fields`, separated by single spaces.
fn encode_request(id: u64, fields: &[&[u8]]) -> Vec<u8> {
    let mut line = format!("{id:x}").into_bytes();
    for field in fields {
        line.push(b' ');
        line.extend_from_slice(hex::encode(field).as_bytes());
    }
    line.push(b'\n');
    line
}

/// Splits a reply line into the id field and the rest of the line (`None` if the line has no other fields).
fn split_id(line: &[u8]) -> (&[u8], Option<&[u8]>) {
    match line.iter().position(|&b| b == b' ') {
        Some(pos) => (&line[..pos], Some(&line[pos + 1..])),
        None => (line, None),
    }
}

/// Parses the id field of a reply (hex, without `0x` prefix)
fn parse_id(field: &[u8]) -> Option<u64> {
    let field = core::str::from_utf8(field).ok()?;
    // `from_str_radix` would also accept a leading `+`
    if field.starts_with('+') {
        return None;
    }
    u64::from_str_radix(field, 16).ok()
}

/// Decodes the hex fields (separated by single spaces) following the id in a reply.
/// Empty fields are valid (empty bytes).
fn decode_fields(payload: Option<&[u8]>) -> Result<Vec<Vec<u8>>, hex::FromHexError> {
    payload.map_or_else(
        || Ok(Vec::new()),
        |payload| payload.split(|&b| b == b' ').map(hex::decode).collect(),
    )
}

/// Replaces the bytes of `input` with `bytes`
fn set_bytes<I>(input: &mut I, bytes: &[u8])
where
    I: HasMutatorBytes + ResizableMutator<u8>,
{
    input.resize(bytes.len(), 0);
    input.mutator_bytes_mut().copy_from_slice(bytes);
}

/// Moves all complete lines (without line terminators) out of `buf`, leaving a trailing partial line.
fn take_lines(buf: &mut Vec<u8>) -> Vec<Vec<u8>> {
    let mut lines = Vec::new();
    while let Some(pos) = buf.iter().position(|&b| b == b'\n') {
        let mut line: Vec<u8> = buf.drain(..=pos).collect();
        while matches!(line.last(), Some(b'\n' | b'\r')) {
            line.pop();
        }
        lines.push(line);
    }
    lines
}

/// Spawns a thread that reads the child's `stdout` and `stderr` and forwards their lines as events.
///
/// Both pipes are handled by a single thread using `poll`, and `stderr` is always drained first.
/// Hence, anything the child wrote to `stderr` *before* writing its reply to `stdout` is guaranteed
/// to be delivered before the reply, which is important for [`ExternalProcessMutator::kill_on_stderr`].
fn spawn_output_reader(
    thread_name: String,
    stdout: ChildStdout,
    stderr: ChildStderr,
    tx: Sender<ProcessEvent>,
) -> io::Result<()> {
    thread::Builder::new()
        .name(thread_name)
        .spawn(move || {
            // An error means that the mutator is gone, nobody is listening anymore.
            let _ = forward_output(stdout, stderr, &tx);
        })
        .map(|_| ())
}

/// The body of the thread spawned by [`spawn_output_reader`].
fn forward_output(
    mut stdout: ChildStdout,
    mut stderr: ChildStderr,
    tx: &Sender<ProcessEvent>,
) -> Result<(), SendError<ProcessEvent>> {
    let mut chunk = vec![0_u8; 64 * 1024];
    let mut stdout_buf = Vec::new();
    let mut stdout_open = true;
    let mut stderr_open = true;

    while stdout_open || stderr_open {
        // Only poll the pipes that are still open (closed ones would report `POLLHUP` forever).
        let mut fds = Vec::with_capacity(2);
        let (mut stderr_idx, mut stdout_idx) = (None, None);
        if stderr_open {
            stderr_idx = Some(fds.len());
            fds.push(PollFd::new(stderr.as_fd(), PollFlags::POLLIN));
        }
        if stdout_open {
            stdout_idx = Some(fds.len());
            fds.push(PollFd::new(stdout.as_fd(), PollFlags::POLLIN));
        }
        match poll(&mut fds, PollTimeout::NONE) {
            Ok(_) => {}
            Err(Errno::EINTR) => continue,
            Err(err) => {
                log::debug!("Failed to poll external mutator pipes: {err}");
                break;
            }
        }
        // Readable, or closed (`read` will return `0` then)
        let ready = |idx: Option<usize>| {
            idx.and_then(|idx| fds[idx].revents())
                .is_some_and(|revents| {
                    revents.intersects(PollFlags::POLLIN | PollFlags::POLLHUP | PollFlags::POLLERR)
                })
        };
        let (stderr_ready, stdout_ready) = (ready(stderr_idx), ready(stdout_idx));
        drop(fds);

        if stderr_ready {
            match stderr.read(&mut chunk) {
                Err(err) if err.kind() == ErrorKind::Interrupted => {}
                Ok(0) | Err(_) => stderr_open = false,
                Ok(len) => {
                    // Forward *everything*, including partial lines, right away.
                    for line in chunk[..len].split(|&b| b == b'\n') {
                        let line = line.strip_suffix(b"\r").unwrap_or(line);
                        if !line.is_empty() {
                            tx.send(ProcessEvent::Stderr(line.to_vec()))?;
                        }
                    }
                }
            }
            // Always check stderr again before looking at stdout.
            continue;
        }

        if stdout_ready {
            match stdout.read(&mut chunk) {
                Err(err) if err.kind() == ErrorKind::Interrupted => {}
                Ok(0) | Err(_) => {
                    stdout_open = false;
                    // A final line without terminator still counts as a line.
                    if !stdout_buf.is_empty() {
                        tx.send(ProcessEvent::Stdout(core::mem::take(&mut stdout_buf)))?;
                    }
                    tx.send(ProcessEvent::StdoutClosed)?;
                }
                Ok(len) => {
                    stdout_buf.extend_from_slice(&chunk[..len]);
                    for line in take_lines(&mut stdout_buf) {
                        tx.send(ProcessEvent::Stdout(line))?;
                    }
                }
            }
        }
    }
    Ok(())
}

/// Spawns a thread that writes all requests it receives to the child's `stdin`.
///
/// Writing happens on a separate thread so that the mutator can never block on a full pipe,
/// which guarantees that the timeout is always honored.
fn spawn_stdin_writer(
    thread_name: String,
    mut stdin: ChildStdin,
    rx: Receiver<Vec<u8>>,
) -> io::Result<()> {
    thread::Builder::new()
        .name(thread_name)
        .spawn(move || {
            // Writing to a process that closed its stdin raises `SIGPIPE` for the writing thread.
            // Block it in this thread, so that the write fails with `EPIPE` instead, independent of
            // the process-wide `SIGPIPE` action (by default, it terminates the fuzzer, and with
            // `LibAFL`'s `handle_sigpipe` feature, it may be reported as a crash of the target).
            let mut sigpipe = SigSet::empty();
            sigpipe.add(Signal::SIGPIPE);
            if let Err(err) = pthread_sigmask(SigmaskHow::SIG_BLOCK, Some(&sigpipe), None) {
                log::warn!("Failed to block SIGPIPE for the external mutator stdin writer: {err}");
            }

            for request in rx {
                if let Err(err) = stdin.write_all(&request).and_then(|()| stdin.flush()) {
                    // The process most likely died (or closed its stdin), the mutator will notice.
                    // Returning also discards the (blocked) pending `SIGPIPE` of this thread.
                    log::debug!("Error writing to external mutator stdin: {err}");
                    return;
                }
            }
            // The sender got dropped: `stdin` is dropped here, closing the pipe.
        })
        .map(|_| ())
}

/// A [`Mutator`] that forwards every mutation to an external program.
///
/// See the [module-level documentation](self) for the details of the protocol.
///
/// # Example
///
/// ```rust,ignore
/// use core::time::Duration;
/// use libafl::mutators::external::ExternalProcessMutator;
///
/// // Runs `python3 ./mutator.py --mutator`
/// let mutator = ExternalProcessMutator::new("python3", ["./mutator.py"])?
///     .with_timeout(Duration::from_millis(500))
///     .with_kill_on_stderr(false);
/// let mut stages = tuple_list!(StdMutationalStage::new(mutator));
/// ```
#[derive(Debug)]
pub struct ExternalProcessMutator {
    name: Cow<'static, str>,
    program: OsString,
    args: Vec<OsString>,
    timeout: Duration,
    startup_timeout: Duration,
    kill_on_stderr: bool,
    process: Option<ExternalProcess>,
    spawn_count: usize,
    /// The id of the next request
    next_id: u64,
}

impl ExternalProcessMutator {
    /// Creates a new [`ExternalProcessMutator`] and spawns the external `program` with the given `args`.
    ///
    /// The program is started with [`EXTERNAL_MUTATOR_SWITCH`] appended to `args`,
    /// uses the [`DEFAULT_EXTERNAL_MUTATOR_TIMEOUT`] and [`DEFAULT_EXTERNAL_MUTATOR_STARTUP_TIMEOUT`],
    /// and ignores (but logs) output on stderr by default.
    ///
    /// Returns an error if the program could not be spawned.
    pub fn new<P, A, IT>(program: P, args: IT) -> Result<Self, Error>
    where
        P: AsRef<OsStr>,
        A: AsRef<OsStr>,
        IT: IntoIterator<Item = A>,
    {
        let mut mutator = Self {
            name: Cow::Borrowed("ExternalProcessMutator"),
            program: program.as_ref().to_os_string(),
            args: args
                .into_iter()
                .map(|arg| arg.as_ref().to_os_string())
                .collect(),
            timeout: DEFAULT_EXTERNAL_MUTATOR_TIMEOUT,
            startup_timeout: DEFAULT_EXTERNAL_MUTATOR_STARTUP_TIMEOUT,
            kill_on_stderr: false,
            process: None,
            spawn_count: 0,
            next_id: 0,
        };
        mutator.spawn()?;
        Ok(mutator)
    }

    /// Sets the timeout for a single mutation roundtrip (sending the input and receiving the result).
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Sets the additional time a freshly (re)spawned process gets to answer its *first* request.
    ///
    /// This avoids endless respawn loops if the program's startup (e.g., of an interpreter)
    /// takes longer than the regular [`Self::timeout`].
    #[must_use]
    pub fn with_startup_timeout(mut self, startup_timeout: Duration) -> Self {
        self.startup_timeout = startup_timeout;
        self
    }

    /// Sets whether the process should be killed and respawned if it writes anything to stderr.
    #[must_use]
    pub fn with_kill_on_stderr(mut self, kill_on_stderr: bool) -> Self {
        self.kill_on_stderr = kill_on_stderr;
        self
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

    /// The timeout for a single mutation roundtrip
    #[must_use]
    pub fn timeout(&self) -> Duration {
        self.timeout
    }

    /// Sets the timeout for a single mutation roundtrip
    pub fn set_timeout(&mut self, timeout: Duration) {
        self.timeout = timeout;
    }

    /// The additional time a freshly (re)spawned process gets to answer its first request
    #[must_use]
    pub fn startup_timeout(&self) -> Duration {
        self.startup_timeout
    }

    /// Sets the additional time a freshly (re)spawned process gets to answer its first request
    pub fn set_startup_timeout(&mut self, startup_timeout: Duration) {
        self.startup_timeout = startup_timeout;
    }

    /// If the process gets killed and respawned whenever it writes anything to stderr.
    /// If `false`, stderr output is only logged.
    #[must_use]
    pub fn kill_on_stderr(&self) -> bool {
        self.kill_on_stderr
    }

    /// Sets if the process gets killed and respawned whenever it writes anything to stderr.
    pub fn set_kill_on_stderr(&mut self, kill_on_stderr: bool) {
        self.kill_on_stderr = kill_on_stderr;
    }

    /// How often the external process has been spawned so far (including the initial spawn).
    #[must_use]
    pub fn spawn_count(&self) -> usize {
        self.spawn_count
    }

    /// The process id of the currently running external process, if any.
    #[must_use]
    pub fn pid(&self) -> Option<u32> {
        self.process.as_ref().map(|process| process.child.id())
    }

    /// Spawns the external process (killing a previous instance, if any).
    fn spawn(&mut self) -> Result<(), Error> {
        self.kill();

        let mut command = Command::new(&self.program);
        command
            .args(&self.args)
            .arg(EXTERNAL_MUTATOR_SWITCH)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        log::debug!(
            "{}: spawning external mutator {:?} with args {:?} + {EXTERNAL_MUTATOR_SWITCH:?}",
            self.name,
            self.program,
            self.args
        );
        let mut child = command.spawn().map_err(|err| {
            Error::os_error(
                err,
                format!(
                    "Failed to spawn external mutator {}",
                    self.program.display()
                ),
            )
        })?;
        let pid = child.id();

        let stdin = child.stdin.take();
        let stdout = child.stdout.take();
        let stderr = child.stderr.take();
        let (event_tx, events) = mpsc::channel();
        let (stdin_tx, stdin_rx) = mpsc::channel();

        // From here on, `process` takes care of killing the child if anything goes wrong.
        let process = ExternalProcess {
            child,
            stdin_tx: Some(stdin_tx),
            events,
            replies: 0,
            reaped: false,
        };

        let (Some(stdin), Some(stdout), Some(stderr)) = (stdin, stdout, stderr) else {
            return Err(Error::illegal_state(
                "Failed to get the stdio pipes of the external mutator",
            ));
        };

        spawn_output_reader(format!("ext-mutator-out-{pid}"), stdout, stderr, event_tx)?;
        spawn_stdin_writer(format!("ext-mutator-stdin-{pid}"), stdin, stdin_rx)?;

        self.process = Some(process);
        self.spawn_count += 1;
        log::debug!(
            "{}: external mutator running with pid {pid} (spawn #{})",
            self.name,
            self.spawn_count
        );
        Ok(())
    }

    /// Kills the running external process, if any.
    fn kill(&mut self) {
        if let Some(mut process) = self.process.take() {
            process.kill();
        }
    }

    /// Kills the running external process and spawns a new one.
    fn respawn(&mut self, reason: &str) -> Result<(), Error> {
        log::warn!(
            "{}: {reason}, respawning external mutator {:?} (pid {:?})",
            self.name,
            self.program,
            self.pid()
        );
        self.spawn()
    }

    /// Logs the exit of the process after it closed its stdout (waiting briefly for its exit status),
    /// followed by `action`. Returns `true` if the process exited cleanly.
    ///
    /// A clean exit (status `0`) is expected for one-shot programs and only logged at debug level,
    /// anything else is considered a crash and logged as warning.
    fn log_exit(&mut self, action: &str) -> bool {
        let status = self
            .process
            .as_mut()
            .and_then(|p| p.wait_timeout(EXIT_STATUS_WAIT));
        let clean = status.is_some_and(|status| status.success());
        let status = status.map_or_else(|| "unknown status".into(), |status| status.to_string());
        let (name, program, pid) = (&self.name, &self.program, self.pid());
        if clean {
            log::debug!(
                "{name}: external mutator (pid {pid:?}) exited cleanly ({status}), {action}"
            );
        } else {
            log::warn!(
                "{name}: external mutator {program:?} (pid {pid:?}) exited ({status}), {action}"
            );
        }
        clean
    }

    /// Respawns the process after it exited (or closed its stdout), see [`Self::log_exit`].
    /// Returns `true` if the process exited cleanly.
    fn respawn_after_exit(&mut self) -> Result<bool, Error> {
        let clean = self.log_exit("respawning");
        self.spawn()?;
        Ok(clean)
    }

    /// Logs a line received on the external process' stderr
    fn log_stderr(&self, line: &[u8]) {
        log::warn!(
            "{}: external mutator (pid {:?}) stderr: {}",
            self.name,
            self.pid(),
            String::from_utf8_lossy(line)
        );
    }

    /// Handles all events that are available right now, without blocking.
    fn drain_pending(&self) -> Pending {
        let mut pending = Pending::default();
        let Some(process) = &self.process else {
            pending.exited = true;
            return pending;
        };
        loop {
            match process.events.try_recv() {
                Ok(ProcessEvent::Stdout(line)) => {
                    log::warn!(
                        "{}: discarding unexpected line on stdout of external mutator (expected exactly one line per input): {:?}",
                        self.name,
                        String::from_utf8_lossy(&line)
                    );
                }
                Ok(ProcessEvent::Stderr(line)) => {
                    self.log_stderr(&line);
                    pending.kill_for_stderr = self.kill_on_stderr;
                }
                Ok(ProcessEvent::StdoutClosed) | Err(TryRecvError::Disconnected) => {
                    pending.exited = true;
                    return pending;
                }
                Err(TryRecvError::Empty) => return pending,
            }
        }
    }

    /// The timeout for the next request, including the startup timeout for fresh processes
    fn effective_timeout(&self) -> Duration {
        match &self.process {
            Some(process) if process.replies == 0 => {
                self.timeout.saturating_add(self.startup_timeout)
            }
            _ => self.timeout,
        }
    }

    /// Sends a request to the external process and waits for its reply (or the `timeout`).
    ///
    /// Checks that the reply carries the request's `id`, and returns the rest of the line.
    fn roundtrip(&mut self, request: &[u8], id: u64, timeout: Duration) -> Reply {
        // For (absurdly) large timeouts, there is no deadline at all.
        let deadline = Instant::now().checked_add(timeout);

        let Some(process) = self.process.as_mut() else {
            return Reply::Exited;
        };
        let sent = process
            .stdin_tx
            .as_ref()
            .is_some_and(|stdin_tx| stdin_tx.send(request.to_vec()).is_ok());
        if !sent {
            log::debug!("{}: the stdin writer thread is gone", self.name);
            return Reply::Exited;
        }

        loop {
            let Some(process) = self.process.as_mut() else {
                return Reply::Exited;
            };
            let event = match deadline {
                Some(deadline) => process
                    .events
                    .recv_timeout(deadline.saturating_duration_since(Instant::now())),
                None => process
                    .events
                    .recv()
                    .map_err(|_| RecvTimeoutError::Disconnected),
            };
            match event {
                Ok(ProcessEvent::Stdout(line)) => {
                    log::trace!("{}: reply: {:?}", self.name, String::from_utf8_lossy(&line));
                    process.replies += 1;
                    let (reply_id, payload) = split_id(&line);
                    return if parse_id(reply_id) == Some(id) {
                        Reply::Payload(payload.map(<[u8]>::to_vec))
                    } else {
                        Reply::WrongId(line)
                    };
                }
                Ok(ProcessEvent::Stderr(line)) => {
                    self.log_stderr(&line);
                    if self.kill_on_stderr {
                        return Reply::Stderr;
                    }
                }
                Ok(ProcessEvent::StdoutClosed) | Err(RecvTimeoutError::Disconnected) => {
                    return Reply::Exited;
                }
                Err(RecvTimeoutError::Timeout) => return Reply::Timeout,
            }
        }
    }

    /// Performs one request/reply exchange with the external process, handling all errors.
    ///
    /// The `fields` are sent hex-encoded (after a fresh request id), and the fields of the reply
    /// are returned decoded. Returns `None` if the mutation has to be skipped (timeout, crash,
    /// stderr output with [`Self::kill_on_stderr`], wrong id, invalid hex, or an unchanged reply).
    fn exchange(&mut self, fields: &[&[u8]]) -> Result<Option<Vec<Vec<u8>>>, Error> {
        // Make sure a healthy process is running, and get rid of leftovers from previous rounds.
        if self.process.is_none() {
            self.spawn()?;
        }
        let pending = self.drain_pending();
        if pending.exited {
            self.respawn_after_exit()?;
        } else if pending.kill_for_stderr {
            self.respawn("external mutator wrote to stderr")?;
        }

        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1);
        let request = encode_request(id, fields);
        log::debug!(
            "{}: sending request {id:x} with {} field(s), {} bytes, to external mutator (pid {:?})",
            self.name,
            fields.len(),
            request.len(),
            self.pid()
        );
        log::trace!(
            "{}: request: {:?}",
            self.name,
            String::from_utf8_lossy(request.trim_ascii_end())
        );

        let start = Instant::now();
        let mut retried = false;
        let payload = loop {
            let timeout = self.effective_timeout();
            let reason = match self.roundtrip(&request, id, timeout) {
                Reply::Payload(payload) => break payload,
                Reply::WrongId(line) => format!(
                    "reply does not match request id {id:x}: {:?}",
                    String::from_utf8_lossy(&line)
                ),
                Reply::Timeout => format!("mutation timed out after {timeout:?}"),
                Reply::Stderr => "external mutator wrote to stderr".into(),
                Reply::Exited => {
                    let replies = self.process.as_ref().map_or(0, |process| process.replies);
                    let clean = self.respawn_after_exit()?;
                    if clean && replies > 0 && !retried {
                        // The process cleanly exited after serving earlier requests (e.g., a one-shot
                        // mutator), most likely before it even read our request. Retry once.
                        log::debug!(
                            "{}: external mutator exited after {replies} replies, retrying with the fresh process",
                            self.name
                        );
                        retried = true;
                        continue;
                    }
                    return Ok(None);
                }
            };
            self.respawn(&reason)?;
            return Ok(None);
        };
        log::debug!(
            "{}: received reply {id:x} ({} bytes) from external mutator after {:?}",
            self.name,
            payload.as_ref().map_or(0, Vec::len),
            start.elapsed()
        );

        // Catch stderr output that arrived together with the reply.
        let pending = self.drain_pending();
        if pending.kill_for_stderr {
            self.respawn("external mutator wrote to stderr")?;
            return Ok(None);
        }
        if pending.exited {
            // We still got a valid reply, just respawn before the next mutation.
            self.log_exit("after replying, will respawn on next mutation");
            self.kill();
        }

        match decode_fields(payload.as_deref()) {
            Ok(reply) if reply.iter().map(Vec::as_slice).eq(fields.iter().copied()) => {
                log::debug!(
                    "{}: external mutator returned the unchanged input",
                    self.name
                );
                Ok(None)
            }
            Ok(reply) => Ok(Some(reply)),
            Err(err) => {
                log::warn!(
                    "{}: external mutator returned invalid hex ({err}) in reply {id:x}: {:?}",
                    self.name,
                    payload.as_deref().map(String::from_utf8_lossy)
                );
                Ok(None)
            }
        }
    }

    /// Checks if the mutated bytes of a (part of an) input exceed the max size.
    fn exceeds_max_size(&self, len: usize, max_size: usize) -> bool {
        if len > max_size {
            log::debug!(
                "{}: external mutator returned {len} bytes, exceeding max size {max_size}, skipping",
                self.name
            );
        }
        len > max_size
    }
}

impl Drop for ExternalProcessMutator {
    fn drop(&mut self) {
        if let Some(mut process) = self.process.take() {
            log::debug!(
                "{}: shutting down external mutator (pid {})",
                self.name,
                process.child.id()
            );
            process.shutdown(SHUTDOWN_GRACE_PERIOD);
        }
    }
}

impl Named for ExternalProcessMutator {
    fn name(&self) -> &Cow<'static, str> {
        &self.name
    }
}

/// Mutates inputs consisting of bytes: the request and reply carry exactly one field (the bytes).
impl<I, S> Mutator<I, S> for ExternalProcessMutator
where
    I: HasMutatorBytes + ResizableMutator<u8>,
    S: HasMaxSize,
{
    fn mutate(&mut self, state: &mut S, input: &mut I) -> Result<MutationResult, Error> {
        let Some(reply) = self.exchange(&[input.mutator_bytes()])? else {
            return Ok(MutationResult::Skipped);
        };
        let mutated = match reply.as_slice() {
            [mutated] if !mutated.is_empty() => mutated,
            [] | [_] => {
                log::debug!(
                    "{}: external mutator returned no bytes, skipping",
                    self.name
                );
                return Ok(MutationResult::Skipped);
            }
            _ => {
                log::warn!(
                    "{}: external mutator returned {} fields, expected one (the mutated bytes)",
                    self.name,
                    reply.len()
                );
                return Ok(MutationResult::Skipped);
            }
        };
        if self.exceeds_max_size(mutated.len(), state.max_size()) {
            return Ok(MutationResult::Skipped);
        }

        log::debug!(
            "{}: applying mutation ({} -> {} bytes)",
            self.name,
            input.mutator_bytes().len(),
            mutated.len()
        );
        set_bytes(input, mutated);
        Ok(MutationResult::Mutated)
    }

    #[inline]
    fn post_exec(&mut self, _state: &mut S, _new_corpus_id: Option<CorpusId>) -> Result<(), Error> {
        Ok(())
    }
}

/// Mutates a [`MultipartInput`]: the request and reply carry a pair of fields for each part,
/// the key (formatted with [`Debug`]) and the part's bytes.
///
/// Since keys can't be created from strings, each key in the reply has to match (the [`Debug`]
/// string of) one of the input's keys. Hence, the external mutator can change the parts' bytes,
/// and reorder, remove, or duplicate parts, but it can't introduce new keys.
#[cfg(feature = "multipart_inputs")]
impl<I, K, S> Mutator<MultipartInput<I, K>, S> for ExternalProcessMutator
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
        let keys: Vec<String> = input
            .parts()
            .iter()
            .map(|(key, _)| format!("{key:?}"))
            .collect();
        let fields: Vec<&[u8]> = input
            .parts()
            .iter()
            .zip(&keys)
            .flat_map(|((_, part), key)| [key.as_bytes(), part.mutator_bytes()])
            .collect();
        let Some(reply) = self.exchange(&fields)? else {
            return Ok(MutationResult::Skipped);
        };
        if reply.is_empty() {
            log::debug!(
                "{}: external mutator returned no parts, skipping",
                self.name
            );
            return Ok(MutationResult::Skipped);
        }
        if reply.len() % 2 != 0 {
            log::warn!(
                "{}: external mutator returned an odd number of fields ({}), expected key/value pairs",
                self.name,
                reply.len()
            );
            return Ok(MutationResult::Skipped);
        }

        let max_size = state.max_size();
        let mut parts = Vec::with_capacity(reply.len() / 2);
        for pair in reply.chunks_exact(2) {
            let (key, bytes) = (&pair[0], &pair[1]);
            let Some(idx) = keys.iter().position(|k| k.as_bytes() == key.as_slice()) else {
                log::warn!(
                    "{}: external mutator returned unknown key {:?}, skipping",
                    self.name,
                    String::from_utf8_lossy(key)
                );
                return Ok(MutationResult::Skipped);
            };
            if self.exceeds_max_size(bytes.len(), max_size) {
                return Ok(MutationResult::Skipped);
            }
            let (key, template) = &input.parts()[idx];
            let mut part = template.clone();
            set_bytes(&mut part, bytes);
            parts.push((key.clone(), part));
        }

        log::debug!(
            "{}: applying mutation ({} -> {} parts)",
            self.name,
            input.len(),
            parts.len()
        );
        *input = MultipartInput::new(parts);
        Ok(MutationResult::Mutated)
    }

    #[inline]
    fn post_exec(&mut self, _state: &mut S, _new_corpus_id: Option<CorpusId>) -> Result<(), Error> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use alloc::{vec, vec::Vec};
    use core::time::Duration;
    use std::time::Instant;

    use super::{ExternalProcessMutator, decode_fields, encode_request, parse_id, split_id};
    use crate::{
        inputs::{BytesInput, HasMutatorBytes},
        mutators::{MutationResult, Mutator},
        state::NopState,
    };

    /// Creates a mutator running the given shell script.
    /// Inside the script, `$0` is `sh`, and `$1` is the appended `--mutator` switch.
    ///
    /// Note: `read id l` splits a request into the id and the rest of the line.
    fn sh(script: &str) -> ExternalProcessMutator {
        ExternalProcessMutator::new("sh", ["-c", script, "sh"])
            .unwrap()
            .with_timeout(Duration::from_secs(5))
    }

    /// Mutates `input` once with `mutator` (and a [`NopState`]).
    fn run<I>(mutator: &mut ExternalProcessMutator, input: &mut I) -> MutationResult
    where
        ExternalProcessMutator: Mutator<I, NopState<I>>,
    {
        mutator.mutate(&mut NopState::new(), input).unwrap()
    }

    /// Runs a single mutation of a bytes input with the given script,
    /// returns the result, the (mutated) input, and the spawn count.
    fn mutate_once(script: &str, input: &[u8]) -> (MutationResult, Vec<u8>, usize) {
        let mut mutator = sh(script);
        let mut input = BytesInput::new(input.to_vec());
        (
            run(&mut mutator, &mut input),
            input.mutator_bytes().to_vec(),
            mutator.spawn_count(),
        )
    }

    #[test]
    fn test_encode_and_parse() {
        assert_eq!(encode_request(0x1f, &[b"abc"]), b"1f 616263\n");
        assert_eq!(
            encode_request(2, &[b"\"a\"", b"xy", b"\"b\"", b""]),
            b"2 226122 7879 226222 \n"
        );
        assert_eq!(encode_request(5, &[]), b"5\n");
        assert_eq!(encode_request(u64::MAX, &[b""]), b"ffffffffffffffff \n");

        assert_eq!(split_id(b"1f 6162"), (&b"1f"[..], Some(&b"6162"[..])));
        assert_eq!(split_id(b"1f "), (&b"1f"[..], Some(&b""[..])));
        assert_eq!(split_id(b"1f"), (&b"1f"[..], None));

        assert_eq!(parse_id(b"1f"), Some(0x1f));
        assert_eq!(parse_id(b"FF"), Some(0xff));
        assert_eq!(parse_id(b"ffffffffffffffff"), Some(u64::MAX));
        assert_eq!(parse_id(b"10000000000000000"), None);
        assert_eq!(parse_id(b"0x1f"), None);
        assert_eq!(parse_id(b"+1f"), None);
        assert_eq!(parse_id(b""), None);

        assert_eq!(decode_fields(None).unwrap(), Vec::<Vec<u8>>::new());
        assert_eq!(decode_fields(Some(b"")).unwrap(), vec![Vec::<u8>::new()]);
        assert_eq!(
            decode_fields(Some(b"61  62")).unwrap(),
            vec![b"a".to_vec(), vec![], b"b".to_vec()]
        );
        assert!(decode_fields(Some(b"6")).is_err());
        assert!(decode_fields(Some(b"zz")).is_err());
    }

    #[test]
    fn test_mutator_switch() {
        // Replies with "ok" as hex if the `--mutator` switch was passed as last (and only) argument
        let (result, bytes, _) = mutate_once(
            r#"read id l; [ "$#" = 1 ] && [ "$1" = --mutator ] && echo "$id 6f6b"; sleep 5"#,
            b"x",
        );
        assert_eq!(result, MutationResult::Mutated);
        assert_eq!(bytes, b"ok");
    }

    #[test]
    fn test_mutated_and_unchanged() {
        // Prepends 'A' to all inputs of length 1, returns everything else as-is.
        let mut mutator = sh(
            r#"while read id l; do if [ ${#l} -eq 2 ]; then echo "$id 41$l"; else echo "$id $l"; fi; done"#,
        );
        let mut input = BytesInput::new(b"b".to_vec());
        assert_eq!(run(&mut mutator, &mut input), MutationResult::Mutated);
        assert_eq!(input.mutator_bytes(), b"Ab");

        assert_eq!(run(&mut mutator, &mut input), MutationResult::Skipped);
        assert_eq!(input.mutator_bytes(), b"Ab");
        assert_eq!(mutator.spawn_count(), 1);
    }

    #[test]
    fn test_request_ids() {
        // Only answers if the ids are consecutive (as hex, starting at 0), otherwise times out.
        let mut mutator = sh(
            r#"n=0; while read id l; do [ "$id" = "$(printf %x $n)" ] && echo "$id 41$l"; n=$((n+1)); done"#,
        );
        let mut input = BytesInput::new(b"b".to_vec());
        for _ in 0..20 {
            assert_eq!(run(&mut mutator, &mut input), MutationResult::Mutated);
        }
        assert_eq!(input.mutator_bytes().len(), 21);
        assert_eq!(mutator.spawn_count(), 1);
    }

    #[test]
    fn test_wrong_id_respawns() {
        let mut mutator = sh(r#"while read id l; do echo "ff$id 41$l"; done"#);
        let mut input = BytesInput::new(b"b".to_vec());
        assert_eq!(run(&mut mutator, &mut input), MutationResult::Skipped);
        assert_eq!(input.mutator_bytes(), b"b");
        assert_eq!(mutator.spawn_count(), 2);

        // A reply without any id respawns, too
        let (result, bytes, spawn_count) =
            mutate_once("while read id l; do echo nothex; done", b"b");
        assert_eq!(
            (result, &bytes[..], spawn_count),
            (MutationResult::Skipped, &b"b"[..], 2)
        );
    }

    #[test]
    fn test_stale_reply_is_never_applied() {
        // Answers every request twice, the second answer must never be used for the next request.
        let mut mutator = sh(r#"while read id l; do echo "$id 41$l"; echo "$id 42$l"; done"#);
        let mut input = BytesInput::new(b"b".to_vec());
        assert_eq!(run(&mut mutator, &mut input), MutationResult::Mutated);
        assert_eq!(input.mutator_bytes(), b"Ab");
        for _ in 0..10 {
            let before = input.mutator_bytes().to_vec();
            match run(&mut mutator, &mut input) {
                // The stale line got discarded before sending the next request
                MutationResult::Mutated => {
                    assert_eq!(input.mutator_bytes(), [b"A", &before[..]].concat());
                }
                // The stale line was received as the reply: wrong id
                MutationResult::Skipped => assert_eq!(input.mutator_bytes(), before),
            }
        }
    }

    #[test]
    fn test_timeout_respawns() {
        let mut mutator = sh("sleep 10")
            .with_timeout(Duration::from_millis(200))
            .with_startup_timeout(Duration::ZERO);
        let mut input = BytesInput::new(b"abc".to_vec());
        let first_pid = mutator.pid();

        let start = Instant::now();
        assert_eq!(run(&mut mutator, &mut input), MutationResult::Skipped);
        assert!(start.elapsed() < Duration::from_secs(5));
        assert_eq!(input.mutator_bytes(), b"abc");
        assert_eq!(mutator.spawn_count(), 2);
        assert_ne!(mutator.pid(), first_pid);
    }

    #[test]
    fn test_startup_timeout() {
        // Slow startup: the first request needs the startup timeout, later ones are fast.
        let mut mutator = sh(r#"sleep 0.3; while read id l; do echo "$id 41$l"; done"#)
            .with_timeout(Duration::from_millis(100))
            .with_startup_timeout(Duration::from_secs(5));
        let mut input = BytesInput::new(b"b".to_vec());
        for _ in 0..3 {
            assert_eq!(run(&mut mutator, &mut input), MutationResult::Mutated);
        }
        assert_eq!(input.mutator_bytes(), b"AAAb");
        assert_eq!(mutator.spawn_count(), 1);
    }

    #[test]
    fn test_graceful_shutdown() {
        let marker = std::env::temp_dir().join(format!(
            "libafl_external_mutator_shutdown_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&marker);
        let script = format!(
            r#"while read l; do echo "$l"; done; touch '{}'"#,
            marker.display()
        );
        drop(sh(&script));
        assert!(marker.exists(), "process did not exit gracefully on EOF");
        std::fs::remove_file(&marker).unwrap();
    }

    #[test]
    fn test_exit_respawns() {
        // One-shot mutator: answers a single request, then exits.
        let mut mutator = sh(r#"read id l; echo "$id ff$l""#);
        let mut input = BytesInput::new(b"a".to_vec());
        for i in 1..=3 {
            assert_eq!(run(&mut mutator, &mut input), MutationResult::Mutated);
            assert_eq!(input.mutator_bytes().len(), 1 + i);
        }
        assert_eq!(mutator.spawn_count(), 3);

        // Crashes (non-zero exit) after answering once: no retry, the mutation is skipped.
        let mut mutator = sh(r#"read id l; echo "$id ff$l"; read id l; exit 3"#);
        let mut input = BytesInput::new(b"a".to_vec());
        assert_eq!(run(&mut mutator, &mut input), MutationResult::Mutated);
        assert_eq!(run(&mut mutator, &mut input), MutationResult::Skipped);
        assert_eq!(mutator.spawn_count(), 2);

        // Exits without answering
        let (result, _, spawn_count) = mutate_once("read id l; exit 3", b"a");
        assert_eq!((result, spawn_count), (MutationResult::Skipped, 2));
    }

    #[test]
    fn test_stderr() {
        let script =
            r#"while read id l; do echo "something went wrong" >&2; echo "$id 41$l"; done"#;
        // stderr gets ignored (only logged)
        let mut mutator = sh(script).with_kill_on_stderr(false);
        let mut input = BytesInput::new(b"b".to_vec());
        assert_eq!(run(&mut mutator, &mut input), MutationResult::Mutated);
        assert_eq!(input.mutator_bytes(), b"Ab");
        assert_eq!(mutator.spawn_count(), 1);

        // stderr kills the process
        let mut mutator = sh(script).with_kill_on_stderr(true);
        let mut input = BytesInput::new(b"b".to_vec());
        assert_eq!(run(&mut mutator, &mut input), MutationResult::Skipped);
        assert_eq!(input.mutator_bytes(), b"b");
        assert_eq!(mutator.spawn_count(), 2);
    }

    #[test]
    fn test_invalid_and_empty_output() {
        let skipped_without_respawn = (MutationResult::Skipped, b"b".to_vec(), 1);
        // invalid hex
        assert_eq!(
            mutate_once(r#"while read id l; do echo "$id nothex"; done"#, b"b"),
            skipped_without_respawn
        );
        // only the id: no mutation
        assert_eq!(
            mutate_once(r#"while read id l; do echo "$id"; done"#, b"b"),
            skipped_without_respawn
        );
        // an empty field: no bytes
        assert_eq!(
            mutate_once(r#"while read id l; do echo "$id "; done"#, b"b"),
            skipped_without_respawn
        );
        // too many fields
        assert_eq!(
            mutate_once(r#"while read id l; do echo "$id 41$l 42"; done"#, b"b"),
            skipped_without_respawn
        );
    }

    /// Tests for [`crate::inputs::MultipartInput`], with `String` keys.
    /// The `Debug` strings of the keys `a` and `b` are `"a"` (hex `226122`) and `"b"` (hex `226222`).
    #[cfg(feature = "multipart_inputs")]
    mod multipart {
        use alloc::{
            string::{String, ToString},
            vec::Vec,
        };

        use super::{run, sh};
        use crate::{
            inputs::{BytesInput, HasMutatorBytes, MultipartInput},
            mutators::MutationResult,
        };

        type Input = MultipartInput<BytesInput, String>;

        fn multipart(parts: &[(&str, &str)]) -> Input {
            MultipartInput::new(
                parts
                    .iter()
                    .map(|(k, v)| ((*k).to_string(), BytesInput::new(v.as_bytes().to_vec())))
                    .collect(),
            )
        }

        fn parts(input: &Input) -> Vec<(String, String)> {
            input
                .parts()
                .iter()
                .map(|(k, v)| {
                    let v = String::from_utf8(v.mutator_bytes().to_vec()).unwrap();
                    (k.clone(), v)
                })
                .collect()
        }

        /// Mutates `[("a", "x"), ("b", "y")]` once with a script that reads `id k1 v1 k2 v2`.
        fn mutate_ab(reply: &str) -> (MutationResult, Vec<(String, String)>, usize) {
            let script = format!("while read id k1 v1 k2 v2; do {reply}; done");
            let mut mutator = sh(&script);
            let mut input = multipart(&[("a", "x"), ("b", "y")]);
            let result = run(&mut mutator, &mut input);
            (result, parts(&input), mutator.spawn_count())
        }

        fn owned(parts: &[(&str, &str)]) -> Vec<(String, String)> {
            parts
                .iter()
                .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
                .collect()
        }

        #[test]
        fn test_request_format_and_mutation() {
            // Only answers if the request has the expected format.
            let (result, parts, _) = mutate_ab(
                r#"[ "$k1 $v1 $k2 $v2" = "226122 78 226222 79" ] && echo "$id $k1 41$v1 $k2 $v2""#,
            );
            assert_eq!(result, MutationResult::Mutated);
            assert_eq!(parts, owned(&[("a", "Ax"), ("b", "y")]));
        }

        #[test]
        fn test_reorder_remove_duplicate() {
            let (result, parts, _) = mutate_ab(r#"echo "$id $k2 $v2 $k1 $v1 $k1 41$v1""#);
            assert_eq!(result, MutationResult::Mutated);
            assert_eq!(parts, owned(&[("b", "y"), ("a", "x"), ("a", "Ax")]));

            let (result, parts, _) = mutate_ab(r#"echo "$id $k2 $v2""#);
            assert_eq!(result, MutationResult::Mutated);
            assert_eq!(parts, owned(&[("b", "y")]));
        }

        #[test]
        fn test_empty_part() {
            // The last field is empty (trailing space)
            let (result, parts, _) = mutate_ab(r#"echo "$id $k1 $v1 $k2 ""#);
            assert_eq!(result, MutationResult::Mutated);
            assert_eq!(parts, owned(&[("a", "x"), ("b", "")]));
        }

        #[test]
        fn test_skipped() {
            let unchanged = owned(&[("a", "x"), ("b", "y")]);
            for reply in [
                r#"echo "$id $k1 $v1 $k2 $v2""#, // unchanged
                r#"echo "$id""#,                 // no mutation
                r#"echo "$id 7a7a $v1""#,        // unknown key
                r#"echo "$id $k1""#,             // odd number of fields
                r#"echo "$id $k1 nothex""#,      // invalid hex
            ] {
                assert_eq!(
                    mutate_ab(reply),
                    (MutationResult::Skipped, unchanged.clone(), 1),
                    "reply: {reply}"
                );
            }
            // Wrong id: respawn
            assert_eq!(
                mutate_ab(r#"echo "1$id $k1 41$v1 $k2 $v2""#),
                (MutationResult::Skipped, unchanged, 2)
            );
        }

        #[test]
        fn test_no_parts() {
            // A request for an input without parts consists of the id only.
            let mut mutator = sh(r#"while read id rest; do [ -z "$rest" ] && echo "$id"; done"#);
            let mut input = multipart(&[]);
            assert_eq!(run(&mut mutator, &mut input), MutationResult::Skipped);
            assert_eq!(mutator.spawn_count(), 1);
        }
    }

    /// Env var that tells [`sigpipe_helper`] that it has been started by [`test_no_sigpipe`]
    const SIGPIPE_HELPER_ENV: &str = "LIBAFL_EXTERNAL_MUTATOR_SIGPIPE_HELPER";

    /// Writing to a mutator that closed its stdin must not raise `SIGPIPE` in the fuzzer.
    ///
    /// Rust binaries ignore `SIGPIPE` by default, but fuzzers with a C `main` don't
    /// (and `LibAFL`'s `handle_sigpipe` feature treats it as a crash).
    /// Hence, we run [`sigpipe_helper`] in a separate process with the default `SIGPIPE` action (terminate).
    #[test]
    fn test_no_sigpipe() {
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "mutators::external::tests::sigpipe_helper",
                "--ignored",
                "--test-threads=1",
            ])
            .env(SIGPIPE_HELPER_ENV, "1")
            .stdout(std::process::Stdio::null())
            .status()
            .unwrap();
        assert!(status.success(), "sigpipe_helper failed: {status}");
    }

    #[test]
    #[ignore = "only a helper for `test_no_sigpipe`"]
    fn sigpipe_helper() {
        use nix::sys::signal::{SigHandler, Signal, signal};

        if std::env::var_os(SIGPIPE_HELPER_ENV).is_none() {
            return;
        }
        // Restore the default action for `SIGPIPE`: terminate the process.
        unsafe { signal(Signal::SIGPIPE, SigHandler::SigDfl) }.unwrap();

        // Closes its stdin *before* answering the first request, then waits.
        let mut mutator = sh(r#"read id l; exec 0<&-; echo "$id 41$l"; exec sleep 10"#)
            .with_timeout(Duration::from_millis(200));
        let mut input = BytesInput::new(b"b".to_vec());
        assert_eq!(run(&mut mutator, &mut input), MutationResult::Mutated);
        // This write hits the closed pipe: EPIPE instead of SIGPIPE, then a timeout.
        assert_eq!(run(&mut mutator, &mut input), MutationResult::Skipped);
        assert_eq!(mutator.spawn_count(), 2);
    }

    #[test]
    fn test_spawn_error() {
        assert!(ExternalProcessMutator::new("/nonexistent/mutator", ["x"]).is_err());
    }
}
