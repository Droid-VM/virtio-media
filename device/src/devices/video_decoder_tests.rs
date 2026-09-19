// Copyright 2026 The DroidVM Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

//! Tests for the stateful [`VideoDecoder`] (`super`).
//!
//! [`FakeDecoderBackend`] turns each queued bitstream buffer of N bytes into one NV12 frame whose
//! luma is a function of the buffer's first byte, on a thread of its own with the session eventfd,
//! the way `FakeCameraBackend` does. The tests replay -- ioctl by ioctl -- the sequences the two
//! guest userland clients issue (`ffmpeg -c:v h264_v4l2m2m`, GStreamer `v4l2h264dec`), plus seek,
//! dynamic resolution change, the §2.5 ordering invariants and the `v4l2-compliance` probes.

use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;
use std::sync::mpsc;
use std::sync::Arc;
use std::sync::Mutex;
use std::thread;
use std::time::Duration;
use std::time::Instant;

use super::*;
use crate::devices::test_pool::PoolBudget;
use crate::ioctl::ffmpeg_wire;
use crate::ioctl::VirtioMediaIoctlHandler;
use crate::MemFdAllocator;

const OUTPUT: QueueType = QueueType::VideoOutputMplane;
const CAPTURE: QueueType = QueueType::VideoCaptureMplane;
const H264: PixelFormat = PixelFormat::from_fourcc(b"H264");
const HEVC: PixelFormat = PixelFormat::from_fourcc(b"HEVC");
const VP90: PixelFormat = PixelFormat::from_fourcc(b"VP90");
const AV01: PixelFormat = PixelFormat::from_fourcc(b"AV01");

// ---------------------------------------------------------------------------------------------
// The VMM-side fakes (event queue, guest memory, host mapper, allocator) -- as in camera::tests.
// ---------------------------------------------------------------------------------------------

#[derive(Default)]
struct EventLog(Rc<RefCell<Vec<V4l2Event>>>);

impl VirtioMediaEventQueue for EventLog {
    fn send_event(&mut self, event: V4l2Event) {
        self.0.borrow_mut().push(event);
    }
}

#[derive(Clone)]
struct FakeGuest {
    memory: Rc<RefCell<Vec<u8>>>,
    live_mappings: Rc<RefCell<usize>>,
    /// So a mapping can assert, as it is released, that the codec thread is not still holding
    /// the buffer it maps (review-m4 R1/R7, F5-crate §2.1).
    log: SharedLog,
}

struct FakeMapping {
    guest: FakeGuest,
    start: usize,
    len: usize,
}

impl GuestMemoryRange for FakeMapping {
    fn as_ptr(&self) -> *const u8 {
        // SAFETY: nothing else resizes `memory` while the test runs.
        unsafe {
            self.guest
                .memory
                .as_ptr()
                .as_ref()
                .unwrap()
                .as_ptr()
                .add(self.start)
        }
    }

    fn as_mut_ptr(&mut self) -> *mut u8 {
        // SAFETY: as above.
        unsafe {
            self.guest
                .memory
                .as_ptr()
                .as_mut()
                .unwrap()
                .as_mut_ptr()
                .add(self.start)
        }
    }

    fn len(&self) -> usize {
        self.len
    }
}

impl Drop for FakeMapping {
    fn drop(&mut self) {
        *self.guest.live_mappings.borrow_mut() -= 1;
        // §2.5: a guest mapping is released only once the backend has stopped writing the buffer
        // behind it -- either because the backend reported the frame, or because it was joined.
        // Skipped while another panic unwinds, so one violation fails one test instead of
        // aborting the binary (F5-crate §2.1).
        if !std::thread::panicking() {
            let ptr = self.as_ptr() as usize;
            assert!(
                !self.guest.log.lock().unwrap().holding.contains(&ptr),
                "a guest mapping was released while the codec thread still held the buffer"
            );
        }
    }
}

impl VirtioMediaGuestMemoryMapper for FakeGuest {
    type GuestMemoryMapping = FakeMapping;

    fn new_mapping(&self, sgs: Vec<SgEntry>) -> anyhow::Result<FakeMapping> {
        let start = sgs.first().map(|sg| sg.start).unwrap_or(0) as usize;
        let total: usize = sgs.iter().map(|sg| sg.len as usize).sum();
        if total == 0 || start + total > self.memory.borrow().len() {
            anyhow::bail!("bad SG list");
        }
        *self.live_mappings.borrow_mut() += 1;
        Ok(FakeMapping {
            guest: self.clone(),
            start,
            len: total,
        })
    }
}

struct FakeHostMapper;

impl VirtioMediaHostMemoryMapper for FakeHostMapper {
    fn add_mapping(&mut self, buffer: &HostBuffer, offset: u64, _rw: bool) -> Result<u64, i32> {
        Ok(buffer.pool_offset.unwrap_or(0x8000_0000 + offset))
    }

    fn remove_mapping(&mut self, _shm_offset: u64) -> Result<(), i32> {
        Ok(())
    }
}

/// Counts what the device gives back, and whether it happened while the backend was still open,
/// over a pool of a size the test chooses.
struct OrderedAllocator {
    inner: MemFdAllocator,
    log: SharedLog,
    /// The `media_host` pool this allocator carves from: unlimited unless a test sizes it (D73).
    pool: Rc<PoolBudget>,
}

impl VirtioMediaBufferAllocator for OrderedAllocator {
    fn allocate(&mut self, len: u64) -> Result<HostBuffer, i32> {
        self.pool.take(len)?;
        match self.inner.allocate(len) {
            Ok(buffer) => Ok(buffer),
            Err(e) => {
                self.pool.give_back(len);
                Err(e)
            }
        }
    }

    fn release(&mut self, buf: HostBuffer) {
        self.pool.give_back(buf.len);
        // §2.5: a buffer must not go back to the allocator while the backend might still be
        // writing the queue's buffers. `capture_active` is true only between a CAPTURE buffer
        // being lent and the backend being joined (clear/stop), so a non-zero count here is a
        // real violation.
        if self.log.lock().unwrap().capture_active {
            *self.log.lock().unwrap().released_while_capture_active.borrow_mut() += 1;
        }
        self.inner.release(buf);
    }
}

// ---------------------------------------------------------------------------------------------
// The fake decoder backend: a codec on a thread of its own.
// ---------------------------------------------------------------------------------------------

/// What the fake decoder did, for the assertions.
#[derive(Default)]
struct FakeLog {
    /// `start` calls, with `(coded_format, coded_size)`.
    started: Vec<(u32, (u32, u32))>,
    /// Whether a backend session is open (thread alive and not stopped).
    open: bool,
    /// Whether a CAPTURE buffer is currently lent to the backend (true from `use_as_capture`
    /// until the backend is joined by `clear_capture_buffers` / `stop`).
    capture_active: bool,
    /// `flush` (seek) calls.
    flushes: usize,
    /// `reinit` (STREAMOFF(OUTPUT) across a pending format change) calls (D53).
    reinits: usize,
    /// `clear_capture_buffers` calls.
    clears: usize,
    /// `stop` calls that completed.
    stops: usize,
    /// Buffers released while `capture_active` -- the §2.5 violation the ordering tests look for.
    released_while_capture_active: RefCell<usize>,
    /// Pointers of the CAPTURE buffers the codec thread holds right now: added by
    /// `use_as_capture` (device thread), removed when the frame is written or when the backend
    /// is joined. A guest mapping released while its pointer is in here is the R1 violation.
    holding: HashSet<usize>,
    /// `resume` calls (`V4L2_DEC_CMD_START`).
    resumes: usize,
    /// The `len` of every OUTPUT buffer lent through `decode`, in order.
    decoded_lens: Vec<usize>,
    /// The next `flush` / `clear_capture_buffers` / `use_as_capture` / `drain` fails with this
    /// errno (once): the wedged-codec case the device's bounds exist for.
    fail_flush: Option<i32>,
    fail_clear: Option<i32>,
    fail_use_capture: Option<i32>,
    fail_drain: Option<i32>,
}

type SharedLog = Arc<Mutex<FakeLog>>;

/// The bitstream's first byte that makes the fake report a mid-stream resolution change from a
/// backend that marks nothing: the new size is announced and no `CAPTURE` buffer is marked. The
/// device then owes the client the `LAST` buffer itself (review-m6 R6-2).
const DRC_MAGIC: u8 = 0xff;
/// The same change, marked as the kernel asks: the frames of the old size, then an empty `LAST`
/// buffer, then the announcement.
const DRC_LAST_MAGIC: u8 = 0xfe;
/// The same change in the order the MediaCodec backend uses (`android.rs` `announce`): the frames
/// of the old size, the announcement, then the empty `LAST` buffer right behind it -- the order
/// the kernel's decoder interface documents, and the one GStreamer and ffmpeg both poll for
/// (`POLLPRI` before `POLLIN`).
const DRC_LAST_AFTER_MAGIC: u8 = 0xfd;
/// A bitstream buffer the backend consumes without being able to announce a format: it produces
/// no `SOURCE_CHANGE` and no frame, and its `InputBufferDone` is **held** until either a later
/// buffer announces a format or the [`FakeBackend::grace`] ceiling passes -- modelling the
/// MediaCodec backend's hold (`android.rs` `note_input_done` / `ANNOUNCE_GRACE`, D55/D56: the
/// first packet of an mp4 our own encoder wrote is a 31-byte header the codec cannot announce
/// from). A one-buffer-in-flight client must still get this buffer back within the grace, or it
/// deadlocks; but it must NOT get it before the `SOURCE_CHANGE` a client that keeps waiting is
/// owed, or that client's OUTPUT queue empties and its poll takes `POLLPRI|POLLERR` (D55).
const NO_ANNOUNCE_MAGIC: u8 = 0xfc;
/// A parameter-set-only buffer fed mid-stream (after the format is announced) that raises a
/// resolution change: the backend holds its `InputBufferDone` until the second `SOURCE_CHANGE`
/// (or the grace), modelling `android.rs` setting `awaiting_drc` on a `CODEC_CONFIG` buffer so a
/// GStreamer client waiting in `wait_for_src_ch` for the change is not emptied (D55, at a DRC).
const DRC_RACE_MAGIC: u8 = 0xfb;
/// The resolution the fake "parses" out of the stream, whatever coded size the client set as a
/// placeholder on `S_FMT(OUTPUT)`. A real decoder reads this from the bitstream.
const FAKE_STREAM_SIZE: (u32, u32) = (320, 240);

/// Luma byte of a frame decoded from a bitstream buffer whose first byte is `b`.
fn luma_of(b: u8) -> u8 {
    0x10u8.wrapping_add(b)
}

struct FakeBackend {
    caps: DecoderCapabilities,
    log: SharedLog,
    /// The `min_capture_buffers` the fake reports in its initial `FormatChanged`, standing in for
    /// the codec's own output-slot count the MediaCodec backend now announces (D69). Default 4.
    announce_min: u32,
    /// `start` fails with this errno.
    fail_start: Option<i32>,
    /// How long a held `InputBufferDone` (a buffer the fake cannot yet announce a format from) is
    /// kept before it is returned anyway, modelling `android.rs`'s `ANNOUNCE_GRACE`. A test that
    /// exercises the grace injects a short one so it need not wait the production 250 ms.
    grace: Duration,
    /// The `CAPTURE` minimum an earlier session "learned" for the coded format, standing in for
    /// the MediaCodec backend's cross-session record (D77). `Some(n)` means the device should
    /// raise a pre-announce `REQBUFS(CAPTURE)` to `n`.
    learned_min: Option<u32>,
}

enum Cmd {
    Start((u32, (u32, u32))),
    Decode {
        index: u32,
        first_byte: u8,
        timestamp: bindings::timeval,
    },
    UseCapture {
        index: u32,
        ptr: SendPtr,
        len: usize,
    },
    Flush(mpsc::Sender<()>),
    ClearCapture(mpsc::Sender<()>),
    Drain,
    Stop(mpsc::Sender<()>),
}

struct FakeSession {
    commands: mpsc::Sender<Cmd>,
    events: mpsc::Receiver<DecoderEvent>,
    thread: Option<thread::JoinHandle<()>>,
    log: SharedLog,
    fail_start: Option<i32>,
    started: bool,
}

/// One frame the fake has decoded and is waiting to write into a CAPTURE buffer.
struct ReadyFrame {
    luma: u8,
    timestamp: bindings::timeval,
}

impl VideoDecoderBackend for FakeBackend {
    type Session = FakeSession;

    fn capabilities(&self) -> &DecoderCapabilities {
        &self.caps
    }

    fn new_session(&mut self, _id: u32, sink: DecoderSink) -> IoctlResult<FakeSession> {
        let (commands, rx) = mpsc::channel::<Cmd>();
        let (events_tx, events) = mpsc::channel::<DecoderEvent>();
        let log = Arc::clone(&self.log);
        let thread_log = Arc::clone(&self.log);
        let grace = self.grace;
        let announce_min = self.announce_min;
        let thread = thread::spawn(move || {
            let emit = |e: DecoderEvent| {
                let _ = events_tx.send(e);
                sink.signal();
            };
            let mut coded_size = (0u32, 0u32);
            let mut ready: std::collections::VecDeque<ReadyFrame> = Default::default();
            let mut captures: std::collections::VecDeque<(u32, SendPtr, usize)> = Default::default();
            let mut draining = false;
            let mut format_announced = false;
            // A resolution change waiting for its `LAST` buffer to go out first.
            let mut pending_format: Option<(u32, u32)> = None;
            // Guest OUTPUT buffer indices whose `InputBufferDone` is held behind a `SOURCE_CHANGE`
            // the client is waiting for (D45/D55), the start of the hold, and whether a
            // mid-stream config buffer started it (`awaiting_drc`). This mirrors `android.rs`'s
            // `note_input_done` / `deferred_input_done` / `ANNOUNCE_GRACE`: a buffer is held while
            // the format is not yet announced or a DRC is expected, and released after the
            // announcement or after `grace` (whichever comes first), never on a bare input-slot
            // recycle.
            let mut deferred: std::collections::VecDeque<u32> = Default::default();
            let mut deferred_since: Option<Instant> = None;
            let mut awaiting_drc = false;
            // The grace has fired once in this pending-format window without an announcement, so no
            // further buffer of it is held (D64, `android.rs`'s `grace_expired`): only the first
            // buffer needs holding for gst's ordering, and a client that keeps feeding must not be
            // re-throttled to one buffer per grace.
            let mut grace_expired = false;
            // Return the guest OUTPUT buffer, or hold it if a `SOURCE_CHANGE` is still owed and the
            // window's grace has not already fired (`grace_expired`, D64).
            let hold = |index: u32,
                        format_announced: bool,
                        awaiting_drc: bool,
                        grace_expired: bool,
                        deferred: &mut std::collections::VecDeque<u32>,
                        deferred_since: &mut Option<Instant>| {
                if (!format_announced || awaiting_drc) && !grace_expired {
                    if deferred_since.is_none() {
                        *deferred_since = Some(Instant::now());
                    }
                    deferred.push_back(index);
                } else {
                    emit(DecoderEvent::InputBufferDone(index));
                }
            };
            // Release every held buffer, after whatever event was just emitted; ends the hold.
            let release_deferred = |deferred: &mut std::collections::VecDeque<u32>,
                                    deferred_since: &mut Option<Instant>,
                                    awaiting_drc: &mut bool| {
                while let Some(index) = deferred.pop_front() {
                    emit(DecoderEvent::InputBufferDone(index));
                }
                *deferred_since = None;
                *awaiting_drc = false;
            };

            // Write a frame into a capture buffer and dequeue it, or -- when draining and no
            // frame is left -- the empty LAST buffer.
            let unhold = |ptr: SendPtr| {
                thread_log.lock().unwrap().holding.remove(&(ptr.as_ptr() as usize));
            };
            let pump = |ready: &mut std::collections::VecDeque<ReadyFrame>,
                            captures: &mut std::collections::VecDeque<(u32, SendPtr, usize)>,
                            coded_size: &mut (u32, u32),
                            draining: &mut bool,
                            pending_format: &mut Option<(u32, u32)>| {
                while !captures.is_empty() {
                    if !ready.is_empty() {
                        let (w, h) = (coded_size.0 as usize, coded_size.1 as usize);
                        let sizeimage = w * h + 2 * (w.div_ceil(2)) * (h.div_ceil(2));
                        // The first lent buffer that can hold a frame of the announced size, as
                        // the MediaCodec backend does (`android.rs` `pump_output`): a smaller one
                        // -- lent for a placeholder size before the SOURCE_CHANGE -- stays lent
                        // and unfilled, and the frame waits with it.
                        let Some(at) = captures.iter().position(|&(_, _, len)| len >= sizeimage)
                        else {
                            break;
                        };
                        let frame = ready.pop_front().expect("not empty");
                        let (index, ptr, _) = captures.remove(at).expect("position was found");
                        let dst = ptr.as_ptr();
                        for row in 0..h {
                            // SAFETY: the buffer was picked for `len >= sizeimage`, which covers
                            // both loops.
                            unsafe { std::ptr::write_bytes(dst.add(row * w), frame.luma, w) };
                        }
                        for row in 0..h.div_ceil(2) {
                            // SAFETY: as above.
                            unsafe { std::ptr::write_bytes(dst.add((h + row) * w), 0x80, w) };
                        }
                        unhold(ptr);
                        emit(DecoderEvent::FrameDecoded {
                            index,
                            bytesused: sizeimage as u32,
                            timestamp: frame.timestamp,
                            is_last: false,
                        });
                    } else if *draining {
                        let (index, ptr, _) = captures.pop_front().expect("not empty");
                        unhold(ptr);
                        *draining = false;
                        emit(DecoderEvent::FrameDecoded {
                            index,
                            bytesused: 0,
                            timestamp: Default::default(),
                            is_last: true,
                        });
                        break;
                    } else {
                        break;
                    }
                }
                // A resolution change held behind the frames of the old size (`Held::Format` in
                // the MediaCodec backend): announced once its `LAST` buffer has gone out.
                if let Some(size) = *pending_format {
                    if !*draining {
                        *pending_format = None;
                        *coded_size = size;
                        emit(DecoderEvent::FormatChanged {
                            coded_size: size,
                            visible_rect: v4l2r::Rect::new(0, 0, size.0, size.1),
                            min_capture_buffers: 4,
                        });
                    }
                }
            };

            loop {
                // While a buffer is held, wait only until its grace deadline: if no command
                // arrives first, the grace fires and the held buffers are returned (D56 -- the
                // release must not depend on a codec event). This is what `GraceTimer` does for the
                // real backend; here the fake's own loop is the timer.
                let cmd = if let Some(since) = deferred_since {
                    let deadline = since + grace;
                    match rx.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
                        Ok(cmd) => cmd,
                        Err(mpsc::RecvTimeoutError::Timeout) => {
                            release_deferred(
                                &mut deferred,
                                &mut deferred_since,
                                &mut awaiting_drc,
                            );
                            // One-shot: the rest of this window is not held (D64).
                            grace_expired = true;
                            continue;
                        }
                        Err(mpsc::RecvTimeoutError::Disconnected) => break,
                    }
                } else {
                    match rx.recv() {
                        Ok(cmd) => cmd,
                        Err(_) => break,
                    }
                };
                match cmd {
                    Cmd::Start((_fmt, size)) => {
                        coded_size = size;
                    }
                    Cmd::Decode {
                        index,
                        first_byte,
                        timestamp,
                    } => {
                        if first_byte == NO_ANNOUNCE_MAGIC {
                            // The codec consumed the buffer but cannot announce a format from it:
                            // no SOURCE_CHANGE, no frame, and its InputBufferDone is HELD, released
                            // only on a later announcement or the grace (D55/D56), never here.
                            hold(
                                index,
                                format_announced,
                                awaiting_drc,
                                grace_expired,
                                &mut deferred,
                                &mut deferred_since,
                            );
                            continue;
                        }
                        if first_byte == DRC_RACE_MAGIC {
                            // A parameter-set-only buffer that raises a mid-stream resolution
                            // change with the D55 race: the input slot recycles before the codec
                            // announces. The backend holds this InputBufferDone until the second
                            // SOURCE_CHANGE, so a GStreamer client in `wait_for_src_ch` is not
                            // emptied. Model it: begin the DRC hold, announce, then release the
                            // held buffer AFTER the event.
                            awaiting_drc = true;
                            // A fresh pending-format window: hold its first buffer again (D64).
                            grace_expired = false;
                            hold(
                                index,
                                format_announced,
                                awaiting_drc,
                                grace_expired,
                                &mut deferred,
                                &mut deferred_since,
                            );
                            let new_size = (
                                (coded_size.0 / 2).max(2) & !1,
                                (coded_size.1 / 2).max(2) & !1,
                            );
                            coded_size = new_size;
                            emit(DecoderEvent::FormatChanged {
                                coded_size,
                                visible_rect: v4l2r::Rect::new(0, 0, new_size.0, new_size.1),
                                min_capture_buffers: 4,
                            });
                            release_deferred(&mut deferred, &mut deferred_since, &mut awaiting_drc);
                            pump(
                                &mut ready,
                                &mut captures,
                                &mut coded_size,
                                &mut draining,
                                &mut pending_format,
                            );
                            continue;
                        }
                        if first_byte == DRC_MAGIC
                            || first_byte == DRC_LAST_MAGIC
                            || first_byte == DRC_LAST_AFTER_MAGIC
                        {
                            // The input is consumed at once (an inline-SPS change, not the config
                            // buffer of DRC_RACE_MAGIC: nothing to hold).
                            emit(DecoderEvent::InputBufferDone(index));
                            // Mid-stream resolution change: halve the size (kept even).
                            let new_size = (
                                (coded_size.0 / 2).max(2) & !1,
                                (coded_size.1 / 2).max(2) & !1,
                            );
                            if first_byte == DRC_LAST_MAGIC {
                                // Every frame of the old size first, then the empty LAST buffer,
                                // then the announcement.
                                pending_format = Some(new_size);
                                draining = true;
                            } else if first_byte == DRC_LAST_AFTER_MAGIC {
                                // The announcement, then the empty LAST buffer from a lent
                                // CAPTURE buffer, if one is lent (the MediaCodec backend's order).
                                coded_size = new_size;
                                emit(DecoderEvent::FormatChanged {
                                    coded_size,
                                    visible_rect: v4l2r::Rect::new(0, 0, new_size.0, new_size.1),
                                    min_capture_buffers: 4,
                                });
                                if let Some((index, ptr, _)) = captures.pop_front() {
                                    unhold(ptr);
                                    emit(DecoderEvent::FrameDecoded {
                                        index,
                                        bytesused: 0,
                                        timestamp: Default::default(),
                                        is_last: true,
                                    });
                                }
                            } else {
                                coded_size = new_size;
                                emit(DecoderEvent::FormatChanged {
                                    coded_size,
                                    visible_rect: v4l2r::Rect::new(0, 0, new_size.0, new_size.1),
                                    min_capture_buffers: 4,
                                });
                            }
                            pump(
                                &mut ready,
                                &mut captures,
                                &mut coded_size,
                                &mut draining,
                                &mut pending_format,
                            );
                            continue;
                        }
                        // The initial format is announced BEFORE the OUTPUT (bitstream) buffer is
                        // returned, so a client polling for `SOURCE_CHANGE` sees it while its
                        // OUTPUT queue is still non-empty: this models the MediaCodec backend
                        // holding `InputBufferDone` until the announcement has been queued (D45,
                        // the `pollrace.py` measurement -- `POLLPRI` before `POLLOUT`). A backend
                        // that returned the buffer first would let the client empty the OUTPUT
                        // queue and take `POLLPRI|POLLERR` on the still-idle CAPTURE side.
                        if !format_announced {
                            format_announced = true;
                            // A real decoder reports the size it parsed, not the placeholder.
                            coded_size = FAKE_STREAM_SIZE;
                            emit(DecoderEvent::FormatChanged {
                                coded_size,
                                visible_rect: v4l2r::Rect::new(0, 0, coded_size.0, coded_size.1),
                                min_capture_buffers: announce_min,
                            });
                            // Any buffers held before the announcement (an earlier NO_ANNOUNCE
                            // one) go back now, AFTER the SOURCE_CHANGE (D45/D55 ordering).
                            release_deferred(&mut deferred, &mut deferred_since, &mut awaiting_drc);
                        }
                        // The input is consumed at once, behind the announcement above.
                        hold(
                            index,
                            format_announced,
                            awaiting_drc,
                            grace_expired,
                            &mut deferred,
                            &mut deferred_since,
                        );
                        ready.push_back(ReadyFrame {
                            luma: luma_of(first_byte),
                            timestamp,
                        });
                        pump(
                            &mut ready,
                            &mut captures,
                            &mut coded_size,
                            &mut draining,
                            &mut pending_format,
                        );
                    }
                    Cmd::UseCapture { index, ptr, len } => {
                        captures.push_back((index, ptr, len));
                        pump(
                            &mut ready,
                            &mut captures,
                            &mut coded_size,
                            &mut draining,
                            &mut pending_format,
                        );
                    }
                    Cmd::Flush(ack) => {
                        // Seek: drop everything queued so far; the CAPTURE queue keeps its buffers
                        // but no pre-seek frame is produced into them. Held input-done is dropped,
                        // not reported late -- the guest takes its OUTPUT buffers back itself
                        // (`android.rs` `flush_codec`).
                        ready.clear();
                        draining = false;
                        pending_format = None;
                        deferred.clear();
                        deferred_since = None;
                        awaiting_drc = false;
                        let _ = ack.send(());
                    }
                    Cmd::ClearCapture(ack) => {
                        // STREAMOFF(CAPTURE): the lent buffers go back, but the frames already
                        // decoded and waiting for a buffer are KEPT and delivered after the CAPTURE
                        // restart -- the MediaCodec backend keeps its held outputs from the last
                        // format change on (`android.rs` `clear_capture_buffers`), so the kernel
                        // loses no frame at a resolution change (review-m6 R6-5). This is what a
                        // reinit relies on: the staged frames survive both STREAMOFF(OUTPUT) (kept
                        // by the reinit, not a seek, D53) and STREAMOFF(CAPTURE). A seek (`Flush`)
                        // is what drops `ready`.
                        captures.clear();
                        draining = false;
                        pending_format = None;
                        let _ = ack.send(());
                    }
                    Cmd::Drain => {
                        draining = true;
                        pump(
                            &mut ready,
                            &mut captures,
                            &mut coded_size,
                            &mut draining,
                            &mut pending_format,
                        );
                    }
                    Cmd::Stop(ack) => {
                        let _ = ack.send(());
                        break;
                    }
                }
            }
            thread_log.lock().unwrap().open = false;
        });
        self.log.lock().unwrap().open = true;
        Ok(FakeSession {
            commands,
            events,
            thread: Some(thread),
            log,
            fail_start: self.fail_start,
            started: false,
        })
    }

    fn close_session(&mut self, mut session: FakeSession) {
        session.stop();
    }

    fn min_capture_buffers(&self, _fourcc: v4l2r::PixelFormat) -> Option<u32> {
        self.learned_min
    }
}

impl Drop for FakeSession {
    fn drop(&mut self) {
        // As the real backend does (`impl Drop for MediaCodecDecoderSession`, crosvm
        // `android_codec_backend/android.rs`): a session that is dropped without a `stop` still
        // joins its codec. This is what makes the device's field order observable.
        self.stop();
    }
}

impl FakeSession {
    fn rendezvous(&self, make: impl FnOnce(mpsc::Sender<()>) -> Cmd) {
        let (tx, rx) = mpsc::channel();
        if self.commands.send(make(tx)).is_ok() {
            let _ = rx.recv();
        }
    }
}

impl VideoDecoderBackendSession for FakeSession {
    fn start(&mut self, coded_format: PixelFormat, coded_size: (u32, u32)) -> IoctlResult<()> {
        if let Some(errno) = self.fail_start {
            return Err(errno);
        }
        self.log
            .lock()
            .unwrap()
            .started
            .push((coded_format.to_u32(), coded_size));
        self.started = true;
        self.commands
            .send(Cmd::Start((coded_format.to_u32(), coded_size)))
            .map_err(|_| libc::EIO)
    }

    fn decode(&mut self, buffer: InputBuffer) -> IoctlResult<()> {
        self.log.lock().unwrap().decoded_lens.push(buffer.len);
        // SAFETY: the device lends a readable pointer valid until we report InputBufferDone.
        let first_byte = if buffer.len > 0 {
            unsafe { *buffer.ptr.as_ptr() }
        } else {
            0
        };
        self.commands
            .send(Cmd::Decode {
                index: buffer.index,
                first_byte,
                timestamp: buffer.timestamp,
            })
            .map_err(|_| libc::EIO)
    }

    fn use_as_capture(&mut self, buffer: OutputBuffer) -> IoctlResult<()> {
        {
            let mut log = self.log.lock().unwrap();
            if let Some(errno) = log.fail_use_capture.take() {
                return Err(errno);
            }
            log.capture_active = true;
            log.holding.insert(buffer.ptr.as_ptr() as usize);
        }
        self.commands
            .send(Cmd::UseCapture {
                index: buffer.index,
                ptr: buffer.ptr,
                len: buffer.len,
            })
            .map_err(|_| libc::EIO)
    }

    fn clear_capture_buffers(&mut self) -> IoctlResult<()> {
        {
            let mut log = self.log.lock().unwrap();
            log.clears += 1;
            // A codec that does not give its buffers back: the real backend has ended its
            // session (`fail`) by the time it answers, so nothing is lent any more either way.
            if let Some(errno) = log.fail_clear.take() {
                log.capture_active = false;
                log.holding.clear();
                return Err(errno);
            }
        }
        // The rendezvous joins the worker w.r.t. CAPTURE buffers; only then is it safe to free
        // them, which is what the `capture_active` flag records.
        self.rendezvous(Cmd::ClearCapture);
        let mut log = self.log.lock().unwrap();
        log.capture_active = false;
        log.holding.clear();
        Ok(())
    }

    fn flush(&mut self) -> IoctlResult<()> {
        {
            let mut log = self.log.lock().unwrap();
            log.flushes += 1;
            if let Some(errno) = log.fail_flush.take() {
                return Err(errno);
            }
        }
        // Nothing is reported for what the flush drops: the inputs were reported the moment
        // they were taken, as the MediaCodec backend does, and the device unqueues the rest
        // itself (the trait's `flush` contract, review-m6 R6-14).
        self.rendezvous(Cmd::Flush);
        Ok(())
    }

    fn reinit(&mut self) -> IoctlResult<()> {
        // A reinit STREAMOFF(OUTPUT) (D53): unlike `flush`, NOTHING is dropped -- no `Cmd::Flush`
        // is sent, so the codec thread keeps its `ready` frames and its state, modelling the
        // MediaCodec backend keeping `pending`/`held_outputs` across the call. The count lets the
        // tests assert a reinit was taken rather than a seek.
        self.log.lock().unwrap().reinits += 1;
        Ok(())
    }

    fn drain(&mut self) -> IoctlResult<()> {
        if let Some(errno) = self.log.lock().unwrap().fail_drain.take() {
            return Err(errno);
        }
        self.commands.send(Cmd::Drain).map_err(|_| libc::EIO)
    }

    fn resume(&mut self) {
        self.log.lock().unwrap().resumes += 1;
    }

    fn stop(&mut self) {
        if let Some(thread) = self.thread.take() {
            self.rendezvous(Cmd::Stop);
            let _ = thread.join();
            let mut log = self.log.lock().unwrap();
            log.stops += 1;
            log.capture_active = false;
            log.holding.clear();
        }
    }

    fn take_events(&mut self) -> Vec<DecoderEvent> {
        self.events.try_iter().collect()
    }
}

// ---------------------------------------------------------------------------------------------
// The rig
// ---------------------------------------------------------------------------------------------

type Device = VideoDecoder<FakeBackend, EventLog, FakeGuest, FakeHostMapper, OrderedAllocator>;
type Session = VideoDecoderSession<FakeMapping, FakeSession>;

struct Rig {
    device: Device,
    events: Rc<RefCell<Vec<V4l2Event>>>,
    guest: FakeGuest,
    log: SharedLog,
    pool: Rc<PoolBudget>,
}

const GUEST_MEMORY: usize = 8 << 20;

/// The profiles and levels the fake backend publishes, as the MediaCodec backend would for
/// `c2.qti.avc.decoder` and `c2.qti.hevc.decoder` on 5566 (`B5-acceptance.md` §"Profiles"), cut
/// down to what makes the control tests readable:
///
/// * **H264** -- Baseline (0), Constrained Baseline (1), Main (2), High (4), Constrained High
///   (17): the store's five, with Extended (3) and every `High 4xx` a hole in between;
/// * **H264 levels** -- 4 (11), 4.1 (12), 5 (14), 5.1 (15), highest first, so the default is 5.1;
/// * **HEVC** -- Main (0) and Main 10 (2), with Main Still Picture (1) the hole, levels 6.2 (12)
///   and 5.1 (8);
/// * **VP9** -- profile 0 only, and **no levels**, the format with a profile control but no level
///   control.
const H264_PROFILES: [i32; 5] = [0, 1, 2, 4, 17];
const H264_LEVELS: [i32; 4] = [15, 14, 12, 11];
const HEVC_PROFILES: [i32; 2] = [0, 2];
const HEVC_LEVELS: [i32; 2] = [12, 8];
const VP9_PROFILES: [i32; 1] = [0];

fn caps() -> DecoderCapabilities {
    let range = SizeRange::new(16, 4096, 2);
    DecoderCapabilities {
        coded_formats: vec![
            CodedFormat {
                fourcc: H264,
                width: range,
                height: range,
                dynamic_resolution: true,
                profiles: H264_PROFILES.to_vec(),
                levels: H264_LEVELS.to_vec(),
            },
            CodedFormat {
                fourcc: HEVC,
                width: range,
                height: range,
                dynamic_resolution: true,
                profiles: HEVC_PROFILES.to_vec(),
                levels: HEVC_LEVELS.to_vec(),
            },
            // A narrower range than the others, as VP9's is on 5566 (B5 §5.1), so a query that
            // answers the wrong format's range is caught.
            CodedFormat {
                fourcc: VP90,
                width: SizeRange::new(16, 2048, 2),
                height: SizeRange::new(16, 2048, 2),
                dynamic_resolution: true,
                profiles: VP9_PROFILES.to_vec(),
                levels: Vec::new(),
            },
        ],
    }
}

/// Capabilities whose formats carry no profile data at all -- a backend that could not ask the
/// host codec store, or a platform whose store has nothing to say. Every coded format is still
/// offered; none of them gets a profile or level control (VA1b).
fn caps_without_profiles() -> DecoderCapabilities {
    let mut caps = caps();
    for f in &mut caps.coded_formats {
        f.profiles.clear();
        f.levels.clear();
    }
    caps
}

/// The default grace the fake holds an unannounceable buffer for. Long enough that a test which
/// announces from a later buffer always releases on the announcement, not the clock; a test that
/// wants the clock injects a short one through [`rig_grace`].
const FAKE_GRACE: Duration = Duration::from_millis(250);

fn rig_with(fail_start: Option<i32>) -> Rig {
    rig_full(fail_start, FAKE_GRACE, 4)
}

fn rig_grace(grace: Duration) -> Rig {
    rig_full(None, grace, 4)
}

/// A rig whose fake backend announces `announce_min` CAPTURE buffers, standing in for a codec
/// whose output-slot count the MediaCodec backend reads and announces (D69).
fn rig_announcing_min(announce_min: u32) -> Rig {
    rig_full(None, FAKE_GRACE, announce_min)
}

/// A rig whose fake backend reports a CAPTURE minimum `learned_min` learned by an earlier session
/// (D77): the device raises this session's pre-announce `REQBUFS(CAPTURE)` to it.
fn rig_with_learned_min(learned_min: u32) -> Rig {
    rig_full_learned(None, FAKE_GRACE, 4, Some(learned_min))
}

fn rig_full(fail_start: Option<i32>, grace: Duration, announce_min: u32) -> Rig {
    rig_full_learned(fail_start, grace, announce_min, None)
}

fn rig_full_learned(
    fail_start: Option<i32>,
    grace: Duration,
    announce_min: u32,
    learned_min: Option<u32>,
) -> Rig {
    rig_caps(caps(), fail_start, grace, announce_min, learned_min)
}

/// A rig whose fake backend publishes `caps`: what the control tests vary, since the device
/// builds its control table from the capabilities at creation.
fn rig_with_caps(caps: DecoderCapabilities) -> Rig {
    rig_caps(caps, None, FAKE_GRACE, 4, None)
}

fn rig_caps(
    caps: DecoderCapabilities,
    fail_start: Option<i32>,
    grace: Duration,
    announce_min: u32,
    learned_min: Option<u32>,
) -> Rig {
    let events = EventLog::default();
    let events_log = Rc::clone(&events.0);
    let log: SharedLog = Default::default();
    let guest = FakeGuest {
        memory: Rc::new(RefCell::new(vec![0u8; GUEST_MEMORY])),
        live_mappings: Rc::new(RefCell::new(0)),
        log: Arc::clone(&log),
    };
    let backend = FakeBackend {
        caps,
        log: Arc::clone(&log),
        announce_min,
        fail_start,
        grace,
        learned_min,
    };
    let pool = PoolBudget::unlimited();
    let device = VideoDecoder::new(
        backend,
        events,
        guest.clone(),
        FakeHostMapper,
        OrderedAllocator {
            inner: MemFdAllocator::new(),
            log: Arc::clone(&log),
            pool: Rc::clone(&pool),
        },
    );
    Rig {
        device,
        events: events_log,
        guest,
        log,
        pool,
    }
}

fn rig() -> Rig {
    rig_with(None)
}

fn session(device: &mut Device) -> Session {
    <Device as VirtioMediaDevice<&[u8], Vec<u8>>>::new_session(device, 0).unwrap()
}

fn new_session(device: &mut Device, id: u32) -> Session {
    <Device as VirtioMediaDevice<&[u8], Vec<u8>>>::new_session(device, id).unwrap()
}

fn close(device: &mut Device, session: Session) {
    <Device as VirtioMediaDevice<&[u8], Vec<u8>>>::close_session(device, session)
}

fn process(device: &mut Device, session: &mut Session) {
    <Device as VirtioMediaDevice<&[u8], Vec<u8>>>::process_events(device, session).unwrap()
}

/// Wait for the session's poll descriptor to become readable, the way the worker does.
fn wait_ready(session: &Session) -> bool {
    let mut pfd = libc::pollfd {
        fd: session.poll_fd().unwrap().as_raw_fd(),
        events: libc::POLLIN,
        revents: 0,
    };
    // SAFETY: one live pollfd.
    unsafe { libc::poll(&mut pfd, 1, 2_000) > 0 }
}

fn dequeued(events: &[V4l2Event]) -> Vec<V4l2Buffer> {
    events
        .iter()
        .filter_map(|e| match e {
            V4l2Event::DequeueBuffer(e) => Some(e.v4l2_buffer().clone()),
            _ => None,
        })
        .collect()
}

fn dequeued_on(events: &[V4l2Event], queue: QueueType) -> Vec<V4l2Buffer> {
    dequeued(events).into_iter().filter(|b| b.queue() == queue).collect()
}

fn source_changes(events: &[V4l2Event]) -> usize {
    events
        .iter()
        .filter(|e| match e {
            V4l2Event::Event(se) => se.event().type_ == bindings::V4L2_EVENT_SOURCE_CHANGE,
            _ => false,
        })
        .count()
}

fn eos_events(events: &[V4l2Event]) -> usize {
    events
        .iter()
        .filter(|e| match e {
            V4l2Event::Event(se) => se.event().type_ == bindings::V4L2_EVENT_EOS,
            _ => false,
        })
        .count()
}

/// The `(id, value)` of every `V4L2_EVENT_CTRL` in the batch, in order (D50).
fn ctrl_events(events: &[V4l2Event]) -> Vec<(u32, i32)> {
    events
        .iter()
        .filter_map(|e| match e {
            V4l2Event::Event(se) if se.event().type_ == bindings::V4L2_EVENT_CTRL => {
                // SAFETY: a CTRL event carries the `ctrl` member.
                let ctrl = unsafe { se.event().u.ctrl };
                Some((se.event().id, unsafe { ctrl.__bindgen_anon_1.value }))
            }
            _ => None,
        })
        .collect()
}

fn errors(events: &[V4l2Event]) -> usize {
    events
        .iter()
        .filter(|e| matches!(e, V4l2Event::Error(_)))
        .count()
}

/// Drive `process_events` until `n` CAPTURE frames have been collected, or time out.
fn collect_capture(r: &mut Rig, s: &mut Session, n: usize) {
    while dequeued_on(&r.events.borrow(), CAPTURE).len() < n {
        assert!(wait_ready(s), "no CAPTURE frame within 2s");
        process(&mut r.device, s);
    }
}

/// Drive `process_events` until a CAPTURE buffer carrying `V4L2_BUF_FLAG_LAST` is dequeued, or
/// time out. A drain delivers ordinary frames before its empty `LAST` buffer, and the grace-model
/// backend thread can split them across `process_events` batches, so a test that wants the `LAST`
/// must collect until it appears rather than assert it is the first CAPTURE dequeue -- doing the
/// latter is what made `a_refused_drain_leaves_no_drain_pending` flaky under load (D63).
fn collect_until_last(r: &mut Rig, s: &mut Session) {
    while !dequeued_on(&r.events.borrow(), CAPTURE)
        .iter()
        .any(|b| b.flags().contains(BufferFlags::LAST))
    {
        assert!(wait_ready(s), "no LAST CAPTURE buffer within 2s");
        process(&mut r.device, s);
    }
}

/// Drive `process_events` for one session until it has produced `n` CAPTURE frames, and return
/// them in order. The rig's event log is device-wide and a `DQBUF` event does not carry its
/// session id, so each batch is sliced off the log around the one `process_events` call that made
/// it -- that call only ever drains the session it is handed, which is what lets a test with two
/// sessions tell whose frames these are.
fn collect_capture_for(r: &mut Rig, s: &mut Session, n: usize) -> Vec<V4l2Buffer> {
    let mut frames = Vec::new();
    while frames.len() < n {
        assert!(wait_ready(s), "no CAPTURE frame within 2s");
        let mark = r.events.borrow().len();
        process(&mut r.device, s);
        frames.extend(dequeued_on(&r.events.borrow()[mark..], CAPTURE));
    }
    frames
}

/// Drain whatever the backend has produced right now (source change, input done) without waiting
/// for a specific count.
fn drain_events(r: &mut Rig, s: &mut Session) {
    if wait_ready(s) {
        process(&mut r.device, s);
    }
}

// buffer builders -----------------------------------------------------------------------------

fn mmap_buffer(queue: QueueType, index: u32, len: u32) -> V4l2Buffer {
    let mut buffer = V4l2Buffer::new(queue, index, MemoryType::Mmap);
    *buffer.get_first_plane_mut().length = len;
    buffer
}

fn userptr_buffer(
    queue: QueueType,
    index: u32,
    gpa: u64,
    len: u32,
    bytesused: u32,
) -> (V4l2Buffer, Vec<Vec<SgEntry>>) {
    let mut buffer = V4l2Buffer::new(queue, index, MemoryType::UserPtr);
    if let V4l2PlanesWithBackingMut::UserPtr(mut planes) = buffer.planes_with_backing_iter_mut() {
        let mut plane = planes.next().unwrap();
        plane.set_userptr(0xc000_0000 + gpa);
        *plane.length = len;
        *plane.bytesused = bytesused;
    }
    (buffer, vec![vec![SgEntry::new(gpa, len)]])
}

/// Write `byte` into the first byte of the OUTPUT buffer `index` (host-owned MMAP buffer).
fn poke_mmap_output(s: &mut Session, index: usize, byte: u8) {
    if let Backing::Host { buffer, .. } = &mut s.input.buffers[index].backing {
        // SAFETY: no guest mapping of this buffer exists; the device is not streaming it yet.
        unsafe { *buffer.as_mut_ptr() = byte };
    } else {
        panic!("not a host-owned OUTPUT buffer");
    }
}

/// The first luma byte of the host-owned CAPTURE buffer `index`: which bitstream buffer the frame
/// sitting in it was decoded from ([`luma_of`]).
fn capture_luma(s: &Session, index: usize) -> u8 {
    match &s.output.buffers[index].backing {
        // SAFETY: the frame has been dequeued, so the backend is no longer writing this buffer.
        Backing::Host { buffer, .. } => unsafe { *buffer.as_ptr() },
        _ => panic!("not a host-owned CAPTURE buffer"),
    }
}

/// The device-wide `MMAP` offset a host-owned buffer was registered at.
fn host_offset(buffer: &Buffer<FakeMapping>) -> u32 {
    match &buffer.backing {
        Backing::Host { offset, .. } => *offset,
        _ => panic!("not a host-owned buffer"),
    }
}

/// A timestamp that carries `n` so a CAPTURE frame can be matched to the OUTPUT buffer it came
/// from (`V4L2_BUF_FLAG_TIMESTAMP_COPY`).
fn ts(n: i64) -> bindings::timeval {
    bindings::timeval {
        tv_sec: n as bindings::time_t,
        tv_usec: 0 as bindings::suseconds_t,
    }
}

/// The fields of a multi-planar format the tests look at, copied out of the packed struct so they
/// can be compared by reference (as `camera::tests` does).
#[derive(Debug, PartialEq, Eq)]
struct Pix {
    width: u32,
    height: u32,
    pixelformat: u32,
    num_planes: u8,
    bytesperline: u32,
    sizeimage: u32,
}

fn pix(format: &v4l2_format) -> Pix {
    // SAFETY: every format these tests see is multi-planar.
    let mp = unsafe { format.fmt.pix_mp };
    Pix {
        width: mp.width,
        height: mp.height,
        pixelformat: mp.pixelformat,
        num_planes: mp.num_planes,
        bytesperline: mp.plane_fmt[0].bytesperline,
        sizeimage: mp.plane_fmt[0].sizeimage,
    }
}

fn output_format(fourcc: PixelFormat, w: u32, h: u32) -> v4l2_format {
    let mut pix = bindings::v4l2_pix_format_mplane {
        width: w,
        height: h,
        pixelformat: fourcc.to_u32(),
        num_planes: 1,
        ..Default::default()
    };
    pix.plane_fmt[0].sizeimage = MIN_BITSTREAM_SIZE;
    v4l2_format {
        type_: OUTPUT as u32,
        fmt: bindings::v4l2_format__bindgen_ty_1 { pix_mp: pix },
    }
}

fn capture_format(w: u32, h: u32) -> v4l2_format {
    capture_format_sized(w, h, 1, 0)
}

fn capture_format_sized(w: u32, h: u32, num_planes: u8, sizeimage: u32) -> v4l2_format {
    let mut pix_mp = bindings::v4l2_pix_format_mplane {
        width: w,
        height: h,
        pixelformat: NV12.to_u32(),
        num_planes,
        ..Default::default()
    };
    // Writing a packed field of a local is allowed; taking a reference to one is not.
    pix_mp.plane_fmt[0].sizeimage = sizeimage;
    v4l2_format {
        type_: CAPTURE as u32,
        fmt: bindings::v4l2_format__bindgen_ty_1 { pix_mp },
    }
}

// ---------------------------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------------------------

/// The V4L2 surface a decoder advertises: coded formats on OUTPUT (from the backend, nothing
/// hard-coded), NV12 on CAPTURE, stepwise frame sizes, no frame rate (ENUM_FRAMEINTERVALS and
/// G_PARM are ENOTTY), the initial coded/visible rectangle.
#[test]
fn formats_are_the_backends_coded_formats_and_nv12() {
    let mut r = rig();
    let s = session(&mut r.device);

    // OUTPUT: the backend's three coded formats, compressed, dynamic-resolution.
    let out: Vec<(u32, u32)> = (0..)
        .map_while(|i| r.device.enum_fmt(&s, OUTPUT, i).ok())
        .map(|d| (d.pixelformat, d.flags))
        .collect();
    assert_eq!(
        out,
        vec![
            (H264.to_u32(), bindings::V4L2_FMT_FLAG_COMPRESSED | bindings::V4L2_FMT_FLAG_DYN_RESOLUTION),
            (HEVC.to_u32(), bindings::V4L2_FMT_FLAG_COMPRESSED | bindings::V4L2_FMT_FLAG_DYN_RESOLUTION),
            (VP90.to_u32(), bindings::V4L2_FMT_FLAG_COMPRESSED | bindings::V4L2_FMT_FLAG_DYN_RESOLUTION),
        ]
    );
    // The kernel's canonical descriptions, which v4l2-compliance checks.
    let h264 = r.device.enum_fmt(&s, OUTPUT, 0).unwrap();
    assert!(h264.description.starts_with(b"H.264\0"));

    // CAPTURE: NV12 only.
    let cap = r.device.enum_fmt(&s, CAPTURE, 0).unwrap();
    assert_eq!(cap.pixelformat, NV12.to_u32());
    assert_eq!(r.device.enum_fmt(&s, CAPTURE, 1).err(), Some(libc::EINVAL));

    // Stepwise frame sizes for a coded format and for NV12.
    let fs = r.device.enum_framesizes(&s, 0, H264.to_u32()).unwrap();
    assert_eq!(fs.type_, bindings::v4l2_frmsizetypes_V4L2_FRMSIZE_TYPE_STEPWISE);
    // SAFETY: stepwise.
    let sw = unsafe { fs.__bindgen_anon_1.stepwise };
    assert_eq!((sw.min_width, sw.max_width, sw.step_width), (16, 4096, 2));
    assert!(r.device.enum_framesizes(&s, 0, NV12.to_u32()).is_ok());
    assert_eq!(r.device.enum_framesizes(&s, 1, H264.to_u32()).err(), Some(libc::EINVAL));
    assert_eq!(
        r.device.enum_framesizes(&s, 0, 0x1234_5678).err(),
        Some(libc::EINVAL)
    );

    // A decoder has no frame rate: G_PARM answers ENOTTY (the trait default), and the driver
    // still forwards it (D6.4).
    assert_eq!(r.device.g_parm(&s, OUTPUT).err(), Some(libc::ENOTTY));
    assert_eq!(r.device.g_parm(&s, CAPTURE).err(), Some(libc::ENOTTY));

    // The CAPTURE format is NV12, tightly packed, at the placeholder coded size.
    let cfmt = pix(&r.device.g_fmt(&s, CAPTURE).unwrap());
    assert_eq!(cfmt.pixelformat, NV12.to_u32());
    assert_eq!(cfmt.num_planes, 1);
    assert_eq!((cfmt.width, cfmt.height), DEFAULT_CODED_SIZE);
    assert_eq!(cfmt.bytesperline, DEFAULT_CODED_SIZE.0);
    assert_eq!(
        cfmt.sizeimage,
        DEFAULT_CODED_SIZE.0 * DEFAULT_CODED_SIZE.1 * 3 / 2
    );

    // The default coded/visible rectangle is non-empty before any SOURCE_CHANGE.
    let bounds = r
        .device
        .g_selection(&s, SelectionType::Capture, SelectionTarget::CropBounds)
        .unwrap();
    assert_eq!((bounds.width, bounds.height), DEFAULT_CODED_SIZE);
    let crop = r
        .device
        .g_selection(&s, SelectionType::Capture, SelectionTarget::Crop)
        .unwrap();
    assert_eq!((crop.width, crop.height), DEFAULT_CODED_SIZE);

    close(&mut r.device, s);
}

/// `S_FMT(OUTPUT)` selects the coded format, sets the CAPTURE placeholder size, and is refused
/// with `EBUSY` once buffers exist; `TRY_FMT(OUTPUT)` snaps an unknown fourcc to the first format.
#[test]
fn s_fmt_output_selects_the_codec_and_is_busy_once_buffers_exist() {
    let mut r = rig();
    let mut s = session(&mut r.device);

    // An unknown fourcc snaps to the first coded format; the coded size is clamped to the range.
    let tried = pix(&r.device.try_fmt(&s, OUTPUT, output_format(PixelFormat::from_fourcc(b"XXXX"), 320, 240)).unwrap());
    assert_eq!(tried.pixelformat, H264.to_u32());
    assert_eq!((tried.width, tried.height), (320, 240));

    let set = pix(&r.device.s_fmt(&mut s, OUTPUT, output_format(VP90, 1920, 1080)).unwrap());
    assert_eq!(set.pixelformat, VP90.to_u32());
    assert_eq!((set.width, set.height), (1920, 1080));
    // The CAPTURE side followed the coded size.
    assert_eq!(pix(&r.device.g_fmt(&s, CAPTURE).unwrap()).width, 1920);

    r.device.reqbufs(&mut s, OUTPUT, MemoryType::Mmap, 2).unwrap();
    assert_eq!(
        r.device.s_fmt(&mut s, OUTPUT, output_format(H264, 640, 480)).err(),
        Some(libc::EBUSY)
    );
    // TRY_FMT still answers while buffers exist.
    assert!(r.device.try_fmt(&s, OUTPUT, output_format(H264, 640, 480)).is_ok());

    close(&mut r.device, s);
}

/// The `ffmpeg -c:v h264_v4l2m2m` sequence, ioctl by ioctl: probe formats, subscribe to
/// SOURCE_CHANGE/EOS, queue OUTPUT and STREAMON(OUTPUT) first, set the CAPTURE queue up from the
/// SOURCE_CHANGE event, decode, then DECODER_CMD(STOP) drain to a LAST buffer and EOS.
///
/// Source: `libavcodec/v4l2_m2m_dec.c` (ffmpeg n7.1.1): `v4l2_prepare_decoder:107-134`
/// (SUBSCRIBE SOURCE_CHANGE then EOS), `v4l2_try_start:37-103` (STREAMON output, then G_FMT /
/// S_SELECTION / G_SELECTION / REQBUFS / STREAMON on capture), `v4l2_receive_frame:136-179` and
/// `v4l2_context.c:v4l2_handle_event:177-228` (DQEVENT -> G_FMT(CAPTURE)), and
/// `ff_v4l2_context_enqueue_packet:602-614` (a zero-size packet -> `V4L2_DEC_CMD_STOP`).
#[test]
fn ffmpeg_v4l2m2m_decode_sequence() {
    let mut r = rig();
    let mut s = session(&mut r.device);

    // ff_v4l2_context_get_format: enumerate OUTPUT coded formats to find H264, TRY_FMT, S_FMT.
    assert_eq!(r.device.enum_fmt(&s, OUTPUT, 0).unwrap().pixelformat, H264.to_u32());
    r.device.try_fmt(&s, OUTPUT, output_format(H264, 0, 0)).unwrap();
    // ffmpeg sizes the OUTPUT (bitstream) buffer itself (v4l2_get_framesize_compressed);
    // `output_format` already sets `sizeimage = MIN_BITSTREAM_SIZE` (1 MiB).
    r.device.s_fmt(&mut s, OUTPUT, output_format(H264, 0, 0)).unwrap();
    // get_raw_format on CAPTURE: ENUM_FMT + TRY_FMT(NV12) + S_FMT.
    assert_eq!(r.device.enum_fmt(&s, CAPTURE, 0).unwrap().pixelformat, NV12.to_u32());
    r.device.try_fmt(&s, CAPTURE, capture_format(0, 0)).unwrap();
    r.device.s_fmt(&mut s, CAPTURE, capture_format(0, 0)).unwrap();

    // ff_v4l2_context_init(output): G_FMT then REQBUFS(OUTPUT) then QUERYBUF.
    r.device.g_fmt(&s, OUTPUT).unwrap();
    r.device.reqbufs(&mut s, OUTPUT, MemoryType::Mmap, 4).unwrap();
    r.device.querybuf(&s, OUTPUT, 0).unwrap();

    // v4l2_prepare_decoder: SUBSCRIBE_EVENT(SOURCE_CHANGE) then EOS.
    r.device
        .subscribe_event(&mut s, EventType::SourceChange(0), SubscribeEventFlags::empty())
        .unwrap();
    r.device
        .subscribe_event(&mut s, EventType::Eos, SubscribeEventFlags::empty())
        .unwrap();

    // First packet: QBUF(OUTPUT) then v4l2_try_start -> STREAMON(OUTPUT).
    poke_mmap_output(&mut s, 0, 0x01);
    let mut ob = mmap_buffer(OUTPUT, 0, 1 << 20);
    ob.set_timestamp(ts(1));
    r.device
        .qbuf(&mut s, ob, vec![], PayloadValidity::ALL)
        .unwrap();
    r.device.streamon(&mut s, OUTPUT).unwrap();
    assert_eq!(r.log.lock().unwrap().started.len(), 1, "codec started at STREAMON(OUTPUT)");

    // The backend parses the stream and raises SOURCE_CHANGE; the input buffer comes back.
    while source_changes(&r.events.borrow()) == 0 {
        assert!(wait_ready(&s), "no SOURCE_CHANGE within 2s");
        process(&mut r.device, &mut s);
    }
    assert_eq!(dequeued_on(&r.events.borrow(), OUTPUT).len(), 1, "OUTPUT buffer returned");

    // v4l2_try_start capture setup, driven off the event: G_FMT(CAPTURE) now shows the stream
    // size, S_SELECTION / G_SELECTION for the crop, then REQBUFS(CAPTURE) + QBUF + STREAMON.
    let cfmt = pix(&r.device.g_fmt(&s, CAPTURE).unwrap());
    assert_eq!((cfmt.width, cfmt.height), (320, 240));
    let sizeimage = cfmt.sizeimage;
    assert_eq!(sizeimage, 320 * 240 * 3 / 2);
    let bounds = r
        .device
        .g_selection(&s, SelectionType::Capture, SelectionTarget::CropBounds)
        .unwrap();
    assert_eq!((bounds.width, bounds.height), (320, 240));

    r.device.reqbufs(&mut s, CAPTURE, MemoryType::Mmap, 4).unwrap();
    for i in 0..4 {
        r.device
            .qbuf(
                &mut s,
                mmap_buffer(CAPTURE, i, sizeimage),
                vec![],
                PayloadValidity::ALL,
            )
            .unwrap();
    }
    r.device.streamon(&mut s, CAPTURE).unwrap();

    // Decode two more OUTPUT buffers; with the buffer 0 frame produced after capture setup, that
    // is three frames, ts 1..3 (the OUTPUT timestamps, copied through TIMESTAMP_COPY).
    for i in 1..3u32 {
        poke_mmap_output(&mut s, i as usize, 0x10 + i as u8);
        let mut ob = mmap_buffer(OUTPUT, i, 1 << 20);
        ob.set_timestamp(ts(i as i64 + 1));
        r.device
            .qbuf(&mut s, ob, vec![], PayloadValidity::ALL)
            .unwrap();
    }
    collect_capture(&mut r, &mut s, 3);
    let frames = dequeued_on(&r.events.borrow(), CAPTURE);
    assert_eq!(frames.len(), 3);
    for (n, f) in frames.iter().enumerate() {
        assert_eq!(f.queue(), CAPTURE);
        assert!(f.flags().contains(BufferFlags::TIMESTAMP_COPY));
        assert_eq!(*f.get_first_plane().bytesused, sizeimage);
        // TIMESTAMP_COPY: the CAPTURE frame carries the OUTPUT buffer's timestamp.
        assert_eq!(f.timestamp().tv_sec, n as i64 + 1);
    }

    // EOF: ffmpeg enqueues a zero-size packet, i.e. DECODER_CMD(STOP), then dequeues until LAST.
    // A CAPTURE buffer is still queued (four were allocated, three consumed), so the drain marks
    // it as the empty LAST buffer.
    r.device.decoder_cmd(&mut s, dec_cmd(bindings::V4L2_DEC_CMD_STOP)).unwrap();
    while !dequeued_on(&r.events.borrow(), CAPTURE).iter().any(|b| b.flags().contains(BufferFlags::LAST)) {
        assert!(wait_ready(&s), "no LAST buffer within 2s");
        process(&mut r.device, &mut s);
    }
    let last = dequeued_on(&r.events.borrow(), CAPTURE);
    let last = last.last().unwrap();
    assert!(last.flags().contains(BufferFlags::LAST));
    assert_eq!(*last.get_first_plane().bytesused, 0, "the LAST drain buffer is empty");
    assert_eq!(eos_events(&r.events.borrow()), 1, "V4L2_EVENT_EOS after the LAST buffer");

    r.device.streamoff(&mut s, OUTPUT).unwrap();
    r.device.streamoff(&mut s, CAPTURE).unwrap();
    close(&mut r.device, s);
    assert_eq!(r.log.lock().unwrap().stops, 1);
    assert_eq!(*r.log.lock().unwrap().released_while_capture_active.borrow(), 0);
}

/// The GStreamer `v4l2h264dec` sequence: probe (ENUM_FMT/ENUM_FRAMESIZES, ENUM_FRAMEINTERVALS
/// ENOTTY tolerated), subscribe SOURCE_CHANGE, S_FMT(OUTPUT) + QBUF + STREAMON(OUTPUT), wait for
/// the source change, then decide_allocation (G_CTRL(MIN_BUFFERS_FOR_CAPTURE)) and capture setup;
/// EXPBUF is refused so gst falls back to MMAP.
///
/// Source: `sys/v4l2/gstv4l2videodec.c` (GStreamer 1.28.2) `gst_v4l2_video_dec_open:120-160`
/// (probe caps, subscribe SOURCE_CHANGE), `gst_v4l2_video_dec_handle_frame:954` (S_FMT output,
/// pool start = REQBUFS, QBUF, STREAMON), `gst_v4l2_video_dec_loop:762` /
/// `wait_for_src_ch:735-760` (poll for SOURCE_CHANGE), `gstv4l2object.c:decide_allocation:5799`
/// via `gst_v4l2_get_driver_min_buffers:940-957` (G_CTRL MIN_BUFFERS_FOR_CAPTURE) and
/// `is_dmabuf_supported:3433-3455` (EXPBUF -> ENOTTY disables DMABuf).
#[test]
fn gstreamer_v4l2videodec_sequence() {
    let mut r = rig();
    let mut s = session(&mut r.device);

    // probe_caps: ENUM_FMT(OUTPUT) + ENUM_FRAMESIZES; ENUM_FRAMEINTERVALS is ENOTTY (a decoder
    // has no frame rate), which gst tolerates.
    assert!(r.device.enum_fmt(&s, OUTPUT, 0).is_ok());
    assert!(r.device.enum_framesizes(&s, 0, H264.to_u32()).is_ok());
    assert_eq!(
        r.device.enum_frameintervals(&s, 0, H264.to_u32(), 320, 240).err(),
        Some(libc::ENOTTY)
    );
    // subscribe SOURCE_CHANGE on the capture side.
    r.device
        .subscribe_event(&mut s, EventType::SourceChange(0), SubscribeEventFlags::empty())
        .unwrap();

    // set_format + handle_frame: S_FMT(OUTPUT), REQBUFS(OUTPUT), QBUF, STREAMON(OUTPUT).
    r.device.s_fmt(&mut s, OUTPUT, output_format(H264, 320, 240)).unwrap();
    r.device.reqbufs(&mut s, OUTPUT, MemoryType::Mmap, 2).unwrap();
    poke_mmap_output(&mut s, 0, 0x22);
    let mut ob = mmap_buffer(OUTPUT, 0, 1 << 20);
    ob.set_timestamp(ts(7));
    r.device
        .qbuf(&mut s, ob, vec![], PayloadValidity::ALL)
        .unwrap();
    r.device.streamon(&mut s, OUTPUT).unwrap();

    // wait_for_src_ch: poll capture for the SOURCE_CHANGE.
    while source_changes(&r.events.borrow()) == 0 {
        assert!(wait_ready(&s), "no SOURCE_CHANGE within 2s");
        process(&mut r.device, &mut s);
    }

    // decide_allocation: G_CTRL(MIN_BUFFERS_FOR_CAPTURE) gives the pool its size.
    let min = r
        .device
        .g_ctrl(&s, bindings::V4L2_CID_MIN_BUFFERS_FOR_CAPTURE)
        .unwrap();
    assert_eq!(min.value, 4);
    // A control gst does not use is refused, not answered.
    assert_eq!(r.device.g_ctrl(&s, 0x0098_0001).err(), Some(libc::EINVAL));

    // negotiate: G_FMT(CAPTURE) + G_SELECTION(COMPOSE) for the visible size.
    let sizeimage = pix(&r.device.g_fmt(&s, CAPTURE).unwrap()).sizeimage;
    let compose = r
        .device
        .g_selection(&s, SelectionType::Capture, SelectionTarget::Compose)
        .unwrap();
    assert_eq!((compose.width, compose.height), (320, 240));

    // capture pool: REQBUFS(CAPTURE, min), QBUF, STREAMON.
    r.device.reqbufs(&mut s, CAPTURE, MemoryType::Mmap, min.value as u32).unwrap();
    for i in 0..min.value as u32 {
        r.device
            .qbuf(
                &mut s,
                mmap_buffer(CAPTURE, i, sizeimage),
                vec![],
                PayloadValidity::ALL,
            )
            .unwrap();
    }
    r.device.streamon(&mut s, CAPTURE).unwrap();

    // A frame comes out.
    collect_capture(&mut r, &mut s, 1);
    let frame = dequeued_on(&r.events.borrow(), CAPTURE).remove(0);
    assert_eq!(frame.timestamp().tv_sec, 7);

    close(&mut r.device, s);
}

/// D27b: after a `V4L2_DEC_CMD_STOP` drain reaches its empty `LAST` buffer the decoder is
/// `Stopped` and "will accept, but not process, any newly queued OUTPUT buffers until the client
/// issues" a resume (`dev-decoder.rst`, "Drain" step 3). `V4L2_DEC_CMD_START` is the resume: it
/// clears the stop, tells the backend (`resume`), and a new stream decodes on the same codec.
///
/// This is the sequence a well-behaved looping client uses. ffmpeg's `-stream_loop` does **not**:
/// its `v4l2m2m` decoder has no `flush` callback, so after the first EOF `s->draining` stays set
/// and it never issues `START` again (`v4l2_m2m_dec.c` `v4l2_receive_frame`: `if (s->draining)
/// goto dequeue;`). That stall (B8 §3.2) is an ffmpeg limitation, not a device one -- the device
/// resumes exactly as the kernel says, which this test pins.
#[test]
fn a_drained_decoder_resumes_on_dec_cmd_start() {
    let mut r = rig();
    let mut s = session(&mut r.device);
    let sizeimage = start_streaming_320x240(&mut r, &mut s);

    // Decode one frame (ts 1), then drain to the empty LAST buffer.
    poke_mmap_output(&mut s, 0, 0x21);
    let mut ob = mmap_buffer(OUTPUT, 0, 1 << 20);
    ob.set_timestamp(ts(1));
    r.device.qbuf(&mut s, ob, vec![], PayloadValidity::ALL).unwrap();
    collect_capture(&mut r, &mut s, 1);
    r.device.decoder_cmd(&mut s, dec_cmd(bindings::V4L2_DEC_CMD_STOP)).unwrap();
    while !dequeued_on(&r.events.borrow(), CAPTURE).iter().any(|b| b.flags().contains(BufferFlags::LAST)) {
        assert!(wait_ready(&s), "no LAST buffer within 2s");
        process(&mut r.device, &mut s);
    }
    assert_eq!(s.drain, Drain::Stopped, "the decoder is stopped after the drain");
    assert_eq!(eos_events(&r.events.borrow()), 1);

    // A new OUTPUT buffer queued now must NOT decode into a CAPTURE buffer: the decoder is
    // stopped (accept, but do not process).
    poke_mmap_output(&mut s, 1, 0x22);
    let mut ob = mmap_buffer(OUTPUT, 1, 1 << 20);
    ob.set_timestamp(ts(2));
    r.device.qbuf(&mut s, ob, vec![], PayloadValidity::ALL).unwrap();
    // Requeue a CAPTURE buffer too; while stopped it is not lent.
    r.device.qbuf(&mut s, mmap_buffer(CAPTURE, 0, sizeimage), vec![], PayloadValidity::ALL).unwrap();
    let non_last_before = dequeued_on(&r.events.borrow(), CAPTURE)
        .iter()
        .filter(|b| !b.flags().contains(BufferFlags::LAST))
        .count();
    drain_events(&mut r, &mut s);
    let non_last_stopped = dequeued_on(&r.events.borrow(), CAPTURE)
        .iter()
        .filter(|b| !b.flags().contains(BufferFlags::LAST))
        .count();
    assert_eq!(non_last_stopped, non_last_before, "a stopped decoder processes no new frame");

    // The resume: V4L2_DEC_CMD_START clears the stop and tells the backend.
    let resumes_before = r.log.lock().unwrap().resumes;
    r.device.decoder_cmd(&mut s, dec_cmd(bindings::V4L2_DEC_CMD_START)).unwrap();
    assert_eq!(s.drain, Drain::None, "START resumes the decoder");
    assert_eq!(r.log.lock().unwrap().resumes, resumes_before + 1, "the backend heard the resume");
    assert_eq!(r.log.lock().unwrap().started.len(), 1, "the same codec, not a new one");

    // The new stream now decodes: a frame carrying the post-drain OUTPUT timestamp comes out.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    loop {
        if dequeued_on(&r.events.borrow(), CAPTURE)
            .iter()
            .any(|b| !b.flags().contains(BufferFlags::LAST) && b.timestamp().tv_sec == 2)
        {
            break;
        }
        assert!(std::time::Instant::now() < deadline, "no post-resume frame within 2s");
        if wait_ready(&s) {
            process(&mut r.device, &mut s);
        }
    }
    close(&mut r.device, s);
}

/// Seek (`STREAMOFF(OUTPUT)` while CAPTURE keeps streaming): the backend flushes and the codec
/// stays; the OUTPUT buffers come back, the CAPTURE queue is untouched, and `STREAMON(OUTPUT)`
/// resumes without recreating the codec. Kernel decoder interface, "Seek".
#[test]
fn seek_streamoff_output_flushes_and_keeps_the_codec() {
    let mut r = rig();
    let mut s = session(&mut r.device);
    let sizeimage = start_streaming_320x240(&mut r, &mut s);

    // Decode one frame so we know the codec is running.
    poke_mmap_output(&mut s, 0, 0x05);
    r.device
        .qbuf(
            &mut s,
            mmap_buffer(OUTPUT, 0, 1 << 20),
            vec![],
            PayloadValidity::ALL,
        )
        .unwrap();
    collect_capture(&mut r, &mut s, 1);

    // Seek: STREAMOFF(OUTPUT). The backend flushes; the codec is not torn down.
    r.device.streamoff(&mut s, OUTPUT).unwrap();
    assert_eq!(r.log.lock().unwrap().flushes, 1);
    assert_eq!(r.log.lock().unwrap().stops, 0, "seek does not stop the codec");
    assert!(r.log.lock().unwrap().open, "the codec is still open after a seek");
    assert!(s.input.buffers.iter().all(|b| !b.queued), "OUTPUT buffers returned by the seek");
    assert!(s.state.capture_streaming, "CAPTURE keeps streaming across a seek");

    // STREAMON(OUTPUT) again: no second codec, and new post-seek data decodes.
    r.device.streamon(&mut s, OUTPUT).unwrap();
    assert_eq!(r.log.lock().unwrap().started.len(), 1, "the codec is not recreated");
    poke_mmap_output(&mut s, 1, 0x06);
    r.device
        .qbuf(
            &mut s,
            mmap_buffer(OUTPUT, 1, 1 << 20),
            vec![],
            PayloadValidity::ALL,
        )
        .unwrap();
    r.device
        .qbuf(
            &mut s,
            mmap_buffer(CAPTURE, 0, sizeimage),
            vec![],
            PayloadValidity::ALL,
        )
        .unwrap();
    collect_capture(&mut r, &mut s, 2);

    close(&mut r.device, s);
}

/// Dynamic resolution change: a magic bitstream buffer makes the backend raise a second
/// SOURCE_CHANGE at a new coded size; the client tears the CAPTURE queue down
/// (STREAMOFF/REQBUFS(0)) while OUTPUT keeps streaming, re-reads G_FMT, reallocates and restarts.
/// Kernel decoder interface, "Dynamic Resolution Change".
#[test]
fn dynamic_resolution_change_reconfigures_capture() {
    let mut r = rig();
    let mut s = session(&mut r.device);
    let sizeimage = start_streaming_320x240(&mut r, &mut s);
    assert_eq!(sizeimage, 320 * 240 * 3 / 2);

    // A frame at the first resolution.
    poke_mmap_output(&mut s, 0, 0x08);
    r.device
        .qbuf(
            &mut s,
            mmap_buffer(OUTPUT, 0, 1 << 20),
            vec![],
            PayloadValidity::ALL,
        )
        .unwrap();
    collect_capture(&mut r, &mut s, 1);
    let before = source_changes(&r.events.borrow());

    // A magic OUTPUT buffer triggers a mid-stream resolution change to 160x120.
    poke_mmap_output(&mut s, 1, DRC_MAGIC);
    r.device
        .qbuf(
            &mut s,
            mmap_buffer(OUTPUT, 1, 1 << 20),
            vec![],
            PayloadValidity::ALL,
        )
        .unwrap();
    while source_changes(&r.events.borrow()) == before {
        assert!(wait_ready(&s), "no second SOURCE_CHANGE within 2s");
        process(&mut r.device, &mut s);
    }
    // Queries now return the new resolution.
    assert_eq!(pix(&r.device.g_fmt(&s, CAPTURE).unwrap()).width, 160);
    let new_sizeimage = pix(&r.device.g_fmt(&s, CAPTURE).unwrap()).sizeimage;
    assert_eq!(new_sizeimage, 160 * 120 * 3 / 2);

    // The DRC reconfiguration, OUTPUT still streaming: STREAMOFF(CAPTURE), REQBUFS(0), G_FMT,
    // REQBUFS, QBUF, STREAMON(CAPTURE).
    r.device.streamoff(&mut s, CAPTURE).unwrap();
    assert_eq!(r.log.lock().unwrap().clears, 1);
    r.device.reqbufs(&mut s, CAPTURE, MemoryType::Mmap, 0).unwrap();
    assert!(s.output.buffers.is_empty());
    r.device.reqbufs(&mut s, CAPTURE, MemoryType::Mmap, 4).unwrap();
    for i in 0..4 {
        r.device
            .qbuf(
                &mut s,
                mmap_buffer(CAPTURE, i, new_sizeimage),
                vec![],
                PayloadValidity::ALL,
            )
            .unwrap();
    }
    r.device.streamon(&mut s, CAPTURE).unwrap();

    // A frame at the new resolution.
    poke_mmap_output(&mut s, 2, 0x09);
    r.device
        .qbuf(
            &mut s,
            mmap_buffer(OUTPUT, 2, 1 << 20),
            vec![],
            PayloadValidity::ALL,
        )
        .unwrap();
    let already = dequeued_on(&r.events.borrow(), CAPTURE).len();
    while dequeued_on(&r.events.borrow(), CAPTURE).len() <= already {
        assert!(wait_ready(&s), "no post-DRC frame within 2s");
        process(&mut r.device, &mut s);
    }
    let frame = dequeued_on(&r.events.borrow(), CAPTURE).pop().unwrap();
    assert_eq!(*frame.get_first_plane().bytesused, new_sizeimage);
    // The sequence counter restarted with the CAPTURE queue (V4L2 counts frames since
    // `STREAMON`, review-m6 R6-16): the frame before the change was number 0 of the old stream.
    assert_eq!(frame.sequence(), 0);

    // D53, test (d): this is GStreamer's spec-conformant reinit -- it reconfigures CAPTURE
    // (STREAMOFF(CAPTURE)/REQBUFS/STREAMON) and never stops the OUTPUT queue -- so neither the seek
    // nor the reinit STREAMOFF(OUTPUT) path is ever taken. The rule leaves it entirely unchanged.
    assert_eq!(r.log.lock().unwrap().flushes, 0, "gst reinit takes no seek");
    assert_eq!(r.log.lock().unwrap().reinits, 0, "gst reinit never stops OUTPUT");

    close(&mut r.device, s);
}

/// D53, test (a): a `STREAMOFF(OUTPUT)` issued to reconfigure CAPTURE across the *initial*
/// `SOURCE_CHANGE` -- ffmpeg's `ff_v4l2_m2m_codec_reinit` path -- must keep the staged bitstream,
/// not drop it as a seek. The device stages each queued bitstream buffer and returns the OUTPUT
/// buffer at once (D48/D28), so the guest believes the packets consumed; a client that then
/// reinits (STREAMOFF(OUTPUT), CAPTURE STREAMOFF/REQBUFS(0)/G_FMT/REQBUFS/STREAMON,
/// STREAMON(OUTPUT)) must get every frame back. The kernel's `dev-decoder.rst` ("Dynamic
/// resolution change") says the client should not stop OUTPUT during a resolution change, so a
/// `STREAMOFF(OUTPUT)` while a format change is pending is a reinit, not a seek: no `flush` is
/// taken, and no "seek #1 ... dropped" is logged by the real backend.
#[test]
fn reinit_across_the_initial_source_change_keeps_staged_input() {
    let mut r = rig();
    let mut s = session(&mut r.device);

    r.device.s_fmt(&mut s, OUTPUT, output_format(H264, 320, 240)).unwrap();
    r.device.reqbufs(&mut s, OUTPUT, MemoryType::Mmap, 4).unwrap();
    r.device
        .subscribe_event(&mut s, EventType::SourceChange(0), SubscribeEventFlags::empty())
        .unwrap();

    // Four bitstream buffers queued before any CAPTURE buffer exists: the backend stages them,
    // announces the format from the first, returns all four (InputBufferDone) so the guest
    // believes them consumed, and holds the four frames until CAPTURE is set up.
    for i in 0..4u32 {
        poke_mmap_output(&mut s, i as usize, 0x10 + i as u8);
        let mut ob = mmap_buffer(OUTPUT, i, 1 << 20);
        ob.set_timestamp(ts(i as i64 + 1));
        r.device.qbuf(&mut s, ob, vec![], PayloadValidity::ALL).unwrap();
    }
    r.device.streamon(&mut s, OUTPUT).unwrap();

    // The SOURCE_CHANGE lands and all four OUTPUT buffers come back.
    while output_dqbuf_at(&r.events.borrow(), 3).is_none() {
        assert!(wait_ready(&s), "no OUTPUT DQBUF for buffer 3 within 2s");
        process(&mut r.device, &mut s);
    }
    assert_eq!(source_changes(&r.events.borrow()), 1);
    assert_eq!(
        dequeued_on(&r.events.borrow(), OUTPUT).len(),
        4,
        "every staged input returned as consumed"
    );

    // The reinit: STREAMOFF(OUTPUT) while the format change is pending -- a reinit, NOT a seek.
    r.device.streamoff(&mut s, OUTPUT).unwrap();
    assert_eq!(r.log.lock().unwrap().flushes, 0, "no seek: the staged bitstream is kept");
    assert_eq!(r.log.lock().unwrap().reinits, 1, "the STREAMOFF(OUTPUT) was taken as a reinit");
    assert!(r.log.lock().unwrap().open, "a reinit does not tear the codec down");

    // CAPTURE STREAMOFF/REQBUFS(0)/G_FMT/REQBUFS/QBUF/STREAMON, then STREAMON(OUTPUT).
    r.device.streamoff(&mut s, CAPTURE).unwrap();
    r.device.reqbufs(&mut s, CAPTURE, MemoryType::Mmap, 0).unwrap();
    let sizeimage = pix(&r.device.g_fmt(&s, CAPTURE).unwrap()).sizeimage;
    r.device.reqbufs(&mut s, CAPTURE, MemoryType::Mmap, 4).unwrap();
    for i in 0..4 {
        r.device
            .qbuf(&mut s, mmap_buffer(CAPTURE, i, sizeimage), vec![], PayloadValidity::ALL)
            .unwrap();
    }
    r.device.streamon(&mut s, CAPTURE).unwrap();
    r.device.streamon(&mut s, OUTPUT).unwrap();

    // Every one of the four staged frames is delivered -- none dropped -- in order (ts 1..4).
    collect_capture(&mut r, &mut s, 4);
    let frames = dequeued_on(&r.events.borrow(), CAPTURE);
    assert_eq!(frames.len(), 4, "every frame delivered, none dropped across the reinit");
    for (n, f) in frames.iter().enumerate() {
        assert_eq!(f.timestamp().tv_sec, n as i64 + 1, "frame {n} in order");
    }
    assert_eq!(r.log.lock().unwrap().flushes, 0, "still no seek across the whole reinit");

    close(&mut r.device, s);
}

/// D53, test (b): the same rule at a *dynamic resolution change*. A mid-stream `SOURCE_CHANGE`
/// leaves a format change pending; a client that reconfigures CAPTURE by stopping OUTPUT first
/// (STREAMOFF(OUTPUT) then the CAPTURE dance, as ffmpeg's reinit would if it stopped OUTPUT) must
/// have that `STREAMOFF(OUTPUT)` taken as a reinit, not a seek -- the codec is kept and decoding
/// resumes at the new size. (The preservation of already-staged frames across a reinit is proven
/// by `reinit_across_the_initial_source_change_keeps_staged_input`; here the fake's non-parking
/// DRC model makes staging across the change ambiguous, so this test pins the classification and
/// the clean resume.)
#[test]
fn reinit_across_a_drc_is_not_a_seek() {
    let mut r = rig();
    let mut s = session(&mut r.device);
    let _sizeimage = start_streaming_320x240(&mut r, &mut s);

    // A frame at the first resolution.
    poke_mmap_output(&mut s, 0, 0x08);
    r.device
        .qbuf(&mut s, mmap_buffer(OUTPUT, 0, 1 << 20), vec![], PayloadValidity::ALL)
        .unwrap();
    collect_capture(&mut r, &mut s, 1);
    let before = source_changes(&r.events.borrow());

    // A magic buffer triggers a mid-stream resolution change to 160x120: a format change is now
    // pending (announced, CAPTURE not yet restarted for it).
    poke_mmap_output(&mut s, 1, DRC_MAGIC);
    r.device
        .qbuf(&mut s, mmap_buffer(OUTPUT, 1, 1 << 20), vec![], PayloadValidity::ALL)
        .unwrap();
    while source_changes(&r.events.borrow()) == before {
        assert!(wait_ready(&s), "no second SOURCE_CHANGE within 2s");
        process(&mut r.device, &mut s);
    }
    let new_sizeimage = pix(&r.device.g_fmt(&s, CAPTURE).unwrap()).sizeimage;
    assert_eq!(new_sizeimage, 160 * 120 * 3 / 2);

    // The reinit: STREAMOFF(OUTPUT) while the DRC is pending -- a reinit, not a seek.
    r.device.streamoff(&mut s, OUTPUT).unwrap();
    assert_eq!(r.log.lock().unwrap().flushes, 0, "no seek taken at the DRC reinit");
    assert_eq!(r.log.lock().unwrap().reinits, 1, "the DRC STREAMOFF(OUTPUT) was a reinit");
    assert!(r.log.lock().unwrap().open, "the codec is kept across the DRC reinit");

    // The CAPTURE reconfiguration at the new size, then STREAMON(OUTPUT).
    r.device.streamoff(&mut s, CAPTURE).unwrap();
    r.device.reqbufs(&mut s, CAPTURE, MemoryType::Mmap, 0).unwrap();
    r.device.reqbufs(&mut s, CAPTURE, MemoryType::Mmap, 4).unwrap();
    for i in 0..4 {
        r.device
            .qbuf(&mut s, mmap_buffer(CAPTURE, i, new_sizeimage), vec![], PayloadValidity::ALL)
            .unwrap();
    }
    r.device.streamon(&mut s, CAPTURE).unwrap();
    r.device.streamon(&mut s, OUTPUT).unwrap();

    // Decoding resumes at the new size on the same codec: a new-size frame comes out.
    poke_mmap_output(&mut s, 2, 0x09);
    r.device
        .qbuf(&mut s, mmap_buffer(OUTPUT, 2, 1 << 20), vec![], PayloadValidity::ALL)
        .unwrap();
    let already = dequeued_on(&r.events.borrow(), CAPTURE).len();
    while dequeued_on(&r.events.borrow(), CAPTURE).len() <= already {
        assert!(wait_ready(&s), "no post-DRC-reinit frame within 2s");
        process(&mut r.device, &mut s);
    }
    let frame = dequeued_on(&r.events.borrow(), CAPTURE).pop().unwrap();
    assert_eq!(*frame.get_first_plane().bytesused, new_sizeimage, "decoding resumed at the new size");
    assert_eq!(r.log.lock().unwrap().flushes, 0, "no seek anywhere in the DRC reinit");
    assert_eq!(r.log.lock().unwrap().started.len(), 1, "the same codec, not a new one");

    close(&mut r.device, s);
}

/// D53, test (c): a *genuine* seek -- `STREAMOFF(OUTPUT)` with CAPTURE streaming and no format
/// change pending -- still flushes: the staged bitstream is dropped, and the next stream is
/// stale-free. This is the behaviour B9 measured (0 stale frames per seek); the reinit rule must
/// not weaken it.
#[test]
fn a_true_seek_still_drops_staging_and_stays_stale_free() {
    let mut r = rig();
    let mut s = session(&mut r.device);
    let sizeimage = start_streaming_320x240(&mut r, &mut s);

    // Decode one frame so the codec is warm, and drain everything the initial setup produced.
    poke_mmap_output(&mut s, 0, 0x05);
    r.device
        .qbuf(&mut s, mmap_buffer(OUTPUT, 0, 1 << 20), vec![], PayloadValidity::ALL)
        .unwrap();
    collect_capture(&mut r, &mut s, 1);
    drain_events(&mut r, &mut s);

    // A genuine seek: CAPTURE is streaming and no format change is pending, so this is a flush,
    // NOT a reinit -- the staged bitstream is dropped, exactly as B9's 0-stale seek measured.
    r.device.streamoff(&mut s, OUTPUT).unwrap();
    assert_eq!(r.log.lock().unwrap().flushes, 1, "a genuine seek flushes the backend (drops staging)");
    assert_eq!(r.log.lock().unwrap().reinits, 0, "not a reinit: nothing was pending");
    assert!(r.log.lock().unwrap().open, "the codec survives a seek");

    // Resume and decode fresh, uniquely-timestamped data: it decodes cleanly on the same codec.
    r.device.streamon(&mut s, OUTPUT).unwrap();
    r.device
        .qbuf(&mut s, mmap_buffer(CAPTURE, 0, sizeimage), vec![], PayloadValidity::ALL)
        .unwrap();
    poke_mmap_output(&mut s, 1, 0x06);
    let mut ob = mmap_buffer(OUTPUT, 1, 1 << 20);
    ob.set_timestamp(ts(42));
    r.device.qbuf(&mut s, ob, vec![], PayloadValidity::ALL).unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    while !dequeued_on(&r.events.borrow(), CAPTURE)
        .iter()
        .any(|f| f.timestamp().tv_sec == 42)
    {
        assert!(std::time::Instant::now() < deadline, "no post-seek frame within 2s");
        if wait_ready(&s) {
            process(&mut r.device, &mut s);
        }
    }
    assert_eq!(r.log.lock().unwrap().flushes, 1, "still exactly one seek");
    assert_eq!(r.log.lock().unwrap().started.len(), 1, "the same codec, not a new one");

    close(&mut r.device, s);
}

/// D71, order STREAMON-then-announce: ffmpeg brings CAPTURE up from the `S_FMT(OUTPUT)`
/// placeholder *before* the `SOURCE_CHANGE` arrives. The announce then finds CAPTURE already
/// streaming at a size it does not change, so nothing has to be reconfigured and no later
/// `STREAMON(CAPTURE)` will come to clear `format_change_pending`. The flag must therefore not be
/// left set: a `STREAMOFF(OUTPUT)` after that (ffmpeg's EOF close, or a genuine seek) must be a
/// seek -- a flush that drops staging -- not a reinit, or B9's 0-stale-seek property is silently
/// disarmed for the life of the session.
#[test]
fn an_announce_after_streamon_capture_leaves_a_later_streamoff_a_seek() {
    let mut r = rig();
    let mut s = session(&mut r.device);

    r.device.s_fmt(&mut s, OUTPUT, output_format(H264, 320, 240)).unwrap();
    r.device.reqbufs(&mut s, OUTPUT, MemoryType::Mmap, 4).unwrap();
    r.device
        .subscribe_event(&mut s, EventType::SourceChange(0), SubscribeEventFlags::empty())
        .unwrap();

    // CAPTURE up at the placeholder size, streaming, before any SOURCE_CHANGE -- the fake will
    // announce FAKE_STREAM_SIZE (320x240), the same size, so the announce changes nothing.
    let sizeimage = pix(&r.device.g_fmt(&s, CAPTURE).unwrap()).sizeimage;
    r.device.reqbufs(&mut s, CAPTURE, MemoryType::Mmap, 4).unwrap();
    for i in 0..4 {
        r.device
            .qbuf(&mut s, mmap_buffer(CAPTURE, i, sizeimage), vec![], PayloadValidity::ALL)
            .unwrap();
    }
    r.device.streamon(&mut s, CAPTURE).unwrap();

    // Now feed OUTPUT and stream it on: the backend parses and raises SOURCE_CHANGE, which the
    // device handles with CAPTURE already streaming.
    poke_mmap_output(&mut s, 0, 0x01);
    r.device
        .qbuf(&mut s, mmap_buffer(OUTPUT, 0, 1 << 20), vec![], PayloadValidity::ALL)
        .unwrap();
    r.device.streamon(&mut s, OUTPUT).unwrap();
    while source_changes(&r.events.borrow()) == 0 {
        assert!(wait_ready(&s), "no SOURCE_CHANGE within 2s");
        process(&mut r.device, &mut s);
    }
    assert_eq!(source_changes(&r.events.borrow()), 1);

    // The STREAMOFF(OUTPUT): a genuine seek (flush), NOT a reinit -- the change was never left
    // pending, because CAPTURE was already streaming for this format (D71).
    r.device.streamoff(&mut s, OUTPUT).unwrap();
    assert_eq!(
        r.log.lock().unwrap().flushes,
        1,
        "STREAMOFF(OUTPUT) after an announce that found CAPTURE streaming is a seek"
    );
    assert_eq!(
        r.log.lock().unwrap().reinits,
        0,
        "not a reinit: the format change was not left pending (D71)"
    );

    close(&mut r.device, s);
}

/// D71, order announce-then-STREAMON: the ordinary GStreamer/ffmpeg flow where CAPTURE is started
/// *after* the `SOURCE_CHANGE`. `start_streaming_320x240` does exactly that (announce, then
/// `REQBUFS`/`STREAMON(CAPTURE)`), and `STREAMON(CAPTURE)` must clear the pending flag so that a
/// later `STREAMOFF(OUTPUT)` is a seek again -- B9's 0-stale seek. (The pending window itself, and
/// the reinit taken inside it, are pinned by `reinit_across_the_initial_source_change_keeps_staged_input`.)
#[test]
fn streamon_capture_after_the_announce_clears_the_pending_change() {
    let mut r = rig();
    let mut s = session(&mut r.device);
    let sizeimage = start_streaming_320x240(&mut r, &mut s);

    // Decode a frame so the codec is warm, then drain.
    poke_mmap_output(&mut s, 0, 0x07);
    r.device
        .qbuf(&mut s, mmap_buffer(OUTPUT, 0, 1 << 20), vec![], PayloadValidity::ALL)
        .unwrap();
    collect_capture(&mut r, &mut s, 1);
    drain_events(&mut r, &mut s);

    // STREAMON(CAPTURE) has already cleared the pending change (it ran inside
    // start_streaming_320x240), so this STREAMOFF(OUTPUT) is a seek, not a reinit.
    r.device.streamoff(&mut s, OUTPUT).unwrap();
    assert_eq!(r.log.lock().unwrap().flushes, 1, "a seek after STREAMON(CAPTURE) cleared the change");
    assert_eq!(r.log.lock().unwrap().reinits, 0, "not a reinit: STREAMON(CAPTURE) cleared it");
    let _ = sizeimage;

    close(&mut r.device, s);
}

/// D72: a `REQBUFS(CAPTURE, n)` with `n` below the codec's announced minimum is raised to the
/// minimum, the way `vb2_core_reqbufs` bumps a count up to the driver's floor. ffmpeg asks for a
/// fixed 20 and never reads `MIN_BUFFERS_FOR_CAPTURE`; the codec here needs 21, so the count must
/// come back 21, not 20. A count already at or above the minimum, and one on the OUTPUT queue, are
/// left as asked (capped at `MAX_BUFFERS`).
#[test]
fn reqbufs_capture_is_raised_to_the_announced_minimum() {
    let mut r = rig_announcing_min(21);
    let mut s = session(&mut r.device);
    start_streaming_320x240(&mut r, &mut s);

    // The codec announced 21; a client asking for 20 (ffmpeg's default) is granted 21.
    let sizeimage = pix(&r.device.g_fmt(&s, CAPTURE).unwrap()).sizeimage;
    r.device.streamoff(&mut s, CAPTURE).unwrap();
    r.device.reqbufs(&mut s, CAPTURE, MemoryType::Mmap, 0).unwrap();
    let reply = r.device.reqbufs(&mut s, CAPTURE, MemoryType::Mmap, 20).unwrap();
    assert_eq!(reply.count, 21, "REQBUFS(CAPTURE, 20) is raised to the announced minimum 21 (D72)");

    // A count already above the minimum is left as asked.
    r.device.reqbufs(&mut s, CAPTURE, MemoryType::Mmap, 0).unwrap();
    let reply = r.device.reqbufs(&mut s, CAPTURE, MemoryType::Mmap, 25).unwrap();
    assert_eq!(reply.count, 25, "a count above the minimum is not lowered");

    // The OUTPUT queue is never raised: only CAPTURE has the codec's minimum.
    r.device.streamoff(&mut s, OUTPUT).unwrap();
    r.device.reqbufs(&mut s, OUTPUT, MemoryType::Mmap, 0).unwrap();
    let reply = r.device.reqbufs(&mut s, OUTPUT, MemoryType::Mmap, 2).unwrap();
    assert_eq!(reply.count, 2, "OUTPUT REQBUFS is granted as asked");

    let _ = sizeimage;
    close(&mut r.device, s);
}

// ---------------------------------------------------------------------------------------------
// F19: D64 (the head-GOP loss is a codec-internal drop, not the device or DEC_CMD_START) and D77
// (the learned CAPTURE floor). See logs/vpu_wp/F19-decoder.md.
// ---------------------------------------------------------------------------------------------

/// D64, the orchestrator's first hypothesis, falsified in code: `V4L2_DEC_CMD_START` answering the
/// *initial* `SOURCE_CHANGE` (ffmpeg's response -- it never re-`REQBUFS`) must not flush, restart
/// or discard anything. B15 §2 proved the head-GOP loss is a codec-internal input drop under the
/// output-slot stall, not the device dropping frames (`300 in, 270 out, 0 held output(s) dropped`).
/// This replays ffmpeg's shape -- feed OUTPUT so the codec announces and produces frames while no
/// `CAPTURE` buffer is lent yet, answer with `DEC_CMD_START`, then provide `CAPTURE` -- and asserts
/// **every produced frame reaches the client, in order**, and that the backend's `resume` /
/// `flush` / `clear` were never called by the initial `DEC_CMD_START` (`drain == None`, so it is a
/// no-op in the device: the frames wait in the backend and flow when buffers arrive).
#[test]
fn dec_cmd_start_on_the_initial_source_change_loses_no_produced_frame() {
    let mut r = rig();
    let mut s = session(&mut r.device);

    r.device.s_fmt(&mut s, OUTPUT, output_format(H264, 320, 240)).unwrap();
    r.device.reqbufs(&mut s, OUTPUT, MemoryType::Mmap, 8).unwrap();
    r.device
        .subscribe_event(&mut s, EventType::SourceChange(0), SubscribeEventFlags::empty())
        .unwrap();
    r.device.streamon(&mut s, OUTPUT).unwrap();

    // Feed six OUTPUT buffers with distinct timestamps. The fake announces on the first and
    // produces one frame per buffer; with no CAPTURE buffer lent yet the frames wait in the
    // backend -- exactly the pre-QBUF window in which the head GOP is at risk on the phone.
    for i in 0..6u32 {
        poke_mmap_output(&mut s, i as usize, (0x01 + i) as u8);
        let mut ob = mmap_buffer(OUTPUT, i, 1 << 20);
        ob.set_timestamp(ts(i as i64));
        r.device.qbuf(&mut s, ob, vec![], PayloadValidity::ALL).unwrap();
    }
    while source_changes(&r.events.borrow()) == 0 {
        assert!(wait_ready(&s), "no SOURCE_CHANGE within 2s");
        process(&mut r.device, &mut s);
    }

    // ffmpeg's answer to the change: DEC_CMD_START, with no re-REQBUFS. On the initial announce
    // the drain is None, so this is a device no-op -- it must not touch the held frames.
    r.device.decoder_cmd(&mut s, dec_cmd(bindings::V4L2_DEC_CMD_START)).unwrap();
    assert_eq!(
        r.log.lock().unwrap().resumes,
        0,
        "the initial DEC_CMD_START does not resume the backend (drain was None): it is a no-op"
    );
    assert_eq!(r.log.lock().unwrap().flushes, 0, "DEC_CMD_START does not flush/seek");
    assert_eq!(r.log.lock().unwrap().clears, 0, "DEC_CMD_START does not clear CAPTURE");

    // Now provide CAPTURE buffers (the QBUF ffmpeg issues after the change). Every one of the six
    // frames the codec produced comes out, in timestamp order -- nothing was lost to the announce
    // or to DEC_CMD_START.
    let sizeimage = pix(&r.device.g_fmt(&s, CAPTURE).unwrap()).sizeimage;
    r.device.reqbufs(&mut s, CAPTURE, MemoryType::Mmap, 8).unwrap();
    for i in 0..8 {
        r.device
            .qbuf(&mut s, mmap_buffer(CAPTURE, i, sizeimage), vec![], PayloadValidity::ALL)
            .unwrap();
    }
    r.device.streamon(&mut s, CAPTURE).unwrap();
    collect_capture(&mut r, &mut s, 6);

    let stamps: Vec<i64> = dequeued_on(&r.events.borrow(), CAPTURE)
        .iter()
        .map(|b| b.timestamp().tv_sec as i64)
        .collect();
    assert_eq!(
        stamps,
        vec![0, 1, 2, 3, 4, 5],
        "every produced frame reaches the client in order across DEC_CMD_START (D64: the device \
         loses none; the phone's head-GOP loss is the codec dropping input under the slot stall)"
    );
    assert_eq!(errors(&r.events.borrow()), 0, "the session did not fail");

    close(&mut r.device, s);
}

/// D77: a client that sizes its CAPTURE pool with a fixed `REQBUFS(CAPTURE)` *before* the
/// `SOURCE_CHANGE` (ffmpeg's 20, which it never re-asks) gets it raised to the codec's output-slot
/// count when an earlier session has learned it -- the backend remembers the last announced
/// minimum per coded format, and the device seeds the session's floor from it. Without the learned
/// floor the pre-announce count is granted verbatim (the D72 raise cannot fire: the session minimum
/// is still the bare 1). This is the whole of D77 (B15 §3): the floor now applies at REQBUFS time,
/// not only after an announce the client never re-`REQBUFS` after.
#[test]
fn a_learned_minimum_raises_a_pre_announce_capture_reqbufs() {
    // The backend learned 21 for this coded format in an earlier session.
    let mut r = rig_with_learned_min(21);
    let mut s = session(&mut r.device);

    r.device.s_fmt(&mut s, OUTPUT, output_format(H264, 320, 240)).unwrap();
    r.device.reqbufs(&mut s, OUTPUT, MemoryType::Mmap, 8).unwrap();

    // G_CTRL(MIN_BUFFERS_FOR_CAPTURE) already reports the learned floor, before any announce: a
    // client that reads it (GStreamer) sizes its pool correctly from the start.
    assert_eq!(
        s.min_capture_buffers, 21,
        "the session's CAPTURE floor is seeded from the learned minimum before the announce (D77)"
    );

    // The pre-announce REQBUFS(CAPTURE, 20) -- ffmpeg's fixed count -- is raised to 21.
    let sizeimage = pix(&r.device.g_fmt(&s, CAPTURE).unwrap()).sizeimage;
    let reply = r.device.reqbufs(&mut s, CAPTURE, MemoryType::Mmap, 20).unwrap();
    assert_eq!(
        reply.count, 21,
        "a pre-announce REQBUFS(CAPTURE, 20) is raised to the learned floor 21 (D77); \
         without it the count would be granted verbatim as 20"
    );
    let _ = sizeimage;

    close(&mut r.device, s);
}

/// D77, the other half: when the announce raises the CAPTURE minimum above the count the client is
/// already streaming with -- ffmpeg brought CAPTURE up before the change and answered with
/// `DEC_CMD_START`, and no learned floor had met the codec's need -- the session must **not** be
/// failed. The backend holds the decoded outputs until more buffers are queued; the shortfall is
/// logged once (`warned_capture_below_announced_min`), and a client that keeps its queue supplied
/// still decodes. Here CAPTURE streams with four buffers (each large enough for the announced
/// size), the codec announces min 21, and frames still flow.
#[test]
fn an_announce_above_the_streaming_capture_count_holds_rather_than_fails() {
    let mut r = rig_announcing_min(21);
    let mut s = session(&mut r.device);

    r.device.s_fmt(&mut s, OUTPUT, output_format(H264, 320, 240)).unwrap();
    r.device.reqbufs(&mut s, OUTPUT, MemoryType::Mmap, 8).unwrap();
    r.device
        .subscribe_event(&mut s, EventType::SourceChange(0), SubscribeEventFlags::empty())
        .unwrap();

    // CAPTURE up with four buffers, streaming, before the SOURCE_CHANGE -- fewer than the 21 the
    // codec will announce. The buffers are the announced size (FAKE_STREAM_SIZE), so they can hold
    // frames; the shortfall is in the count, not the size.
    let sizeimage = pix(&r.device.g_fmt(&s, CAPTURE).unwrap()).sizeimage;
    r.device.reqbufs(&mut s, CAPTURE, MemoryType::Mmap, 4).unwrap();
    for i in 0..4 {
        r.device
            .qbuf(&mut s, mmap_buffer(CAPTURE, i, sizeimage), vec![], PayloadValidity::ALL)
            .unwrap();
    }
    r.device.streamon(&mut s, CAPTURE).unwrap();

    // Feed OUTPUT so the codec announces min 21 while CAPTURE already streams with four.
    poke_mmap_output(&mut s, 0, 0x01);
    let mut ob = mmap_buffer(OUTPUT, 0, 1 << 20);
    ob.set_timestamp(ts(0));
    r.device.qbuf(&mut s, ob, vec![], PayloadValidity::ALL).unwrap();
    r.device.streamon(&mut s, OUTPUT).unwrap();
    while source_changes(&r.events.borrow()) == 0 {
        assert!(wait_ready(&s), "no SOURCE_CHANGE within 2s");
        process(&mut r.device, &mut s);
    }

    // The session was not failed, and the shortfall was noted once (D77).
    assert_eq!(errors(&r.events.borrow()), 0, "the announce above the streaming count does not fail the session");
    assert!(
        s.warned_capture_below_announced_min,
        "the count shortfall at the announce is made visible (D77)"
    );
    assert_eq!(s.min_capture_buffers, 21, "the announced minimum is recorded");

    // A frame still reaches the client: the four buffers hold the announced size.
    collect_capture(&mut r, &mut s, 1);
    let stamps: Vec<i64> = dequeued_on(&r.events.borrow(), CAPTURE)
        .iter()
        .map(|b| b.timestamp().tv_sec as i64)
        .collect();
    assert_eq!(stamps.first(), Some(&0), "the first frame is delivered despite the count shortfall");

    close(&mut r.device, s);
}

/// D58 instrument + reproduction: `v4l2-compliance -d /dev/videoN -s`'s streaming test on the
/// decoder's OUTPUT (m2m) queue queues three zero-filled bitstream buffers and expects each to be
/// returned (`DQBUF(OUTPUT)`) as the codec consumes them (v4l-utils 1.32.0
/// `v4l2-test-buffers.cpp`'s `captureBufs`: for an OUTPUT queue it fills, queues, streams, and
/// dequeues every buffer, printing "Frame #NNN" per dequeue). B11-acceptance §3.2 saw the third
/// `DQBUF(OUTPUT)` never complete (D58). This test replays that sequence against a backend that
/// announces after the second buffer and never produces a frame, and asserts the DEVICE returns
/// all three OUTPUT buffers: if it does (it does), the device is not where the buffer is lost, and
/// the wedge is downstream (the crosvm event queue, or the guest driver's DQBUF wake-up) -- the
/// B13 dig, see the report.
#[test]
fn compliance_zero_buffer_streaming_returns_every_output_buffer() {
    let mut r = rig();
    let mut s = session(&mut r.device);

    r.device.s_fmt(&mut s, OUTPUT, output_format(H264, 320, 240)).unwrap();
    r.device.reqbufs(&mut s, OUTPUT, MemoryType::Mmap, 3).unwrap();
    r.device
        .subscribe_event(&mut s, EventType::SourceChange(0), SubscribeEventFlags::empty())
        .unwrap();

    // Three zero-filled OUTPUT buffers (compliance fills with zero), queued and streamed. Buffers 0
    // and 2 are NO_ANNOUNCE (consumed, no frame, InputBufferDone held while awaiting the format);
    // buffer 1 -- the second buffer -- announces the format, which releases the held ones. This is
    // the device-side signature B11-acceptance §3.2 measured ("announces after buffer 2, 1 format
    // change, 0 frames out"), and the third buffer (index 2) is the `DQBUF(OUTPUT)` that never
    // completed there (D58). The one frame the announcer would stage stays undelivered (no CAPTURE
    // buffer), so 0 frames go out, as measured.
    for i in 0..3u32 {
        let byte = if i == 1 { 0x11 } else { NO_ANNOUNCE_MAGIC };
        poke_mmap_output(&mut s, i as usize, byte);
        let mut ob = mmap_buffer(OUTPUT, i, 1 << 20);
        ob.set_timestamp(ts(i as i64 + 1));
        r.device.qbuf(&mut s, ob, vec![], PayloadValidity::ALL).unwrap();
    }
    r.device.streamon(&mut s, OUTPUT).unwrap();

    // All three OUTPUT buffers must come back (DQBUF(OUTPUT)), the third included: the two held
    // NO_ANNOUNCE buffers are released when buffer 2 announces, and buffer 2 (a plain buffer after
    // the announcement) is returned at once. Poll until buffer 2's DQBUF, or time out.
    while output_dqbuf_at(&r.events.borrow(), 2).is_none() {
        assert!(wait_ready(&s), "the third OUTPUT buffer's DQBUF never arrived (D58 on the device)");
        process(&mut r.device, &mut s);
    }
    let returned: HashSet<u32> = dequeued_on(&r.events.borrow(), OUTPUT)
        .iter()
        .map(|b| b.index())
        .collect();
    assert_eq!(returned.len(), 3, "every OUTPUT buffer returned: 0, 1 and 2");
    assert!(returned.contains(&0) && returned.contains(&1) && returned.contains(&2));
    assert_eq!(source_changes(&r.events.borrow()), 1, "one format change, as the repro measured");

    close(&mut r.device, s);
}

/// Guest-owned buffers (`driver_owned_queues=all`): the OUTPUT bitstream is a read-only guest
/// mapping held until `InputBufferDone`, the CAPTURE frame lands in a writable guest mapping
/// dropped before the DQBUF event, and a CAPTURE buffer too small for a frame is refused.
#[test]
fn userptr_output_and_capture_go_through_guest_mappings() {
    let mut r = rig();
    let mut s = session(&mut r.device);
    r.device.s_fmt(&mut s, OUTPUT, output_format(H264, 320, 240)).unwrap();

    // OUTPUT bitstream buffers are guest-owned (USERPTR).
    let reply = r.device.reqbufs(&mut s, OUTPUT, MemoryType::UserPtr, 2).unwrap();
    assert!(reply.capabilities & BufferCapabilities::SUPPORTS_USERPTR.bits() != 0);

    r.device
        .subscribe_event(&mut s, EventType::SourceChange(0), SubscribeEventFlags::empty())
        .unwrap();

    // Put a bitstream byte into guest memory and queue the OUTPUT buffer over it; its frame is
    // produced (with luma derived from that byte) once the CAPTURE queue is set up.
    let out_gpa = 0x1000u64;
    r.guest.memory.borrow_mut()[out_gpa as usize] = 0x44;
    let (ob, sgs) = userptr_buffer(OUTPUT, 0, out_gpa, 1 << 20, 4096);
    r.device
        .qbuf(&mut s, ob, sgs, PayloadValidity::ALL)
        .unwrap();
    assert_eq!(
        *r.guest.live_mappings.borrow(),
        1,
        "OUTPUT mapping held while queued"
    );
    r.device.streamon(&mut s, OUTPUT).unwrap();

    // SOURCE_CHANGE, then the OUTPUT buffer's mapping is dropped when InputBufferDone arrives.
    while source_changes(&r.events.borrow()) == 0 {
        assert!(wait_ready(&s), "no SOURCE_CHANGE within 2s");
        process(&mut r.device, &mut s);
    }
    assert_eq!(dequeued_on(&r.events.borrow(), OUTPUT).len(), 1);
    assert_eq!(*r.guest.live_mappings.borrow(), 0, "OUTPUT mapping released after InputBufferDone");

    // CAPTURE buffers guest-owned too.
    let sizeimage = pix(&r.device.g_fmt(&s, CAPTURE).unwrap()).sizeimage;
    r.device.reqbufs(&mut s, CAPTURE, MemoryType::UserPtr, 2).unwrap();
    // A buffer whose length cannot hold a frame is refused, nothing mapped.
    let (small, sgs) = userptr_buffer(CAPTURE, 0, 0x200000, sizeimage - 1, sizeimage - 1);
    assert_eq!(
        r.device
            .qbuf(&mut s, small, sgs, PayloadValidity::ALL)
            .err(),
        Some(libc::EINVAL)
    );
    assert_eq!(*r.guest.live_mappings.borrow(), 0);

    let cap_gpa = 0x200000u64;
    let (cb, sgs) = userptr_buffer(CAPTURE, 0, cap_gpa, sizeimage, sizeimage);
    r.device
        .qbuf(&mut s, cb, sgs, PayloadValidity::ALL)
        .unwrap();
    assert_eq!(
        *r.guest.live_mappings.borrow(),
        1,
        "CAPTURE mapping held while lent"
    );
    r.device.streamon(&mut s, CAPTURE).unwrap();

    // The frame lands in the guest pages, then its mapping is dropped before the DQBUF event.
    collect_capture(&mut r, &mut s, 1);
    assert_eq!(*r.guest.live_mappings.borrow(), 0, "CAPTURE mapping released before the DQBUF");
    {
        let mem = r.guest.memory.borrow();
        assert_eq!(mem[cap_gpa as usize], luma_of(0x44), "the frame landed in guest memory");
        assert_eq!(mem[cap_gpa as usize + (320 * 240)], 0x80, "chroma");
    }

    close(&mut r.device, s);
}

/// §2.5: nothing is written to a buffer after it is unqueued or freed; `STREAMOFF`/`REQBUFS(0)`/
/// close join the backend before a buffer goes back to the allocator; a second session is EBUSY.
#[test]
fn lifecycle_invariants_join_the_backend_before_freeing() {
    let mut r = rig();
    let mut s = session(&mut r.device);
    let sizeimage = start_streaming_320x240(&mut r, &mut s);

    // Decode a couple of frames.
    for i in 0..2u32 {
        poke_mmap_output(&mut s, i as usize, 0x30 + i as u8);
        r.device
            .qbuf(
                &mut s,
                mmap_buffer(OUTPUT, i, 1 << 20),
                vec![],
                PayloadValidity::ALL,
            )
            .unwrap();
    }
    // Requeue capture buffers as they come back.
    collect_capture(&mut r, &mut s, 2);

    // REQBUFS(0) on CAPTURE joins (clear_capture) before freeing; nothing written meanwhile.
    let clears_before = r.log.lock().unwrap().clears;
    r.device.reqbufs(&mut s, CAPTURE, MemoryType::Mmap, 0).unwrap();
    assert_eq!(r.log.lock().unwrap().clears, clears_before + 1, "REQBUFS(0) clears once");
    assert!(s.output.buffers.is_empty());
    assert_eq!(*r.log.lock().unwrap().released_while_capture_active.borrow(), 0);

    // Rebuild capture, then close the whole session: stop() joins, then buffers are freed.
    r.device
        .reqbufs(&mut s, CAPTURE, MemoryType::Mmap, 2)
        .unwrap();
    r.device
        .qbuf(
            &mut s,
            mmap_buffer(CAPTURE, 0, sizeimage),
            vec![],
            PayloadValidity::ALL,
        )
        .unwrap();
    close(&mut r.device, s);
    assert_eq!(r.log.lock().unwrap().stops, 1, "close joined the backend exactly once");
    assert_eq!(*r.log.lock().unwrap().released_while_capture_active.borrow(), 0);
    assert_eq!(r.pool.used(), 0, "and every buffer went back to the pool");
}

/// vb2's rule (`vb2_core_reqbufs`, Linux 6.18.21 `videobuf2-core.c:883-886`): `REQBUFS` with a
/// non-zero count on a streaming queue is `EBUSY`, because the buffers it would free may be lent
/// to the codec (`M7-crate` §9 item 3). `REQBUFS(0)` is still the implicit `STREAMOFF` and joins
/// before freeing, and `CREATE_BUFS` -- which only appends -- keeps working, as in vb2
/// (`vb2_core_create_bufs`, `videobuf2-core.c:1038-1081`, no `q->streaming` check).
#[test]
fn reqbufs_on_a_streaming_queue_is_busy() {
    let mut r = rig();
    let mut s = session(&mut r.device);
    let sizeimage = start_streaming_320x240(&mut r, &mut s);
    assert!(s.state.output_streaming && s.state.capture_streaming);

    // Both queues stream and their buffers are lent: no REQBUFS(n) may free them.
    assert_eq!(
        r.device.reqbufs(&mut s, CAPTURE, MemoryType::Mmap, 2).err(),
        Some(libc::EBUSY)
    );
    assert_eq!(s.output.buffers.len(), 4, "the CAPTURE buffers are still there");
    assert_eq!(
        r.device.reqbufs(&mut s, OUTPUT, MemoryType::Mmap, 2).err(),
        Some(libc::EBUSY)
    );
    assert_eq!(s.input.buffers.len(), 4, "the OUTPUT buffers are still there");
    assert_eq!(*r.log.lock().unwrap().released_while_capture_active.borrow(), 0);
    assert_eq!(r.log.lock().unwrap().clears, 0, "nothing joined the backend either");

    // CREATE_BUFS appends to a streaming queue and the new buffer can be queued at once.
    let created = r
        .device
        .create_bufs(
            &mut s,
            2,
            CAPTURE,
            MemoryType::Mmap,
            capture_format_sized(320, 240, 1, sizeimage),
        )
        .unwrap();
    assert_eq!((created.index, created.count), (4, 2));
    assert_eq!(s.output.buffers.len(), 6);
    r.device
        .qbuf(
            &mut s,
            mmap_buffer(CAPTURE, 4, sizeimage),
            vec![],
            PayloadValidity::ALL,
        )
        .unwrap();

    // REQBUFS(0) joins the backend first, then frees everything.
    r.device.reqbufs(&mut s, CAPTURE, MemoryType::Mmap, 0).unwrap();
    assert_eq!(r.log.lock().unwrap().clears, 1, "REQBUFS(0) clears once");
    assert!(s.output.buffers.is_empty());
    assert!(!s.state.capture_streaming);
    assert_eq!(*r.log.lock().unwrap().released_while_capture_active.borrow(), 0);

    // And with the queue stopped, a fresh set is fine again.
    r.device.reqbufs(&mut s, CAPTURE, MemoryType::Mmap, 2).unwrap();
    close(&mut r.device, s);
}

/// review-m4 R1 for the decoder: a session that ends without a `CLOSE` -- a worker returning on
/// its kill event, a `stop_queue`, a `reset` -- is only dropped, and must still join the codec
/// before the buffers it was lent go away. The session's field order (`backend` before the two
/// queues) is the mechanism; a mapping released too early trips `FakeMapping::drop`.
#[test]
fn a_dropped_session_joins_the_backend_before_its_buffers() {
    let mut r = rig();
    let mut s = session(&mut r.device);
    r.device.s_fmt(&mut s, OUTPUT, output_format(H264, 320, 240)).unwrap();
    r.device.reqbufs(&mut s, OUTPUT, MemoryType::Mmap, 2).unwrap();
    r.device
        .subscribe_event(&mut s, EventType::SourceChange(0), SubscribeEventFlags::empty())
        .unwrap();
    poke_mmap_output(&mut s, 0, 0x21);
    r.device
        .qbuf(
            &mut s,
            mmap_buffer(OUTPUT, 0, 1 << 20),
            vec![],
            PayloadValidity::ALL,
        )
        .unwrap();
    r.device.streamon(&mut s, OUTPUT).unwrap();
    while source_changes(&r.events.borrow()) == 0 {
        assert!(wait_ready(&s), "no SOURCE_CHANGE within 2s");
        process(&mut r.device, &mut s);
    }

    // Two guest-owned CAPTURE buffers for one ready frame: the second stays held by the codec
    // thread, its guest mapping alive, when the session goes away.
    let sizeimage = pix(&r.device.g_fmt(&s, CAPTURE).unwrap()).sizeimage;
    r.device.reqbufs(&mut s, CAPTURE, MemoryType::UserPtr, 2).unwrap();
    for i in 0..2u32 {
        let (cb, sgs) = userptr_buffer(CAPTURE, i, 0x200000 + i as u64 * 0x40000, sizeimage, 0);
        r.device
            .qbuf(&mut s, cb, sgs, PayloadValidity::ALL)
            .unwrap();
    }
    r.device.streamon(&mut s, CAPTURE).unwrap();
    collect_capture(&mut r, &mut s, 1);
    assert_eq!(*r.guest.live_mappings.borrow(), 1, "the second buffer is still lent");
    assert!(r.log.lock().unwrap().holding.len() == 1, "the codec holds it");

    // No close_session, no stop: just drop it, as the runner's kill path does.
    drop(s);
    let log = r.log.lock().unwrap();
    assert_eq!(log.stops, 1, "the drop joined the codec");
    assert!(!log.open, "the codec thread is gone");
    assert!(!log.capture_active);
    drop(log);
    assert_eq!(*r.guest.live_mappings.borrow(), 0, "and only then were the mappings released");
}

/// review-m4 R2 for the decoder: `PREPARE_BUF` maps nothing, so the `QBUF` that follows carries
/// the scatter list again -- and `ioctl::get_userptr_regions` reads that list against *this*
/// call's `length`. A `QBUF` that shrinks the plane is refused on both queues, whatever
/// `PREPARE_BUF` accepted, and nothing is mapped from the short list.
#[test]
fn prepare_buf_does_not_let_qbuf_shrink_the_mapping() {
    let mut r = rig();
    let mut s = session(&mut r.device);
    r.device.s_fmt(&mut s, OUTPUT, output_format(H264, 320, 240)).unwrap();
    r.device.reqbufs(&mut s, OUTPUT, MemoryType::UserPtr, 1).unwrap();
    let bitstream = MIN_BITSTREAM_SIZE;

    // OUTPUT: prepared at a full-length plane, nothing mapped yet.
    let (pb, sgs) = userptr_buffer(OUTPUT, 0, 0x1000, bitstream, 4096);
    let prepared = r
        .device
        .prepare_buf(&mut s, pb, sgs, PayloadValidity::ALL)
        .unwrap();
    assert!(prepared.flags().contains(BufferFlags::PREPARED));
    assert_eq!(*r.guest.live_mappings.borrow(), 0, "PREPARE_BUF maps nothing");

    // The same buffer queued over an eight-byte scatter list: refused, still prepared.
    let (short, sgs) = userptr_buffer(OUTPUT, 0, 0x1000, 8, 8);
    assert_eq!(
        r.device
            .qbuf(&mut s, short, sgs, PayloadValidity::ALL)
            .err(),
        Some(libc::EINVAL)
    );
    assert_eq!(*r.guest.live_mappings.borrow(), 0, "nothing was mapped");
    assert!(s.input.buffers[0].prepared.is_some(), "the buffer is still prepared");
    assert!(!s.input.buffers[0].queued);

    // A plane longer than the buffer was allocated for is refused too.
    let (big, sgs) = userptr_buffer(OUTPUT, 0, 0x1000, bitstream + 1, 4096);
    assert_eq!(
        r.device.qbuf(&mut s, big, sgs, PayloadValidity::ALL).err(),
        Some(libc::EINVAL)
    );

    // The honest QBUF maps the full list and keeps PREPARE_BUF's payload.
    let (ok, sgs) = userptr_buffer(OUTPUT, 0, 0x1000, bitstream, 0);
    let queued = r
        .device
        .qbuf(&mut s, ok, sgs, PayloadValidity::ALL)
        .unwrap();
    assert_eq!(
        *queued.get_first_plane().bytesused,
        4096,
        "the prepared payload survived"
    );
    assert_eq!(*r.guest.live_mappings.borrow(), 1);

    // CAPTURE: the same, over a frame-sized plane.
    let sizeimage = pix(&r.device.g_fmt(&s, CAPTURE).unwrap()).sizeimage;
    r.device.reqbufs(&mut s, CAPTURE, MemoryType::UserPtr, 1).unwrap();
    let cap_gpa = 0x300000u64;
    let (pb, sgs) = userptr_buffer(CAPTURE, 0, cap_gpa, sizeimage, 0);
    r.device
        .prepare_buf(&mut s, pb, sgs, PayloadValidity::ALL)
        .unwrap();
    let (short, sgs) = userptr_buffer(CAPTURE, 0, cap_gpa, 8, 0);
    assert_eq!(
        r.device
            .qbuf(&mut s, short, sgs, PayloadValidity::ALL)
            .err(),
        Some(libc::EINVAL)
    );
    assert_eq!(
        *r.guest.live_mappings.borrow(),
        1,
        "still only the OUTPUT mapping"
    );
    // A full-length plane whose scatter list is short is caught by the backstop.
    let (mut lying, _) = userptr_buffer(CAPTURE, 0, cap_gpa, sizeimage, 0);
    *lying.get_first_plane_mut().length = sizeimage;
    let short_sgs = vec![vec![SgEntry::new(cap_gpa, sizeimage - 1)]];
    assert_eq!(
        r.device
            .qbuf(&mut s, lying, short_sgs, PayloadValidity::ALL)
            .err(),
        Some(libc::EINVAL)
    );
    assert_eq!(
        *r.guest.live_mappings.borrow(),
        1,
        "the short mapping was dropped"
    );

    let (ok, sgs) = userptr_buffer(CAPTURE, 0, cap_gpa, sizeimage, 0);
    r.device
        .qbuf(&mut s, ok, sgs, PayloadValidity::ALL)
        .unwrap();
    assert_eq!(*r.guest.live_mappings.borrow(), 2);

    close(&mut r.device, s);
}

/// The kernel's resolution-change sequence ends the *decoder*, not the *stream*: the last CAPTURE
/// buffer of the old resolution carries `V4L2_BUF_FLAG_LAST` -- it may be empty -- and no
/// `V4L2_EVENT_EOS` follows it (`dev-decoder.rst`, "Dynamic Resolution Change" step 2 asks for the
/// flag "similarly to the Drain sequence" and lists the event nowhere; "Drain" step 3 is the only
/// place it appears). The decoder then stays stopped -- a CAPTURE buffer queued in the meantime
/// waits -- until `V4L2_DEC_CMD_START` or the `STREAMON(CAPTURE)` of the reallocation.
///
/// This is the behaviour a backend gets when it marks the change (`DRC_LAST_MAGIC`; `F7.md` §3 is
/// the exact crosvm change that makes the MediaCodec backend do it), and the reason
/// `is_last` no longer implies `EOS`.
#[test]
fn a_resolution_change_marks_its_last_buffer_and_sends_no_eos() {
    let mut r = rig();
    let mut s = session(&mut r.device);
    start_streaming_320x240(&mut r, &mut s);

    // A frame at the first resolution, then the change.
    poke_mmap_output(&mut s, 0, 0x08);
    r.device
        .qbuf(
            &mut s,
            mmap_buffer(OUTPUT, 0, 1 << 20),
            vec![],
            PayloadValidity::ALL,
        )
        .unwrap();
    collect_capture(&mut r, &mut s, 1);
    let before = source_changes(&r.events.borrow());
    poke_mmap_output(&mut s, 1, DRC_LAST_MAGIC);
    r.device
        .qbuf(
            &mut s,
            mmap_buffer(OUTPUT, 1, 1 << 20),
            vec![],
            PayloadValidity::ALL,
        )
        .unwrap();
    while source_changes(&r.events.borrow()) == before {
        assert!(wait_ready(&s), "no SOURCE_CHANGE within 2s");
        process(&mut r.device, &mut s);
    }

    // The buffer before the announcement is the LAST one, and it ends no stream.
    let frames = dequeued_on(&r.events.borrow(), CAPTURE);
    let last = frames.last().unwrap().clone();
    assert!(
        last.flags().contains(BufferFlags::LAST),
        "the last buffer of the old resolution carries LAST"
    );
    assert_eq!(*last.get_first_plane().bytesused, 0, "and may be empty");
    assert!(
        !frames[0].flags().contains(BufferFlags::LAST),
        "and only the last one"
    );
    assert_eq!(
        eos_events(&r.events.borrow()),
        0,
        "a resolution change sends no EOS, unlike a drain"
    );
    assert_eq!(s.drain, Drain::Stopped, "but the decoder is stopped");

    // Stopped: a CAPTURE buffer queued now waits instead of being lent, and `DEC_CMD_START`
    // resumes (so does the `STREAMON(CAPTURE)` of the reallocation, which the DRC test covers).
    let index = last.index() as usize;
    let sizeimage = pix(&r.device.g_fmt(&s, CAPTURE).unwrap()).sizeimage;
    r.device
        .qbuf(
            &mut s,
            mmap_buffer(CAPTURE, index as u32, sizeimage),
            vec![],
            PayloadValidity::ALL,
        )
        .unwrap();
    assert_eq!(s.output.pending.len(), 1, "not lent while the decoder is stopped");
    assert!(!s.output.buffers[index].lent);
    r.device.decoder_cmd(&mut s, dec_cmd(bindings::V4L2_DEC_CMD_START)).unwrap();
    assert_eq!(s.drain, Drain::None);
    assert!(s.output.pending.is_empty());
    assert!(s.output.buffers[index].lent, "DEC_CMD_START resumes the decoder");
    assert_eq!(eos_events(&r.events.borrow()), 0);

    close(&mut r.device, s);
}

/// The same sequence from a backend that marks nothing, which the fork must keep accepting. The
/// `SOURCE_CHANGE` alone still stops the decoder (the kernel's implicit drain), so no CAPTURE
/// buffer is handed to a codec that has stopped decoding at that size, and no `EOS` is invented
/// -- but the client is still owed the `LAST` buffer it waits for before it renegotiates
/// (GStreamer, review-m6 R6-2), so the device supplies it from the next CAPTURE buffer the guest
/// queues, empty, as `v4l2_m2m_qbuf` does for a stopped decoder. One only: the buffer after it
/// waits for the resume.
#[test]
fn a_resolution_change_stops_the_decoder_even_when_nothing_is_marked() {
    let mut r = rig();
    let mut s = session(&mut r.device);
    start_streaming_320x240(&mut r, &mut s);
    poke_mmap_output(&mut s, 0, 0x08);
    r.device
        .qbuf(
            &mut s,
            mmap_buffer(OUTPUT, 0, 1 << 20),
            vec![],
            PayloadValidity::ALL,
        )
        .unwrap();
    collect_capture(&mut r, &mut s, 1);
    let before = source_changes(&r.events.borrow());

    poke_mmap_output(&mut s, 1, DRC_MAGIC);
    r.device
        .qbuf(
            &mut s,
            mmap_buffer(OUTPUT, 1, 1 << 20),
            vec![],
            PayloadValidity::ALL,
        )
        .unwrap();
    while source_changes(&r.events.borrow()) == before {
        assert!(wait_ready(&s), "no SOURCE_CHANGE within 2s");
        process(&mut r.device, &mut s);
    }
    assert!(
        dequeued_on(&r.events.borrow(), CAPTURE)
            .iter()
            .all(|b| !b.flags().contains(BufferFlags::LAST)),
        "this backend marks nothing"
    );
    assert_eq!(eos_events(&r.events.borrow()), 0);
    assert_eq!(s.drain, Drain::Stopped);
    assert!(
        s.last_owed,
        "the client is owed the LAST buffer of the old resolution"
    );

    // The first announcement is not a change and must not stop anything: the frame above proves
    // the initial SOURCE_CHANGE left the decoder running. The CAPTURE buffer queued now comes
    // straight back, empty and LAST, from the device itself.
    let index = 0usize;
    let sizeimage = pix(&r.device.g_fmt(&s, CAPTURE).unwrap()).sizeimage;
    let dequeued_before = dequeued_on(&r.events.borrow(), CAPTURE).len();
    r.device
        .qbuf(
            &mut s,
            mmap_buffer(CAPTURE, index as u32, sizeimage),
            vec![],
            PayloadValidity::ALL,
        )
        .unwrap();
    let frames = dequeued_on(&r.events.borrow(), CAPTURE);
    assert_eq!(frames.len(), dequeued_before + 1, "returned at once");
    let last = frames.last().unwrap();
    assert_eq!(last.index() as usize, index);
    assert!(last.flags().contains(BufferFlags::LAST));
    assert!(!last.flags().contains(BufferFlags::QUEUED));
    assert_eq!(*last.get_first_plane().bytesused, 0);
    assert!(!s.output.buffers[index].queued && !s.output.buffers[index].lent);
    assert!(!s.last_owed);
    assert_eq!(
        eos_events(&r.events.borrow()),
        0,
        "still no EOS: nothing was drained"
    );
    assert_eq!(s.drain, Drain::Stopped, "and the decoder stays stopped");

    // The next one waits, not lent, until the resume.
    r.device
        .qbuf(
            &mut s,
            mmap_buffer(CAPTURE, index as u32, sizeimage),
            vec![],
            PayloadValidity::ALL,
        )
        .unwrap();
    assert!(s.output.buffers[index].queued);
    assert!(
        !s.output.buffers[index].lent,
        "not lent while the decoder is stopped"
    );
    assert_eq!(
        dequeued_on(&r.events.borrow(), CAPTURE).len(),
        dequeued_before + 1
    );
    r.device
        .decoder_cmd(&mut s, dec_cmd(bindings::V4L2_DEC_CMD_START))
        .unwrap();
    assert!(s.output.buffers[index].lent);
    assert_eq!(
        r.log.lock().unwrap().resumes,
        1,
        "the backend hears of DEC_CMD_START"
    );

    close(&mut r.device, s);
}

/// The MediaCodec backend's order (`android.rs` `announce`): the frames of the old size, the
/// announcement, then the empty `LAST` buffer right behind it, which is the order the kernel
/// documents ("Dynamic Resolution Change" step 1 then 2) and the one both clients poll for. The
/// device takes it as it takes the other order: `LAST` and no `EOS`, the decoder stopped, and
/// nothing owed -- the next CAPTURE buffer queued waits for the resume instead of being
/// answered a second time.
#[test]
fn a_resolution_change_may_mark_its_last_buffer_after_the_announcement() {
    let mut r = rig();
    let mut s = session(&mut r.device);
    start_streaming_320x240(&mut r, &mut s);
    poke_mmap_output(&mut s, 0, 0x08);
    r.device
        .qbuf(
            &mut s,
            mmap_buffer(OUTPUT, 0, 1 << 20),
            vec![],
            PayloadValidity::ALL,
        )
        .unwrap();
    collect_capture(&mut r, &mut s, 1);
    let before = source_changes(&r.events.borrow());

    poke_mmap_output(&mut s, 1, DRC_LAST_AFTER_MAGIC);
    r.device
        .qbuf(
            &mut s,
            mmap_buffer(OUTPUT, 1, 1 << 20),
            vec![],
            PayloadValidity::ALL,
        )
        .unwrap();
    // Until both the announcement and the LAST buffer behind it are in.
    loop {
        let events = r.events.borrow();
        let done = source_changes(&events) > before
            && dequeued_on(&events, CAPTURE)
                .last()
                .is_some_and(|b| b.flags().contains(BufferFlags::LAST));
        drop(events);
        if done {
            break;
        }
        assert!(wait_ready(&s), "no SOURCE_CHANGE + LAST within 2s");
        process(&mut r.device, &mut s);
    }
    assert_eq!(source_changes(&r.events.borrow()), before + 1);
    let frames = dequeued_on(&r.events.borrow(), CAPTURE);
    let last = frames.last().unwrap().clone();
    assert_eq!(*last.get_first_plane().bytesused, 0);
    assert!(
        frames[..frames.len() - 1]
            .iter()
            .all(|b| !b.flags().contains(BufferFlags::LAST)),
        "only the last one"
    );
    assert_eq!(
        eos_events(&r.events.borrow()),
        0,
        "no EOS for a resolution change"
    );
    assert_eq!(s.drain, Drain::Stopped);
    assert!(!s.last_owed, "the backend's LAST buffer settled the debt");
    // The event came first, so a query made on it already answers the new size.
    assert_eq!(pix(&r.device.g_fmt(&s, CAPTURE).unwrap()).width, 160);

    let index = last.index() as usize;
    let sizeimage = pix(&r.device.g_fmt(&s, CAPTURE).unwrap()).sizeimage;
    let dequeued_before = frames.len();
    r.device
        .qbuf(
            &mut s,
            mmap_buffer(CAPTURE, index as u32, sizeimage),
            vec![],
            PayloadValidity::ALL,
        )
        .unwrap();
    assert!(s.output.buffers[index].queued && !s.output.buffers[index].lent);
    assert_eq!(
        dequeued_on(&r.events.borrow(), CAPTURE).len(),
        dequeued_before,
        "no second LAST"
    );
    r.device
        .decoder_cmd(&mut s, dec_cmd(bindings::V4L2_DEC_CMD_START))
        .unwrap();
    assert!(s.output.buffers[index].lent);

    close(&mut r.device, s);
}

/// A CAPTURE buffer smaller than the announced canvas -- a queue sized from the client's own
/// placeholder before the `SOURCE_CHANGE`, and never reallocated -- is lent anyway (the guest may
/// still take it back), but a backend can only hold it unfilled, so decoding stalls with nothing
/// in the log to say why (`M6-backend` §10 item 6). The device says it once per session.
#[test]
fn a_capture_buffer_too_small_for_the_canvas_is_reported_once() {
    let mut r = rig();
    let mut s = session(&mut r.device);
    r.device.s_fmt(&mut s, OUTPUT, output_format(H264, 160, 120)).unwrap();
    r.device.reqbufs(&mut s, OUTPUT, MemoryType::Mmap, 2).unwrap();
    r.device
        .subscribe_event(&mut s, EventType::SourceChange(0), SubscribeEventFlags::empty())
        .unwrap();

    // The CAPTURE queue sized from the placeholder, before the stream has been parsed.
    let placeholder = pix(&r.device.g_fmt(&s, CAPTURE).unwrap()).sizeimage;
    assert_eq!(placeholder, 160 * 120 * 3 / 2);
    r.device.reqbufs(&mut s, CAPTURE, MemoryType::Mmap, 2).unwrap();
    for i in 0..2 {
        r.device
            .qbuf(
                &mut s,
                mmap_buffer(CAPTURE, i, placeholder),
                vec![],
                PayloadValidity::ALL,
            )
            .unwrap();
    }

    // The stream turns out to be 320x240: those buffers hold a quarter of a frame.
    poke_mmap_output(&mut s, 0, 0x03);
    r.device
        .qbuf(
            &mut s,
            mmap_buffer(OUTPUT, 0, 1 << 20),
            vec![],
            PayloadValidity::ALL,
        )
        .unwrap();
    r.device.streamon(&mut s, OUTPUT).unwrap();
    while source_changes(&r.events.borrow()) == 0 {
        assert!(wait_ready(&s), "no SOURCE_CHANGE within 2s");
        process(&mut r.device, &mut s);
    }
    let sizeimage = pix(&r.device.g_fmt(&s, CAPTURE).unwrap()).sizeimage;
    assert_eq!(sizeimage, 320 * 240 * 3 / 2);
    assert!(!s.warned_small_capture, "nothing has been lent yet");

    // STREAMON(CAPTURE) lends them, warns once, and no frame can come out of them.
    r.device.streamon(&mut s, CAPTURE).unwrap();
    assert!(s.warned_small_capture, "the too-small buffer was reported");
    assert!(s.output.buffers.iter().all(|b| b.lent), "and lent all the same");
    process(&mut r.device, &mut s);
    assert!(
        dequeued_on(&r.events.borrow(), CAPTURE).is_empty(),
        "the frame waits for a buffer that fits"
    );

    // The reallocation the client owes: buffers of the announced size, and frames flow.
    r.device.streamoff(&mut s, CAPTURE).unwrap();
    r.device.reqbufs(&mut s, CAPTURE, MemoryType::Mmap, 0).unwrap();
    r.device.reqbufs(&mut s, CAPTURE, MemoryType::Mmap, 2).unwrap();
    for i in 0..2 {
        r.device
            .qbuf(
                &mut s,
                mmap_buffer(CAPTURE, i, sizeimage),
                vec![],
                PayloadValidity::ALL,
            )
            .unwrap();
    }
    r.device.streamon(&mut s, CAPTURE).unwrap();
    poke_mmap_output(&mut s, 1, 0x04);
    r.device
        .qbuf(
            &mut s,
            mmap_buffer(OUTPUT, 1, 1 << 20),
            vec![],
            PayloadValidity::ALL,
        )
        .unwrap();
    collect_capture(&mut r, &mut s, 1);
    let frame = dequeued_on(&r.events.borrow(), CAPTURE).remove(0);
    assert_eq!(*frame.get_first_plane().bytesused, sizeimage);
    assert_eq!(*r.log.lock().unwrap().released_while_capture_active.borrow(), 0);

    close(&mut r.device, s);
}

/// One guest `open()` is one decode session and sessions are independent: both provision OUTPUT
/// and CAPTURE, both stream, each gets its own frames in its own buffers, and closing one leaves
/// the other streaming with everything it holds intact.
///
/// The device used to let one session hold buffers at a time and answered a second's
/// `REQBUFS`/`CREATE_BUFS` with `EBUSY`. `P6b-verify` §3.2/§6.1 measured what that cost: libva
/// opens a V4L2 node per context, so a second context -- two independent processes reproduce it --
/// could not provision until the first let go, and Firefox holds the previous page's decoder
/// 2.4-5.6 s across a navigation, well past libva's 2000 ms budget.
#[test]
fn two_sessions_decode_side_by_side() {
    let mut r = rig();
    let mut a = new_session(&mut r.device, 0);
    let a_sizeimage = start_streaming_320x240(&mut r, &mut a);

    // The second session provisions and streams while the first holds buffers and streams: its
    // `REQBUFS(OUTPUT)` is the call the gate refused.
    let mut b = new_session(&mut r.device, 1);
    let b_sizeimage = start_streaming_320x240(&mut r, &mut b);
    assert_eq!(a_sizeimage, b_sizeimage);
    assert!(a.state.output_streaming && a.state.capture_streaming, "A still streams");
    assert!(b.state.output_streaming && b.state.capture_streaming, "B streams too");
    assert_eq!((a.input.buffers.len(), a.output.buffers.len()), (4, 4));
    assert_eq!((b.input.buffers.len(), b.output.buffers.len()), (4, 4));
    assert_eq!(r.log.lock().unwrap().started.len(), 2, "two codecs are running at once");

    // Nothing is shared: the sixteen buffers are sixteen distinct `MMAP` offsets.
    let offsets: HashSet<u32> = a
        .input
        .buffers
        .iter()
        .chain(a.output.buffers.iter())
        .chain(b.input.buffers.iter())
        .chain(b.output.buffers.iter())
        .map(host_offset)
        .collect();
    assert_eq!(offsets.len(), 16, "every buffer of both sessions has its own offset");

    // The frame each was already decoding (`start_streaming_320x240` feeds both a buffer whose
    // first byte is 0x01) comes back to that session alone.
    assert_eq!(collect_capture_for(&mut r, &mut a, 1).len(), 1);
    assert_eq!(collect_capture_for(&mut r, &mut b, 1).len(), 1);

    // Now a different bitstream into each, at the same time: neither frame lands in the other's
    // buffers.
    poke_mmap_output(&mut a, 1, 0x20);
    r.device
        .qbuf(&mut a, mmap_buffer(OUTPUT, 1, 1 << 20), vec![], PayloadValidity::ALL)
        .unwrap();
    poke_mmap_output(&mut b, 1, 0x40);
    r.device
        .qbuf(&mut b, mmap_buffer(OUTPUT, 1, 1 << 20), vec![], PayloadValidity::ALL)
        .unwrap();
    let a_frame = collect_capture_for(&mut r, &mut a, 1).remove(0);
    let b_frame = collect_capture_for(&mut r, &mut b, 1).remove(0);
    assert_eq!(capture_luma(&a, a_frame.index() as usize), luma_of(0x20), "A decoded A's stream");
    assert_eq!(capture_luma(&b, b_frame.index() as usize), luma_of(0x40), "B decoded B's stream");

    // Closing B leaves A streaming, with its buffers where they were and its last frame intact.
    let a_offsets: Vec<u32> = a
        .input
        .buffers
        .iter()
        .chain(a.output.buffers.iter())
        .map(host_offset)
        .collect();
    close(&mut r.device, b);
    assert!(a.state.output_streaming && a.state.capture_streaming, "A survives B's close");
    assert_eq!((a.input.buffers.len(), a.output.buffers.len()), (4, 4));
    assert_eq!(
        a.input
            .buffers
            .iter()
            .chain(a.output.buffers.iter())
            .map(host_offset)
            .collect::<Vec<_>>(),
        a_offsets,
        "not one of A's buffers was freed or re-registered"
    );
    assert_eq!(
        capture_luma(&a, a_frame.index() as usize),
        luma_of(0x20),
        "A's frame is still there"
    );
    assert_eq!(*r.log.lock().unwrap().released_while_capture_active.borrow(), 0);

    // And A decodes on.
    poke_mmap_output(&mut a, 2, 0x60);
    r.device
        .qbuf(&mut a, mmap_buffer(OUTPUT, 2, 1 << 20), vec![], PayloadValidity::ALL)
        .unwrap();
    let last = collect_capture_for(&mut r, &mut a, 1).remove(0);
    assert_eq!(capture_luma(&a, last.index() as usize), luma_of(0x60), "A keeps decoding after B");

    close(&mut r.device, a);
    assert_eq!(r.pool.used(), 0, "both sessions gave every buffer back");
}

/// A second session provisions while the first is mid-decode -- its CAPTURE buffers lent to the
/// codec thread -- and nothing of the first's is touched. Both queues come out of the one pool,
/// so what matters is that provisioning only ever *adds*: no lent buffer is freed, re-registered
/// or given back to the allocator behind the codec that is writing it (§2.5).
#[test]
fn a_second_session_provisions_while_the_first_holds_lent_buffers() {
    let mut r = rig();
    let mut a = new_session(&mut r.device, 0);
    start_streaming_320x240(&mut r, &mut a);
    collect_capture_for(&mut r, &mut a, 1);

    // A is mid-decode: the three CAPTURE buffers its first frame did not use are still lent.
    let lent_before: Vec<bool> = a.output.buffers.iter().map(|b| b.lent).collect();
    let held_before = r.log.lock().unwrap().holding.clone();
    assert_eq!(lent_before.iter().filter(|l| **l).count(), 3, "three CAPTURE buffers are lent");
    assert_eq!(held_before.len(), 3, "and the codec thread holds exactly those");
    let a_offsets: Vec<u32> = a
        .input
        .buffers
        .iter()
        .chain(a.output.buffers.iter())
        .map(host_offset)
        .collect();
    let used_before = r.pool.used();

    // All of the second session's provisioning happens while that is true. `CREATE_BUFS` had a
    // gate of its own, so B builds its whole OUTPUT queue with it -- the way a GStreamer pool
    // grows -- and `REQBUFS` provisions CAPTURE.
    let mut b = new_session(&mut r.device, 1);
    r.device.s_fmt(&mut b, OUTPUT, output_format(H264, 320, 240)).unwrap();
    let first = r
        .device
        .create_bufs(&mut b, 1, OUTPUT, MemoryType::Mmap, output_format(H264, 320, 240))
        .unwrap();
    assert_eq!((first.index, first.count), (0, 1), "B's first OUTPUT buffer");
    let grown = r
        .device
        .create_bufs(&mut b, 2, OUTPUT, MemoryType::Mmap, output_format(H264, 320, 240))
        .unwrap();
    assert_eq!((grown.index, grown.count), (1, 2), "and two more, appended to B's own queue");
    assert_eq!(r.device.reqbufs(&mut b, CAPTURE, MemoryType::Mmap, 4).unwrap().count, 4);
    r.device.streamon(&mut b, OUTPUT).unwrap();
    assert_eq!(r.log.lock().unwrap().started.len(), 2, "B's codec started under A's");

    // A is exactly as it was, and its buffers are still the codec thread's.
    assert_eq!(a.output.buffers.iter().map(|b| b.lent).collect::<Vec<_>>(), lent_before);
    assert_eq!(r.log.lock().unwrap().holding.clone(), held_before, "the same buffers, still lent");
    assert_eq!(
        a.input
            .buffers
            .iter()
            .chain(a.output.buffers.iter())
            .map(host_offset)
            .collect::<Vec<_>>(),
        a_offsets,
        "not one of A's buffers was freed or re-registered"
    );
    assert_eq!(*r.log.lock().unwrap().released_while_capture_active.borrow(), 0);
    assert!(r.pool.used() > used_before, "B's buffers came out of the same pool, on top of A's");

    // And A decodes into its own lent buffers as if nothing had happened.
    poke_mmap_output(&mut a, 1, 0x20);
    r.device
        .qbuf(&mut a, mmap_buffer(OUTPUT, 1, 1 << 20), vec![], PayloadValidity::ALL)
        .unwrap();
    let a_frame = collect_capture_for(&mut r, &mut a, 1).remove(0);
    assert_eq!(capture_luma(&a, a_frame.index() as usize), luma_of(0x20));

    close(&mut r.device, b);
    close(&mut r.device, a);
    assert_eq!(r.pool.used(), 0, "both sessions gave every buffer back");
}

/// A backend whose codec will not start (reclaimed, out of memory) fails `STREAMON(OUTPUT)` with
/// that errno and leaves the buffers queued for another try.
#[test]
fn a_codec_that_will_not_start_fails_streamon() {
    let mut r = rig_with(Some(libc::EBUSY));
    let mut s = session(&mut r.device);
    r.device.s_fmt(&mut s, OUTPUT, output_format(H264, 320, 240)).unwrap();
    r.device.reqbufs(&mut s, OUTPUT, MemoryType::Mmap, 2).unwrap();
    poke_mmap_output(&mut s, 0, 0x01);
    r.device
        .qbuf(
            &mut s,
            mmap_buffer(OUTPUT, 0, 1 << 20),
            vec![],
            PayloadValidity::ALL,
        )
        .unwrap();
    assert_eq!(r.device.streamon(&mut s, OUTPUT).err(), Some(libc::EBUSY));
    assert!(!s.state.output_streaming);
    assert!(s.input.buffers[0].queued, "the buffer stays queued for a retry");
    close(&mut r.device, s);
}

/// The `v4l2-compliance` decoder-command probe (`v4l2-test-codecs.cpp:testDecoder`, non-stateless
/// branch): TRY/DECODER_CMD(STOP) and (START) succeed with the flags a decoder does not act on
/// cleared, PAUSE/RESUME are EINVAL.
#[test]
fn decoder_cmd_matches_the_compliance_probe() {
    let mut r = rig();
    let mut s = session(&mut r.device);

    // STOP with every flag set comes back with flags cleared (a decoder honours none).
    let mut stop = dec_cmd(bindings::V4L2_DEC_CMD_STOP);
    stop.flags = !0;
    let out = r.device.try_decoder_cmd(&s, stop).unwrap();
    assert_eq!(out.cmd, bindings::V4L2_DEC_CMD_STOP);
    assert_eq!(out.flags, 0);

    // START with speed/format set comes back cleared.
    let mut start = dec_cmd(bindings::V4L2_DEC_CMD_START);
    start.flags = !0;
    // Writing a Copy union field is safe; only reads need `unsafe`.
    start.__bindgen_anon_1.start.speed = 0x7fff;
    let out = r.device.try_decoder_cmd(&s, start).unwrap();
    assert_eq!(out.flags, 0);
    // SAFETY: START variant.
    assert_eq!(unsafe { out.__bindgen_anon_1.start.speed }, 0);

    // PAUSE / RESUME are EINVAL for a decoder.
    assert_eq!(r.device.decoder_cmd(&mut s, dec_cmd(bindings::V4L2_DEC_CMD_PAUSE)).err(), Some(libc::EINVAL));
    assert_eq!(r.device.decoder_cmd(&mut s, dec_cmd(bindings::V4L2_DEC_CMD_RESUME)).err(), Some(libc::EINVAL));

    // STOP with the queues not streaming is a no-op success, not a failure.
    assert!(r.device.decoder_cmd(&mut s, dec_cmd(bindings::V4L2_DEC_CMD_STOP)).is_ok());

    close(&mut r.device, s);
}

/// `CREATE_BUFS` sizes buffers itself: it refuses a set with the wrong plane count or a
/// `sizeimage` smaller than the queue's format needs (D6.2/D9), and honours a larger one.
#[test]
fn create_bufs_refuses_a_set_too_small() {
    let mut r = rig();
    let mut s = session(&mut r.device);
    r.device.s_fmt(&mut s, OUTPUT, output_format(H264, 320, 240)).unwrap();
    r.device.reqbufs(&mut s, CAPTURE, MemoryType::Mmap, 0).unwrap();
    let sizeimage = 320 * 240 * 3 / 2;

    // A set with no plane, or a plane too small, is refused.
    let noplane = capture_format_sized(320, 240, 0, sizeimage);
    assert_eq!(
        r.device.create_bufs(&mut s, 1, CAPTURE, MemoryType::Mmap, noplane).err(),
        Some(libc::EINVAL)
    );
    let small = capture_format_sized(320, 240, 1, sizeimage / 2);
    assert_eq!(
        r.device.create_bufs(&mut s, 1, CAPTURE, MemoryType::Mmap, small).err(),
        Some(libc::EINVAL)
    );

    // A larger set is honoured, and the buffers really are that large.
    let big = capture_format_sized(320, 240, 1, sizeimage * 2);
    let reply = r.device.create_bufs(&mut s, 1, CAPTURE, MemoryType::Mmap, big).unwrap();
    assert_eq!((reply.index, reply.count), (0, 1));
    assert_eq!(pix(&reply.format).sizeimage, sizeimage * 2);
    let queried = r.device.querybuf(&s, CAPTURE, 0).unwrap();
    assert_eq!(*queried.get_first_plane().length, sizeimage * 2);

    close(&mut r.device, s);
}

/// D21 -- the exact `QBUF` `ffmpeg -f v4l2` sends, replayed on the wire, on both queues.
///
/// ffmpeg declares `length = VIDEO_MAX_PLANES` and fills only `planes[0]` from `QUERYBUF`;
/// `planes[1..8]` are its own stack (`logs/vpu_wp/B5-acceptance.md` §4.3). Judging all eight
/// slots refused every buffer it queued. These formats have one plane, and vb2 looks at
/// `vb->num_planes` entries only (`__verify_length`, `videobuf2-v4l2.c:105`) -- but plane 0 on
/// an output queue is the payload length, and that one is still checked.
#[test]
fn ffmpegs_dirty_plane_array_is_judged_on_the_planes_the_format_has() {
    let mut r = rig();
    let mut s = session(&mut r.device);
    r.device.s_fmt(&mut s, OUTPUT, output_format(H264, 320, 240)).unwrap();
    r.device.reqbufs(&mut s, OUTPUT, MemoryType::Mmap, 2).unwrap();

    // OUTPUT (the bitstream) with a sane plane 0 under the dirty tail: queued, and the length
    // the guest declared is kept.
    let bytes = ffmpeg_wire::ffmpeg_qbuf_bytes(OUTPUT, MemoryType::Mmap, 0, (4096, 1 << 20));
    assert_eq!(
        ffmpeg_wire::dispatch_qbuf(&mut r.device, &mut s, &bytes),
        0,
        "ffmpeg's plane array was refused on the output queue (D21)"
    );
    assert_eq!(*s.input.buffers[0].v4l2_buffer.get_first_plane().bytesused, 4096);

    // The same array with the garbage in plane 0: `bytesused > length` is what
    // `__verify_length` does check on an output queue, so this is still `EINVAL`.
    let bytes =
        ffmpeg_wire::ffmpeg_qbuf_bytes(OUTPUT, MemoryType::Mmap, 1, ((1 << 20) + 1, 1 << 20));
    assert_eq!(
        ffmpeg_wire::dispatch_qbuf(&mut r.device, &mut s, &bytes),
        libc::EINVAL,
        "a bitstream length the buffer cannot hold was accepted"
    );
    assert!(!s.input.buffers[1].queued);

    // CAPTURE, `MMAP`: the frame is this device's to fill, so the guest's payload is ignored.
    let sizeimage = pix(&r.device.g_fmt(&s, CAPTURE).unwrap()).sizeimage;
    r.device.reqbufs(&mut s, CAPTURE, MemoryType::Mmap, 2).unwrap();
    let bytes = ffmpeg_wire::ffmpeg_qbuf_bytes(CAPTURE, MemoryType::Mmap, 0, (0, sizeimage));
    assert_eq!(ffmpeg_wire::dispatch_qbuf(&mut r.device, &mut s, &bytes), 0);
    assert!(s.output.buffers[0].queued);

    close(&mut r.device, s);
}

/// `PREPARE_BUF` validates a payload without queueing, and the `QBUF` that follows keeps the
/// prepared description and ignores its own (D6.1), for a guest-owned buffer.
#[test]
fn prepare_buf_then_qbuf_keeps_the_prepared_payload() {
    let mut r = rig();
    let mut s = session(&mut r.device);
    r.device.s_fmt(&mut s, OUTPUT, output_format(H264, 320, 240)).unwrap();
    r.device.reqbufs(&mut s, OUTPUT, MemoryType::UserPtr, 1).unwrap();

    let out_gpa = 0x4000u64;
    let (pb, _sgs) = userptr_buffer(OUTPUT, 0, out_gpa, 1 << 20, 5000);
    let prepared = r
        .device
        .prepare_buf(&mut s, pb, vec![], PayloadValidity::ALL)
        .unwrap();
    assert!(prepared.flags().contains(BufferFlags::PREPARED));
    assert_eq!(*prepared.get_first_plane().bytesused, 5000);

    // A following QBUF with nonsense payload keeps the prepared description.
    let (mut qb, sgs) = userptr_buffer(OUTPUT, 0, out_gpa, 1 << 20, 0xdead_beef);
    *qb.get_first_plane_mut().bytesused = 0xdead_beef;
    let queued = r
        .device
        .qbuf(&mut s, qb, sgs, PayloadValidity::ALL)
        .unwrap();
    assert!(queued.flags().contains(BufferFlags::QUEUED));
    assert_eq!(*queued.get_first_plane().bytesused, 5000, "the prepared bytesused survived");

    close(&mut r.device, s);
}

/// `SUBSCRIBE_EVENT` accepts `SOURCE_CHANGE`, `EOS`, and -- since D29 gave the decoder a control
/// table -- a `V4L2_EVENT_CTRL` on the class marker and each exposed control (D50); anything else
/// is `EINVAL`. A stateful decoder must offer all three (the `v4l2-compliance` `testEvents` rule;
/// refusing the control event cost the decoder subtest 48/47/1 -> 48/46/2, B9-acceptance §4.2).
/// With `SEND_INITIAL` a control event is answered at once with the control's state -- for every
/// control but the class marker, which the kernel's `v4l2_ctrl_add_event` sends no initial value
/// for. Modelled on the encoder's `events_are_eos_and_control_events_only`.
#[test]
fn events_are_source_change_eos_and_control_events() {
    let mut r = rig();
    let mut s = session(&mut r.device);
    const CID_MIN_CAP: u32 = bindings::V4L2_CID_MIN_BUFFERS_FOR_CAPTURE;

    assert!(r
        .device
        .subscribe_event(&mut s, EventType::SourceChange(0), SubscribeEventFlags::empty())
        .is_ok());
    assert!(r
        .device
        .subscribe_event(&mut s, EventType::Eos, SubscribeEventFlags::empty())
        .is_ok());
    assert_eq!(
        r.device
            .subscribe_event(&mut s, EventType::VSync, SubscribeEventFlags::empty())
            .err(),
        Some(libc::EINVAL)
    );

    // The control event compliance subscribes to: MIN_BUFFERS_FOR_CAPTURE. With SEND_INITIAL the
    // current value goes out at once. Before any SOURCE_CHANGE it is the floor, 1.
    assert!(r
        .device
        .subscribe_event(
            &mut s,
            EventType::Ctrl(CID_MIN_CAP),
            SubscribeEventFlags::SEND_INITIAL,
        )
        .is_ok());
    // The class marker is accepted too, but carries no initial value.
    assert!(r
        .device
        .subscribe_event(
            &mut s,
            EventType::Ctrl(bindings::V4L2_CID_USER_CLASS),
            SubscribeEventFlags::SEND_INITIAL,
        )
        .is_ok());
    // Subscribing without SEND_INITIAL emits nothing.
    assert!(r
        .device
        .subscribe_event(&mut s, EventType::Ctrl(CID_MIN_CAP), SubscribeEventFlags::empty())
        .is_ok());
    // A control the decoder does not have: EINVAL, not silently accepted.
    assert_eq!(
        r.device
            .subscribe_event(
                &mut s,
                EventType::Ctrl(bindings::V4L2_CID_BRIGHTNESS),
                SubscribeEventFlags::SEND_INITIAL,
            )
            .err(),
        Some(libc::EINVAL)
    );

    // Exactly one initial control event went out: MIN_BUFFERS_FOR_CAPTURE = 1 (the class marker
    // and the no-flag subscription send none).
    assert_eq!(ctrl_events(&r.events.borrow()), vec![(CID_MIN_CAP, 1)]);

    // Unsubscribe accepts SOURCE_CHANGE, EOS and a control id, and V4L2_EVENT_ALL.
    let unsub = |type_: u32, id: u32| bindings::v4l2_event_subscription {
        type_,
        id,
        ..Default::default()
    };
    assert!(r
        .device
        .unsubscribe_event(&mut s, unsub(bindings::V4L2_EVENT_CTRL, CID_MIN_CAP))
        .is_ok());
    assert!(r
        .device
        .unsubscribe_event(&mut s, unsub(bindings::V4L2_EVENT_SOURCE_CHANGE, 0))
        .is_ok());
    assert!(!s.src_change_subscribed);
    assert!(r
        .device
        .unsubscribe_event(&mut s, unsub(bindings::V4L2_EVENT_ALL, 0))
        .is_ok());
    assert!(!s.eos_subscribed);
    close(&mut r.device, s);
}

/// A codec that dies mid-stream ends the session with one error event; every ioctl that would
/// touch it then answers `ENODEV`, and `REQBUFS(0)` / close still work.
#[test]
fn a_codec_error_ends_the_session() {
    let mut r = rig();
    let mut s = session(&mut r.device);
    let _ = start_streaming_320x240(&mut r, &mut s);

    // Inject a backend error by hand-delivering it (the codec would report AMEDIACODEC_ERROR_*).
    r.device.handle_event(&mut s, DecoderEvent::Error("codec reclaimed".into()));
    assert!(s.dead);
    assert_eq!(errors(&r.events.borrow()), 1);

    poke_mmap_output(&mut s, 0, 0x01);
    assert_eq!(
        r.device
            .qbuf(
                &mut s,
                mmap_buffer(OUTPUT, 0, 1 << 20),
                vec![],
                PayloadValidity::ALL
            )
            .err(),
        Some(libc::ENODEV)
    );
    assert_eq!(r.device.streamon(&mut s, OUTPUT).err(), Some(libc::ENODEV));
    // A dead session takes no more pool space (review-m6 R6-11): the three allocating ioctls
    // are refused, and `REQBUFS(0)` still frees.
    assert_eq!(
        r.device.reqbufs(&mut s, CAPTURE, MemoryType::Mmap, 4).err(),
        Some(libc::ENODEV)
    );
    assert_eq!(
        r.device
            .create_bufs(
                &mut s,
                1,
                CAPTURE,
                MemoryType::Mmap,
                capture_format(320, 240)
            )
            .err(),
        Some(libc::ENODEV)
    );
    assert_eq!(
        r.device
            .prepare_buf(
                &mut s,
                mmap_buffer(OUTPUT, 0, 1 << 20),
                vec![],
                PayloadValidity::ALL
            )
            .err(),
        Some(libc::ENODEV)
    );
    r.device
        .reqbufs(&mut s, CAPTURE, MemoryType::Mmap, 0)
        .unwrap();
    assert!(s.output.buffers.is_empty());
    assert_eq!(
        errors(&r.events.borrow()),
        1,
        "one error event, however many causes"
    );

    close(&mut r.device, s);
}

/// `STREAMOFF` never fails (review-m6 R6-4): V4L2 has it remove every buffer from the queue,
/// and a guest that cannot `STREAMOFF` cannot `REQBUFS(0)` either, so a flush or a clear the
/// backend cannot complete -- a wedged codec past its bound -- ends the session instead, and the
/// queue is reset, the buffers unqueued and the pool space freeable all the same.
#[test]
fn streamoff_never_fails_a_backend_that_will_not_flush() {
    // A seek whose flush times out.
    let mut r = rig();
    let mut s = session(&mut r.device);
    let sizeimage = start_streaming_320x240(&mut r, &mut s);
    poke_mmap_output(&mut s, 0, 0x05);
    r.device
        .qbuf(
            &mut s,
            mmap_buffer(OUTPUT, 0, 1 << 20),
            vec![],
            PayloadValidity::ALL,
        )
        .unwrap();
    collect_capture(&mut r, &mut s, 1);
    r.log.lock().unwrap().fail_flush = Some(libc::ETIMEDOUT);
    assert_eq!(
        r.device.streamoff(&mut s, OUTPUT),
        Ok(()),
        "STREAMOFF answers Ok"
    );
    assert!(s.dead, "and the session is over");
    assert_eq!(errors(&r.events.borrow()), 1);
    assert_eq!(r.log.lock().unwrap().stops, 1, "the backend was joined");
    assert!(!s.state.output_streaming);
    assert!(
        s.input
            .buffers
            .iter()
            .chain(s.output.buffers.iter())
            .all(|b| !b.queued && !b.lent),
        "every buffer of both queues is back"
    );
    // The guest can still free everything.
    r.device
        .reqbufs(&mut s, OUTPUT, MemoryType::Mmap, 0)
        .unwrap();
    r.device
        .reqbufs(&mut s, CAPTURE, MemoryType::Mmap, 0)
        .unwrap();
    assert!(s.input.buffers.is_empty() && s.output.buffers.is_empty());
    close(&mut r.device, s);

    // A CAPTURE reset whose clear fails.
    let mut r = rig();
    let mut s = session(&mut r.device);
    start_streaming_320x240(&mut r, &mut s);
    r.log.lock().unwrap().fail_clear = Some(libc::EIO);
    assert_eq!(r.device.streamoff(&mut s, CAPTURE), Ok(()));
    assert!(s.dead);
    assert!(!s.state.capture_streaming);
    assert!(s.output.buffers.iter().all(|b| !b.queued));
    // `REQBUFS(0)` goes through the (now trivially successful) STREAMOFF and frees.
    r.device
        .reqbufs(&mut s, CAPTURE, MemoryType::Mmap, 0)
        .unwrap();
    assert!(s.output.buffers.is_empty());
    let _ = sizeimage;
    close(&mut r.device, s);
}

/// A CAPTURE buffer the backend refuses ends the session from every caller of the lend, not
/// only from `QBUF` (review-m6 R6-6), and the buffer is unqueued with the rest rather than left
/// neither lent nor pending. `DEC_CMD_START` is the caller exercised: the buffer waits while
/// the decoder is stopped and is lent by the resume.
#[test]
fn a_refused_capture_buffer_ends_the_session_from_dec_cmd_start_too() {
    let mut r = rig();
    let mut s = session(&mut r.device);
    start_streaming_320x240(&mut r, &mut s);
    poke_mmap_output(&mut s, 0, 0x08);
    r.device
        .qbuf(
            &mut s,
            mmap_buffer(OUTPUT, 0, 1 << 20),
            vec![],
            PayloadValidity::ALL,
        )
        .unwrap();
    collect_capture(&mut r, &mut s, 1);
    let before = source_changes(&r.events.borrow());
    poke_mmap_output(&mut s, 1, DRC_LAST_MAGIC);
    r.device
        .qbuf(
            &mut s,
            mmap_buffer(OUTPUT, 1, 1 << 20),
            vec![],
            PayloadValidity::ALL,
        )
        .unwrap();
    while source_changes(&r.events.borrow()) == before {
        assert!(wait_ready(&s), "no SOURCE_CHANGE within 2s");
        process(&mut r.device, &mut s);
    }
    assert_eq!(s.drain, Drain::Stopped);
    let index = 0usize;
    let sizeimage = pix(&r.device.g_fmt(&s, CAPTURE).unwrap()).sizeimage;
    r.device
        .qbuf(
            &mut s,
            mmap_buffer(CAPTURE, index as u32, sizeimage),
            vec![],
            PayloadValidity::ALL,
        )
        .unwrap();
    assert_eq!(s.output.pending, [index]);

    r.log.lock().unwrap().fail_use_capture = Some(libc::EIO);
    assert_eq!(
        r.device
            .decoder_cmd(&mut s, dec_cmd(bindings::V4L2_DEC_CMD_START))
            .err(),
        Some(libc::EIO)
    );
    assert!(s.dead);
    assert_eq!(errors(&r.events.borrow()), 1);
    assert!(s.output.pending.is_empty());
    assert!(
        !s.output.buffers[index].queued && !s.output.buffers[index].lent,
        "the refused buffer went back with the rest"
    );
    close(&mut r.device, s);
}

/// The `MMAP` half of `prepare_buf_then_qbuf_keeps_the_prepared_payload` (review-m6 R6-7): a
/// host-owned OUTPUT buffer prepared with `bytesused = 5000` and queued with `bytesused = 0`
/// lends the backend 5000 bytes, not the whole buffer.
#[test]
fn prepare_buf_keeps_the_prepared_payload_for_an_mmap_buffer_too() {
    let mut r = rig();
    let mut s = session(&mut r.device);
    r.device
        .s_fmt(&mut s, OUTPUT, output_format(H264, 320, 240))
        .unwrap();
    r.device
        .reqbufs(&mut s, OUTPUT, MemoryType::Mmap, 1)
        .unwrap();

    let mut pb = mmap_buffer(OUTPUT, 0, 1 << 20);
    *pb.get_first_plane_mut().bytesused = 5000;
    let prepared = r
        .device
        .prepare_buf(&mut s, pb, vec![], PayloadValidity::ALL)
        .unwrap();
    assert_eq!(*prepared.get_first_plane().bytesused, 5000);
    let mut qb = mmap_buffer(OUTPUT, 0, 1 << 20);
    *qb.get_first_plane_mut().bytesused = 0;
    let queued = r
        .device
        .qbuf(&mut s, qb, vec![], PayloadValidity::ALL)
        .unwrap();
    assert_eq!(
        *queued.get_first_plane().bytesused,
        5000,
        "the prepared bytesused survived"
    );

    r.device.streamon(&mut s, OUTPUT).unwrap();
    assert_eq!(
        r.log.lock().unwrap().decoded_lens,
        vec![5000],
        "the backend was lent the prepared payload, not the buffer"
    );
    close(&mut r.device, s);
}

/// `ENUM_FRAMESIZES` for NV12 follows the coded format `S_FMT(OUTPUT)` selected, not always the
/// first one offered (review-m6 R6-16).
#[test]
fn enum_framesizes_for_nv12_follows_the_selected_coded_format() {
    let mut r = rig();
    let mut s = session(&mut r.device);
    let max_width = |r: &mut Rig, s: &Session| {
        let fs = r.device.enum_framesizes(s, 0, NV12.to_u32()).unwrap();
        // SAFETY: stepwise.
        unsafe { fs.__bindgen_anon_1.stepwise }.max_width
    };
    assert_eq!(max_width(&mut r, &s), 4096, "H264's range before any S_FMT");
    r.device
        .s_fmt(&mut s, OUTPUT, output_format(VP90, 640, 480))
        .unwrap();
    assert_eq!(
        max_width(&mut r, &s),
        2048,
        "VP9's range once VP9 is selected"
    );
    close(&mut r.device, s);
}

/// A drain the backend refuses leaves no drain pending (review-m6 R6-6 second half): the
/// session is not left waiting for a `LAST` buffer no drain will produce.
#[test]
fn a_refused_drain_leaves_no_drain_pending() {
    let mut r = rig();
    let mut s = session(&mut r.device);
    start_streaming_320x240(&mut r, &mut s);
    r.log.lock().unwrap().fail_drain = Some(libc::EIO);
    assert_eq!(
        r.device
            .decoder_cmd(&mut s, dec_cmd(bindings::V4L2_DEC_CMD_STOP))
            .err(),
        Some(libc::EIO)
    );
    assert_eq!(s.drain, Drain::None);
    // The next STOP is a real one.
    r.device
        .decoder_cmd(&mut s, dec_cmd(bindings::V4L2_DEC_CMD_STOP))
        .unwrap();
    assert_eq!(s.drain, Drain::Pending);
    // Collect until the drain's LAST buffer appears -- not just the first CAPTURE dequeue, which
    // an ordinary frame or the grace thread can beat under load (D63).
    collect_until_last(&mut r, &mut s);
    assert!(
        dequeued_on(&r.events.borrow(), CAPTURE)
            .iter()
            .any(|b| b.flags().contains(BufferFlags::LAST)),
        "the drain returned a LAST buffer"
    );
    assert_eq!(eos_events(&r.events.borrow()), 1);
    close(&mut r.device, s);
}

/// `nv12_sizeimage` saturates rather than aborting the helper on a dimension the codec store
/// published (review-m6 R6-8): `u32::MAX` bytes is a buffer the allocator refuses.
#[test]
fn nv12_sizeimage_saturates_instead_of_aborting() {
    assert_eq!(nv12_sizeimage(320, 240), 320 * 240 * 3 / 2);
    assert_eq!(nv12_sizeimage(3, 3), 9 + 2 * 2 * 2);
    assert_eq!(nv12_sizeimage(65_536, 65_536), u32::MAX);
    assert_eq!(nv12_sizeimage(u32::MAX, u32::MAX), u32::MAX);
    assert_eq!(nv12_sizeimage(u32::MAX, 1), u32::MAX);
}

/// D45: the initial `SOURCE_CHANGE` reaches the guest **before** the OUTPUT (bitstream) buffer
/// that produced it comes back. The B8 `pollrace.py` measurement showed the MediaCodec backend
/// returning the OUTPUT buffer 16--19 ms *before* it emitted `SOURCE_CHANGE`; a GStreamer client
/// that dequeued it emptied its OUTPUT queue and took `POLLPRI|POLLERR` on the still-idle CAPTURE
/// side, and discarded the event. With the backend holding `InputBufferDone` until the
/// announcement is queued, the device sends the `SOURCE_CHANGE` event first and the OUTPUT
/// `DQBUF` second -- `POLLPRI` before `POLLOUT`. A client that has queued no CAPTURE buffer yet
/// still gets the event (it is a device event, not a buffer).
#[test]
fn source_change_precedes_the_output_buffer_that_produced_it() {
    let mut r = rig();
    let mut s = session(&mut r.device);
    r.device.s_fmt(&mut s, OUTPUT, output_format(H264, 320, 240)).unwrap();
    r.device.reqbufs(&mut s, OUTPUT, MemoryType::Mmap, 4).unwrap();
    r.device
        .subscribe_event(&mut s, EventType::SourceChange(0), SubscribeEventFlags::empty())
        .unwrap();
    // One OUTPUT buffer, no CAPTURE buffer queued yet: exactly the window pollrace measured.
    poke_mmap_output(&mut s, 0, 0x01);
    r.device
        .qbuf(&mut s, mmap_buffer(OUTPUT, 0, 1 << 20), vec![], PayloadValidity::ALL)
        .unwrap();
    r.device.streamon(&mut s, OUTPUT).unwrap();
    while source_changes(&r.events.borrow()) == 0 {
        assert!(wait_ready(&s), "no SOURCE_CHANGE within 2s");
        process(&mut r.device, &mut s);
    }

    let events = r.events.borrow();
    let src_at = events.iter().position(|e| {
        matches!(e, V4l2Event::Event(se) if se.event().type_ == bindings::V4L2_EVENT_SOURCE_CHANGE)
    });
    let out_at = events.iter().position(|e| {
        matches!(e, V4l2Event::DequeueBuffer(d) if d.v4l2_buffer().queue() == OUTPUT)
    });
    let src_at = src_at.expect("a SOURCE_CHANGE event");
    // The OUTPUT buffer may not have come back in the same batch, but if it has it must be after.
    if let Some(out_at) = out_at {
        assert!(
            src_at < out_at,
            "SOURCE_CHANGE (idx {src_at}) must precede the OUTPUT DQBUF (idx {out_at}): POLLPRI before POLLOUT"
        );
    }
    // No CAPTURE buffer was ever queued, yet the client saw the event.
    assert_eq!(source_changes(&events), 1);
    drop(events);
    close(&mut r.device, s);
}

/// The position of the OUTPUT `DQBUF` for a given buffer index in the accumulated event log.
fn output_dqbuf_at(events: &[V4l2Event], index: u32) -> Option<usize> {
    events.iter().position(|e| {
        matches!(e, V4l2Event::DequeueBuffer(d)
            if d.v4l2_buffer().queue() == OUTPUT && d.v4l2_buffer().index() == index)
    })
}

/// The position of the `n`-th (1-based) `SOURCE_CHANGE` event in the accumulated event log.
fn nth_source_change_at(events: &[V4l2Event], n: usize) -> Option<usize> {
    events
        .iter()
        .enumerate()
        .filter(|(_, e)| {
            matches!(e, V4l2Event::Event(se)
                if se.event().type_ == bindings::V4L2_EVENT_SOURCE_CHANGE)
        })
        .map(|(i, _)| i)
        .nth(n - 1)
}

/// D55/D56 (a): a backend that never announces a format holds the `InputBufferDone` and returns it
/// only once [`ANNOUNCE_GRACE`] passes, never on the codec's bare input-slot recycle. This is the
/// one-buffer-in-flight case (a stateful ffmpeg on an mp4 whose first packet is a 31-byte header,
/// D44; a `v4l2-compliance -s` feeding undecodable bytes): the buffer must come back, or the
/// client deadlocks -- but on the *clock*, not on a codec event, because a codec that goes silent
/// after consuming the garbage emits no callback to drive an event-driven release (D56, the
/// residue of D28/D48). Here a short grace is injected so the test need not wait the production
/// 250 ms; the point is the buffer is held for at least that grace and comes back with **no**
/// `SOURCE_CHANGE` ever.
#[test]
fn a_held_input_buffer_returns_after_the_grace_when_the_codec_never_announces() {
    let grace = Duration::from_millis(40);
    let mut r = rig_grace(grace);
    let mut s = session(&mut r.device);
    r.device.s_fmt(&mut s, OUTPUT, output_format(H264, 320, 240)).unwrap();
    r.device.reqbufs(&mut s, OUTPUT, MemoryType::Mmap, 4).unwrap();
    r.device
        .subscribe_event(&mut s, EventType::SourceChange(0), SubscribeEventFlags::empty())
        .unwrap();

    // Buffer 0 carries bytes the backend cannot announce a format from: it is HELD, not returned
    // on the input-slot recycle.
    poke_mmap_output(&mut s, 0, NO_ANNOUNCE_MAGIC);
    let mut ob = mmap_buffer(OUTPUT, 0, 1 << 20);
    ob.set_timestamp(ts(1));
    r.device.qbuf(&mut s, ob, vec![], PayloadValidity::ALL).unwrap();
    let start = Instant::now();
    r.device.streamon(&mut s, OUTPUT).unwrap();
    while dequeued_on(&r.events.borrow(), OUTPUT).is_empty() {
        assert!(wait_ready(&s), "no OUTPUT DQBUF within 2s");
        process(&mut r.device, &mut s);
    }
    // It was held for at least the grace (the hold began after `start`, so the release is strictly
    // later than `start + grace`), and it came back with no announcement -- freed on the clock.
    assert!(
        start.elapsed() >= grace,
        "the buffer must be held for the grace, not returned on the recycle"
    );
    assert_eq!(dequeued_on(&r.events.borrow(), OUTPUT).len(), 1, "buffer 0 returned after the grace");
    assert_eq!(source_changes(&r.events.borrow()), 0, "the codec never announced");
    close(&mut r.device, s);
}

/// D64: the grace is one-shot per pending-format window. A client that keeps feeding after the
/// first held buffer comes back (ffmpeg, one `OUTPUT` buffer in flight, re-queuing it) must not be
/// re-throttled to one buffer per grace: only the *first* buffer needs holding for gst's ordering
/// (gst queues one and stops), and throttling ffmpeg for a whole 7.5 s announce delay under encode
/// contention starved the codec into dropping a mid-stream band of ~30 pictures (30 grace lines ==
/// 30 lost frames). So the first buffer is held for the grace, and every buffer after the grace
/// fires (with no announcement) comes back at full rate.
#[test]
fn a_never_announcing_codec_feeds_a_one_buffer_client_at_full_rate_after_the_first_grace() {
    let grace = Duration::from_millis(100);
    let mut r = rig_grace(grace);
    let mut s = session(&mut r.device);
    r.device.s_fmt(&mut s, OUTPUT, output_format(H264, 320, 240)).unwrap();
    r.device.reqbufs(&mut s, OUTPUT, MemoryType::Mmap, 4).unwrap();
    r.device
        .subscribe_event(&mut s, EventType::SourceChange(0), SubscribeEventFlags::empty())
        .unwrap();

    // One OUTPUT buffer in flight (ffmpeg-shaped), re-queued each time it comes back. The first
    // one carries bytes the codec cannot announce from: it is held for the grace.
    poke_mmap_output(&mut s, 0, NO_ANNOUNCE_MAGIC);
    r.device
        .qbuf(&mut s, mmap_buffer(OUTPUT, 0, 1 << 20), vec![], PayloadValidity::ALL)
        .unwrap();
    let start = Instant::now();
    r.device.streamon(&mut s, OUTPUT).unwrap();
    while dequeued_on(&r.events.borrow(), OUTPUT).is_empty() {
        assert!(wait_ready(&s), "no first OUTPUT DQBUF within 2s");
        process(&mut r.device, &mut s);
    }
    assert!(start.elapsed() >= grace, "the first buffer is held for the grace");

    // Now keep feeding the same one buffer. After the first grace has fired without an
    // announcement, each re-queue must come back at full rate. Six re-queues that were each
    // throttled would take at least 6*grace; assert all six come back inside a single grace.
    let feeds = 6usize;
    let resume = Instant::now();
    for _ in 0..feeds {
        let have = dequeued_on(&r.events.borrow(), OUTPUT).len();
        poke_mmap_output(&mut s, 0, NO_ANNOUNCE_MAGIC);
        r.device
            .qbuf(&mut s, mmap_buffer(OUTPUT, 0, 1 << 20), vec![], PayloadValidity::ALL)
            .unwrap();
        while dequeued_on(&r.events.borrow(), OUTPUT).len() <= have {
            assert!(wait_ready(&s), "no prompt InputBufferDone after the first grace");
            process(&mut r.device, &mut s);
        }
    }
    let elapsed = resume.elapsed();
    assert_eq!(
        dequeued_on(&r.events.borrow(), OUTPUT).len(),
        1 + feeds,
        "every fed buffer returned"
    );
    assert!(
        elapsed < grace,
        "after the first grace the client runs at full rate ({elapsed:?} for {feeds} feeds, grace {grace:?}); \
         re-throttling would take at least {feeds}*grace"
    );
    assert_eq!(source_changes(&r.events.borrow()), 0, "the codec never announced");
    close(&mut r.device, s);
}

/// D55 (b): the race the F12 fix opened. MediaCodec can recycle the input slot
/// (`onInputAvailable`) *before* it announces (`onOutputFormatChanged`); the F12 backend released
/// the held `InputBufferDone` on that recycle, which returned the OUTPUT buffer **before** the
/// `SOURCE_CHANGE` a GStreamer client was waiting for -- so `wait_for_src_ch` emptied its OUTPUT
/// queue and its CAPTURE poll took `POLLPRI|POLLERR` (4/20 DRC runs died, B10-acceptance §1.4).
/// With the recycle no longer a release point, the held buffer comes back only *after* the
/// announcement, so the `SOURCE_CHANGE` still precedes the OUTPUT `DQBUF` (`POLLPRI` before
/// `POLLOUT`) even though the codec recycled the slot first. Buffer 0 is one the codec cannot
/// announce from (its slot recycles with nothing announced); buffer 1 lets it announce.
#[test]
fn a_held_input_buffer_returns_after_the_source_change_when_the_codec_announces_late() {
    // A long grace, so the announcement -- not the clock -- is what ends the hold here.
    let mut r = rig();
    let mut s = session(&mut r.device);
    r.device.s_fmt(&mut s, OUTPUT, output_format(H264, 320, 240)).unwrap();
    r.device.reqbufs(&mut s, OUTPUT, MemoryType::Mmap, 4).unwrap();
    r.device
        .subscribe_event(&mut s, EventType::SourceChange(0), SubscribeEventFlags::empty())
        .unwrap();

    // Buffer 0: the slot recycles with nothing announced -- held, NOT returned yet.
    poke_mmap_output(&mut s, 0, NO_ANNOUNCE_MAGIC);
    let mut ob = mmap_buffer(OUTPUT, 0, 1 << 20);
    ob.set_timestamp(ts(1));
    r.device.qbuf(&mut s, ob, vec![], PayloadValidity::ALL).unwrap();
    r.device.streamon(&mut s, OUTPUT).unwrap();

    // Buffer 1: the codec can announce from it. Its arrival ends the hold -- after the event.
    poke_mmap_output(&mut s, 1, 0x01);
    let mut ob = mmap_buffer(OUTPUT, 1, 1 << 20);
    ob.set_timestamp(ts(2));
    r.device.qbuf(&mut s, ob, vec![], PayloadValidity::ALL).unwrap();
    while dequeued_on(&r.events.borrow(), OUTPUT).len() < 2 {
        assert!(wait_ready(&s), "both OUTPUT buffers back within 2s");
        process(&mut r.device, &mut s);
    }
    let events = r.events.borrow();
    assert_eq!(source_changes(&events), 1, "one announcement");
    let src_at = nth_source_change_at(&events, 1).expect("a SOURCE_CHANGE");
    let out0_at = output_dqbuf_at(&events, 0).expect("buffer 0's OUTPUT DQBUF");
    // The buffer whose slot recycled first still comes back AFTER the SOURCE_CHANGE (D55): the
    // recycle is no longer a release point.
    assert!(
        src_at < out0_at,
        "SOURCE_CHANGE (idx {src_at}) must precede the recycled buffer's OUTPUT DQBUF (idx {out0_at})"
    );
    drop(events);
    close(&mut r.device, s);
}

/// D55 (c): the same race at a mid-stream resolution change. A parameter-set-only buffer
/// (`CODEC_CONFIG`) carries the new SPS; MediaCodec may recycle its input slot before it announces
/// the second `SOURCE_CHANGE`. The backend holds that `InputBufferDone` (`android.rs`
/// `awaiting_drc`) until the change is out, so a GStreamer client waiting in `wait_for_src_ch` for
/// the second change is not emptied. The second `SOURCE_CHANGE` must precede the OUTPUT `DQBUF` of
/// the buffer that carried the new parameter sets.
#[test]
fn a_drc_input_buffer_returns_after_the_second_source_change() {
    let mut r = rig();
    let mut s = session(&mut r.device);
    let _ = start_streaming_320x240(&mut r, &mut s);
    let before = source_changes(&r.events.borrow());

    // A parameter-set-only buffer that raises the mid-stream change with the recycle-first race.
    poke_mmap_output(&mut s, 1, DRC_RACE_MAGIC);
    let mut ob = mmap_buffer(OUTPUT, 1, 1 << 20);
    ob.set_timestamp(ts(9));
    r.device.qbuf(&mut s, ob, vec![], PayloadValidity::ALL).unwrap();
    while output_dqbuf_at(&r.events.borrow(), 1).is_none()
        || source_changes(&r.events.borrow()) <= before
    {
        assert!(wait_ready(&s), "the second SOURCE_CHANGE and buffer 1 within 2s");
        process(&mut r.device, &mut s);
    }
    let events = r.events.borrow();
    assert_eq!(source_changes(&events), before + 1, "a second announcement");
    let src2_at = nth_source_change_at(&events, before + 1).expect("the second SOURCE_CHANGE");
    let out1_at = output_dqbuf_at(&events, 1).expect("buffer 1's OUTPUT DQBUF");
    assert!(
        src2_at < out1_at,
        "the second SOURCE_CHANGE (idx {src2_at}) must precede the new-SPS buffer's OUTPUT DQBUF (idx {out1_at})"
    );
    drop(events);
    close(&mut r.device, s);
}

/// D54: every fourcc the decoder and encoder advertise is spelled exactly as its `videodev2.h`
/// `V4L2_PIX_FMT_*` macro (`v4l2_fourcc(a,b,c,d) = a | b<<8 | c<<16 | d<<24`), so
/// `v4l2-compliance`'s `determine_codec_mask` recognises every compressed OUTPUT format and
/// classifies the node as a stateful decoder (and any client matching on the fourcc can find the
/// format). AV1 was advertised as `AV10`, which is not a V4L2 format; `determine_codec_mask` bailed
/// on it and left the node unclassified, failing `testEvents` for the very control D29 gave it
/// (B10-acceptance §2.1). The fix is `AV01` = `v4l2_fourcc('A','V','0','1')` = `V4L2_PIX_FMT_AV1`.
#[test]
fn advertised_fourccs_match_videodev2_h() {
    let f = |s: &[u8; 4]| PixelFormat::from_fourcc(s);
    // The exact `V4L2_PIX_FMT_*` wire values from the guest's `videodev2.h`.
    assert_eq!(f(b"H264").to_u32(), 0x3436_3248, "V4L2_PIX_FMT_H264");
    assert_eq!(f(b"HEVC").to_u32(), 0x4356_4548, "V4L2_PIX_FMT_HEVC");
    assert_eq!(f(b"VP80").to_u32(), 0x3038_5056, "V4L2_PIX_FMT_VP8");
    assert_eq!(f(b"VP90").to_u32(), 0x3039_5056, "V4L2_PIX_FMT_VP9");
    assert_eq!(f(b"AV01").to_u32(), 0x3130_5641, "V4L2_PIX_FMT_AV1");
    assert_eq!(f(b"NV12").to_u32(), 0x3231_564e, "V4L2_PIX_FMT_NV12");

    // Each advertised fourcc has a real (non-`Unknown`) ENUM_FMT description, so compliance's
    // description check passes and `determine_codec_mask` recognises the compressed formats.
    for spelled in [b"H264", b"HEVC", b"VP80", b"VP90", b"AV01"] {
        assert_ne!(
            fourcc_description(f(spelled)),
            b"Unknown",
            "advertised fourcc {} must be a known V4L2 format",
            std::str::from_utf8(spelled).unwrap()
        );
    }
    assert_eq!(fourcc_description(f(b"AV01")), b"AV1");
    assert_eq!(fourcc_description(NV12), b"Y/UV 4:2:0");
    // The D54 bug spelling is NOT a V4L2 format: it must not be recognised (a guard against
    // regressing to `AV10`, which is what made compliance bail).
    assert_eq!(
        fourcc_description(f(b"AV10")),
        b"Unknown",
        "AV10 is not a V4L2 pixel format (D54)"
    );
}

/// D29: the decoder exposes `V4L2_CID_MIN_BUFFERS_FOR_CAPTURE` through the whole control
/// interface -- `QUERY_EXT_CTRL` / `QUERYCTRL` walk the two-entry list and end in `EINVAL`,
/// `QUERYMENU` is `EINVAL` (never `ENOTTY`), and `G_EXT_CTRLS` / `G_CTRL` read the value. On a
/// 6.15+ guest kernel `G_CTRL` arrives as `G_EXT_CTRLS`, which is why answering only `g_ctrl`
/// (the old code) left the control invisible.
#[test]
fn min_buffers_for_capture_is_enumerated_and_readable() {
    let mut r = rig();
    let mut s = session(&mut r.device);
    const CID_MIN_CAP: u32 = bindings::V4L2_CID_MIN_BUFFERS_FOR_CAPTURE;

    // The NEXT_CTRL walk starts at the User Controls class marker and reaches
    // MIN_BUFFERS_FOR_CAPTURE; past it the walk goes on into the codec class (VA1b's profile and
    // level menus, whose own walk `profile_and_level_menus_enumerate_what_the_backend_published`
    // checks end to end).
    let walk = |r: &mut Rig, s: &Session, from: u32| -> Result<u32, i32> {
        let (id, flags) =
            v4l2r::ioctl::parse_ctrl_id_and_flags(from | bindings::V4L2_CTRL_FLAG_NEXT_CTRL);
        r.device.query_ext_ctrl(s, id, flags).map(|q| q.id)
    };
    assert_eq!(walk(&mut r, &s, 0).unwrap(), bindings::V4L2_CID_USER_CLASS);
    assert_eq!(
        walk(&mut r, &s, bindings::V4L2_CID_USER_CLASS).unwrap(),
        CID_MIN_CAP
    );
    assert_eq!(
        walk(&mut r, &s, CID_MIN_CAP).unwrap(),
        bindings::V4L2_CID_CODEC_CLASS,
        "the user class ends at MIN_BUFFERS_FOR_CAPTURE; the codec class follows"
    );

    // QUERY_EXT_CTRL of the control itself: read-only integer, 1..=MAX_BUFFERS.
    let (id, flags) = v4l2r::ioctl::parse_ctrl_id_and_flags(CID_MIN_CAP);
    let q = r.device.query_ext_ctrl(&s, id, flags).unwrap();
    assert_eq!(q.type_, bindings::v4l2_ctrl_type_V4L2_CTRL_TYPE_INTEGER);
    assert_eq!((q.minimum, q.maximum, q.step), (1, MAX_BUFFERS as i64, 1));
    assert_ne!(q.flags & bindings::V4L2_CTRL_FLAG_READ_ONLY, 0, "read-only");
    let name: Vec<u8> = q.name.iter().take_while(|c| **c != 0).map(|c| *c as u8).collect();
    assert_eq!(&name, b"Min Number of Capture Buffers");

    // The old QUERYCTRL says the same and walks the same list.
    let (id, flags) = v4l2r::ioctl::parse_ctrl_id_and_flags(CID_MIN_CAP);
    let qc = r.device.queryctrl(&s, id, flags).unwrap();
    assert_eq!((qc.minimum, qc.maximum, qc.step), (1, MAX_BUFFERS as i32, 1));
    // QUERYMENU is EINVAL for the integer control, not ENOTTY.
    assert_eq!(r.device.querymenu(&s, CID_MIN_CAP, 0).map(|_| ()), Err(libc::EINVAL));
    // A control the decoder does not have: EINVAL, never ENOTTY.
    let (id, flags) = v4l2r::ioctl::parse_ctrl_id_and_flags(bindings::V4L2_CID_BRIGHTNESS);
    assert_eq!(r.device.query_ext_ctrl(&s, id, flags).map(|_| ()), Err(libc::EINVAL));

    // Before any SOURCE_CHANGE the value is the floor (1); after it, the backend's number.
    assert_eq!(r.device.g_ctrl(&s, CID_MIN_CAP).unwrap().value, 1);
    assert_eq!(g_ctrl_ext(&mut r, &mut s, CID_MIN_CAP), Ok(1));
    start_streaming_320x240(&mut r, &mut s);
    // The FakeDecoderBackend announces min_capture_buffers = 4.
    assert_eq!(r.device.g_ctrl(&s, CID_MIN_CAP).unwrap().value, 4);
    assert_eq!(g_ctrl_ext(&mut r, &mut s, CID_MIN_CAP), Ok(4), "via G_EXT_CTRLS");
    close(&mut r.device, s);
}

/// D69: the CAPTURE minimum the backend announces flows through to the control and the
/// `SOURCE_CHANGE`, whatever its value. A client that honours the announcement -- GStreamer sizes
/// its CAPTURE pool from `MIN_BUFFERS_FOR_CAPTURE`, and ffmpeg's `-num_capture_buffers` needs it
/// -- was told `min 4` while the codec needed 21 output slots, and undershooting cost frames in
/// silence (8 buffers -> 73/300, rc 0, B12-acceptance §2/§15). The MediaCodec backend now reads
/// `num-output-slots` and announces `max(4, slots)`; here the fake stands in for the codec by
/// announcing 21, and the device must carry it to both `G_CTRL(MIN_BUFFERS_FOR_CAPTURE)` (and its
/// `G_EXT_CTRLS` form) and the `SOURCE_CHANGE`. Before any format change the floor (1) still
/// stands, and the control is read-only and volatile.
#[test]
fn the_announced_capture_minimum_reaches_the_control() {
    let mut r = rig_announcing_min(21);
    let mut s = session(&mut r.device);
    const CID_MIN_CAP: u32 = bindings::V4L2_CID_MIN_BUFFERS_FOR_CAPTURE;

    // Before the codec has parsed the stream the value is the floor, not the codec's number.
    assert_eq!(r.device.g_ctrl(&s, CID_MIN_CAP).unwrap().value, 1);

    // The SOURCE_CHANGE the fake raises carries min 21, and the control now answers 21 both ways.
    start_streaming_320x240(&mut r, &mut s);
    assert_eq!(source_changes(&r.events.borrow()), 1, "one SOURCE_CHANGE");
    assert_eq!(
        r.device.g_ctrl(&s, CID_MIN_CAP).unwrap().value,
        21,
        "the codec's output-slot count reached MIN_BUFFERS_FOR_CAPTURE"
    );
    assert_eq!(g_ctrl_ext(&mut r, &mut s, CID_MIN_CAP), Ok(21), "via G_EXT_CTRLS");

    // Still read-only: a client cannot force it down and then under-provision without a refusal.
    assert_eq!(r.device.s_ctrl(&mut s, CID_MIN_CAP, 4).map(|_| ()), Err(libc::EACCES));
    assert_eq!(r.device.g_ctrl(&s, CID_MIN_CAP).unwrap().value, 21);
    close(&mut r.device, s);
}

/// D29, the write side: `MIN_BUFFERS_FOR_CAPTURE` is read-only, so `S_CTRL` / `S_EXT_CTRLS` /
/// `TRY_EXT_CTRLS` of it answer `EACCES`, and the class marker in `G_EXT_CTRLS` answers `EACCES`
/// too (it carries `WRITE_ONLY`) -- exactly the kernel's rules `v4l2-compliance` checks.
#[test]
fn decoder_controls_reject_writes_and_the_class_marker() {
    let mut r = rig();
    let mut s = session(&mut r.device);
    const CID_MIN_CAP: u32 = bindings::V4L2_CID_MIN_BUFFERS_FOR_CAPTURE;

    assert_eq!(r.device.s_ctrl(&mut s, CID_MIN_CAP, 8).map(|_| ()), Err(libc::EACCES));
    assert_eq!(
        r.device.s_ctrl(&mut s, bindings::V4L2_CID_USER_CLASS, 0).map(|_| ()),
        Err(libc::EINVAL),
        "the class marker is not an int"
    );
    assert_eq!(
        r.device.s_ctrl(&mut s, bindings::V4L2_CID_BRIGHTNESS, 0).map(|_| ()),
        Err(libc::EINVAL)
    );

    // S/TRY_EXT_CTRLS of the read-only control: EACCES, error_idx = count (1).
    let mut ctrls = ext_controls_current(1);
    let mut arr = vec![ext_ctrl(CID_MIN_CAP, 8)];
    assert_eq!(
        r.device.s_ext_ctrls(&mut s, CtrlWhich::Current, &mut ctrls, &mut arr, vec![]),
        Err(libc::EACCES)
    );
    assert_eq!(ctrls.error_idx, 1);
    let mut ctrls = ext_controls_current(1);
    let mut arr = vec![ext_ctrl(CID_MIN_CAP, 8)];
    assert_eq!(
        r.device.try_ext_ctrls(&s, CtrlWhich::Current, &mut ctrls, &mut arr, vec![]),
        Err(libc::EACCES),
        "TRY names the failing control"
    );
    assert_eq!(ctrls.error_idx, 0, "the first (only) control, index 0");

    // G_EXT_CTRLS of the class marker: EACCES (WRITE_ONLY), unlike G_CTRL of it (EINVAL).
    let mut ctrls = ext_controls_current(1);
    let mut arr = vec![ext_ctrl(bindings::V4L2_CID_USER_CLASS, 0)];
    assert_eq!(
        r.device.g_ext_ctrls(&s, CtrlWhich::Current, &mut ctrls, &mut arr, vec![]),
        Err(libc::EACCES)
    );
    assert_eq!(
        r.device.g_ctrl(&s, bindings::V4L2_CID_USER_CLASS).map(|_| ()),
        Err(libc::EINVAL)
    );
    close(&mut r.device, s);
}

/// VA1b (`VPU_DESIGN.md` §7.6 item 2): the decoder's profile and level menus are the host codec
/// store's, item for item. The fake publishes H.264 {Baseline, Constrained Baseline, Main, High,
/// Constrained High} with levels {4, 4.1, 5, 5.1}, HEVC {Main, Main 10} with levels {5.1, 6.2}, and
/// VP9 {0} with no levels at all; the device must enumerate exactly that -- the codec class
/// marker, one profile control per format, a level control only where the backend gave levels --
/// with the kernel's own menu values and names, and answer `EINVAL` for every value in between.
/// The holes are the point: `v4l2-ctl -L`, GStreamer and a VA driver all learn the supported set
/// by walking the range and keeping what `QUERYMENU` accepts, so a menu that answered for an
/// unsupported profile would advertise a codec the host does not have.
#[test]
fn profile_and_level_menus_enumerate_what_the_backend_published() {
    let mut r = rig();
    let s = session(&mut r.device);

    // The whole NEXT_CTRL walk, in id order: the two user controls (D29), then the codec class
    // marker and one control per (format, kind) the backend published. VP9 published no levels,
    // so there is no VP9 level control -- and there is no control for a format the fake does not
    // offer at all (AV1).
    let mut expected = vec![
        bindings::V4L2_CID_USER_CLASS,
        bindings::V4L2_CID_MIN_BUFFERS_FOR_CAPTURE,
        bindings::V4L2_CID_CODEC_CLASS,
        CID_H264_PROFILE,
        CID_H264_LEVEL,
        CID_HEVC_PROFILE,
        CID_HEVC_LEVEL,
        CID_VP9_PROFILE,
    ];
    expected.sort_unstable();
    assert_eq!(walk_controls(&mut r, &s), expected, "the NEXT_CTRL walk");
    for absent in [
        bindings::V4L2_CID_MPEG_VIDEO_VP9_LEVEL,
        bindings::V4L2_CID_MPEG_VIDEO_AV1_PROFILE,
    ] {
        let (id, flags) = v4l2r::ioctl::parse_ctrl_id_and_flags(absent);
        assert_eq!(
            r.device.query_ext_ctrl(&s, id, flags).map(|_| ()),
            Err(libc::EINVAL),
            "no control for what the backend published nothing for ({absent:#x})"
        );
    }

    // Each menu is a read-only V4L2_CTRL_TYPE_MENU whose bounds are its lowest and highest item
    // (so QUERYMENU of either end always finds one, which v4l2-compliance checks) and whose
    // default is the first value the backend listed.
    let q = query_ext(&mut r, &s, CID_H264_PROFILE);
    assert_eq!(q.type_, bindings::v4l2_ctrl_type_V4L2_CTRL_TYPE_MENU);
    assert_eq!((q.minimum, q.maximum, q.step), (0, 17, 1));
    assert_eq!(q.default_value, 0, "the first profile the backend listed");
    assert_ne!(q.flags & bindings::V4L2_CTRL_FLAG_READ_ONLY, 0, "read-only");
    assert_eq!(
        q.flags & bindings::V4L2_CTRL_FLAG_VOLATILE,
        0,
        "a codec's profiles do not change with the stream, unlike MIN_BUFFERS_FOR_CAPTURE"
    );
    assert_eq!(ext_ctrl_name(&q), "H264 Profile", "the kernel's own name");

    // The H.264 profile menu: exactly the five published, with the kernel's names. Extended (3)
    // and the High 4xx family (5..16) are holes the codec does not support.
    assert_eq!(
        menu_items(&mut r, &s, CID_H264_PROFILE),
        vec![
            (0, "Baseline".to_string()),
            (1, "Constrained Baseline".to_string()),
            (2, "Main".to_string()),
            (4, "High".to_string()),
            (17, "Constrained High".to_string()),
        ]
    );
    for hole in [3, 5, 9, 16] {
        assert_eq!(
            r.device.querymenu(&s, CID_H264_PROFILE, hole).map(|_| ()),
            Err(libc::EINVAL),
            "profile {hole} is not supported, so it is not a menu item"
        );
    }
    // Past the enum entirely, and past the control's own range.
    assert_eq!(
        r.device.querymenu(&s, CID_H264_PROFILE, 18).map(|_| ()),
        Err(libc::EINVAL)
    );
    assert_eq!(
        r.device.querymenu(&s, CID_H264_PROFILE, 4096).map(|_| ()),
        Err(libc::EINVAL)
    );

    // The H.264 level menu: the four published, the highest first in the backend's list, so the
    // default is 5.1 -- what a decoder's level conveys is the most it can decode.
    let q = query_ext(&mut r, &s, CID_H264_LEVEL);
    assert_eq!((q.minimum, q.maximum, q.default_value), (11, 15, 15));
    assert_eq!(ext_ctrl_name(&q), "H264 Level");
    assert_eq!(
        menu_items(&mut r, &s, CID_H264_LEVEL),
        vec![
            (11, "4".to_string()),
            (12, "4.1".to_string()),
            (14, "5".to_string()),
            (15, "5.1".to_string()),
        ]
    );
    assert_eq!(
        r.device.querymenu(&s, CID_H264_LEVEL, 13).map(|_| ()),
        Err(libc::EINVAL),
        "4.2 is between two supported levels and is still a hole"
    );

    // HEVC: Main and Main 10, with Main Still Picture the hole in between; its levels too.
    assert_eq!(
        menu_items(&mut r, &s, CID_HEVC_PROFILE),
        vec![(0, "Main".to_string()), (2, "Main 10".to_string())]
    );
    assert_eq!(
        r.device.querymenu(&s, CID_HEVC_PROFILE, 1).map(|_| ()),
        Err(libc::EINVAL)
    );
    assert_eq!(
        menu_items(&mut r, &s, CID_HEVC_LEVEL),
        vec![(8, "5.1".to_string()), (12, "6.2".to_string())]
    );

    // VP9: one profile, and QUERYMENU of the old integer control is still EINVAL (D29).
    assert_eq!(
        menu_items(&mut r, &s, CID_VP9_PROFILE),
        vec![(0, "0".to_string())]
    );
    assert_eq!(
        r.device
            .querymenu(&s, bindings::V4L2_CID_MIN_BUFFERS_FOR_CAPTURE, 0)
            .map(|_| ()),
        Err(libc::EINVAL),
        "an integer control has no menu"
    );

    // The legacy QUERYCTRL says the same about a menu as QUERY_EXT_CTRL does.
    let (id, flags) = v4l2r::ioctl::parse_ctrl_id_and_flags(CID_H264_PROFILE);
    let qc = r.device.queryctrl(&s, id, flags).unwrap();
    assert_eq!(qc.type_, bindings::v4l2_ctrl_type_V4L2_CTRL_TYPE_MENU);
    assert_eq!((qc.minimum, qc.maximum, qc.default_value), (0, 17, 0));
    close(&mut r.device, s);
}

/// VA1b: what a client reads off a profile or level control is always one of the values the host
/// codec reported supporting, through every path -- `G_CTRL`, the `G_EXT_CTRLS` form a 6.15+ guest
/// kernel turns it into, and `V4L2_CTRL_WHICH_DEF_VAL` -- and the menus stay read-only, because a
/// stateful decoder selects nothing: the profile and level of what it decodes are in the
/// bitstream.
#[test]
fn profile_and_level_controls_read_a_supported_value_and_refuse_writes() {
    let mut r = rig();
    let mut s = session(&mut r.device);

    for (id, expected) in [(CID_H264_PROFILE, 0), (CID_H264_LEVEL, 15), (CID_VP9_PROFILE, 0)] {
        let value = r.device.g_ctrl(&s, id).unwrap().value;
        assert_eq!(value, expected);
        assert_eq!(g_ctrl_ext(&mut r, &mut s, id), Ok(expected), "via G_EXT_CTRLS");
        assert!(
            menu_items(&mut r, &s, id).iter().any(|i| i.0 == value),
            "G_CTRL({id:#x}) answered {value}, which is not one of its menu items"
        );
    }

    // G_EXT_CTRLS with WHICH_DEF_VAL answers the same default.
    let mut ctrls = ext_controls_current(1);
    let mut arr = vec![ext_ctrl(CID_H264_LEVEL, 0)];
    r.device
        .g_ext_ctrls(&s, CtrlWhich::Default, &mut ctrls, &mut arr, vec![])
        .unwrap();
    // SAFETY: a plain value control.
    assert_eq!(unsafe { arr[0].__bindgen_anon_1.value }, 15);

    // Read-only: S_CTRL is EACCES, and so are S/TRY_EXT_CTRLS, with the kernel's error_idx.
    assert_eq!(
        r.device.s_ctrl(&mut s, CID_H264_PROFILE, 2).map(|_| ()),
        Err(libc::EACCES),
        "a supported value is refused too: the control is read-only"
    );
    assert_eq!(
        r.device.s_ctrl(&mut s, CID_H264_PROFILE, 3).map(|_| ()),
        Err(libc::EACCES),
        "read-only is checked before the value (the kernel's order)"
    );
    let mut ctrls = ext_controls_current(1);
    let mut arr = vec![ext_ctrl(CID_HEVC_PROFILE, 0)];
    assert_eq!(
        r.device.try_ext_ctrls(&s, CtrlWhich::Current, &mut ctrls, &mut arr, vec![]),
        Err(libc::EACCES)
    );
    assert_eq!(ctrls.error_idx, 0, "TRY names the failing control");
    assert_eq!(r.device.g_ctrl(&s, CID_H264_PROFILE).unwrap().value, 0);

    // The codec class marker follows the same rules as the user one: EINVAL for G_CTRL / S_CTRL
    // (it is not an int), EACCES for G_EXT_CTRLS (it carries WRITE_ONLY).
    assert_eq!(
        r.device.g_ctrl(&s, bindings::V4L2_CID_CODEC_CLASS).map(|_| ()),
        Err(libc::EINVAL)
    );
    assert_eq!(
        r.device.s_ctrl(&mut s, bindings::V4L2_CID_CODEC_CLASS, 0).map(|_| ()),
        Err(libc::EINVAL)
    );
    let mut ctrls = ext_controls_current(1);
    let mut arr = vec![ext_ctrl(bindings::V4L2_CID_CODEC_CLASS, 0)];
    assert_eq!(
        r.device.g_ext_ctrls(&s, CtrlWhich::Current, &mut ctrls, &mut arr, vec![]),
        Err(libc::EACCES)
    );

    // A control event can be subscribed for a menu, and its initial value is that same supported
    // value (D50: v4l2-compliance's testEvents subscribes to every control it enumerated). The
    // codec class marker is accepted too and carries no initial value.
    r.device
        .subscribe_event(
            &mut s,
            EventType::Ctrl(CID_H264_PROFILE),
            SubscribeEventFlags::SEND_INITIAL,
        )
        .unwrap();
    r.device
        .subscribe_event(
            &mut s,
            EventType::Ctrl(bindings::V4L2_CID_CODEC_CLASS),
            SubscribeEventFlags::SEND_INITIAL,
        )
        .unwrap();
    assert_eq!(
        ctrl_events(&r.events.borrow()),
        vec![(CID_H264_PROFILE, 0)],
        "one initial event, carrying the menu's value"
    );
    close(&mut r.device, s);
}

/// VA1b's floor: a coded format the host codec store said nothing about gets **no** profile
/// control -- the device never invents a list. With no format carrying profile data the codec
/// control class disappears entirely and the control interface is what D29 left it; with only
/// some formats carrying it, the ones that do get their menus and the ones that do not are
/// absent, which is exactly how a client tells "this decoder does not publish its profiles" from
/// "this decoder does not support that profile".
#[test]
fn a_format_the_store_said_nothing_about_gets_no_profile_control() {
    let mut r = rig_with_caps(caps_without_profiles());
    let s = session(&mut r.device);

    assert_eq!(
        walk_controls(&mut r, &s),
        vec![
            bindings::V4L2_CID_USER_CLASS,
            bindings::V4L2_CID_MIN_BUFFERS_FOR_CAPTURE,
        ],
        "no profile data anywhere: no codec class, no codec controls"
    );
    for id in [CID_H264_PROFILE, CID_H264_LEVEL, CID_HEVC_PROFILE, CID_VP9_PROFILE] {
        let (qid, flags) = v4l2r::ioctl::parse_ctrl_id_and_flags(id);
        assert_eq!(r.device.query_ext_ctrl(&s, qid, flags).map(|_| ()), Err(libc::EINVAL));
        assert_eq!(r.device.querymenu(&s, id, 0).map(|_| ()), Err(libc::EINVAL));
        assert_eq!(r.device.g_ctrl(&s, id).map(|_| ()), Err(libc::EINVAL));
    }
    // The decoder still decodes: the formats are offered, they just describe nothing.
    assert_eq!(r.device.capabilities().coded_formats.len(), 3);
    close(&mut r.device, s);

    // One format with data, one without: H.264 loses both its controls, HEVC keeps both, and the
    // codec class marker stays because something is in that class.
    let mut mixed = caps();
    mixed.coded_formats[0].profiles.clear();
    mixed.coded_formats[0].levels.clear();
    let mut r = rig_with_caps(mixed);
    let s = session(&mut r.device);
    let mut expected = vec![
        bindings::V4L2_CID_USER_CLASS,
        bindings::V4L2_CID_MIN_BUFFERS_FOR_CAPTURE,
        bindings::V4L2_CID_CODEC_CLASS,
        CID_HEVC_PROFILE,
        CID_HEVC_LEVEL,
        CID_VP9_PROFILE,
    ];
    expected.sort_unstable();
    assert_eq!(walk_controls(&mut r, &s), expected);
    close(&mut r.device, s);

    // A format with profiles but no levels keeps its profile control and has no level control --
    // the VP9 case above, checked here on H.264, where a level control does exist for other
    // backends.
    let mut no_levels = caps();
    no_levels.coded_formats[0].levels.clear();
    let mut r = rig_with_caps(no_levels);
    let s = session(&mut r.device);
    assert_eq!(
        menu_items(&mut r, &s, CID_H264_PROFILE).len(),
        H264_PROFILES.len()
    );
    let (id, flags) = v4l2r::ioctl::parse_ctrl_id_and_flags(CID_H264_LEVEL);
    assert_eq!(
        r.device.query_ext_ctrl(&s, id, flags).map(|_| ()),
        Err(libc::EINVAL),
        "no levels published, no level control"
    );
    close(&mut r.device, s);
}

/// VA1b: AV1. `V4L2_CID_MPEG_VIDEO_AV1_PROFILE` is the one profile menu the encoder device has no
/// table for, and the kernel gives AV1 no level control a stateful decoder would fill, so an AV1
/// format gets a profile menu and nothing else. Its menu values and names are the kernel's
/// (`v4l2_ctrl_get_menu`'s `av1_profile[]` against `enum v4l2_mpeg_video_av1_profile` in
/// `v4l2-controls.h`: `MAIN = 0`, `HIGH = 1`, `PROFESSIONAL = 2`). A value the running kernel has
/// no name for is dropped rather than enumerated with an empty name.
#[test]
fn the_av1_profile_menu_comes_from_the_backend() {
    let range = SizeRange::new(16, 4096, 2);
    let av1_only = DecoderCapabilities {
        coded_formats: vec![CodedFormat {
            fourcc: AV01,
            width: range,
            height: range,
            dynamic_resolution: true,
            // Main (what `c2.qti.av1.decoder` reports for both `AV1ProfileMain8` and
            // `AV1ProfileMain10`: one AV1 `seq_profile`), plus a value past the kernel's enum.
            profiles: vec![0, 9],
            levels: Vec::new(),
        }],
    };
    let mut r = rig_with_caps(av1_only);
    let s = session(&mut r.device);

    let mut expected = vec![
        bindings::V4L2_CID_USER_CLASS,
        bindings::V4L2_CID_MIN_BUFFERS_FOR_CAPTURE,
        bindings::V4L2_CID_CODEC_CLASS,
        bindings::V4L2_CID_MPEG_VIDEO_AV1_PROFILE,
    ];
    expected.sort_unstable();
    assert_eq!(walk_controls(&mut r, &s), expected);

    let id = bindings::V4L2_CID_MPEG_VIDEO_AV1_PROFILE;
    assert_eq!(
        menu_items(&mut r, &s, id),
        vec![(0, "Main".to_string())],
        "the profile past the kernel's enum was dropped"
    );
    let q = query_ext(&mut r, &s, id);
    assert_eq!(ext_ctrl_name(&q), "AV1 Profile");
    assert_eq!((q.minimum, q.maximum, q.default_value), (0, 0, 0));
    assert_eq!(r.device.g_ctrl(&s, id).unwrap().value, 0);
    for absent in [1, 2] {
        assert_eq!(
            r.device.querymenu(&s, id, absent).map(|_| ()),
            Err(libc::EINVAL),
            "AV1 profile {absent} is not supported by this codec"
        );
    }
    assert_eq!(
        r.device
            .query_ext_ctrl(
                &s,
                v4l2r::ioctl::parse_ctrl_id_and_flags(bindings::V4L2_CID_MPEG_VIDEO_AV1_LEVEL).0,
                v4l2r::ioctl::parse_ctrl_id_and_flags(bindings::V4L2_CID_MPEG_VIDEO_AV1_LEVEL).1,
            )
            .map(|_| ()),
        Err(libc::EINVAL),
        "no AV1 level control"
    );
    close(&mut r.device, s);
}

// helpers used by several tests ---------------------------------------------------------------

const CID_H264_PROFILE: u32 = bindings::V4L2_CID_MPEG_VIDEO_H264_PROFILE;
const CID_H264_LEVEL: u32 = bindings::V4L2_CID_MPEG_VIDEO_H264_LEVEL;
const CID_HEVC_PROFILE: u32 = bindings::V4L2_CID_MPEG_VIDEO_HEVC_PROFILE;
const CID_HEVC_LEVEL: u32 = bindings::V4L2_CID_MPEG_VIDEO_HEVC_LEVEL;
const CID_VP9_PROFILE: u32 = bindings::V4L2_CID_MPEG_VIDEO_VP9_PROFILE;

/// Every control id the `V4L2_CTRL_FLAG_NEXT_CTRL` walk visits, in order -- what `v4l2-ctl -L`
/// and `v4l2-compliance` enumerate the device's controls with.
fn walk_controls(r: &mut Rig, s: &Session) -> Vec<u32> {
    let mut out = Vec::new();
    let mut from = 0u32;
    loop {
        let (id, flags) =
            v4l2r::ioctl::parse_ctrl_id_and_flags(from | bindings::V4L2_CTRL_FLAG_NEXT_CTRL);
        match r.device.query_ext_ctrl(s, id, flags) {
            Ok(q) => {
                assert!(q.id > from, "the walk must advance");
                out.push(q.id);
                from = q.id;
            }
            Err(e) => {
                assert_eq!(e, libc::EINVAL, "the walk ends in EINVAL");
                return out;
            }
        }
        assert!(out.len() < 64, "the walk does not terminate");
    }
}

/// `QUERY_EXT_CTRL` of one control by id.
fn query_ext(r: &mut Rig, s: &Session, id: u32) -> bindings::v4l2_query_ext_ctrl {
    let (id, flags) = v4l2r::ioctl::parse_ctrl_id_and_flags(id);
    r.device.query_ext_ctrl(s, id, flags).unwrap()
}

/// The `name` a `v4l2_query_ext_ctrl` carries (a `[c_char; 32]`).
fn ext_ctrl_name(q: &bindings::v4l2_query_ext_ctrl) -> String {
    let bytes: Vec<u8> = q.name.iter().take_while(|c| **c != 0).map(|c| *c as u8).collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

/// Every `(value, name)` `QUERYMENU` accepts over a menu control's whole range: the supported set
/// as a client discovers it.
fn menu_items(r: &mut Rig, s: &Session, id: u32) -> Vec<(i32, String)> {
    let q = query_ext(r, s, id);
    (q.minimum..=q.maximum)
        .filter_map(|v| {
            let m = r.device.querymenu(s, id, v as u32).ok()?;
            // SAFETY: `name` is the member a menu (not integer-menu) control fills.
            let name = unsafe { m.__bindgen_anon_1.name };
            let bytes: Vec<u8> = name.iter().take_while(|c| **c != 0).copied().collect();
            Some((v as i32, String::from_utf8_lossy(&bytes).into_owned()))
        })
        .collect()
}

/// A `v4l2_ext_control` for a plain (value) control.
fn ext_ctrl(id: u32, value: i32) -> bindings::v4l2_ext_control {
    bindings::v4l2_ext_control {
        id,
        size: 0,
        reserved2: [0],
        __bindgen_anon_1: bindings::v4l2_ext_control__bindgen_ty_1 { value },
    }
}

/// A `v4l2_ext_controls` header for `count` controls (the `which` the handler acts on is passed
/// to the ioctl separately, so the union value here is immaterial).
fn ext_controls_current(count: u32) -> bindings::v4l2_ext_controls {
    bindings::v4l2_ext_controls {
        __bindgen_anon_1: bindings::v4l2_ext_controls__bindgen_ty_1 {
            ctrl_class: bindings::V4L2_CTRL_WHICH_CUR_VAL,
        },
        count,
        error_idx: 0,
        request_fd: 0,
        reserved: [0],
        controls: std::ptr::null_mut(),
    }
}

/// Read one control through `G_EXT_CTRLS`, the path a 6.15+ guest kernel turns `G_CTRL` into.
fn g_ctrl_ext(r: &mut Rig, s: &mut Session, id: u32) -> Result<i32, i32> {
    let mut ctrls = ext_controls_current(1);
    let mut arr = vec![ext_ctrl(id, 0)];
    r.device
        .g_ext_ctrls(s, CtrlWhich::Current, &mut ctrls, &mut arr, vec![])
        .map(|()| {
            let anon = arr[0].__bindgen_anon_1;
            // SAFETY: plain value control.
            unsafe { anon.value }
        })
}

fn dec_cmd(cmd: u32) -> bindings::v4l2_decoder_cmd {
    bindings::v4l2_decoder_cmd {
        cmd,
        flags: 0,
        __bindgen_anon_1: bindings::v4l2_decoder_cmd__bindgen_ty_1 {
            stop: bindings::v4l2_decoder_cmd__bindgen_ty_1__bindgen_ty_1 { pts: 0 },
        },
    }
}

/// Take a session from OPEN to both queues streaming at 320x240, MMAP buffers, and return the
/// CAPTURE `sizeimage`. Leaves four CAPTURE buffers queued.
///
/// Every wait below counts only the events logged from `mark` on. The rig's log is device-wide,
/// so a second session brought up while a first is already streaming would otherwise see the
/// first's `SOURCE_CHANGE` and `DQBUF` and stop waiting for its own.
fn start_streaming_320x240(r: &mut Rig, s: &mut Session) -> u32 {
    let mark = r.events.borrow().len();
    r.device.s_fmt(s, OUTPUT, output_format(H264, 320, 240)).unwrap();
    r.device.reqbufs(s, OUTPUT, MemoryType::Mmap, 4).unwrap();
    r.device
        .subscribe_event(s, EventType::SourceChange(0), SubscribeEventFlags::empty())
        .unwrap();
    r.device
        .subscribe_event(s, EventType::Eos, SubscribeEventFlags::empty())
        .unwrap();
    // One OUTPUT buffer + STREAMON(OUTPUT) makes the backend parse and raise SOURCE_CHANGE.
    poke_mmap_output(s, 0, 0x01);
    r.device
        .qbuf(
            s,
            mmap_buffer(OUTPUT, 0, 1 << 20),
            vec![],
            PayloadValidity::ALL,
        )
        .unwrap();
    r.device.streamon(s, OUTPUT).unwrap();
    while source_changes(&r.events.borrow()[mark..]) == 0 {
        assert!(wait_ready(s), "no SOURCE_CHANGE within 2s");
        process(&mut r.device, s);
    }
    // Buffer 0's `InputBufferDone` follows the announcement (the backend holds it behind the
    // SOURCE_CHANGE, D45/D55), and may land in the batch after the event. Drain it before
    // returning, so a caller can re-queue buffer 0 -- several tests do -- without racing that
    // trailing DQBUF.
    while output_dqbuf_at(&r.events.borrow()[mark..], 0).is_none() {
        assert!(wait_ready(s), "no OUTPUT DQBUF for buffer 0 within 2s");
        process(&mut r.device, s);
    }
    let sizeimage = pix(&r.device.g_fmt(s, CAPTURE).unwrap()).sizeimage;
    r.device.reqbufs(s, CAPTURE, MemoryType::Mmap, 4).unwrap();
    for i in 0..4 {
        r.device
            .qbuf(
                s,
                mmap_buffer(CAPTURE, i, sizeimage),
                vec![],
                PayloadValidity::ALL,
            )
            .unwrap();
    }
    r.device.streamon(s, CAPTURE).unwrap();
    let _ = drain_events; // silence dead-code in builds where it is unused
    sizeimage
}

/// D73: a `REQBUFS` the pool cannot serve in full grants what it *can*, as vb2 does
/// (`__vb2_queue_alloc` keeps what it allocated; `vb2_core_reqbufs` only turns a short
/// allocation into `-ENOMEM` below the queue's own floor), instead of failing the whole
/// request -- the 4K `ffmpeg -f v4l2` failure of `logs/vpu_wp/B14-accept-B.md` section 2. The
/// OUTPUT queue's floor is one bitstream buffer, so a pool that holds five answers five.
#[test]
fn reqbufs_output_grants_what_the_pool_can_hold() {
    let mut r = rig();
    let mut s = session(&mut r.device);
    r.device
        .s_fmt(&mut s, OUTPUT, output_format(H264, 320, 240))
        .unwrap();
    let sizeimage = pix(&r.device.g_fmt(&s, OUTPUT).unwrap()).sizeimage;
    r.pool.holds(5, sizeimage as u64);

    let reply = r
        .device
        .reqbufs(&mut s, OUTPUT, MemoryType::Mmap, 256)
        .unwrap();
    assert_eq!(reply.count, 5, "the pool held five, so five were granted");
    assert_eq!(s.input.buffers.len(), 5);
    assert_eq!(r.pool.used(), 5 * sizeimage as u64);
    for index in 0..5 {
        let buf = r.device.querybuf(&s, OUTPUT, index).unwrap();
        assert_eq!(buf.index(), index);
        // The R2 length rule: a partial answer's buffers are bounded like any other's.
        assert_eq!(*buf.get_first_plane().length, sizeimage);
    }
    assert_eq!(r.device.querybuf(&s, OUTPUT, 5).err(), Some(libc::EINVAL));

    // `REQBUFS(0)` gives every byte back.
    r.device
        .reqbufs(&mut s, OUTPUT, MemoryType::Mmap, 0)
        .unwrap();
    assert!(s.input.buffers.is_empty());
    assert_eq!(r.pool.used(), 0);
    close(&mut r.device, s);
}

/// D73's other half on the decoder: a pool that cannot serve even one bitstream buffer is still
/// `ENOMEM`, with nothing kept -- and the session is left able to allocate what does fit.
#[test]
fn reqbufs_output_on_an_empty_pool_is_still_enomem_and_holds_nothing() {
    let mut r = rig();
    let mut s = session(&mut r.device);
    r.device
        .s_fmt(&mut s, OUTPUT, output_format(H264, 320, 240))
        .unwrap();
    let sizeimage = pix(&r.device.g_fmt(&s, OUTPUT).unwrap()).sizeimage;
    r.pool.holds(0, sizeimage as u64);

    assert_eq!(
        r.device
            .reqbufs(&mut s, OUTPUT, MemoryType::Mmap, 256)
            .err(),
        Some(libc::ENOMEM)
    );
    assert!(s.input.buffers.is_empty());
    assert_eq!(s.input.memory, None);
    assert_eq!(r.pool.used(), 0);

    r.pool.holds(2, sizeimage as u64);
    assert_eq!(
        r.device
            .reqbufs(&mut s, OUTPUT, MemoryType::Mmap, 2)
            .unwrap()
            .count,
        2
    );
    assert_eq!(r.pool.used(), 2 * sizeimage as u64);
    close(&mut r.device, s);
}

/// D73 meets D72: the CAPTURE queue's floor is the count the codec announced, not one. A decode
/// given fewer output slots than the codec needs does not run slowly, it stalls with every slot
/// held (D69/D72), so a short CAPTURE answer below the announced minimum is no answer at all --
/// `ENOMEM`, nothing kept. At or above it, the short answer stands.
#[test]
fn reqbufs_capture_below_the_announced_minimum_is_enomem() {
    let mut r = rig_announcing_min(8);
    let mut s = session(&mut r.device);
    let sizeimage = start_streaming_320x240(&mut r, &mut s);

    // Put the CAPTURE queue back and leave the pool room for five buffers only: below the eight
    // the codec announced.
    r.device.streamoff(&mut s, CAPTURE).unwrap();
    r.device
        .reqbufs(&mut s, CAPTURE, MemoryType::Mmap, 0)
        .unwrap();
    let output_held = r.pool.used();
    r.pool.holds_more(5, sizeimage as u64);

    assert_eq!(
        r.device
            .reqbufs(&mut s, CAPTURE, MemoryType::Mmap, 32)
            .err(),
        Some(libc::ENOMEM),
        "five is below the announced eight, so nothing is granted"
    );
    assert!(s.output.buffers.is_empty());
    assert_eq!(s.output.memory, None);
    assert_eq!(
        r.pool.used(),
        output_held,
        "the refused CAPTURE set left the OUTPUT queue's bytes and nothing else"
    );

    // Room for ten: the short answer is above the floor, so it stands.
    r.pool.holds_more(10, sizeimage as u64);
    let reply = r
        .device
        .reqbufs(&mut s, CAPTURE, MemoryType::Mmap, 32)
        .unwrap();
    assert_eq!(
        reply.count, 10,
        "ten fit, ten were granted (above the floor of eight)"
    );
    assert_eq!(s.output.buffers.len(), 10);
    assert_eq!(r.pool.used(), output_held + 10 * sizeimage as u64);
    for index in 0..10 {
        assert_eq!(
            r.device.querybuf(&s, CAPTURE, index).unwrap().index(),
            index
        );
    }
    assert_eq!(r.device.querybuf(&s, CAPTURE, 10).err(), Some(libc::EINVAL));

    r.device
        .reqbufs(&mut s, CAPTURE, MemoryType::Mmap, 0)
        .unwrap();
    assert_eq!(r.pool.used(), output_held);
    close(&mut r.device, s);
}

/// D73 on `CREATE_BUFS`: `index` + `count` are what the guest indexes the new buffers by, so a
/// short answer reports what was really created. The announced CAPTURE minimum is a `REQBUFS`
/// floor, not a `CREATE_BUFS` one (this call adds to a queue that already has buffers, and vb2's
/// `vb2_core_create_bufs` fails only when it could create none) -- so one new buffer is enough,
/// and a set the pool cannot start at all is `ENOMEM` with the existing buffers untouched.
#[test]
fn create_bufs_grants_what_the_pool_can_hold() {
    let mut r = rig();
    let mut s = session(&mut r.device);
    r.device
        .s_fmt(&mut s, OUTPUT, output_format(H264, 320, 240))
        .unwrap();
    let sizeimage = pix(&r.device.g_fmt(&s, OUTPUT).unwrap()).sizeimage;
    r.pool.holds(5, sizeimage as u64);

    r.device
        .reqbufs(&mut s, OUTPUT, MemoryType::Mmap, 2)
        .unwrap();
    // `g_fmt` already answers the one-plane OUTPUT format `create_bufs` requires.
    let fmt = r.device.g_fmt(&s, OUTPUT).unwrap();
    let reply = r
        .device
        .create_bufs(&mut s, 8, OUTPUT, MemoryType::Mmap, fmt)
        .unwrap();
    assert_eq!((reply.index, reply.count), (2, 3));
    assert_eq!(s.input.buffers.len(), 5);
    assert_eq!(r.pool.used(), 5 * sizeimage as u64);
    assert!(r.device.querybuf(&s, OUTPUT, 4).is_ok());
    assert_eq!(r.device.querybuf(&s, OUTPUT, 5).err(), Some(libc::EINVAL));

    assert_eq!(
        r.device
            .create_bufs(&mut s, 4, OUTPUT, MemoryType::Mmap, fmt)
            .err(),
        Some(libc::ENOMEM)
    );
    assert_eq!(s.input.buffers.len(), 5);
    assert_eq!(r.pool.used(), 5 * sizeimage as u64);

    r.device
        .reqbufs(&mut s, OUTPUT, MemoryType::Mmap, 0)
        .unwrap();
    assert_eq!(r.pool.used(), 0);
    close(&mut r.device, s);
}
