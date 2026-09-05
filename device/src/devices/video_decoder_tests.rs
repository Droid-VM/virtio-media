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

use super::*;
use crate::ioctl::ffmpeg_wire;
use crate::ioctl::VirtioMediaIoctlHandler;
use crate::MemFdAllocator;

const OUTPUT: QueueType = QueueType::VideoOutputMplane;
const CAPTURE: QueueType = QueueType::VideoCaptureMplane;
const H264: PixelFormat = PixelFormat::from_fourcc(b"H264");
const HEVC: PixelFormat = PixelFormat::from_fourcc(b"HEVC");
const VP90: PixelFormat = PixelFormat::from_fourcc(b"VP90");

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

/// Counts what the device gives back, and whether it happened while the backend was still open.
struct OrderedAllocator {
    inner: MemFdAllocator,
    log: SharedLog,
}

impl VirtioMediaBufferAllocator for OrderedAllocator {
    fn allocate(&mut self, len: u64) -> Result<HostBuffer, i32> {
        self.inner.allocate(len)
    }

    fn release(&mut self, buf: HostBuffer) {
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
/// A bitstream buffer the backend consumes without being able to announce a format: it returns
/// the OUTPUT buffer (`InputBufferDone`) but emits no `SOURCE_CHANGE`, and produces no frame. This
/// models the MediaCodec backend releasing a held `InputBufferDone` when the codec asks for more
/// input than the buffer carried (D48, `android.rs` -- the first packet of an mp4 our own encoder
/// wrote is a 31-byte header the codec cannot announce from). A one-buffer-in-flight client must
/// still get this buffer back, or it never queues the next.
const NO_ANNOUNCE_MAGIC: u8 = 0xfc;
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
    /// `start` fails with this errno.
    fail_start: Option<i32>,
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

            for cmd in rx {
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
                            // The codec consumed the buffer but cannot announce a format from it,
                            // and asks for more input: the OUTPUT buffer goes back with no
                            // SOURCE_CHANGE and no frame (D48).
                            emit(DecoderEvent::InputBufferDone(index));
                            continue;
                        }
                        if first_byte == DRC_MAGIC
                            || first_byte == DRC_LAST_MAGIC
                            || first_byte == DRC_LAST_AFTER_MAGIC
                        {
                            // The input is consumed at once.
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
                                min_capture_buffers: 4,
                            });
                        }
                        // The input is consumed at once, behind the announcement above.
                        emit(DecoderEvent::InputBufferDone(index));
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
                        // but no pre-seek frame is produced into them.
                        ready.clear();
                        draining = false;
                        pending_format = None;
                        let _ = ack.send(());
                    }
                    Cmd::ClearCapture(ack) => {
                        captures.clear();
                        ready.clear();
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
}

const GUEST_MEMORY: usize = 8 << 20;

fn caps() -> DecoderCapabilities {
    let range = SizeRange::new(16, 4096, 2);
    DecoderCapabilities {
        coded_formats: vec![
            CodedFormat {
                fourcc: H264,
                width: range,
                height: range,
                dynamic_resolution: true,
            },
            CodedFormat {
                fourcc: HEVC,
                width: range,
                height: range,
                dynamic_resolution: true,
            },
            // A narrower range than the others, as VP9's is on 5566 (B5 §5.1), so a query that
            // answers the wrong format's range is caught.
            CodedFormat {
                fourcc: VP90,
                width: SizeRange::new(16, 2048, 2),
                height: SizeRange::new(16, 2048, 2),
                dynamic_resolution: true,
            },
        ],
    }
}

fn rig_with(fail_start: Option<i32>) -> Rig {
    let events = EventLog::default();
    let events_log = Rc::clone(&events.0);
    let log: SharedLog = Default::default();
    let guest = FakeGuest {
        memory: Rc::new(RefCell::new(vec![0u8; GUEST_MEMORY])),
        live_mappings: Rc::new(RefCell::new(0)),
        log: Arc::clone(&log),
    };
    let backend = FakeBackend {
        caps: caps(),
        log: Arc::clone(&log),
        fail_start,
    };
    let device = VideoDecoder::new(
        backend,
        events,
        guest.clone(),
        FakeHostMapper,
        OrderedAllocator {
            inner: MemFdAllocator::new(),
            log: Arc::clone(&log),
        },
    );
    Rig {
        device,
        events: events_log,
        guest,
        log,
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
    assert_eq!(r.device.active_session, None);
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

/// One decode at a time: a second session's `REQBUFS`/`CREATE_BUFS` is refused with `EBUSY` while
/// the first holds buffers, and succeeds once it lets go.
#[test]
fn a_second_session_is_refused_while_one_holds_buffers() {
    let mut r = rig();
    let mut s = session(&mut r.device);
    r.device.s_fmt(&mut s, OUTPUT, output_format(H264, 320, 240)).unwrap();
    r.device.reqbufs(&mut s, OUTPUT, MemoryType::Mmap, 2).unwrap();

    let mut other = new_session(&mut r.device, 1);
    assert_eq!(
        r.device.reqbufs(&mut other, OUTPUT, MemoryType::Mmap, 1).err(),
        Some(libc::EBUSY)
    );
    assert_eq!(
        r.device
            .create_bufs(&mut other, 1, OUTPUT, MemoryType::Mmap, output_format(H264, 320, 240))
            .err(),
        Some(libc::EBUSY)
    );

    // The first session lets go; the second can take the device.
    r.device.reqbufs(&mut s, OUTPUT, MemoryType::Mmap, 0).unwrap();
    close(&mut r.device, s);
    r.device.reqbufs(&mut other, OUTPUT, MemoryType::Mmap, 1).unwrap();
    close(&mut r.device, other);
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
    assert_eq!(r.device.active_session, None);
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
    collect_capture(&mut r, &mut s, 1);
    let last = dequeued_on(&r.events.borrow(), CAPTURE).pop().unwrap();
    assert!(last.flags().contains(BufferFlags::LAST));
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

/// D48: a backend that consumes the first OUTPUT buffer but cannot announce a format from it (an
/// mp4 whose first packet is a 31-byte header, D44) still returns the buffer -- **before** any
/// `SOURCE_CHANGE` -- and the device must dequeue it for the guest. A one-buffer-in-flight client
/// (a stateful ffmpeg on an mp4, `v4l2-compliance -s`) polls for its OUTPUT buffer to come back
/// before it queues the next; if the device held the buffer behind the initial `SOURCE_CHANGE`
/// the codec can never reach, the two wait on each other forever (B9-build §6 measured exactly
/// this: `1 bitstream buffers in, 0 frames out`). The MediaCodec backend bounds that hold
/// (`android.rs`: it releases the held `InputBufferDone` the moment the codec asks for more input
/// than the buffer carried, plus a 500 ms backstop); this pins the device end of the contract --
/// an `InputBufferDone` reported before the first `FormatChanged` reaches the guest as a `DQBUF`,
/// so the D45 hold (which lives entirely in the backend) can end without waiting for a
/// `SOURCE_CHANGE` that is not coming. The companion
/// `source_change_precedes_the_output_buffer_that_produced_it` pins the other half: when the codec
/// *can* announce, the event still precedes the buffer (`POLLPRI` before `POLLOUT`).
#[test]
fn an_output_buffer_returns_before_the_first_source_change_when_the_codec_cannot_announce() {
    let mut r = rig();
    let mut s = session(&mut r.device);
    r.device.s_fmt(&mut s, OUTPUT, output_format(H264, 320, 240)).unwrap();
    r.device.reqbufs(&mut s, OUTPUT, MemoryType::Mmap, 4).unwrap();
    r.device
        .subscribe_event(&mut s, EventType::SourceChange(0), SubscribeEventFlags::empty())
        .unwrap();

    // Buffer 0 carries bytes the backend cannot announce a format from: it comes back with no
    // SOURCE_CHANGE, modelling the release condition the MediaCodec backend implements (D48).
    poke_mmap_output(&mut s, 0, NO_ANNOUNCE_MAGIC);
    let mut ob = mmap_buffer(OUTPUT, 0, 1 << 20);
    ob.set_timestamp(ts(1));
    r.device.qbuf(&mut s, ob, vec![], PayloadValidity::ALL).unwrap();
    r.device.streamon(&mut s, OUTPUT).unwrap();
    while dequeued_on(&r.events.borrow(), OUTPUT).is_empty() {
        assert!(wait_ready(&s), "no OUTPUT DQBUF within 2s");
        process(&mut r.device, &mut s);
    }
    // The buffer came back, and no SOURCE_CHANGE preceded it: a one-buffer client can now queue
    // the next OUTPUT buffer instead of deadlocking.
    assert_eq!(dequeued_on(&r.events.borrow(), OUTPUT).len(), 1, "buffer 0 returned");
    assert_eq!(source_changes(&r.events.borrow()), 0, "no announcement yet");

    // Queue buffer 1, which the backend CAN announce from: the SOURCE_CHANGE arrives now, and
    // this buffer too comes back.
    poke_mmap_output(&mut s, 1, 0x01);
    let mut ob = mmap_buffer(OUTPUT, 1, 1 << 20);
    ob.set_timestamp(ts(2));
    r.device.qbuf(&mut s, ob, vec![], PayloadValidity::ALL).unwrap();
    while source_changes(&r.events.borrow()) == 0 {
        assert!(wait_ready(&s), "no SOURCE_CHANGE within 2s");
        process(&mut r.device, &mut s);
    }
    assert_eq!(source_changes(&r.events.borrow()), 1);
    assert_eq!(
        dequeued_on(&r.events.borrow(), OUTPUT).len(),
        2,
        "both OUTPUT buffers returned"
    );
    close(&mut r.device, s);
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

    // The NEXT_CTRL walk: id 0 -> the User Controls class marker -> MIN_BUFFERS_FOR_CAPTURE ->
    // EINVAL (the end).
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
    assert_eq!(walk(&mut r, &s, CID_MIN_CAP), Err(libc::EINVAL), "walk ends");

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

// helpers used by several tests ---------------------------------------------------------------

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
fn start_streaming_320x240(r: &mut Rig, s: &mut Session) -> u32 {
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
    while source_changes(&r.events.borrow()) == 0 {
        assert!(wait_ready(s), "no SOURCE_CHANGE within 2s");
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
