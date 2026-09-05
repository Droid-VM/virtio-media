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
use std::rc::Rc;
use std::sync::mpsc;
use std::sync::Arc;
use std::sync::Mutex;
use std::thread;

use super::*;
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
}

type SharedLog = Arc<Mutex<FakeLog>>;

/// The bitstream's first byte that makes the fake report a mid-stream resolution change.
const DRC_MAGIC: u8 = 0xff;
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

            // Write a frame into a capture buffer and dequeue it, or -- when draining and no
            // frame is left -- the empty LAST buffer.
            let pump = |ready: &mut std::collections::VecDeque<ReadyFrame>,
                            captures: &mut std::collections::VecDeque<(u32, SendPtr, usize)>,
                            coded_size: (u32, u32),
                            draining: &mut bool| {
                while let Some(&(index, ptr, len)) = captures.front() {
                    if let Some(frame) = ready.pop_front() {
                        let (w, h) = (coded_size.0 as usize, coded_size.1 as usize);
                        let sizeimage = w * h + 2 * (w.div_ceil(2)) * (h.div_ceil(2));
                        assert!(len >= sizeimage, "capture buffer too small for a frame");
                        let dst = ptr.as_ptr();
                        for row in 0..h {
                            // SAFETY: `len >= sizeimage` covers both loops.
                            unsafe { std::ptr::write_bytes(dst.add(row * w), frame.luma, w) };
                        }
                        for row in 0..h.div_ceil(2) {
                            // SAFETY: as above.
                            unsafe { std::ptr::write_bytes(dst.add((h + row) * w), 0x80, w) };
                        }
                        captures.pop_front();
                        emit(DecoderEvent::FrameDecoded {
                            index,
                            bytesused: sizeimage as u32,
                            timestamp: frame.timestamp,
                            is_last: false,
                        });
                    } else if *draining {
                        captures.pop_front();
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
                        // The input is consumed at once.
                        emit(DecoderEvent::InputBufferDone(index));
                        if first_byte == DRC_MAGIC {
                            // Mid-stream resolution change: halve the size (kept even).
                            coded_size = (
                                (coded_size.0 / 2).max(2) & !1,
                                (coded_size.1 / 2).max(2) & !1,
                            );
                            emit(DecoderEvent::FormatChanged {
                                coded_size,
                                visible_rect: v4l2r::Rect::new(0, 0, coded_size.0, coded_size.1),
                                min_capture_buffers: 4,
                            });
                            continue;
                        }
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
                        ready.push_back(ReadyFrame {
                            luma: luma_of(first_byte),
                            timestamp,
                        });
                        pump(&mut ready, &mut captures, coded_size, &mut draining);
                    }
                    Cmd::UseCapture { index, ptr, len } => {
                        captures.push_back((index, ptr, len));
                        pump(&mut ready, &mut captures, coded_size, &mut draining);
                    }
                    Cmd::Flush(ack) => {
                        // Seek: drop everything queued so far; the CAPTURE queue keeps its buffers
                        // but no pre-seek frame is produced into them.
                        ready.clear();
                        draining = false;
                        let _ = ack.send(());
                    }
                    Cmd::ClearCapture(ack) => {
                        captures.clear();
                        ready.clear();
                        draining = false;
                        let _ = ack.send(());
                    }
                    Cmd::Drain => {
                        draining = true;
                        pump(&mut ready, &mut captures, coded_size, &mut draining);
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
        self.log.lock().unwrap().capture_active = true;
        self.commands
            .send(Cmd::UseCapture {
                index: buffer.index,
                ptr: buffer.ptr,
                len: buffer.len,
            })
            .map_err(|_| libc::EIO)
    }

    fn clear_capture_buffers(&mut self) -> IoctlResult<()> {
        self.log.lock().unwrap().clears += 1;
        // The rendezvous joins the worker w.r.t. CAPTURE buffers; only then is it safe to free
        // them, which is what the `capture_active` flag records.
        self.rendezvous(Cmd::ClearCapture);
        self.log.lock().unwrap().capture_active = false;
        Ok(())
    }

    fn flush(&mut self) -> IoctlResult<()> {
        self.log.lock().unwrap().flushes += 1;
        self.rendezvous(Cmd::Flush);
        Ok(())
    }

    fn drain(&mut self) -> IoctlResult<()> {
        self.commands.send(Cmd::Drain).map_err(|_| libc::EIO)
    }

    fn stop(&mut self) {
        if let Some(thread) = self.thread.take() {
            self.rendezvous(Cmd::Stop);
            let _ = thread.join();
            let mut log = self.log.lock().unwrap();
            log.stops += 1;
            log.capture_active = false;
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
            CodedFormat {
                fourcc: VP90,
                width: range,
                height: range,
                dynamic_resolution: true,
            },
        ],
    }
}

fn rig_with(fail_start: Option<i32>) -> Rig {
    let events = EventLog::default();
    let events_log = Rc::clone(&events.0);
    let guest = FakeGuest {
        memory: Rc::new(RefCell::new(vec![0u8; GUEST_MEMORY])),
        live_mappings: Rc::new(RefCell::new(0)),
    };
    let log: SharedLog = Default::default();
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
    r.device.qbuf(&mut s, ob, vec![], true).unwrap();
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
        r.device.qbuf(&mut s, mmap_buffer(CAPTURE, i, sizeimage), vec![], true).unwrap();
    }
    r.device.streamon(&mut s, CAPTURE).unwrap();

    // Decode two more OUTPUT buffers; with the buffer 0 frame produced after capture setup, that
    // is three frames, ts 1..3 (the OUTPUT timestamps, copied through TIMESTAMP_COPY).
    for i in 1..3u32 {
        poke_mmap_output(&mut s, i as usize, 0x10 + i as u8);
        let mut ob = mmap_buffer(OUTPUT, i, 1 << 20);
        ob.set_timestamp(ts(i as i64 + 1));
        r.device.qbuf(&mut s, ob, vec![], true).unwrap();
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
    r.device.qbuf(&mut s, ob, vec![], true).unwrap();
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
        r.device.qbuf(&mut s, mmap_buffer(CAPTURE, i, sizeimage), vec![], true).unwrap();
    }
    r.device.streamon(&mut s, CAPTURE).unwrap();

    // A frame comes out.
    collect_capture(&mut r, &mut s, 1);
    let frame = dequeued_on(&r.events.borrow(), CAPTURE).remove(0);
    assert_eq!(frame.timestamp().tv_sec, 7);

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
    r.device.qbuf(&mut s, mmap_buffer(OUTPUT, 0, 1 << 20), vec![], true).unwrap();
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
    r.device.qbuf(&mut s, mmap_buffer(OUTPUT, 1, 1 << 20), vec![], true).unwrap();
    r.device.qbuf(&mut s, mmap_buffer(CAPTURE, 0, sizeimage), vec![], true).unwrap();
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
    r.device.qbuf(&mut s, mmap_buffer(OUTPUT, 0, 1 << 20), vec![], true).unwrap();
    collect_capture(&mut r, &mut s, 1);
    let before = source_changes(&r.events.borrow());

    // A magic OUTPUT buffer triggers a mid-stream resolution change to 160x120.
    poke_mmap_output(&mut s, 1, DRC_MAGIC);
    r.device.qbuf(&mut s, mmap_buffer(OUTPUT, 1, 1 << 20), vec![], true).unwrap();
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
        r.device.qbuf(&mut s, mmap_buffer(CAPTURE, i, new_sizeimage), vec![], true).unwrap();
    }
    r.device.streamon(&mut s, CAPTURE).unwrap();

    // A frame at the new resolution.
    poke_mmap_output(&mut s, 2, 0x09);
    r.device.qbuf(&mut s, mmap_buffer(OUTPUT, 2, 1 << 20), vec![], true).unwrap();
    let already = dequeued_on(&r.events.borrow(), CAPTURE).len();
    while dequeued_on(&r.events.borrow(), CAPTURE).len() <= already {
        assert!(wait_ready(&s), "no post-DRC frame within 2s");
        process(&mut r.device, &mut s);
    }
    let frame = dequeued_on(&r.events.borrow(), CAPTURE).pop().unwrap();
    assert_eq!(*frame.get_first_plane().bytesused, new_sizeimage);

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
    r.device.qbuf(&mut s, ob, sgs, true).unwrap();
    assert_eq!(*r.guest.live_mappings.borrow(), 1, "OUTPUT mapping held while queued");
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
    assert_eq!(r.device.qbuf(&mut s, small, sgs, true).err(), Some(libc::EINVAL));
    assert_eq!(*r.guest.live_mappings.borrow(), 0);

    let cap_gpa = 0x200000u64;
    let (cb, sgs) = userptr_buffer(CAPTURE, 0, cap_gpa, sizeimage, sizeimage);
    r.device.qbuf(&mut s, cb, sgs, true).unwrap();
    assert_eq!(*r.guest.live_mappings.borrow(), 1, "CAPTURE mapping held while lent");
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
        r.device.qbuf(&mut s, mmap_buffer(OUTPUT, i, 1 << 20), vec![], true).unwrap();
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
    r.device.reqbufs(&mut s, CAPTURE, MemoryType::Mmap, 2).unwrap();
    r.device.qbuf(&mut s, mmap_buffer(CAPTURE, 0, sizeimage), vec![], true).unwrap();
    close(&mut r.device, s);
    assert_eq!(r.log.lock().unwrap().stops, 1, "close joined the backend exactly once");
    assert_eq!(*r.log.lock().unwrap().released_while_capture_active.borrow(), 0);
    assert_eq!(r.device.active_session, None);
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
    r.device.qbuf(&mut s, mmap_buffer(OUTPUT, 0, 1 << 20), vec![], true).unwrap();
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
    let prepared = r.device.prepare_buf(&mut s, pb, vec![], true).unwrap();
    assert!(prepared.flags().contains(BufferFlags::PREPARED));
    assert_eq!(*prepared.get_first_plane().bytesused, 5000);

    // A following QBUF with nonsense payload keeps the prepared description.
    let (mut qb, sgs) = userptr_buffer(OUTPUT, 0, out_gpa, 1 << 20, 0xdead_beef);
    *qb.get_first_plane_mut().bytesused = 0xdead_beef;
    let queued = r.device.qbuf(&mut s, qb, sgs, true).unwrap();
    assert!(queued.flags().contains(BufferFlags::QUEUED));
    assert_eq!(*queued.get_first_plane().bytesused, 5000, "the prepared bytesused survived");

    close(&mut r.device, s);
}

/// `SUBSCRIBE_EVENT` accepts only `SOURCE_CHANGE` and `EOS`; anything else is `EINVAL`. A decoder
/// must offer both (the `v4l2-compliance` `testEvents` rule for a stateful decoder).
#[test]
fn events_are_source_change_and_eos_only() {
    let mut r = rig();
    let mut s = session(&mut r.device);
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
        r.device.qbuf(&mut s, mmap_buffer(OUTPUT, 0, 1 << 20), vec![], true).err(),
        Some(libc::ENODEV)
    );
    assert_eq!(r.device.streamon(&mut s, OUTPUT).err(), Some(libc::ENODEV));

    close(&mut r.device, s);
}

// helpers used by several tests ---------------------------------------------------------------

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
    r.device.qbuf(s, mmap_buffer(OUTPUT, 0, 1 << 20), vec![], true).unwrap();
    r.device.streamon(s, OUTPUT).unwrap();
    while source_changes(&r.events.borrow()) == 0 {
        assert!(wait_ready(s), "no SOURCE_CHANGE within 2s");
        process(&mut r.device, s);
    }
    let sizeimage = pix(&r.device.g_fmt(s, CAPTURE).unwrap()).sizeimage;
    r.device.reqbufs(s, CAPTURE, MemoryType::Mmap, 4).unwrap();
    for i in 0..4 {
        r.device.qbuf(s, mmap_buffer(CAPTURE, i, sizeimage), vec![], true).unwrap();
    }
    r.device.streamon(s, CAPTURE).unwrap();
    let _ = drain_events; // silence dead-code in builds where it is unused
    sizeimage
}
