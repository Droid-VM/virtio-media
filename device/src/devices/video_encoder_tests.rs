// Copyright 2026 The DroidVM Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

//! Tests for the stateful [`VideoEncoder`] (`super`).
//!
//! [`FakeBackend`] runs a codec thread with the session eventfd, the way `FakeDecoderBackend`
//! does: each raw NV12 frame becomes one synthetic packet whose bytes carry the frame's luma
//! digest, keyframes fall on GOP boundaries and on `FORCE_KEY_FRAME`, stream headers are
//! emitted per `HEADER_MODE`, and a raw frame is released only once its packet is written (as a
//! real encoder does, which is what makes the reset path observable). The tests replay --
//! ioctl by ioctl -- the sequences the two guest userland clients issue
//! (`ffmpeg -c:v h264_v4l2m2m`, GStreamer `v4l2h264enc`), the controls, drain/EOS, the §2.5
//! ordering invariants and the `v4l2-compliance` probes.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::mpsc;
use std::sync::Arc;
use std::sync::Mutex;
use std::thread;

use v4l2r::controls::codec::VideoH264Level;
use v4l2r::controls::codec::VideoH264Profile;
use v4l2r::controls::codec::VideoHEVCLevel;
use v4l2r::controls::codec::VideoHEVCProfile;

use super::*;
use crate::ioctl::ffmpeg_wire;
use crate::ioctl::VirtioMediaIoctlHandler;
use crate::MemFdAllocator;

const OUTPUT: QueueType = QueueType::VideoOutputMplane;
const CAPTURE: QueueType = QueueType::VideoCaptureMplane;
const H264: PixelFormat = PixelFormat::from_fourcc(b"H264");
const HEVC: PixelFormat = PixelFormat::from_fourcc(b"HEVC");
const VP80: PixelFormat = PixelFormat::from_fourcc(b"VP80");

const CID_GOP: u32 = bindings::V4L2_CID_MPEG_VIDEO_GOP_SIZE;
const CID_BITRATE: u32 = bindings::V4L2_CID_MPEG_VIDEO_BITRATE;
const CID_BITRATE_MODE: u32 = bindings::V4L2_CID_MPEG_VIDEO_BITRATE_MODE;
const CID_HEADER_MODE: u32 = bindings::V4L2_CID_MPEG_VIDEO_HEADER_MODE;
const CID_FORCE_KEY: u32 = bindings::V4L2_CID_MPEG_VIDEO_FORCE_KEY_FRAME;
const CID_PREPEND: u32 = bindings::V4L2_CID_MPEG_VIDEO_PREPEND_SPSPPS_TO_IDR;
const CID_H264_PROFILE: u32 = bindings::V4L2_CID_MPEG_VIDEO_H264_PROFILE;
const CID_H264_LEVEL: u32 = bindings::V4L2_CID_MPEG_VIDEO_H264_LEVEL;
const CID_H264_MIN_QP: u32 = bindings::V4L2_CID_MPEG_VIDEO_H264_MIN_QP;
const CID_H264_MAX_QP: u32 = bindings::V4L2_CID_MPEG_VIDEO_H264_MAX_QP;
const CID_HEVC_PROFILE: u32 = bindings::V4L2_CID_MPEG_VIDEO_HEVC_PROFILE;
const CID_HEVC_LEVEL: u32 = bindings::V4L2_CID_MPEG_VIDEO_HEVC_LEVEL;
const CID_HEVC_MIN_QP: u32 = bindings::V4L2_CID_MPEG_VIDEO_HEVC_MIN_QP;
const CID_HEVC_MAX_QP: u32 = bindings::V4L2_CID_MPEG_VIDEO_HEVC_MAX_QP;
const CID_VP8_PROFILE: u32 = bindings::V4L2_CID_MPEG_VIDEO_VP8_PROFILE;
const CID_MIN_OUT: u32 = bindings::V4L2_CID_MIN_BUFFERS_FOR_OUTPUT;
const CID_MIN_CAP: u32 = bindings::V4L2_CID_MIN_BUFFERS_FOR_CAPTURE;
const CID_B_FRAMES: u32 = bindings::V4L2_CID_MPEG_VIDEO_B_FRAMES;
const CID_FRAME_RC: u32 = bindings::V4L2_CID_MPEG_VIDEO_FRAME_RC_ENABLE;
const CODEC_CLASS: u32 = bindings::V4L2_CTRL_CLASS_CODEC;
const USER_CLASS: u32 = bindings::V4L2_CTRL_CLASS_USER;

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
        // touching the queue's buffers. `capture_active` is true only between a CAPTURE buffer
        // being lent and the backend being joined (stop), so a non-zero count here is a real
        // violation.
        if self.log.lock().unwrap().capture_active {
            *self
                .log
                .lock()
                .unwrap()
                .released_while_capture_active
                .borrow_mut() += 1;
        }
        self.inner.release(buf);
    }
}

// ---------------------------------------------------------------------------------------------
// The fake encoder backend: a codec on a thread of its own.
// ---------------------------------------------------------------------------------------------

/// The synthetic bitstream: Annex-B-looking start codes, then a NAL byte that says what the
/// unit is. Headers are an SPS and a PPS; a frame unit carries its number and its luma digest.
const START_CODE: [u8; 4] = [0, 0, 0, 1];
const NAL_SPS: u8 = 0x67;
const NAL_PPS: u8 = 0x68;
const NAL_KEY: u8 = 0x65;
const NAL_INTER: u8 = 0x41;
const HEADERS_LEN: usize = 16;
const FRAME_LEN: usize = 10;

fn headers_bytes() -> Vec<u8> {
    let mut v = Vec::new();
    v.extend_from_slice(&START_CODE);
    v.extend_from_slice(&[NAL_SPS, b'S', b'P', b'S']);
    v.extend_from_slice(&START_CODE);
    v.extend_from_slice(&[NAL_PPS, b'P', b'P', b'S']);
    assert_eq!(v.len(), HEADERS_LEN);
    v
}

fn frame_bytes(key: bool, frame_no: u8, digest: u32) -> Vec<u8> {
    let mut v = Vec::new();
    v.extend_from_slice(&START_CODE);
    v.push(if key { NAL_KEY } else { NAL_INTER });
    v.push(frame_no);
    v.extend_from_slice(&digest.to_le_bytes());
    assert_eq!(v.len(), FRAME_LEN);
    v
}

/// A unit of the synthetic bitstream, parsed back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Unit {
    Sps,
    Pps,
    Frame {
        key: bool,
        frame_no: u8,
        digest: u32,
    },
}

fn parse_units(bytes: &[u8]) -> Vec<Unit> {
    let mut units = Vec::new();
    let mut i = 0;
    while i + 5 <= bytes.len() {
        assert_eq!(&bytes[i..i + 4], &START_CODE, "start code at {i}");
        match bytes[i + 4] {
            NAL_SPS => {
                units.push(Unit::Sps);
                i += 8;
            }
            NAL_PPS => {
                units.push(Unit::Pps);
                i += 8;
            }
            nal @ (NAL_KEY | NAL_INTER) => {
                let digest = u32::from_le_bytes(bytes[i + 6..i + 10].try_into().unwrap());
                units.push(Unit::Frame {
                    key: nal == NAL_KEY,
                    frame_no: bytes[i + 5],
                    digest,
                });
                i += FRAME_LEN;
            }
            other => panic!("unknown NAL {other:#x} at {i}"),
        }
    }
    assert_eq!(i, bytes.len(), "trailing bytes");
    units
}

/// The luma digest of a frame whose luma plane is all `luma`: the sum of its bytes.
fn digest_of(luma: u8, (w, h): (u32, u32)) -> u32 {
    luma as u32 * w * h
}

/// What the fake encoder did, for the assertions.
#[derive(Default)]
struct FakeLog {
    /// `start` calls, with the config each was given.
    started: Vec<EncoderConfig>,
    /// Whether a backend session is open (thread alive and not stopped).
    open: bool,
    /// Whether a CAPTURE buffer is currently lent to the backend (true from `use_as_capture`
    /// until the backend is joined by `stop`).
    capture_active: bool,
    /// `flush` (STREAMOFF(OUTPUT) / ENC_CMD_START) calls.
    flushes: usize,
    /// `stop` calls that completed.
    stops: usize,
    /// `set_bitrate` calls.
    bitrates: Vec<u32>,
    /// `force_keyframe` calls.
    forced: usize,
    /// Buffers released while `capture_active` -- the §2.5 violation the ordering tests look for.
    released_while_capture_active: RefCell<usize>,
    /// Raw frames a `flush` dropped without a word: the ones the codec had not written out, and
    /// the ones parked behind a drain's `EOS` (the MediaCodec backend's rule, review-m7 R7-11).
    dropped_by_flush: usize,
    /// The next `flush` / `drain` fails with this errno (once): the wedged-codec case the
    /// device's bounds exist for.
    fail_flush: Option<i32>,
    fail_drain: Option<i32>,
}

type SharedLog = Arc<Mutex<FakeLog>>;

struct FakeBackend {
    caps: EncoderCapabilities,
    log: SharedLog,
    /// `start` fails with this errno.
    fail_start: Option<i32>,
}

enum Cmd {
    Start(EncoderConfig),
    Encode {
        index: u32,
        ptr: SendPtr,
        len: usize,
        timestamp: bindings::timeval,
    },
    UseCapture {
        index: u32,
        ptr: SendPtr,
        len: usize,
    },
    ForceKey,
    Flush(mpsc::Sender<()>),
    Drain,
    Stop(mpsc::Sender<()>),
}

struct FakeSession {
    sink: EncoderSink,
    log: SharedLog,
    fail_start: Option<i32>,
    /// The codec thread's command channel, while a codec exists.
    commands: Option<mpsc::Sender<Cmd>>,
    events_tx: mpsc::Sender<EncoderEvent>,
    events: mpsc::Receiver<EncoderEvent>,
    thread: Option<thread::JoinHandle<()>>,
}

/// A packet the fake has produced and is waiting to write into a CAPTURE buffer.
struct Packet {
    bytes: Vec<u8>,
    kind: FrameKind,
    timestamp: bindings::timeval,
    /// The OUTPUT buffer it came from, released when the packet is written.
    input: Option<u32>,
}

/// The codec thread: one per `start`, gone after `stop`, as a MediaCodec instance is.
fn run_codec(
    rx: mpsc::Receiver<Cmd>,
    events_tx: mpsc::Sender<EncoderEvent>,
    sink: EncoderSink,
    thread_log: SharedLog,
) {
    let emit = |e: EncoderEvent| {
        let _ = events_tx.send(e);
        sink.signal();
    };
    let mut config: Option<EncoderConfig> = None;
    let mut frame_no: u8 = 0;
    let mut force_key = false;
    let mut headers_due = false;
    let mut ready: VecDeque<Packet> = VecDeque::new();
    let mut captures: VecDeque<(u32, SendPtr, usize)> = VecDeque::new();
    let mut draining = false;
    // From the drain's `EOS` on, a codec takes no input until it is flushed (the async rule):
    // frames lent in the meantime sit in the backend's FIFO, encoded by nobody, and go with the
    // flush -- the MediaCodec backend's `pending`.
    let mut stopped = false;
    let mut parked: Vec<u32> = Vec::new();

    // Write packets into capture buffers while both are there; when draining, the
    // last packet (or an empty buffer) carries LAST.
    let pump = |ready: &mut VecDeque<Packet>,
                captures: &mut VecDeque<(u32, SendPtr, usize)>,
                draining: &mut bool,
                stopped: &mut bool| {
        while let Some(&(index, ptr, len)) = captures.front() {
            if let Some(packet) = ready.pop_front() {
                assert!(len >= packet.bytes.len(), "capture buffer too small");
                // SAFETY: the device lent `len` writable bytes at `ptr` until we
                // report the buffer.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        packet.bytes.as_ptr(),
                        ptr.as_ptr(),
                        packet.bytes.len(),
                    )
                };
                captures.pop_front();
                if let Some(input) = packet.input {
                    emit(EncoderEvent::InputBufferDone(input));
                }
                let is_last = *draining && ready.is_empty();
                if is_last {
                    *draining = false;
                    *stopped = true;
                }
                emit(EncoderEvent::FrameEncoded {
                    index,
                    bytesused: packet.bytes.len() as u32,
                    timestamp: packet.timestamp,
                    kind: packet.kind,
                    is_last,
                });
            } else if *draining {
                captures.pop_front();
                *draining = false;
                *stopped = true;
                emit(EncoderEvent::FrameEncoded {
                    index,
                    bytesused: 0,
                    timestamp: Default::default(),
                    kind: FrameKind::Key,
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
            Cmd::Start(c) => {
                config = Some(c);
                frame_no = 0;
                force_key = false;
                headers_due = true;
                draining = false;
                stopped = false;
                parked.clear();
            }
            Cmd::Encode {
                index,
                ptr,
                len,
                timestamp,
            } => {
                let Some(c) = config.as_ref() else {
                    emit(EncoderEvent::Error("frame before start".into()));
                    continue;
                };
                if draining || stopped {
                    parked.push(index);
                    continue;
                }
                let (w, h) = c.coded_size;
                let luma_len = (w * h) as usize;
                assert!(len >= luma_len * 3 / 2, "raw buffer too small for a frame");
                // SAFETY: the device lent `len` readable bytes at `ptr` until we
                // report InputBufferDone.
                let luma = unsafe { std::slice::from_raw_parts(ptr.as_ptr(), luma_len) };
                let digest = luma.iter().map(|b| *b as u32).sum::<u32>();
                let key = force_key || c.gop_size == 0 || frame_no as u32 % c.gop_size == 0;
                force_key = false;
                let mut bytes = Vec::new();
                let mut kind = if key {
                    FrameKind::Key
                } else {
                    FrameKind::Inter
                };
                if headers_due {
                    headers_due = false;
                    match c.header_mode {
                        VideoHeaderMode::Separate => ready.push_back(Packet {
                            bytes: headers_bytes(),
                            kind: FrameKind::Headers,
                            timestamp,
                            input: None,
                        }),
                        VideoHeaderMode::JoinedWith1stFrame => {
                            bytes.extend(headers_bytes());
                        }
                    }
                } else if key && c.prepend_sps_pps_to_idr {
                    bytes.extend(headers_bytes());
                }
                if key {
                    kind = FrameKind::Key;
                }
                bytes.extend(frame_bytes(key, frame_no, digest));
                frame_no = frame_no.wrapping_add(1);
                ready.push_back(Packet {
                    bytes,
                    kind,
                    timestamp,
                    input: Some(index),
                });
                pump(&mut ready, &mut captures, &mut draining, &mut stopped);
            }
            Cmd::UseCapture { index, ptr, len } => {
                captures.push_back((index, ptr, len));
                pump(&mut ready, &mut captures, &mut draining, &mut stopped);
            }
            Cmd::ForceKey => force_key = true,
            Cmd::Flush(ack) => {
                // Drop what is not written yet and what was parked behind the EOS, and say
                // nothing about either: the device returns or re-queues the raw frames itself
                // (the trait's `flush` contract), and the MediaCodec backend does exactly this,
                // because a report delivered after the device has unqueued a buffer could land
                // on one the guest has queued again (review-m7 R7-11).
                let dropped = ready.iter().filter(|p| p.input.is_some()).count() + parked.len();
                thread_log.lock().unwrap().dropped_by_flush += dropped;
                ready.clear();
                parked.clear();
                draining = false;
                stopped = false;
                let _ = ack.send(());
            }
            Cmd::Drain => {
                draining = true;
                pump(&mut ready, &mut captures, &mut draining, &mut stopped);
            }
            Cmd::Stop(ack) => {
                let _ = ack.send(());
                break;
            }
        }
    }
    thread_log.lock().unwrap().open = false;
}

impl VideoEncoderBackend for FakeBackend {
    type Session = FakeSession;

    fn capabilities(&self) -> &EncoderCapabilities {
        &self.caps
    }

    fn new_session(&mut self, _id: u32, sink: EncoderSink) -> IoctlResult<FakeSession> {
        let (events_tx, events) = mpsc::channel::<EncoderEvent>();
        Ok(FakeSession {
            sink,
            log: Arc::clone(&self.log),
            fail_start: self.fail_start,
            commands: None,
            events_tx,
            events,
            thread: None,
        })
    }

    fn close_session(&mut self, mut session: FakeSession) {
        session.stop();
    }
}

impl FakeSession {
    fn send(&self, cmd: Cmd) -> IoctlResult<()> {
        self.commands
            .as_ref()
            .ok_or(libc::EIO)?
            .send(cmd)
            .map_err(|_| libc::EIO)
    }

    fn rendezvous(&self, make: impl FnOnce(mpsc::Sender<()>) -> Cmd) {
        let (tx, rx) = mpsc::channel();
        if self.send(make(tx)).is_ok() {
            let _ = rx.recv();
        }
    }
}

impl VideoEncoderBackendSession for FakeSession {
    fn start(&mut self, config: &EncoderConfig) -> IoctlResult<()> {
        if let Some(errno) = self.fail_start {
            return Err(errno);
        }
        self.log.lock().unwrap().started.push(config.clone());
        if self.thread.is_none() {
            let (commands, rx) = mpsc::channel::<Cmd>();
            let (events_tx, sink, log) = (
                self.events_tx.clone(),
                self.sink.clone(),
                Arc::clone(&self.log),
            );
            self.thread = Some(thread::spawn(move || run_codec(rx, events_tx, sink, log)));
            self.commands = Some(commands);
            self.log.lock().unwrap().open = true;
        }
        self.send(Cmd::Start(config.clone()))
    }

    fn encode(&mut self, buffer: InputBuffer) -> IoctlResult<()> {
        self.send(Cmd::Encode {
            index: buffer.index,
            ptr: buffer.ptr,
            len: buffer.len,
            timestamp: buffer.timestamp,
        })
    }

    fn use_as_capture(&mut self, buffer: OutputBuffer) -> IoctlResult<()> {
        self.log.lock().unwrap().capture_active = true;
        self.send(Cmd::UseCapture {
            index: buffer.index,
            ptr: buffer.ptr,
            len: buffer.len,
        })
    }

    fn set_bitrate(&mut self, bitrate: u32) -> IoctlResult<()> {
        self.log.lock().unwrap().bitrates.push(bitrate);
        Ok(())
    }

    fn force_keyframe(&mut self) -> IoctlResult<()> {
        self.log.lock().unwrap().forced += 1;
        self.send(Cmd::ForceKey)
    }

    fn flush(&mut self) -> IoctlResult<()> {
        {
            let mut log = self.log.lock().unwrap();
            log.flushes += 1;
            if let Some(errno) = log.fail_flush.take() {
                return Err(errno);
            }
        }
        self.rendezvous(Cmd::Flush);
        Ok(())
    }

    fn drain(&mut self) -> IoctlResult<()> {
        if let Some(errno) = self.log.lock().unwrap().fail_drain.take() {
            return Err(errno);
        }
        self.send(Cmd::Drain)
    }

    fn stop(&mut self) {
        if let Some(thread) = self.thread.take() {
            self.rendezvous(Cmd::Stop);
            let _ = thread.join();
            self.commands = None;
            let mut log = self.log.lock().unwrap();
            log.stops += 1;
            log.capture_active = false;
        }
    }

    fn take_events(&mut self) -> Vec<EncoderEvent> {
        self.events.try_iter().collect()
    }
}

/// A session dropped without `stop` still joins its thread, as the real backend's does
/// (review-m4 R1).
impl Drop for FakeSession {
    fn drop(&mut self) {
        self.stop();
    }
}

// ---------------------------------------------------------------------------------------------
// The rig
// ---------------------------------------------------------------------------------------------

type Device = VideoEncoder<FakeBackend, EventLog, FakeGuest, FakeHostMapper, OrderedAllocator>;
type Session = VideoEncoderSession<FakeMapping, FakeSession>;

struct Rig {
    device: Device,
    events: Rc<RefCell<Vec<V4l2Event>>>,
    guest: FakeGuest,
    log: SharedLog,
}

const GUEST_MEMORY: usize = 8 << 20;
const SIZE: (u32, u32) = (320, 240);
const RAW_SIZEIMAGE: u32 = 320 * 240 * 3 / 2;

/// Three codecs with deliberately different ranges, so the union rules show.
fn caps() -> EncoderCapabilities {
    EncoderCapabilities {
        coded_formats: vec![
            CodedFormat {
                fourcc: H264,
                width: SizeRange::new(16, 4096, 2),
                height: SizeRange::new(16, 4096, 2),
                frame_rate: FrameRateRange { min: 1, max: 120 },
                bitrate: ControlRange::new(64_000, 100_000_000, 2_000_000),
                gop_size: ControlRange::new(0, 300, 30),
                bitrate_modes: vec![
                    VideoBitrateMode::VariableBitrate,
                    VideoBitrateMode::ConstantBitrate,
                ],
                profiles: vec![
                    VideoH264Profile::High as i32,
                    VideoH264Profile::Baseline as i32,
                    VideoH264Profile::Main as i32,
                ],
                levels: vec![
                    VideoH264Level::L4_1 as i32,
                    VideoH264Level::L3_0 as i32,
                    VideoH264Level::L4_0 as i32,
                    VideoH264Level::L5_1 as i32,
                ],
                qp: Some(QpRange { min: 0, max: 51 }),
            },
            CodedFormat {
                fourcc: HEVC,
                width: SizeRange::new(64, 8192, 8),
                height: SizeRange::new(64, 8192, 8),
                frame_rate: FrameRateRange { min: 1, max: 60 },
                bitrate: ControlRange::new(100_000, 200_000_000, 4_000_000),
                gop_size: ControlRange::new(1, 600, 60),
                bitrate_modes: vec![
                    VideoBitrateMode::VariableBitrate,
                    VideoBitrateMode::ConstantBitrate,
                    VideoBitrateMode::ConstantQuality,
                ],
                profiles: vec![VideoHEVCProfile::Main as i32],
                levels: vec![VideoHEVCLevel::L4_1 as i32],
                qp: Some(QpRange { min: 0, max: 51 }),
            },
            CodedFormat {
                fourcc: VP80,
                width: SizeRange::new(16, 4096, 2),
                height: SizeRange::new(16, 4096, 2),
                frame_rate: FrameRateRange { min: 1, max: 60 },
                bitrate: ControlRange::new(32_000, 50_000_000, 1_000_000),
                gop_size: ControlRange::new(0, 128, 30),
                bitrate_modes: vec![VideoBitrateMode::VariableBitrate],
                profiles: vec![0],
                levels: vec![],
                qp: None,
            },
        ],
        min_output_buffers: 4,
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
    let device = VideoEncoder::new(
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
    dequeued(events)
        .into_iter()
        .filter(|b| b.queue() == queue)
        .collect()
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

/// Drive `process_events` until `n` CAPTURE buffers have been collected, or time out.
fn collect_capture(r: &mut Rig, s: &mut Session, n: usize) {
    while dequeued_on(&r.events.borrow(), CAPTURE).len() < n {
        assert!(wait_ready(s), "no CAPTURE buffer within 2s");
        process(&mut r.device, s);
    }
}

/// Drive `process_events` until `n` OUTPUT buffers have come back.
fn collect_output(r: &mut Rig, s: &mut Session, n: usize) {
    while dequeued_on(&r.events.borrow(), OUTPUT).len() < n {
        assert!(wait_ready(s), "no OUTPUT buffer within 2s");
        process(&mut r.device, s);
    }
}

// buffer builders -----------------------------------------------------------------------------

fn mmap_buffer(queue: QueueType, index: u32, len: u32) -> V4l2Buffer {
    let mut buffer = V4l2Buffer::new(queue, index, MemoryType::Mmap);
    *buffer.get_first_plane_mut().length = len;
    *buffer.get_first_plane_mut().bytesused = if queue == OUTPUT { len } else { 0 };
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

/// Fill the OUTPUT MMAP buffer `index` with a frame whose luma is `luma` (chroma 0x80).
fn fill_mmap_output(s: &mut Session, index: usize, luma: u8) {
    let (w, h) = s.coded_size;
    if let Backing::Host { buffer, .. } = &mut s.input.buffers[index].backing {
        let luma_len = (w * h) as usize;
        // SAFETY: no guest mapping of this buffer exists; the device is not streaming it yet.
        unsafe {
            std::ptr::write_bytes(buffer.as_mut_ptr(), luma, luma_len);
            std::ptr::write_bytes(buffer.as_mut_ptr().add(luma_len), 0x80, luma_len / 2);
        }
    } else {
        panic!("not a host-owned OUTPUT buffer");
    }
}

/// The first `len` bytes of the CAPTURE MMAP buffer `index`, after it was dequeued.
fn capture_bytes(s: &Session, index: usize, len: usize) -> Vec<u8> {
    if let Backing::Host { buffer, .. } = &s.output.buffers[index].backing {
        // SAFETY: the buffer was dequeued, so the backend is done with it.
        unsafe { std::slice::from_raw_parts(buffer.as_ptr(), len).to_vec() }
    } else {
        panic!("not a host-owned CAPTURE buffer");
    }
}

/// Queue a frame with luma `luma` and timestamp `n` on OUTPUT buffer `index` (MMAP).
fn queue_frame(r: &mut Rig, s: &mut Session, index: u32, luma: u8, n: i64) {
    fill_mmap_output(s, index as usize, luma);
    let mut ob = mmap_buffer(OUTPUT, index, RAW_SIZEIMAGE);
    ob.set_timestamp(ts(n));
    r.device.qbuf(s, ob, vec![], PayloadValidity::ALL).unwrap();
}

/// A timestamp that carries `n` so a CAPTURE buffer can be matched to the OUTPUT buffer it came
/// from (`V4L2_BUF_FLAG_TIMESTAMP_COPY`).
fn ts(n: i64) -> bindings::timeval {
    bindings::timeval {
        tv_sec: n as bindings::time_t,
        tv_usec: 0 as bindings::suseconds_t,
    }
}

/// The fields of a multi-planar format the tests look at, copied out of the packed struct so they
/// can be compared by reference.
#[derive(Debug, PartialEq, Eq)]
struct Pix {
    width: u32,
    height: u32,
    pixelformat: u32,
    num_planes: u8,
    bytesperline: u32,
    sizeimage: u32,
    colorspace: u32,
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
        colorspace: mp.colorspace,
    }
}

fn output_format(fourcc: PixelFormat, w: u32, h: u32) -> v4l2_format {
    let pix_mp = bindings::v4l2_pix_format_mplane {
        width: w,
        height: h,
        pixelformat: fourcc.to_u32(),
        num_planes: 1,
        ..Default::default()
    };
    v4l2_format {
        type_: OUTPUT as u32,
        fmt: bindings::v4l2_format__bindgen_ty_1 { pix_mp },
    }
}

fn capture_format(fourcc: PixelFormat, sizeimage: u32) -> v4l2_format {
    capture_format_sized(fourcc, 1, sizeimage)
}

fn capture_format_sized(fourcc: PixelFormat, num_planes: u8, sizeimage: u32) -> v4l2_format {
    let mut pix_mp = bindings::v4l2_pix_format_mplane {
        width: 0,
        height: 0,
        pixelformat: fourcc.to_u32(),
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

fn parm(queue: QueueType, numerator: u32, denominator: u32) -> v4l2_streamparm {
    v4l2_streamparm {
        type_: queue as u32,
        parm: bindings::v4l2_streamparm__bindgen_ty_1 {
            output: bindings::v4l2_outputparm {
                capability: 0,
                outputmode: 0,
                timeperframe: bindings::v4l2_fract {
                    numerator,
                    denominator,
                },
                extendedmode: 0,
                writebuffers: 0,
                reserved: [0; 4],
            },
        },
    }
}

/// `(numerator, denominator, capability)` of an OUTPUT streamparm.
fn timeperframe(p: &v4l2_streamparm) -> (u32, u32, u32) {
    // SAFETY: output.
    let o = unsafe { p.parm.output };
    (
        o.timeperframe.numerator,
        o.timeperframe.denominator,
        o.capability,
    )
}

fn enc_cmd(cmd: u32) -> v4l2_encoder_cmd {
    v4l2_encoder_cmd {
        cmd,
        flags: 0,
        __bindgen_anon_1: bindings::v4l2_encoder_cmd__bindgen_ty_1 {
            raw: bindings::v4l2_encoder_cmd__bindgen_ty_1__bindgen_ty_1 { data: [0; 8] },
        },
    }
}

// control builders ----------------------------------------------------------------------------

fn ext_controls(class: u32, count: usize) -> v4l2_ext_controls {
    v4l2_ext_controls {
        __bindgen_anon_1: bindings::v4l2_ext_controls__bindgen_ty_1 { ctrl_class: class },
        count: count as u32,
        error_idx: 0,
        request_fd: 0,
        reserved: [0],
        controls: std::ptr::null_mut(),
    }
}

fn ext_control(id: u32, value: i32) -> v4l2_ext_control {
    v4l2_ext_control {
        id,
        size: 0,
        reserved2: [0],
        __bindgen_anon_1: bindings::v4l2_ext_control__bindgen_ty_1 { value },
    }
}

fn ext_value(c: &v4l2_ext_control) -> i32 {
    let anon = c.__bindgen_anon_1;
    // SAFETY: plain value controls.
    unsafe { anon.value }
}

/// `S_EXT_CTRLS` of one control the way ffmpeg does it (`ctrl_class = V4L2_CTRL_CLASS_MPEG`,
/// `v4l2_m2m_enc.c:53-72`): the errno, or the value the control now holds.
fn s_ext(r: &mut Rig, s: &mut Session, id: u32, value: i32) -> Result<i32, i32> {
    let mut ctrls = ext_controls(CODEC_CLASS, 1);
    let mut arr = vec![ext_control(id, value)];
    r.device
        .s_ext_ctrls(
            s,
            CtrlWhich::Class(CODEC_CLASS),
            &mut ctrls,
            &mut arr,
            vec![],
        )
        .map(|()| ext_value(&arr[0]))
}

fn g_ext(r: &mut Rig, s: &Session, id: u32) -> Result<i32, i32> {
    let mut ctrls = ext_controls(CODEC_CLASS, 1);
    let mut arr = vec![ext_control(id, 0)];
    r.device
        .g_ext_ctrls(
            s,
            CtrlWhich::Class(CODEC_CLASS),
            &mut ctrls,
            &mut arr,
            vec![],
        )
        .map(|()| ext_value(&arr[0]))
}

fn query(r: &mut Rig, s: &Session, id: u32) -> Result<v4l2_query_ext_ctrl, i32> {
    let (id, flags) = v4l2r::ioctl::parse_ctrl_id_and_flags(id);
    r.device.query_ext_ctrl(s, id, flags)
}

fn query_name(q: &v4l2_query_ext_ctrl) -> String {
    let bytes: Vec<u8> = q
        .name
        .iter()
        .take_while(|c| **c != 0)
        .map(|c| *c as u8)
        .collect();
    String::from_utf8(bytes).unwrap()
}

fn menu_name(m: &v4l2_querymenu) -> String {
    // SAFETY: a menu control's name.
    let name = unsafe { m.__bindgen_anon_1.name };
    let bytes: Vec<u8> = name.iter().take_while(|c| **c != 0).copied().collect();
    String::from_utf8(bytes).unwrap()
}

/// Walk `QUERY_EXT_CTRL` with `NEXT_CTRL`, as `v4l2-compliance` does, and return the ids.
fn enumerate_controls(r: &mut Rig, s: &Session) -> Vec<u32> {
    let mut out = Vec::new();
    let mut id = 0;
    loop {
        match query(r, s, id | bindings::V4L2_CTRL_FLAG_NEXT_CTRL) {
            Ok(q) => {
                assert!(q.id > id, "id did not increase");
                id = q.id;
                out.push(id);
            }
            Err(e) => {
                assert_eq!(e, libc::EINVAL);
                break;
            }
        }
    }
    out
}

/// Take a session from OPEN to both queues streaming at 320x240 H.264, MMAP buffers (four on
/// each queue, the CAPTURE ones queued), and return the CAPTURE `sizeimage`.
fn start_streaming(r: &mut Rig, s: &mut Session) -> u32 {
    r.device.s_fmt(s, CAPTURE, capture_format(H264, 0)).unwrap();
    r.device
        .s_fmt(s, OUTPUT, output_format(NV12, SIZE.0, SIZE.1))
        .unwrap();
    r.device
        .subscribe_event(s, EventType::Eos, SubscribeEventFlags::empty())
        .unwrap();
    r.device.reqbufs(s, OUTPUT, MemoryType::Mmap, 4).unwrap();
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
    r.device.streamon(s, OUTPUT).unwrap();
    r.device.streamon(s, CAPTURE).unwrap();
    assert_eq!(
        r.log.lock().unwrap().started.len(),
        1,
        "codec created once both stream"
    );
    sizeimage
}

// ---------------------------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------------------------

/// The V4L2 surface an encoder advertises: the backend's coded formats on CAPTURE (nothing
/// hard-coded), NV12 on OUTPUT, stepwise frame sizes, frame intervals from the backend's rates
/// (a size the format does not accept is EINVAL, as `v4l2-compliance` probes), G/S_PARM on
/// OUTPUT and ENOTTY on CAPTURE, the default formats and selection rectangles.
#[test]
fn formats_are_the_backends_coded_formats_and_nv12() {
    let mut r = rig();
    let s = session(&mut r.device);

    // CAPTURE: the backend's three coded formats, compressed, in its order.
    let cap: Vec<(u32, u32)> = (0..)
        .map_while(|i| r.device.enum_fmt(&s, CAPTURE, i).ok())
        .map(|d| (d.pixelformat, d.flags))
        .collect();
    assert_eq!(
        cap,
        vec![
            (H264.to_u32(), bindings::V4L2_FMT_FLAG_COMPRESSED),
            (HEVC.to_u32(), bindings::V4L2_FMT_FLAG_COMPRESSED),
            (VP80.to_u32(), bindings::V4L2_FMT_FLAG_COMPRESSED),
        ]
    );
    assert!(r
        .device
        .enum_fmt(&s, CAPTURE, 0)
        .unwrap()
        .description
        .starts_with(b"H.264\0"));
    // OUTPUT: NV12 only.
    let out = r.device.enum_fmt(&s, OUTPUT, 0).unwrap();
    assert_eq!((out.pixelformat, out.flags), (NV12.to_u32(), 0));
    assert!(out.description.starts_with(b"Y/UV 4:2:0\0"));
    assert_eq!(r.device.enum_fmt(&s, OUTPUT, 1).err(), Some(libc::EINVAL));

    // Stepwise frame sizes: a coded format's own; NV12 follows the current CAPTURE format.
    let fs = r.device.enum_framesizes(&s, 0, HEVC.to_u32()).unwrap();
    assert_eq!(
        fs.type_,
        bindings::v4l2_frmsizetypes_V4L2_FRMSIZE_TYPE_STEPWISE
    );
    // SAFETY: stepwise.
    let sw = unsafe { fs.__bindgen_anon_1.stepwise };
    assert_eq!((sw.min_width, sw.max_width, sw.step_width), (64, 8192, 8));
    let fs = r.device.enum_framesizes(&s, 0, NV12.to_u32()).unwrap();
    // SAFETY: stepwise.
    let sw = unsafe { fs.__bindgen_anon_1.stepwise };
    assert_eq!(
        (sw.min_width, sw.max_width, sw.step_width),
        (16, 4096, 2),
        "H264's, the default"
    );
    assert_eq!(
        r.device.enum_framesizes(&s, 1, H264.to_u32()).err(),
        Some(libc::EINVAL)
    );
    assert_eq!(
        r.device.enum_framesizes(&s, 0, 0x1234_5678).err(),
        Some(libc::EINVAL)
    );

    // Frame intervals: one continuous range 1/120 .. 1/1 at an accepted size; EINVAL off the
    // grid (`v4l2-test-formats.cpp:195-210` probes min-1 and max+1).
    let fi = r
        .device
        .enum_frameintervals(&s, 0, H264.to_u32(), 16, 16)
        .unwrap();
    assert_eq!(
        fi.type_,
        bindings::v4l2_frmivaltypes_V4L2_FRMIVAL_TYPE_CONTINUOUS
    );
    // SAFETY: stepwise/continuous.
    let sw = unsafe { fi.__bindgen_anon_1.stepwise };
    assert_eq!((sw.min.numerator, sw.min.denominator), (1, 120));
    assert_eq!((sw.max.numerator, sw.max.denominator), (1, 1));
    assert_eq!((sw.step.numerator, sw.step.denominator), (1, 1));
    assert!(r
        .device
        .enum_frameintervals(&s, 0, H264.to_u32(), 4096, 4096)
        .is_ok());
    assert!(r
        .device
        .enum_frameintervals(&s, 0, NV12.to_u32(), 320, 240)
        .is_ok());
    assert_eq!(
        r.device
            .enum_frameintervals(&s, 1, H264.to_u32(), 16, 16)
            .err(),
        Some(libc::EINVAL)
    );
    assert_eq!(
        r.device
            .enum_frameintervals(&s, 0, H264.to_u32(), 15, 16)
            .err(),
        Some(libc::EINVAL)
    );
    assert_eq!(
        r.device
            .enum_frameintervals(&s, 0, H264.to_u32(), 4096, 4097)
            .err(),
        Some(libc::EINVAL)
    );
    assert_eq!(
        r.device
            .enum_frameintervals(&s, 0, H264.to_u32(), 17, 16)
            .err(),
        Some(libc::EINVAL)
    );

    // An encoder has a frame rate: G_PARM(OUTPUT) answers with TIMEPERFRAME and the default
    // 1/30; the CAPTURE side is ENOTTY for both G and S (`v4l2-test-formats.cpp:1418-1441`).
    assert_eq!(
        timeperframe(&r.device.g_parm(&s, OUTPUT).unwrap()),
        (1, 30, bindings::V4L2_CAP_TIMEPERFRAME)
    );
    assert_eq!(r.device.g_parm(&s, CAPTURE).err(), Some(libc::ENOTTY));

    // Default formats: NV12 640x480 tightly packed on OUTPUT, H264 at the same size on CAPTURE
    // with the device's bitstream buffer size.
    let ofmt = pix(&r.device.g_fmt(&s, OUTPUT).unwrap());
    assert_eq!(ofmt.pixelformat, NV12.to_u32());
    assert_eq!((ofmt.width, ofmt.height, ofmt.num_planes), (640, 480, 1));
    assert_eq!(ofmt.bytesperline, 640);
    assert_eq!(ofmt.sizeimage, 640 * 480 * 3 / 2);
    let cfmt = pix(&r.device.g_fmt(&s, CAPTURE).unwrap());
    assert_eq!(cfmt.pixelformat, H264.to_u32());
    assert_eq!((cfmt.width, cfmt.height, cfmt.bytesperline), (640, 480, 0));
    assert_eq!(cfmt.sizeimage, DEFAULT_BITSTREAM_FLOOR);

    // The OUTPUT crop is the whole frame; CAPTURE selection and composition are refused
    // (`v4l2-test-formats.cpp:1746`, `:1841`).
    let crop = r
        .device
        .g_selection(&s, SelectionType::Output, SelectionTarget::Crop)
        .unwrap();
    assert_eq!(
        (crop.left, crop.top, crop.width, crop.height),
        (0, 0, 640, 480)
    );
    for target in [SelectionTarget::CropBounds, SelectionTarget::CropDefault] {
        let b = r
            .device
            .g_selection(&s, SelectionType::Output, target)
            .unwrap();
        assert_eq!((b.width, b.height), (640, 480));
    }
    assert_eq!(
        r.device
            .g_selection(&s, SelectionType::Capture, SelectionTarget::Crop)
            .err(),
        Some(libc::EINVAL)
    );
    assert_eq!(
        r.device
            .g_selection(&s, SelectionType::Output, SelectionTarget::Compose)
            .err(),
        Some(libc::EINVAL)
    );

    close(&mut r.device, s);
}

/// `S_FMT(CAPTURE)` selects the codec and sizes the bitstream buffers, `S_FMT(OUTPUT)` the raw
/// size (aligned up, the crop reset, the colorimetry propagated to CAPTURE), and the `EBUSY`
/// rules of the kernel's commit points hold. The size/crop half is `v4l2-compliance`'s
/// `testM2MFormats` for a stateful encoder (`v4l2-test-formats.cpp:963-1019`).
#[test]
fn s_fmt_capture_selects_the_codec_and_output_the_raw_size() {
    let mut r = rig();
    let mut s = session(&mut r.device);

    // An unknown coded fourcc snaps to the first format; width/height are read-only.
    let tried = pix(&r
        .device
        .try_fmt(
            &s,
            CAPTURE,
            capture_format(PixelFormat::from_fourcc(b"XXXX"), 0),
        )
        .unwrap());
    assert_eq!(tried.pixelformat, H264.to_u32());
    assert_eq!((tried.width, tried.height), (640, 480));

    // HEVC with a client-sized bitstream buffer; the raw size is refitted to HEVC's step of 8.
    let set = pix(&r
        .device
        .s_fmt(&mut s, CAPTURE, capture_format(HEVC, 1 << 20))
        .unwrap());
    assert_eq!(set.pixelformat, HEVC.to_u32());
    assert_eq!(set.sizeimage, 1 << 20);
    assert_eq!((set.width, set.height), (640, 480));
    // A bitstream size below the floor is replaced by the device's default.
    let set = pix(&r
        .device
        .s_fmt(&mut s, CAPTURE, capture_format(H264, 100))
        .unwrap());
    assert_eq!(set.sizeimage, DEFAULT_BITSTREAM_FLOOR);

    // The raw size: an odd request is aligned up, the crop follows, CAPTURE reports it.
    let mut fmt = output_format(NV12, 635, 475);
    // Writing a packed field of a local union member is safe; only reads need `unsafe`.
    fmt.fmt.pix_mp.colorspace = bindings::v4l2_colorspace_V4L2_COLORSPACE_SMPTE170M;
    let set = pix(&r.device.s_fmt(&mut s, OUTPUT, fmt).unwrap());
    assert_eq!((set.width, set.height), (636, 476));
    assert_eq!(set.bytesperline, 636);
    assert_eq!(set.sizeimage, 636 * 476 * 3 / 2);
    assert_eq!(
        set.colorspace,
        bindings::v4l2_colorspace_V4L2_COLORSPACE_SMPTE170M
    );
    let cfmt = pix(&r.device.g_fmt(&s, CAPTURE).unwrap());
    assert_eq!((cfmt.width, cfmt.height), (636, 476));
    assert_eq!(
        cfmt.colorspace,
        bindings::v4l2_colorspace_V4L2_COLORSPACE_SMPTE170M,
        "propagated"
    );
    let crop = r
        .device
        .g_selection(&s, SelectionType::Output, SelectionTarget::Crop)
        .unwrap();
    assert_eq!(
        (crop.left, crop.top, crop.width, crop.height),
        (0, 0, 636, 476)
    );
    // A YU12 request snaps to NV12: the pixel format is not negotiable.
    let set = pix(&r
        .device
        .s_fmt(
            &mut s,
            OUTPUT,
            output_format(PixelFormat::from_fourcc(b"YU12"), 320, 240),
        )
        .unwrap());
    assert_eq!(set.pixelformat, NV12.to_u32());
    assert_eq!((set.width, set.height), (320, 240));

    // A smaller crop is honoured; bounds/default stay the frame; read-only targets and the
    // CAPTURE side are refused by S_SELECTION.
    let rect = bindings::v4l2_rect {
        left: 0,
        top: 0,
        width: 316,
        height: 236,
    };
    let got = r
        .device
        .s_selection(
            &mut s,
            SelectionType::Output,
            SelectionTarget::Crop,
            rect,
            SelectionFlags::empty(),
        )
        .unwrap();
    assert_eq!((got.width, got.height), (316, 236));
    let got = r
        .device
        .g_selection(&s, SelectionType::Output, SelectionTarget::Crop)
        .unwrap();
    assert_eq!((got.width, got.height), (316, 236));
    let b = r
        .device
        .g_selection(&s, SelectionType::Output, SelectionTarget::CropBounds)
        .unwrap();
    assert_eq!((b.width, b.height), (320, 240));
    for target in [
        SelectionTarget::CropBounds,
        SelectionTarget::CropDefault,
        SelectionTarget::Compose,
    ] {
        assert_eq!(
            r.device
                .s_selection(
                    &mut s,
                    SelectionType::Output,
                    target,
                    rect,
                    SelectionFlags::empty()
                )
                .err(),
            Some(libc::EINVAL)
        );
    }
    assert_eq!(
        r.device
            .s_selection(
                &mut s,
                SelectionType::Capture,
                SelectionTarget::Crop,
                rect,
                SelectionFlags::empty()
            )
            .err(),
        Some(libc::EINVAL)
    );
    // An oversized crop is held inside the frame.
    let rect = bindings::v4l2_rect {
        left: 10,
        top: 10,
        width: 1000,
        height: 1000,
    };
    let got = r
        .device
        .s_selection(
            &mut s,
            SelectionType::Output,
            SelectionTarget::Crop,
            rect,
            SelectionFlags::empty(),
        )
        .unwrap();
    assert_eq!(
        (got.left, got.top, got.width, got.height),
        (10, 10, 310, 230)
    );

    // Commit points: CAPTURE buffers block S_FMT(CAPTURE) but not S_FMT(OUTPUT); OUTPUT buffers
    // block both; TRY_FMT always answers.
    r.device
        .reqbufs(&mut s, CAPTURE, MemoryType::Mmap, 2)
        .unwrap();
    assert_eq!(
        r.device
            .s_fmt(&mut s, CAPTURE, capture_format(HEVC, 0))
            .err(),
        Some(libc::EBUSY)
    );
    assert!(r
        .device
        .s_fmt(&mut s, OUTPUT, output_format(NV12, 320, 240))
        .is_ok());
    r.device
        .reqbufs(&mut s, OUTPUT, MemoryType::Mmap, 2)
        .unwrap();
    assert_eq!(
        r.device
            .s_fmt(&mut s, OUTPUT, output_format(NV12, 640, 480))
            .err(),
        Some(libc::EBUSY)
    );
    assert!(r
        .device
        .try_fmt(&s, OUTPUT, output_format(NV12, 640, 480))
        .is_ok());
    assert!(r
        .device
        .try_fmt(&s, CAPTURE, capture_format(HEVC, 0))
        .is_ok());

    close(&mut r.device, s);
}

/// `S_PARM(OUTPUT)` sets the frame rate the codec is created with: a fraction the format
/// sustains is kept as given, one outside the range snaps to the nearest end, `0/x` and `x/0`
/// mean the default (`v4l2-test-formats.cpp:1502-1521` sends both); `S_PARM(CAPTURE)` is
/// ENOTTY. The rate reaches the backend in the config.
#[test]
fn s_parm_output_sets_the_frame_rate() {
    let mut r = rig();
    let mut s = session(&mut r.device);

    let p = r.device.s_parm(&mut s, parm(OUTPUT, 1001, 30000)).unwrap();
    assert_eq!(
        timeperframe(&p),
        (1001, 30000, bindings::V4L2_CAP_TIMEPERFRAME)
    );
    assert_eq!(
        timeperframe(&r.device.g_parm(&s, OUTPUT).unwrap()),
        (1001, 30000, bindings::V4L2_CAP_TIMEPERFRAME)
    );
    // 240 fps is above H264's 120.
    assert_eq!(
        timeperframe(&r.device.s_parm(&mut s, parm(OUTPUT, 1, 240)).unwrap()).1,
        120
    );
    // 0/1 and 1/0: the default.
    assert_eq!(
        timeperframe(&r.device.s_parm(&mut s, parm(OUTPUT, 0, 1)).unwrap()),
        (1, 30, bindings::V4L2_CAP_TIMEPERFRAME)
    );
    assert_eq!(
        timeperframe(&r.device.s_parm(&mut s, parm(OUTPUT, 1, 0)).unwrap()),
        (1, 30, bindings::V4L2_CAP_TIMEPERFRAME)
    );
    assert_eq!(
        r.device.s_parm(&mut s, parm(CAPTURE, 1, 30)).err(),
        Some(libc::ENOTTY)
    );

    // The rate set last is what the codec is created with.
    r.device.s_parm(&mut s, parm(OUTPUT, 1, 60)).unwrap();
    start_streaming(&mut r, &mut s);
    let config = r.log.lock().unwrap().started[0].clone();
    assert_eq!(config.frame_rate, FrameRate { num: 60, den: 1 });
    assert_eq!(config.coded_format, H264);
    assert_eq!(config.coded_size, SIZE);

    close(&mut r.device, s);
}

/// The control table is built from the capabilities and enumerated the way `v4l2-compliance`
/// walks it (`v4l2-test-controls.cpp:176-328`): class controls first in each class, ids
/// increasing, ranges the union of the formats', menus the formats' items with holes, names and
/// flags the kernel's, `NEXT_COMPOUND` finding nothing, `QUERYMENU` only on menus.
#[test]
fn controls_are_enumerated_from_the_capabilities() {
    let mut r = rig();
    let s = session(&mut r.device);

    let ids = enumerate_controls(&mut r, &s);
    assert_eq!(
        ids,
        vec![
            bindings::V4L2_CID_USER_CLASS,
            CID_MIN_OUT,
            bindings::V4L2_CID_CODEC_CLASS,
            CID_GOP,
            CID_BITRATE_MODE,
            CID_BITRATE,
            CID_HEADER_MODE,
            CID_FORCE_KEY,
            CID_H264_MIN_QP,
            CID_H264_MAX_QP,
            CID_H264_LEVEL,
            CID_H264_PROFILE,
            CID_VP8_PROFILE,
            CID_HEVC_MIN_QP,
            CID_HEVC_MAX_QP,
            CID_HEVC_PROFILE,
            CID_HEVC_LEVEL,
            CID_PREPEND,
        ]
    );
    // Every id also answers a direct query, with the same fields as the walk gave.
    for id in &ids {
        assert_eq!(query(&mut r, &s, *id).unwrap().id, *id);
    }
    // Nothing is a compound control; an unknown id is EINVAL; so is MIN_BUFFERS_FOR_CAPTURE,
    // which an encoder must not have (`v4l2-test-controls.cpp:1180`).
    assert_eq!(
        query(&mut r, &s, bindings::V4L2_CTRL_FLAG_NEXT_COMPOUND).err(),
        Some(libc::EINVAL)
    );
    assert_eq!(
        query(&mut r, &s, 0x0098_0001 + 0x100).err(),
        Some(libc::EINVAL)
    );
    assert_eq!(query(&mut r, &s, CID_MIN_CAP).err(), Some(libc::EINVAL));
    assert_eq!(query(&mut r, &s, CID_B_FRAMES).err(), Some(libc::EINVAL));

    // The class controls: read- and write-only, zero bounds.
    let class = query(&mut r, &s, bindings::V4L2_CID_CODEC_CLASS).unwrap();
    assert_eq!(
        class.type_,
        bindings::v4l2_ctrl_type_V4L2_CTRL_TYPE_CTRL_CLASS
    );
    assert_eq!(
        class.flags,
        bindings::V4L2_CTRL_FLAG_READ_ONLY | bindings::V4L2_CTRL_FLAG_WRITE_ONLY
    );
    assert_eq!(query_name(&class), "Codec Controls");
    assert_eq!(
        (
            class.minimum,
            class.maximum,
            class.step,
            class.default_value
        ),
        (0, 0, 0, 0)
    );

    // BITRATE: the union of the three ranges, the first format's default.
    let q = query(&mut r, &s, CID_BITRATE).unwrap();
    assert_eq!(q.type_, bindings::v4l2_ctrl_type_V4L2_CTRL_TYPE_INTEGER);
    assert_eq!(
        (q.minimum, q.maximum, q.step, q.default_value),
        (32_000, 200_000_000, 1, 2_000_000)
    );
    assert_eq!(query_name(&q), "Video Bitrate");
    assert_eq!((q.elem_size, q.elems, q.nr_of_dims), (4, 1, 0));
    let q = query(&mut r, &s, CID_GOP).unwrap();
    assert_eq!((q.minimum, q.maximum, q.default_value), (0, 600, 30));

    // BITRATE_MODE: VBR, CBR and CQ between them, VBR default; every item named.
    let q = query(&mut r, &s, CID_BITRATE_MODE).unwrap();
    assert_eq!(q.type_, bindings::v4l2_ctrl_type_V4L2_CTRL_TYPE_MENU);
    assert_eq!(
        (q.minimum, q.maximum, q.step, q.default_value),
        (0, 2, 1, 0)
    );
    let names: Vec<String> = (0..=2)
        .map(|i| menu_name(&r.device.querymenu(&s, CID_BITRATE_MODE, i).unwrap()))
        .collect();
    assert_eq!(
        names,
        vec!["Variable Bitrate", "Constant Bitrate", "Constant Quality"]
    );
    assert_eq!(
        r.device.querymenu(&s, CID_BITRATE_MODE, 3).err(),
        Some(libc::EINVAL)
    );

    // H264_PROFILE: Baseline(0)..High(4) with holes at 1 and 3, default High (the backend's
    // first); QUERYMENU names the kernel's strings and refuses the holes.
    let q = query(&mut r, &s, CID_H264_PROFILE).unwrap();
    assert_eq!(
        (q.minimum, q.maximum, q.default_value),
        (0, 4, VideoH264Profile::High as i64)
    );
    assert_eq!(
        menu_name(&r.device.querymenu(&s, CID_H264_PROFILE, 4).unwrap()),
        "High"
    );
    assert_eq!(
        menu_name(&r.device.querymenu(&s, CID_H264_PROFILE, 0).unwrap()),
        "Baseline"
    );
    assert_eq!(
        r.device.querymenu(&s, CID_H264_PROFILE, 1).err(),
        Some(libc::EINVAL)
    );
    assert_eq!(
        r.device.querymenu(&s, CID_H264_PROFILE, 3).err(),
        Some(libc::EINVAL)
    );
    assert_eq!(
        r.device.querymenu(&s, CID_H264_PROFILE, 5).err(),
        Some(libc::EINVAL)
    );
    let q = query(&mut r, &s, CID_H264_LEVEL).unwrap();
    assert_eq!(
        (q.minimum, q.maximum, q.default_value),
        (
            VideoH264Level::L3_0 as i64,
            VideoH264Level::L5_1 as i64,
            VideoH264Level::L4_1 as i64
        )
    );
    assert_eq!(
        menu_name(
            &r.device
                .querymenu(&s, CID_H264_LEVEL, VideoH264Level::L4_1 as u32)
                .unwrap()
        ),
        "4.1"
    );
    let q = query(&mut r, &s, CID_HEVC_PROFILE).unwrap();
    assert_eq!((q.minimum, q.maximum), (0, 0));
    assert_eq!(query_name(&q), "HEVC Profile");
    assert_eq!(
        menu_name(
            &r.device
                .querymenu(&s, CID_HEVC_LEVEL, VideoHEVCLevel::L4_1 as u32)
                .unwrap()
        ),
        "4.1"
    );
    assert_eq!(
        menu_name(&r.device.querymenu(&s, CID_VP8_PROFILE, 0).unwrap()),
        "0"
    );

    // QP: the codec's range, MIN starting at the bottom and MAX at the top.
    let q = query(&mut r, &s, CID_H264_MIN_QP).unwrap();
    assert_eq!((q.minimum, q.maximum, q.default_value), (0, 51, 0));
    let q = query(&mut r, &s, CID_HEVC_MAX_QP).unwrap();
    assert_eq!((q.minimum, q.maximum, q.default_value), (0, 51, 51));
    assert_eq!(query_name(&q), "HEVC Maximum QP Value");

    // The button: write-only, executes on write, zero bounds, no menu.
    let q = query(&mut r, &s, CID_FORCE_KEY).unwrap();
    assert_eq!(q.type_, bindings::v4l2_ctrl_type_V4L2_CTRL_TYPE_BUTTON);
    assert_eq!(
        q.flags,
        bindings::V4L2_CTRL_FLAG_WRITE_ONLY | bindings::V4L2_CTRL_FLAG_EXECUTE_ON_WRITE
    );
    assert_eq!(
        r.device.querymenu(&s, CID_FORCE_KEY, 0).err(),
        Some(libc::EINVAL)
    );
    // HEADER_MODE defaults to joined; PREPEND is a boolean off.
    let q = query(&mut r, &s, CID_HEADER_MODE).unwrap();
    assert_eq!(
        (q.minimum, q.maximum, q.default_value),
        (0, 1, VideoHeaderMode::JoinedWith1stFrame as i64)
    );
    let q = query(&mut r, &s, CID_PREPEND).unwrap();
    assert_eq!(
        (q.type_, q.minimum, q.maximum, q.default_value),
        (bindings::v4l2_ctrl_type_V4L2_CTRL_TYPE_BOOLEAN, 0, 1, 0)
    );
    // MIN_BUFFERS_FOR_OUTPUT: read-only, the backend's number.
    let q = query(&mut r, &s, CID_MIN_OUT).unwrap();
    assert_eq!(q.flags, bindings::V4L2_CTRL_FLAG_READ_ONLY);
    assert_eq!(q.default_value, 4);
    assert_eq!(query_name(&q), "Min Number of Output Buffers");

    // The old QUERYCTRL says the same, and walks the same list.
    let (id, flags) = v4l2r::ioctl::parse_ctrl_id_and_flags(CID_BITRATE);
    let q = r.device.queryctrl(&s, id, flags).unwrap();
    assert_eq!(
        (q.minimum, q.maximum, q.step, q.default_value),
        (32_000, 200_000_000, 1, 2_000_000)
    );
    assert!(q.name.starts_with(b"Video Bitrate\0"));
    let (id, flags) = v4l2r::ioctl::parse_ctrl_id_and_flags(bindings::V4L2_CTRL_FLAG_NEXT_CTRL);
    assert_eq!(
        r.device.queryctrl(&s, id, flags).unwrap().id,
        bindings::V4L2_CID_USER_CLASS
    );

    close(&mut r.device, s);
}

/// Getting, setting and trying controls, and what is refused: the rules `v4l2-compliance`'s
/// `testSimpleControls` / `testExtendedControls` check (`v4l2-test-controls.cpp:434-589`,
/// `:846-1109`) -- clamping, ERANGE / EINVAL on menus, EACCES on read-only and write-only,
/// EINVAL on a class or an unknown id, `error_idx` per ioctl, `which` handling, and a set that
/// fails leaving nothing applied.
#[test]
fn controls_get_set_try_and_reject_invalid_values() {
    let mut r = rig();
    let mut s = session(&mut r.device);

    // Defaults through G_CTRL and G_EXT_CTRLS.
    assert_eq!(r.device.g_ctrl(&s, CID_BITRATE).unwrap().value, 2_000_000);
    assert_eq!(
        g_ext(&mut r, &s, CID_H264_LEVEL).unwrap(),
        VideoH264Level::L4_1 as i32
    );
    assert_eq!(r.device.g_ctrl(&s, CID_MIN_OUT).unwrap().value, 4);
    // Class: EINVAL; button: EACCES (write-only); unknown: EINVAL.
    assert_eq!(
        r.device.g_ctrl(&s, bindings::V4L2_CID_CODEC_CLASS).err(),
        Some(libc::EINVAL)
    );
    assert_eq!(r.device.g_ctrl(&s, CID_FORCE_KEY).err(), Some(libc::EACCES));
    assert_eq!(r.device.g_ctrl(&s, 0).err(), Some(libc::EINVAL));
    assert_eq!(r.device.s_ctrl(&mut s, 0, 0).err(), Some(libc::EINVAL));
    assert_eq!(
        r.device
            .s_ctrl(&mut s, bindings::V4L2_CID_CODEC_CLASS, 0)
            .err(),
        Some(libc::EINVAL)
    );
    assert_eq!(
        r.device.s_ctrl(&mut s, CID_MIN_OUT, 4).err(),
        Some(libc::EACCES)
    );

    // Integers clamp (the kernel's behaviour, which compliance accepts alongside ERANGE).
    assert_eq!(
        r.device.s_ctrl(&mut s, CID_BITRATE, 1_000).unwrap().value,
        32_000
    );
    assert_eq!(
        r.device
            .s_ctrl(&mut s, CID_BITRATE, 1_000_000_000)
            .unwrap()
            .value,
        200_000_000
    );
    assert_eq!(
        r.device
            .s_ctrl(&mut s, CID_BITRATE, 5_000_000)
            .unwrap()
            .value,
        5_000_000
    );
    assert_eq!(r.device.g_ctrl(&s, CID_BITRATE).unwrap().value, 5_000_000);
    // Menus: outside the range ERANGE, in a hole EINVAL, an item is taken.
    assert_eq!(
        r.device.s_ctrl(&mut s, CID_H264_PROFILE, 5).err(),
        Some(libc::ERANGE)
    );
    assert_eq!(
        r.device.s_ctrl(&mut s, CID_H264_PROFILE, -1).err(),
        Some(libc::ERANGE)
    );
    assert_eq!(
        r.device.s_ctrl(&mut s, CID_H264_PROFILE, 1).err(),
        Some(libc::EINVAL)
    );
    assert_eq!(
        r.device
            .s_ctrl(&mut s, CID_H264_PROFILE, VideoH264Profile::Main as i32)
            .unwrap()
            .value,
        2
    );
    assert_eq!(r.device.g_ctrl(&s, CID_H264_PROFILE).unwrap().value, 2);
    // Booleans normalise; the button takes any value and, with no codec, does nothing.
    assert_eq!(r.device.s_ctrl(&mut s, CID_PREPEND, 7).unwrap().value, 1);
    assert_eq!(r.device.s_ctrl(&mut s, CID_FORCE_KEY, 0).unwrap().value, 0);
    assert_eq!(r.log.lock().unwrap().forced, 0);

    // Ext controls: count 0 succeeds and changes nothing.
    let mut ctrls = ext_controls(0, 0);
    assert!(r
        .device
        .g_ext_ctrls(&s, CtrlWhich::Current, &mut ctrls, &mut vec![], vec![])
        .is_ok());
    assert!(r
        .device
        .try_ext_ctrls(&s, CtrlWhich::Current, &mut ctrls, &mut vec![], vec![])
        .is_ok());
    // An invalid id: EINVAL, error_idx = index for TRY, = count for G and S.
    let mut ctrls = ext_controls(0, 1);
    let mut arr = vec![ext_control(0, 0)];
    assert_eq!(
        r.device
            .g_ext_ctrls(&s, CtrlWhich::Current, &mut ctrls, &mut arr, vec![])
            .err(),
        Some(libc::EINVAL)
    );
    assert_eq!(ctrls.error_idx, 1);
    assert_eq!(
        r.device
            .try_ext_ctrls(&s, CtrlWhich::Current, &mut ctrls, &mut arr, vec![])
            .err(),
        Some(libc::EINVAL)
    );
    assert_eq!(ctrls.error_idx, 0);
    assert_eq!(
        r.device
            .s_ext_ctrls(&mut s, CtrlWhich::Current, &mut ctrls, &mut arr, vec![])
            .err(),
        Some(libc::EINVAL)
    );
    assert_eq!(ctrls.error_idx, 1);
    // Read-only through TRY/S: EACCES with the same error_idx rule; write-only through G.
    let mut arr = vec![ext_control(CID_MIN_OUT, 4)];
    assert_eq!(
        r.device
            .try_ext_ctrls(&s, CtrlWhich::Current, &mut ctrls, &mut arr, vec![])
            .err(),
        Some(libc::EACCES)
    );
    assert_eq!(ctrls.error_idx, 0);
    assert_eq!(
        r.device
            .s_ext_ctrls(&mut s, CtrlWhich::Current, &mut ctrls, &mut arr, vec![])
            .err(),
        Some(libc::EACCES)
    );
    assert_eq!(ctrls.error_idx, 1);
    let mut arr = vec![ext_control(CID_FORCE_KEY, 0)];
    assert_eq!(
        r.device
            .g_ext_ctrls(&s, CtrlWhich::Current, &mut ctrls, &mut arr, vec![])
            .err(),
        Some(libc::EACCES)
    );
    assert_eq!(ctrls.error_idx, 1);
    // A class `which` that does not match: EINVAL.
    let mut arr = vec![ext_control(CID_MIN_OUT, 0)];
    assert_eq!(
        r.device
            .g_ext_ctrls(
                &s,
                CtrlWhich::Class(CODEC_CLASS),
                &mut ctrls,
                &mut arr,
                vec![]
            )
            .err(),
        Some(libc::EINVAL)
    );
    assert!(r
        .device
        .g_ext_ctrls(
            &s,
            CtrlWhich::Class(USER_CLASS),
            &mut ctrls,
            &mut arr,
            vec![]
        )
        .is_ok());
    assert_eq!(ext_value(&arr[0]), 4);
    // WHICH_DEF_VAL: G answers defaults, S and TRY refuse.
    let mut arr = vec![ext_control(CID_BITRATE, 0)];
    assert!(r
        .device
        .g_ext_ctrls(&s, CtrlWhich::Default, &mut ctrls, &mut arr, vec![])
        .is_ok());
    assert_eq!(
        ext_value(&arr[0]),
        2_000_000,
        "the default, not the 5 Mbit/s set above"
    );
    assert_eq!(
        r.device
            .s_ext_ctrls(&mut s, CtrlWhich::Default, &mut ctrls, &mut arr, vec![])
            .err(),
        Some(libc::EINVAL)
    );
    assert_eq!(
        r.device
            .try_ext_ctrls(&s, CtrlWhich::Default, &mut ctrls, &mut arr, vec![])
            .err(),
        Some(libc::EINVAL)
    );

    // A set of several controls is all or nothing: a bad second one leaves the first as is.
    let mut ctrls = ext_controls(CODEC_CLASS, 2);
    let mut arr = vec![ext_control(CID_GOP, 12), ext_control(CID_H264_PROFILE, 3)];
    assert_eq!(
        r.device
            .s_ext_ctrls(
                &mut s,
                CtrlWhich::Class(CODEC_CLASS),
                &mut ctrls,
                &mut arr,
                vec![]
            )
            .err(),
        Some(libc::EINVAL)
    );
    assert_eq!(ctrls.error_idx, 2);
    assert_eq!(
        r.device.g_ctrl(&s, CID_GOP).unwrap().value,
        30,
        "nothing applied"
    );
    let mut arr = vec![
        ext_control(CID_GOP, 12),
        ext_control(CID_H264_PROFILE, VideoH264Profile::Baseline as i32),
    ];
    assert!(r
        .device
        .s_ext_ctrls(
            &mut s,
            CtrlWhich::Class(CODEC_CLASS),
            &mut ctrls,
            &mut arr,
            vec![]
        )
        .is_ok());
    assert_eq!(ctrls.error_idx, 0);
    assert_eq!(r.device.g_ctrl(&s, CID_GOP).unwrap().value, 12);
    assert_eq!(r.device.g_ctrl(&s, CID_H264_PROFILE).unwrap().value, 0);
    // TRY echoes what a value would become without applying it.
    let mut arr = vec![ext_control(CID_BITRATE, 1)];
    assert!(r
        .device
        .try_ext_ctrls(&s, CtrlWhich::Current, &mut ctrls, &mut arr, vec![])
        .is_ok());
    assert_eq!(ext_value(&arr[0]), 32_000);
    assert_eq!(r.device.g_ctrl(&s, CID_BITRATE).unwrap().value, 5_000_000);

    close(&mut r.device, s);
}

/// The `ffmpeg -c:v h264_v4l2m2m` sequence, ioctl by ioctl (libavcodec, FFmpeg n7.1.1):
/// `v4l2_probe_driver` = TRY_FMT(OUTPUT raw) + ENUM_FMT(CAPTURE)/TRY_FMT(CAPTURE)
/// (`v4l2_m2m.c:101-135`, `v4l2_context.c:467-555`, `:667-692`); `v4l2_configure_contexts` =
/// S_FMT(OUTPUT), S_FMT(CAPTURE) with `v4l2_get_framesize_compressed` (`v4l2_m2m.c:163-173`,
/// `v4l2_context.c:112-124`), then `ff_v4l2_context_init` on both = G_FMT + REQBUFS + QUERYBUF
/// (`v4l2_context.c:713-762`, capture too for an encoder: `v4l2_m2m.c:182-188`);
/// `v4l2_prepare_encoder` (`v4l2_m2m_enc.c:174-272`) = SUBSCRIBE_EVENT(EOS) (`:163-172`),
/// S/G_EXT_CTRLS(B_FRAMES) tolerated failing (`:148-161`), S_PARM(OUTPUT) (`:41-51`, `:193`),
/// S_EXT_CTRLS of HEADER_MODE=SEPARATE, BITRATE, FRAME_RC_ENABLE (fails, debug level), GOP_SIZE
/// (`:196-199`), H264_PROFILE (`:214`), H264_MIN/MAX_QP (`:266-269`); per frame
/// `v4l2_send_frame` = FORCE_KEY_FRAME for an I picture (`:279-282`) + QBUF(OUTPUT), then
/// STREAMON(OUTPUT) and STREAMON(CAPTURE) (`:314-328`); EOF = `v4l2_stop_encode` = ENCODER_CMD
/// (STOP) (`v4l2_context.c:250-268`, `:583-589`) and dequeue until `bytesused == 0` or LAST
/// (`:399-410`); close = STREAMOFF both + REQBUFS(0) (`v4l2_m2m.c:272-282`,
/// `v4l2_context.c:444-465`).
#[test]
fn ffmpeg_h264_v4l2m2m_encode_sequence() {
    let mut r = rig();
    let mut s = session(&mut r.device);

    // probe: TRY_FMT(OUTPUT, NV12); ENUM_FMT(CAPTURE) until H264; TRY_FMT(CAPTURE).
    assert!(r
        .device
        .try_fmt(&s, OUTPUT, output_format(NV12, 0, 0))
        .is_ok());
    let mut i = 0;
    while r.device.enum_fmt(&s, CAPTURE, i).unwrap().pixelformat != H264.to_u32() {
        i += 1;
    }
    assert!(r
        .device
        .try_fmt(&s, CAPTURE, capture_format(H264, 0))
        .is_ok());

    // configure: S_FMT(OUTPUT, avctx w x h), S_FMT(CAPTURE, H264, sizeimage from
    // `v4l2_get_framesize_compressed`'s encoder branch: 32-aligned w*h*3/2/2, 4 KiB-aligned
    // (`v4l2_context.c:121-123`) -- 61440 for 320x240, below the device's floor.
    r.device
        .s_fmt(&mut s, OUTPUT, output_format(NV12, SIZE.0, SIZE.1))
        .unwrap();
    let asked =
        (SIZE.1.div_ceil(32) * 32 * SIZE.0.div_ceil(32) * 32 * 3 / 2 / 2).div_ceil(4096) * 4096;
    assert_eq!(asked, 61440);
    let cfmt = pix(&r
        .device
        .s_fmt(&mut s, CAPTURE, capture_format(H264, asked))
        .unwrap());
    assert_eq!(
        cfmt.sizeimage, DEFAULT_BITSTREAM_FLOOR,
        "below the floor: the device's default"
    );
    // context_init(output): G_FMT, REQBUFS(16), QUERYBUF; the same for capture (4).
    r.device.g_fmt(&s, OUTPUT).unwrap();
    assert_eq!(
        r.device
            .reqbufs(&mut s, OUTPUT, MemoryType::Mmap, 16)
            .unwrap()
            .count,
        16
    );
    r.device.querybuf(&s, OUTPUT, 0).unwrap();
    let sizeimage = pix(&r.device.g_fmt(&s, CAPTURE).unwrap()).sizeimage;
    assert_eq!(
        r.device
            .reqbufs(&mut s, CAPTURE, MemoryType::Mmap, 4)
            .unwrap()
            .count,
        4
    );
    r.device.querybuf(&s, CAPTURE, 3).unwrap();

    // prepare_encoder.
    r.device
        .subscribe_event(&mut s, EventType::Eos, SubscribeEventFlags::empty())
        .unwrap();
    assert_eq!(
        s_ext(&mut r, &mut s, CID_B_FRAMES, 0).err(),
        Some(libc::EINVAL)
    );
    assert_eq!(g_ext(&mut r, &s, CID_B_FRAMES).err(), Some(libc::EINVAL));
    r.device.s_parm(&mut s, parm(OUTPUT, 1, 25)).unwrap();
    assert_eq!(
        s_ext(
            &mut r,
            &mut s,
            CID_HEADER_MODE,
            VideoHeaderMode::Separate as i32
        )
        .unwrap(),
        0
    );
    assert_eq!(
        s_ext(&mut r, &mut s, CID_BITRATE, 3_000_000).unwrap(),
        3_000_000
    );
    assert_eq!(
        s_ext(&mut r, &mut s, CID_FRAME_RC, 1).err(),
        Some(libc::EINVAL)
    );
    assert_eq!(s_ext(&mut r, &mut s, CID_GOP, 2).unwrap(), 2);
    assert_eq!(
        s_ext(
            &mut r,
            &mut s,
            CID_H264_PROFILE,
            VideoH264Profile::High as i32
        )
        .unwrap(),
        4
    );
    assert_eq!(s_ext(&mut r, &mut s, CID_H264_MIN_QP, 0).unwrap(), 0);
    assert_eq!(s_ext(&mut r, &mut s, CID_H264_MAX_QP, 51).unwrap(), 51);

    // First frame: FORCE_KEY_FRAME (pict_type I, value 0 -- a button ignores it), QBUF(OUTPUT),
    // STREAMON(OUTPUT), STREAMON(CAPTURE). ffmpeg queues no CAPTURE buffer by hand: they are
    // queued by `ff_v4l2_context_init`'s QBUF loop in real life; here explicitly.
    assert_eq!(s_ext(&mut r, &mut s, CID_FORCE_KEY, 0).unwrap(), 0);
    queue_frame(&mut r, &mut s, 0, 0x10, 1);
    r.device.streamon(&mut s, OUTPUT).unwrap();
    assert!(
        r.log.lock().unwrap().started.is_empty(),
        "no codec until both queues stream"
    );
    r.device.streamon(&mut s, CAPTURE).unwrap();
    let config = r
        .log
        .lock()
        .unwrap()
        .started
        .first()
        .cloned()
        .expect("codec created");
    assert_eq!(config.header_mode, VideoHeaderMode::Separate);
    assert_eq!(config.bitrate, 3_000_000);
    assert_eq!(config.gop_size, 2);
    assert_eq!(config.profile, Some(VideoH264Profile::High as i32));
    assert_eq!(
        config.level,
        Some(VideoH264Level::L4_1 as i32),
        "the backend's default level"
    );
    assert_eq!(config.qp, Some((0, 51)));
    assert_eq!(config.frame_rate, FrameRate { num: 25, den: 1 });
    assert_eq!(config.bitrate_mode, VideoBitrateMode::VariableBitrate);
    assert_eq!(config.visible_rect, v4l2r::Rect::new(0, 0, SIZE.0, SIZE.1));
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
    // Three more frames; the fourth is an I picture again (FORCE_KEY_FRAME).
    queue_frame(&mut r, &mut s, 1, 0x11, 2);
    queue_frame(&mut r, &mut s, 2, 0x12, 3);
    assert_eq!(s_ext(&mut r, &mut s, CID_FORCE_KEY, 0).unwrap(), 0);
    queue_frame(&mut r, &mut s, 3, 0x13, 4);

    // Headers first (SEPARATE), then the frames; GOP 2 makes frames 1 and 3 keyframes, the
    // forced one is frame 4 (index 3), which waits: four CAPTURE buffers hold four packets.
    // Timestamps are copied from the OUTPUT buffers.
    collect_capture(&mut r, &mut s, 4);
    let packets = dequeued_on(&r.events.borrow(), CAPTURE);
    assert_eq!(packets.len(), 4);
    let headers = &packets[0];
    assert_eq!(*headers.get_first_plane().bytesused, HEADERS_LEN as u32);
    assert!(!headers
        .flags()
        .intersects(BufferFlags::KEYFRAME | BufferFlags::PFRAME | BufferFlags::BFRAME));
    assert!(headers.flags().contains(BufferFlags::TIMESTAMP_COPY));
    assert_eq!(
        headers.timestamp().tv_sec,
        1,
        "the headers carry the first frame's timestamp"
    );
    assert_eq!(
        parse_units(&capture_bytes(&s, headers.index() as usize, HEADERS_LEN)),
        vec![Unit::Sps, Unit::Pps]
    );
    let expect = [
        (0x10u8, true, 1i64),
        (0x11, false, 2),
        (0x12, true, 3),
        (0x13, true, 4),
    ];
    let check = |s: &Session, p: &V4l2Buffer, n: usize| {
        let (luma, key, t) = expect[n];
        assert_eq!(p.timestamp().tv_sec, t);
        assert_eq!(p.flags().contains(BufferFlags::KEYFRAME), key, "frame {n}");
        assert_eq!(p.flags().contains(BufferFlags::PFRAME), !key, "frame {n}");
        assert_eq!(p.sequence(), n as u32 + 1);
        let units = parse_units(&capture_bytes(
            s,
            p.index() as usize,
            *p.get_first_plane().bytesused as usize,
        ));
        assert_eq!(
            units,
            vec![Unit::Frame {
                key,
                frame_no: n as u8,
                digest: digest_of(luma, SIZE)
            }]
        );
    };
    for n in 0..3 {
        check(&s, &packets[n + 1], n);
    }
    assert_eq!(
        r.log.lock().unwrap().forced,
        1,
        "the second FORCE_KEY_FRAME reached the codec"
    );
    // Three OUTPUT buffers came back; the fourth frame is still with the codec.
    collect_output(&mut r, &mut s, 3);
    assert!(s.input.buffers[3].lent);

    // EOF: ENCODER_CMD(STOP), then the CAPTURE buffer ffmpeg re-queues after dequeuing the
    // headers takes the fourth frame, which is the last one and says so; then EOS.
    let free = packets[0].index();
    r.device
        .encoder_cmd(&mut s, enc_cmd(bindings::V4L2_ENC_CMD_STOP))
        .unwrap();
    r.device
        .qbuf(
            &mut s,
            mmap_buffer(CAPTURE, free, sizeimage),
            vec![],
            PayloadValidity::ALL,
        )
        .unwrap();
    collect_capture(&mut r, &mut s, 5);
    let last = dequeued_on(&r.events.borrow(), CAPTURE).pop().unwrap();
    assert!(last.flags().contains(BufferFlags::LAST));
    check(&s, &last, 3);
    assert_eq!(eos_events(&r.events.borrow()), 1);
    collect_output(&mut r, &mut s, 4);
    assert!(s.input.buffers.iter().all(|b| !b.queued));

    // close: STREAMOFF both, REQBUFS(OUTPUT, 0).
    r.device.streamoff(&mut s, OUTPUT).unwrap();
    r.device.streamoff(&mut s, CAPTURE).unwrap();
    r.device
        .reqbufs(&mut s, OUTPUT, MemoryType::Mmap, 0)
        .unwrap();
    close(&mut r.device, s);
    assert_eq!(
        *r.log.lock().unwrap().released_while_capture_active.borrow(),
        0
    );
    assert!(!r.log.lock().unwrap().open);
}

/// The GStreamer `v4l2h264enc` sequence (gst-plugins-good, GStreamer 1.28.2): at plugin
/// registration QUERYCTRL(H264_PROFILE)+QUERYMENU and QUERYCTRL(H264_LEVEL)+QUERYMENU(max)
/// (`gstv4l2codec.c:38-126`, `gstv4l2videoenc.c:1228-1238`); open = probe_caps on both queues
/// = ENUM_FMT + ENUM_FRAMESIZES + ENUM_FRAMEINTERVALS at the minimum size
/// (`gstv4l2videoenc.c:124-141`, `gstv4l2object.c:3088-3200`, `:2738-2760`); negotiate =
/// S_CTRL(profile) / G_CTRL(level) (`gstv4l2videoenc.c:436-443`, `:588-597`);
/// decide_allocation = S_FMT(CAPTURE) with `calculate_max_sizeimage` then G_PARM(CAPTURE)
/// tolerated failing (`gstv4l2videoenc.c:922-933`, `gstv4l2object.c:4183-4185`, `:4526-4527`,
/// `:4783-4792`), G_CTRL(MIN_BUFFERS_FOR_CAPTURE) tolerated failing (`:940-957`); set_format =
/// S_FMT(OUTPUT) + G_PARM/S_PARM(OUTPUT) (`gstv4l2videoenc.c:363`, `gstv4l2object.c:4604-4636`)
/// and propose_allocation = G_CTRL(MIN_BUFFERS_FOR_OUTPUT); handle_frame = REQBUFS(OUTPUT),
/// REQBUFS(CAPTURE) + QBUF all + STREAMON(CAPTURE) **first** (`gstv4l2videoenc.c:774-822`,
/// `gstv4l2bufferpool.c:963-970`), S_CTRL(FORCE_KEY_FRAME=1) for a forced frame (`:836-844`),
/// QBUF(OUTPUT) with `timestamp = (frame number, 0)` (`gstv4l2bufferpool.c:1208-1210`) and
/// STREAMON(OUTPUT); the loop matches a CAPTURE buffer to its frame by that timestamp
/// (`gstv4l2videoenc.c:678-685`) and takes KEYFRAME as the sync point (`:712-715`,
/// `gstv4l2bufferpool.c:1446-1454`); finish = ENCODER_CMD(STOP) until LAST
/// (`gstv4l2videoenc.c:285`, `gstv4l2bufferpool.c:1335-1341`); stop = STREAMOFF + REQBUFS(0).
#[test]
fn gstreamer_v4l2h264enc_sequence() {
    let mut r = rig();
    let mut s = session(&mut r.device);

    // registration: profile and level menus.
    let q = query(&mut r, &s, CID_H264_PROFILE).unwrap();
    assert_eq!(q.type_, bindings::v4l2_ctrl_type_V4L2_CTRL_TYPE_MENU);
    let profiles: Vec<u32> = (q.minimum as u32..=q.maximum as u32)
        .filter(|i| r.device.querymenu(&s, CID_H264_PROFILE, *i).is_ok())
        .collect();
    assert_eq!(profiles, vec![0, 2, 4]);
    let q = query(&mut r, &s, CID_H264_LEVEL).unwrap();
    assert!(r
        .device
        .querymenu(&s, CID_H264_LEVEL, q.maximum as u32)
        .is_ok());

    // open: probe both queues.
    assert!(r.device.enum_fmt(&s, OUTPUT, 0).is_ok());
    let fs = r.device.enum_framesizes(&s, 0, NV12.to_u32()).unwrap();
    // SAFETY: stepwise.
    let sw = unsafe { fs.__bindgen_anon_1.stepwise };
    assert!(r
        .device
        .enum_frameintervals(&s, 0, NV12.to_u32(), sw.min_width, sw.min_height)
        .is_ok());
    assert!(r.device.enum_fmt(&s, CAPTURE, 0).is_ok());
    assert!(r.device.enum_framesizes(&s, 0, H264.to_u32()).is_ok());
    assert!(r
        .device
        .enum_frameintervals(&s, 0, H264.to_u32(), 16, 16)
        .is_ok());

    // negotiate: the downstream caps ask for "main"; the level is read back.
    assert_eq!(
        r.device
            .s_ctrl(&mut s, CID_H264_PROFILE, VideoH264Profile::Main as i32)
            .unwrap()
            .value,
        2
    );
    assert_eq!(
        r.device.g_ctrl(&s, CID_H264_LEVEL).unwrap().value,
        VideoH264Level::L4_1 as i32
    );

    // decide_allocation: S_FMT(CAPTURE) sized from the maximum frame size (`gstv4l2object.c`
    // `calculate_max_sizeimage`: half the *probed maximum* frame, 8 MiB here, 32 MiB against the
    // phone's 8192x8192 codecs) -- which the device refits to what the raw size can need, 4 MiB
    // at this size, so a pool of those does not take the shared `media_host` pool (review-m7
    // R7-4); G_PARM(CAPTURE) and MIN_BUFFERS_FOR_CAPTURE fail harmlessly.
    let max_sizeimage = 4096 * 4096 * 8 / 8 / 2;
    let cfmt = pix(&r
        .device
        .s_fmt(&mut s, CAPTURE, capture_format(H264, max_sizeimage))
        .unwrap());
    assert_eq!(
        cfmt.sizeimage,
        4 << 20,
        "the client's size is refitted to the raw size, and the answer is what it gets"
    );
    assert_eq!(r.device.g_parm(&s, CAPTURE).err(), Some(libc::ENOTTY));
    assert_eq!(r.device.g_ctrl(&s, CID_MIN_CAP).err(), Some(libc::EINVAL));
    // set_format: S_FMT(OUTPUT) then G_PARM + S_PARM(OUTPUT, 30/1) on the raw side.
    r.device
        .s_fmt(&mut s, OUTPUT, output_format(NV12, SIZE.0, SIZE.1))
        .unwrap();
    let p = r.device.g_parm(&s, OUTPUT).unwrap();
    assert_ne!(timeperframe(&p).2 & bindings::V4L2_CAP_TIMEPERFRAME, 0);
    r.device.s_parm(&mut s, parm(OUTPUT, 1, 30)).unwrap();
    // propose_allocation: the OUTPUT pool size.
    assert_eq!(r.device.g_ctrl(&s, CID_MIN_OUT).unwrap().value, 4);

    // handle_frame: pools. CAPTURE streams before any raw frame exists.
    r.device
        .reqbufs(&mut s, OUTPUT, MemoryType::Mmap, 4)
        .unwrap();
    r.device
        .reqbufs(&mut s, CAPTURE, MemoryType::Mmap, 4)
        .unwrap();
    for i in 0..4 {
        r.device
            .qbuf(
                &mut s,
                mmap_buffer(CAPTURE, i, max_sizeimage),
                vec![],
                PayloadValidity::ALL,
            )
            .unwrap();
    }
    r.device.streamon(&mut s, CAPTURE).unwrap();
    assert!(
        r.log.lock().unwrap().started.is_empty(),
        "no codec until both queues stream"
    );
    // The first frame is a forced keyframe in gst's eyes; frame numbers ride in tv_sec.
    assert!(r.device.s_ctrl(&mut s, CID_FORCE_KEY, 1).is_ok());
    queue_frame(&mut r, &mut s, 0, 0x20, 0);
    r.device.streamon(&mut s, OUTPUT).unwrap();
    let config = r
        .log
        .lock()
        .unwrap()
        .started
        .first()
        .cloned()
        .expect("codec created");
    assert_eq!(config.profile, Some(VideoH264Profile::Main as i32));
    assert_eq!(
        config.header_mode,
        VideoHeaderMode::JoinedWith1stFrame,
        "the default gst relies on"
    );
    queue_frame(&mut r, &mut s, 1, 0x21, 1);
    queue_frame(&mut r, &mut s, 2, 0x22, 2);

    // The loop: one CAPTURE buffer per frame, timestamp exactly (N, 0), headers joined with
    // the first keyframe, KEYFRAME on it and PFRAME on the rest.
    collect_capture(&mut r, &mut s, 3);
    let packets = dequeued_on(&r.events.borrow(), CAPTURE);
    for (n, p) in packets.iter().enumerate() {
        assert_eq!((p.timestamp().tv_sec, p.timestamp().tv_usec), (n as i64, 0));
        assert_eq!(p.flags().contains(BufferFlags::KEYFRAME), n == 0);
        let units = parse_units(&capture_bytes(
            &s,
            p.index() as usize,
            *p.get_first_plane().bytesused as usize,
        ));
        let frame = Unit::Frame {
            key: n == 0,
            frame_no: n as u8,
            digest: digest_of(0x20 + n as u8, SIZE),
        };
        if n == 0 {
            assert_eq!(
                units,
                vec![Unit::Sps, Unit::Pps, frame],
                "headers joined with the first frame"
            );
        } else {
            assert_eq!(units, vec![frame]);
        }
    }

    // finish: ENCODER_CMD(STOP) -> the one CAPTURE buffer still queued comes back LAST, empty.
    r.device
        .encoder_cmd(&mut s, enc_cmd(bindings::V4L2_ENC_CMD_STOP))
        .unwrap();
    collect_capture(&mut r, &mut s, 4);
    let last = dequeued_on(&r.events.borrow(), CAPTURE).pop().unwrap();
    assert!(last.flags().contains(BufferFlags::LAST));
    assert_eq!(*last.get_first_plane().bytesused, 0);

    // stop: STREAMOFF both, REQBUFS(0) both.
    r.device.streamoff(&mut s, OUTPUT).unwrap();
    r.device.streamoff(&mut s, CAPTURE).unwrap();
    r.device
        .reqbufs(&mut s, OUTPUT, MemoryType::Mmap, 0)
        .unwrap();
    r.device
        .reqbufs(&mut s, CAPTURE, MemoryType::Mmap, 0)
        .unwrap();
    assert_eq!(
        r.log.lock().unwrap().stops,
        1,
        "STREAMOFF(CAPTURE) reset the codec"
    );
    close(&mut r.device, s);
}

/// `PREPEND_SPSPPS_TO_IDR` puts the headers in front of every keyframe; the GOP size decides
/// where the keyframes fall.
#[test]
fn prepend_sps_pps_to_idr_repeats_the_headers_on_every_keyframe() {
    let mut r = rig();
    let mut s = session(&mut r.device);
    r.device.s_ctrl(&mut s, CID_PREPEND, 1).unwrap();
    r.device.s_ctrl(&mut s, CID_GOP, 2).unwrap();
    start_streaming(&mut r, &mut s);
    assert!(r.log.lock().unwrap().started[0].prepend_sps_pps_to_idr);

    for i in 0..4u32 {
        queue_frame(&mut r, &mut s, i, 0x30 + i as u8, i as i64);
    }
    collect_capture(&mut r, &mut s, 4);
    let packets = dequeued_on(&r.events.borrow(), CAPTURE);
    for (n, p) in packets.iter().enumerate() {
        let units = parse_units(&capture_bytes(
            &s,
            p.index() as usize,
            *p.get_first_plane().bytesused as usize,
        ));
        let key = n % 2 == 0;
        assert_eq!(p.flags().contains(BufferFlags::KEYFRAME), key);
        assert_eq!(
            units.len(),
            if key { 3 } else { 1 },
            "frame {n}: headers on keyframes only"
        );
    }
    close(&mut r.device, s);
}

/// Controls while the codec runs: BITRATE and FORCE_KEY_FRAME reach the session, anything else
/// is EBUSY (kernel encoder interface, "Encoding Parameter Changes"); a S_PARM does not disturb
/// the running codec.
#[test]
fn runtime_bitrate_and_force_keyframe_reach_the_session() {
    let mut r = rig();
    let mut s = session(&mut r.device);
    start_streaming(&mut r, &mut s);

    assert_eq!(
        r.device
            .s_ctrl(&mut s, CID_BITRATE, 8_000_000)
            .unwrap()
            .value,
        8_000_000
    );
    assert_eq!(
        s_ext(&mut r, &mut s, CID_BITRATE, 9_000_000).unwrap(),
        9_000_000
    );
    assert_eq!(r.log.lock().unwrap().bitrates, vec![8_000_000, 9_000_000]);
    // Setting the same value again is not a change.
    assert_eq!(
        r.device
            .s_ctrl(&mut s, CID_BITRATE, 9_000_000)
            .unwrap()
            .value,
        9_000_000
    );
    assert_eq!(r.log.lock().unwrap().bitrates.len(), 2);
    assert_eq!(r.device.s_ctrl(&mut s, CID_GOP, 5).err(), Some(libc::EBUSY));
    assert_eq!(
        r.device
            .s_ctrl(&mut s, CID_H264_PROFILE, VideoH264Profile::Baseline as i32)
            .err(),
        Some(libc::EBUSY)
    );
    assert_eq!(
        r.device.s_ctrl(&mut s, CID_GOP, 30).unwrap().value,
        30,
        "the current value is fine"
    );
    // TRY never minds the codec.
    let mut ctrls = ext_controls(CODEC_CLASS, 1);
    let mut arr = vec![ext_control(CID_GOP, 5)];
    assert!(r
        .device
        .try_ext_ctrls(
            &s,
            CtrlWhich::Class(CODEC_CLASS),
            &mut ctrls,
            &mut arr,
            vec![]
        )
        .is_ok());
    assert!(r.device.s_parm(&mut s, parm(OUTPUT, 1, 15)).is_ok());

    // Two frames, the second forced: both keyframes (GOP 30 would make only the first one).
    queue_frame(&mut r, &mut s, 0, 0x40, 0);
    r.device.s_ctrl(&mut s, CID_FORCE_KEY, 1).unwrap();
    queue_frame(&mut r, &mut s, 1, 0x41, 1);
    queue_frame(&mut r, &mut s, 2, 0x42, 2);
    collect_capture(&mut r, &mut s, 3);
    let keys: Vec<bool> = dequeued_on(&r.events.borrow(), CAPTURE)
        .iter()
        .map(|p| p.flags().contains(BufferFlags::KEYFRAME))
        .collect();
    assert_eq!(keys, vec![true, true, false]);
    assert_eq!(r.log.lock().unwrap().forced, 1);
    // A crop cannot change under a running codec either.
    let rect = bindings::v4l2_rect {
        left: 0,
        top: 0,
        width: 16,
        height: 16,
    };
    assert_eq!(
        r.device
            .s_selection(
                &mut s,
                SelectionType::Output,
                SelectionTarget::Crop,
                rect,
                SelectionFlags::empty()
            )
            .err(),
        Some(libc::EBUSY)
    );
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
    r.device
        .s_fmt(&mut s, CAPTURE, capture_format(H264, 0))
        .unwrap();
    r.device
        .s_fmt(&mut s, OUTPUT, output_format(NV12, SIZE.0, SIZE.1))
        .unwrap();
    r.device
        .reqbufs(&mut s, OUTPUT, MemoryType::Mmap, 2)
        .unwrap();

    // OUTPUT (the raw frame) with a sane plane 0 under the dirty tail: queued, and the length
    // the guest declared is kept.
    let bytes =
        ffmpeg_wire::ffmpeg_qbuf_bytes(OUTPUT, MemoryType::Mmap, 0, (RAW_SIZEIMAGE, RAW_SIZEIMAGE));
    assert_eq!(
        ffmpeg_wire::dispatch_qbuf(&mut r.device, &mut s, &bytes),
        0,
        "ffmpeg's plane array was refused on the output queue (D21)"
    );
    assert_eq!(
        *s.input.buffers[0].v4l2_buffer.get_first_plane().bytesused,
        RAW_SIZEIMAGE
    );

    // The same array with the garbage in plane 0: still `EINVAL`.
    let bytes = ffmpeg_wire::ffmpeg_qbuf_bytes(
        OUTPUT,
        MemoryType::Mmap,
        1,
        (RAW_SIZEIMAGE + 1, RAW_SIZEIMAGE),
    );
    assert_eq!(
        ffmpeg_wire::dispatch_qbuf(&mut r.device, &mut s, &bytes),
        libc::EINVAL,
        "a payload the buffer cannot hold was accepted"
    );
    assert!(!s.input.buffers[1].queued);

    // CAPTURE, `MMAP`: the bitstream is this device's to write, so the guest's payload is
    // ignored.
    let bitstream = pix(&r.device.g_fmt(&s, CAPTURE).unwrap()).sizeimage;
    r.device
        .reqbufs(&mut s, CAPTURE, MemoryType::Mmap, 2)
        .unwrap();
    let bytes = ffmpeg_wire::ffmpeg_qbuf_bytes(CAPTURE, MemoryType::Mmap, 0, (0, bitstream));
    assert_eq!(ffmpeg_wire::dispatch_qbuf(&mut r.device, &mut s, &bytes), 0);
    assert!(s.output.buffers[0].queued);

    close(&mut r.device, s);
}

/// Guest-owned buffers (`driver_owned_queues=output`, the DroidVM default, and `all`): the raw
/// frame is a read-only guest mapping held until `InputBufferDone`, the bitstream lands in a
/// writable guest mapping dropped before the DQBUF event; a buffer too short for the queue's
/// format is refused on either queue, sized from *this* call's length however `PREPARE_BUF`
/// went (review-m4 R2), and a scatter list shorter than the length claims is the backstop.
#[test]
fn userptr_raw_input_and_bitstream_output_go_through_guest_mappings() {
    let mut r = rig();
    let mut s = session(&mut r.device);
    r.device
        .s_fmt(&mut s, CAPTURE, capture_format(H264, 0))
        .unwrap();
    r.device
        .s_fmt(&mut s, OUTPUT, output_format(NV12, SIZE.0, SIZE.1))
        .unwrap();
    let bitstream = pix(&r.device.g_fmt(&s, CAPTURE).unwrap()).sizeimage;

    let reply = r
        .device
        .reqbufs(&mut s, OUTPUT, MemoryType::UserPtr, 2)
        .unwrap();
    assert!(reply.capabilities & BufferCapabilities::SUPPORTS_USERPTR.bits() != 0);
    r.device
        .reqbufs(&mut s, CAPTURE, MemoryType::UserPtr, 2)
        .unwrap();

    // Too short on either queue: refused, nothing mapped.
    let (short, sgs) = userptr_buffer(OUTPUT, 0, 0x1000, RAW_SIZEIMAGE - 1, RAW_SIZEIMAGE - 1);
    assert_eq!(
        r.device
            .qbuf(&mut s, short, sgs, PayloadValidity::ALL)
            .err(),
        Some(libc::EINVAL)
    );
    let (short, sgs) = userptr_buffer(CAPTURE, 0, 0x100000, bitstream - 1, 0);
    assert_eq!(
        r.device
            .qbuf(&mut s, short, sgs, PayloadValidity::ALL)
            .err(),
        Some(libc::EINVAL)
    );
    // A scatter list shorter than the declared length: refused too.
    let (mut lying, _) = userptr_buffer(OUTPUT, 0, 0x1000, RAW_SIZEIMAGE, RAW_SIZEIMAGE);
    *lying.get_first_plane_mut().length = RAW_SIZEIMAGE;
    let sgs = vec![vec![SgEntry::new(0x1000, 4096)]];
    assert_eq!(
        r.device
            .qbuf(&mut s, lying, sgs, PayloadValidity::ALL)
            .err(),
        Some(libc::EINVAL)
    );
    // Prepared at full length, then queued with an 8-byte list: refused, still prepared.
    let (pb, _) = userptr_buffer(OUTPUT, 1, 0x1000, RAW_SIZEIMAGE, RAW_SIZEIMAGE);
    assert!(r
        .device
        .prepare_buf(&mut s, pb, vec![], PayloadValidity::ALL)
        .unwrap()
        .flags()
        .contains(BufferFlags::PREPARED));
    let (qb, sgs) = userptr_buffer(OUTPUT, 1, 0x1000, 8, 8);
    assert_eq!(
        r.device.qbuf(&mut s, qb, sgs, PayloadValidity::ALL).err(),
        Some(libc::EINVAL)
    );
    assert!(s.input.buffers[1].prepared.is_some());
    assert_eq!(*r.guest.live_mappings.borrow(), 0);

    // A raw frame in guest memory: luma 0x55.
    let raw_gpa = 0x10000u64;
    {
        let mut mem = r.guest.memory.borrow_mut();
        let luma = (SIZE.0 * SIZE.1) as usize;
        mem[raw_gpa as usize..raw_gpa as usize + luma].fill(0x55);
    }
    let (ob, sgs) = userptr_buffer(OUTPUT, 0, raw_gpa, RAW_SIZEIMAGE, RAW_SIZEIMAGE);
    let mut ob = ob;
    ob.set_timestamp(ts(9));
    r.device
        .qbuf(&mut s, ob, sgs, PayloadValidity::ALL)
        .unwrap();
    assert_eq!(
        *r.guest.live_mappings.borrow(),
        1,
        "OUTPUT mapping held while queued"
    );
    let cap_gpa = 0x200000u64;
    let (cb, sgs) = userptr_buffer(CAPTURE, 0, cap_gpa, bitstream, 0);
    r.device
        .qbuf(&mut s, cb, sgs, PayloadValidity::ALL)
        .unwrap();
    assert_eq!(
        *r.guest.live_mappings.borrow(),
        2,
        "CAPTURE mapping held while queued"
    );
    r.device.streamon(&mut s, OUTPUT).unwrap();
    r.device.streamon(&mut s, CAPTURE).unwrap();

    // The packet lands in the guest pages; both mappings are dropped before the DQBUF events.
    collect_capture(&mut r, &mut s, 1);
    collect_output(&mut r, &mut s, 1);
    assert_eq!(
        *r.guest.live_mappings.borrow(),
        0,
        "mappings released before the DQBUFs"
    );
    let packet = dequeued_on(&r.events.borrow(), CAPTURE).remove(0);
    assert_eq!(packet.timestamp().tv_sec, 9);
    let len = *packet.get_first_plane().bytesused as usize;
    let bytes = r.guest.memory.borrow()[cap_gpa as usize..cap_gpa as usize + len].to_vec();
    assert_eq!(
        parse_units(&bytes),
        vec![
            Unit::Sps,
            Unit::Pps,
            Unit::Frame {
                key: true,
                frame_no: 0,
                digest: digest_of(0x55, SIZE)
            }
        ]
    );
    close(&mut r.device, s);
}

/// The kernel's "Stopped" and "Reset": `STREAMOFF(OUTPUT)` flushes and keeps the codec, the
/// OUTPUT buffers come back and `STREAMON(OUTPUT)` resumes without a new codec;
/// `STREAMOFF(CAPTURE)` tears the codec down (the CAPTURE buffers come back) and `STREAMON
/// (CAPTURE)` creates a new one that starts with headers again and encodes the raw frame the
/// old one had not got to.
#[test]
fn streamoff_output_pauses_and_streamoff_capture_resets_the_codec() {
    let mut r = rig();
    let mut s = session(&mut r.device);
    let sizeimage = start_streaming(&mut r, &mut s);
    // Four frames use up the four CAPTURE buffers; a fifth stays with the codec, unwritten.
    for i in 0..4u32 {
        queue_frame(&mut r, &mut s, i, 0x60 + i as u8, i as i64);
    }
    collect_capture(&mut r, &mut s, 4);
    collect_output(&mut r, &mut s, 4);
    queue_frame(&mut r, &mut s, 0, 0x64, 4);
    assert!(s.input.buffers[0].lent);

    // Stopped: STREAMOFF(OUTPUT) flushes the pending frame and keeps the codec.
    r.device.streamoff(&mut s, OUTPUT).unwrap();
    assert_eq!(r.log.lock().unwrap().flushes, 1);
    assert_eq!(r.log.lock().unwrap().stops, 0, "the codec is kept");
    assert!(
        s.input.buffers.iter().all(|b| !b.queued),
        "OUTPUT buffers returned by the pause"
    );
    assert!(s.state.capture_streaming);
    r.device.streamon(&mut s, OUTPUT).unwrap();
    assert_eq!(r.log.lock().unwrap().started.len(), 1, "no second codec");
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
    queue_frame(&mut r, &mut s, 1, 0x65, 5);
    collect_capture(&mut r, &mut s, 5);
    let packets = dequeued_on(&r.events.borrow(), CAPTURE);
    assert_eq!(packets.len(), 5, "the flushed frame never came out");
    let p = &packets[4];
    assert_eq!(p.timestamp().tv_sec, 5);
    assert!(
        !p.flags().contains(BufferFlags::KEYFRAME),
        "the same stream continues"
    );
    let units = parse_units(&capture_bytes(
        &s,
        p.index() as usize,
        *p.get_first_plane().bytesused as usize,
    ));
    assert_eq!(
        units,
        vec![Unit::Frame {
            key: false,
            frame_no: 5,
            digest: digest_of(0x65, SIZE)
        }]
    );

    // Reset: again a frame the codec cannot write out, then STREAMOFF(CAPTURE).
    for i in 0..3u32 {
        queue_frame(&mut r, &mut s, 2 + i % 2, 0x66 + i as u8, 6 + i as i64);
        collect_capture(&mut r, &mut s, 6 + i as usize);
        collect_output(&mut r, &mut s, 6 + i as usize);
    }
    queue_frame(&mut r, &mut s, 0, 0x69, 9);
    assert!(
        s.input.buffers[0].lent,
        "frame 9 is with the codec, waiting for a CAPTURE buffer"
    );
    r.device.streamoff(&mut s, CAPTURE).unwrap();
    assert_eq!(
        r.log.lock().unwrap().stops,
        1,
        "STREAMOFF(CAPTURE) joins the codec"
    );
    assert!(
        s.output.buffers.iter().all(|b| !b.queued),
        "CAPTURE buffers returned by the reset"
    );
    assert!(
        s.input.buffers[0].queued && !s.input.buffers[0].lent,
        "frame 9 is queued again"
    );
    assert_eq!(
        *r.log.lock().unwrap().released_while_capture_active.borrow(),
        0
    );

    // A fresh codec: headers again, frame 9 first, as a keyframe of the new stream.
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
    assert_eq!(
        r.log.lock().unwrap().started.len(),
        2,
        "a new codec for the new stream"
    );
    collect_capture(&mut r, &mut s, 9);
    let p = dequeued_on(&r.events.borrow(), CAPTURE).pop().unwrap();
    assert_eq!(p.timestamp().tv_sec, 9);
    assert!(p.flags().contains(BufferFlags::KEYFRAME));
    let units = parse_units(&capture_bytes(
        &s,
        p.index() as usize,
        *p.get_first_plane().bytesused as usize,
    ));
    assert_eq!(
        units,
        vec![
            Unit::Sps,
            Unit::Pps,
            Unit::Frame {
                key: true,
                frame_no: 0,
                digest: digest_of(0x69, SIZE)
            }
        ]
    );

    close(&mut r.device, s);
    assert_eq!(r.log.lock().unwrap().stops, 2);
}

/// §2.5: nothing is written to a buffer after it is unqueued or freed; `REQBUFS(0)` and close
/// join the backend before a buffer goes back to the allocator; `REQBUFS(n)` on a streaming
/// queue is refused rather than freeing lent buffers; a session that is only dropped joins
/// its thread before its buffers go (review-m4 R1).
#[test]
fn lifecycle_invariants_join_the_backend_before_freeing() {
    let mut r = rig();
    let mut s = session(&mut r.device);
    let sizeimage = start_streaming(&mut r, &mut s);
    for i in 0..2u32 {
        queue_frame(&mut r, &mut s, i, 0x70 + i as u8, i as i64);
    }
    collect_capture(&mut r, &mut s, 2);

    assert_eq!(
        r.device.reqbufs(&mut s, CAPTURE, MemoryType::Mmap, 4).err(),
        Some(libc::EBUSY)
    );
    assert_eq!(
        r.device.reqbufs(&mut s, OUTPUT, MemoryType::Mmap, 4).err(),
        Some(libc::EBUSY)
    );
    // REQBUFS(0) on CAPTURE joins (stop) before freeing; nothing released meanwhile.
    r.device
        .reqbufs(&mut s, CAPTURE, MemoryType::Mmap, 0)
        .unwrap();
    assert_eq!(
        r.log.lock().unwrap().stops,
        1,
        "REQBUFS(0) stopped the codec first"
    );
    assert!(s.output.buffers.is_empty());
    assert_eq!(
        *r.log.lock().unwrap().released_while_capture_active.borrow(),
        0
    );
    assert!(!s.state.capture_streaming);
    assert!(s.state.output_streaming, "OUTPUT keeps streaming");

    // Rebuild capture and restart; then drop the session bare: the fake's Drop joins.
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
    r.device.streamon(&mut s, CAPTURE).unwrap();
    assert_eq!(r.log.lock().unwrap().started.len(), 2);
    queue_frame(&mut r, &mut s, 2, 0x72, 2);
    assert!(r.log.lock().unwrap().open);
    drop(s);
    assert!(
        !r.log.lock().unwrap().open,
        "the dropped session joined its codec"
    );
    assert_eq!(r.log.lock().unwrap().stops, 2);
    assert_eq!(
        *r.log.lock().unwrap().released_while_capture_active.borrow(),
        0
    );
}

/// One encode at a time: a second session's `REQBUFS`/`CREATE_BUFS` is refused with `EBUSY` while
/// the first holds buffers, and succeeds once it lets go.
#[test]
fn a_second_session_is_refused_while_one_holds_buffers() {
    let mut r = rig();
    let mut s = session(&mut r.device);
    r.device
        .reqbufs(&mut s, OUTPUT, MemoryType::Mmap, 2)
        .unwrap();

    let mut other = new_session(&mut r.device, 1);
    assert_eq!(
        r.device
            .reqbufs(&mut other, OUTPUT, MemoryType::Mmap, 1)
            .err(),
        Some(libc::EBUSY)
    );
    let big = capture_format(H264, 1 << 20);
    assert_eq!(
        r.device
            .create_bufs(&mut other, 1, CAPTURE, MemoryType::Mmap, big)
            .err(),
        Some(libc::EBUSY)
    );
    // The second session's own controls are its own, untouched by the first's.
    r.device.s_ctrl(&mut s, CID_GOP, 7).unwrap();
    assert_eq!(r.device.g_ctrl(&other, CID_GOP).unwrap().value, 30);

    r.device
        .reqbufs(&mut s, OUTPUT, MemoryType::Mmap, 0)
        .unwrap();
    close(&mut r.device, s);
    r.device
        .reqbufs(&mut other, OUTPUT, MemoryType::Mmap, 1)
        .unwrap();
    close(&mut r.device, other);
    assert_eq!(r.device.active_session, None);
}

/// A backend whose codec will not start (reclaimed, out of memory) fails the `STREAMON` that
/// would have created it with that errno and leaves the queue as it was, buffers queued for a
/// retry.
#[test]
fn a_codec_that_will_not_start_fails_streamon() {
    let mut r = rig_with(Some(libc::EBUSY));
    let mut s = session(&mut r.device);
    r.device
        .s_fmt(&mut s, OUTPUT, output_format(NV12, SIZE.0, SIZE.1))
        .unwrap();
    r.device
        .reqbufs(&mut s, OUTPUT, MemoryType::Mmap, 2)
        .unwrap();
    r.device
        .reqbufs(&mut s, CAPTURE, MemoryType::Mmap, 2)
        .unwrap();
    queue_frame(&mut r, &mut s, 0, 0x01, 0);
    r.device.streamon(&mut s, OUTPUT).unwrap();
    assert_eq!(r.device.streamon(&mut s, CAPTURE).err(), Some(libc::EBUSY));
    assert!(s.state.output_streaming);
    assert!(!s.state.capture_streaming);
    assert!(!s.codec_started);
    assert!(
        s.input.buffers[0].queued,
        "the frame stays queued for a retry"
    );
    close(&mut r.device, s);
}

/// The `v4l2-compliance` encoder-command probe (`v4l2-test-codecs.cpp:testEncoder:28-79`): an
/// unknown command is EINVAL for both ioctls, STOP and START come back with their flags cleared,
/// PAUSE/RESUME are EINVAL, STOP with nothing streaming is a no-op success; and the drain corner
/// cases of `v4l2-test-buffers.cpp:1742-1775`: STOP with no raw frame queued makes the next
/// CAPTURE buffer come back empty and LAST, twice over; a command during a drain is EBUSY; START
/// after the LAST buffer resumes.
#[test]
fn encoder_cmd_matches_the_compliance_probe() {
    let mut r = rig();
    let mut s = session(&mut r.device);

    let mut bad = enc_cmd(0xffff_ffff);
    bad.flags = !0;
    assert_eq!(r.device.encoder_cmd(&mut s, bad).err(), Some(libc::EINVAL));
    assert_eq!(r.device.try_encoder_cmd(&s, bad).err(), Some(libc::EINVAL));
    let mut stop = enc_cmd(bindings::V4L2_ENC_CMD_STOP);
    stop.flags = !0;
    let out = r.device.try_encoder_cmd(&s, stop).unwrap();
    assert_eq!((out.cmd, out.flags), (bindings::V4L2_ENC_CMD_STOP, 0));
    assert!(r
        .device
        .encoder_cmd(&mut s, enc_cmd(bindings::V4L2_ENC_CMD_STOP))
        .is_ok());
    let mut start = enc_cmd(bindings::V4L2_ENC_CMD_START);
    start.flags = !0;
    assert_eq!(r.device.try_encoder_cmd(&s, start).unwrap().flags, 0);
    assert_eq!(
        r.device
            .encoder_cmd(&mut s, enc_cmd(bindings::V4L2_ENC_CMD_PAUSE))
            .err(),
        Some(libc::EINVAL)
    );
    assert_eq!(
        r.device
            .encoder_cmd(&mut s, enc_cmd(bindings::V4L2_ENC_CMD_RESUME))
            .err(),
        Some(libc::EINVAL)
    );

    // Both queues stream, no raw frame queued, STOP: the next CAPTURE buffer is LAST and empty.
    r.device
        .s_fmt(&mut s, OUTPUT, output_format(NV12, SIZE.0, SIZE.1))
        .unwrap();
    r.device
        .reqbufs(&mut s, OUTPUT, MemoryType::Mmap, 2)
        .unwrap();
    let sizeimage = pix(&r.device.g_fmt(&s, CAPTURE).unwrap()).sizeimage;
    r.device
        .reqbufs(&mut s, CAPTURE, MemoryType::Mmap, 2)
        .unwrap();
    r.device.streamon(&mut s, OUTPUT).unwrap();
    r.device.streamon(&mut s, CAPTURE).unwrap();
    assert!(r
        .device
        .encoder_cmd(&mut s, enc_cmd(bindings::V4L2_ENC_CMD_STOP))
        .is_ok());
    assert_eq!(
        r.device
            .encoder_cmd(&mut s, enc_cmd(bindings::V4L2_ENC_CMD_STOP))
            .err(),
        Some(libc::EBUSY)
    );
    assert_eq!(
        r.device
            .encoder_cmd(&mut s, enc_cmd(bindings::V4L2_ENC_CMD_START))
            .err(),
        Some(libc::EBUSY)
    );
    r.device
        .qbuf(
            &mut s,
            mmap_buffer(CAPTURE, 0, sizeimage),
            vec![],
            PayloadValidity::ALL,
        )
        .unwrap();
    collect_capture(&mut r, &mut s, 1);
    let last = dequeued_on(&r.events.borrow(), CAPTURE).pop().unwrap();
    assert!(last.flags().contains(BufferFlags::LAST));
    assert_eq!(*last.get_first_plane().bytesused, 0);
    assert_eq!(
        eos_events(&r.events.borrow()),
        0,
        "EOS only when subscribed"
    );
    // A CAPTURE buffer queued after the LAST one is held, not lent, until START.
    r.device
        .qbuf(
            &mut s,
            mmap_buffer(CAPTURE, 1, sizeimage),
            vec![],
            PayloadValidity::ALL,
        )
        .unwrap();
    assert!(!s.output.buffers[1].lent);
    assert!(
        r.device
            .encoder_cmd(&mut s, enc_cmd(bindings::V4L2_ENC_CMD_STOP))
            .is_ok(),
        "STOP after a drain is a no-op"
    );
    assert!(r
        .device
        .encoder_cmd(&mut s, enc_cmd(bindings::V4L2_ENC_CMD_START))
        .is_ok());
    assert_eq!(r.log.lock().unwrap().flushes, 1, "START resumes the codec");
    assert!(s.output.buffers[1].lent);
    queue_frame(&mut r, &mut s, 0, 0x01, 0);
    collect_capture(&mut r, &mut s, 2);
    // The compliance probe's second case: STREAMOFF/STREAMON both, one CAPTURE buffer, STOP.
    r.device.streamoff(&mut s, OUTPUT).unwrap();
    r.device.streamoff(&mut s, CAPTURE).unwrap();
    r.device.streamon(&mut s, OUTPUT).unwrap();
    r.device.streamon(&mut s, CAPTURE).unwrap();
    r.device
        .qbuf(
            &mut s,
            mmap_buffer(CAPTURE, 0, sizeimage),
            vec![],
            PayloadValidity::ALL,
        )
        .unwrap();
    assert!(r
        .device
        .encoder_cmd(&mut s, enc_cmd(bindings::V4L2_ENC_CMD_STOP))
        .is_ok());
    collect_capture(&mut r, &mut s, 3);
    let last = dequeued_on(&r.events.borrow(), CAPTURE).pop().unwrap();
    assert!(last.flags().contains(BufferFlags::LAST));
    assert_eq!(*last.get_first_plane().bytesused, 0);
    close(&mut r.device, s);
}

/// `CREATE_BUFS` sizes buffers itself: it refuses a set with the wrong plane count or a
/// `sizeimage` smaller than the queue's format needs (D6.2/D9) on either queue, and honours a
/// larger one; `PREPARE_BUF` validates a payload without queueing, and the `QBUF` that follows
/// keeps the prepared description (D6.1).
#[test]
fn create_bufs_refuses_a_set_too_small_and_prepare_buf_keeps_its_payload() {
    let mut r = rig();
    let mut s = session(&mut r.device);
    r.device
        .s_fmt(&mut s, OUTPUT, output_format(NV12, SIZE.0, SIZE.1))
        .unwrap();
    let bitstream = pix(&r.device.g_fmt(&s, CAPTURE).unwrap()).sizeimage;

    let noplane = capture_format_sized(H264, 0, bitstream);
    assert_eq!(
        r.device
            .create_bufs(&mut s, 1, CAPTURE, MemoryType::Mmap, noplane)
            .err(),
        Some(libc::EINVAL)
    );
    let small = capture_format_sized(H264, 1, bitstream / 2);
    assert_eq!(
        r.device
            .create_bufs(&mut s, 1, CAPTURE, MemoryType::Mmap, small)
            .err(),
        Some(libc::EINVAL)
    );
    let mut small_raw = output_format(NV12, SIZE.0, SIZE.1);
    // SAFETY: multi-planar; writing a packed field of a local.
    unsafe { small_raw.fmt.pix_mp.plane_fmt[0].sizeimage = RAW_SIZEIMAGE - 1 };
    assert_eq!(
        r.device
            .create_bufs(&mut s, 1, OUTPUT, MemoryType::Mmap, small_raw)
            .err(),
        Some(libc::EINVAL)
    );

    let big = capture_format_sized(H264, 1, bitstream * 2);
    let reply = r
        .device
        .create_bufs(&mut s, 1, CAPTURE, MemoryType::Mmap, big)
        .unwrap();
    assert_eq!((reply.index, reply.count), (0, 1));
    assert_eq!(pix(&reply.format).sizeimage, bitstream * 2);
    assert_eq!(
        *r.device
            .querybuf(&s, CAPTURE, 0)
            .unwrap()
            .get_first_plane()
            .length,
        bitstream * 2
    );

    // PREPARE_BUF on a guest-owned raw frame, then a QBUF whose own payload is nonsense.
    r.device
        .reqbufs(&mut s, OUTPUT, MemoryType::UserPtr, 1)
        .unwrap();
    let (pb, _) = userptr_buffer(OUTPUT, 0, 0x4000, RAW_SIZEIMAGE, RAW_SIZEIMAGE);
    let prepared = r
        .device
        .prepare_buf(&mut s, pb, vec![], PayloadValidity::ALL)
        .unwrap();
    assert!(prepared.flags().contains(BufferFlags::PREPARED));
    assert_eq!(*prepared.get_first_plane().bytesused, RAW_SIZEIMAGE);
    let (mut qb, sgs) = userptr_buffer(OUTPUT, 0, 0x4000, RAW_SIZEIMAGE, 0xdead_beef);
    *qb.get_first_plane_mut().bytesused = 0xdead_beef;
    let queued = r
        .device
        .qbuf(&mut s, qb, sgs, PayloadValidity::ALL)
        .unwrap();
    assert!(queued.flags().contains(BufferFlags::QUEUED));
    assert_eq!(
        *queued.get_first_plane().bytesused,
        RAW_SIZEIMAGE,
        "the prepared bytesused survived"
    );
    // Preparing a queued buffer, or twice, is refused.
    let (pb, _) = userptr_buffer(OUTPUT, 0, 0x4000, RAW_SIZEIMAGE, RAW_SIZEIMAGE);
    assert_eq!(
        r.device
            .prepare_buf(&mut s, pb, vec![], PayloadValidity::ALL)
            .err(),
        Some(libc::EINVAL)
    );

    close(&mut r.device, s);
}

/// `SUBSCRIBE_EVENT`: EOS is accepted, SOURCE_CHANGE refused (`v4l2-test-controls.cpp:1200-1201`
/// for a stateful encoder), a control event is accepted for a known control and, with
/// `SEND_INITIAL`, answered with the control's state at once -- for every control but the class
/// ones (`testEvents`, `:1114-1150`); an unknown control is EINVAL.
#[test]
fn events_are_eos_and_control_events_only() {
    let mut r = rig();
    let mut s = session(&mut r.device);
    assert!(r
        .device
        .subscribe_event(&mut s, EventType::Eos, SubscribeEventFlags::empty())
        .is_ok());
    assert_eq!(
        r.device
            .subscribe_event(
                &mut s,
                EventType::SourceChange(0),
                SubscribeEventFlags::empty()
            )
            .err(),
        Some(libc::EINVAL)
    );
    assert_eq!(
        r.device
            .subscribe_event(&mut s, EventType::VSync, SubscribeEventFlags::empty())
            .err(),
        Some(libc::EINVAL)
    );
    r.device.s_ctrl(&mut s, CID_BITRATE, 6_000_000).unwrap();
    assert!(r
        .device
        .subscribe_event(
            &mut s,
            EventType::Ctrl(CID_BITRATE),
            SubscribeEventFlags::SEND_INITIAL
        )
        .is_ok());
    assert!(r
        .device
        .subscribe_event(
            &mut s,
            EventType::Ctrl(bindings::V4L2_CID_CODEC_CLASS),
            SubscribeEventFlags::SEND_INITIAL
        )
        .is_ok());
    assert!(r
        .device
        .subscribe_event(
            &mut s,
            EventType::Ctrl(CID_FORCE_KEY),
            SubscribeEventFlags::SEND_INITIAL
        )
        .is_ok());
    assert!(r
        .device
        .subscribe_event(
            &mut s,
            EventType::Ctrl(CID_GOP),
            SubscribeEventFlags::empty()
        )
        .is_ok());
    assert_eq!(
        r.device
            .subscribe_event(
                &mut s,
                EventType::Ctrl(CID_MIN_CAP),
                SubscribeEventFlags::SEND_INITIAL
            )
            .err(),
        Some(libc::EINVAL)
    );
    assert_eq!(
        ctrl_events(&r.events.borrow()),
        vec![(CID_BITRATE, 6_000_000), (CID_FORCE_KEY, 0)]
    );
    let unsub = |type_: u32, id: u32| v4l2_event_subscription {
        type_,
        id,
        ..Default::default()
    };
    assert!(r
        .device
        .unsubscribe_event(&mut s, unsub(bindings::V4L2_EVENT_CTRL, CID_BITRATE))
        .is_ok());
    assert!(r
        .device
        .unsubscribe_event(&mut s, unsub(bindings::V4L2_EVENT_EOS, 0))
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
    let _ = start_streaming(&mut r, &mut s);

    r.device
        .handle_event(&mut s, EncoderEvent::Error("codec reclaimed".into()));
    assert!(s.dead);
    assert_eq!(errors(&r.events.borrow()), 1);
    assert_eq!(r.log.lock().unwrap().stops, 1);

    fill_mmap_output(&mut s, 0, 0x01);
    assert_eq!(
        r.device
            .qbuf(
                &mut s,
                mmap_buffer(OUTPUT, 0, RAW_SIZEIMAGE),
                vec![],
                PayloadValidity::ALL
            )
            .err(),
        Some(libc::ENODEV)
    );
    assert_eq!(r.device.streamon(&mut s, OUTPUT).err(), Some(libc::ENODEV));
    assert_eq!(
        r.device.s_ctrl(&mut s, CID_BITRATE, 1_000_000).err(),
        Some(libc::ENODEV)
    );
    assert_eq!(
        r.device
            .encoder_cmd(&mut s, enc_cmd(bindings::V4L2_ENC_CMD_STOP))
            .err(),
        Some(libc::ENODEV)
    );
    // A dead session takes no more pool space (review-m6 R6-11's shape): the three allocating
    // ioctls are refused, and `REQBUFS(0)` still frees.
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
                capture_format(H264, 1 << 20)
            )
            .err(),
        Some(libc::ENODEV)
    );
    assert_eq!(
        r.device
            .prepare_buf(
                &mut s,
                mmap_buffer(OUTPUT, 0, RAW_SIZEIMAGE),
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

/// `V4L2_ENC_CMD_START` after a finished drain queues again every raw frame the codec never got
/// to (review-m7 R7-2): the frames lent while the drain ran sat behind its `EOS` in the backend,
/// which drops them at the flush without a word (the trait's `flush` contract, R7-11), so the
/// device must put them back itself, as it does for a `STREAMOFF(CAPTURE)` reset. Before the
/// fix they stayed `queued && lent` for the rest of the stream: no `DQBUF` could ever return
/// them.
#[test]
fn enc_cmd_start_after_a_drain_queues_the_frames_the_codec_never_took() {
    let mut r = rig();
    let mut s = session(&mut r.device);
    let sizeimage = start_streaming(&mut r, &mut s);
    queue_frame(&mut r, &mut s, 0, 0x30, 0);
    queue_frame(&mut r, &mut s, 1, 0x31, 1);
    collect_capture(&mut r, &mut s, 2);
    collect_output(&mut r, &mut s, 2);

    // Drain, with two frames arriving while it runs: lent, parked behind the EOS.
    r.device
        .encoder_cmd(&mut s, enc_cmd(bindings::V4L2_ENC_CMD_STOP))
        .unwrap();
    queue_frame(&mut r, &mut s, 2, 0x32, 2);
    queue_frame(&mut r, &mut s, 3, 0x33, 3);
    assert!(s.input.buffers[2].lent && s.input.buffers[3].lent);
    collect_capture(&mut r, &mut s, 3);
    let last = dequeued_on(&r.events.borrow(), CAPTURE).pop().unwrap();
    assert!(last.flags().contains(BufferFlags::LAST));
    assert_eq!(*last.get_first_plane().bytesused, 0);
    assert_eq!(s.drain, Drain::Done);
    assert!(
        s.input.buffers[2].lent && s.input.buffers[3].lent,
        "still with the backend when the LAST buffer is out"
    );

    // START: the backend flushes (and says nothing about the two), the device queues them again
    // in front of anything newer, and the resumed codec encodes them in order.
    r.device
        .encoder_cmd(&mut s, enc_cmd(bindings::V4L2_ENC_CMD_START))
        .unwrap();
    assert_eq!(r.log.lock().unwrap().flushes, 1);
    assert_eq!(
        r.log.lock().unwrap().dropped_by_flush,
        2,
        "the backend dropped both without a report"
    );
    assert_eq!(s.drain, Drain::None);
    assert!(
        s.input.buffers[2].queued && s.input.buffers[3].queued,
        "queued again for the resumed codec"
    );
    // One CAPTURE buffer was left lent; a second one takes the other frame.
    r.device
        .qbuf(
            &mut s,
            mmap_buffer(CAPTURE, 0, sizeimage),
            vec![],
            PayloadValidity::ALL,
        )
        .unwrap();
    collect_capture(&mut r, &mut s, 5);
    collect_output(&mut r, &mut s, 4);
    let packets = dequeued_on(&r.events.borrow(), CAPTURE);
    assert_eq!(
        (packets[3].timestamp().tv_sec, packets[4].timestamp().tv_sec),
        (2, 3),
        "the two frames, in order"
    );
    assert!(
        s.input.buffers.iter().all(|b| !b.queued && !b.lent),
        "every raw frame is back with the guest"
    );
    let units = parse_units(&capture_bytes(
        &s,
        packets[4].index() as usize,
        *packets[4].get_first_plane().bytesused as usize,
    ));
    assert_eq!(
        units,
        vec![Unit::Frame {
            key: false,
            frame_no: 3,
            digest: digest_of(0x33, SIZE)
        }],
        "the same stream continues"
    );
    close(&mut r.device, s);
}

/// `STREAMOFF(OUTPUT)` never fails (review-m7 R7-3, review-m6 R6-4): a flush the backend cannot
/// complete -- a wedged codec past its bound -- ends the session instead, the queue is reset,
/// the buffers unqueued, and `REQBUFS(0)` frees the pool space.
#[test]
fn streamoff_never_fails_a_backend_that_will_not_flush() {
    let mut r = rig();
    let mut s = session(&mut r.device);
    start_streaming(&mut r, &mut s);
    queue_frame(&mut r, &mut s, 0, 0x40, 0);
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
    r.device
        .reqbufs(&mut s, OUTPUT, MemoryType::Mmap, 0)
        .unwrap();
    r.device
        .reqbufs(&mut s, CAPTURE, MemoryType::Mmap, 0)
        .unwrap();
    assert!(s.input.buffers.is_empty() && s.output.buffers.is_empty());
    assert_eq!(r.device.active_session, None);
    close(&mut r.device, s);
}

/// `S_FMT(CAPTURE)` honours a client's bitstream buffer size only up to what the raw size can
/// need (review-m7 R7-4): GStreamer asks for half the *probed maximum* frame -- 32 MiB against an
/// 8192x8192 codec -- for every buffer of its pool, out of a `media_host` pool the decoder
/// shares. Four times the device's own default for the negotiated size, never under 4 MiB.
#[test]
fn s_fmt_capture_caps_the_bitstream_size_at_what_the_raw_size_can_need() {
    let mut r = rig();
    let mut s = session(&mut r.device);
    // 640x480 (the default raw size): the default is the 256 KiB floor, the cap 4 MiB.
    let set = pix(&r
        .device
        .s_fmt(&mut s, CAPTURE, capture_format(H264, 32 << 20))
        .unwrap());
    assert_eq!(set.sizeimage, 4 << 20);
    // A plausible ask is still honoured as it is.
    let set = pix(&r
        .device
        .s_fmt(&mut s, CAPTURE, capture_format(H264, 3 << 20))
        .unwrap());
    assert_eq!(set.sizeimage, 3 << 20);
    // At 4096x2160 the default is 6 635 520 bytes, so the cap is four times that.
    r.device
        .s_fmt(&mut s, OUTPUT, output_format(NV12, 4096, 2160))
        .unwrap();
    let tried = pix(&r
        .device
        .try_fmt(&s, CAPTURE, capture_format(H264, 32 << 20))
        .unwrap());
    assert_eq!(tried.sizeimage, 4 * (4096 * 2160 * 3 / 2 / 2));
    assert_eq!(
        nv12_sizeimage(u32::MAX, u32::MAX),
        u32::MAX,
        "and never an abort"
    );
    close(&mut r.device, s);
}

/// `S_SELECTION(OUTPUT, CROP)` fits the rectangle to the coded format the way `S_FMT` fits the
/// frame (review-m7 R7-7): the codec is configured for exactly this rectangle, so a legal-looking
/// crop it cannot take (641x481, 64x64 on a codec whose minimum is 128) is adjusted here, where
/// the ioctl may adjust, instead of failing at `STREAMON`.
#[test]
fn s_selection_fits_the_crop_to_the_coded_format() {
    let mut r = rig();
    let mut s = session(&mut r.device);
    r.device
        .s_fmt(&mut s, OUTPUT, output_format(NV12, SIZE.0, SIZE.1))
        .unwrap();
    let crop = |r: &mut Rig, s: &mut Session, left: i32, top: i32, width: u32, height: u32| {
        let got = r
            .device
            .s_selection(
                s,
                SelectionType::Output,
                SelectionTarget::Crop,
                bindings::v4l2_rect {
                    left,
                    top,
                    width,
                    height,
                },
                SelectionFlags::empty(),
            )
            .unwrap();
        (got.left, got.top, got.width, got.height)
    };
    // H264: 16..4096 step 2. Odd extents round up (an encoder pads), an odd origin rounds down
    // (its chroma samples must exist whole), and nothing leaves the frame.
    assert_eq!(crop(&mut r, &mut s, 2, 2, 301, 201), (2, 2, 302, 202));
    assert_eq!(crop(&mut r, &mut s, 0, 0, 319, 239), (0, 0, 320, 240));
    assert_eq!(crop(&mut r, &mut s, 1, 3, 0, 0), (0, 2, 320, 238));
    assert_eq!(crop(&mut r, &mut s, 0, 0, 317, 237), (0, 0, 318, 238));
    // HEVC: 64..8192 step 8. A crop below the minimum grows to it; an origin too far in for a
    // minimum crop moves back.
    r.device
        .s_fmt(&mut s, CAPTURE, capture_format(HEVC, 0))
        .unwrap();
    assert_eq!(crop(&mut r, &mut s, 0, 0, 100, 100), (0, 0, 104, 104));
    assert_eq!(crop(&mut r, &mut s, 0, 0, 30, 30), (0, 0, 64, 64));
    assert_eq!(crop(&mut r, &mut s, 300, 0, 0, 0), (256, 0, 64, 240));
    let got = r
        .device
        .g_selection(&s, SelectionType::Output, SelectionTarget::Crop)
        .unwrap();
    assert_eq!((got.left, got.width), (256, 64));
    close(&mut r.device, s);
}

/// A drain the backend refuses leaves no drain pending (review-m7 R7-12): the session is not
/// left answering `EBUSY` to every later command while waiting for a `LAST` buffer no drain
/// will produce.
#[test]
fn a_refused_drain_leaves_no_drain_pending() {
    let mut r = rig();
    let mut s = session(&mut r.device);
    start_streaming(&mut r, &mut s);
    r.log.lock().unwrap().fail_drain = Some(libc::EIO);
    assert_eq!(
        r.device
            .encoder_cmd(&mut s, enc_cmd(bindings::V4L2_ENC_CMD_STOP))
            .err(),
        Some(libc::EIO)
    );
    assert_eq!(s.drain, Drain::None);
    r.device
        .encoder_cmd(&mut s, enc_cmd(bindings::V4L2_ENC_CMD_STOP))
        .unwrap();
    assert_eq!(s.drain, Drain::Pending);
    collect_capture(&mut r, &mut s, 1);
    let last = dequeued_on(&r.events.borrow(), CAPTURE).pop().unwrap();
    assert!(last.flags().contains(BufferFlags::LAST));
    assert_eq!(eos_events(&r.events.borrow()), 1);
    close(&mut r.device, s);
}
