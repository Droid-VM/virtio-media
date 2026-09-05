// Copyright 2026 The DroidVM Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

//! A V4L2 capture device over a host camera (`VPU_DESIGN.md` §7.1).
//!
//! This module is the V4L2 half of the camera and knows nothing about where the frames come
//! from: a [`CameraBackend`] describes one camera ([`CameraInfo`]) and opens a [`CameraStream`]
//! at a size and frame rate; the stream fills the buffers it is lent and says so. The host-side
//! half -- on DroidVM, `AndroidCameraBackend` in crosvm over the Camera2 NDK -- lives in the VMM,
//! so this crate stays free of Android.
//!
//! # What the guest sees
//!
//! `V4L2_CAP_VIDEO_CAPTURE_MPLANE | V4L2_CAP_STREAMING`, one format (`NV12`, one plane, tightly
//! packed: `bytesperline = width`), the camera's `YUV_420_888` output sizes as discrete frame
//! sizes, and the frame intervals the camera's target-fps ranges allow at each size. `G_PARM` /
//! `S_PARM` carry a single `timeperframe`, which the design maps to the *widest* supported fps
//! range whose maximum is that rate (`VIDEO_MEDIA_PLAN.md` D12 row 12): pinning the minimum too
//! is what makes a phone camera darker than its own camera app in low light. `S_FMT` is refused
//! with `EBUSY` while buffers exist, as `vb2` does. Buffers are host-owned (`MMAP`, from the
//! device's [`VirtioMediaBufferAllocator`] -- the `media_host` pool on DroidVM) or guest-owned
//! (`USERPTR`, `driver_owned_queues=all`), on the one `CAPTURE` queue.
//!
//! # Threads and buffers
//!
//! Frames arrive on a thread the backend owns -- a camera API is a stream of callbacks, and on
//! Android the camera handle cannot even leave the thread that opened it -- while every ioctl
//! runs on the device's worker thread. The two meet in three places, none of which blocks the
//! worker:
//!
//! * a buffer the guest queued is *lent* to the stream ([`CameraStream::give_empty`]) as a raw
//!   pointer and length ([`EmptyBuffer`]); the stream owns those bytes until it hands the buffer
//!   back, and nothing else touches them meanwhile;
//! * a filled buffer comes back through [`CameraStream::take_filled`] with its size, timestamp
//!   and sequence number, and the stream bumps the session's [`CaptureSink`] -- an eventfd the
//!   worker polls (`poll_fd`) -- so `process_events` runs and sends the `DQBUF` event;
//! * `STREAMOFF`, `REQBUFS(0)`, a session close and a camera error all *close the stream first*,
//!   which joins its thread, and only then unqueue or free the buffers (`VPU_DESIGN.md` §2.5):
//!   a lent buffer is never released while the thread that may be writing into it is alive.
//!
//! The stream is opened at `STREAMON` -- that is when the camera is taken from the host, and a
//! first frame is a couple of hundred milliseconds away -- and closed at `STREAMOFF`, which gives
//! the camera back to Android.
//!
//! # The frame copy contract
//!
//! An [`EmptyBuffer`] is `len` bytes at `ptr`, sized for one frame of the stream's dimensions,
//! and the stream fills it as tightly packed `NV12`: `height` rows of `stride` (`= width`) luma
//! bytes, then `height / 2` rows of `stride` interleaved Cb/Cr bytes. `bytesused` of the returned
//! [`FilledBuffer`] is the number of bytes written, normally exactly `width * height * 3 / 2`.
//! Converting from whatever the camera produces (NV21, I420, padded rows) is the backend's job;
//! the device never looks at the pixels.

use std::collections::VecDeque;
use std::io::Result as IoResult;
use std::os::fd::AsFd;
use std::os::fd::AsRawFd;
use std::os::fd::BorrowedFd;
use std::os::fd::FromRawFd;
use std::os::fd::OwnedFd;
use std::sync::Arc;

use v4l2r::bindings;
use v4l2r::bindings::v4l2_create_buffers;
use v4l2r::bindings::v4l2_event_subscription;
use v4l2r::bindings::v4l2_fmtdesc;
use v4l2r::bindings::v4l2_format;
use v4l2r::bindings::v4l2_frmivalenum;
use v4l2r::bindings::v4l2_frmsizeenum;
use v4l2r::bindings::v4l2_requestbuffers;
use v4l2r::bindings::v4l2_streamparm;
use v4l2r::ioctl::BufferCapabilities;
use v4l2r::ioctl::BufferField;
use v4l2r::ioctl::BufferFlags;
use v4l2r::ioctl::EventType;
use v4l2r::ioctl::SubscribeEventFlags;
use v4l2r::ioctl::V4l2Buffer;
use v4l2r::ioctl::V4l2PlanesWithBackingMut;
use v4l2r::memory::MemoryType;
use v4l2r::PixelFormat;
use v4l2r::QueueType;

use crate::guest_mapping_errno;
use crate::ioctl::virtio_media_dispatch_ioctl;
use crate::ioctl::IoctlResult;
use crate::ioctl::VirtioMediaIoctlHandler;
use crate::mmap::MmapMappingManager;
use crate::mmap::RetiredBuffers;
use crate::protocol::DequeueBufferEvent;
use crate::protocol::SgEntry;
use crate::protocol::V4l2Event;
use crate::protocol::V4l2Ioctl;
use crate::protocol::VIRTIO_MEDIA_MMAP_FLAG_RW;
use crate::GuestMemoryRange;
use crate::HostBuffer;
use crate::ReadFromDescriptorChain;
use crate::VirtioMediaBufferAllocator;
use crate::VirtioMediaDevice;
use crate::VirtioMediaDeviceSession;
use crate::VirtioMediaEventQueue;
use crate::VirtioMediaGuestMemoryMapper;
use crate::VirtioMediaHostMemoryMapper;
use crate::WriteToDescriptorChain;

/// The one pixel format offered: Y plane then interleaved Cb/Cr, tightly packed.
pub const NV12: PixelFormat = PixelFormat::from_fourcc(b"NV12");
/// Most buffers on the queue, the usual V4L2 ceiling.
pub const MAX_BUFFERS: usize = 32;
/// The queue this device has.
const QUEUE: QueueType = QueueType::VideoCaptureMplane;
/// The frame rate a session starts at, when the camera offers it; also what an `S_PARM` asking
/// for `0/0` (or any other fraction that is not a rate) falls back to.
const DEFAULT_FPS: u32 = 30;
/// The size a session starts at, or the nearest the camera offers: what a guest that never calls
/// `S_FMT` records at. Not the largest size, which on a phone is a 12 MiB still-photo frame.
const DEFAULT_SIZE: (u32, u32) = (1280, 720);

// ---------------------------------------------------------------------------------------------
// What a backend provides
// ---------------------------------------------------------------------------------------------

/// One output size a camera offers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameSize {
    pub width: u32,
    pub height: u32,
    /// The shortest frame duration the camera sustains at this size, when it says
    /// (`SCALER_AVAILABLE_MIN_FRAME_DURATIONS` on Android); caps the frame intervals offered.
    pub min_frame_duration_ns: Option<u64>,
}

impl FrameSize {
    pub const fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            min_frame_duration_ns: None,
        }
    }

    /// The fastest whole frame rate this size sustains, if the camera said.
    fn max_fps(&self) -> Option<u32> {
        self.min_frame_duration_ns
            .filter(|&ns| ns > 0)
            .map(|ns| (1_000_000_000 / ns).clamp(1, u32::MAX as u64) as u32)
    }

    /// Bytes of one tightly packed NV12 frame. Odd dimensions round the chroma plane up.
    ///
    /// Saturating, not because a guest reaches this -- every `FrameSize` here comes from the
    /// camera's own list, and a guest request is snapped to one of those by [`nearest_size`]
    /// first -- but because a plain `u32` product of two dimensions is one host camera away from
    /// aborting the helper, and a saturated size merely fails to allocate (D18's shape).
    ///
    /// [`nearest_size`]: CameraInfo::nearest_size
    fn sizeimage(&self) -> u32 {
        let luma = self.width.saturating_mul(self.height);
        let chroma = self
            .width
            .div_ceil(2)
            .saturating_mul(self.height.div_ceil(2))
            .saturating_mul(2);
        luma.saturating_add(chroma)
    }
}

/// What a camera can do, in the terms `ENUM_FRAMESIZES`, `ENUM_FRAMEINTERVALS` and `G/S_PARM`
/// are answered with. Controls (`VPU_DESIGN.md` §7.1, M5) extend this.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CameraInfo {
    /// The host's name for the camera (`"0"` on Android).
    pub id: String,
    /// A human-readable name, what `ENUMINPUT` reports.
    pub name: String,
    /// Output sizes, in `ENUM_FRAMESIZES` order.
    pub sizes: Vec<FrameSize>,
    /// `(min, max)` target frame-rate ranges the camera accepts
    /// (`CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES` on Android). A range whose ends differ lets the
    /// camera slow down in low light.
    pub fps_ranges: Vec<(u32, u32)>,
}

impl CameraInfo {
    /// The frame rates a guest can ask for at `size`, fastest first: every distinct maximum of
    /// the fps ranges -- `S_PARM` selects by maximum, so these are exactly the rates it can
    /// honour -- that the size sustains. A camera that lists no ranges offers the size's own
    /// ceiling, or [`DEFAULT_FPS`].
    fn frame_rates(&self, size: &FrameSize) -> Vec<u32> {
        let cap = size.max_fps();
        let mut rates: Vec<u32> = self
            .fps_ranges
            .iter()
            .map(|&(_, max)| max)
            .filter(|&max| max > 0 && cap.is_none_or(|cap| max <= cap))
            .collect();
        if rates.is_empty() {
            rates.push(cap.unwrap_or(DEFAULT_FPS));
        }
        rates.sort_unstable_by(|a, b| b.cmp(a));
        rates.dedup();
        rates
    }

    /// The fps range to run at for a guest that asked for `fps` frames per second: the widest
    /// range whose maximum is `fps`; failing an exact match, the widest among those whose
    /// maximum is nearest (the faster one on a tie). A camera that lists no usable range gets
    /// `(fps, fps)`.
    ///
    /// `cap` is the ceiling the chosen range must also respect -- the size's own, when a size
    /// is in question. A range whose maximum is above it must not be chosen: the camera would
    /// be opened at a rate `ENUM_FRAMEINTERVALS` never offered, and `G_PARM` would report a
    /// rate the size cannot deliver. [`Self::frame_rates`] filters exactly the same way, so the
    /// two ioctls answer from one list (review-m4 R5); when nothing fits under the ceiling,
    /// both answer the ceiling itself.
    fn range_for(&self, fps: u32, cap: Option<u32>) -> (u32, u32) {
        let fits = |&&(_, max): &&(u32, u32)| max > 0 && cap.is_none_or(|cap| max <= cap);
        let nearest = self
            .fps_ranges
            .iter()
            .filter(fits)
            .map(|&(_, max)| max)
            .min_by_key(|&max| (max.abs_diff(fps), std::cmp::Reverse(max)));
        match nearest {
            Some(max) => self
                .fps_ranges
                .iter()
                .copied()
                .filter(|&(_, m)| m == max)
                .min_by_key(|&(min, _)| min)
                .unwrap_or((max, max)),
            None => (fps, fps),
        }
    }

    /// The range to open `size` at when the guest asked for `fps`: the rate is first held to
    /// what the size sustains, and so is the range chosen for it.
    fn range_at(&self, size: &FrameSize, fps: u32) -> (u32, u32) {
        let cap = size.max_fps();
        let fps = match cap {
            Some(cap) => fps.min(cap),
            None => fps,
        };
        self.range_for(fps, cap)
    }

    /// The size nearest to `width`x`height` (least squared distance in both dimensions), or
    /// `None` for a camera without sizes.
    ///
    /// The guest picks `width` and `height` and V4L2 puts no ceiling on either, so the metric is
    /// computed in `u128`. It used to be `i64`, which overflows: the largest squared distance a
    /// pair of `u32`s can produce is just under `2 * (2^32)^2`, past `i64` and past `u64` too.
    /// The helper is built with `-C overflow-checks=on` and `panic = abort`, so the overflow was
    /// an abort of the whole process -- any guest could end its own VM with one `TRY_FMT`, and
    /// `v4l2-compliance` did it by accident (D18).
    ///
    /// `u128` needs no clamp in front of it: the metric is then exact for every pair a guest can
    /// send, and since it only grows past the largest listed size, an absurd request snaps to
    /// that size, which is what a clamp would have arranged anyway.
    fn nearest_size(&self, width: u32, height: u32) -> Option<FrameSize> {
        self.sizes.iter().copied().min_by_key(|s| {
            let dw = s.width.abs_diff(width) as u128;
            let dh = s.height.abs_diff(height) as u128;
            dw * dw + dh * dh
        })
    }

    fn default_size(&self) -> Option<FrameSize> {
        self.nearest_size(DEFAULT_SIZE.0, DEFAULT_SIZE.1)
    }
}

/// A pointer into a buffer the guest maps too, handed to the capture thread.
///
/// Raw pointers are not `Send`; this one is, because what it points at -- a host buffer from the
/// allocator or a guest mapping held for as long as the buffer is queued -- is plain shared
/// memory with no thread affinity, and because the device lends each buffer to exactly one
/// stream and takes it back only after that stream's thread is gone.
#[derive(Clone, Copy, Debug)]
pub struct SendPtr(*mut u8);

// SAFETY: see the type's documentation: the pointee is shared memory that stays mapped for the
// life of the loan, and the loan is exclusive.
unsafe impl Send for SendPtr {}

impl SendPtr {
    pub fn as_ptr(&self) -> *mut u8 {
        self.0
    }
}

/// A buffer lent to a stream to be filled. See the module documentation for the copy contract.
#[derive(Clone, Copy, Debug)]
pub struct EmptyBuffer {
    /// The V4L2 buffer index; comes back in [`FilledBuffer::index`].
    pub index: u32,
    pub ptr: SendPtr,
    /// Bytes available at `ptr`: one frame at the stream's dimensions.
    pub len: usize,
    /// Bytes per row of the destination, both planes: the stream's width.
    pub stride: u32,
}

/// A buffer the stream has filled and gives back.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FilledBuffer {
    pub index: u32,
    /// Bytes written, from the start of the buffer.
    pub bytesused: u32,
    /// The frame's capture timestamp, on the host's monotonic clock.
    pub timestamp_ns: i64,
    /// Frame number since the stream opened, from 0.
    pub sequence: u32,
}

/// Something the stream reports besides frames.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CameraEvent {
    /// The host took the camera away (another app, a policy). The stream is over.
    Disconnected,
    /// The stream failed and produces no more frames; the string is for the log.
    Error(String),
}

/// A control applied to an open stream. M5 adds the camera controls proper.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CameraControl {
    /// `CONTROL_AE_TARGET_FPS_RANGE`: what `S_PARM` during streaming asks for.
    FpsRange(u32, u32),
}

/// What a stream is opened for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StreamRequest {
    pub width: u32,
    pub height: u32,
    /// `(min, max)` frames per second, already resolved by the fps-range rule.
    pub fps: (u32, u32),
    /// How many buffers the guest allocated, for a backend that sizes a queue of its own.
    pub buffers: u32,
}

/// An eventfd a session's worker polls. Bumped once per filled buffer and per event; drained by
/// the device before it collects them, so a bump that lands in between leaves it readable.
pub struct FrameSignal(OwnedFd);

impl FrameSignal {
    pub fn new() -> Result<Self, i32> {
        // SAFETY: eventfd takes no pointers; the descriptor is checked before it is owned.
        let fd = unsafe { libc::eventfd(0, libc::EFD_CLOEXEC | libc::EFD_NONBLOCK) };
        if fd < 0 {
            return Err(std::io::Error::last_os_error()
                .raw_os_error()
                .unwrap_or(libc::EIO));
        }
        // SAFETY: `fd` was just returned by eventfd and is owned by no one else.
        Ok(Self(unsafe { OwnedFd::from_raw_fd(fd) }))
    }

    /// Add one to the counter. Never blocks; a counter about to overflow (2^64 - 1 bumps
    /// without a drain) would fail with `EAGAIN`, which is ignored because it cannot happen at
    /// frame rates.
    pub fn signal(&self) {
        let one: u64 = 1;
        // SAFETY: writing 8 bytes from a live u64 to a descriptor we own.
        unsafe {
            libc::write(
                self.0.as_raw_fd(),
                &one as *const u64 as *const libc::c_void,
                std::mem::size_of::<u64>(),
            )
        };
    }

    /// Reset the counter, so the descriptor stops being readable until the next bump.
    pub fn drain(&self) {
        let mut count: u64 = 0;
        // SAFETY: reading 8 bytes into a live u64 from a non-blocking descriptor we own; an empty
        // counter is `EAGAIN`, which is the wanted outcome.
        unsafe {
            libc::read(
                self.0.as_raw_fd(),
                &mut count as *mut u64 as *mut libc::c_void,
                std::mem::size_of::<u64>(),
            )
        };
    }
}

impl AsFd for FrameSignal {
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.0.as_fd()
    }
}

/// The session's end of the wake-up path, cloned into whatever thread produces frames. Bump it
/// after every [`FilledBuffer`] and [`CameraEvent`] made available.
#[derive(Clone)]
pub struct CaptureSink(Arc<FrameSignal>);

impl CaptureSink {
    pub fn signal(&self) {
        self.0.signal()
    }
}

/// An open capture stream: frames flowing from a camera into the buffers it is lent.
///
/// Implementations run their own thread; every method here is called on the device's worker
/// thread and must return promptly. Dropping a stream without [`CameraStream::close`] must stop
/// it too, but `close` is what the device calls, so a backend may make it the one that waits.
pub trait CameraStream {
    /// Lend `buffer` to be filled with the next frame. The stream owns the bytes until it
    /// returns the buffer through [`Self::take_filled`] or is closed.
    fn give_empty(&mut self, buffer: EmptyBuffer) -> Result<(), i32>;

    /// Every buffer filled since the last call, oldest first.
    fn take_filled(&mut self) -> Vec<FilledBuffer>;

    /// Every event since the last call.
    fn take_events(&mut self) -> Vec<CameraEvent>;

    /// Apply `controls` to the running stream, in one submission.
    fn set_controls(&mut self, controls: &[CameraControl]) -> Result<(), i32>;

    /// Stop the stream and wait for its thread to end. When this returns no lent buffer is
    /// touched any more; the ones not returned are simply forgotten.
    fn close(self);
}

/// One camera, as a device sees it.
pub trait CameraBackend {
    type Stream: CameraStream;

    fn info(&self) -> &CameraInfo;

    /// Open the camera at `request`'s size and rate. `sink` is what the stream bumps when a
    /// buffer is filled or an event is pending. Errors are `libc` error codes and become the
    /// guest's `STREAMON` result: `EBUSY` for a camera in use, `EACCES` for one this process may
    /// not open.
    fn open_stream(
        &mut self,
        request: StreamRequest,
        sink: CaptureSink,
    ) -> Result<Self::Stream, i32>;
}

// ---------------------------------------------------------------------------------------------
// The device
// ---------------------------------------------------------------------------------------------

/// Where a buffer's bytes live.
enum Backing<GM> {
    /// Host-owned, from the allocator; mappable by the guest at `offset`.
    Host { buffer: HostBuffer, offset: u32 },
    /// Guest-owned; mapped from the guest's `USERPTR` SG list while the buffer is queued.
    Guest(Option<GM>),
}

struct Buffer<GM> {
    /// V4L2 representation of this buffer, what `QUERYBUF` and the dequeue event return.
    v4l2_buffer: V4l2Buffer,
    backing: Backing<GM>,
    /// Queued by the guest and not yet returned.
    queued: bool,
    /// Lent to the stream (implies `queued`).
    lent: bool,
    /// Bytes this buffer was created for: the `sizeimage` `REQBUFS`/`CREATE_BUFS` sized it with.
    size: u32,
    /// `(bytesused, length)` `PREPARE_BUF` accepted for this buffer, while it is prepared.
    prepared: Option<(u32, u32)>,
}

impl<GM: GuestMemoryRange> Buffer<GM> {
    /// Pointer to the first byte, for writing. `None` for a guest buffer that is not mapped.
    fn data_mut_ptr(&mut self) -> Option<*mut u8> {
        match &mut self.backing {
            Backing::Host { buffer, .. } => Some(buffer.as_mut_ptr()),
            Backing::Guest(Some(mapping)) => Some(mapping.as_mut_ptr()),
            Backing::Guest(None) => None,
        }
    }

    /// Bytes the backing can hold. `None` for a guest buffer that is not mapped; for one that
    /// is, whatever the mapping says (see [`GuestMemoryRange::len`]).
    fn data_len(&self) -> Option<usize> {
        match &self.backing {
            Backing::Host { buffer, .. } => Some(buffer.len as usize),
            Backing::Guest(Some(mapping)) => Some(mapping.len()),
            Backing::Guest(None) => None,
        }
    }

    fn drop_guest_mapping(&mut self) {
        if let Backing::Guest(mapping) = &mut self.backing {
            *mapping = None;
        }
    }

    /// Return the buffer to the not-queued state, releasing any guest mapping. Only after the
    /// stream that may have been lent it is gone.
    fn unqueue(&mut self) {
        self.drop_guest_mapping();
        self.queued = false;
        self.lent = false;
        self.prepared = None;
        self.v4l2_buffer
            .clear_flags(BufferFlags::QUEUED | BufferFlags::PREPARED);
    }
}

/// Session data of [`CameraDevice`].
pub struct CameraSession<GM, S> {
    id: u32,
    /// What the worker polls; the stream bumps it through a [`CaptureSink`].
    signal: Arc<FrameSignal>,
    /// The current format's dimensions.
    size: FrameSize,
    /// The frame rate the guest asked for (`S_PARM`), or the default; resolved to a range when
    /// the stream opens.
    fps: u32,
    /// Memory type the buffers were allocated with; `None` while there are none.
    memory: Option<MemoryType>,
    /// The open stream; `Some` exactly while streaming.
    ///
    /// Declared before `buffers`, and the order is load-bearing: a session that is merely
    /// dropped rather than closed drops its fields in declaration order, and a
    /// [`CameraStream`] joins its capture thread when it goes. Putting it first means even a
    /// bare drop joins before a single buffer -- a host buffer's mapping, or a guest mapping
    /// whose `Drop` unmaps the guest's own pages -- is released underneath a thread that is
    /// still writing a frame into it (`VPU_DESIGN.md` §2.5, review-m4 R1). A `Drop` impl
    /// cannot do this job: freeing a host buffer needs the device's allocator, which the
    /// session does not have; [`VirtioMediaDeviceRunner`]'s `Drop` calls `close_session` for
    /// exactly that reason, and this order is what protects the session dropped any other way.
    stream: Option<S>,
    buffers: Vec<Buffer<GM>>,
    /// Indices queued before `STREAMON`, in order; lent when the stream opens.
    queued: VecDeque<usize>,
    /// The camera went away or failed. The guest was sent an error event and treats the session
    /// as closed (the driver fails every further ioctl with `ENODEV`); so does this device for
    /// anything that would touch the stream, until the guest closes it.
    dead: bool,
}

impl<GM, S> VirtioMediaDeviceSession for CameraSession<GM, S> {
    fn poll_fd(&self) -> Option<BorrowedFd> {
        Some(self.signal.as_fd())
    }
}

impl<GM, S> CameraSession<GM, S> {
    fn sizeimage(&self) -> u32 {
        self.size.sizeimage()
    }
}

/// A V4L2 capture device over a [`CameraBackend`]. See the module documentation.
pub struct CameraDevice<
    B: CameraBackend,
    Q: VirtioMediaEventQueue,
    M: VirtioMediaGuestMemoryMapper,
    HM: VirtioMediaHostMemoryMapper,
    A: VirtioMediaBufferAllocator,
> {
    backend: B,
    evt_queue: Q,
    /// Guest memory mapper, for `USERPTR` buffers.
    mem: M,
    mmap_manager: MmapMappingManager<HM>,
    /// Where `MMAP` buffers come from.
    allocator: A,
    /// Freed `MMAP` buffers the guest still maps.
    retired: RetiredBuffers,
    /// The one session allowed to hold buffers: a camera cannot be shared, and
    /// `v4l2-compliance` checks that a second session is refused (see `SimpleCaptureDevice`).
    active_session: Option<u32>,
}

impl<B, Q, M, HM, A> CameraDevice<B, Q, M, HM, A>
where
    B: CameraBackend,
    Q: VirtioMediaEventQueue,
    M: VirtioMediaGuestMemoryMapper,
    HM: VirtioMediaHostMemoryMapper,
    A: VirtioMediaBufferAllocator,
{
    pub fn new(backend: B, evt_queue: Q, mem: M, mapper: HM, allocator: A) -> Self {
        Self {
            backend,
            evt_queue,
            mem,
            mmap_manager: MmapMappingManager::from(mapper),
            allocator,
            retired: RetiredBuffers::new(),
            active_session: None,
        }
    }

    pub fn info(&self) -> &CameraInfo {
        self.backend.info()
    }

    /// Close the stream, if one is open, and wait for its thread. Every buffer it was lent is
    /// still queued afterwards; the caller decides what becomes of them.
    fn stop_stream(&mut self, session: &mut CameraSession<M::GuestMemoryMapping, B::Stream>) {
        if let Some(stream) = session.stream.take() {
            stream.close();
        }
    }

    /// The stream is over, one way or another: join it, give every buffer back to the
    /// not-queued state, and leave the session dead.
    fn end_session(
        &mut self,
        session: &mut CameraSession<M::GuestMemoryMapping, B::Stream>,
        why: &str,
    ) {
        log::error!(
            "camera {}: session {} ends: {}",
            self.backend.info().id,
            session.id,
            why
        );
        self.stop_stream(session);
        session.queued.clear();
        for buffer in session.buffers.iter_mut() {
            buffer.unqueue();
        }
        session.dead = true;
        self.evt_queue.send_error(session.id, libc::ENODEV);
    }

    /// Drop every buffer, returning host buffers to the allocator (or holding them until the
    /// guest unmaps them) and releasing guest mappings. The stream must be closed already.
    fn free_buffers(&mut self, session: &mut CameraSession<M::GuestMemoryMapping, B::Stream>) {
        debug_assert!(
            session.stream.is_none(),
            "buffers freed under a live stream"
        );
        session.queued.clear();
        for buffer in session.buffers.drain(..) {
            if let Backing::Host { buffer, offset } = buffer.backing {
                self.retired
                    .retire(&mut self.mmap_manager, &mut self.allocator, offset, buffer);
            }
        }
    }

    /// Append `count` buffers of `sizeimage` bytes. All or nothing.
    fn add_buffers(
        &mut self,
        session: &mut CameraSession<M::GuestMemoryMapping, B::Stream>,
        memory: MemoryType,
        count: usize,
        sizeimage: u32,
    ) -> IoctlResult<()> {
        let first = session.buffers.len();
        let mut added: Vec<Buffer<M::GuestMemoryMapping>> = Vec::with_capacity(count);

        for index in first..first + count {
            let mut v4l2_buffer = V4l2Buffer::new(QUEUE, index as u32, memory);
            v4l2_buffer.set_field(BufferField::None);
            v4l2_buffer.set_flags(BufferFlags::TIMESTAMP_MONOTONIC);

            let backing = match memory {
                MemoryType::Mmap => {
                    let host_buffer = match self.allocator.allocate(sizeimage as u64) {
                        Ok(b) => b,
                        Err(e) => {
                            self.undo_added(added);
                            return Err(e);
                        }
                    };
                    let offset = match self.mmap_manager.register_buffer(None, sizeimage) {
                        Ok(offset) => offset,
                        Err(e) => {
                            log::error!("failed to register MMAP buffer: {:#}", e);
                            self.allocator.release(host_buffer);
                            self.undo_added(added);
                            return Err(libc::EINVAL);
                        }
                    };
                    if let V4l2PlanesWithBackingMut::Mmap(mut planes) =
                        v4l2_buffer.planes_with_backing_iter_mut()
                    {
                        // SAFETY: every buffer has at least one plane.
                        let mut plane = planes.next().unwrap();
                        plane.set_mem_offset(offset);
                        *plane.length = sizeimage;
                    }
                    Backing::Host {
                        buffer: host_buffer,
                        offset,
                    }
                }
                MemoryType::UserPtr => {
                    *v4l2_buffer.get_first_plane_mut().length = sizeimage;
                    Backing::Guest(None)
                }
                _ => {
                    self.undo_added(added);
                    return Err(libc::EINVAL);
                }
            };

            added.push(Buffer {
                v4l2_buffer,
                backing,
                queued: false,
                lent: false,
                size: sizeimage,
                prepared: None,
            });
        }

        session.buffers.extend(added);
        Ok(())
    }

    /// Give back what `add_buffers` allocated before it failed.
    fn undo_added(&mut self, added: Vec<Buffer<M::GuestMemoryMapping>>) {
        for buffer in added {
            if let Backing::Host { buffer, offset } = buffer.backing {
                self.mmap_manager.unregister_buffer(offset);
                self.allocator.release(buffer);
            }
        }
    }

    /// The size this device would use for `format` on `queue`: a zero dimension is "whatever
    /// you have", anything else is matched to the nearest size the camera offers. The pixel
    /// format is always NV12.
    fn adjust_size(
        &self,
        session: &CameraSession<M::GuestMemoryMapping, B::Stream>,
        queue: QueueType,
        format: &v4l2_format,
    ) -> IoctlResult<FrameSize> {
        if queue != QUEUE {
            return Err(libc::EINVAL);
        }
        // SAFETY: the queue is multi-planar, so `pix_mp` is the live member.
        let pix_mp = unsafe { format.fmt.pix_mp };
        let (width, height) = (pix_mp.width, pix_mp.height);
        if width == 0 || height == 0 {
            return Ok(session.size);
        }
        self.backend
            .info()
            .nearest_size(width, height)
            .ok_or(libc::ENODEV)
    }

    /// Lend the queued buffer `index` to the open stream.
    fn lend(
        session: &mut CameraSession<M::GuestMemoryMapping, B::Stream>,
        index: usize,
    ) -> IoctlResult<()> {
        let len = session.sizeimage() as usize;
        let stride = session.size.width;
        let stream = session.stream.as_mut().ok_or(libc::EIO)?;
        let entry = session.buffers.get_mut(index).ok_or(libc::EINVAL)?;
        // The loan is `len` bytes wide and the capture thread writes all of them, so whatever
        // backs the buffer must be at least that long. `qbuf` has already refused a `length`
        // that could not hold a frame; this is the mapping's own account of itself, the last
        // thing between a mis-sized one and a `copy_nonoverlapping` (review-m4 R2).
        if entry.data_len().is_some_and(|have| have < len) {
            return Err(libc::EINVAL);
        }
        let ptr = entry.data_mut_ptr().ok_or(libc::EIO)?;
        stream.give_empty(EmptyBuffer {
            index: index as u32,
            ptr: SendPtr(ptr),
            len,
            stride,
        })?;
        entry.lent = true;
        Ok(())
    }

    /// A filled buffer back from the stream: finish its V4L2 description and tell the guest.
    fn return_filled(
        &mut self,
        session: &mut CameraSession<M::GuestMemoryMapping, B::Stream>,
        filled: FilledBuffer,
    ) {
        let Some(entry) = session.buffers.get_mut(filled.index as usize) else {
            log::error!(
                "camera {}: the stream returned buffer {}, which does not exist",
                self.backend.info().id,
                filled.index
            );
            return;
        };
        if !entry.lent {
            log::error!(
                "camera {}: the stream returned buffer {}, which it was not lent",
                self.backend.info().id,
                filled.index
            );
            return;
        }
        // The stream is done with the bytes: a shadowed guest mapping is written back here, and
        // the guest must see the frame before it is told the buffer is done.
        entry.unqueue();
        let plane = entry.v4l2_buffer.get_first_plane_mut();
        *plane.bytesused = filled.bytesused.min(entry.size);
        entry.v4l2_buffer.set_sequence(filled.sequence);
        let ns = filled.timestamp_ns.max(0);
        entry.v4l2_buffer.set_timestamp(bindings::timeval {
            tv_sec: (ns / 1_000_000_000) as bindings::time_t,
            tv_usec: ((ns % 1_000_000_000) / 1_000) as bindings::time_t,
        });
        let event = entry.v4l2_buffer.clone();
        self.evt_queue
            .send_event(V4l2Event::DequeueBuffer(DequeueBufferEvent::new(
                session.id, event,
            )));
    }
}

impl<B, Q, M, HM, A, Reader, Writer> VirtioMediaDevice<Reader, Writer>
    for CameraDevice<B, Q, M, HM, A>
where
    B: CameraBackend,
    Q: VirtioMediaEventQueue,
    M: VirtioMediaGuestMemoryMapper,
    HM: VirtioMediaHostMemoryMapper,
    A: VirtioMediaBufferAllocator,
    Reader: ReadFromDescriptorChain,
    Writer: WriteToDescriptorChain,
{
    type Session = CameraSession<M::GuestMemoryMapping, B::Stream>;

    fn new_session(&mut self, session_id: u32) -> Result<Self::Session, i32> {
        let info = self.backend.info();
        let size = info.default_size().ok_or_else(|| {
            log::error!("camera {}: no output size at all", info.id);
            libc::ENODEV
        })?;
        Ok(CameraSession {
            id: session_id,
            signal: Arc::new(FrameSignal::new()?),
            size,
            fps: DEFAULT_FPS,
            memory: None,
            stream: None,
            buffers: Vec::new(),
            queued: VecDeque::new(),
            dead: false,
        })
    }

    fn close_session(&mut self, mut session: Self::Session) {
        if self.active_session == Some(session.id) {
            self.active_session = None;
        }
        // The stream first, so no thread is writing into a buffer that goes away below.
        self.stop_stream(&mut session);
        self.free_buffers(&mut session);
    }

    fn do_ioctl(
        &mut self,
        session: &mut Self::Session,
        ioctl: V4l2Ioctl,
        reader: &mut Reader,
        writer: &mut Writer,
    ) -> IoResult<()> {
        virtio_media_dispatch_ioctl(self, session, ioctl, reader, writer)
    }

    fn do_mmap(
        &mut self,
        session: &mut Self::Session,
        flags: u32,
        offset: u32,
    ) -> Result<(u64, u64), i32> {
        let host_buffer = session
            .buffers
            .iter()
            .find_map(|b| match &b.backing {
                Backing::Host { buffer, offset: o } if *o == offset => Some(buffer),
                _ => None,
            })
            .ok_or(libc::EINVAL)?;
        let rw = (flags & VIRTIO_MEDIA_MMAP_FLAG_RW) != 0;

        self.mmap_manager
            .create_mapping(offset, host_buffer, rw)
            .map_err(|e| {
                log::error!("failed to map MMAP buffer at offset {:#x}: {:#}", offset, e);
                libc::EINVAL
            })
    }

    fn do_munmap(&mut self, guest_addr: u64) -> Result<(), i32> {
        let res = self
            .mmap_manager
            .remove_mapping(guest_addr)
            .map(|_| ())
            .map_err(|_| libc::EINVAL);
        self.retired.reap(&self.mmap_manager, &mut self.allocator);
        res
    }

    /// The session's eventfd is readable: collect what the stream has produced. Frames first,
    /// then events, so a frame that arrived before the camera died is still delivered.
    fn process_events(&mut self, session: &mut Self::Session) -> Result<(), i32> {
        session.signal.drain();
        let (filled, events) = match session.stream.as_mut() {
            Some(stream) => (stream.take_filled(), stream.take_events()),
            // A bump from a stream that has since been closed; nothing to collect.
            None => return Ok(()),
        };
        for buffer in filled {
            self.return_filled(session, buffer);
        }
        if let Some(event) = events.into_iter().next() {
            let why = match event {
                CameraEvent::Disconnected => "the camera was disconnected".to_owned(),
                CameraEvent::Error(reason) => reason,
            };
            self.end_session(session, &why);
        }
        Ok(())
    }
}

/// The format as a single-plane multi-planar `v4l2_format`, `sizeimage` bytes per buffer
/// (`CREATE_BUFS` may ask for more than a frame needs).
///
/// The colorimetry is what an Android camera's `YUV_420_888` output is (dataspace JFIF): sRGB
/// primaries, BT.601 encoding, full range.
fn to_v4l2_sized(size: FrameSize, sizeimage: u32) -> v4l2_format {
    let mut pix_mp = bindings::v4l2_pix_format_mplane {
        width: size.width,
        height: size.height,
        pixelformat: NV12.to_u32(),
        field: bindings::v4l2_field_V4L2_FIELD_NONE,
        colorspace: bindings::v4l2_colorspace_V4L2_COLORSPACE_SRGB,
        num_planes: 1,
        ..Default::default()
    };
    pix_mp.__bindgen_anon_1.ycbcr_enc = bindings::v4l2_ycbcr_encoding_V4L2_YCBCR_ENC_601 as u8;
    pix_mp.quantization = bindings::v4l2_quantization_V4L2_QUANTIZATION_FULL_RANGE as u8;
    pix_mp.xfer_func = bindings::v4l2_xfer_func_V4L2_XFER_FUNC_SRGB as u8;
    pix_mp.plane_fmt[0] = bindings::v4l2_plane_pix_format {
        sizeimage,
        bytesperline: size.width,
        ..Default::default()
    };

    v4l2_format {
        type_: QUEUE as u32,
        fmt: bindings::v4l2_format__bindgen_ty_1 { pix_mp },
    }
}

fn to_v4l2(size: FrameSize) -> v4l2_format {
    to_v4l2_sized(size, size.sizeimage())
}

/// Whether `queue` is a buffer type `G_PARM` and `S_PARM` answer for.
///
/// The frame rate belongs to the camera, not to a queue's plane layout, so both capture types
/// name the same thing here. A vb2 driver behaves the same way: the V4L2 core's `check_fmt()`
/// lets `V4L2_BUF_TYPE_VIDEO_CAPTURE` reach a driver that only implements
/// `vidioc_g_fmt_vid_cap_mplane`, and nothing under it looks at the type again. It matters
/// because v4l-utils 1.32.0 hardcodes the single-planar type in `v4l2-ctl --get-parm` and
/// `--set-parm`, so with only the mplane type accepted both failed with `EINVAL` while
/// `v4l2-compliance`, which sends the mplane type, passed (D16).
fn is_parm_queue(queue: QueueType) -> bool {
    matches!(queue, QUEUE | QueueType::VideoCapture)
}

/// `G_PARM`'s answer for a session running at `fps`, in the buffer type the caller asked with.
fn streamparm(queue: QueueType, fps: u32) -> v4l2_streamparm {
    v4l2_streamparm {
        type_: queue as u32,
        parm: bindings::v4l2_streamparm__bindgen_ty_1 {
            capture: bindings::v4l2_captureparm {
                capability: bindings::V4L2_CAP_TIMEPERFRAME,
                capturemode: 0,
                timeperframe: bindings::v4l2_fract {
                    numerator: 1,
                    denominator: fps.max(1),
                },
                extendedmode: 0,
                readbuffers: 0,
                reserved: [0; 4],
            },
        },
    }
}

impl<B, Q, M, HM, A> VirtioMediaIoctlHandler for CameraDevice<B, Q, M, HM, A>
where
    B: CameraBackend,
    Q: VirtioMediaEventQueue,
    M: VirtioMediaGuestMemoryMapper,
    HM: VirtioMediaHostMemoryMapper,
    A: VirtioMediaBufferAllocator,
{
    type Session = CameraSession<M::GuestMemoryMapping, B::Stream>;

    fn enum_fmt(
        &mut self,
        _session: &Self::Session,
        queue: QueueType,
        index: u32,
    ) -> IoctlResult<v4l2_fmtdesc> {
        if queue != QUEUE || index != 0 {
            return Err(libc::EINVAL);
        }
        let mut desc = v4l2_fmtdesc {
            index,
            type_: queue as u32,
            pixelformat: NV12.to_u32(),
            ..Default::default()
        };
        let description = b"Y/UV 4:2:0";
        desc.description[..description.len()].copy_from_slice(description);
        Ok(desc)
    }

    fn enum_framesizes(
        &mut self,
        _session: &Self::Session,
        index: u32,
        pixel_format: u32,
    ) -> IoctlResult<v4l2_frmsizeenum> {
        if pixel_format != NV12.to_u32() {
            return Err(libc::EINVAL);
        }
        let size = self
            .backend
            .info()
            .sizes
            .get(index as usize)
            .ok_or(libc::EINVAL)?;
        Ok(v4l2_frmsizeenum {
            index,
            pixel_format,
            type_: bindings::v4l2_frmsizetypes_V4L2_FRMSIZE_TYPE_DISCRETE,
            __bindgen_anon_1: bindings::v4l2_frmsizeenum__bindgen_ty_1 {
                discrete: bindings::v4l2_frmsize_discrete {
                    width: size.width,
                    height: size.height,
                },
            },
            ..Default::default()
        })
    }

    fn enum_frameintervals(
        &mut self,
        _session: &Self::Session,
        index: u32,
        pixel_format: u32,
        width: u32,
        height: u32,
    ) -> IoctlResult<v4l2_frmivalenum> {
        if pixel_format != NV12.to_u32() {
            return Err(libc::EINVAL);
        }
        let info = self.backend.info();
        let size = info
            .sizes
            .iter()
            .find(|s| s.width == width && s.height == height)
            .ok_or(libc::EINVAL)?;
        let fps = *info
            .frame_rates(size)
            .get(index as usize)
            .ok_or(libc::EINVAL)?;
        Ok(v4l2_frmivalenum {
            index,
            pixel_format,
            width,
            height,
            type_: bindings::v4l2_frmivaltypes_V4L2_FRMIVAL_TYPE_DISCRETE,
            __bindgen_anon_1: bindings::v4l2_frmivalenum__bindgen_ty_1 {
                discrete: bindings::v4l2_fract {
                    numerator: 1,
                    denominator: fps,
                },
            },
            ..Default::default()
        })
    }

    fn g_fmt(&mut self, session: &Self::Session, queue: QueueType) -> IoctlResult<v4l2_format> {
        if queue != QUEUE {
            return Err(libc::EINVAL);
        }
        Ok(to_v4l2(session.size))
    }

    fn try_fmt(
        &mut self,
        session: &Self::Session,
        queue: QueueType,
        format: v4l2_format,
    ) -> IoctlResult<v4l2_format> {
        Ok(to_v4l2(self.adjust_size(session, queue, &format)?))
    }

    fn s_fmt(
        &mut self,
        session: &mut Self::Session,
        queue: QueueType,
        format: v4l2_format,
    ) -> IoctlResult<v4l2_format> {
        // The camera is gone; see `reqbufs` (review-m4 R12).
        if session.dead {
            return Err(libc::ENODEV);
        }
        let size = self.adjust_size(session, queue, &format)?;
        // Buffers were sized for the old format, and a stream is running at it.
        if !session.buffers.is_empty() || session.stream.is_some() {
            return Err(libc::EBUSY);
        }
        session.size = size;
        Ok(to_v4l2(size))
    }

    fn reqbufs(
        &mut self,
        session: &mut Self::Session,
        queue: QueueType,
        memory: MemoryType,
        count: u32,
    ) -> IoctlResult<v4l2_requestbuffers> {
        if queue != QUEUE {
            return Err(libc::EINVAL);
        }
        if !matches!(memory, MemoryType::Mmap | MemoryType::UserPtr) {
            return Err(libc::EINVAL);
        }
        // Nothing that takes or holds resources for a session the camera has already left:
        // `end_session` has told the guest the session is over, and the fork driver answers
        // `ENODEV` itself from then on. Without this a guest that ignores the error event could
        // still REQBUFS(32) and sit on a queue's worth of `media_host` (review-m4 R12).
        if session.dead {
            return Err(libc::ENODEV);
        }
        match self.active_session {
            Some(id) if id != session.id => return Err(libc::EBUSY),
            _ => (),
        }
        // `REQBUFS(0)` is an implicit `STREAMOFF`; a reallocation under a running stream is
        // refused, as `vb2` does.
        if count == 0 {
            self.streamoff(session, queue)?;
        } else if session.stream.is_some() {
            return Err(libc::EBUSY);
        }

        // Old buffers go first, mappings and all, so the reply never races a stale view. The
        // stream is closed by now (`streamoff` above, or there was none), so nothing writes into
        // them.
        self.free_buffers(session);
        let count = (count as usize).min(MAX_BUFFERS);
        session.memory = None;
        self.active_session = None;
        if count > 0 {
            let sizeimage = session.sizeimage();
            self.add_buffers(session, memory, count, sizeimage)?;
            session.memory = Some(memory);
            self.active_session = Some(session.id);
        }

        Ok(v4l2_requestbuffers {
            count: count as u32,
            type_: queue as u32,
            memory: memory as u32,
            capabilities: (BufferCapabilities::SUPPORTS_MMAP
                | BufferCapabilities::SUPPORTS_USERPTR
                | BufferCapabilities::SUPPORTS_ORPHANED_BUFS)
                .bits(),
            ..Default::default()
        })
    }

    fn create_bufs(
        &mut self,
        session: &mut Self::Session,
        count: u32,
        queue: QueueType,
        memory: MemoryType,
        format: v4l2_format,
    ) -> IoctlResult<v4l2_create_buffers> {
        if queue != QUEUE {
            return Err(libc::EINVAL);
        }
        if !matches!(memory, MemoryType::Mmap | MemoryType::UserPtr) {
            return Err(libc::EINVAL);
        }
        // The camera is gone; see `reqbufs` (review-m4 R12).
        if session.dead {
            return Err(libc::ENODEV);
        }
        // `CREATE_BUFS(count = 0)` is V4L2's capability probe, and vb2 answers it without ever
        // looking at the format: `vb2_ioctl_create_bufs` verifies the memory and buffer types,
        // fills in the index the next buffer would take and the queue's `V4L2_BUF_CAP_*` word,
        // and returns -- "If count == 0, then just check if memory and type are valid",
        // videobuf2-v4l2.c:1054-1059, and `vb2_create_bufs` returns at :757-758 before the
        // switch that reads `num_planes` and `sizeimage` (GKI 6.18). It takes no resources, so
        // it does not contend for the queue either: vb2's owner check (`vb2_queue_is_busy`) sits
        // after that return, and so does this one. `v4l2-ctl --stream-mmap` sends exactly this
        // probe -- a zeroed format -- before every stream, and warned on each one while it was
        // answered `EINVAL` (D19). The format goes back untouched, as the kernel leaves it.
        if count == 0 {
            return Ok(v4l2_create_buffers {
                index: session.buffers.len() as u32,
                count: 0,
                memory: memory as u32,
                format,
                capabilities: (BufferCapabilities::SUPPORTS_MMAP
                    | BufferCapabilities::SUPPORTS_USERPTR
                    | BufferCapabilities::SUPPORTS_ORPHANED_BUFS)
                    .bits(),
                ..Default::default()
            });
        }
        match self.active_session {
            Some(id) if id != session.id => return Err(libc::EBUSY),
            _ => (),
        }
        let wanted = self.adjust_size(session, queue, &format)?;
        // `CREATE_BUFS` is the one call where the guest sizes the buffers itself, so the format
        // it hands over is checked rather than adjusted: `EINVAL` for a plane count the format
        // does not have, for a `sizeimage` too small for the requested frame, and for one too
        // small for the queue's own format (D9; the loopback device says why). While buffers
        // exist a set in another geometry may not join them.
        // SAFETY: the queue is multi-planar, so `pix_mp` is the live member.
        let asked_mp = unsafe { format.fmt.pix_mp };
        if asked_mp.num_planes != 1 {
            return Err(libc::EINVAL);
        }
        let asked = asked_mp.plane_fmt[0].sizeimage;
        if asked < wanted.sizeimage() || asked < session.sizeimage() {
            return Err(libc::EINVAL);
        }
        if !session.buffers.is_empty()
            && (wanted.width != session.size.width || wanted.height != session.size.height)
        {
            return Err(libc::EINVAL);
        }
        // One memory type per queue.
        if let Some(existing) = session.memory {
            if existing != memory {
                return Err(libc::EINVAL);
            }
        }

        let first = session.buffers.len();
        let count = (count as usize).min(MAX_BUFFERS.saturating_sub(first));
        if count > 0 {
            self.add_buffers(session, memory, count, asked)?;
            session.memory = Some(memory);
            self.active_session = Some(session.id);
        }

        Ok(v4l2_create_buffers {
            index: first as u32,
            count: count as u32,
            memory: memory as u32,
            format: to_v4l2_sized(wanted, asked),
            capabilities: (BufferCapabilities::SUPPORTS_MMAP
                | BufferCapabilities::SUPPORTS_USERPTR
                | BufferCapabilities::SUPPORTS_ORPHANED_BUFS)
                .bits(),
            ..Default::default()
        })
    }

    fn querybuf(
        &mut self,
        session: &Self::Session,
        queue: QueueType,
        index: u32,
    ) -> IoctlResult<V4l2Buffer> {
        if queue != QUEUE {
            return Err(libc::EINVAL);
        }
        let buffer = session.buffers.get(index as usize).ok_or(libc::EINVAL)?;
        Ok(buffer.v4l2_buffer.clone())
    }

    fn qbuf(
        &mut self,
        session: &mut Self::Session,
        buffer: V4l2Buffer,
        guest_regions: Vec<Vec<SgEntry>>,
        payload_valid: bool,
    ) -> IoctlResult<V4l2Buffer> {
        if buffer.queue() != QUEUE {
            return Err(libc::EINVAL);
        }
        if session.dead {
            return Err(libc::ENODEV);
        }
        let sizeimage = session.sizeimage();
        let entry = session
            .buffers
            .get_mut(buffer.index() as usize)
            .ok_or(libc::EINVAL)?;
        if entry.queued || Some(buffer.memory()) != session.memory {
            return Err(libc::EINVAL);
        }
        // A prepared buffer keeps the payload description `PREPARE_BUF` accepted, and V4L2 says
        // this call's own `bytesused` / `data_offset` are ignored.
        let prepared = entry.prepared;
        if prepared.is_none() && !payload_valid {
            return Err(libc::EINVAL);
        }

        // A guest-supplied MPLANE buffer may carry no plane at all; the first plane is asked
        // for, never assumed (this VMM aborts on panic).
        let guest_plane = buffer.planes_iter().next().ok_or(libc::EINVAL)?;
        let guest_length = match prepared {
            Some((_, length)) => length,
            None => *guest_plane.length,
        };

        match &mut entry.backing {
            Backing::Host { .. } => {
                *entry.v4l2_buffer.get_first_plane_mut().bytesused = 0;
            }
            Backing::Guest(slot) => {
                // `length` is the guest's number and sizes the mapping; the stream writes a
                // whole frame into it, so it must hold one, and no more than the buffer was
                // allocated for.
                //
                // Checked on *this* call's plane whatever `PREPARE_BUF` accepted earlier, and
                // not only on `guest_length`: the SG list about to be mapped was read against
                // this call's `length` (`ioctl::get_userptr_regions`), so that is the number
                // that decides how much guest memory the loan really covers. Trusting the
                // prepared one let a guest prepare a full-sized buffer and then queue an
                // 8-byte scatter list, over which the capture thread wrote a whole frame
                // (review-m4 R2). V4L2 is right that `QBUF` ignores a prepared buffer's
                // *payload*; the scatter list is not payload, it is the mapping.
                if *guest_plane.length < sizeimage || *guest_plane.length > entry.size {
                    return Err(libc::EINVAL);
                }
                if guest_length < sizeimage || guest_length > entry.size {
                    return Err(libc::EINVAL);
                }
                let sgs = guest_regions.into_iter().next().ok_or(libc::EINVAL)?;
                // CAPTURE: the stream fills the guest's pages, so the mapping is writable.
                let mapping = self.mem.new_mapping_for(sgs, true).map_err(|e| {
                    log::error!("failed to map USERPTR buffer: {:#}", e);
                    guest_mapping_errno(&e)
                })?;
                *slot = Some(mapping);
                if prepared.is_none() {
                    // The guest's view of its own buffer -- userptr and length -- is what must
                    // be echoed back in the dequeue event.
                    let mut v4l2_buffer = buffer.clone();
                    v4l2_buffer.set_field(BufferField::None);
                    v4l2_buffer.set_flags(BufferFlags::TIMESTAMP_MONOTONIC);
                    *v4l2_buffer.get_first_plane_mut().bytesused = 0;
                    entry.v4l2_buffer = v4l2_buffer;
                }
            }
        }

        entry.queued = true;
        entry.prepared = None;
        entry
            .v4l2_buffer
            .clear_flags(BufferFlags::PREPARED | BufferFlags::LAST | BufferFlags::DONE);
        entry.v4l2_buffer.add_flags(BufferFlags::QUEUED);
        let reply = entry.v4l2_buffer.clone();

        let index = buffer.index() as usize;
        if session.stream.is_some() {
            if let Err(e) = Self::lend(session, index) {
                // The stream would not take it: the buffer stays queued for the guest's sake,
                // but the stream is not going to fill it, which is the end of it.
                self.end_session(session, &format!("the stream refused a buffer: errno {e}"));
                return Err(libc::EIO);
            }
        } else {
            session.queued.push_back(index);
        }

        Ok(reply)
    }

    /// `VIDIOC_PREPARE_BUF`: everything `QBUF` validates, minus the queueing; the loopback
    /// device explains the rules.
    fn prepare_buf(
        &mut self,
        session: &mut Self::Session,
        buffer: V4l2Buffer,
        _guest_regions: Vec<Vec<SgEntry>>,
        payload_valid: bool,
    ) -> IoctlResult<V4l2Buffer> {
        if !payload_valid {
            return Err(libc::EINVAL);
        }
        if buffer.queue() != QUEUE {
            return Err(libc::EINVAL);
        }
        // The camera is gone; see `reqbufs` (review-m4 R12).
        if session.dead {
            return Err(libc::ENODEV);
        }
        let sizeimage = session.sizeimage();
        let entry = session
            .buffers
            .get_mut(buffer.index() as usize)
            .ok_or(libc::EINVAL)?;
        if entry.queued || entry.prepared.is_some() || Some(buffer.memory()) != session.memory {
            return Err(libc::EINVAL);
        }

        let guest_plane = buffer.planes_iter().next().ok_or(libc::EINVAL)?;
        let guest_length = *guest_plane.length;

        if let Backing::Guest(_) = &entry.backing {
            if guest_length < sizeimage || guest_length > entry.size {
                return Err(libc::EINVAL);
            }
            let mut v4l2_buffer = buffer.clone();
            v4l2_buffer.set_field(BufferField::None);
            v4l2_buffer.set_flags(BufferFlags::TIMESTAMP_MONOTONIC);
            entry.v4l2_buffer = v4l2_buffer;
        }

        // A prepared buffer is neither queued nor done, and carries no timestamp or sequence
        // yet; on a capture queue it has no payload either.
        entry.v4l2_buffer.set_timestamp(Default::default());
        entry.v4l2_buffer.set_sequence(0);
        entry
            .v4l2_buffer
            .clear_flags(BufferFlags::QUEUED | BufferFlags::DONE | BufferFlags::LAST);
        *entry.v4l2_buffer.get_first_plane_mut().bytesused = 0;
        entry.v4l2_buffer.add_flags(BufferFlags::PREPARED);
        entry.prepared = Some((0, guest_length));

        Ok(entry.v4l2_buffer.clone())
    }

    /// Open the camera and lend it every buffer the guest has queued.
    fn streamon(&mut self, session: &mut Self::Session, queue: QueueType) -> IoctlResult<()> {
        if queue != QUEUE {
            return Err(libc::EINVAL);
        }
        if session.dead {
            return Err(libc::ENODEV);
        }
        if session.buffers.is_empty() {
            return Err(libc::EINVAL);
        }
        if session.stream.is_some() {
            // Already streaming: V4L2 says this is not an error.
            return Ok(());
        }

        let info = self.backend.info();
        let id = info.id.clone();
        let request = StreamRequest {
            width: session.size.width,
            height: session.size.height,
            fps: info.range_at(&session.size, session.fps),
            buffers: session.buffers.len() as u32,
        };
        let sink = CaptureSink(Arc::clone(&session.signal));
        let stream = self.backend.open_stream(request, sink).map_err(|e| {
            log::error!(
                "camera {}: cannot open a {}x{} stream at {:?} fps: errno {}",
                id,
                request.width,
                request.height,
                request.fps,
                e
            );
            e
        })?;
        session.stream = Some(stream);

        // Over a copy, so that `session.queued` is left exactly as the guest built it if one of
        // the loans is refused: rebuilding it from the buffer list would hand a guest that
        // queued 1, 0, 2 a retry in index order, and a different DQBUF order with it
        // (review-m4 R13).
        let order: Vec<usize> = session.queued.iter().copied().collect();
        for index in order {
            if let Err(e) = Self::lend(session, index) {
                // Undo: the stream is closed and every buffer stays queued but not lent, as
                // before the call; the guest may try again.
                log::error!(
                    "camera {}: the stream refused buffer {} at STREAMON: errno {}",
                    self.backend.info().id,
                    index,
                    e
                );
                self.stop_stream(session);
                for buffer in session.buffers.iter_mut() {
                    buffer.lent = false;
                }
                return Err(libc::EIO);
            }
        }
        session.queued.clear();
        Ok(())
    }

    /// Close the camera, then give every buffer back to the not-queued state. The order is the
    /// §2.5 invariant: no buffer is unqueued -- and no guest mapping dropped -- while the stream
    /// thread may still write into it.
    fn streamoff(&mut self, session: &mut Self::Session, queue: QueueType) -> IoctlResult<()> {
        if queue != QUEUE {
            return Err(libc::EINVAL);
        }
        self.stop_stream(session);
        session.queued.clear();
        for buffer in session.buffers.iter_mut() {
            buffer.unqueue();
        }
        Ok(())
    }

    fn g_parm(
        &mut self,
        session: &Self::Session,
        queue: QueueType,
    ) -> IoctlResult<v4l2_streamparm> {
        if !is_parm_queue(queue) {
            return Err(libc::EINVAL);
        }
        let (_, max) = self.backend.info().range_at(&session.size, session.fps);
        Ok(streamparm(queue, max))
    }

    /// A single `timeperframe` selects, by the design's rule, the widest fps range whose maximum
    /// is that rate; the reply carries the rate actually chosen. A fraction that is not a rate
    /// (`0/0`, `1/0`, `0/1` -- `v4l2-compliance` sends all three and expects success) asks for
    /// the default. While streaming the range is applied to the running stream.
    fn s_parm(
        &mut self,
        session: &mut Self::Session,
        parm: v4l2_streamparm,
    ) -> IoctlResult<v4l2_streamparm> {
        let queue = QueueType::n(parm.type_)
            .filter(|&queue| is_parm_queue(queue))
            .ok_or(libc::EINVAL)?;
        // The camera is gone; see `reqbufs` (review-m4 R12).
        if session.dead {
            return Err(libc::ENODEV);
        }
        // SAFETY: the type says capture, so `capture` is the live member.
        let asked = unsafe { parm.parm.capture.timeperframe };
        let fps = if asked.numerator == 0 || asked.denominator == 0 {
            DEFAULT_FPS
        } else {
            // Round to the nearest whole rate; anything under one frame a second is one.
            ((asked.denominator as u64 + asked.numerator as u64 / 2) / asked.numerator as u64)
                .clamp(1, u32::MAX as u64) as u32
        };
        let range = self.backend.info().range_at(&session.size, fps);
        if let Some(stream) = session.stream.as_mut() {
            stream.set_controls(&[CameraControl::FpsRange(range.0, range.1)])?;
        }
        session.fps = fps;
        Ok(streamparm(queue, range.1))
    }

    fn enuminput(
        &mut self,
        _session: &Self::Session,
        index: u32,
    ) -> IoctlResult<bindings::v4l2_input> {
        if index != 0 {
            return Err(libc::EINVAL);
        }
        let mut input = bindings::v4l2_input {
            index: 0,
            type_: bindings::V4L2_INPUT_TYPE_CAMERA,
            // SAFETY: an all-zero `v4l2_input` is a valid (if empty) value of a plain C struct.
            ..unsafe { std::mem::zeroed() }
        };
        let name = self.backend.info().name.as_bytes();
        let n = name.len().min(input.name.len() - 1);
        input.name[..n].copy_from_slice(&name[..n]);
        Ok(input)
    }

    fn g_input(&mut self, _session: &Self::Session) -> IoctlResult<i32> {
        Ok(0)
    }

    fn s_input(&mut self, _session: &mut Self::Session, input: i32) -> IoctlResult<i32> {
        if input != 0 {
            Err(libc::EINVAL)
        } else {
            Ok(0)
        }
    }

    /// `EOS` and `SOURCE_CHANGE` are accepted and never emitted (a camera has neither); the
    /// controls' `V4L2_EVENT_CTRL` is M5.
    fn subscribe_event(
        &mut self,
        _session: &mut Self::Session,
        event: EventType,
        _flags: SubscribeEventFlags,
    ) -> IoctlResult<()> {
        match event {
            EventType::Eos | EventType::SourceChange(0) => Ok(()),
            _ => Err(libc::EINVAL),
        }
    }

    fn unsubscribe_event(
        &mut self,
        _session: &mut Self::Session,
        event: v4l2_event_subscription,
    ) -> IoctlResult<()> {
        if event.type_ == bindings::V4L2_EVENT_ALL {
            return Ok(());
        }
        match EventType::try_from(&event) {
            Ok(EventType::Eos) | Ok(EventType::SourceChange(0)) => Ok(()),
            _ => Err(libc::EINVAL),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::collections::HashSet;
    use std::rc::Rc;
    use std::sync::mpsc;
    use std::sync::Mutex;
    use std::thread;

    use super::*;
    use crate::poll::SessionPoller;
    use crate::protocol::VIRTIO_MEDIA_CMD_OPEN;
    use crate::MemFdAllocator;
    use crate::VirtioMediaDeviceRunner;

    /// Collects the events the device sends.
    #[derive(Default)]
    struct EventLog(Rc<RefCell<Vec<V4l2Event>>>);

    impl VirtioMediaEventQueue for EventLog {
        fn send_event(&mut self, event: V4l2Event) {
            self.0.borrow_mut().push(event);
        }
    }

    /// A pretend guest: one flat byte array standing in for guest-physical memory, from which SG
    /// lists are "mapped" by pointing straight into it. Live mappings are counted so the tests
    /// can check they are released when the design says they must be.
    #[derive(Clone)]
    struct FakeGuest {
        memory: Rc<RefCell<Vec<u8>>>,
        live_mappings: Rc<RefCell<usize>>,
        /// The camera's log, so a mapping can say *when* it was released. See
        /// [`FakeMapping::drop`].
        log: SharedLog,
    }

    struct FakeMapping {
        guest: FakeGuest,
        start: usize,
        len: usize,
    }

    impl GuestMemoryRange for FakeMapping {
        /// What the SG list this mapping was built from adds up to, as a real mapping reports
        /// (`GuestArenaMapping::len`, `GuestShadowMapping::len`).
        fn len(&self) -> usize {
            self.len
        }

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
    }

    impl Drop for FakeMapping {
        /// Dropping a guest mapping is what gives the guest's pages back (a real one unmaps an
        /// arena, or writes a shadow buffer back into it), so it is one of the two events §2.5
        /// orders against the capture thread -- the other being [`OrderedAllocator::release`],
        /// which checks the same thing for host buffers. Releasing one while the fake camera is
        /// still streaming *and* still holds this very buffer is the invariant broken
        /// (review-m4 R1, R7); releasing one the camera has handed back is the ordinary
        /// per-frame case `return_filled` does on purpose.
        fn drop(&mut self) {
            let violated = match self.guest.log.lock() {
                Ok(log) => log.streaming && log.holding.contains(&(self.as_ptr() as usize)),
                Err(_) => false,
            };
            *self.guest.live_mappings.borrow_mut() -= 1;
            // A session holds several mappings and loses them all at once, so the second
            // violation would panic while the first is unwinding -- which aborts the whole test
            // binary instead of failing this one test.
            assert!(
                !violated || std::thread::panicking(),
                "a guest mapping was released while the capture thread still held the buffer"
            );
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

    /// The worker's wait context, as far as [`VirtioMediaDeviceRunner`] can tell: it records the
    /// session descriptors added to it and taken out of it.
    #[derive(Clone, Default)]
    struct FakePoller {
        added: Rc<RefCell<Vec<i32>>>,
        removed: Rc<RefCell<Vec<i32>>>,
    }

    impl SessionPoller for FakePoller {
        fn add_session(&self, session: BorrowedFd, _session_id: u32) -> Result<(), i32> {
            self.added.borrow_mut().push(session.as_raw_fd());
            Ok(())
        }

        fn remove_session(&self, session: BorrowedFd) {
            self.removed.borrow_mut().push(session.as_raw_fd());
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

    /// What the fake camera saw, for the assertions.
    #[derive(Default)]
    struct FakeLog {
        /// `open_stream` calls, with what they asked for.
        opened: Vec<StreamRequest>,
        /// `close` calls that have completed (thread joined).
        closed: usize,
        /// Whether a stream is open right now.
        streaming: bool,
        /// Every buffer index lent, in order, across streams.
        lent: Vec<u32>,
        /// Controls applied to an open stream.
        controls: Vec<CameraControl>,
        /// Host buffers the allocator was handed back while a stream was open -- the §2.5
        /// violation the ordering tests look for.
        released_while_streaming: usize,
        /// The first byte of every buffer the stream has been lent and not yet handed back,
        /// i.e. the memory the capture thread may be writing into right now. Filled in by
        /// `give_empty` on the device thread and emptied by the capture thread just before it
        /// returns the buffer, so it is exact at every point either side can observe.
        holding: HashSet<usize>,
    }

    type SharedLog = Arc<Mutex<FakeLog>>;

    /// A camera producing synthetic frames from a thread of its own, the way a real backend
    /// does: every buffer it is lent is filled at once with a pattern derived from the frame
    /// number and handed back through a channel, with the sink bumped.
    struct FakeCamera {
        info: CameraInfo,
        log: SharedLog,
        /// `open_stream` fails with this errno.
        fail_open: Option<i32>,
        /// The camera "disconnects" after returning this many frames.
        disconnect_after: Option<u32>,
        /// From this frame on the camera keeps every buffer it is lent instead of filling and
        /// returning it, the way a real one holds the buffer it is writing into: the sink is
        /// bumped so a test can tell the thread has taken it, and only the join ends that.
        hold_from: Option<u32>,
        /// The stream refuses the loans after this many (`give_empty` answers `EIO`), which is
        /// what a backend whose capture thread has died looks like.
        refuse_lend_after: Option<usize>,
    }

    enum FakeCommand {
        Lend(EmptyBuffer),
        Stop,
    }

    struct FakeStream {
        commands: mpsc::Sender<FakeCommand>,
        filled: mpsc::Receiver<FilledBuffer>,
        events: mpsc::Receiver<CameraEvent>,
        thread: Option<thread::JoinHandle<()>>,
        log: SharedLog,
        /// Loans accepted so far, against [`FakeCamera::refuse_lend_after`].
        lends: usize,
        refuse_lend_after: Option<usize>,
    }

    /// Luma byte of frame `sequence`, so a filled buffer says which frame it holds.
    fn luma_of(sequence: u32) -> u8 {
        0x10 + sequence as u8
    }

    impl CameraBackend for FakeCamera {
        type Stream = FakeStream;

        fn info(&self) -> &CameraInfo {
            &self.info
        }

        fn open_stream(
            &mut self,
            request: StreamRequest,
            sink: CaptureSink,
        ) -> Result<FakeStream, i32> {
            self.log.lock().unwrap().opened.push(request);
            if let Some(errno) = self.fail_open {
                return Err(errno);
            }
            let (commands, rx) = mpsc::channel();
            let (filled_tx, filled) = mpsc::channel();
            let (events_tx, events) = mpsc::channel();
            let log = Arc::clone(&self.log);
            let thread_log = Arc::clone(&self.log);
            let disconnect_after = self.disconnect_after;
            let hold_from = self.hold_from;
            let (width, height) = (request.width, request.height);
            let thread = thread::spawn(move || {
                let mut sequence = 0u32;
                for command in rx {
                    let buffer = match command {
                        FakeCommand::Lend(buffer) => buffer,
                        FakeCommand::Stop => break,
                    };
                    thread_log.lock().unwrap().lent.push(buffer.index);
                    if disconnect_after == Some(sequence) {
                        let _ = events_tx.send(CameraEvent::Disconnected);
                        sink.signal();
                        break;
                    }
                    if hold_from.is_some_and(|first| sequence >= first) {
                        // Kept, not filled: the buffer stays in `log.holding` until the thread
                        // is joined, which is what makes the ordering assertions bite.
                        sink.signal();
                        continue;
                    }
                    // The copy contract: `height` rows of luma, `height / 2` rows of chroma,
                    // `stride` bytes each, tightly packed.
                    let stride = buffer.stride as usize;
                    assert_eq!(stride, width as usize);
                    assert_eq!(buffer.len, (width * height * 3 / 2) as usize);
                    let dst = buffer.ptr.as_ptr();
                    for row in 0..height as usize {
                        // SAFETY: the buffer is `len` bytes, which the two loops cover exactly.
                        unsafe {
                            std::ptr::write_bytes(dst.add(row * stride), luma_of(sequence), stride)
                        };
                    }
                    for row in 0..height as usize / 2 {
                        // SAFETY: as above.
                        unsafe {
                            std::ptr::write_bytes(
                                dst.add((height as usize + row) * stride),
                                0x80,
                                stride,
                            )
                        };
                    }
                    // Done with the bytes: the device may release what backs them from here
                    // on, and the ordering assertions stop applying to this buffer.
                    thread_log.lock().unwrap().holding.remove(&(dst as usize));
                    let _ = filled_tx.send(FilledBuffer {
                        index: buffer.index,
                        bytesused: buffer.len as u32,
                        timestamp_ns: 1_500_000_000 + sequence as i64 * 33_333_333,
                        sequence,
                    });
                    sequence += 1;
                    sink.signal();
                }
            });
            self.log.lock().unwrap().streaming = true;
            Ok(FakeStream {
                commands,
                filled,
                events,
                thread: Some(thread),
                log,
                lends: 0,
                refuse_lend_after: self.refuse_lend_after,
            })
        }
    }

    impl CameraStream for FakeStream {
        fn give_empty(&mut self, buffer: EmptyBuffer) -> Result<(), i32> {
            if self.refuse_lend_after.is_some_and(|n| self.lends >= n) {
                return Err(libc::EIO);
            }
            self.lends += 1;
            // From here the capture thread owns these bytes until it hands them back.
            self.log
                .lock()
                .unwrap()
                .holding
                .insert(buffer.ptr.as_ptr() as usize);
            self.commands
                .send(FakeCommand::Lend(buffer))
                .map_err(|_| libc::EIO)
        }

        fn take_filled(&mut self) -> Vec<FilledBuffer> {
            self.filled.try_iter().collect()
        }

        fn take_events(&mut self) -> Vec<CameraEvent> {
            self.events.try_iter().collect()
        }

        fn set_controls(&mut self, controls: &[CameraControl]) -> Result<(), i32> {
            self.log
                .lock()
                .unwrap()
                .controls
                .extend_from_slice(controls);
            Ok(())
        }

        fn close(self) {
            // Everything `close` has to do, `Drop` does; see [`FakeStream::stop`].
        }
    }

    impl FakeStream {
        /// Stop the capture thread and join it, marking the stream closed only once the thread
        /// is really gone -- what `AndroidCameraStream` does in *its* `Drop`
        /// (`android_camera_backend/android.rs`). Idempotent.
        fn stop(&mut self) {
            let _ = self.commands.send(FakeCommand::Stop);
            if let Some(thread) = self.thread.take() {
                thread.join().unwrap();
                let mut log = self.log.lock().unwrap();
                log.closed += 1;
                log.streaming = false;
            }
        }
    }

    /// The real backend joins its capture thread when the stream is dropped, not only when
    /// `close` is called, and that is what makes a session's field order matter (review-m4 R1).
    /// The fake would hide the bug without this.
    impl Drop for FakeStream {
        fn drop(&mut self) {
            self.stop();
        }
    }

    /// Counts what the device gives back, and when.
    struct OrderedAllocator {
        inner: MemFdAllocator,
        log: SharedLog,
        released: Rc<RefCell<usize>>,
    }

    impl VirtioMediaBufferAllocator for OrderedAllocator {
        fn allocate(&mut self, len: u64) -> Result<HostBuffer, i32> {
            self.inner.allocate(len)
        }

        fn release(&mut self, buf: HostBuffer) {
            {
                let mut log = self.log.lock().unwrap();
                if log.streaming {
                    log.released_while_streaming += 1;
                }
            }
            *self.released.borrow_mut() += 1;
            self.inner.release(buf);
        }
    }

    type Device = CameraDevice<FakeCamera, EventLog, FakeGuest, FakeHostMapper, OrderedAllocator>;
    type Session = CameraSession<FakeMapping, FakeStream>;

    struct Rig {
        device: Device,
        events: Rc<RefCell<Vec<V4l2Event>>>,
        guest: FakeGuest,
        log: SharedLog,
        released: Rc<RefCell<usize>>,
    }

    const GUEST_MEMORY: usize = 1 << 20;

    /// A camera shaped like a phone's: three sizes, the largest too slow for 60 fps, and the
    /// fps ranges 5566's back camera reports for the most part.
    fn info() -> CameraInfo {
        CameraInfo {
            id: "0".into(),
            name: "Back camera".into(),
            sizes: vec![
                FrameSize {
                    width: 1920,
                    height: 1080,
                    min_frame_duration_ns: Some(33_333_333),
                },
                FrameSize {
                    width: 1280,
                    height: 720,
                    min_frame_duration_ns: Some(16_666_666),
                },
                FrameSize {
                    width: 64,
                    height: 48,
                    min_frame_duration_ns: None,
                },
            ],
            fps_ranges: vec![(15, 15), (7, 30), (15, 30), (30, 30), (24, 24), (60, 60)],
        }
    }

    fn rig_with(camera: FakeCamera) -> Rig {
        let events = EventLog::default();
        let events_log = Rc::clone(&events.0);
        let log = Arc::clone(&camera.log);
        let guest = FakeGuest {
            memory: Rc::new(RefCell::new(vec![0u8; GUEST_MEMORY])),
            live_mappings: Rc::new(RefCell::new(0)),
            log: Arc::clone(&log),
        };
        let released = Rc::new(RefCell::new(0));
        let device = CameraDevice::new(
            camera,
            events,
            guest.clone(),
            FakeHostMapper,
            OrderedAllocator {
                inner: MemFdAllocator::new(),
                log: Arc::clone(&log),
                released: Rc::clone(&released),
            },
        );
        Rig {
            device,
            events: events_log,
            guest,
            log,
            released,
        }
    }

    fn rig() -> Rig {
        rig_with(camera())
    }

    /// A camera that does everything right: opens, fills every buffer it is lent, never goes
    /// away. Tests that want one of the failure modes set the knob they need.
    fn camera() -> FakeCamera {
        FakeCamera {
            info: info(),
            log: Default::default(),
            fail_open: None,
            disconnect_after: None,
            hold_from: None,
            refuse_lend_after: None,
        }
    }

    fn session(device: &mut Device) -> Session {
        <Device as VirtioMediaDevice<&[u8], Vec<u8>>>::new_session(device, 0).unwrap()
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

    /// Wait until `n` DQBUF events have been collected, processing as they arrive.
    fn collect_frames(r: &mut Rig, s: &mut Session, n: usize) -> Vec<V4l2Buffer> {
        while dequeued(&r.events.borrow()).len() < n {
            assert!(wait_ready(s), "no frame within 2s");
            process(&mut r.device, s);
        }
        dequeued(&r.events.borrow())
    }

    fn format(width: u32, height: u32) -> v4l2_format {
        to_v4l2(FrameSize::new(width, height))
    }

    /// The fields of a multi-planar format the tests look at, copied out of the packed struct
    /// so they can be compared by reference.
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

    fn pix_mp(format: &v4l2_format) -> Pix {
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

    fn mmap_buffer(index: u32, len: u32) -> V4l2Buffer {
        let mut buffer = V4l2Buffer::new(QUEUE, index, MemoryType::Mmap);
        *buffer.get_first_plane_mut().length = len;
        buffer
    }

    fn userptr_buffer(index: u32, gpa: u64, len: u32) -> (V4l2Buffer, Vec<Vec<SgEntry>>) {
        userptr_buffer_sized(index, gpa, len, len)
    }

    /// A `USERPTR` buffer whose plane declares `length` bytes while its scatter list covers
    /// `mapped`: two numbers the guest picks separately, which is the point of review-m4 R2.
    /// The ioctl layer reads the list against the declared length, so a real guest can only make
    /// `mapped` *larger*; a device must not depend on that.
    fn userptr_buffer_sized(
        index: u32,
        gpa: u64,
        length: u32,
        mapped: u32,
    ) -> (V4l2Buffer, Vec<Vec<SgEntry>>) {
        let mut buffer = V4l2Buffer::new(QUEUE, index, MemoryType::UserPtr);
        if let V4l2PlanesWithBackingMut::UserPtr(mut planes) = buffer.planes_with_backing_iter_mut()
        {
            let mut plane = planes.next().unwrap();
            plane.set_userptr(0xc000_0000 + index as u64);
            *plane.length = length;
        }
        (buffer, vec![vec![SgEntry::new(gpa, mapped)]])
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

    fn errors(events: &[V4l2Event]) -> usize {
        events
            .iter()
            .filter(|e| matches!(e, V4l2Event::Error(_)))
            .count()
    }

    fn timeperframe(parm: &v4l2_streamparm) -> (u32, u32) {
        // SAFETY: capture.
        let tpf = unsafe { parm.parm.capture.timeperframe };
        (tpf.numerator, tpf.denominator)
    }

    fn parm_for(fps: (u32, u32)) -> v4l2_streamparm {
        let mut parm = streamparm(QUEUE, 1);
        parm.parm.capture.timeperframe = bindings::v4l2_fract {
            numerator: fps.0,
            denominator: fps.1,
        };
        parm
    }

    /// The V4L2 surface: one format, the camera's sizes, the intervals each size sustains, the
    /// nearest size for `TRY_FMT`, and `EBUSY` for `S_FMT` once buffers exist.
    #[test]
    fn formats_are_nv12_over_the_camera_sizes() {
        let mut r = rig();
        let mut s = session(&mut r.device);

        let desc = r.device.enum_fmt(&s, QUEUE, 0).unwrap();
        assert_eq!(desc.pixelformat, NV12.to_u32());
        assert_eq!(r.device.enum_fmt(&s, QUEUE, 1).err(), Some(libc::EINVAL));
        assert_eq!(
            r.device.enum_fmt(&s, QueueType::VideoCapture, 0).err(),
            Some(libc::EINVAL)
        );

        let sizes: Vec<(u32, u32)> = (0..)
            .map_while(|i| r.device.enum_framesizes(&s, i, NV12.to_u32()).ok())
            .map(|f| {
                // SAFETY: discrete.
                let d = unsafe { f.__bindgen_anon_1.discrete };
                (d.width, d.height)
            })
            .collect();
        assert_eq!(sizes, vec![(1920, 1080), (1280, 720), (64, 48)]);
        assert_eq!(
            r.device.enum_framesizes(&s, 0, 0x1234_5678).err(),
            Some(libc::EINVAL)
        );

        let mut intervals = |w, h| -> Vec<u32> {
            (0..)
                .map_while(|i| {
                    r.device
                        .enum_frameintervals(&s, i, NV12.to_u32(), w, h)
                        .ok()
                })
                .map(|f| {
                    // SAFETY: discrete.
                    let d = unsafe { f.__bindgen_anon_1.discrete };
                    assert_eq!(d.numerator, 1);
                    d.denominator
                })
                .collect()
        };
        // 1080p sustains 30 fps: the 60 fps range drops out; 720p takes it.
        assert_eq!(intervals(1920, 1080), vec![30, 24, 15]);
        assert_eq!(intervals(1280, 720), vec![60, 30, 24, 15]);
        // A size that says nothing about its speed offers every range.
        assert_eq!(intervals(64, 48), vec![60, 30, 24, 15]);
        assert_eq!(
            r.device
                .enum_frameintervals(&s, 0, NV12.to_u32(), 640, 480)
                .err(),
            Some(libc::EINVAL)
        );

        // The session starts at 720p, and the pixel format is not negotiable.
        let cur = pix_mp(&r.device.g_fmt(&s, QUEUE).unwrap());
        assert_eq!((cur.width, cur.height), (1280, 720));
        assert_eq!(cur.num_planes, 1);
        assert_eq!(cur.bytesperline, 1280);
        assert_eq!(cur.sizeimage, 1280 * 720 * 3 / 2);
        assert_eq!(
            cur.colorspace,
            bindings::v4l2_colorspace_V4L2_COLORSPACE_SRGB
        );

        let mut asked = format(1900, 1000);
        asked.fmt.pix_mp.pixelformat = PixelFormat::from_fourcc(b"RGB3").to_u32();
        let tried = pix_mp(&r.device.try_fmt(&s, QUEUE, asked).unwrap());
        assert_eq!((tried.width, tried.height), (1920, 1080));
        assert_eq!(tried.pixelformat, NV12.to_u32());
        // TRY_FMT changed nothing.
        assert_eq!(pix_mp(&r.device.g_fmt(&s, QUEUE).unwrap()).width, 1280);

        let set = pix_mp(&r.device.s_fmt(&mut s, QUEUE, asked).unwrap());
        assert_eq!((set.width, set.height), (1920, 1080));
        assert_eq!(pix_mp(&r.device.g_fmt(&s, QUEUE).unwrap()).height, 1080);

        r.device
            .reqbufs(&mut s, QUEUE, MemoryType::Mmap, 2)
            .unwrap();
        assert_eq!(
            r.device.s_fmt(&mut s, QUEUE, format(64, 48)).err(),
            Some(libc::EBUSY)
        );
        // TRY_FMT still answers.
        assert_eq!(
            pix_mp(&r.device.try_fmt(&s, QUEUE, format(64, 48)).unwrap()).width,
            64
        );
        // CREATE_BUFS in another geometry may not join the 1080p buffers, and a set too small
        // for the frame is refused; a bigger one is honoured.
        assert_eq!(
            r.device
                .create_bufs(&mut s, 1, QUEUE, MemoryType::Mmap, format(1280, 720))
                .err(),
            Some(libc::EINVAL)
        );
        let mut small = format(1920, 1080);
        // SAFETY: multi-planar.
        unsafe { small.fmt.pix_mp.plane_fmt[0].sizeimage /= 2 };
        assert_eq!(
            r.device
                .create_bufs(&mut s, 1, QUEUE, MemoryType::Mmap, small)
                .err(),
            Some(libc::EINVAL)
        );
        let mut big = format(1920, 1080);
        // SAFETY: multi-planar.
        unsafe { big.fmt.pix_mp.plane_fmt[0].sizeimage *= 2 };
        let reply = r
            .device
            .create_bufs(&mut s, 1, QUEUE, MemoryType::Mmap, big)
            .unwrap();
        assert_eq!((reply.index, reply.count), (2, 1));
        assert_eq!(pix_mp(&reply.format).sizeimage, 1920 * 1080 * 3);
        assert!(r.device.querybuf(&s, QUEUE, 2).is_ok());
        assert_eq!(r.device.querybuf(&s, QUEUE, 3).err(), Some(libc::EINVAL));

        // One input, a camera.
        let input = r.device.enuminput(&s, 0).unwrap();
        assert_eq!(input.type_, bindings::V4L2_INPUT_TYPE_CAMERA);
        assert!(input.name.starts_with(b"Back camera\0"));
        assert_eq!(r.device.enuminput(&s, 1).err(), Some(libc::EINVAL));
        assert_eq!(r.device.s_input(&mut s, 1).err(), Some(libc::EINVAL));

        close(&mut r.device, s);
    }

    /// The fps rule of design §7.1: a single rate selects the widest range whose maximum is that
    /// rate, the reply is the rate chosen, a fraction that is not a rate means the default, and
    /// the size's own ceiling wins.
    #[test]
    fn parm_follows_the_widest_range_whose_max_matches() {
        let mut r = rig();
        let mut s = session(&mut r.device);

        let parm = r.device.g_parm(&s, QUEUE).unwrap();
        // SAFETY: capture.
        assert_eq!(
            unsafe { parm.parm.capture.capability },
            bindings::V4L2_CAP_TIMEPERFRAME
        );
        assert_eq!(timeperframe(&parm), (1, 30));
        assert_eq!(
            r.device.g_parm(&s, QueueType::VideoOutputMplane).err(),
            Some(libc::EINVAL)
        );

        // 30 fps: (7, 30) is the widest of the three ranges ending at 30.
        assert_eq!(info().range_for(30, None), (7, 30));
        assert_eq!(info().range_for(24, None), (24, 24));
        assert_eq!(info().range_for(15, None), (15, 15));
        // No range ends at 20: 24 and 15 are equally far, the faster one wins.
        assert_eq!(info().range_for(20, None), (24, 24));
        assert_eq!(info().range_for(1000, None), (60, 60));
        // A camera that lists none pins both ends.
        let mute = CameraInfo {
            fps_ranges: vec![],
            ..info()
        };
        assert_eq!(mute.range_for(25, None), (25, 25));

        let reply = r.device.s_parm(&mut s, parm_for((1, 24))).unwrap();
        assert_eq!(timeperframe(&reply), (1, 24));
        assert_eq!(timeperframe(&r.device.g_parm(&s, QUEUE).unwrap()), (1, 24));
        // 720p sustains 60.
        assert_eq!(
            timeperframe(&r.device.s_parm(&mut s, parm_for((1, 60))).unwrap()),
            (1, 60)
        );
        // ... 1080p does not: the rate is held to the size's ceiling, and comes back as such.
        r.device.s_fmt(&mut s, QUEUE, format(1920, 1080)).unwrap();
        assert_eq!(timeperframe(&r.device.g_parm(&s, QUEUE).unwrap()), (1, 30));
        // 2/60 is 30.
        assert_eq!(
            timeperframe(&r.device.s_parm(&mut s, parm_for((2, 60))).unwrap()),
            (1, 30)
        );
        // v4l2-compliance sends 0/1 and 1/0 and expects both to succeed.
        for bad in [(0, 1), (1, 0), (0, 0)] {
            assert_eq!(
                timeperframe(&r.device.s_parm(&mut s, parm_for(bad)).unwrap()),
                (1, 30)
            );
        }
        let mut wrong_queue = parm_for((1, 30));
        wrong_queue.type_ = QueueType::VideoOutputMplane as u32;
        assert_eq!(
            r.device.s_parm(&mut s, wrong_queue).err(),
            Some(libc::EINVAL)
        );

        // A size whose own duration ceiling is under every range the camera lists: both ioctls
        // answer with the ceiling. ENUM_FRAMEINTERVALS offered 1/15 and S_PARM used to snap
        // back to the nearest listed maximum, 30 -- a rate the size cannot sustain
        // (review-m4 R5).
        let slow = CameraInfo {
            sizes: vec![FrameSize {
                width: 3840,
                height: 2160,
                min_frame_duration_ns: Some(66_666_666),
            }],
            fps_ranges: vec![(30, 30)],
            ..info()
        };
        let uhd = slow.sizes[0];
        assert_eq!(slow.frame_rates(&uhd), vec![15]);
        assert_eq!(slow.range_at(&uhd, 15), (15, 15));
        assert_eq!(slow.range_at(&uhd, 30), (15, 15));
        // The same camera at a size with no ceiling still gets its listed range.
        assert_eq!(slow.range_for(30, None), (30, 30));
        let mut slow_rig = rig_with(FakeCamera {
            info: slow,
            ..camera()
        });
        let mut slow_s = session(&mut slow_rig.device);
        let intervals: Vec<(u32, u32)> = (0..)
            .map_while(|i| {
                slow_rig
                    .device
                    .enum_frameintervals(&slow_s, i, NV12.to_u32(), 3840, 2160)
                    .ok()
            })
            .map(|f| {
                // SAFETY: discrete.
                let d = unsafe { f.__bindgen_anon_1.discrete };
                (d.numerator, d.denominator)
            })
            .collect();
        assert_eq!(intervals, vec![(1, 15)]);
        assert_eq!(
            timeperframe(
                &slow_rig
                    .device
                    .s_parm(&mut slow_s, parm_for((1, 15)))
                    .unwrap()
            ),
            (1, 15)
        );
        assert_eq!(
            timeperframe(&slow_rig.device.g_parm(&slow_s, QUEUE).unwrap()),
            (1, 15)
        );
        close(&mut slow_rig.device, slow_s);

        // While streaming, S_PARM reaches the running stream as a control.
        r.device
            .reqbufs(&mut s, QUEUE, MemoryType::Mmap, 1)
            .unwrap();
        r.device.streamon(&mut s, QUEUE).unwrap();
        assert_eq!(r.log.lock().unwrap().opened[0].fps, (7, 30));
        r.device.s_parm(&mut s, parm_for((1, 15))).unwrap();
        assert_eq!(
            r.log.lock().unwrap().controls,
            vec![CameraControl::FpsRange(15, 15)]
        );

        close(&mut r.device, s);
    }

    /// The stream state machine on host-owned buffers: the camera opens at `STREAMON` with every
    /// buffer queued so far lent in order, a buffer queued while streaming is lent at once,
    /// frames come back as `DQBUF` events carrying the stream's sequence, timestamp and size --
    /// and the bytes -- and `STREAMOFF` closes the camera before the buffers are unqueued.
    #[test]
    fn streamon_opens_the_camera_and_lends_the_queued_buffers() {
        let mut r = rig();
        let mut s = session(&mut r.device);
        r.device.s_fmt(&mut s, QUEUE, format(64, 48)).unwrap();
        let size = 64 * 48 * 3 / 2;

        r.device
            .reqbufs(&mut s, QUEUE, MemoryType::Mmap, 3)
            .unwrap();
        assert_eq!(
            r.device.streamon(&mut s, QueueType::VideoCapture).err(),
            Some(libc::EINVAL)
        );

        // Two buffers queued before STREAMON: nothing is opened yet.
        for index in [1, 0] {
            let reply = r
                .device
                .qbuf(&mut s, mmap_buffer(index, size), vec![], true)
                .unwrap();
            assert!(reply.flags().contains(BufferFlags::QUEUED));
        }
        assert!(r.log.lock().unwrap().opened.is_empty());
        // Queueing a queued buffer is refused.
        assert_eq!(
            r.device
                .qbuf(&mut s, mmap_buffer(0, size), vec![], true)
                .err(),
            Some(libc::EINVAL)
        );

        r.device.streamon(&mut s, QUEUE).unwrap();
        {
            let log = r.log.lock().unwrap();
            assert_eq!(log.opened.len(), 1);
            assert_eq!(
                log.opened[0],
                StreamRequest {
                    width: 64,
                    height: 48,
                    fps: (7, 30),
                    buffers: 3,
                }
            );
        }
        // STREAMON twice is fine and opens nothing more.
        r.device.streamon(&mut s, QUEUE).unwrap();
        assert_eq!(r.log.lock().unwrap().opened.len(), 1);

        // The third buffer, queued while streaming, is lent straight away.
        r.device
            .qbuf(&mut s, mmap_buffer(2, size), vec![], true)
            .unwrap();

        let frames = collect_frames(&mut r, &mut s, 3);
        assert_eq!(
            r.log.lock().unwrap().lent,
            vec![1, 0, 2],
            "lent in queueing order"
        );
        for (n, frame) in frames.iter().enumerate() {
            assert_eq!(frame.queue(), QUEUE);
            assert_eq!(frame.sequence(), n as u32);
            assert_eq!(*frame.get_first_plane().bytesused, size);
            assert!(!frame.flags().contains(BufferFlags::QUEUED));
            assert!(frame.flags().contains(BufferFlags::TIMESTAMP_MONOTONIC));
            let ts = frame.timestamp();
            let ns = 1_500_000_000 + n as i64 * 33_333_333;
            assert_eq!(
                (ts.tv_sec, ts.tv_usec),
                (ns / 1_000_000_000, (ns % 1_000_000_000) / 1_000)
            );
        }
        assert_eq!(
            frames.iter().map(|f| f.index()).collect::<Vec<_>>(),
            vec![1, 0, 2]
        );
        // The bytes: buffer 0 holds frame 1, luma then 0x80 chroma.
        let Backing::Host { buffer, .. } = &s.buffers[0].backing else {
            panic!("not host-owned");
        };
        // SAFETY: no guest mapping of this buffer was ever made.
        let bytes = unsafe { buffer.as_slice() };
        assert_eq!(bytes[0], luma_of(1));
        assert_eq!(bytes[64 * 48 - 1], luma_of(1));
        assert_eq!(bytes[64 * 48], 0x80);
        assert_eq!(bytes[size as usize - 1], 0x80);
        // QUERYBUF agrees the buffer is done and dequeued.
        let queried = r.device.querybuf(&s, QUEUE, 0).unwrap();
        assert!(!queried.flags().contains(BufferFlags::QUEUED));
        assert_eq!(queried.sequence(), 1);

        // Requeue one: the next frame goes into it.
        r.device
            .qbuf(&mut s, mmap_buffer(0, size), vec![], true)
            .unwrap();
        let frames = collect_frames(&mut r, &mut s, 4);
        assert_eq!(frames[3].index(), 0);
        assert_eq!(frames[3].sequence(), 3);

        // STREAMOFF: the camera is closed (joined) first, then the buffers are given back.
        r.device
            .qbuf(&mut s, mmap_buffer(1, size), vec![], true)
            .unwrap();
        r.device.streamoff(&mut s, QUEUE).unwrap();
        {
            let log = r.log.lock().unwrap();
            assert_eq!(log.closed, 1);
            assert!(!log.streaming);
        }
        assert!(s.buffers.iter().all(|b| !b.queued && !b.lent));
        assert!(!r
            .device
            .querybuf(&s, QUEUE, 1)
            .unwrap()
            .flags()
            .contains(BufferFlags::QUEUED));
        // No stray frame after the close.
        process(&mut r.device, &mut s);
        assert_eq!(dequeued(&r.events.borrow()).len(), 4);

        // A second round starts a fresh stream with a fresh sequence.
        r.device
            .qbuf(&mut s, mmap_buffer(2, size), vec![], true)
            .unwrap();
        r.device.streamon(&mut s, QUEUE).unwrap();
        assert_eq!(r.log.lock().unwrap().opened.len(), 2);
        let frames = collect_frames(&mut r, &mut s, 5);
        assert_eq!((frames[4].index(), frames[4].sequence()), (2, 0));

        close(&mut r.device, s);
        let log = r.log.lock().unwrap();
        assert_eq!(log.closed, 2, "close joined the second stream");
        assert_eq!(log.released_while_streaming, 0);
        assert_eq!(*r.released.borrow(), 3);
    }

    /// §2.5: `REQBUFS(0)` and a session close both join the stream before a single buffer goes
    /// back to the allocator; `REQBUFS(n)` under a stream is refused.
    #[test]
    fn reqbufs_zero_and_close_join_the_stream_before_freeing() {
        let mut r = rig();
        let mut s = session(&mut r.device);
        r.device.s_fmt(&mut s, QUEUE, format(64, 48)).unwrap();
        let size = 64 * 48 * 3 / 2;

        r.device
            .reqbufs(&mut s, QUEUE, MemoryType::Mmap, 2)
            .unwrap();
        r.device
            .qbuf(&mut s, mmap_buffer(0, size), vec![], true)
            .unwrap();
        r.device.streamon(&mut s, QUEUE).unwrap();
        assert_eq!(
            r.device.reqbufs(&mut s, QUEUE, MemoryType::Mmap, 4).err(),
            Some(libc::EBUSY)
        );
        assert_eq!(
            r.device.s_fmt(&mut s, QUEUE, format(64, 48)).err(),
            Some(libc::EBUSY)
        );

        r.device
            .reqbufs(&mut s, QUEUE, MemoryType::Mmap, 0)
            .unwrap();
        {
            let log = r.log.lock().unwrap();
            assert_eq!(log.closed, 1);
            assert_eq!(log.released_while_streaming, 0);
        }
        assert_eq!(*r.released.borrow(), 2);
        assert!(s.buffers.is_empty());
        assert_eq!(s.memory, None);
        assert_eq!(r.device.active_session, None);

        // Again, ending with a close instead.
        r.device
            .reqbufs(&mut s, QUEUE, MemoryType::Mmap, 2)
            .unwrap();
        r.device
            .qbuf(&mut s, mmap_buffer(1, size), vec![], true)
            .unwrap();
        r.device.streamon(&mut s, QUEUE).unwrap();
        // A second session cannot take the camera meanwhile.
        let mut other =
            <Device as VirtioMediaDevice<&[u8], Vec<u8>>>::new_session(&mut r.device, 1).unwrap();
        assert_eq!(
            r.device
                .reqbufs(&mut other, QUEUE, MemoryType::Mmap, 1)
                .err(),
            Some(libc::EBUSY)
        );
        close(&mut r.device, s);
        {
            let log = r.log.lock().unwrap();
            assert_eq!(log.closed, 2);
            assert_eq!(log.released_while_streaming, 0);
        }
        assert_eq!(*r.released.borrow(), 4);
        // ... and can afterwards.
        r.device
            .reqbufs(&mut other, QUEUE, MemoryType::Mmap, 1)
            .unwrap();
        close(&mut r.device, other);
    }

    /// `PREPARE_BUF` on this device, and the scatter list `QBUF` must not be allowed to smuggle
    /// past it. The buffer is prepared with a full-sized plane and then queued with a short one:
    /// the list the ioctl layer read is *this* call's, so it is this call's `length` that
    /// decides how much guest memory the loan covers, and a prepared buffer may not shrink it
    /// (review-m4 R2). Nothing is mapped by `PREPARE_BUF` itself.
    #[test]
    fn prepare_buf_does_not_let_qbuf_shrink_the_mapping() {
        let mut r = rig();
        let mut s = session(&mut r.device);
        r.device.s_fmt(&mut s, QUEUE, format(64, 48)).unwrap();
        let size = 64 * 48 * 3 / 2;
        r.device
            .reqbufs(&mut s, QUEUE, MemoryType::UserPtr, 2)
            .unwrap();
        let gpa = 4 * 0x1000u64;

        // The payload is what `PREPARE_BUF` validates, so a dispatcher that could not represent
        // it is refused, and so are the lengths `QBUF` refuses.
        let (buf, _) = userptr_buffer(0, gpa, size);
        assert_eq!(
            r.device.prepare_buf(&mut s, buf, vec![], false).err(),
            Some(libc::EINVAL)
        );
        for len in [size - 1, size + 1] {
            let (buf, _) = userptr_buffer(0, gpa, len);
            assert_eq!(
                r.device.prepare_buf(&mut s, buf, vec![], true).err(),
                Some(libc::EINVAL)
            );
        }

        let (buf, _) = userptr_buffer(0, gpa, size);
        let reply = r.device.prepare_buf(&mut s, buf, vec![], true).unwrap();
        assert_eq!(
            (reply.flags() & (BufferFlags::QUEUED | BufferFlags::PREPARED | BufferFlags::DONE))
                .bits(),
            BufferFlags::PREPARED.bits()
        );
        assert_eq!(reply.sequence(), 0);
        assert_eq!(reply.timestamp().tv_sec, 0);
        assert_eq!(*reply.get_first_plane().bytesused, 0);
        assert!(r
            .device
            .querybuf(&s, QUEUE, 0)
            .unwrap()
            .flags()
            .contains(BufferFlags::PREPARED));
        // No guest memory is held between PREPARE_BUF and QBUF.
        assert_eq!(*r.guest.live_mappings.borrow(), 0);
        // Preparing it twice is refused.
        let (buf, _) = userptr_buffer(0, gpa, size);
        assert_eq!(
            r.device.prepare_buf(&mut s, buf, vec![], true).err(),
            Some(libc::EINVAL)
        );

        // The blocker: queueing the prepared buffer with a plane the guest has shrunk. The
        // prepared length says a frame fits; this call's does not, and this call's is the one
        // the scatter list was read against.
        let (buf, sgs) = userptr_buffer(0, gpa, 8);
        assert_eq!(
            r.device.qbuf(&mut s, buf, sgs, true).err(),
            Some(libc::EINVAL)
        );
        assert_eq!(*r.guest.live_mappings.borrow(), 0, "nothing was mapped");
        assert!(!s.buffers[0].queued);
        assert!(s.buffers[0].prepared.is_some(), "still prepared");

        // Queued honestly it works, and the payload PREPARE_BUF accepted is what comes back.
        let (buf, sgs) = userptr_buffer(0, gpa, size);
        let reply = r.device.qbuf(&mut s, buf, sgs, false).unwrap();
        assert!(reply.flags().contains(BufferFlags::QUEUED));
        assert!(!reply.flags().contains(BufferFlags::PREPARED));
        assert_eq!(*reply.get_first_plane().length, size);
        assert_eq!(*r.guest.live_mappings.borrow(), 1);

        r.device.streamon(&mut s, QUEUE).unwrap();
        let frames = collect_frames(&mut r, &mut s, 1);
        assert_eq!(frames[0].index(), 0);
        close(&mut r.device, s);
    }

    /// The second line of defence behind the check above: a mapping that turns out to be
    /// shorter than a frame is never lent, whatever the guest's `length` field claimed
    /// (review-m4 R2). At `STREAMON` that is the undo path -- the camera is closed again and
    /// every buffer stays queued, in the order the guest queued them; on a queue that is
    /// already streaming it ends the session, because a stream that will not take a buffer is
    /// not going to fill it.
    #[test]
    fn a_mapping_too_short_for_a_frame_is_not_lent() {
        let mut r = rig();
        let mut s = session(&mut r.device);
        r.device.s_fmt(&mut s, QUEUE, format(64, 48)).unwrap();
        let size = 64 * 48 * 3 / 2;
        r.device
            .reqbufs(&mut s, QUEUE, MemoryType::UserPtr, 2)
            .unwrap();
        let gpa = 4 * 0x1000u64;

        // A well-formed `length` over a scatter list that covers a quarter of it.
        let (buf, sgs) = userptr_buffer_sized(1, gpa + 0x1_0000, size, size / 4);
        r.device.qbuf(&mut s, buf, sgs, true).unwrap();
        let (buf, sgs) = userptr_buffer(0, gpa, size);
        r.device.qbuf(&mut s, buf, sgs, true).unwrap();
        assert_eq!(s.queued, VecDeque::from([1, 0]));

        assert_eq!(r.device.streamon(&mut s, QUEUE).err(), Some(libc::EIO));
        assert!(s.stream.is_none(), "the camera was closed again");
        assert!(!s.dead);
        assert_eq!(r.log.lock().unwrap().closed, 1);
        assert!(s.buffers.iter().all(|b| b.queued && !b.lent));
        assert_eq!(
            s.queued,
            VecDeque::from([1, 0]),
            "the retry keeps the guest's queueing order"
        );

        // Queued honestly, both stream.
        r.device.streamoff(&mut s, QUEUE).unwrap();
        for index in 0..2u32 {
            let (buf, sgs) = userptr_buffer(index, gpa + index as u64 * 0x1_0000, size);
            r.device.qbuf(&mut s, buf, sgs, true).unwrap();
        }
        r.device.streamon(&mut s, QUEUE).unwrap();
        let _ = collect_frames(&mut r, &mut s, 2);
        assert_eq!(r.log.lock().unwrap().lent, vec![0, 1]);

        // Now the same short mapping while the queue streams: the session ends.
        let (buf, sgs) = userptr_buffer_sized(1, gpa + 0x1_0000, size, size / 4);
        assert_eq!(r.device.qbuf(&mut s, buf, sgs, true).err(), Some(libc::EIO));
        assert!(s.dead);
        assert!(s.stream.is_none());
        assert_eq!(errors(&r.events.borrow()), 1);
        assert_eq!(*r.guest.live_mappings.borrow(), 0);
        close(&mut r.device, s);
    }

    /// §2.5 on the path that is not an ioctl: a session that is merely *dropped* -- which is what
    /// becomes of every session the guest has not closed when the worker thread ends -- must
    /// still join the capture thread before the buffers it was lent, and the guest mappings
    /// behind them, go away. `FakeMapping::drop` makes the assertion; `CameraSession`'s field
    /// order is what makes it hold (review-m4 R1).
    #[test]
    fn a_dropped_session_joins_the_stream_before_its_buffers() {
        // The camera keeps the buffers it is lent, as one writing a frame does.
        let mut r = rig_with(FakeCamera {
            hold_from: Some(0),
            ..camera()
        });
        let mut s = session(&mut r.device);
        r.device.s_fmt(&mut s, QUEUE, format(64, 48)).unwrap();
        let size = 64 * 48 * 3 / 2;
        r.device
            .reqbufs(&mut s, QUEUE, MemoryType::UserPtr, 2)
            .unwrap();
        let gpa = 4 * 0x1000u64;
        for index in 0..2u32 {
            let (buf, sgs) = userptr_buffer(index, gpa + index as u64 * 0x1_0000, size);
            r.device.qbuf(&mut s, buf, sgs, true).unwrap();
        }
        r.device.streamon(&mut s, QUEUE).unwrap();
        // The capture thread has written into a lent buffer and is still running: nothing
        // collects the frames, which is the state a killed worker leaves behind.
        assert!(wait_ready(&s), "no frame within 2s");
        assert_eq!(*r.guest.live_mappings.borrow(), 2);
        assert!(r.log.lock().unwrap().streaming);

        drop(s);

        {
            let log = r.log.lock().unwrap();
            assert_eq!(log.closed, 1, "dropping the session joined the stream");
            assert!(!log.streaming);
        }
        assert_eq!(*r.guest.live_mappings.borrow(), 0);
    }

    /// The same invariant one level up: the worker never sends `CLOSE` when it ends on its kill
    /// event, on `stop_queue` or on `reset` -- it drops the runner. `VirtioMediaDeviceRunner`'s
    /// `Drop` closes whatever sessions are left through the device, so the stream is joined, the
    /// buffers go back to the allocator and the session's descriptor leaves the wait context;
    /// with the device dropped first none of that happened (review-m4 R1).
    #[test]
    fn a_dropped_runner_closes_its_sessions() {
        let r = rig_with(FakeCamera {
            hold_from: Some(0),
            ..camera()
        });
        let poller = FakePoller::default();
        // OPEN, as the worker would dispatch it. Declared before the runner: the reader type is
        // part of the runner's own type, so these bytes must outlive it.
        let cmd: Vec<u8> = VIRTIO_MEDIA_CMD_OPEN
            .to_le_bytes()
            .into_iter()
            .chain(0u32.to_le_bytes())
            .collect();
        let mut writer: Vec<u8> = Vec::new();
        let mut runner: VirtioMediaDeviceRunner<&[u8], Vec<u8>, Device, FakePoller> =
            VirtioMediaDeviceRunner::new(r.device, poller.clone());

        runner.handle_command(&mut cmd.as_slice(), &mut writer);
        assert_eq!(runner.sessions.len(), 1);
        assert_eq!(poller.added.borrow().len(), 1, "the session is polled");

        let size = 64 * 48 * 3 / 2;
        {
            let s = runner.sessions.get_mut(&0).unwrap();
            runner.device.s_fmt(s, QUEUE, format(64, 48)).unwrap();
            runner
                .device
                .reqbufs(s, QUEUE, MemoryType::Mmap, 2)
                .unwrap();
            for index in 0..2 {
                runner
                    .device
                    .qbuf(s, mmap_buffer(index, size), vec![], true)
                    .unwrap();
            }
            runner.device.streamon(s, QUEUE).unwrap();
            assert!(wait_ready(s), "no frame within 2s");
        }
        assert!(r.log.lock().unwrap().streaming);

        drop(runner);

        {
            let log = r.log.lock().unwrap();
            assert_eq!(log.closed, 1, "dropping the runner joined the stream");
            assert_eq!(log.released_while_streaming, 0);
        }
        assert_eq!(*r.released.borrow(), 2, "the buffers went back");
        assert_eq!(poller.removed.borrow().len(), 1, "and stopped being polled");
    }

    /// Guest-owned CAPTURE buffers (`driver_owned_queues=all`): the frame lands in the guest's
    /// pages through a writable mapping held while the buffer is lent and dropped before the
    /// `DQBUF` event; a buffer that cannot hold a frame is refused.
    #[test]
    fn userptr_capture_buffers_are_filled_through_the_guest_mapping() {
        let mut r = rig();
        let mut s = session(&mut r.device);
        r.device.s_fmt(&mut s, QUEUE, format(64, 48)).unwrap();
        let size = 64 * 48 * 3 / 2;

        let reply = r
            .device
            .reqbufs(&mut s, QUEUE, MemoryType::UserPtr, 2)
            .unwrap();
        assert!(reply.capabilities & BufferCapabilities::SUPPORTS_USERPTR.bits() != 0);

        let gpa = 4 * 0x1000u64;
        // Too small for a frame, and larger than the buffer was allocated for.
        for len in [size - 1, size + 1] {
            let (buf, sgs) = userptr_buffer(0, gpa, len);
            assert_eq!(
                r.device.qbuf(&mut s, buf, sgs, true).err(),
                Some(libc::EINVAL)
            );
        }
        assert_eq!(*r.guest.live_mappings.borrow(), 0);
        // The memory type is fixed by REQBUFS.
        assert_eq!(
            r.device
                .qbuf(&mut s, mmap_buffer(0, size), vec![], true)
                .err(),
            Some(libc::EINVAL)
        );

        let (buf, sgs) = userptr_buffer(0, gpa, size);
        r.device.qbuf(&mut s, buf, sgs, true).unwrap();
        assert_eq!(*r.guest.live_mappings.borrow(), 1, "held while queued");
        r.device.streamon(&mut s, QUEUE).unwrap();

        let frames = collect_frames(&mut r, &mut s, 1);
        assert_eq!(frames[0].index(), 0);
        assert_eq!(frames[0].memory(), MemoryType::UserPtr);
        if let v4l2r::ioctl::V4l2PlanesWithBacking::UserPtr(mut planes) =
            frames[0].planes_with_backing_iter()
        {
            assert_eq!(planes.next().unwrap().userptr(), 0xc000_0000);
        } else {
            panic!("the buffer lost its memory type");
        }
        assert_eq!(*frames[0].get_first_plane().bytesused, size);
        // The mapping went before the event.
        assert_eq!(*r.guest.live_mappings.borrow(), 0);
        {
            let mem = r.guest.memory.borrow();
            let dst = &mem[gpa as usize..gpa as usize + size as usize];
            assert!(dst[..64 * 48].iter().all(|&b| b == luma_of(0)));
            assert!(dst[64 * 48..].iter().all(|&b| b == 0x80));
            // Not one byte past the buffer.
            assert_eq!(mem[gpa as usize + size as usize], 0);
        }

        // A buffer still lent at STREAMOFF has its mapping released only after the join.
        let (buf, sgs) = userptr_buffer(1, gpa + 0x10000, size);
        r.device.qbuf(&mut s, buf, sgs, true).unwrap();
        let _ = collect_frames(&mut r, &mut s, 2);
        let (buf, sgs) = userptr_buffer(0, gpa, size);
        r.device.qbuf(&mut s, buf, sgs, true).unwrap();
        r.device.streamoff(&mut s, QUEUE).unwrap();
        assert_eq!(r.log.lock().unwrap().closed, 1);
        assert_eq!(*r.guest.live_mappings.borrow(), 0);

        close(&mut r.device, s);
    }

    /// A camera that goes away mid-stream: the frames that made it are delivered, the session
    /// gets one error event, its buffers are given back, and everything that would touch the
    /// stream answers `ENODEV` until the guest closes -- which still works.
    #[test]
    fn a_camera_that_dies_ends_the_session_with_an_error_event() {
        let mut r = rig_with(FakeCamera {
            disconnect_after: Some(2),
            ..camera()
        });
        let mut s = session(&mut r.device);
        r.device.s_fmt(&mut s, QUEUE, format(64, 48)).unwrap();
        let size = 64 * 48 * 3 / 2;
        r.device
            .reqbufs(&mut s, QUEUE, MemoryType::Mmap, 3)
            .unwrap();
        for index in 0..3 {
            r.device
                .qbuf(&mut s, mmap_buffer(index, size), vec![], true)
                .unwrap();
        }
        r.device.streamon(&mut s, QUEUE).unwrap();

        // Two frames, then the disconnect.
        while errors(&r.events.borrow()) == 0 {
            assert!(wait_ready(&s), "no event within 2s");
            process(&mut r.device, &mut s);
        }
        assert_eq!(dequeued(&r.events.borrow()).len(), 2);
        assert_eq!(errors(&r.events.borrow()), 1);
        assert!(s.dead);
        assert!(s.stream.is_none());
        assert_eq!(r.log.lock().unwrap().closed, 1);
        assert!(s.buffers.iter().all(|b| !b.queued));

        assert_eq!(
            r.device
                .qbuf(&mut s, mmap_buffer(2, size), vec![], true)
                .err(),
            Some(libc::ENODEV)
        );
        assert_eq!(r.device.streamon(&mut s, QUEUE).err(), Some(libc::ENODEV));
        // Nothing to collect any more, and no second error.
        process(&mut r.device, &mut s);
        assert_eq!(errors(&r.events.borrow()), 1);

        close(&mut r.device, s);
        assert_eq!(*r.released.borrow(), 3);
        assert_eq!(r.log.lock().unwrap().released_while_streaming, 0);
    }

    /// §2.5 for the path `end_session` takes, on guest-owned buffers. The camera goes away
    /// while it holds a buffer it never filled: the stream must be joined before that buffer's
    /// mapping is released, or the guest's pages are handed back under a thread that still has
    /// a pointer into them. `FakeMapping::drop` is what checks it -- with host buffers the same
    /// test proves nothing, because `end_session` releases none (review-m4 R7).
    #[test]
    fn a_dying_camera_releases_the_guest_mappings_after_the_join() {
        let mut r = rig_with(FakeCamera {
            disconnect_after: Some(1),
            ..camera()
        });
        let mut s = session(&mut r.device);
        r.device.s_fmt(&mut s, QUEUE, format(64, 48)).unwrap();
        let size = 64 * 48 * 3 / 2;
        r.device
            .reqbufs(&mut s, QUEUE, MemoryType::UserPtr, 2)
            .unwrap();
        let gpa = 4 * 0x1000u64;
        for index in 0..2u32 {
            let (buf, sgs) = userptr_buffer(index, gpa + index as u64 * 0x1_0000, size);
            r.device.qbuf(&mut s, buf, sgs, true).unwrap();
        }
        assert_eq!(*r.guest.live_mappings.borrow(), 2);
        r.device.streamon(&mut s, QUEUE).unwrap();

        // One frame, then the camera disconnects while it holds buffer 1.
        while errors(&r.events.borrow()) == 0 {
            assert!(wait_ready(&s), "no event within 2s");
            process(&mut r.device, &mut s);
        }
        assert_eq!(dequeued(&r.events.borrow()).len(), 1);
        assert!(s.dead);
        assert_eq!(r.log.lock().unwrap().closed, 1, "joined");
        assert!(s.buffers.iter().all(|b| !b.queued && !b.lent));
        assert_eq!(*r.guest.live_mappings.borrow(), 0, "and only then released");

        close(&mut r.device, s);
    }

    /// A session the camera has left holds nothing more: everything that would take or keep
    /// `media_host` space, or set the session up to, answers `ENODEV` the way `QBUF` and
    /// `STREAMON` already did -- a guest that ignores the error event could otherwise
    /// `REQBUFS(32)` on a session it can never stream (review-m4 R12). Closing it still works.
    #[test]
    fn a_dead_session_is_enodev_for_everything_that_takes_buffers() {
        let mut r = rig_with(FakeCamera {
            disconnect_after: Some(0),
            ..camera()
        });
        let mut s = session(&mut r.device);
        r.device.s_fmt(&mut s, QUEUE, format(64, 48)).unwrap();
        let size = 64 * 48 * 3 / 2;
        r.device
            .reqbufs(&mut s, QUEUE, MemoryType::Mmap, 1)
            .unwrap();
        r.device
            .qbuf(&mut s, mmap_buffer(0, size), vec![], true)
            .unwrap();
        r.device.streamon(&mut s, QUEUE).unwrap();
        while errors(&r.events.borrow()) == 0 {
            assert!(wait_ready(&s), "no event within 2s");
            process(&mut r.device, &mut s);
        }
        assert!(s.dead);

        assert_eq!(
            r.device.reqbufs(&mut s, QUEUE, MemoryType::Mmap, 4).err(),
            Some(libc::ENODEV)
        );
        assert_eq!(
            r.device.reqbufs(&mut s, QUEUE, MemoryType::Mmap, 0).err(),
            Some(libc::ENODEV)
        );
        assert_eq!(
            r.device
                .create_bufs(&mut s, 2, QUEUE, MemoryType::Mmap, format(64, 48))
                .err(),
            Some(libc::ENODEV)
        );
        assert_eq!(
            r.device
                .prepare_buf(&mut s, mmap_buffer(0, size), vec![], true)
                .err(),
            Some(libc::ENODEV)
        );
        assert_eq!(
            r.device.s_fmt(&mut s, QUEUE, format(1280, 720)).err(),
            Some(libc::ENODEV)
        );
        assert_eq!(
            r.device.s_parm(&mut s, parm_for((1, 15))).err(),
            Some(libc::ENODEV)
        );
        // The buffers it already had are still exactly the one it allocated.
        assert_eq!(s.buffers.len(), 1);
        assert_eq!(*r.released.borrow(), 0);

        close(&mut r.device, s);
        assert_eq!(*r.released.borrow(), 1);
    }

    /// `CREATE_BUFS` on a guest-owned queue: the set is appended to the `REQBUFS` one, it may
    /// be sized larger than a frame, the memory type may not change, and a buffer created that
    /// way streams -- the loan is sized from the session's format, never from the buffer's own
    /// size (review-m4 R7).
    #[test]
    fn create_bufs_appends_userptr_buffers() {
        let mut r = rig();
        let mut s = session(&mut r.device);
        r.device.s_fmt(&mut s, QUEUE, format(64, 48)).unwrap();
        let size = 64 * 48 * 3 / 2;
        r.device
            .reqbufs(&mut s, QUEUE, MemoryType::UserPtr, 1)
            .unwrap();

        // A set smaller than the queue's own format is refused; a bigger one is not.
        assert_eq!(
            r.device
                .create_bufs(
                    &mut s,
                    1,
                    QUEUE,
                    MemoryType::UserPtr,
                    to_v4l2_sized(FrameSize::new(64, 48), size - 1),
                )
                .err(),
            Some(libc::EINVAL)
        );
        // One memory type per queue.
        assert_eq!(
            r.device
                .create_bufs(&mut s, 1, QUEUE, MemoryType::Mmap, format(64, 48))
                .err(),
            Some(libc::EINVAL)
        );

        let big = size + 0x1000;
        let reply = r
            .device
            .create_bufs(
                &mut s,
                1,
                QUEUE,
                MemoryType::UserPtr,
                to_v4l2_sized(FrameSize::new(64, 48), big),
            )
            .unwrap();
        assert_eq!((reply.index, reply.count), (1, 1));
        assert_eq!(pix_mp(&reply.format).sizeimage, big);
        assert_eq!(s.buffers.len(), 2);
        assert_eq!(s.buffers[1].size, big);
        assert_eq!(s.memory, Some(MemoryType::UserPtr));

        // The created buffer takes a plane between a frame and its own size, and no more.
        let gpa = 4 * 0x1000u64;
        for len in [size - 1, big + 1] {
            let (buf, sgs) = userptr_buffer(1, gpa, len);
            assert_eq!(
                r.device.qbuf(&mut s, buf, sgs, true).err(),
                Some(libc::EINVAL)
            );
        }
        let (buf, sgs) = userptr_buffer(1, gpa, big);
        r.device.qbuf(&mut s, buf, sgs, true).unwrap();
        r.device.streamon(&mut s, QUEUE).unwrap();
        let frames = collect_frames(&mut r, &mut s, 1);
        assert_eq!(frames[0].index(), 1);
        // Filled with a frame's worth, not the buffer's worth.
        assert_eq!(*frames[0].get_first_plane().bytesused, size);

        close(&mut r.device, s);
    }

    /// A camera that will not open (in use, no permission) fails `STREAMON` with that errno and
    /// leaves the buffers queued for another try.
    #[test]
    fn a_stream_that_cannot_open_fails_streamon_and_keeps_the_buffers_queued() {
        let mut r = rig_with(FakeCamera {
            fail_open: Some(libc::EBUSY),
            ..camera()
        });
        let mut s = session(&mut r.device);
        r.device.s_fmt(&mut s, QUEUE, format(64, 48)).unwrap();
        let size = 64 * 48 * 3 / 2;
        r.device
            .reqbufs(&mut s, QUEUE, MemoryType::Mmap, 2)
            .unwrap();
        r.device
            .qbuf(&mut s, mmap_buffer(0, size), vec![], true)
            .unwrap();

        assert_eq!(r.device.streamon(&mut s, QUEUE).err(), Some(libc::EBUSY));
        assert!(s.stream.is_none());
        assert!(!s.dead);
        assert!(s.buffers[0].queued);
        assert_eq!(s.queued, VecDeque::from([0]));
        assert_eq!(r.log.lock().unwrap().opened.len(), 1);

        // Once the camera is free the same session streams.
        r.device.backend.fail_open = None;
        r.device.streamon(&mut s, QUEUE).unwrap();
        let frames = collect_frames(&mut r, &mut s, 1);
        assert_eq!(frames[0].index(), 0);

        close(&mut r.device, s);
    }

    #[test]
    fn events_are_eos_and_source_change_only() {
        let mut r = rig();
        let mut s = session(&mut r.device);
        assert_eq!(
            r.device
                .subscribe_event(&mut s, EventType::Eos, SubscribeEventFlags::empty()),
            Ok(())
        );
        assert_eq!(
            r.device.subscribe_event(
                &mut s,
                EventType::SourceChange(0),
                SubscribeEventFlags::empty()
            ),
            Ok(())
        );
        assert_eq!(
            r.device
                .subscribe_event(&mut s, EventType::VSync, SubscribeEventFlags::empty()),
            Err(libc::EINVAL)
        );
        close(&mut r.device, s);
    }

    /// D18: the guest picks the dimensions of a `TRY_FMT`/`S_FMT` and V4L2 caps neither, so
    /// `nearest_size`'s metric has to hold every `u32` pair. It did not: the squared distance
    /// was computed in `i64`, `width = 3_100_000_000` overflowed it, and with overflow checks on
    /// the panic aborted the helper -- one ioctl from any guest process ended the VM, and
    /// `v4l2-compliance -s` hit it by accident a third of the way in.
    #[test]
    fn an_enormous_try_fmt_snaps_to_the_largest_size_instead_of_panicking() {
        let mut r = rig();
        let mut s = session(&mut r.device);

        // The reproduction from M4-acceptance §9 D18, and both ends of the range around it.
        for (w, h) in [
            (3_000_000_000, 100),
            (3_100_000_000, 100),
            (u32::MAX, 100),
            (u32::MAX, u32::MAX),
            (100, u32::MAX),
            (u32::MAX - 1, u32::MAX - 1),
        ] {
            let tried = pix_mp(&r.device.try_fmt(&s, QUEUE, format(w, h)).unwrap());
            assert_eq!(
                (tried.width, tried.height),
                (1920, 1080),
                "{}x{} did not snap to the largest size",
                w,
                h
            );
            // Nothing was changed by trying, and the buffer size stays a real number.
            assert_eq!(tried.sizeimage, 1920 * 1080 * 3 / 2);
            assert_eq!(pix_mp(&r.device.g_fmt(&s, QUEUE).unwrap()).width, 1280);
        }

        // `S_FMT` and `CREATE_BUFS` take the same path. `CREATE_BUFS` still judges the
        // `sizeimage` against the size the request snapped to, so an impossible geometry with a
        // small `sizeimage` is `EINVAL` rather than a panic or a 4 GiB allocation.
        let set = pix_mp(
            &r.device
                .s_fmt(&mut s, QUEUE, format(u32::MAX, u32::MAX))
                .unwrap(),
        );
        assert_eq!((set.width, set.height), (1920, 1080));
        let mut huge = format(u32::MAX, 1);
        // SAFETY: multi-planar.
        unsafe { huge.fmt.pix_mp.plane_fmt[0].sizeimage = 100 };
        assert_eq!(
            r.device
                .create_bufs(&mut s, 1, QUEUE, MemoryType::Mmap, huge)
                .err(),
            Some(libc::EINVAL)
        );
        assert!(s.buffers.is_empty());

        close(&mut r.device, s);
    }

    /// D16: `G_PARM` and `S_PARM` answer for the single-planar capture type as well as the
    /// mplane one, and echo back the type they were called with. v4l-utils 1.32.0 hardcodes
    /// `V4L2_BUF_TYPE_VIDEO_CAPTURE` in `v4l2-ctl --get-parm` / `--set-parm`, so with only the
    /// mplane type accepted neither could be run against this device.
    #[test]
    fn parm_answers_the_single_planar_capture_type_too() {
        const SPLANE: QueueType = QueueType::VideoCapture;
        let mut r = rig();
        let mut s = session(&mut r.device);

        for queue in [QUEUE, SPLANE] {
            let parm = r.device.g_parm(&s, queue).unwrap();
            assert_eq!(parm.type_, queue as u32);
            assert_eq!(timeperframe(&parm), (1, 30));
        }

        // A rate set through the single-planar type is the same session state the mplane type
        // reads back, and the reply carries the caller's own type.
        let mut asked = parm_for((1, 15));
        asked.type_ = SPLANE as u32;
        let set = r.device.s_parm(&mut s, asked).unwrap();
        assert_eq!(set.type_, SPLANE as u32);
        assert_eq!(timeperframe(&set), (1, 15));
        assert_eq!(s.fps, 15);
        let read_back = r.device.g_parm(&s, QUEUE).unwrap();
        assert_eq!(read_back.type_, QUEUE as u32);
        assert_eq!(timeperframe(&read_back), (1, 15));

        // And the other way round, so neither type is a second-class citizen.
        let set = r.device.s_parm(&mut s, parm_for((1, 24))).unwrap();
        assert_eq!(set.type_, QUEUE as u32);
        assert_eq!(timeperframe(&r.device.g_parm(&s, SPLANE).unwrap()), (1, 24));

        // Every other buffer type is still `EINVAL`, on both ioctls.
        for queue in [QueueType::VideoOutput, QueueType::VideoOutputMplane] {
            assert_eq!(r.device.g_parm(&s, queue).err(), Some(libc::EINVAL));
            let mut asked = parm_for((1, 30));
            asked.type_ = queue as u32;
            assert_eq!(r.device.s_parm(&mut s, asked).err(), Some(libc::EINVAL));
        }

        close(&mut r.device, s);
    }

    /// D19: `CREATE_BUFS(count = 0)` is the capability probe `v4l2-ctl --stream-mmap` sends
    /// before every stream. vb2 answers it from the memory and buffer types alone -- index,
    /// capabilities, success -- without reading the format, which is why the zeroed format it
    /// carries is not an error.
    #[test]
    fn create_bufs_with_count_zero_is_a_capability_probe() {
        let mut r = rig();
        let mut s = session(&mut r.device);

        let caps = (BufferCapabilities::SUPPORTS_MMAP
            | BufferCapabilities::SUPPORTS_USERPTR
            | BufferCapabilities::SUPPORTS_ORPHANED_BUFS)
            .bits();
        // Exactly what v4l2-ctl sends: the queue type and nothing else.
        let zeroed = v4l2_format {
            type_: QUEUE as u32,
            fmt: bindings::v4l2_format__bindgen_ty_1 {
                pix_mp: Default::default(),
            },
        };

        let reply = r
            .device
            .create_bufs(&mut s, 0, QUEUE, MemoryType::Mmap, zeroed)
            .unwrap();
        assert_eq!((reply.index, reply.count), (0, 0));
        assert_eq!(reply.capabilities, caps);
        // No buffers were made, and the session's own format is untouched.
        assert!(s.buffers.is_empty());
        assert!(s.memory.is_none());
        assert_eq!(pix_mp(&r.device.g_fmt(&s, QUEUE).unwrap()).width, 1280);

        // With buffers on the queue the probe reports the index the next one would take, and
        // still allocates nothing.
        r.device
            .reqbufs(&mut s, QUEUE, MemoryType::Mmap, 3)
            .unwrap();
        let reply = r
            .device
            .create_bufs(&mut s, 0, QUEUE, MemoryType::Mmap, zeroed)
            .unwrap();
        assert_eq!((reply.index, reply.count), (3, 0));
        assert_eq!(s.buffers.len(), 3);
        // A probe with the other memory type is not the "one memory type per queue" error
        // either: vb2 checks only that the type is one it supports.
        assert!(r
            .device
            .create_bufs(&mut s, 0, QUEUE, MemoryType::UserPtr, zeroed)
            .is_ok());

        // What is still refused: a memory type the device has no backing for, and a buffer type
        // that is not this queue.
        assert_eq!(
            r.device
                .create_bufs(&mut s, 0, QUEUE, MemoryType::DmaBuf, zeroed)
                .err(),
            Some(libc::EINVAL)
        );
        let output = QueueType::VideoOutputMplane;
        assert_eq!(
            r.device
                .create_bufs(&mut s, 0, output, MemoryType::Mmap, zeroed)
                .err(),
            Some(libc::EINVAL)
        );

        close(&mut r.device, s);
    }
}
