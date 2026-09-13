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
//! # Controls
//!
//! The camera's features -- zoom, exposure, focus, white balance, flash, and the rest of plan
//! §3.1's table -- are standard V4L2 controls, plus a private block for what V4L2 has no control
//! for (the metering and focus regions, the active lens, the auto-exposure state). Which ones
//! exist is decided once from [`CameraInfo`]; their values belong to the device, not to a
//! session, so a `v4l2-ctl --set-ctrl` from one process reaches the stream another runs, and
//! what is set with no stream open is applied when one opens. Changes -- a set from another
//! session, a flag flip, a state the backend observed -- are `V4L2_EVENT_CTRL` to the sessions
//! that subscribed. The [`controls`] module has the table, the units and the wire formats.
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

use std::collections::HashMap;
use std::collections::VecDeque;
use std::io::Result as IoResult;
use std::os::fd::AsFd;
use std::os::fd::AsRawFd;
use std::os::fd::BorrowedFd;
use std::os::fd::FromRawFd;
use std::os::fd::OwnedFd;
use std::sync::Arc;

use v4l2r::bindings;
use v4l2r::bindings::v4l2_control;
use v4l2r::bindings::v4l2_create_buffers;
use v4l2r::bindings::v4l2_event_subscription;
use v4l2r::bindings::v4l2_ext_control;
use v4l2r::bindings::v4l2_ext_controls;
use v4l2r::bindings::v4l2_fmtdesc;
use v4l2r::bindings::v4l2_format;
use v4l2r::bindings::v4l2_frmivalenum;
use v4l2r::bindings::v4l2_frmsizeenum;
use v4l2r::bindings::v4l2_query_ext_ctrl;
use v4l2r::bindings::v4l2_queryctrl;
use v4l2r::bindings::v4l2_querymenu;
use v4l2r::bindings::v4l2_requestbuffers;
use v4l2r::bindings::v4l2_streamparm;
use v4l2r::ioctl::BufferCapabilities;
use v4l2r::ioctl::BufferField;
use v4l2r::ioctl::BufferFlags;
use v4l2r::ioctl::CtrlWhich;
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
use crate::ioctl::PayloadValidity;
use crate::ioctl::VirtioMediaIoctlHandler;
use crate::mmap::MmapMappingManager;
use crate::mmap::RetiredBuffers;
use crate::protocol::DequeueBufferEvent;
use crate::protocol::SessionEvent;
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

pub mod controls;
pub use controls::AeMode;
pub use controls::AeState;
pub use controls::AfMode;
pub use controls::AfTrigger;
pub use controls::CameraControl;
pub use controls::ColorEffect;
pub use controls::ControlDesc;
pub use controls::Controls;
pub use controls::ExposureBias;
pub use controls::ExposureMode;
pub use controls::FlashLed;
pub use controls::IsoMode;
pub use controls::Kind;
pub use controls::MaxRegions;
pub use controls::PowerLine;
pub use controls::Region;
pub use controls::SceneMode;
pub use controls::Value;
pub use controls::WhiteBalance;
pub use controls::AF_STATUS_BUSY;
pub use controls::AF_STATUS_FAILED;
pub use controls::AF_STATUS_IDLE;
pub use controls::AF_STATUS_REACHED;
pub use controls::REGION_MAX_WEIGHT;
pub use controls::REGION_SCALE;
pub use controls::REGION_WORDS;
pub use controls::VCAM_CID_ACTIVE_PHYSICAL_ID;
pub use controls::VCAM_CID_AE_REGIONS;
pub use controls::VCAM_CID_AE_STATE;
pub use controls::VCAM_CID_AF_REGIONS;
pub use controls::VCAM_CID_AWB_REGIONS;
pub use controls::VCAM_CID_BASE;

/// The one pixel format offered: Y plane then interleaved Cb/Cr, tightly packed.
pub const NV12: PixelFormat = PixelFormat::from_fourcc(b"NV12");
/// Most buffers on the queue, the usual V4L2 ceiling.
pub const MAX_BUFFERS: usize = 32;
/// The queue this device has.
const QUEUE: QueueType = QueueType::VideoCaptureMplane;
/// Planes per buffer in every format this queue has. `QBUF` and `PREPARE_BUF` judge the guest's
/// payload description on these slots only: the rest of the plane array it sends is scratch
/// space it may leave dirty, which is what ffmpeg does (defect D21; the rule is
/// [`PayloadValidity::is_accepted_by`]).
const NUM_PLANES: usize = 1;
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

/// What a camera can do, in the terms `ENUM_FRAMESIZES`, `ENUM_FRAMEINTERVALS`, `G/S_PARM` and
/// the controls are answered with. Every control field is a capability: a control is offered
/// only when the field says the camera has the feature (`VPU_DESIGN.md` §7.1); the table
/// [`Controls::new`] builds says exactly which field makes which control.
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
    /// The zoom ratio range in hundredths (`CONTROL_ZOOM_RATIO_RANGE` x100): `ZOOM_ABSOLUTE`.
    pub zoom_range: Option<(u32, u32)>,
    /// The autofocus modes (`CONTROL_AF_AVAILABLE_MODES`): `FOCUS_AUTO`, the trigger buttons and
    /// `AUTO_FOCUS_STATUS`.
    pub af_modes: Vec<AfMode>,
    /// A flash unit (`FLASH_INFO_AVAILABLE`): `FLASH_LED_MODE`.
    pub flash: bool,
    /// The auto-exposure modes (`CONTROL_AE_AVAILABLE_MODES`); `Off` and `On` together make
    /// `EXPOSURE_AUTO` and `ISO_SENSITIVITY_AUTO`.
    pub ae_modes: Vec<AeMode>,
    /// The exposure time range in nanoseconds (`SENSOR_INFO_EXPOSURE_TIME_RANGE`):
    /// `EXPOSURE_ABSOLUTE`, in 100 µs units.
    pub exposure_range_ns: Option<(u64, u64)>,
    /// The sensitivity range (`SENSOR_INFO_SENSITIVITY_RANGE`): the `ISO_SENSITIVITY` menu.
    pub iso_range: Option<(u32, u32)>,
    /// The exposure compensation range and step: the `AUTO_EXPOSURE_BIAS` menu.
    pub exposure_bias: Option<ExposureBias>,
    /// The white-balance modes (`CONTROL_AWB_AVAILABLE_MODES`), in the crate's names:
    /// `AUTO_N_PRESET_WHITE_BALANCE`.
    pub awb_modes: Vec<WhiteBalance>,
    /// The antibanding modes (`CONTROL_AE_AVAILABLE_ANTIBANDING_MODES`): `POWER_LINE_FREQUENCY`.
    pub antibanding: Vec<PowerLine>,
    /// The colour effects (`CONTROL_AVAILABLE_EFFECTS`), those with a V4L2 name: `COLORFX`.
    pub effects: Vec<ColorEffect>,
    /// The scene modes (`CONTROL_AVAILABLE_SCENE_MODES`), those with a V4L2 name: `SCENE_MODE`.
    pub scenes: Vec<SceneMode>,
    /// Video stabilisation can be switched on (`CONTROL_AVAILABLE_VIDEO_STABILIZATION_MODES`):
    /// `IMAGE_STABILIZATION`.
    pub stabilization: bool,
    /// How many metering / focus / white-balance regions the camera takes
    /// (`CONTROL_MAX_REGIONS`): the private regions controls.
    pub max_regions: MaxRegions,
    /// The physical lenses behind a logical camera (`LOGICAL_MULTI_CAMERA_PHYSICAL_IDS`): the
    /// private active-physical-id control indexes this list.
    pub physical_ids: Vec<String>,
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
    /// A control's value as the camera reports it: the autofocus state, the auto-exposure
    /// state, the lens in use, or a value the backend had to change itself. The backend sends
    /// one only when the value differs from what it last sent, and the device compares again
    /// before it tells the guest, so a report is never a redundant `V4L2_EVENT_CTRL`.
    Control(CameraControl),
}

/// What a stream is opened for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StreamRequest {
    pub width: u32,
    pub height: u32,
    /// `(min, max)` frames per second, already resolved by the fps-range rule.
    pub fps: (u32, u32),
    /// How many buffers the guest allocated, for a backend that sizes a queue of its own.
    pub buffers: u32,
    /// Every settable control at its current value, to be applied before the first frame --
    /// what a guest set while nothing was streaming, or the defaults.
    pub controls: Vec<CameraControl>,
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

    /// Apply `controls` to the stream this backend opened last, if it is still running, in one
    /// submission; `Ok(())` and nothing done when none is.
    ///
    /// The device calls this, not [`CameraStream::set_controls`], for the V4L2 controls: they
    /// belong to the device, the session setting them need not be the one streaming (the
    /// stream is another session's, out of this one's reach), and a value set with no stream
    /// open is carried in the next [`StreamRequest`] instead. `S_PARM` still goes through the
    /// stream: the frame rate is the streaming session's own.
    fn set_controls(&mut self, controls: &[CameraControl]) -> Result<(), i32>;
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
    /// The control table and the current values: the device's, shared by every session.
    controls: Controls,
    /// `V4L2_EVENT_CTRL` subscriptions, by session id: control id to "allow feedback".
    subscriptions: HashMap<u32, HashMap<u32, bool>>,
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
        let controls = Controls::new(backend.info());
        Self {
            backend,
            evt_queue,
            mem,
            mmap_manager: MmapMappingManager::from(mapper),
            allocator,
            retired: RetiredBuffers::new(),
            active_session: None,
            controls,
            subscriptions: HashMap::new(),
        }
    }

    /// The control table, for a VMM that wants to log it.
    pub fn controls(&self) -> &Controls {
        &self.controls
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

    /// Append up to `count` buffers of `sizeimage` bytes and return how many were actually
    /// appended -- fewer than `count` when the pool ran out on the way (D73).
    ///
    /// This was all-or-nothing, and that cost a 4K capture the whole request: `libavdevice`'s
    /// v4l2 input asks `REQBUFS(count = 256)` and exposes no option to lower it, the device
    /// clamps that to `MAX_BUFFERS` (32), and 32 x 12 443 648 B = 379.75 MiB does not fit the
    /// 320 MiB `media_host` pool -- so `ffmpeg -f v4l2 -video_size 3840x2160` failed `ENOMEM` on
    /// buffer 27 while the 26 the pool could hold would have streamed, and `REQBUFS(3)` at the
    /// same size worked (`logs/vpu_wp/B14-accept-B.md` section 2). V4L2 has always allowed the
    /// shorter answer: "the driver may allocate fewer buffers than requested" and the count
    /// field comes back with what it got (`vidioc-reqbufs.rst`). vb2 is built that way --
    /// `__vb2_queue_alloc` allocates what it can, and `vb2_core_reqbufs` turns that into
    /// `-ENOMEM` only when it lands below the queue's own floor (`allocated_buffers <
    /// q->min_reqbufs_allocation`, `videobuf2-core.c:977`), `vb2_core_create_bufs` only when it
    /// is zero (`:1102`).
    ///
    /// So a refusal from the allocator ends the loop, not the request: what was allocated stays,
    /// as long as it is at least `min`. Below `min` nothing is kept -- `undo_added` gives back
    /// every buffer this call made, and the errno the allocator refused with (the `ENOMEM` whose
    /// cause the VMM's own pool-exhaustion line names) goes back to the guest exactly as before.
    /// A partial answer is made of ordinary buffers: indices `first..first + granted`, each
    /// `sizeimage` bytes, each carrying the plane `length` that bounds its mapping (review-m4
    /// R2), and the pool holds exactly `granted` of them -- nothing allocated is dropped on the
    /// floor and nothing counted is missing.
    fn add_buffers(
        &mut self,
        session: &mut CameraSession<M::GuestMemoryMapping, B::Stream>,
        memory: MemoryType,
        count: usize,
        min: usize,
        sizeimage: u32,
        what: &str,
    ) -> IoctlResult<usize> {
        let first = session.buffers.len();
        let mut added: Vec<Buffer<M::GuestMemoryMapping>> = Vec::with_capacity(count);
        // The errno that stopped the loop, if one did; kept for the below-`min` case.
        let mut stopped: Option<i32> = None;

        for index in first..first + count {
            let mut v4l2_buffer = V4l2Buffer::new(QUEUE, index as u32, memory);
            v4l2_buffer.set_field(BufferField::None);
            v4l2_buffer.set_flags(BufferFlags::TIMESTAMP_MONOTONIC);

            let backing = match memory {
                MemoryType::Mmap => {
                    let host_buffer = match self.allocator.allocate(sizeimage as u64) {
                        Ok(b) => b,
                        Err(e) => {
                            stopped = Some(e);
                            break;
                        }
                    };
                    let offset = match self.mmap_manager.register_buffer(None, sizeimage) {
                        Ok(offset) => offset,
                        Err(e) => {
                            log::error!("failed to register MMAP buffer: {:#}", e);
                            self.allocator.release(host_buffer);
                            stopped = Some(libc::EINVAL);
                            break;
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

        let granted = added.len();
        if granted < min {
            self.undo_added(added);
            return Err(stopped.unwrap_or(libc::ENOMEM));
        }
        if stopped.is_some() {
            // One line for the short answer, at `info!`: the VMM's pool logs the refusal that
            // stopped the loop as an error and says nothing about the request that survived it.
            log::info!(
                "camera: {}: asked {}, pool holds {}, granted {}",
                what,
                count,
                granted,
                granted
            );
        }
        session.buffers.extend(added);
        Ok(granted)
    }

    /// Give back what `add_buffers` allocated before it failed, or beyond what it may keep.
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

    // -----------------------------------------------------------------------------------------
    // Controls
    // -----------------------------------------------------------------------------------------

    /// `V4L2_EVENT_CTRL` for control `id` to every session subscribed to it. `from` is the
    /// session whose set caused it, which is told only if it asked for feedback; `None` sends
    /// to all, as the kernel does for a flag change or a value the hardware changed.
    fn send_control_event(&mut self, id: u32, changes: u32, from: Option<u32>) {
        let Some(desc) = self.controls.find(id) else {
            return;
        };
        let event = self.controls.event(desc, changes);
        let targets: Vec<u32> = self
            .subscriptions
            .iter()
            .filter_map(|(sid, subs)| {
                let feedback = *subs.get(&id)?;
                (from != Some(*sid) || feedback).then_some(*sid)
            })
            .collect();
        for sid in targets {
            self.evt_queue
                .send_event(V4l2Event::Event(SessionEvent::new(sid, event)));
        }
    }

    /// Store `values` (validated already), hand what changed to the camera in one submission,
    /// and tell the subscribers. `from` is the session that set them.
    ///
    /// A value that did not change is neither sent nor announced, as the kernel's control
    /// framework does; a button always executes. A camera that will not take the set (`EIO`:
    /// its capture thread is gone) leaves the values stored -- they are what the next stream
    /// opens with -- and the error goes to the guest.
    fn commit_controls(&mut self, from: u32, values: Vec<(u32, Value)>) -> IoctlResult<()> {
        let manual_before = self.controls.manual_exposure();
        let mut changed: Vec<CameraControl> = Vec::new();
        let mut announce: Vec<u32> = Vec::new();
        for (id, value) in values {
            if !self.controls.set(id, value.clone()) {
                continue;
            }
            if let Some(control) = self.controls.control_of(id, &value) {
                changed.push(control);
            }
            // A button has no value to announce.
            if self.controls.current(id).is_some() {
                announce.push(id);
            }
        }
        let result = if changed.is_empty() {
            Ok(())
        } else {
            self.backend.set_controls(&changed).map_err(|e| {
                log::error!(
                    "camera {}: the stream would not take {} control(s): errno {}",
                    self.backend.info().id,
                    changed.len(),
                    e
                );
                e
            })
        };
        for id in announce {
            self.send_control_event(id, bindings::V4L2_EVENT_CTRL_CH_VALUE, Some(from));
        }
        if manual_before != self.controls.manual_exposure() {
            for id in [
                bindings::V4L2_CID_EXPOSURE_ABSOLUTE,
                bindings::V4L2_CID_ISO_SENSITIVITY,
            ] {
                self.send_control_event(id, bindings::V4L2_EVENT_CTRL_CH_FLAGS, None);
            }
        }
        result
    }

    /// A value the running camera reported: stored, and announced if it differs.
    fn observe_control(&mut self, control: &CameraControl) {
        if let Some(id) = self.controls.observe(control) {
            self.send_control_event(id, bindings::V4L2_EVENT_CTRL_CH_VALUE, None);
        }
    }

    /// The control `QUERYCTRL` / `QUERY_EXT_CTRL` asks for, from the `id` word as the guest
    /// sent it, and the id to report: the next one after `id` under the `V4L2_CTRL_FLAG_NEXT_*`
    /// bits, the private control an old-style `V4L2_CID_PRIVATE_BASE + n` names (reported under
    /// that alias, as the kernel does), or the control itself.
    fn lookup_query(&self, raw: u32) -> IoctlResult<(&ControlDesc, u32)> {
        let next =
            raw & (bindings::V4L2_CTRL_FLAG_NEXT_CTRL | bindings::V4L2_CTRL_FLAG_NEXT_COMPOUND);
        let id = raw & bindings::V4L2_CTRL_ID_MASK;
        if next != 0 {
            let regular = next & bindings::V4L2_CTRL_FLAG_NEXT_CTRL != 0;
            let compound = next & bindings::V4L2_CTRL_FLAG_NEXT_COMPOUND != 0;
            let desc = self
                .controls
                .next(id, regular, compound)
                .ok_or(libc::EINVAL)?;
            Ok((desc, desc.id))
        } else {
            let desc = self.controls.find_legacy(id).ok_or(libc::EINVAL)?;
            Ok((desc, id))
        }
    }

    /// What every ext-controls ioctl checks first: the `which` word, the reserved words, and a
    /// `count == 0` call, which is a class probe and ends there.
    fn ext_ctrls_prelude(
        &self,
        which: CtrlWhich,
        ctrls: &mut v4l2_ext_controls,
        ctrl_array: &mut [v4l2_ext_control],
        set: bool,
    ) -> IoctlResult<Option<ExtCtrls>> {
        ctrls.error_idx = ctrls.count;
        ctrls.reserved = [0];
        for ctrl in ctrl_array.iter_mut() {
            ctrl.reserved2 = [0];
        }
        let (class, default) = match which {
            CtrlWhich::Current => (None, false),
            // The defaults can be read but not changed.
            CtrlWhich::Default if set => return Err(libc::EINVAL),
            CtrlWhich::Default => (None, true),
            CtrlWhich::Class(class) => (Some(class), false),
            CtrlWhich::Request(_) => return Err(libc::EINVAL),
        };
        if ctrl_array.is_empty() {
            // "Does this class exist": the kernel's `class_check`.
            return match class {
                Some(class) if !self.controls.has_class(class) => Err(libc::EINVAL),
                _ => Ok(None),
            };
        }
        Ok(Some(ExtCtrls { class, default }))
    }

    /// The values an `S_EXT_CTRLS` / `TRY_EXT_CTRLS` carries, each resolved and validated; a
    /// compound control's payload is read from the guest memory the ioctl named. `Err((i, e))`
    /// is the control that failed.
    fn read_ext_values(
        &self,
        class: Option<u32>,
        ctrl_array: &mut [v4l2_ext_control],
        user_regions: Vec<Vec<SgEntry>>,
    ) -> Result<Vec<(u32, Value)>, (usize, i32)> {
        let mut regions = user_regions.into_iter();
        let mut values = Vec::with_capacity(ctrl_array.len());
        for (i, ctrl) in ctrl_array.iter_mut().enumerate() {
            let fail = |e: i32| (i, e);
            // The SG list of a payload control was read in order, whatever else fails.
            let sgs = (ctrl.size > 0)
                .then(|| regions.next().ok_or(fail(libc::EINVAL)))
                .transpose()?;
            let id = ctrl.id & bindings::V4L2_CTRL_ID_MASK;
            if class.is_some_and(|class| controls::ctrl_class(id) != class) {
                return Err(fail(libc::EINVAL));
            }
            // Old-style private ids are for `G_CTRL`/`S_CTRL` only (the kernel's rule).
            if id >= controls::V4L2_CID_PRIVATE_BASE {
                return Err(fail(libc::EINVAL));
            }
            let desc = self.controls.find(id).ok_or(fail(libc::EINVAL))?;
            if desc.is_read_only() {
                return Err(fail(libc::EACCES));
            }
            let value = if desc.is_compound() {
                let needed = desc.payload_len();
                if ctrl.size < needed {
                    return Err(fail(libc::EFAULT));
                }
                ctrl.size = needed;
                let sgs = sgs.ok_or(fail(libc::EFAULT))?;
                let mapping = self.mem.new_mapping_for(sgs, false).map_err(|e| {
                    log::error!("failed to map a control payload: {:#}", e);
                    fail(guest_mapping_errno(&e))
                })?;
                Value::Array(read_words(&mapping, needed as usize).ok_or(fail(libc::EFAULT))?)
            } else {
                // A plain control travels in `value`; none of this device's is 64-bit.
                let union = ctrl.__bindgen_anon_1;
                // SAFETY: `value` is the member a plain control carries; every bit pattern is
                // a valid `i32`.
                Value::Int(unsafe { union.value } as i64)
            };
            let value = self.controls.validate(desc, &value).map_err(fail)?;
            values.push((id, value));
        }
        Ok(values)
    }

    /// `G_EXT_CTRLS`: every control checked first -- an unknown id, a write-only control, a
    /// payload buffer too small (`ENOSPC`, with the size it needs written back) -- then the
    /// values filled in, the current ones or the defaults. A payload goes into the guest
    /// memory the ioctl named.
    fn write_ext_values(
        &self,
        class: Option<u32>,
        default: bool,
        ctrl_array: &mut [v4l2_ext_control],
        user_regions: Vec<Vec<SgEntry>>,
    ) -> IoctlResult<()> {
        let mut regions = user_regions.into_iter();
        let mut planned: Vec<(&ControlDesc, Option<Vec<SgEntry>>)> =
            Vec::with_capacity(ctrl_array.len());
        for ctrl in ctrl_array.iter_mut() {
            let sgs = (ctrl.size > 0)
                .then(|| regions.next().ok_or(libc::EINVAL))
                .transpose()?;
            let id = ctrl.id & bindings::V4L2_CTRL_ID_MASK;
            if class.is_some_and(|class| controls::ctrl_class(id) != class) {
                return Err(libc::EINVAL);
            }
            if id >= controls::V4L2_CID_PRIVATE_BASE {
                return Err(libc::EINVAL);
            }
            let desc = self.controls.find(id).ok_or(libc::EINVAL)?;
            if desc.is_write_only() {
                return Err(libc::EACCES);
            }
            if desc.is_compound() {
                let needed = desc.payload_len();
                if ctrl.size < needed {
                    // "In the get case the application first queries to obtain the size."
                    ctrl.size = needed;
                    return Err(libc::ENOSPC);
                }
                ctrl.size = needed;
            }
            planned.push((desc, sgs));
        }
        for (ctrl, (desc, sgs)) in ctrl_array.iter_mut().zip(planned) {
            let value = if default {
                desc.kind.default_value()
            } else {
                self.controls
                    .current(desc.id)
                    .cloned()
                    .unwrap_or_else(|| desc.kind.default_value())
            };
            match value {
                Value::Int(v) => {
                    ctrl.__bindgen_anon_1 =
                        bindings::v4l2_ext_control__bindgen_ty_1 { value: v as i32 };
                }
                Value::Array(words) => {
                    let sgs = sgs.ok_or(libc::EFAULT)?;
                    let mut mapping = self.mem.new_mapping_for(sgs, true).map_err(|e| {
                        log::error!("failed to map a control payload: {:#}", e);
                        guest_mapping_errno(&e)
                    })?;
                    if !write_words(&mut mapping, &words) {
                        return Err(libc::EFAULT);
                    }
                }
            }
        }
        Ok(())
    }
}

/// What an ext-controls ioctl with controls in it asks for.
struct ExtCtrls {
    /// The class every control must be in, when `which` named one.
    class: Option<u32>,
    /// `V4L2_CTRL_WHICH_DEF_VAL`: the defaults rather than the current values.
    default: bool,
}

/// `bytes` bytes of a control payload as little-endian words, out of the guest memory the
/// ioctl named; `None` for a mapping too short to hold them. Copied out with a raw pointer:
/// the guest can write those pages at any time, so no slice is ever made over them.
fn read_words<GM: GuestMemoryRange>(mapping: &GM, bytes: usize) -> Option<Vec<u32>> {
    if mapping.len() < bytes {
        return None;
    }
    let mut raw = vec![0u8; bytes];
    // SAFETY: the mapping holds at least `bytes` bytes (checked), `raw` is exactly that long,
    // and a fresh `Vec` cannot overlap guest memory.
    unsafe { std::ptr::copy_nonoverlapping(mapping.as_ptr(), raw.as_mut_ptr(), bytes) };
    Some(
        raw.chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    )
}

/// `words` into a control payload in guest memory, little-endian; `false` for a mapping too
/// short to take them.
fn write_words<GM: GuestMemoryRange>(mapping: &mut GM, words: &[u32]) -> bool {
    let raw: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    if mapping.len() < raw.len() {
        return false;
    }
    // SAFETY: the mapping holds at least `raw.len()` bytes (checked) and `raw` is a `Vec` of
    // our own, so the two cannot overlap.
    unsafe { std::ptr::copy_nonoverlapping(raw.as_ptr(), mapping.as_mut_ptr(), raw.len()) };
    true
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
        self.subscriptions.remove(&session.id);
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
        // A control the camera reported is announced; the first fatal event ends the session
        // and nothing after it matters.
        for event in events {
            let why = match event {
                CameraEvent::Control(control) => {
                    self.observe_control(&control);
                    continue;
                }
                CameraEvent::Disconnected => "the camera was disconnected".to_owned(),
                CameraEvent::Error(reason) => reason,
            };
            self.end_session(session, &why);
            break;
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

    /// The format set here belongs to `session`, not to the device: virtio-media gives every
    /// `open(2)` its own session, so a second `open` of the same node keeps its own format and
    /// never sees this one. V4L2 says the format is a property of the *device*, and
    /// `v4l2-compliance`'s `testGlobalFormat` asserts exactly that -- so this is one of the
    /// three failures in the camera's 59 / 56 / 3 run, and it is **accepted, not a defect**
    /// (D22, `logs/vpu_wp/B5-acceptance.md` §4.6 and §6; `VPU_DESIGN.md` §7.1's acceptance
    /// paragraph lists all three). Moving the state up to the device would mean redesigning the
    /// whole fork's session and stream lifetime, and the per-open session is what makes "every
    /// format test is one `v4l2-ctl` invocation" work.
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
        let want = (count as usize).min(MAX_BUFFERS);
        session.memory = None;
        self.active_session = None;
        // What the guest is told it got: `want`, unless the pool could only serve some of it
        // (D73). The camera's own floor is **one** buffer -- `streamon`'s only buffer gate is
        // `session.buffers.is_empty()`, the backend is handed `session.buffers.len()` and
        // streams whatever that is, and the device promises no double buffering (it exposes no
        // `V4L2_CID_MIN_BUFFERS_FOR_CAPTURE`), so a one-buffer queue captures, slowly. Below it
        // there is no queue at all and the answer stays `ENOMEM`.
        let mut count = want;
        if want > 0 {
            let sizeimage = session.sizeimage();
            count = self.add_buffers(session, memory, want, 1, sizeimage, "REQBUFS")?;
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
        let want = (count as usize).min(MAX_BUFFERS.saturating_sub(first));
        // `index` + `count` are what the guest indexes the new buffers by, so `count` must be
        // what was really created -- vb2 answers a short `CREATE_BUFS` the same way, failing
        // only when it could make none (D73).
        let mut count = want;
        if want > 0 {
            count = self.add_buffers(session, memory, want, 1, asked, "CREATE_BUFS")?;
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
        payload: PayloadValidity,
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
        // this call's own `bytesused` / `data_offset` are ignored. Otherwise it is the queue's
        // own plane count that decides which of the guest's slots are a description at all, and
        // an `MMAP` capture buffer has no guest description to check: this device reports the
        // payload (D21, `ioctl::PayloadValidity`).
        let prepared = entry.prepared;
        if prepared.is_none()
            && !payload.is_accepted_by(QUEUE.direction(), buffer.memory(), NUM_PLANES)
        {
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
        payload: PayloadValidity,
    ) -> IoctlResult<V4l2Buffer> {
        // The same rule `qbuf` applies, with no prepared description to fall back on.
        if !payload.is_accepted_by(QUEUE.direction(), buffer.memory(), NUM_PLANES) {
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
            controls: self.controls.all_settable(),
        };
        let sink = CaptureSink(Arc::clone(&session.signal));
        let (width, height, fps) = (request.width, request.height, request.fps);
        let stream = self.backend.open_stream(request, sink).map_err(|e| {
            log::error!(
                "camera {}: cannot open a {}x{} stream at {:?} fps: errno {}",
                id,
                width,
                height,
                fps,
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

    // -----------------------------------------------------------------------------------------
    // Controls: see `controls.rs` for the table and the units.
    // -----------------------------------------------------------------------------------------

    fn queryctrl_raw(&mut self, _session: &Self::Session, id: u32) -> IoctlResult<v4l2_queryctrl> {
        let (desc, id) = self.lookup_query(id)?;
        Ok(self.controls.query(desc, id))
    }

    fn query_ext_ctrl_raw(
        &mut self,
        _session: &Self::Session,
        id: u32,
    ) -> IoctlResult<v4l2_query_ext_ctrl> {
        let (desc, id) = self.lookup_query(id)?;
        Ok(self.controls.query_ext(desc, id))
    }

    fn querymenu(
        &mut self,
        _session: &Self::Session,
        id: u32,
        index: u32,
    ) -> IoctlResult<v4l2_querymenu> {
        self.controls.menu_item(id, index)
    }

    /// `G_CTRL`: the plain controls only, as the kernel's `is_int`; a write-only one is
    /// `EACCES`. An old-style `V4L2_CID_PRIVATE_BASE + n` id names its control here too, and
    /// the answer carries the id as asked (D33).
    fn g_ctrl(&mut self, _session: &Self::Session, id: u32) -> IoctlResult<v4l2_control> {
        let desc = self.controls.find_legacy(id).ok_or(libc::EINVAL)?;
        if !desc.is_int() {
            return Err(libc::EINVAL);
        }
        if desc.is_write_only() {
            return Err(libc::EACCES);
        }
        let value = self.controls.current(desc.id).map(Value::int).unwrap_or(0);
        Ok(v4l2_control {
            id,
            value: value as i32,
        })
    }

    /// `S_CTRL`: validated against the range, stored, sent to the camera and announced, like a
    /// one-control `S_EXT_CTRLS`. Resolves an old-style private alias, as `G_CTRL` does (D33);
    /// the control is set under its real id and the answer carries the id as asked.
    fn s_ctrl(
        &mut self,
        session: &mut Self::Session,
        id: u32,
        value: i32,
    ) -> IoctlResult<v4l2_control> {
        let desc = self.controls.find_legacy(id).ok_or(libc::EINVAL)?;
        if !desc.is_int() {
            return Err(libc::EINVAL);
        }
        if desc.is_read_only() {
            return Err(libc::EACCES);
        }
        let real_id = desc.id;
        let value = self.controls.validate(desc, &Value::Int(value as i64))?;
        let reply = value.int() as i32;
        self.commit_controls(session.id, vec![(real_id, value)])?;
        Ok(v4l2_control { id, value: reply })
    }

    /// `G_EXT_CTRLS`. `error_idx` is `count` on every failure, as the kernel leaves it for a get.
    fn g_ext_ctrls(
        &mut self,
        _session: &Self::Session,
        which: CtrlWhich,
        ctrls: &mut v4l2_ext_controls,
        ctrl_array: &mut Vec<v4l2_ext_control>,
        user_regions: Vec<Vec<SgEntry>>,
    ) -> IoctlResult<()> {
        let Some(ExtCtrls { class, default }) =
            self.ext_ctrls_prelude(which, ctrls, ctrl_array, false)?
        else {
            return Ok(());
        };
        self.write_ext_values(class, default, ctrl_array, user_regions)
    }

    /// `S_EXT_CTRLS`: every control validated before any is applied, then all of them stored
    /// and handed to the camera in one submission. A failure applies nothing and leaves
    /// `error_idx` at `count` (the whole set failed, as the kernel reports a set); the reply
    /// carries the values as stored.
    fn s_ext_ctrls(
        &mut self,
        session: &mut Self::Session,
        which: CtrlWhich,
        ctrls: &mut v4l2_ext_controls,
        ctrl_array: &mut Vec<v4l2_ext_control>,
        user_regions: Vec<Vec<SgEntry>>,
    ) -> IoctlResult<()> {
        let Some(ExtCtrls { class, .. }) =
            self.ext_ctrls_prelude(which, ctrls, ctrl_array, true)?
        else {
            return Ok(());
        };
        let values = self
            .read_ext_values(class, ctrl_array, user_regions)
            .map_err(|(_, e)| e)?;
        for (ctrl, (_, value)) in ctrl_array.iter_mut().zip(values.iter()) {
            if let Value::Int(v) = value {
                ctrl.__bindgen_anon_1 =
                    bindings::v4l2_ext_control__bindgen_ty_1 { value: *v as i32 };
            }
        }
        self.commit_controls(session.id, values)
    }

    /// `TRY_EXT_CTRLS`: the validation of `S_EXT_CTRLS` and nothing else; `error_idx` names
    /// the control refused.
    fn try_ext_ctrls(
        &mut self,
        _session: &Self::Session,
        which: CtrlWhich,
        ctrls: &mut v4l2_ext_controls,
        ctrl_array: &mut Vec<v4l2_ext_control>,
        user_regions: Vec<Vec<SgEntry>>,
    ) -> IoctlResult<()> {
        let Some(ExtCtrls { class, .. }) =
            self.ext_ctrls_prelude(which, ctrls, ctrl_array, true)?
        else {
            return Ok(());
        };
        let values = self
            .read_ext_values(class, ctrl_array, user_regions)
            .map_err(|(i, e)| {
                ctrls.error_idx = i as u32;
                e
            })?;
        for (ctrl, (_, value)) in ctrl_array.iter_mut().zip(values.iter()) {
            if let Value::Int(v) = value {
                ctrl.__bindgen_anon_1 =
                    bindings::v4l2_ext_control__bindgen_ty_1 { value: *v as i32 };
            }
        }
        Ok(())
    }

    /// `EOS` and `SOURCE_CHANGE` are accepted and never emitted (a camera has neither).
    /// `V4L2_EVENT_CTRL` is per control id; with `SEND_INITIAL` the current value and flags go
    /// out at once, for every control but a class (the kernel's `v4l2_ctrl_add_event`).
    fn subscribe_event(
        &mut self,
        session: &mut Self::Session,
        event: EventType,
        flags: SubscribeEventFlags,
    ) -> IoctlResult<()> {
        match event {
            EventType::Eos | EventType::SourceChange(0) => Ok(()),
            EventType::Ctrl(id) => {
                let (is_class, write_only) = {
                    let desc = self.controls.find(id).ok_or(libc::EINVAL)?;
                    (desc.kind == Kind::Class, desc.is_write_only())
                };
                self.subscriptions
                    .entry(session.id)
                    .or_default()
                    .insert(id, flags.contains(SubscribeEventFlags::ALLOW_FEEDBACK));
                if flags.contains(SubscribeEventFlags::SEND_INITIAL) && !is_class {
                    let mut changes = bindings::V4L2_EVENT_CTRL_CH_FLAGS;
                    if !write_only {
                        changes |= bindings::V4L2_EVENT_CTRL_CH_VALUE;
                    }
                    if let Some(desc) = self.controls.find(id) {
                        let event = self.controls.event(desc, changes);
                        self.evt_queue
                            .send_event(V4l2Event::Event(SessionEvent::new(session.id, event)));
                    }
                }
                Ok(())
            }
            _ => Err(libc::EINVAL),
        }
    }

    /// `V4L2_EVENT_ALL` drops every subscription of the session; a control id drops that one,
    /// subscribed or not (the kernel answers 0 either way).
    fn unsubscribe_event(
        &mut self,
        session: &mut Self::Session,
        event: v4l2_event_subscription,
    ) -> IoctlResult<()> {
        if event.type_ == bindings::V4L2_EVENT_ALL {
            self.subscriptions.remove(&session.id);
            return Ok(());
        }
        match EventType::try_from(&event) {
            Ok(EventType::Eos) | Ok(EventType::SourceChange(0)) => Ok(()),
            Ok(EventType::Ctrl(id)) => {
                if let Some(subs) = self.subscriptions.get_mut(&session.id) {
                    subs.remove(&id);
                }
                Ok(())
            }
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
    use crate::devices::test_pool::PoolBudget;
    use crate::ioctl::ffmpeg_wire;
    use crate::poll::SessionPoller;
    use crate::protocol::VIRTIO_MEDIA_CMD_IOCTL;
    use crate::protocol::VIRTIO_MEDIA_CMD_OPEN;
    use crate::MemFdAllocator;
    use crate::RespHeader;
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
        /// Controls applied to an open stream through it (`S_PARM`).
        controls: Vec<CameraControl>,
        /// Every `CameraBackend::set_controls`, with whether a stream was open at the time.
        backend_controls: Vec<(bool, Vec<CameraControl>)>,
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
        /// The way to the open stream's thread, for `set_controls`; `None` before the first
        /// stream, stale (and harmlessly so: the send fails) after one closes.
        stream_commands: Option<mpsc::Sender<FakeCommand>>,
    }

    enum FakeCommand {
        Lend(EmptyBuffer),
        /// V4L2 controls, through the backend. The thread plays a small script for an AF
        /// trigger -- the states a real scan reports, with repeats, and a lens id the camera
        /// never listed -- so the device's throttling and lookups are exercised.
        Controls(Vec<CameraControl>),
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
            let (width, height) = (request.width, request.height);
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
            let thread = thread::spawn(move || {
                let mut sequence = 0u32;
                for command in rx {
                    let buffer = match command {
                        FakeCommand::Lend(buffer) => buffer,
                        FakeCommand::Controls(controls) => {
                            for control in controls {
                                let script: &[CameraEvent] = match control {
                                    CameraControl::AfTrigger(AfTrigger::Start) => &[
                                        CameraEvent::Control(CameraControl::AfStatus(
                                            AF_STATUS_BUSY,
                                        )),
                                        CameraEvent::Control(CameraControl::AeState(
                                            AeState::Searching,
                                        )),
                                        CameraEvent::Control(CameraControl::AfStatus(
                                            AF_STATUS_REACHED,
                                        )),
                                        CameraEvent::Control(CameraControl::AfStatus(
                                            AF_STATUS_REACHED,
                                        )),
                                        CameraEvent::Control(CameraControl::ActivePhysicalId(
                                            "4".into(),
                                        )),
                                        CameraEvent::Control(CameraControl::ActivePhysicalId(
                                            "4".into(),
                                        )),
                                        CameraEvent::Control(CameraControl::ActivePhysicalId(
                                            "nope".into(),
                                        )),
                                    ],
                                    CameraControl::AfTrigger(AfTrigger::Cancel) => {
                                        &[CameraEvent::Control(CameraControl::AfStatus(
                                            AF_STATUS_IDLE,
                                        ))]
                                    }
                                    _ => &[],
                                };
                                for event in script {
                                    let _ = events_tx.send(event.clone());
                                }
                                if !script.is_empty() {
                                    sink.signal();
                                }
                            }
                            continue;
                        }
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
            self.stream_commands = Some(commands.clone());
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

        fn set_controls(&mut self, controls: &[CameraControl]) -> Result<(), i32> {
            {
                let mut log = self.log.lock().unwrap();
                let streaming = log.streaming;
                log.backend_controls.push((streaming, controls.to_vec()));
            }
            if let Some(commands) = &self.stream_commands {
                // A closed stream's thread is gone: nothing to apply to, not an error.
                let _ = commands.send(FakeCommand::Controls(controls.to_vec()));
            }
            Ok(())
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

    /// Counts what the device gives back, and when, over a pool of a size the test chooses.
    struct OrderedAllocator {
        inner: MemFdAllocator,
        log: SharedLog,
        released: Rc<RefCell<usize>>,
        /// The `media_host` pool this allocator carves from: unlimited unless a test sizes it
        /// (D73).
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
        pool: Rc<PoolBudget>,
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
            // The 5566 back camera's controls, near enough: 0.67x-20x zoom, every AF mode, a
            // flash, manual exposure from 85 µs to 1 s, ISO 100-16000, ±4 EV in sixths.
            zoom_range: Some((67, 2000)),
            af_modes: vec![
                AfMode::Off,
                AfMode::Auto,
                AfMode::Macro,
                AfMode::ContinuousVideo,
                AfMode::ContinuousPicture,
            ],
            flash: true,
            ae_modes: vec![
                AeMode::Off,
                AeMode::On,
                AeMode::OnAutoFlash,
                AeMode::OnAlwaysFlash,
            ],
            exposure_range_ns: Some((85_000, 1_000_000_000)),
            iso_range: Some((100, 16_000)),
            exposure_bias: Some(ExposureBias {
                min: -24,
                max: 24,
                step_num: 1,
                step_den: 6,
            }),
            awb_modes: vec![
                WhiteBalance::Auto,
                WhiteBalance::Incandescent,
                WhiteBalance::Fluorescent,
                WhiteBalance::FluorescentH,
                WhiteBalance::Daylight,
                WhiteBalance::Cloudy,
                WhiteBalance::Horizon,
                WhiteBalance::Shade,
            ],
            antibanding: vec![
                PowerLine::Disabled,
                PowerLine::Hz50,
                PowerLine::Hz60,
                PowerLine::Auto,
            ],
            effects: vec![
                ColorEffect::None,
                ColorEffect::BlackWhite,
                ColorEffect::Negative,
                ColorEffect::Sepia,
                ColorEffect::Aqua,
                ColorEffect::Solarization,
            ],
            scenes: vec![
                SceneMode::None,
                SceneMode::Portrait,
                SceneMode::Landscape,
                SceneMode::Night,
                SceneMode::Sports,
                SceneMode::Fireworks,
            ],
            stabilization: true,
            max_regions: MaxRegions {
                ae: 1,
                awb: 0,
                af: 1,
            },
            physical_ids: vec!["3".into(), "2".into(), "4".into()],
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
        let pool = PoolBudget::unlimited();
        let device = CameraDevice::new(
            camera,
            events,
            guest.clone(),
            FakeHostMapper,
            OrderedAllocator {
                inner: MemFdAllocator::new(),
                log: Arc::clone(&log),
                released: Rc::clone(&released),
                pool: Rc::clone(&pool),
            },
        );
        Rig {
            device,
            events: events_log,
            guest,
            log,
            released,
            pool,
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
            stream_commands: None,
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
                .qbuf(
                    &mut s,
                    mmap_buffer(index, size),
                    vec![],
                    PayloadValidity::ALL,
                )
                .unwrap();
            assert!(reply.flags().contains(BufferFlags::QUEUED));
        }
        assert!(r.log.lock().unwrap().opened.is_empty());
        // Queueing a queued buffer is refused.
        assert_eq!(
            r.device
                .qbuf(&mut s, mmap_buffer(0, size), vec![], PayloadValidity::ALL)
                .err(),
            Some(libc::EINVAL)
        );

        r.device.streamon(&mut s, QUEUE).unwrap();
        {
            let log = r.log.lock().unwrap();
            assert_eq!(log.opened.len(), 1);
            let opened = &log.opened[0];
            assert_eq!(
                (opened.width, opened.height, opened.fps, opened.buffers),
                (64, 48, (7, 30), 3)
            );
        }
        // STREAMON twice is fine and opens nothing more.
        r.device.streamon(&mut s, QUEUE).unwrap();
        assert_eq!(r.log.lock().unwrap().opened.len(), 1);

        // The third buffer, queued while streaming, is lent straight away.
        r.device
            .qbuf(&mut s, mmap_buffer(2, size), vec![], PayloadValidity::ALL)
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
            .qbuf(&mut s, mmap_buffer(0, size), vec![], PayloadValidity::ALL)
            .unwrap();
        let frames = collect_frames(&mut r, &mut s, 4);
        assert_eq!(frames[3].index(), 0);
        assert_eq!(frames[3].sequence(), 3);

        // STREAMOFF: the camera is closed (joined) first, then the buffers are given back.
        r.device
            .qbuf(&mut s, mmap_buffer(1, size), vec![], PayloadValidity::ALL)
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
            .qbuf(&mut s, mmap_buffer(2, size), vec![], PayloadValidity::ALL)
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
            .qbuf(&mut s, mmap_buffer(0, size), vec![], PayloadValidity::ALL)
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
            .qbuf(&mut s, mmap_buffer(1, size), vec![], PayloadValidity::ALL)
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
            r.device
                .prepare_buf(&mut s, buf, vec![], PayloadValidity::ALL.without(0))
                .err(),
            Some(libc::EINVAL)
        );
        for len in [size - 1, size + 1] {
            let (buf, _) = userptr_buffer(0, gpa, len);
            assert_eq!(
                r.device
                    .prepare_buf(&mut s, buf, vec![], PayloadValidity::ALL)
                    .err(),
                Some(libc::EINVAL)
            );
        }

        let (buf, _) = userptr_buffer(0, gpa, size);
        let reply = r
            .device
            .prepare_buf(&mut s, buf, vec![], PayloadValidity::ALL)
            .unwrap();
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
            r.device
                .prepare_buf(&mut s, buf, vec![], PayloadValidity::ALL)
                .err(),
            Some(libc::EINVAL)
        );

        // The blocker: queueing the prepared buffer with a plane the guest has shrunk. The
        // prepared length says a frame fits; this call's does not, and this call's is the one
        // the scatter list was read against.
        let (buf, sgs) = userptr_buffer(0, gpa, 8);
        assert_eq!(
            r.device.qbuf(&mut s, buf, sgs, PayloadValidity::ALL).err(),
            Some(libc::EINVAL)
        );
        assert_eq!(*r.guest.live_mappings.borrow(), 0, "nothing was mapped");
        assert!(!s.buffers[0].queued);
        assert!(s.buffers[0].prepared.is_some(), "still prepared");

        // Queued honestly it works, and the payload PREPARE_BUF accepted is what comes back.
        let (buf, sgs) = userptr_buffer(0, gpa, size);
        let reply = r
            .device
            .qbuf(&mut s, buf, sgs, PayloadValidity::ALL.without(0))
            .unwrap();
        assert!(reply.flags().contains(BufferFlags::QUEUED));
        assert!(!reply.flags().contains(BufferFlags::PREPARED));
        assert_eq!(*reply.get_first_plane().length, size);
        assert_eq!(*r.guest.live_mappings.borrow(), 1);

        r.device.streamon(&mut s, QUEUE).unwrap();
        let frames = collect_frames(&mut r, &mut s, 1);
        assert_eq!(frames[0].index(), 0);
        close(&mut r.device, s);
    }

    /// D21 -- the exact `QBUF` `ffmpeg -f v4l2` sends, replayed on the wire.
    ///
    /// ffmpeg declares `length = VIDEO_MAX_PLANES` and fills only `planes[0]` from `QUERYBUF`;
    /// `planes[1..8]` are its own stack (`logs/vpu_wp/B5-acceptance.md` §4.3). Judging all eight
    /// slots refused every capture buffer it queued, so `ffmpeg -f v4l2 -i /dev/video0 -t 5`
    /// produced no file at all. This queue's format has one plane, and vb2 looks at
    /// `vb->num_planes` entries only.
    #[test]
    fn ffmpegs_dirty_plane_array_queues_a_capture_buffer() {
        let mut r = rig();
        let mut s = session(&mut r.device);
        r.device.s_fmt(&mut s, QUEUE, format(1280, 720)).unwrap();
        let sizeimage = s.sizeimage();
        r.device
            .reqbufs(&mut s, QUEUE, MemoryType::Mmap, 2)
            .unwrap();

        let bytes = ffmpeg_wire::ffmpeg_qbuf_bytes(QUEUE, MemoryType::Mmap, 0, (0, sizeimage));
        assert_eq!(
            ffmpeg_wire::dispatch_qbuf(&mut r.device, &mut s, &bytes),
            0,
            "ffmpeg's plane array was refused (D21)"
        );
        assert!(s.buffers[0].queued);

        // On an `MMAP` capture buffer the payload is this device's to report, so even a plane 0
        // the guest made nonsensical is ignored rather than refused -- `__verify_length` returns
        // before it looks (`videobuf2-v4l2.c:101`). What comes back is the device's own zero.
        let bytes = ffmpeg_wire::ffmpeg_qbuf_bytes(QUEUE, MemoryType::Mmap, 1, (sizeimage + 1, 8));
        assert_eq!(ffmpeg_wire::dispatch_qbuf(&mut r.device, &mut s, &bytes), 0);
        assert_eq!(
            *s.buffers[1].v4l2_buffer.get_first_plane().bytesused,
            0,
            "the device reports the payload, not the guest"
        );

        // A capture buffer whose pages the guest lends is held to the stricter rule this device
        // keeps: there is no vb2 underneath to re-check the description it sent.
        close(&mut r.device, s);
        let mut s = session(&mut r.device);
        r.device.s_fmt(&mut s, QUEUE, format(64, 48)).unwrap();
        let size = s.sizeimage();
        r.device
            .reqbufs(&mut s, QUEUE, MemoryType::UserPtr, 1)
            .unwrap();
        let (buf, sgs) = userptr_buffer(0, 4 * 0x1000, size);
        assert_eq!(
            r.device
                .qbuf(&mut s, buf, sgs, PayloadValidity::ALL.without(0))
                .err(),
            Some(libc::EINVAL)
        );
        // The slots past the one plane the format has are still scratch, even there.
        let (buf, sgs) = userptr_buffer(0, 4 * 0x1000, size);
        r.device
            .qbuf(&mut s, buf, sgs, PayloadValidity::ALL.without(1).without(7))
            .unwrap();
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
        r.device
            .qbuf(&mut s, buf, sgs, PayloadValidity::ALL)
            .unwrap();
        let (buf, sgs) = userptr_buffer(0, gpa, size);
        r.device
            .qbuf(&mut s, buf, sgs, PayloadValidity::ALL)
            .unwrap();
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
            r.device
                .qbuf(&mut s, buf, sgs, PayloadValidity::ALL)
                .unwrap();
        }
        r.device.streamon(&mut s, QUEUE).unwrap();
        let _ = collect_frames(&mut r, &mut s, 2);
        assert_eq!(r.log.lock().unwrap().lent, vec![0, 1]);

        // Now the same short mapping while the queue streams: the session ends.
        let (buf, sgs) = userptr_buffer_sized(1, gpa + 0x1_0000, size, size / 4);
        assert_eq!(
            r.device.qbuf(&mut s, buf, sgs, PayloadValidity::ALL).err(),
            Some(libc::EIO)
        );
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
            r.device
                .qbuf(&mut s, buf, sgs, PayloadValidity::ALL)
                .unwrap();
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
                    .qbuf(s, mmap_buffer(index, size), vec![], PayloadValidity::ALL)
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
                r.device.qbuf(&mut s, buf, sgs, PayloadValidity::ALL).err(),
                Some(libc::EINVAL)
            );
        }
        assert_eq!(*r.guest.live_mappings.borrow(), 0);
        // The memory type is fixed by REQBUFS.
        assert_eq!(
            r.device
                .qbuf(&mut s, mmap_buffer(0, size), vec![], PayloadValidity::ALL)
                .err(),
            Some(libc::EINVAL)
        );

        let (buf, sgs) = userptr_buffer(0, gpa, size);
        r.device
            .qbuf(&mut s, buf, sgs, PayloadValidity::ALL)
            .unwrap();
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
        r.device
            .qbuf(&mut s, buf, sgs, PayloadValidity::ALL)
            .unwrap();
        let _ = collect_frames(&mut r, &mut s, 2);
        let (buf, sgs) = userptr_buffer(0, gpa, size);
        r.device
            .qbuf(&mut s, buf, sgs, PayloadValidity::ALL)
            .unwrap();
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
                .qbuf(
                    &mut s,
                    mmap_buffer(index, size),
                    vec![],
                    PayloadValidity::ALL,
                )
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
                .qbuf(&mut s, mmap_buffer(2, size), vec![], PayloadValidity::ALL)
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
            r.device
                .qbuf(&mut s, buf, sgs, PayloadValidity::ALL)
                .unwrap();
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
            .qbuf(&mut s, mmap_buffer(0, size), vec![], PayloadValidity::ALL)
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
                .prepare_buf(&mut s, mmap_buffer(0, size), vec![], PayloadValidity::ALL)
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
                r.device.qbuf(&mut s, buf, sgs, PayloadValidity::ALL).err(),
                Some(libc::EINVAL)
            );
        }
        let (buf, sgs) = userptr_buffer(1, gpa, big);
        r.device
            .qbuf(&mut s, buf, sgs, PayloadValidity::ALL)
            .unwrap();
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
            .qbuf(&mut s, mmap_buffer(0, size), vec![], PayloadValidity::ALL)
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

    // -----------------------------------------------------------------------------------------
    // Controls
    // -----------------------------------------------------------------------------------------

    fn ext_controls(count: usize) -> v4l2_ext_controls {
        v4l2_ext_controls {
            count: count as u32,
            error_idx: 0xdead,
            ..Default::default()
        }
    }

    fn ext_control(id: u32, value: i32) -> v4l2_ext_control {
        v4l2_ext_control {
            id,
            size: 0,
            reserved2: [0xdead],
            __bindgen_anon_1: bindings::v4l2_ext_control__bindgen_ty_1 { value },
        }
    }

    fn ext_payload(id: u32, size: u32) -> v4l2_ext_control {
        v4l2_ext_control {
            id,
            size,
            ..Default::default()
        }
    }

    fn ctrl_value(ctrl: &v4l2_ext_control) -> i32 {
        let union = ctrl.__bindgen_anon_1;
        // SAFETY: every control these tests set or read is a plain one.
        unsafe { union.value }
    }

    fn menu_name(qm: &v4l2_querymenu) -> String {
        let union = qm.__bindgen_anon_1;
        // SAFETY: the tests ask this only of a MENU item.
        let name = unsafe { union.name };
        let end = name.iter().position(|&b| b == 0).unwrap_or(name.len());
        String::from_utf8_lossy(&name[..end]).into_owned()
    }

    fn menu_value(qm: &v4l2_querymenu) -> i64 {
        let union = qm.__bindgen_anon_1;
        // SAFETY: the tests ask this only of an INTEGER_MENU item.
        unsafe { union.value }
    }

    fn qc_name(qc: &v4l2_query_ext_ctrl) -> String {
        let bytes: Vec<u8> = qc
            .name
            .iter()
            .take_while(|&&c| c != 0)
            .map(|&c| c as u8)
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// The `QUERY_EXT_CTRL` walk under `flags` (`V4L2_CTRL_FLAG_NEXT_*`), as v4l2-ctl and
    /// v4l2-compliance do it.
    fn walk(device: &mut Device, session: &Session, flags: u32) -> Vec<v4l2_query_ext_ctrl> {
        let mut out = Vec::new();
        let mut id = 0;
        while let Ok(qc) = device.query_ext_ctrl_raw(session, id | flags) {
            assert!(qc.id > id, "{:#x} after {:#x}", qc.id, id);
            id = qc.id;
            out.push(qc);
        }
        out
    }

    /// A control event as the guest would see it: which session, which control, what changed.
    #[derive(Debug, PartialEq, Eq)]
    struct CtrlEv {
        session: u32,
        id: u32,
        changes: u32,
        value: i64,
        flags: u32,
    }

    fn ctrl_events(events: &[V4l2Event]) -> Vec<CtrlEv> {
        events
            .iter()
            .filter_map(|e| match e {
                V4l2Event::Event(se) if se.event().type_ == bindings::V4L2_EVENT_CTRL => {
                    let ev = se.event();
                    // SAFETY: a CTRL event carries `ctrl`.
                    let ctrl = unsafe { ev.u.ctrl };
                    let union = ctrl.__bindgen_anon_1;
                    // SAFETY: as above.
                    let value = unsafe { union.value64 };
                    Some(CtrlEv {
                        session: se.hdr.session_id(),
                        id: ev.id,
                        changes: ctrl.changes,
                        value,
                        flags: ctrl.flags,
                    })
                }
                _ => None,
            })
            .collect()
    }

    const NEXT: u32 = bindings::V4L2_CTRL_FLAG_NEXT_CTRL;
    const NEXT_COMPOUND: u32 = bindings::V4L2_CTRL_FLAG_NEXT_COMPOUND;
    const CH_VALUE: u32 = bindings::V4L2_EVENT_CTRL_CH_VALUE;
    const CH_FLAGS: u32 = bindings::V4L2_EVENT_CTRL_CH_FLAGS;

    /// The table as `v4l2-compliance` walks it: ids ascending, a class control ahead of every
    /// class, the plain walk and the compound walk partitioning it, every control answering
    /// by its own id and the old private aliases answering for the private integer ones; the
    /// ranges and menus the design's conventions give a phone-shaped camera.
    #[test]
    fn controls_enumerate_as_v4l2_compliance_walks_them() {
        let mut r = rig();
        let s = session(&mut r.device);

        let all = walk(&mut r.device, &s, NEXT | NEXT_COMPOUND);
        let ids: Vec<u32> = all.iter().map(|q| q.id).collect();
        assert_eq!(
            ids,
            vec![
                bindings::V4L2_CID_USER_CLASS,
                bindings::V4L2_CID_POWER_LINE_FREQUENCY,
                bindings::V4L2_CID_COLORFX,
                VCAM_CID_ACTIVE_PHYSICAL_ID,
                VCAM_CID_AE_STATE,
                VCAM_CID_AE_REGIONS,
                VCAM_CID_AF_REGIONS,
                bindings::V4L2_CID_CAMERA_CLASS,
                bindings::V4L2_CID_EXPOSURE_AUTO,
                bindings::V4L2_CID_EXPOSURE_ABSOLUTE,
                bindings::V4L2_CID_FOCUS_AUTO,
                bindings::V4L2_CID_ZOOM_ABSOLUTE,
                bindings::V4L2_CID_AUTO_EXPOSURE_BIAS,
                bindings::V4L2_CID_AUTO_N_PRESET_WHITE_BALANCE,
                bindings::V4L2_CID_IMAGE_STABILIZATION,
                bindings::V4L2_CID_ISO_SENSITIVITY,
                bindings::V4L2_CID_ISO_SENSITIVITY_AUTO,
                bindings::V4L2_CID_SCENE_MODE,
                bindings::V4L2_CID_AUTO_FOCUS_START,
                bindings::V4L2_CID_AUTO_FOCUS_STOP,
                bindings::V4L2_CID_AUTO_FOCUS_STATUS,
                bindings::V4L2_CID_FLASH_CLASS,
                bindings::V4L2_CID_FLASH_LED_MODE,
            ]
        );
        // The private ids are where the design puts them, in the USER class.
        assert_eq!(VCAM_CID_BASE, 0x0098_0900 + 0x1200);
        assert_eq!(
            controls::ctrl_class(VCAM_CID_AF_REGIONS),
            bindings::V4L2_CTRL_CLASS_USER
        );
        assert!(controls::is_driver_private(VCAM_CID_AF_REGIONS));

        // A class control per class, with the flags and zeroed range compliance checks.
        for qc in all
            .iter()
            .filter(|q| q.type_ == bindings::v4l2_ctrl_type_V4L2_CTRL_TYPE_CTRL_CLASS)
        {
            assert_eq!(qc.id & 0xffff, 1);
            assert_eq!(
                qc.flags,
                bindings::V4L2_CTRL_FLAG_READ_ONLY | bindings::V4L2_CTRL_FLAG_WRITE_ONLY
            );
            assert_eq!(
                (qc.minimum, qc.maximum, qc.step, qc.default_value),
                (0, 0, 0, 0)
            );
        }
        assert_eq!(qc_name(&all[0]), "User Controls");
        assert_eq!(qc_name(&all[7]), "Camera Controls");
        assert_eq!(qc_name(&all[21]), "Flash Controls");

        // The plain walk and the compound walk partition the table.
        let plain: Vec<u32> = walk(&mut r.device, &s, NEXT).iter().map(|q| q.id).collect();
        let compound: Vec<u32> = walk(&mut r.device, &s, NEXT_COMPOUND)
            .iter()
            .map(|q| q.id)
            .collect();
        assert_eq!(plain.len(), 21);
        assert_eq!(compound, vec![VCAM_CID_AE_REGIONS, VCAM_CID_AF_REGIONS]);
        assert!(plain.iter().all(|id| !compound.contains(id)));

        // Every control answers by its own id, and nothing else does.
        for id in &ids {
            assert_eq!(r.device.query_ext_ctrl_raw(&s, *id).unwrap().id, *id);
            assert_eq!(r.device.queryctrl_raw(&s, *id).unwrap().id, *id);
        }
        for id in [
            0,
            bindings::V4L2_CID_BRIGHTNESS,
            bindings::V4L2_CID_EXPOSURE_METERING,
            VCAM_CID_AWB_REGIONS,
            VCAM_CID_BASE + 15,
        ] {
            assert_eq!(
                r.device.query_ext_ctrl_raw(&s, id).err(),
                Some(libc::EINVAL)
            );
            assert_eq!(r.device.queryctrl_raw(&s, id).err(), Some(libc::EINVAL));
        }
        // The old-style private aliases: the n-th private USER-class integer control,
        // reported under the alias, as the kernel's `find_private_ref` resolves them.
        let alias = r
            .device
            .query_ext_ctrl_raw(&s, controls::V4L2_CID_PRIVATE_BASE)
            .unwrap();
        assert_eq!(alias.id, controls::V4L2_CID_PRIVATE_BASE);
        assert_eq!(qc_name(&alias), "Active Physical Camera");
        let alias = r
            .device
            .queryctrl_raw(&s, controls::V4L2_CID_PRIVATE_BASE + 1)
            .unwrap();
        assert_eq!(alias.id, controls::V4L2_CID_PRIVATE_BASE + 1);
        assert_eq!(alias.type_, bindings::v4l2_ctrl_type_V4L2_CTRL_TYPE_MENU);
        // The regions are not integer controls, so the aliases stop there.
        assert_eq!(
            r.device
                .query_ext_ctrl_raw(&s, controls::V4L2_CID_PRIVATE_BASE + 2)
                .err(),
            Some(libc::EINVAL)
        );

        // Ranges and menus, by the design's conventions.
        let zoom = r
            .device
            .query_ext_ctrl_raw(&s, bindings::V4L2_CID_ZOOM_ABSOLUTE)
            .unwrap();
        assert_eq!(zoom.type_, bindings::v4l2_ctrl_type_V4L2_CTRL_TYPE_INTEGER);
        assert_eq!(
            (zoom.minimum, zoom.maximum, zoom.step, zoom.default_value),
            (67, 2000, 1, 100)
        );
        assert_eq!(qc_name(&zoom), "Zoom, Absolute");
        assert_eq!(zoom.flags, 0);
        let exposure = r
            .device
            .query_ext_ctrl_raw(&s, bindings::V4L2_CID_EXPOSURE_ABSOLUTE)
            .unwrap();
        // 85 µs .. 1 s in 100 µs units: 1 .. 10000, and inactive while AE is on.
        assert_eq!(
            (exposure.minimum, exposure.maximum, exposure.default_value),
            (1, 10_000, 333)
        );
        assert_eq!(exposure.flags, bindings::V4L2_CTRL_FLAG_INACTIVE);
        let iso = r
            .device
            .query_ext_ctrl_raw(&s, bindings::V4L2_CID_ISO_SENSITIVITY)
            .unwrap();
        assert_eq!(
            iso.type_,
            bindings::v4l2_ctrl_type_V4L2_CTRL_TYPE_INTEGER_MENU
        );
        assert_eq!(
            (iso.minimum, iso.maximum, iso.step, iso.default_value),
            (0, 8, 1, 0)
        );
        let ladder: Vec<i64> = (0..=8)
            .map(|i| {
                menu_value(
                    &r.device
                        .querymenu(&s, bindings::V4L2_CID_ISO_SENSITIVITY, i)
                        .unwrap(),
                )
            })
            .collect();
        assert_eq!(
            ladder,
            vec![100, 200, 400, 800, 1600, 3200, 6400, 12800, 16000]
        );
        assert_eq!(
            r.device
                .querymenu(&s, bindings::V4L2_CID_ISO_SENSITIVITY, 9)
                .err(),
            Some(libc::EINVAL)
        );
        let bias = r
            .device
            .query_ext_ctrl_raw(&s, bindings::V4L2_CID_AUTO_EXPOSURE_BIAS)
            .unwrap();
        // -24..=24 sixths of an EV: 49 items in 0.001 EV, no compensation in the middle.
        assert_eq!(
            (bias.minimum, bias.maximum, bias.default_value),
            (0, 48, 24)
        );
        let mut item = |i| {
            menu_value(
                &r.device
                    .querymenu(&s, bindings::V4L2_CID_AUTO_EXPOSURE_BIAS, i)
                    .unwrap(),
            )
        };
        assert_eq!(
            (item(0), item(24), item(25), item(48)),
            (-4000, 0, 166, 4000)
        );

        // Menus over what the camera offers: a missing item is `EINVAL`, the default is the
        // conventional one when offered.
        let power = r
            .device
            .query_ext_ctrl_raw(&s, bindings::V4L2_CID_POWER_LINE_FREQUENCY)
            .unwrap();
        assert_eq!(
            (power.minimum, power.maximum, power.default_value),
            (0, 3, 3)
        );
        assert_eq!(
            menu_name(
                &r.device
                    .querymenu(&s, bindings::V4L2_CID_POWER_LINE_FREQUENCY, 1)
                    .unwrap()
            ),
            "50 Hz"
        );
        let fx = r
            .device
            .query_ext_ctrl_raw(&s, bindings::V4L2_CID_COLORFX)
            .unwrap();
        assert_eq!((fx.minimum, fx.maximum, fx.default_value), (0, 13, 0));
        let offered: Vec<u32> = (0..=13)
            .filter(|&i| {
                r.device
                    .querymenu(&s, bindings::V4L2_CID_COLORFX, i)
                    .is_ok()
            })
            .collect();
        assert_eq!(offered, vec![0, 1, 2, 3, 10, 13]);
        assert_eq!(
            menu_name(
                &r.device
                    .querymenu(&s, bindings::V4L2_CID_COLORFX, 13)
                    .unwrap()
            ),
            "Solarization"
        );
        let flash = r
            .device
            .query_ext_ctrl_raw(&s, bindings::V4L2_CID_FLASH_LED_MODE)
            .unwrap();
        assert_eq!(
            (flash.minimum, flash.maximum, flash.default_value),
            (0, 2, 0)
        );
        assert_eq!(
            menu_name(
                &r.device
                    .querymenu(&s, bindings::V4L2_CID_FLASH_LED_MODE, 2)
                    .unwrap()
            ),
            "Torch"
        );
        // `Flash` is the skipped item.
        assert_eq!(
            r.device
                .querymenu(&s, bindings::V4L2_CID_FLASH_LED_MODE, 1)
                .err(),
            Some(libc::EINVAL)
        );
        let wb = r
            .device
            .query_ext_ctrl_raw(&s, bindings::V4L2_CID_AUTO_N_PRESET_WHITE_BALANCE)
            .unwrap();
        assert_eq!((wb.minimum, wb.maximum, wb.default_value), (0, 9, 1));
        let offered: Vec<u32> = (0..=9)
            .filter(|&i| {
                r.device
                    .querymenu(&s, bindings::V4L2_CID_AUTO_N_PRESET_WHITE_BALANCE, i)
                    .is_ok()
            })
            .collect();
        assert_eq!(offered, vec![1, 2, 3, 4, 5, 6, 8, 9]);
        let scene = r
            .device
            .query_ext_ctrl_raw(&s, bindings::V4L2_CID_SCENE_MODE)
            .unwrap();
        assert_eq!(
            (scene.minimum, scene.maximum, scene.default_value),
            (0, 11, 0)
        );
        let offered: Vec<u32> = (0..=11)
            .filter(|&i| {
                r.device
                    .querymenu(&s, bindings::V4L2_CID_SCENE_MODE, i)
                    .is_ok()
            })
            .collect();
        assert_eq!(offered, vec![0, 6, 7, 8, 10, 11]);
        let ae_state = r.device.query_ext_ctrl_raw(&s, VCAM_CID_AE_STATE).unwrap();
        assert_eq!(
            ae_state.flags,
            bindings::V4L2_CTRL_FLAG_READ_ONLY | bindings::V4L2_CTRL_FLAG_VOLATILE
        );
        assert_eq!(
            menu_name(&r.device.querymenu(&s, VCAM_CID_AE_STATE, 4).unwrap()),
            "Flash Required"
        );
        let lens = r
            .device
            .query_ext_ctrl_raw(&s, VCAM_CID_ACTIVE_PHYSICAL_ID)
            .unwrap();
        assert_eq!((lens.minimum, lens.maximum), (0, 2));
        let lenses: Vec<i64> = (0..=2)
            .map(|i| {
                menu_value(
                    &r.device
                        .querymenu(&s, VCAM_CID_ACTIVE_PHYSICAL_ID, i)
                        .unwrap(),
                )
            })
            .collect();
        assert_eq!(lenses, vec![3, 2, 4]);
        // Not a menu.
        assert_eq!(
            r.device
                .querymenu(&s, bindings::V4L2_CID_ZOOM_ABSOLUTE, 0)
                .err(),
            Some(libc::EINVAL)
        );

        // The buttons, the status, and the regions' shape.
        let start = r
            .device
            .query_ext_ctrl_raw(&s, bindings::V4L2_CID_AUTO_FOCUS_START)
            .unwrap();
        assert_eq!(start.type_, bindings::v4l2_ctrl_type_V4L2_CTRL_TYPE_BUTTON);
        assert_eq!(
            start.flags,
            bindings::V4L2_CTRL_FLAG_WRITE_ONLY | bindings::V4L2_CTRL_FLAG_EXECUTE_ON_WRITE
        );
        let status = r
            .device
            .query_ext_ctrl_raw(&s, bindings::V4L2_CID_AUTO_FOCUS_STATUS)
            .unwrap();
        assert_eq!(
            status.type_,
            bindings::v4l2_ctrl_type_V4L2_CTRL_TYPE_BITMASK
        );
        assert_eq!((status.minimum, status.maximum, status.step), (0, 7, 0));
        assert_eq!(
            status.flags,
            bindings::V4L2_CTRL_FLAG_READ_ONLY | bindings::V4L2_CTRL_FLAG_VOLATILE
        );
        let regions = r
            .device
            .query_ext_ctrl_raw(&s, VCAM_CID_AF_REGIONS)
            .unwrap();
        assert_eq!(regions.type_, bindings::v4l2_ctrl_type_V4L2_CTRL_TYPE_U32);
        assert_eq!(regions.flags, bindings::V4L2_CTRL_FLAG_HAS_PAYLOAD);
        assert_eq!(
            (regions.elem_size, regions.elems, regions.nr_of_dims),
            (4, 5, 2)
        );
        assert_eq!(&regions.dims[..2], &[1, 5]);
        assert_eq!((regions.minimum, regions.maximum), (0, REGION_SCALE as i64));
        assert_eq!(qc_name(&regions), "Auto Focus, Regions");
        // `QUERYCTRL` cannot carry a compound range and zeroes it, as the kernel does.
        let old = r.device.queryctrl_raw(&s, VCAM_CID_AF_REGIONS).unwrap();
        assert_eq!(
            (old.minimum, old.maximum, old.step, old.default_value),
            (0, 0, 0, 0)
        );
        let old = r
            .device
            .queryctrl_raw(&s, bindings::V4L2_CID_ZOOM_ABSOLUTE)
            .unwrap();
        assert_eq!(
            (old.minimum, old.maximum, old.step, old.default_value),
            (67, 2000, 1, 100)
        );
        assert!(old.name.starts_with(b"Zoom, Absolute\0"));

        // The wiring: the guest's id word, flags included, reaches the device through the
        // dispatcher's `_raw` path.
        let mut input = vec![0u8; std::mem::size_of::<v4l2_query_ext_ctrl>()];
        input[..4].copy_from_slice(&(NEXT | NEXT_COMPOUND).to_le_bytes());
        let mut out = Vec::new();
        let mut other = session(&mut r.device);
        <Device as VirtioMediaDevice<&[u8], Vec<u8>>>::do_ioctl(
            &mut r.device,
            &mut other,
            crate::protocol::V4l2Ioctl::VIDIOC_QUERY_EXT_CTRL,
            &mut input.as_slice(),
            &mut out,
        )
        .unwrap();
        assert_eq!(i32::from_le_bytes(out[0..4].try_into().unwrap()), 0);
        assert_eq!(
            u32::from_le_bytes(out[8..12].try_into().unwrap()),
            bindings::V4L2_CID_USER_CLASS
        );

        close(&mut r.device, s);
    }

    /// Set and get: a mixed-class `S_EXT_CTRLS` is validated as a whole and handed to the camera
    /// in one call, a value set again is not; `G_EXT_CTRLS` and `G_CTRL` read it back; what is
    /// set with no stream open is applied when one opens, and what is set while one runs
    /// reaches it; `S_PARM` keeps its own path.
    #[test]
    fn controls_round_trip_and_reach_the_camera() {
        let mut r = rig();
        let mut s = session(&mut r.device);

        let mut ctrls = ext_controls(4);
        let mut array = vec![
            ext_control(bindings::V4L2_CID_ZOOM_ABSOLUTE, 300),
            ext_control(
                bindings::V4L2_CID_AUTO_N_PRESET_WHITE_BALANCE,
                WhiteBalance::Daylight as i32,
            ),
            ext_control(
                bindings::V4L2_CID_POWER_LINE_FREQUENCY,
                PowerLine::Hz50 as i32,
            ),
            ext_control(bindings::V4L2_CID_FLASH_LED_MODE, FlashLed::Torch as i32),
        ];
        r.device
            .s_ext_ctrls(&mut s, CtrlWhich::Current, &mut ctrls, &mut array, vec![])
            .unwrap();
        assert_eq!(ctrls.error_idx, 4);
        assert_eq!(ctrls.reserved, [0]);
        assert!(array.iter().all(|c| {
            let reserved = c.reserved2;
            reserved == [0]
        }));
        {
            let log = r.log.lock().unwrap();
            assert_eq!(
                log.backend_controls,
                vec![(
                    false,
                    vec![
                        CameraControl::Zoom(300),
                        CameraControl::WhiteBalance(WhiteBalance::Daylight),
                        CameraControl::PowerLine(PowerLine::Hz50),
                        CameraControl::FlashLed(FlashLed::Torch),
                    ]
                )]
            );
        }
        // Read back, both ways.
        let mut ctrls = ext_controls(2);
        let mut array = vec![
            ext_control(bindings::V4L2_CID_ZOOM_ABSOLUTE, 0),
            ext_control(bindings::V4L2_CID_FLASH_LED_MODE, 0),
        ];
        r.device
            .g_ext_ctrls(&s, CtrlWhich::Current, &mut ctrls, &mut array, vec![])
            .unwrap();
        assert_eq!(ctrl_value(&array[0]), 300);
        assert_eq!(ctrl_value(&array[1]), FlashLed::Torch as i32);
        assert_eq!(
            r.device
                .g_ctrl(&s, bindings::V4L2_CID_AUTO_N_PRESET_WHITE_BALANCE)
                .unwrap()
                .value,
            WhiteBalance::Daylight as i32
        );
        // The same values again change nothing and reach nobody.
        let mut ctrls = ext_controls(1);
        let mut array = vec![ext_control(bindings::V4L2_CID_ZOOM_ABSOLUTE, 300)];
        r.device
            .s_ext_ctrls(&mut s, CtrlWhich::Current, &mut ctrls, &mut array, vec![])
            .unwrap();
        assert_eq!(r.log.lock().unwrap().backend_controls.len(), 1);

        // `which` as a class: every control must be in it.
        let mut ctrls = ext_controls(2);
        let mut array = vec![
            ext_control(bindings::V4L2_CID_ZOOM_ABSOLUTE, 310),
            ext_control(bindings::V4L2_CID_POWER_LINE_FREQUENCY, 2),
        ];
        assert_eq!(
            r.device.s_ext_ctrls(
                &mut s,
                CtrlWhich::Class(bindings::V4L2_CTRL_CLASS_CAMERA),
                &mut ctrls,
                &mut array,
                vec![]
            ),
            Err(libc::EINVAL)
        );
        assert_eq!(ctrls.error_idx, 2);
        assert_eq!(
            r.device
                .g_ctrl(&s, bindings::V4L2_CID_ZOOM_ABSOLUTE)
                .unwrap()
                .value,
            300,
            "nothing of a refused set is applied"
        );
        let mut ctrls = ext_controls(1);
        let mut array = vec![ext_control(bindings::V4L2_CID_ZOOM_ABSOLUTE, 310)];
        r.device
            .s_ext_ctrls(
                &mut s,
                CtrlWhich::Class(bindings::V4L2_CTRL_CLASS_CAMERA),
                &mut ctrls,
                &mut array,
                vec![],
            )
            .unwrap();
        // `count == 0` asks whether the class exists.
        let mut none = ext_controls(0);
        r.device
            .g_ext_ctrls(
                &s,
                CtrlWhich::Class(bindings::V4L2_CTRL_CLASS_FLASH),
                &mut none,
                &mut vec![],
                vec![],
            )
            .unwrap();
        assert_eq!(none.error_idx, 0);
        assert_eq!(
            r.device.g_ext_ctrls(
                &s,
                CtrlWhich::Class(bindings::V4L2_CTRL_CLASS_CODEC),
                &mut none,
                &mut vec![],
                vec![]
            ),
            Err(libc::EINVAL)
        );

        // The plain ioctls: a class control, a button, a payload control, the read-only
        // status, an unknown id.
        assert_eq!(
            r.device.g_ctrl(&s, bindings::V4L2_CID_CAMERA_CLASS).err(),
            Some(libc::EINVAL)
        );
        assert_eq!(
            r.device
                .g_ctrl(&s, bindings::V4L2_CID_AUTO_FOCUS_START)
                .err(),
            Some(libc::EACCES)
        );
        assert_eq!(
            r.device.g_ctrl(&s, VCAM_CID_AF_REGIONS).err(),
            Some(libc::EINVAL)
        );
        assert_eq!(
            r.device
                .s_ctrl(&mut s, bindings::V4L2_CID_AUTO_FOCUS_STATUS, 0)
                .err(),
            Some(libc::EACCES)
        );
        assert_eq!(r.device.g_ctrl(&s, 0).err(), Some(libc::EINVAL));
        assert_eq!(r.device.s_ctrl(&mut s, 0, 0).err(), Some(libc::EINVAL));
        // A menu's value is its index: the ISO and the bias reach the camera as numbers.
        r.device
            .s_ctrl(&mut s, bindings::V4L2_CID_ISO_SENSITIVITY, 3)
            .unwrap();
        r.device
            .s_ctrl(&mut s, bindings::V4L2_CID_AUTO_EXPOSURE_BIAS, 30)
            .unwrap();
        // A boolean is normalised, and only a change is sent.
        assert_eq!(
            r.device
                .s_ctrl(&mut s, bindings::V4L2_CID_FOCUS_AUTO, 0)
                .unwrap()
                .value,
            0
        );
        assert_eq!(
            r.device
                .s_ctrl(&mut s, bindings::V4L2_CID_FOCUS_AUTO, 7)
                .unwrap()
                .value,
            1
        );
        {
            let log = r.log.lock().unwrap();
            let sent: Vec<CameraControl> = log
                .backend_controls
                .iter()
                .flat_map(|(_, c)| c.clone())
                .collect();
            assert!(sent.contains(&CameraControl::Iso(800)));
            assert!(sent.contains(&CameraControl::ExposureBias(6)));
            assert!(sent.contains(&CameraControl::FocusAuto(false)));
            assert!(sent.ends_with(&[CameraControl::FocusAuto(true)]));
            assert!(log.backend_controls.iter().all(|(streaming, _)| !streaming));
        }
        // Manual exposure: either switch turns the auto off, and the manual values become
        // active.
        r.device
            .s_ctrl(
                &mut s,
                bindings::V4L2_CID_EXPOSURE_AUTO,
                ExposureMode::Manual as i32,
            )
            .unwrap();
        assert_eq!(
            r.device
                .query_ext_ctrl_raw(&s, bindings::V4L2_CID_EXPOSURE_ABSOLUTE)
                .unwrap()
                .flags,
            0
        );
        assert_eq!(
            r.device
                .query_ext_ctrl_raw(&s, bindings::V4L2_CID_ISO_SENSITIVITY)
                .unwrap()
                .flags,
            0
        );
        r.device
            .s_ctrl(&mut s, bindings::V4L2_CID_EXPOSURE_ABSOLUTE, 500)
            .unwrap();
        r.device
            .s_ctrl(
                &mut s,
                bindings::V4L2_CID_EXPOSURE_AUTO,
                ExposureMode::Auto as i32,
            )
            .unwrap();
        assert_eq!(
            r.device
                .query_ext_ctrl_raw(&s, bindings::V4L2_CID_EXPOSURE_ABSOLUTE)
                .unwrap()
                .flags,
            bindings::V4L2_CTRL_FLAG_INACTIVE
        );
        r.device
            .s_ctrl(
                &mut s,
                bindings::V4L2_CID_ISO_SENSITIVITY_AUTO,
                IsoMode::Manual as i32,
            )
            .unwrap();
        assert_eq!(
            r.device
                .query_ext_ctrl_raw(&s, bindings::V4L2_CID_EXPOSURE_ABSOLUTE)
                .unwrap()
                .flags,
            0
        );

        // Everything set so far opens the stream.
        r.device.s_fmt(&mut s, QUEUE, format(64, 48)).unwrap();
        let size = 64 * 48 * 3 / 2;
        r.device
            .reqbufs(&mut s, QUEUE, MemoryType::Mmap, 1)
            .unwrap();
        r.device
            .qbuf(&mut s, mmap_buffer(0, size), vec![], PayloadValidity::ALL)
            .unwrap();
        r.device.streamon(&mut s, QUEUE).unwrap();
        {
            let log = r.log.lock().unwrap();
            let opened = &log.opened[0].controls;
            for expected in [
                CameraControl::Zoom(310),
                CameraControl::WhiteBalance(WhiteBalance::Daylight),
                CameraControl::PowerLine(PowerLine::Hz50),
                CameraControl::FlashLed(FlashLed::Torch),
                CameraControl::Iso(800),
                CameraControl::ExposureBias(6),
                CameraControl::FocusAuto(true),
                CameraControl::ExposureMode(ExposureMode::Auto),
                CameraControl::IsoMode(IsoMode::Manual),
                CameraControl::ExposureTime(500),
                CameraControl::ColorEffect(ColorEffect::None),
                CameraControl::SceneMode(SceneMode::None),
                CameraControl::Stabilization(false),
                CameraControl::AeRegions(vec![Region::default()]),
                CameraControl::AfRegions(vec![Region::default()]),
            ] {
                assert!(
                    opened.contains(&expected),
                    "{:?} missing from {:?}",
                    expected,
                    opened
                );
            }
            assert!(!opened
                .iter()
                .any(|c| matches!(c, CameraControl::AfTrigger(_) | CameraControl::AfStatus(_))));
        }
        let _ = collect_frames(&mut r, &mut s, 1);
        // While streaming, a set reaches the running camera.
        r.device
            .s_ctrl(&mut s, bindings::V4L2_CID_ZOOM_ABSOLUTE, 150)
            .unwrap();
        assert_eq!(
            r.log.lock().unwrap().backend_controls.last().unwrap(),
            &(true, vec![CameraControl::Zoom(150)])
        );
        assert_eq!(
            r.device
                .g_ctrl(&s, bindings::V4L2_CID_ZOOM_ABSOLUTE)
                .unwrap()
                .value,
            150
        );
        // `S_PARM` still goes through the stream, not the backend.
        let before = r.log.lock().unwrap().backend_controls.len();
        r.device.s_parm(&mut s, parm_for((1, 15))).unwrap();
        {
            let log = r.log.lock().unwrap();
            assert_eq!(log.controls, vec![CameraControl::FpsRange(15, 15)]);
            assert_eq!(log.backend_controls.len(), before);
        }
        r.device.streamoff(&mut s, QUEUE).unwrap();
        r.device
            .s_ctrl(&mut s, bindings::V4L2_CID_ZOOM_ABSOLUTE, 200)
            .unwrap();
        assert_eq!(
            r.log.lock().unwrap().backend_controls.last().unwrap(),
            &(false, vec![CameraControl::Zoom(200)])
        );

        close(&mut r.device, s);
    }

    /// `TRY_EXT_CTRLS` refuses what `S_EXT_CTRLS` would -- out of range, a missing menu item, a
    /// read-only control, the defaults, an unknown id -- naming the control, and changes
    /// nothing; a set that is refused applies none of it.
    #[test]
    fn try_ext_ctrls_refuses_out_of_range_and_changes_nothing() {
        let mut r = rig();
        let mut s = session(&mut r.device);

        let mut ctrls = ext_controls(2);
        let mut array = vec![
            ext_control(
                bindings::V4L2_CID_AUTO_N_PRESET_WHITE_BALANCE,
                WhiteBalance::Daylight as i32,
            ),
            ext_control(bindings::V4L2_CID_ZOOM_ABSOLUTE, 2001),
        ];
        assert_eq!(
            r.device
                .try_ext_ctrls(&s, CtrlWhich::Current, &mut ctrls, &mut array, vec![]),
            Err(libc::ERANGE)
        );
        assert_eq!(ctrls.error_idx, 1, "the control refused");
        assert_eq!(
            r.device
                .s_ext_ctrls(&mut s, CtrlWhich::Current, &mut ctrls, &mut array, vec![]),
            Err(libc::ERANGE)
        );
        assert_eq!(ctrls.error_idx, 2, "a set fails as a whole");
        assert_eq!(
            r.device
                .g_ctrl(&s, bindings::V4L2_CID_AUTO_N_PRESET_WHITE_BALANCE)
                .unwrap()
                .value,
            WhiteBalance::Auto as i32,
            "the valid half was not applied either"
        );
        assert!(r.log.lock().unwrap().backend_controls.is_empty());
        // Below the minimum, through `S_CTRL`.
        assert_eq!(
            r.device
                .s_ctrl(&mut s, bindings::V4L2_CID_ZOOM_ABSOLUTE, 66)
                .err(),
            Some(libc::ERANGE)
        );
        // A menu index past the end is out of range; an item the camera lacks is invalid.
        assert_eq!(
            r.device
                .s_ctrl(&mut s, bindings::V4L2_CID_POWER_LINE_FREQUENCY, 4)
                .err(),
            Some(libc::ERANGE)
        );
        let mut ctrls = ext_controls(1);
        let mut array = vec![ext_control(
            bindings::V4L2_CID_FLASH_LED_MODE,
            FlashLed::Flash as i32,
        )];
        assert_eq!(
            r.device
                .try_ext_ctrls(&s, CtrlWhich::Current, &mut ctrls, &mut array, vec![]),
            Err(libc::EINVAL)
        );
        assert_eq!(ctrls.error_idx, 0);
        // Read-only: `EACCES`, at the control for a try and at `count` for a set.
        let mut ctrls = ext_controls(1);
        let mut array = vec![ext_control(bindings::V4L2_CID_AUTO_FOCUS_STATUS, 0)];
        assert_eq!(
            r.device
                .try_ext_ctrls(&s, CtrlWhich::Current, &mut ctrls, &mut array, vec![]),
            Err(libc::EACCES)
        );
        assert_eq!(ctrls.error_idx, 0);
        assert_eq!(
            r.device
                .s_ext_ctrls(&mut s, CtrlWhich::Current, &mut ctrls, &mut array, vec![]),
            Err(libc::EACCES)
        );
        assert_eq!(ctrls.error_idx, 1);
        // A class control can be neither read (write-only) nor written (read-only).
        let mut array = vec![ext_control(bindings::V4L2_CID_CAMERA_CLASS, 0)];
        assert_eq!(
            r.device
                .g_ext_ctrls(&s, CtrlWhich::Current, &mut ctrls, &mut array, vec![]),
            Err(libc::EACCES)
        );
        assert_eq!(ctrls.error_idx, 1);
        assert_eq!(
            r.device
                .try_ext_ctrls(&s, CtrlWhich::Current, &mut ctrls, &mut array, vec![]),
            Err(libc::EACCES)
        );
        // A button cannot be read.
        let mut array = vec![ext_control(bindings::V4L2_CID_AUTO_FOCUS_START, 0)];
        assert_eq!(
            r.device
                .g_ext_ctrls(&s, CtrlWhich::Current, &mut ctrls, &mut array, vec![]),
            Err(libc::EACCES)
        );
        // A try normalises what a set would store, and stores nothing.
        let mut array = vec![ext_control(bindings::V4L2_CID_FOCUS_AUTO, 5)];
        r.device
            .try_ext_ctrls(&s, CtrlWhich::Current, &mut ctrls, &mut array, vec![])
            .unwrap();
        assert_eq!(ctrl_value(&array[0]), 1);
        assert!(r.log.lock().unwrap().backend_controls.is_empty());
        // The defaults can be read, not set or tried.
        r.device
            .s_ctrl(&mut s, bindings::V4L2_CID_ZOOM_ABSOLUTE, 300)
            .unwrap();
        let mut array = vec![ext_control(bindings::V4L2_CID_ZOOM_ABSOLUTE, 0)];
        r.device
            .g_ext_ctrls(&s, CtrlWhich::Default, &mut ctrls, &mut array, vec![])
            .unwrap();
        assert_eq!(ctrl_value(&array[0]), 100);
        r.device
            .g_ext_ctrls(&s, CtrlWhich::Current, &mut ctrls, &mut array, vec![])
            .unwrap();
        assert_eq!(ctrl_value(&array[0]), 300);
        assert_eq!(
            r.device
                .s_ext_ctrls(&mut s, CtrlWhich::Default, &mut ctrls, &mut array, vec![]),
            Err(libc::EINVAL)
        );
        assert_eq!(
            r.device
                .try_ext_ctrls(&s, CtrlWhich::Default, &mut ctrls, &mut array, vec![]),
            Err(libc::EINVAL)
        );
        // `count == 0` is fine; id 0 is not, with the error index the kernel gives.
        let mut none = ext_controls(0);
        r.device
            .try_ext_ctrls(&s, CtrlWhich::Current, &mut none, &mut vec![], vec![])
            .unwrap();
        assert_eq!((none.count, none.error_idx), (0, 0));
        let mut ctrls = ext_controls(1);
        let mut array = vec![ext_control(0, 0)];
        assert_eq!(
            r.device
                .g_ext_ctrls(&s, CtrlWhich::Current, &mut ctrls, &mut array, vec![]),
            Err(libc::EINVAL)
        );
        assert_eq!(ctrls.error_idx, 1);
        assert_eq!(
            r.device
                .try_ext_ctrls(&s, CtrlWhich::Current, &mut ctrls, &mut array, vec![]),
            Err(libc::EINVAL)
        );
        assert_eq!(ctrls.error_idx, 0);
        assert_eq!(
            r.device
                .s_ext_ctrls(&mut s, CtrlWhich::Current, &mut ctrls, &mut array, vec![]),
            Err(libc::EINVAL)
        );
        assert_eq!(ctrls.error_idx, 1);
        // An old-style private id is for `G_CTRL`/`S_CTRL` only.
        let mut array = vec![ext_control(controls::V4L2_CID_PRIVATE_BASE, 0)];
        assert_eq!(
            r.device
                .g_ext_ctrls(&s, CtrlWhich::Current, &mut ctrls, &mut array, vec![]),
            Err(libc::EINVAL)
        );

        close(&mut r.device, s);
    }

    /// D33: every ioctl the kernel resolves an old-style `V4L2_CID_PRIVATE_BASE + n` id in --
    /// `QUERYCTRL`, `QUERY_EXT_CTRL`, `QUERYMENU`, `G_CTRL` and `S_CTRL`, the five
    /// `find_private_ref` serves through `find_ref` -- resolves it here too, so a menu control
    /// reached through its alias has items and a value.
    ///
    /// `v4l2-compliance` walks the aliases with `QUERY_EXT_CTRL` from `PRIVATE_BASE` up and runs
    /// `checkQCtrl` on each; for a menu that is a `QUERYMENU(minimum..=maximum + 1)` walk which
    /// must find at least one item, an item at the default value, `EINVAL` past the maximum, and
    /// the id and index it asked for (`v4l2-test-controls.cpp:145-172`, then `:301-315`). This
    /// test is that walk. Before the fix every one of those `QUERYMENU`s was `EINVAL`, which is
    /// `no menu items found` at `:171` and `invalid control 08000000` at `:315`.
    #[test]
    fn the_private_aliases_answer_every_old_style_ioctl() {
        let mut r = rig();
        let mut s = session(&mut r.device);
        let base = controls::V4L2_CID_PRIVATE_BASE;

        // The alias walk, stopping at the first `EINVAL` as compliance does.
        let mut aliases = Vec::new();
        let mut id = base;
        while let Ok(qc) = r.device.query_ext_ctrl_raw(&s, id) {
            assert_eq!(qc.id, id, "the alias answers under the alias id");
            aliases.push(qc);
            id += 1;
        }
        assert_eq!(
            aliases.iter().map(qc_name).collect::<Vec<_>>(),
            vec!["Active Physical Camera", "Auto Exposure, State"],
            "the two private USER-class integer controls, in order"
        );

        // `checkQCtrl`'s menu walk, for each alias.
        for qc in &aliases {
            assert!(
                qc.type_ == bindings::v4l2_ctrl_type_V4L2_CTRL_TYPE_MENU
                    || qc.type_ == bindings::v4l2_ctrl_type_V4L2_CTRL_TYPE_INTEGER_MENU
            );
            let mut items = 0;
            let mut have_default = false;
            for index in 0..=(qc.maximum as u32 + 1) {
                match r.device.querymenu(&s, qc.id, index) {
                    Ok(qm) => {
                        assert!(
                            index <= qc.maximum as u32,
                            "menu item for an out-of-range index"
                        );
                        assert_eq!((qm.id, qm.index), (qc.id, index), "id or index changed");
                        assert_eq!({ qm.reserved }, 0, "reserved is non-zero");
                        items += 1;
                        have_default |= index as i64 == qc.default_value;
                    }
                    Err(e) => assert_eq!(e, libc::EINVAL, "invalid QUERYMENU return code"),
                }
            }
            assert_ne!(items, 0, "no menu items found ({})", qc_name(qc));
            assert!(have_default, "no item at the default value");
        }

        // The alias and the real id name one control: the same menu items and the same value.
        assert_eq!(
            menu_name(&r.device.querymenu(&s, base + 1, 2).unwrap()),
            menu_name(&r.device.querymenu(&s, VCAM_CID_AE_STATE, 2).unwrap()),
        );
        assert_eq!(
            menu_value(&r.device.querymenu(&s, base, 1).unwrap()),
            menu_value(
                &r.device
                    .querymenu(&s, VCAM_CID_ACTIVE_PHYSICAL_ID, 1)
                    .unwrap()
            ),
        );
        let by_alias = r.device.g_ctrl(&s, base).unwrap();
        let by_id = r.device.g_ctrl(&s, VCAM_CID_ACTIVE_PHYSICAL_ID).unwrap();
        assert_eq!(by_alias.value, by_id.value);
        assert_eq!(by_alias.id, base, "the answer carries the id as asked");
        // Both are read-only, so a set through the alias is `EACCES` -- the control was found.
        assert_eq!(r.device.s_ctrl(&mut s, base, 0).err(), Some(libc::EACCES));
        assert_eq!(
            r.device.s_ctrl(&mut s, base + 1, 0).err(),
            Some(libc::EACCES)
        );

        // The compound controls have no alias, in any of the five (a payload cannot travel in
        // a `v4l2_control`), and the walk therefore ends at `+2`.
        assert_eq!(
            r.device.query_ext_ctrl_raw(&s, base + 2).err(),
            Some(libc::EINVAL)
        );
        assert_eq!(
            r.device.queryctrl_raw(&s, base + 2).err(),
            Some(libc::EINVAL)
        );
        assert_eq!(
            r.device.querymenu(&s, base + 2, 0).err(),
            Some(libc::EINVAL)
        );
        assert_eq!(r.device.g_ctrl(&s, base + 2).err(), Some(libc::EINVAL));
        assert_eq!(
            r.device.s_ctrl(&mut s, base + 2, 0).err(),
            Some(libc::EINVAL)
        );
        // And a control that is not a menu still refuses `QUERYMENU`, by either name.
        assert_eq!(
            r.device
                .querymenu(&s, bindings::V4L2_CID_ZOOM_ABSOLUTE, 0)
                .err(),
            Some(libc::EINVAL)
        );

        close(&mut r.device, s);
    }

    /// D37: a failed `G/S/TRY_EXT_CTRLS` writes the `v4l2_ext_controls` header back, so
    /// `error_idx` reaches the guest.
    ///
    /// This is asserted on the wire, through the whole command path, because it is the response
    /// *length* that decides it: `virtio_media_send_ext_controls_ioctl()` copies the header out
    /// of the response only when the device wrote at least
    /// `sizeof(virtio_media_resp_ioctl) + sizeof(v4l2_ext_controls)` bytes, and skips it -- with
    /// no error and no log line -- when the response is shorter. So the test checks the exact
    /// byte count of an error response and reads `error_idx` back out of it, and it pins the two
    /// structure sizes the guest ABI depends on.
    ///
    /// **D37 itself is closed as a test artefact of the guest-side Python client** and this test
    /// is now a regression guard, not an open investigation: on hardware a C client reads the
    /// device's `error_idx` back on all seven refusals, including the `TRY_EXT_CTRLS` this
    /// device fails at index 1 in `try_ext_ctrls`, while CPython's `fcntl.ioctl` reads `0`
    /// for the same calls because it writes its argument buffer back only when the ioctl returns
    /// >= 0 (`logs/vpu_wp/B10-acceptance.md` §7). The client that settles it is checked in as
    /// `deploy/vpu/tests/ext_ctrls_error_idx.c`.
    #[test]
    fn a_failed_ext_ctrls_writes_the_header_back_with_error_idx() {
        const RESP_HEADER: usize = std::mem::size_of::<RespHeader>();
        const CTRLS: usize = std::mem::size_of::<v4l2_ext_controls>();
        const CTRL: usize = std::mem::size_of::<v4l2_ext_control>();
        assert_eq!((RESP_HEADER, CTRLS, CTRL), (8, 32, 20), "the guest ABI");

        /// One `VIRTIO_MEDIA_CMD_IOCTL` for an ext-controls ioctl, as the driver builds it:
        /// the command header, the ioctl header, the `v4l2_ext_controls` and its array. The
        /// `error_idx` the guest sends is a sentinel, so the test can tell "the device wrote
        /// the header" from "the header came back untouched".
        fn command(code: V4l2Ioctl, which: u32, sentinel: u32, values: &[(u32, i32)]) -> Vec<u8> {
            let mut out = Vec::new();
            out.extend(VIRTIO_MEDIA_CMD_IOCTL.to_le_bytes());
            out.extend(0u32.to_le_bytes());
            out.extend(0u32.to_le_bytes()); // session id
            out.extend((code as u32).to_le_bytes());
            out.extend(which.to_le_bytes());
            out.extend((values.len() as u32).to_le_bytes());
            out.extend(sentinel.to_le_bytes());
            out.extend(0u32.to_le_bytes()); // request_fd
            out.extend(0u32.to_le_bytes()); // reserved[0]
            out.extend(0u32.to_le_bytes()); // padding before the pointer
            out.extend(0u64.to_le_bytes()); // controls, nulled by the driver
            for (id, value) in values {
                out.extend(id.to_le_bytes());
                out.extend(0u32.to_le_bytes()); // size: a plain control
                out.extend(0u32.to_le_bytes()); // reserved2[0]
                out.extend(value.to_le_bytes());
                out.extend(0u32.to_le_bytes()); // the rest of the union
            }
            out
        }

        const SENTINEL: u32 = 0xbeef;
        let refused = &[
            (bindings::V4L2_CID_AUTO_N_PRESET_WHITE_BALANCE, 1),
            (bindings::V4L2_CID_ZOOM_ABSOLUTE, 2001),
        ];
        let unknown = &[(bindings::V4L2_CID_BRIGHTNESS, 0)];
        // Every command is built before the runner: the reader type is part of the runner's own
        // type, so these bytes must outlive it.
        let commands: Vec<(&str, Vec<u8>, usize, (i32, u32))> = vec![
            // A try names the control it refused; a set fails as a whole, so `count`.
            (
                "TRY out of range",
                command(V4l2Ioctl::VIDIOC_TRY_EXT_CTRLS, 0, SENTINEL, refused),
                2,
                (libc::ERANGE, 1),
            ),
            (
                "S out of range",
                command(V4l2Ioctl::VIDIOC_S_EXT_CTRLS, 0, SENTINEL, refused),
                2,
                (libc::ERANGE, 2),
            ),
            // A get that fails: `error_idx` is `count`, as the kernel leaves it for a get.
            (
                "G unknown id",
                command(V4l2Ioctl::VIDIOC_G_EXT_CTRLS, 0, SENTINEL, unknown),
                1,
                (libc::EINVAL, 1),
            ),
            // A write-only control read: `EACCES` and `error_idx = count`, both of which
            // `v4l2-test-controls.cpp:901-905` checks.
            (
                "G write-only",
                command(
                    V4l2Ioctl::VIDIOC_G_EXT_CTRLS,
                    0,
                    SENTINEL,
                    &[(bindings::V4L2_CID_AUTO_FOCUS_START, 0)],
                ),
                1,
                (libc::EACCES, 1),
            ),
            // A `which` word no control set can be named under is refused before the device is
            // asked at all -- and still carries the header.
            (
                "G which=MIN_VAL",
                command(
                    V4l2Ioctl::VIDIOC_G_EXT_CTRLS,
                    bindings::V4L2_CTRL_WHICH_MIN_VAL,
                    SENTINEL,
                    unknown,
                ),
                1,
                (libc::EINVAL, 1),
            ),
            // And a set that succeeds answers the same shape, with `error_idx` at `count`.
            (
                "S accepted",
                command(
                    V4l2Ioctl::VIDIOC_S_EXT_CTRLS,
                    0,
                    SENTINEL,
                    &[(bindings::V4L2_CID_ZOOM_ABSOLUTE, 200)],
                ),
                1,
                (0, 1),
            ),
        ];

        let r = rig();
        let poller = FakePoller::default();
        let open: Vec<u8> = VIRTIO_MEDIA_CMD_OPEN
            .to_le_bytes()
            .into_iter()
            .chain(0u32.to_le_bytes())
            .collect();
        let mut runner: VirtioMediaDeviceRunner<&[u8], Vec<u8>, Device, FakePoller> =
            VirtioMediaDeviceRunner::new(r.device, poller.clone());
        let mut writer: Vec<u8> = Vec::new();
        runner.handle_command(&mut open.as_slice(), &mut writer);
        assert_eq!(runner.sessions.len(), 1);

        for (what, cmd, count, expected) in &commands {
            let mut writer: Vec<u8> = Vec::new();
            runner.handle_command(&mut cmd.as_slice(), &mut writer);
            assert_eq!(
                writer.len(),
                RESP_HEADER + CTRLS + count * CTRL,
                "{}: the header and the array must both come back",
                what
            );
            let errno = i32::from_le_bytes(writer[0..4].try_into().unwrap());
            let error_idx = u32::from_le_bytes(writer[16..20].try_into().unwrap());
            assert_eq!((errno, error_idx), *expected, "{}", what);
        }
    }

    /// The autofocus buttons reach the camera every time they are pressed, and what the camera
    /// reports back -- the focus state, the exposure state, the lens in use -- becomes a
    /// `V4L2_EVENT_CTRL` only when it differs from what the guest was last told: the fake
    /// reports each state twice and a lens the camera never listed.
    #[test]
    fn an_af_trigger_produces_status_events_only_on_change() {
        let mut r = rig();
        let mut s = session(&mut r.device);
        for id in [
            bindings::V4L2_CID_AUTO_FOCUS_STATUS,
            VCAM_CID_AE_STATE,
            VCAM_CID_ACTIVE_PHYSICAL_ID,
        ] {
            r.device
                .subscribe_event(&mut s, EventType::Ctrl(id), SubscribeEventFlags::empty())
                .unwrap();
        }
        r.device.s_fmt(&mut s, QUEUE, format(64, 48)).unwrap();
        r.device
            .reqbufs(&mut s, QUEUE, MemoryType::Mmap, 1)
            .unwrap();
        r.device
            .qbuf(
                &mut s,
                mmap_buffer(0, 64 * 48 * 3 / 2),
                vec![],
                PayloadValidity::ALL,
            )
            .unwrap();
        r.device.streamon(&mut s, QUEUE).unwrap();
        let _ = collect_frames(&mut r, &mut s, 1);

        // A button is a press, not a value: the camera hears every one.
        for _ in 0..2 {
            r.device
                .s_ctrl(&mut s, bindings::V4L2_CID_AUTO_FOCUS_START, 0)
                .unwrap();
        }
        let sent: Vec<CameraControl> = r
            .log
            .lock()
            .unwrap()
            .backend_controls
            .iter()
            .flat_map(|(_, c)| c.clone())
            .collect();
        assert_eq!(
            sent.iter()
                .filter(|c| **c == CameraControl::AfTrigger(AfTrigger::Start))
                .count(),
            2
        );
        // Seven reports per press. The first press is four changes; the second repeats the
        // exposure state and the lens, which are no events, and scans again -- busy, then
        // reached -- which are two.
        while ctrl_events(&r.events.borrow()).len() < 6 {
            assert!(wait_ready(&s), "no event within 2s");
            process(&mut r.device, &mut s);
        }
        process(&mut r.device, &mut s);
        let events = ctrl_events(&r.events.borrow());
        assert_eq!(events.len(), 6, "{events:?}");
        assert_eq!(
            events[0],
            CtrlEv {
                session: 0,
                id: bindings::V4L2_CID_AUTO_FOCUS_STATUS,
                changes: CH_VALUE,
                value: AF_STATUS_BUSY as i64,
                flags: bindings::V4L2_CTRL_FLAG_READ_ONLY | bindings::V4L2_CTRL_FLAG_VOLATILE,
            }
        );
        assert_eq!(
            (events[1].id, events[1].value),
            (VCAM_CID_AE_STATE, AeState::Searching as i64)
        );
        assert_eq!(
            (events[2].id, events[2].value),
            (
                bindings::V4L2_CID_AUTO_FOCUS_STATUS,
                AF_STATUS_REACHED as i64
            )
        );
        // Lens "4" is the third physical id.
        assert_eq!(
            (events[3].id, events[3].value),
            (VCAM_CID_ACTIVE_PHYSICAL_ID, 2)
        );
        assert_eq!(
            (events[4].id, events[4].value, events[5].value),
            (
                bindings::V4L2_CID_AUTO_FOCUS_STATUS,
                AF_STATUS_BUSY as i64,
                AF_STATUS_REACHED as i64
            )
        );
        // The volatile controls read what was last reported.
        assert_eq!(
            r.device
                .g_ctrl(&s, bindings::V4L2_CID_AUTO_FOCUS_STATUS)
                .unwrap()
                .value,
            AF_STATUS_REACHED as i32
        );
        assert_eq!(
            r.device
                .g_ctrl(&s, VCAM_CID_ACTIVE_PHYSICAL_ID)
                .unwrap()
                .value,
            2
        );
        assert_eq!(
            menu_value(
                &r.device
                    .querymenu(&s, VCAM_CID_ACTIVE_PHYSICAL_ID, 2)
                    .unwrap()
            ),
            4
        );
        // A cancel reports idle, which is a change again.
        r.device
            .s_ctrl(&mut s, bindings::V4L2_CID_AUTO_FOCUS_STOP, 0)
            .unwrap();
        while ctrl_events(&r.events.borrow()).len() < 7 {
            assert!(wait_ready(&s), "no event within 2s");
            process(&mut r.device, &mut s);
        }
        let events = ctrl_events(&r.events.borrow());
        assert_eq!(
            (events[6].id, events[6].value),
            (bindings::V4L2_CID_AUTO_FOCUS_STATUS, AF_STATUS_IDLE as i64)
        );
        // And the session is still streaming: reports are not errors.
        assert!(!s.dead);
        assert_eq!(errors(&r.events.borrow()), 0);

        close(&mut r.device, s);
    }

    /// `V4L2_EVENT_CTRL` subscriptions are per session: an initial event on request (never for
    /// a class), a value set by one session announced to the others (and to itself only with
    /// feedback), a flag change announced to all, `V4L2_EVENT_ALL` dropping everything.
    #[test]
    fn control_events_go_to_subscribed_sessions() {
        let mut r = rig();
        let mut a = session(&mut r.device);
        let mut b =
            <Device as VirtioMediaDevice<&[u8], Vec<u8>>>::new_session(&mut r.device, 1).unwrap();
        let zoom = bindings::V4L2_CID_ZOOM_ABSOLUTE;

        // The initial event: the current value, the flags and the range.
        r.device
            .subscribe_event(
                &mut b,
                EventType::Ctrl(zoom),
                SubscribeEventFlags::SEND_INITIAL,
            )
            .unwrap();
        let events = ctrl_events(&r.events.borrow());
        assert_eq!(
            events,
            vec![CtrlEv {
                session: 1,
                id: zoom,
                changes: CH_FLAGS | CH_VALUE,
                value: 100,
                flags: 0,
            }]
        );
        {
            let events = r.events.borrow();
            let V4l2Event::Event(se) = &events[0] else {
                panic!("not a session event");
            };
            // SAFETY: a CTRL event.
            let ctrl = unsafe { se.event().u.ctrl };
            assert_eq!(
                (
                    ctrl.type_,
                    ctrl.minimum,
                    ctrl.maximum,
                    ctrl.step,
                    ctrl.default_value
                ),
                (
                    bindings::v4l2_ctrl_type_V4L2_CTRL_TYPE_INTEGER,
                    67,
                    2000,
                    1,
                    100
                )
            );
        }
        // A class control has no value to send; a button's initial event carries no value.
        r.device
            .subscribe_event(
                &mut a,
                EventType::Ctrl(bindings::V4L2_CID_CAMERA_CLASS),
                SubscribeEventFlags::SEND_INITIAL,
            )
            .unwrap();
        assert_eq!(ctrl_events(&r.events.borrow()).len(), 1);
        r.device
            .subscribe_event(
                &mut a,
                EventType::Ctrl(bindings::V4L2_CID_AUTO_FOCUS_START),
                SubscribeEventFlags::SEND_INITIAL,
            )
            .unwrap();
        assert_eq!(
            ctrl_events(&r.events.borrow()).last().unwrap().changes,
            CH_FLAGS
        );
        // What cannot be subscribed to.
        assert_eq!(
            r.device
                .subscribe_event(&mut a, EventType::Ctrl(0), SubscribeEventFlags::empty()),
            Err(libc::EINVAL)
        );
        assert_eq!(
            r.device.subscribe_event(
                &mut a,
                EventType::Ctrl(bindings::V4L2_CID_BRIGHTNESS),
                SubscribeEventFlags::empty()
            ),
            Err(libc::EINVAL)
        );
        assert_eq!(
            r.device
                .subscribe_event(&mut a, EventType::VSync, SubscribeEventFlags::empty()),
            Err(libc::EINVAL)
        );

        // A sets zoom: B hears, A does not.
        r.device
            .subscribe_event(&mut a, EventType::Ctrl(zoom), SubscribeEventFlags::empty())
            .unwrap();
        let before = ctrl_events(&r.events.borrow()).len();
        r.device.s_ctrl(&mut a, zoom, 150).unwrap();
        let events = ctrl_events(&r.events.borrow());
        assert_eq!(
            &events[before..],
            &[CtrlEv {
                session: 1,
                id: zoom,
                changes: CH_VALUE,
                value: 150,
                flags: 0,
            }]
        );
        // With feedback, A hears its own set too.
        r.device
            .unsubscribe_event(
                &mut a,
                v4l2_event_subscription {
                    type_: bindings::V4L2_EVENT_CTRL,
                    id: zoom,
                    ..Default::default()
                },
            )
            .unwrap();
        r.device
            .subscribe_event(
                &mut a,
                EventType::Ctrl(zoom),
                SubscribeEventFlags::ALLOW_FEEDBACK,
            )
            .unwrap();
        let before = ctrl_events(&r.events.borrow()).len();
        r.device.s_ctrl(&mut a, zoom, 160).unwrap();
        let events = ctrl_events(&r.events.borrow());
        let mut heard: Vec<u32> = events[before..].iter().map(|e| e.session).collect();
        heard.sort_unstable();
        assert_eq!(heard, vec![0, 1]);
        assert!(events[before..].iter().all(|e| e.value == 160));
        // The same value again is no event.
        let before = events.len();
        r.device.s_ctrl(&mut a, zoom, 160).unwrap();
        assert_eq!(ctrl_events(&r.events.borrow()).len(), before);

        // A flag change goes to everyone subscribed to the control whose flags changed, the
        // setter included, with the new flags.
        r.device
            .subscribe_event(
                &mut b,
                EventType::Ctrl(bindings::V4L2_CID_EXPOSURE_ABSOLUTE),
                SubscribeEventFlags::empty(),
            )
            .unwrap();
        let before = ctrl_events(&r.events.borrow()).len();
        r.device
            .s_ctrl(
                &mut b,
                bindings::V4L2_CID_EXPOSURE_AUTO,
                ExposureMode::Manual as i32,
            )
            .unwrap();
        let events = ctrl_events(&r.events.borrow());
        assert_eq!(
            &events[before..],
            &[CtrlEv {
                session: 1,
                id: bindings::V4L2_CID_EXPOSURE_ABSOLUTE,
                changes: CH_FLAGS,
                value: 333,
                flags: 0,
            }]
        );
        let before = events.len();
        r.device
            .s_ctrl(
                &mut b,
                bindings::V4L2_CID_EXPOSURE_AUTO,
                ExposureMode::Auto as i32,
            )
            .unwrap();
        let events = ctrl_events(&r.events.borrow());
        assert_eq!(events[before..].len(), 1);
        assert_eq!(events[before].flags, bindings::V4L2_CTRL_FLAG_INACTIVE);

        // `V4L2_EVENT_ALL` drops B's subscriptions; unsubscribing what was never subscribed is
        // fine; a closed session's subscriptions go with it.
        r.device
            .unsubscribe_event(
                &mut b,
                v4l2_event_subscription {
                    type_: bindings::V4L2_EVENT_ALL,
                    ..Default::default()
                },
            )
            .unwrap();
        r.device
            .unsubscribe_event(
                &mut b,
                v4l2_event_subscription {
                    type_: bindings::V4L2_EVENT_CTRL,
                    id: bindings::V4L2_CID_COLORFX,
                    ..Default::default()
                },
            )
            .unwrap();
        let before = ctrl_events(&r.events.borrow()).len();
        r.device.s_ctrl(&mut a, zoom, 170).unwrap();
        let events = ctrl_events(&r.events.borrow());
        assert_eq!(events[before..].len(), 1);
        assert_eq!(events[before].session, 0);
        assert!(r.device.subscriptions.contains_key(&0));
        close(&mut r.device, a);
        assert!(!r.device.subscriptions.contains_key(&0));
        close(&mut r.device, b);
    }

    /// The regions control round trip: its payload is read from and written into the guest
    /// memory the ioctl names, a buffer too small is `ENOSPC` with the size filled in for a
    /// get and `EFAULT` for a set, a rectangle outside the frame or an over-weighted one is
    /// `ERANGE`, and the regions reach the camera and open the stream.
    #[test]
    fn the_regions_control_round_trips_through_its_payload() {
        let mut r = rig();
        let mut s = session(&mut r.device);
        let words = REGION_WORDS * 4;
        let gpa = 4 * 0x1000u64;
        let region = Region {
            x: 2500,
            y: 2500,
            width: 5000,
            height: 5000,
            weight: 1000,
        };
        let put = |r: &Rig, at: u64, region: Region| {
            let mut mem = r.guest.memory.borrow_mut();
            let bytes: Vec<u8> = [
                region.x,
                region.y,
                region.width,
                region.height,
                region.weight,
            ]
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect();
            mem[at as usize..at as usize + words].copy_from_slice(&bytes);
        };
        let get = |r: &Rig, at: u64| -> Region {
            let mem = r.guest.memory.borrow();
            let w: Vec<u32> = mem[at as usize..at as usize + words]
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            Region {
                x: w[0],
                y: w[1],
                width: w[2],
                height: w[3],
                weight: w[4],
            }
        };

        // "How big": a get with no room answers the size and `ENOSPC`.
        let mut ctrls = ext_controls(1);
        let mut array = vec![ext_payload(VCAM_CID_AF_REGIONS, 0)];
        assert_eq!(
            r.device
                .g_ext_ctrls(&s, CtrlWhich::Current, &mut ctrls, &mut array, vec![]),
            Err(libc::ENOSPC)
        );
        let size = array[0].size;
        assert_eq!(size, words as u32);
        assert_eq!(ctrls.error_idx, 1);
        // The default is no region.
        let mut array = vec![ext_payload(VCAM_CID_AF_REGIONS, words as u32)];
        r.device
            .g_ext_ctrls(
                &s,
                CtrlWhich::Current,
                &mut ctrls,
                &mut array,
                vec![vec![SgEntry::new(gpa, words as u32)]],
            )
            .unwrap();
        assert_eq!(get(&r, gpa), Region::default());
        assert_eq!(*r.guest.live_mappings.borrow(), 0);

        // Set the centre quarter as the focus region.
        put(&r, gpa, region);
        let mut array = vec![ext_payload(VCAM_CID_AF_REGIONS, words as u32 + 4)];
        r.device
            .s_ext_ctrls(
                &mut s,
                CtrlWhich::Current,
                &mut ctrls,
                &mut array,
                vec![vec![SgEntry::new(gpa, words as u32 + 4)]],
            )
            .unwrap();
        let size = array[0].size;
        assert_eq!(size, words as u32, "the size is normalised");
        assert_eq!(
            r.log.lock().unwrap().backend_controls.last().unwrap(),
            &(false, vec![CameraControl::AfRegions(vec![region])])
        );
        // Read back into other guest memory.
        let gpa2 = gpa + 0x1000;
        let mut array = vec![ext_payload(VCAM_CID_AF_REGIONS, words as u32)];
        r.device
            .g_ext_ctrls(
                &s,
                CtrlWhich::Current,
                &mut ctrls,
                &mut array,
                vec![vec![SgEntry::new(gpa2, words as u32)]],
            )
            .unwrap();
        assert_eq!(get(&r, gpa2), region);
        // ... and the default is still empty.
        r.device
            .g_ext_ctrls(
                &s,
                CtrlWhich::Default,
                &mut ctrls,
                &mut array,
                vec![vec![SgEntry::new(gpa2, words as u32)]],
            )
            .unwrap();
        assert_eq!(get(&r, gpa2), Region::default());

        // Refused: a rectangle past the frame's edge, an over-weighted one, a payload too
        // small, a mapping shorter than the payload claims.
        put(
            &r,
            gpa,
            Region {
                x: 6000,
                width: 5000,
                ..region
            },
        );
        let mut array = vec![ext_payload(VCAM_CID_AF_REGIONS, words as u32)];
        assert_eq!(
            r.device.try_ext_ctrls(
                &s,
                CtrlWhich::Current,
                &mut ctrls,
                &mut array,
                vec![vec![SgEntry::new(gpa, words as u32)]]
            ),
            Err(libc::ERANGE)
        );
        assert_eq!(ctrls.error_idx, 0);
        put(
            &r,
            gpa,
            Region {
                weight: REGION_MAX_WEIGHT + 1,
                ..region
            },
        );
        assert_eq!(
            r.device.s_ext_ctrls(
                &mut s,
                CtrlWhich::Current,
                &mut ctrls,
                &mut array,
                vec![vec![SgEntry::new(gpa, words as u32)]]
            ),
            Err(libc::ERANGE)
        );
        put(&r, gpa, region);
        let mut short = vec![ext_payload(VCAM_CID_AF_REGIONS, 8)];
        assert_eq!(
            r.device.s_ext_ctrls(
                &mut s,
                CtrlWhich::Current,
                &mut ctrls,
                &mut short,
                vec![vec![SgEntry::new(gpa, 8)]]
            ),
            Err(libc::EFAULT)
        );
        let mut array = vec![ext_payload(VCAM_CID_AF_REGIONS, words as u32)];
        assert_eq!(
            r.device.s_ext_ctrls(
                &mut s,
                CtrlWhich::Current,
                &mut ctrls,
                &mut array,
                vec![vec![SgEntry::new(gpa, 8)]]
            ),
            Err(libc::EFAULT)
        );
        assert_eq!(
            *r.guest.live_mappings.borrow(),
            0,
            "every payload mapping was released"
        );
        // A camera that takes no white-balance regions has no such control.
        assert_eq!(
            r.device.query_ext_ctrl_raw(&s, VCAM_CID_AWB_REGIONS).err(),
            Some(libc::EINVAL)
        );

        // All zeros is "no region", and what is set opens the stream.
        put(&r, gpa, Region::default());
        let mut ae = vec![ext_payload(VCAM_CID_AE_REGIONS, words as u32)];
        r.device
            .s_ext_ctrls(
                &mut s,
                CtrlWhich::Current,
                &mut ctrls,
                &mut ae,
                vec![vec![SgEntry::new(gpa, words as u32)]],
            )
            .unwrap();
        assert_eq!(
            r.log.lock().unwrap().backend_controls.len(),
            1,
            "unchanged: not sent"
        );
        r.device.s_fmt(&mut s, QUEUE, format(64, 48)).unwrap();
        r.device
            .reqbufs(&mut s, QUEUE, MemoryType::Mmap, 1)
            .unwrap();
        r.device.streamon(&mut s, QUEUE).unwrap();
        {
            let log = r.log.lock().unwrap();
            let opened = &log.opened[0].controls;
            assert!(opened.contains(&CameraControl::AfRegions(vec![region])));
            assert!(opened.contains(&CameraControl::AeRegions(vec![Region::default()])));
        }
        close(&mut r.device, s);
    }

    /// A camera that describes nothing but sizes gets nothing but the state the backend can
    /// always report: no zoom, no flash, no exposure, no focus -- and a set of any of those is
    /// an unknown control.
    #[test]
    fn controls_follow_what_the_camera_offers() {
        let bare = CameraInfo {
            id: "1".into(),
            name: "Front camera".into(),
            sizes: vec![FrameSize::new(640, 480)],
            fps_ranges: vec![(30, 30)],
            ..Default::default()
        };
        let mut r = rig_with(FakeCamera {
            info: bare,
            ..camera()
        });
        let mut s = session(&mut r.device);
        let ids: Vec<u32> = walk(&mut r.device, &s, NEXT | NEXT_COMPOUND)
            .iter()
            .map(|q| q.id)
            .collect();
        assert_eq!(ids, vec![bindings::V4L2_CID_USER_CLASS, VCAM_CID_AE_STATE]);
        for id in [
            bindings::V4L2_CID_ZOOM_ABSOLUTE,
            bindings::V4L2_CID_FLASH_LED_MODE,
            bindings::V4L2_CID_EXPOSURE_AUTO,
            bindings::V4L2_CID_FOCUS_AUTO,
            bindings::V4L2_CID_AUTO_FOCUS_STATUS,
            VCAM_CID_ACTIVE_PHYSICAL_ID,
            VCAM_CID_AF_REGIONS,
        ] {
            assert_eq!(
                r.device.query_ext_ctrl_raw(&s, id).err(),
                Some(libc::EINVAL)
            );
            assert_eq!(r.device.s_ctrl(&mut s, id, 1).err(), Some(libc::EINVAL));
            assert_eq!(
                r.device
                    .subscribe_event(&mut s, EventType::Ctrl(id), SubscribeEventFlags::empty()),
                Err(libc::EINVAL)
            );
        }
        // No camera class, no flash class either.
        let mut none = ext_controls(0);
        assert_eq!(
            r.device.g_ext_ctrls(
                &s,
                CtrlWhich::Class(bindings::V4L2_CTRL_CLASS_CAMERA),
                &mut none,
                &mut vec![],
                vec![]
            ),
            Err(libc::EINVAL)
        );
        r.device
            .reqbufs(&mut s, QUEUE, MemoryType::Mmap, 1)
            .unwrap();
        r.device.streamon(&mut s, QUEUE).unwrap();
        assert!(r.log.lock().unwrap().opened[0].controls.is_empty());
        close(&mut r.device, s);
    }

    /// D73: a `REQBUFS` the pool cannot serve in full grants what it *can*, as vb2 does, instead
    /// of failing the whole request.
    ///
    /// `libavdevice`'s v4l2 input asks `REQBUFS(256)` and offers no way to lower it; the device
    /// clamps it to `MAX_BUFFERS` and used to allocate all 32 or nothing, which is why
    /// `ffmpeg -f v4l2 -video_size 3840x2160` failed `ENOMEM` on a 320 MiB pool while
    /// `REQBUFS(3)` at the same size worked (`logs/vpu_wp/B14-accept-B.md` section 2). Here the
    /// pool holds five buffers: five come back, they are ordinary buffers with the queue's own
    /// `sizeimage` on their plane, the books say five, and the sixth index does not exist.
    #[test]
    fn reqbufs_grants_what_the_pool_can_hold() {
        let mut r = rig();
        let mut s = session(&mut r.device);
        r.device.s_fmt(&mut s, QUEUE, format(64, 48)).unwrap();
        let size = 64 * 48 * 3 / 2;
        r.pool.holds(5, size as u64);

        let reply = r
            .device
            .reqbufs(&mut s, QUEUE, MemoryType::Mmap, 256)
            .unwrap();
        assert_eq!(reply.count, 5, "the pool held five, so five were granted");
        assert_eq!(s.buffers.len(), 5);
        assert_eq!(
            r.pool.used(),
            5 * size as u64,
            "the pool's books match the count the guest was told"
        );
        for index in 0..5 {
            let buf = r.device.querybuf(&s, QUEUE, index).unwrap();
            assert_eq!(buf.index(), index);
            // The R2 length rule: every granted buffer carries the mapping bound the queue's
            // format asks for, partial answer or not.
            assert_eq!(*buf.get_first_plane().length, size);
        }
        assert_eq!(r.device.querybuf(&s, QUEUE, 5).err(), Some(libc::EINVAL));

        // Five buffers are a queue that streams: the camera's floor is one.
        for index in 0..5 {
            r.device
                .qbuf(
                    &mut s,
                    mmap_buffer(index, size),
                    vec![],
                    PayloadValidity::ALL,
                )
                .unwrap();
        }
        r.device.streamon(&mut s, QUEUE).unwrap();
        assert_eq!(collect_frames(&mut r, &mut s, 5).len(), 5);
        r.device.streamoff(&mut s, QUEUE).unwrap();

        // And `REQBUFS(0)` gives every byte back.
        assert_eq!(
            r.device
                .reqbufs(&mut s, QUEUE, MemoryType::Mmap, 0)
                .unwrap()
                .count,
            0
        );
        assert!(s.buffers.is_empty());
        assert_eq!(r.pool.used(), 0);
        assert_eq!(*r.released.borrow(), 5);
        close(&mut r.device, s);
    }

    /// D73's other half: below the device's floor there is no queue to grant, so a pool that
    /// cannot serve even one buffer is still `ENOMEM` with nothing held -- the exhaustion the
    /// VMM's pool line names, and the behaviour every caller had before the partial rule.
    #[test]
    fn reqbufs_on_an_empty_pool_is_still_enomem_and_holds_nothing() {
        let mut r = rig();
        let mut s = session(&mut r.device);
        r.device.s_fmt(&mut s, QUEUE, format(64, 48)).unwrap();
        let size = 64 * 48 * 3 / 2;
        r.pool.holds(0, size as u64);

        assert_eq!(
            r.device.reqbufs(&mut s, QUEUE, MemoryType::Mmap, 256).err(),
            Some(libc::ENOMEM)
        );
        assert!(s.buffers.is_empty());
        assert_eq!(s.memory, None);
        assert_eq!(r.device.active_session, None);
        assert_eq!(r.pool.used(), 0, "nothing was kept");
        assert_eq!(*r.released.borrow(), 0, "and nothing was allocated to keep");

        // The session is untouched, so a request the pool can serve still works.
        r.pool.holds(2, size as u64);
        assert_eq!(
            r.device
                .reqbufs(&mut s, QUEUE, MemoryType::Mmap, 2)
                .unwrap()
                .count,
            2
        );
        assert_eq!(r.pool.used(), 2 * size as u64);
        close(&mut r.device, s);
    }

    /// D73 on `CREATE_BUFS`: the reply's `index` + `count` are what the guest indexes the new
    /// buffers by, so a short answer must report what was really created and the buffers must be
    /// there at those indices. A `CREATE_BUFS` that can create none is `ENOMEM`, and it leaves
    /// the buffers the queue already had alone.
    #[test]
    fn create_bufs_grants_what_the_pool_can_hold() {
        let mut r = rig();
        let mut s = session(&mut r.device);
        r.device.s_fmt(&mut s, QUEUE, format(64, 48)).unwrap();
        let size = 64 * 48 * 3 / 2;
        r.pool.holds(5, size as u64);

        assert_eq!(
            r.device
                .reqbufs(&mut s, QUEUE, MemoryType::Mmap, 2)
                .unwrap()
                .count,
            2
        );
        // Three of the eight asked for still fit.
        let reply = r
            .device
            .create_bufs(&mut s, 8, QUEUE, MemoryType::Mmap, format(64, 48))
            .unwrap();
        assert_eq!((reply.index, reply.count), (2, 3));
        assert_eq!(s.buffers.len(), 5);
        assert_eq!(r.pool.used(), 5 * size as u64);
        assert!(r.device.querybuf(&s, QUEUE, 4).is_ok());
        assert_eq!(r.device.querybuf(&s, QUEUE, 5).err(), Some(libc::EINVAL));

        // The pool is full: a set that can create nothing is refused, and the five stay.
        assert_eq!(
            r.device
                .create_bufs(&mut s, 4, QUEUE, MemoryType::Mmap, format(64, 48))
                .err(),
            Some(libc::ENOMEM)
        );
        assert_eq!(s.buffers.len(), 5);
        assert_eq!(r.pool.used(), 5 * size as u64);

        // `REQBUFS(0)` returns everything, `CREATE_BUFS` buffers included.
        r.device
            .reqbufs(&mut s, QUEUE, MemoryType::Mmap, 0)
            .unwrap();
        assert_eq!(r.pool.used(), 0);
        assert_eq!(*r.released.borrow(), 5);
        close(&mut r.device, s);
    }
}
