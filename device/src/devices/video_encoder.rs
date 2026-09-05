// Copyright 2026 The DroidVM Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

//! A stateful V4L2 memory-to-memory video encoder over a [`VideoEncoderBackend`]
//! (`VPU_DESIGN.md` §7.3), the mirror image of `video_decoder.rs`.
//!
//! This module is the V4L2 half of the encoder and knows nothing about how frames are actually
//! encoded: a [`VideoEncoderBackend`] enumerates what it can encode ([`EncoderCapabilities`]:
//! coded formats with their size, frame-rate and bitrate ranges, bitrate modes, profiles and
//! levels) and, once the guest streams both queues, opens a [`VideoEncoderBackendSession`] that
//! turns `OUTPUT` (raw NV12) frames into `CAPTURE` (bitstream) buffers. On DroidVM the backend
//! is `MediaCodecEncoderBackend` in crosvm over the Android MediaCodec NDK, so this crate stays
//! free of Android. Nothing about the codec is hard-coded here: every format, range, menu and
//! default the guest can query comes from the capabilities.
//!
//! # What the guest sees
//!
//! `V4L2_CAP_VIDEO_M2M_MPLANE | V4L2_CAP_STREAMING`, the kernel's stateful encoder interface
//! (`Documentation/userspace-api/media/v4l/dev-encoder.rst`): the `CAPTURE` queue advertises the
//! backend's coded formats (`ENUM_FMT`, `ENUM_FRAMESIZES` stepwise, `ENUM_FRAMEINTERVALS`), the
//! `OUTPUT` queue advertises **NV12 single-plane, tightly packed** only (`bytesperline = width`,
//! `sizeimage = w*h*3/2` for even dimensions). `S_FMT(CAPTURE)` selects the codec,
//! `S_FMT(OUTPUT)` the raw frame size, `S_PARM(OUTPUT)` the frame rate (an encoder has one, so
//! `G/S_PARM` work on `OUTPUT` and answer `ENOTTY` on `CAPTURE`, which is what
//! `v4l2-compliance` requires of a stateful encoder), `G/S_SELECTION(OUTPUT, CROP)` the visible
//! rectangle. The encoder's parameters are the `V4L2_CID_MPEG_VIDEO_*` controls
//! (`QUERYCTRL`/`QUERY_EXT_CTRL`/`QUERYMENU`/`G|S|TRY_EXT_CTRLS`), whose ranges and menus are the
//! backend's. The codec itself is created when **both queues stream** (in either order -- ffmpeg
//! starts `OUTPUT` first, GStreamer `CAPTURE` first) from the formats, frame rate and control
//! values current at that moment. Drain is `V4L2_ENC_CMD_STOP` (a `LAST` buffer, then
//! `V4L2_EVENT_EOS`); `STREAMOFF(OUTPUT)` pauses the encoder and keeps it, `STREAMOFF(CAPTURE)`
//! resets it so the next stream starts afresh, headers included.
//!
//! Buffers are guest-owned (`USERPTR`) or host-owned (`MMAP`, from the device's
//! [`VirtioMediaBufferAllocator`] -- the `media_host` pool on DroidVM): a raw `OUTPUT` frame is
//! guest-owned in the usual mode (`VPU_DESIGN.md` §2.1), a `CAPTURE` bitstream buffer is
//! host-owned unless `driver_owned_queues=all`. Only one encoding session per device instance
//! is allowed; a second session's `REQBUFS`/`CREATE_BUFS` is refused with `EBUSY`.
//!
//! # Threads and buffers
//!
//! Encoded frames arrive on a thread the backend owns while every ioctl runs on the device's
//! worker thread. The two meet the way they do in `camera.rs` and `video_decoder.rs`:
//!
//! * a raw frame the guest queues is *lent* to the backend ([`InputBuffer`]) as a raw read-only
//!   pointer; the backend owns those bytes until it reports
//!   [`EncoderEvent::InputBufferDone`], and the device holds the buffer's guest mapping until
//!   then (`VPU_DESIGN.md` §2.5);
//! * a `CAPTURE` buffer the guest queues is lent to the backend ([`OutputBuffer`]) as a writable
//!   pointer and the backend writes the bitstream into it and reports
//!   [`EncoderEvent::FrameEncoded`];
//! * `STREAMOFF`, `REQBUFS(0)`, a session close and an encoder error all stop the backend from
//!   touching a queue's buffers *before* the device unqueues or frees them
//!   ([`VideoEncoderBackendSession::flush`] for `OUTPUT`, [`VideoEncoderBackendSession::stop`]
//!   for `CAPTURE` and the whole session): a lent buffer is never released while the thread
//!   that may be reading or writing it is alive (review-m4 R1), and a guest mapping is always
//!   sized from the call that carries its scatter list (review-m4 R2).
//!
//! # The frame contract
//!
//! An [`InputBuffer`] is `len` bytes at `ptr` holding one tightly packed NV12 frame of the
//! [`EncoderConfig::coded_size`]: `height` rows of `width` luma bytes, then `height / 2` rows of
//! `width` interleaved Cb/Cr bytes; only [`EncoderConfig::visible_rect`] of it is to be encoded.
//! An [`OutputBuffer`] is `len` bytes the backend fills with one coded frame (or, in
//! `V4L2_MPEG_VIDEO_HEADER_MODE_SEPARATE`, with the stream headers alone) and reports with
//! `bytesused`, the frame's kind (which becomes `V4L2_BUF_FLAG_KEYFRAME` / `PFRAME` / `BFRAME`)
//! and the `OUTPUT` timestamp it came from (`V4L2_BUF_FLAG_TIMESTAMP_COPY`).

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
use v4l2r::bindings::v4l2_encoder_cmd;
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
use v4l2r::controls::codec::VideoBitrateMode;
use v4l2r::controls::codec::VideoHeaderMode;
use v4l2r::ioctl::BufferCapabilities;
use v4l2r::ioctl::BufferField;
use v4l2r::ioctl::BufferFlags;
use v4l2r::ioctl::CtrlId;
use v4l2r::ioctl::CtrlWhich;
use v4l2r::ioctl::EventType;
use v4l2r::ioctl::QueryCtrlFlags;
use v4l2r::ioctl::SelectionFlags;
use v4l2r::ioctl::SelectionTarget;
use v4l2r::ioctl::SelectionType;
use v4l2r::ioctl::SubscribeEventFlags;
use v4l2r::ioctl::V4l2Buffer;
use v4l2r::ioctl::V4l2PlanesWithBackingMut;
use v4l2r::memory::MemoryType;
use v4l2r::PixelFormat;
use v4l2r::QueueDirection;
use v4l2r::QueueType;

use crate::guest_mapping_errno;
use crate::ioctl::virtio_media_dispatch_ioctl;
use crate::ioctl::IoctlResult;
use crate::ioctl::PayloadValidity;
use crate::ioctl::VirtioMediaIoctlHandler;

/// Planes per buffer in every format these queues have. `QBUF` and `PREPARE_BUF` judge the
/// guest's payload description on these slots only: the rest of the plane array it sends is
/// scratch space it may leave dirty, which is what ffmpeg does (defect D21; the rule is
/// [`PayloadValidity::is_accepted_by`]).
const NUM_PLANES: usize = 1;
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

/// The one OUTPUT (raw) format accepted: Y plane then interleaved Cb/Cr, tightly packed.
pub const NV12: PixelFormat = PixelFormat::from_fourcc(b"NV12");
/// Most buffers on a queue, the usual V4L2 ceiling.
pub const MAX_BUFFERS: usize = 32;
/// The raw size a session starts at (fitted into the first coded format's range), so that
/// `G_FMT`/`G_SELECTION` answer a non-empty rectangle before the client sets anything.
const DEFAULT_CODED_SIZE: (u32, u32) = (640, 480);
/// The frame rate a session starts at, and what `S_PARM` with a `0/x` or `x/0` fraction means
/// (`v4l2-compliance` sends both and requires success); clamped into the codec's range.
const DEFAULT_FRAME_RATE: u32 = 30;
/// The smallest CAPTURE (bitstream) buffer a client may size through `S_FMT(CAPTURE)`; a smaller
/// or absent `sizeimage` gets the device's own default for the raw size.
const MIN_BITSTREAM_SIZE: u32 = 64 << 10;
/// Floor of the device's default bitstream buffer size (a keyframe at a high bitrate).
const DEFAULT_BITSTREAM_FLOOR: u32 = 256 << 10;

// ---------------------------------------------------------------------------------------------
// What a backend provides
// ---------------------------------------------------------------------------------------------

/// A `[min, max]` range with an alignment `step`, as `ENUM_FRAMESIZES` reports it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SizeRange {
    pub min: u32,
    pub max: u32,
    pub step: u32,
}

impl SizeRange {
    pub const fn new(min: u32, max: u32, step: u32) -> Self {
        Self { min, max, step }
    }

    /// `v` held into the range and aligned to the step, rounding up when that stays within the
    /// range (an encoder pads a picture up to its alignment; the visible rectangle says what
    /// is really there) and down otherwise.
    fn fit(&self, v: u32) -> u32 {
        // A backend's range is trusted for its meaning, not for its consistency: `clamp`
        // panics on `min > max`, and a panic aborts the VMM.
        let step = self.step.max(1);
        let lo = self.min.max(1);
        let hi = self.max.max(lo);
        let v = v.clamp(lo, hi);
        let up = v.div_ceil(step) * step;
        if up <= hi {
            up
        } else {
            ((v / step) * step).max(lo)
        }
    }

    /// Whether `v` is a size `fit` would return unchanged: in range and aligned.
    fn accepts(&self, v: u32) -> bool {
        v >= self.min && v <= self.max && v % self.step.max(1) == 0
    }
}

/// Frames per second an encoder sustains, `[min, max]`; `ENUM_FRAMEINTERVALS` reports
/// `1/max .. 1/min`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameRateRange {
    pub min: u32,
    pub max: u32,
}

/// An integer control's range and starting value, as `QUERYCTRL` reports them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ControlRange {
    pub min: i32,
    pub max: i32,
    pub default: i32,
}

impl ControlRange {
    pub const fn new(min: i32, max: i32, default: i32) -> Self {
        Self { min, max, default }
    }

    fn clamp(&self, v: i32) -> i32 {
        v.max(self.min).min(self.max.max(self.min))
    }
}

/// The quantiser range a codec accepts for its `MIN_QP` / `MAX_QP` controls (0..51 for 8-bit
/// H.264 and HEVC). `MIN_QP` starts at `min`, `MAX_QP` at `max`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QpRange {
    pub min: i32,
    pub max: i32,
}

/// One coded (compressed) format the backend produces on the `CAPTURE` queue, with everything
/// the guest can ask about it. The device advertises nothing of its own: everything comes from
/// here.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CodedFormat {
    /// The V4L2 CAPTURE fourcc: `H264`, `HEVC`, `VP80`, `VP90` or `AV10`.
    pub fourcc: PixelFormat,
    pub width: SizeRange,
    pub height: SizeRange,
    /// Frame rates the encoder sustains at any size of the range.
    pub frame_rate: FrameRateRange,
    /// `V4L2_CID_MPEG_VIDEO_BITRATE`, bits per second.
    pub bitrate: ControlRange,
    /// `V4L2_CID_MPEG_VIDEO_GOP_SIZE`, frames between keyframes (`0` = every frame is one).
    pub gop_size: ControlRange,
    /// `V4L2_CID_MPEG_VIDEO_BITRATE_MODE` menu items the codec accepts. The first is the
    /// default; the order is otherwise irrelevant.
    pub bitrate_modes: Vec<VideoBitrateMode>,
    /// Profiles, as the V4L2 menu values of the codec's profile control
    /// (`V4L2_MPEG_VIDEO_H264_PROFILE_*`, `_HEVC_PROFILE_*`, `_VP8_PROFILE_*`,
    /// `_VP9_PROFILE_*`). The first is the default. Empty: the codec has no profile control.
    pub profiles: Vec<i32>,
    /// Levels, as the V4L2 menu values of the codec's level control (H.264 and HEVC). The first
    /// is the default. Empty: no level control.
    pub levels: Vec<i32>,
    /// The quantiser range behind `MIN_QP` / `MAX_QP`, if the codec exposes one.
    pub qp: Option<QpRange>,
}

/// What an encoder can do, enumerated once at device creation (on Android by warming the
/// `AMediaCodecStore` up on a single thread, `VPU_DESIGN.md` §7.2/§7.3).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EncoderCapabilities {
    /// The coded formats, in `ENUM_FMT(CAPTURE)` order. The first is the format a session
    /// starts with.
    pub coded_formats: Vec<CodedFormat>,
    /// Least `OUTPUT` (raw) buffers the encoder needs queued to make progress; answered by
    /// `V4L2_CID_MIN_BUFFERS_FOR_OUTPUT`, which GStreamer reads to size its pool.
    pub min_output_buffers: u32,
}

impl EncoderCapabilities {
    fn coded_format(&self, fourcc: PixelFormat) -> Option<&CodedFormat> {
        self.coded_formats.iter().find(|f| f.fourcc == fourcc)
    }
}

/// The kernel's canonical `ENUM_FMT` description for a fourcc; `v4l2-compliance` checks it against
/// its own table (`v4l2-test-formats.cpp:271`).
fn fourcc_description(fourcc: PixelFormat) -> &'static [u8] {
    match &fourcc.to_fourcc() {
        b"H264" => b"H.264",
        b"HEVC" => b"HEVC",
        b"VP80" => b"VP8",
        b"VP90" => b"VP9",
        b"AV10" => b"AV1",
        b"NV12" => b"Y/UV 4:2:0",
        _ => b"Unknown",
    }
}

/// A pointer into a buffer the backend reads or fills, handed to the codec thread.
///
/// Raw pointers are not `Send`; this one is, because what it points at -- a host buffer from the
/// allocator or a guest mapping held for as long as the buffer is lent -- is plain shared memory
/// with no thread affinity, and because the device lends each buffer to exactly one backend
/// session and takes it back only after that session has stopped touching it.
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

/// An `OUTPUT` (raw frame) buffer lent to the backend to encode. The backend reads at most `len`
/// bytes at `ptr` -- one NV12 frame of the configured coded size -- and reports
/// [`EncoderEvent::InputBufferDone`] when done with them.
#[derive(Clone, Copy, Debug)]
pub struct InputBuffer {
    /// The V4L2 buffer index; comes back in [`EncoderEvent::InputBufferDone`].
    pub index: u32,
    /// Read-only pointer to the frame.
    pub ptr: SendPtr,
    /// Bytes at `ptr`: at least one NV12 frame of the coded size.
    pub len: usize,
    /// The frame's timestamp, copied to the `CAPTURE` buffer(s) encoded from it
    /// (`V4L2_BUF_FLAG_TIMESTAMP_COPY`).
    pub timestamp: bindings::timeval,
}

/// A `CAPTURE` (bitstream) buffer lent to the backend to encode into.
#[derive(Clone, Copy, Debug)]
pub struct OutputBuffer {
    /// The V4L2 buffer index; comes back in [`EncoderEvent::FrameEncoded`].
    pub index: u32,
    /// Writable pointer to the bitstream buffer.
    pub ptr: SendPtr,
    /// Bytes available at `ptr`.
    pub len: usize,
}

/// What a filled `CAPTURE` buffer holds; becomes the buffer's frame-type flag.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameKind {
    /// Stream headers alone (SPS/PPS, VPS/SPS/PPS), `V4L2_MPEG_VIDEO_HEADER_MODE_SEPARATE`'s
    /// first buffer. Carries no frame-type flag.
    Headers,
    /// A key (intra, IDR) frame: `V4L2_BUF_FLAG_KEYFRAME`.
    Key,
    /// A predicted frame: `V4L2_BUF_FLAG_PFRAME`.
    Inter,
    /// A bidirectionally predicted frame: `V4L2_BUF_FLAG_BFRAME`.
    Bidirectional,
}

/// Something the backend reports on its event path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EncoderEvent {
    /// The `OUTPUT` buffer at `index` is done being read and can be returned to the guest.
    InputBufferDone(u32),
    /// The `CAPTURE` buffer at `index` holds a coded frame (or the stream headers).
    FrameEncoded {
        index: u32,
        /// Bytes written, from the start of the buffer.
        bytesused: u32,
        /// The `OUTPUT` timestamp the frame came from.
        timestamp: bindings::timeval,
        kind: FrameKind,
        /// The last buffer of a drain: carries `V4L2_BUF_FLAG_LAST`, and may be empty
        /// (`bytesused == 0`, `kind` then irrelevant).
        is_last: bool,
    },
    /// The session failed and produces nothing more; the string is for the log.
    Error(String),
}

/// The colorimetry of the raw frames, as the guest declared it on `S_FMT(OUTPUT)`: raw V4L2
/// `v4l2_colorspace` / `v4l2_ycbcr_encoding` / `v4l2_quantization` / `v4l2_xfer_func` values,
/// for a backend that writes them into the stream's VUI.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Colorimetry {
    pub colorspace: u32,
    pub ycbcr_enc: u32,
    pub quantization: u32,
    pub xfer_func: u32,
}

/// Frames per second as a fraction, `num / den` (`30000 / 1001` for 29.97).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameRate {
    pub num: u32,
    pub den: u32,
}

/// Everything the codec is created with, gathered when both queues start streaming: the
/// formats, the frame rate and the control values the guest set until then. Every value is one
/// the backend declared it accepts (ranges clamped, menus held to the selected format's items).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncoderConfig {
    /// The CAPTURE fourcc selected by `S_FMT(CAPTURE)`.
    pub coded_format: PixelFormat,
    /// Size of the NV12 frames in every [`InputBuffer`] (`bytesperline = width`).
    pub coded_size: (u32, u32),
    /// The part of each frame to encode (`G/S_SELECTION(OUTPUT, CROP)`); the whole frame unless
    /// the client cropped.
    pub visible_rect: v4l2r::Rect,
    /// From `S_PARM(OUTPUT)`.
    pub frame_rate: FrameRate,
    /// `V4L2_CID_MPEG_VIDEO_BITRATE`, bits per second.
    pub bitrate: u32,
    pub bitrate_mode: VideoBitrateMode,
    /// `V4L2_CID_MPEG_VIDEO_GOP_SIZE`, frames.
    pub gop_size: u32,
    /// Whether the stream headers go into a `CAPTURE` buffer of their own (reported as
    /// [`FrameKind::Headers`]) or in front of the first frame.
    pub header_mode: VideoHeaderMode,
    /// `V4L2_CID_MPEG_VIDEO_PREPEND_SPSPPS_TO_IDR`: headers again in front of every keyframe.
    pub prepend_sps_pps_to_idr: bool,
    /// The selected format's profile control value, if it has one.
    pub profile: Option<i32>,
    /// The selected format's level control value, if it has one.
    pub level: Option<i32>,
    /// `(MIN_QP, MAX_QP)` of the selected format, if it has them.
    pub qp: Option<(i32, i32)>,
    pub colorimetry: Colorimetry,
}

/// An eventfd a session's worker polls. Bumped once per event; drained by the device before it
/// collects them, so a bump that lands in between leaves it readable. Same shape as
/// `camera.rs`'s `FrameSignal` and `video_decoder.rs`'s `DecoderSignal`.
pub struct EncoderSignal(OwnedFd);

impl EncoderSignal {
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

    /// Add one to the counter. Never blocks; `EAGAIN` (2^64 - 1 bumps without a drain) is ignored
    /// because it cannot happen at frame rates.
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

impl AsFd for EncoderSignal {
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.0.as_fd()
    }
}

/// The session's end of the wake-up path, cloned into whatever thread the backend runs. Bump it
/// after every [`EncoderEvent`] made available.
#[derive(Clone)]
pub struct EncoderSink(Arc<EncoderSignal>);

impl EncoderSink {
    pub fn signal(&self) {
        self.0.signal()
    }
}

/// An encoding session's backend: the actual codec. Every method here is called on the device's
/// worker thread and must return promptly; the codec runs on a thread of its own and reports back
/// through [`Self::take_events`], bumping the [`EncoderSink`] it was given at creation.
pub trait VideoEncoderBackendSession {
    /// Create the codec for `config` and start it. Called when both queues start streaming,
    /// and again after a `STREAMOFF(CAPTURE)` reset. Errors become the guest's `STREAMON`
    /// result.
    fn start(&mut self, config: &EncoderConfig) -> IoctlResult<()>;

    /// Lend an `OUTPUT` raw frame to be encoded. The backend owns the bytes until it reports
    /// [`EncoderEvent::InputBufferDone`] for the same index.
    fn encode(&mut self, buffer: InputBuffer) -> IoctlResult<()>;

    /// Lend a `CAPTURE` buffer to be encoded into. The backend owns the bytes until it reports
    /// an [`EncoderEvent::FrameEncoded`] for the same index, or until [`Self::stop`].
    fn use_as_capture(&mut self, buffer: OutputBuffer) -> IoctlResult<()>;

    /// `V4L2_CID_MPEG_VIDEO_BITRATE` changed while the codec runs (`setParameters`
    /// `video-bitrate` on MediaCodec).
    fn set_bitrate(&mut self, bitrate: u32) -> IoctlResult<()>;

    /// `V4L2_CID_MPEG_VIDEO_FORCE_KEY_FRAME`: the next frame encoded is a keyframe
    /// (`request-sync` on MediaCodec).
    fn force_keyframe(&mut self) -> IoctlResult<()>;

    /// `STREAMOFF(OUTPUT)`, and `V4L2_ENC_CMD_START` after a finished drain: drop every pending
    /// `OUTPUT` frame and be ready to take new ones. When this returns the backend touches no
    /// `OUTPUT` buffer any more (the device returns them all to the guest), and the codec is
    /// kept. On an async codec this is flush-then-start.
    fn flush(&mut self) -> IoctlResult<()>;

    /// `V4L2_ENC_CMD_STOP`: encode everything queued so far, then report the last `CAPTURE`
    /// buffer with [`EncoderEvent::FrameEncoded`] `is_last = true` (empty if no frame is left;
    /// if no `CAPTURE` buffer is lent at that moment, the next one lent).
    fn drain(&mut self) -> IoctlResult<()>;

    /// Tear the codec down and join its thread: `STREAMOFF(CAPTURE)`, `REQBUFS(0)`, close, an
    /// error. When this returns the backend touches nothing; lent buffers not yet reported are
    /// simply forgotten (the device re-queues the raw frames and returns the bitstream buffers).
    fn stop(&mut self);

    /// Every event since the last call, oldest first.
    fn take_events(&mut self) -> Vec<EncoderEvent>;
}

/// An encoder, as a device sees it: what it can encode, and how to open a session.
pub trait VideoEncoderBackend {
    type Session: VideoEncoderBackendSession;

    /// What this encoder can do. Enumerated once; must not change over the device's life.
    fn capabilities(&self) -> &EncoderCapabilities;

    /// Prepare a session with the given `id`. `sink` is what the session bumps when an event is
    /// pending. The codec itself is not created until [`VideoEncoderBackendSession::start`].
    fn new_session(&mut self, id: u32, sink: EncoderSink) -> IoctlResult<Self::Session>;

    /// Close and destroy `session`, joining its thread.
    fn close_session(&mut self, session: Self::Session);
}

// ---------------------------------------------------------------------------------------------
// The controls
// ---------------------------------------------------------------------------------------------

/// The kernel's menu strings (`drivers/media/v4l2-core/v4l2-ctrls-defs.c`, `v4l2_ctrl_get_menu`),
/// indexed by menu value.
const BITRATE_MODE_NAMES: [&str; 3] = ["Variable Bitrate", "Constant Bitrate", "Constant Quality"];
const HEADER_MODE_NAMES: [&str; 2] = ["Separate Buffer", "Joined With 1st Frame"];
const H264_PROFILE_NAMES: [&str; 18] = [
    "Baseline",
    "Constrained Baseline",
    "Main",
    "Extended",
    "High",
    "High 10",
    "High 422",
    "High 444 Predictive",
    "High 10 Intra",
    "High 422 Intra",
    "High 444 Intra",
    "CAVLC 444 Intra",
    "Scalable Baseline",
    "Scalable High",
    "Scalable High Intra",
    "Stereo High",
    "Multiview High",
    "Constrained High",
];
const H264_LEVEL_NAMES: [&str; 20] = [
    "1", "1b", "1.1", "1.2", "1.3", "2", "2.1", "2.2", "3", "3.1", "3.2", "4", "4.1", "4.2", "5",
    "5.1", "5.2", "6.0", "6.1", "6.2",
];
const HEVC_PROFILE_NAMES: [&str; 3] = ["Main", "Main Still Picture", "Main 10"];
const HEVC_LEVEL_NAMES: [&str; 13] = [
    "1", "2", "2.1", "3", "3.1", "4", "4.1", "5", "5.1", "5.2", "6", "6.1", "6.2",
];
const VPX_PROFILE_NAMES: [&str; 4] = ["0", "1", "2", "3"];

/// The class a control id belongs to (`V4L2_CTRL_ID2WHICH`).
fn ctrl_class(id: u32) -> u32 {
    id & 0x0fff_0000
}

/// What kind of control an entry of the table is, with its range.
#[derive(Clone, Debug)]
enum CtrlType {
    /// A `V4L2_CTRL_TYPE_CTRL_CLASS` marker.
    Class,
    Integer {
        range: ControlRange,
        read_only: bool,
    },
    /// `items` are `(value, name)` pairs, sorted by value; values in between are holes.
    Menu {
        items: Vec<(i32, &'static str)>,
        default: i32,
    },
    Boolean {
        default: bool,
    },
    /// Write-only, executes on write.
    Button,
}

/// One control the device answers for, built from the capabilities at device creation.
#[derive(Clone, Debug)]
struct CtrlDef {
    id: u32,
    name: &'static str,
    ty: CtrlType,
}

impl CtrlDef {
    fn v4l2_type(&self) -> u32 {
        match self.ty {
            CtrlType::Class => bindings::v4l2_ctrl_type_V4L2_CTRL_TYPE_CTRL_CLASS,
            CtrlType::Integer { .. } => bindings::v4l2_ctrl_type_V4L2_CTRL_TYPE_INTEGER,
            CtrlType::Menu { .. } => bindings::v4l2_ctrl_type_V4L2_CTRL_TYPE_MENU,
            CtrlType::Boolean { .. } => bindings::v4l2_ctrl_type_V4L2_CTRL_TYPE_BOOLEAN,
            CtrlType::Button => bindings::v4l2_ctrl_type_V4L2_CTRL_TYPE_BUTTON,
        }
    }

    fn flags(&self) -> u32 {
        match self.ty {
            // "You can neither read nor write these" (v4l2_ctrl_fill).
            CtrlType::Class => {
                bindings::V4L2_CTRL_FLAG_READ_ONLY | bindings::V4L2_CTRL_FLAG_WRITE_ONLY
            }
            CtrlType::Integer {
                read_only: true, ..
            } => bindings::V4L2_CTRL_FLAG_READ_ONLY,
            CtrlType::Button => {
                bindings::V4L2_CTRL_FLAG_WRITE_ONLY | bindings::V4L2_CTRL_FLAG_EXECUTE_ON_WRITE
            }
            _ => 0,
        }
    }

    fn read_only(&self) -> bool {
        self.flags() & bindings::V4L2_CTRL_FLAG_READ_ONLY != 0
    }

    /// `(minimum, maximum, step, default)`.
    fn bounds(&self) -> (i32, i32, i32, i32) {
        match &self.ty {
            CtrlType::Class | CtrlType::Button => (0, 0, 0, 0),
            CtrlType::Integer { range, .. } => (range.min, range.max, 1, range.default),
            CtrlType::Menu { items, default } => (
                items.first().map(|i| i.0).unwrap_or(0),
                items.last().map(|i| i.0).unwrap_or(0),
                1,
                *default,
            ),
            CtrlType::Boolean { default } => (0, 1, 1, *default as i32),
        }
    }

    fn default(&self) -> i32 {
        self.bounds().3
    }

    /// The name of menu item `index`, if the control is a menu and the item exists.
    fn menu_name(&self, index: i32) -> Option<&'static str> {
        match &self.ty {
            CtrlType::Menu { items, .. } => items.iter().find(|i| i.0 == index).map(|i| i.1),
            _ => None,
        }
    }

    /// What a value the guest sets becomes, or the errno refusing it: integers are clamped,
    /// booleans normalised, menu values must be an item (`ERANGE` outside the range, `EINVAL`
    /// in a hole), as the kernel's `validate_new` does.
    fn validate(&self, value: i32) -> IoctlResult<i32> {
        match &self.ty {
            CtrlType::Class => Err(libc::EINVAL),
            CtrlType::Integer { range, .. } => Ok(range.clamp(value)),
            CtrlType::Menu { items, .. } => {
                let (min, max, _, _) = self.bounds();
                if value < min || value > max {
                    return Err(libc::ERANGE);
                }
                if items.iter().any(|i| i.0 == value) {
                    Ok(value)
                } else {
                    Err(libc::EINVAL)
                }
            }
            CtrlType::Boolean { .. } => Ok((value != 0) as i32),
            CtrlType::Button => Ok(0),
        }
    }
}

/// A menu control from the values a backend accepts, held to the kernel's enum (`names`), sorted
/// and deduplicated; the first accepted value is the default. `None` if nothing is left.
fn menu_from(values: &[i32], names: &[&'static str]) -> Option<CtrlType> {
    let mut items: Vec<(i32, &'static str)> = Vec::new();
    for &v in values {
        match usize::try_from(v).ok().and_then(|i| names.get(i)) {
            Some(name) if !items.iter().any(|i| i.0 == v) => items.push((v, *name)),
            Some(_) => (),
            None => log::warn!(
                "encoder: menu value {} is not in the kernel's enum, dropped",
                v
            ),
        }
    }
    let default = values
        .iter()
        .copied()
        .find(|v| items.iter().any(|i| i.0 == *v))?;
    items.sort_by_key(|i| i.0);
    Some(CtrlType::Menu { items, default })
}

/// The union of several formats' ranges: the widest bounds, the first format's default.
fn union_range<'a>(ranges: impl Iterator<Item = &'a ControlRange>) -> Option<ControlRange> {
    let mut out: Option<ControlRange> = None;
    for r in ranges {
        out = Some(match out {
            None => *r,
            Some(o) => ControlRange::new(o.min.min(r.min), o.max.max(r.max), o.default),
        });
    }
    out
}

/// The control table for a set of capabilities, sorted by id (the `NEXT_CTRL` enumeration
/// order). Codec-specific controls (profile, level, QP) exist for every coded format the backend
/// offers, whatever the current `CAPTURE` format: GStreamer reads a codec's profile and level
/// with `G_CTRL` before it sets that codec on `CAPTURE` (`gstv4l2videoenc.c:577-597`).
fn controls_for(caps: &EncoderCapabilities) -> Vec<CtrlDef> {
    let mut out = vec![
        CtrlDef {
            id: bindings::V4L2_CID_USER_CLASS,
            name: "User Controls",
            ty: CtrlType::Class,
        },
        CtrlDef {
            id: bindings::V4L2_CID_MIN_BUFFERS_FOR_OUTPUT,
            name: "Min Number of Output Buffers",
            ty: CtrlType::Integer {
                range: ControlRange::new(
                    1,
                    MAX_BUFFERS as i32,
                    (caps.min_output_buffers.max(1) as i32).min(MAX_BUFFERS as i32),
                ),
                read_only: true,
            },
        },
        CtrlDef {
            id: bindings::V4L2_CID_CODEC_CLASS,
            name: "Codec Controls",
            ty: CtrlType::Class,
        },
        CtrlDef {
            id: bindings::V4L2_CID_MPEG_VIDEO_HEADER_MODE,
            name: "Sequence Header Mode",
            ty: CtrlType::Menu {
                items: HEADER_MODE_NAMES
                    .iter()
                    .enumerate()
                    .map(|(i, n)| (i as i32, *n))
                    .collect(),
                // Joined: a headers-only buffer confuses GStreamer's frame matching
                // (`gstv4l2videoenc.c:683-725`); ffmpeg asks for SEPARATE itself.
                default: VideoHeaderMode::JoinedWith1stFrame as i32,
            },
        },
        CtrlDef {
            id: bindings::V4L2_CID_MPEG_VIDEO_FORCE_KEY_FRAME,
            name: "Force Key Frame",
            ty: CtrlType::Button,
        },
        CtrlDef {
            id: bindings::V4L2_CID_MPEG_VIDEO_PREPEND_SPSPPS_TO_IDR,
            name: "Prepend SPS and PPS to IDR",
            ty: CtrlType::Boolean { default: false },
        },
    ];
    let formats = &caps.coded_formats;
    if let Some(range) = union_range(formats.iter().map(|f| &f.gop_size)) {
        out.push(CtrlDef {
            id: bindings::V4L2_CID_MPEG_VIDEO_GOP_SIZE,
            name: "Video GOP Size",
            ty: CtrlType::Integer {
                range,
                read_only: false,
            },
        });
    }
    if let Some(range) = union_range(formats.iter().map(|f| &f.bitrate)) {
        out.push(CtrlDef {
            id: bindings::V4L2_CID_MPEG_VIDEO_BITRATE,
            name: "Video Bitrate",
            ty: CtrlType::Integer {
                range,
                read_only: false,
            },
        });
    }
    let modes: Vec<i32> = formats
        .iter()
        .flat_map(|f| f.bitrate_modes.iter().map(|m| *m as i32))
        .collect();
    if let Some(ty) = menu_from(&modes, &BITRATE_MODE_NAMES) {
        out.push(CtrlDef {
            id: bindings::V4L2_CID_MPEG_VIDEO_BITRATE_MODE,
            name: "Video Bitrate Mode",
            ty,
        });
    }
    for f in formats {
        let (profile, level, qp, profile_names, level_names): (
            u32,
            u32,
            Option<(u32, u32)>,
            &[&'static str],
            &[&'static str],
        ) = match &f.fourcc.to_fourcc() {
            b"H264" => (
                bindings::V4L2_CID_MPEG_VIDEO_H264_PROFILE,
                bindings::V4L2_CID_MPEG_VIDEO_H264_LEVEL,
                Some((
                    bindings::V4L2_CID_MPEG_VIDEO_H264_MIN_QP,
                    bindings::V4L2_CID_MPEG_VIDEO_H264_MAX_QP,
                )),
                &H264_PROFILE_NAMES,
                &H264_LEVEL_NAMES,
            ),
            b"HEVC" => (
                bindings::V4L2_CID_MPEG_VIDEO_HEVC_PROFILE,
                bindings::V4L2_CID_MPEG_VIDEO_HEVC_LEVEL,
                Some((
                    bindings::V4L2_CID_MPEG_VIDEO_HEVC_MIN_QP,
                    bindings::V4L2_CID_MPEG_VIDEO_HEVC_MAX_QP,
                )),
                &HEVC_PROFILE_NAMES,
                &HEVC_LEVEL_NAMES,
            ),
            b"VP80" => (
                bindings::V4L2_CID_MPEG_VIDEO_VP8_PROFILE,
                0,
                None,
                &VPX_PROFILE_NAMES,
                &[],
            ),
            b"VP90" => (
                bindings::V4L2_CID_MPEG_VIDEO_VP9_PROFILE,
                0,
                None,
                &VPX_PROFILE_NAMES,
                &[],
            ),
            _ => continue,
        };
        if out.iter().any(|c| c.id == profile) {
            // The same fourcc twice: the first entry's controls stand.
            continue;
        }
        if let Some(ty) = menu_from(&f.profiles, profile_names) {
            out.push(CtrlDef {
                id: profile,
                name: match &f.fourcc.to_fourcc() {
                    b"H264" => "H264 Profile",
                    b"HEVC" => "HEVC Profile",
                    b"VP80" => "VP8 Profile",
                    _ => "VP9 Profile",
                },
                ty,
            });
        }
        if level != 0 {
            if let Some(ty) = menu_from(&f.levels, level_names) {
                out.push(CtrlDef {
                    id: level,
                    name: if level == bindings::V4L2_CID_MPEG_VIDEO_H264_LEVEL {
                        "H264 Level"
                    } else {
                        "HEVC Level"
                    },
                    ty,
                });
            }
        }
        if let (Some((min_id, max_id)), Some(qp)) = (qp, f.qp) {
            let h264 = min_id == bindings::V4L2_CID_MPEG_VIDEO_H264_MIN_QP;
            out.push(CtrlDef {
                id: min_id,
                name: if h264 {
                    "H264 Minimum QP Value"
                } else {
                    "HEVC Minimum QP Value"
                },
                ty: CtrlType::Integer {
                    range: ControlRange::new(qp.min, qp.max, qp.min),
                    read_only: false,
                },
            });
            out.push(CtrlDef {
                id: max_id,
                name: if h264 {
                    "H264 Maximum QP Value"
                } else {
                    "HEVC Maximum QP Value"
                },
                ty: CtrlType::Integer {
                    range: ControlRange::new(qp.min, qp.max, qp.max),
                    read_only: false,
                },
            });
        }
    }
    out.sort_by_key(|c| c.id);
    out
}

/// Whether a control may change while the codec runs (forwarded to the session); every other
/// control answers `EBUSY` then, as the kernel encoder interface allows ("Encoding Parameter
/// Changes").
fn settable_while_running(id: u32) -> bool {
    matches!(
        id,
        bindings::V4L2_CID_MPEG_VIDEO_BITRATE | bindings::V4L2_CID_MPEG_VIDEO_FORCE_KEY_FRAME
    )
}

/// Fill a `[u8; 32]`-shaped name field (`v4l2_queryctrl`, `v4l2_querymenu`).
fn copy_name(dst: &mut [u8], name: &str) {
    let n = name.len().min(dst.len() - 1);
    dst[..n].copy_from_slice(&name.as_bytes()[..n]);
}

/// Fill a `[c_char; 32]`-shaped name field (`v4l2_query_ext_ctrl`; `c_char` is `u8` on aarch64
/// and `i8` on x86-64).
fn copy_c_name(dst: &mut [std::os::raw::c_char], name: &str) {
    let n = name.len().min(dst.len() - 1);
    for (d, s) in dst.iter_mut().zip(name.as_bytes()[..n].iter()) {
        *d = *s as std::os::raw::c_char;
    }
}

// ---------------------------------------------------------------------------------------------
// The device
// ---------------------------------------------------------------------------------------------

/// Validated colorspace information for a format, propagated `OUTPUT` -> `CAPTURE` on an m2m
/// device (`v4l2-compliance`'s `testM2MFormats`). Same as `video_decoder.rs`'s.
#[derive(Debug, Clone, Copy)]
struct V4l2FormatColorspace {
    colorspace: u32,
    xfer_func: u32,
    ycbcr_enc: u32,
    quantization: u32,
}

impl Default for V4l2FormatColorspace {
    fn default() -> Self {
        Self {
            colorspace: bindings::v4l2_colorspace_V4L2_COLORSPACE_REC709,
            xfer_func: bindings::v4l2_xfer_func_V4L2_XFER_FUNC_DEFAULT,
            ycbcr_enc: bindings::v4l2_ycbcr_encoding_V4L2_YCBCR_ENC_DEFAULT,
            quantization: bindings::v4l2_quantization_V4L2_QUANTIZATION_DEFAULT,
        }
    }
}

impl V4l2FormatColorspace {
    /// Take the colorimetry from a guest-supplied format, keeping defaults for values a
    /// `v4l2_format` cannot carry back or that `v4l2-compliance` refuses for a non-JPEG codec.
    fn from_pix_mp(pix_mp: &bindings::v4l2_pix_format_mplane) -> Self {
        let default = Self::default();
        let usable = |v: u32, fallback: u32| if v == 0 || v >= 0xff { fallback } else { v };
        let colorspace = match pix_mp.colorspace {
            bindings::v4l2_colorspace_V4L2_COLORSPACE_BT878
            | bindings::v4l2_colorspace_V4L2_COLORSPACE_JPEG => default.colorspace,
            other => usable(other, default.colorspace),
        };
        Self {
            colorspace,
            // SAFETY: `ycbcr_enc` and `hsv_enc` are the same `__u8`; only the meaning differs.
            ycbcr_enc: usable(
                unsafe { pix_mp.__bindgen_anon_1.ycbcr_enc } as u32,
                default.ycbcr_enc,
            ),
            quantization: usable(pix_mp.quantization as u32, default.quantization),
            xfer_func: usable(pix_mp.xfer_func as u32, default.xfer_func),
        }
    }

    fn apply(self, pix_mp: &mut bindings::v4l2_pix_format_mplane) {
        pix_mp.colorspace = self.colorspace;
        pix_mp.__bindgen_anon_1 = bindings::v4l2_pix_format_mplane__bindgen_ty_1 {
            ycbcr_enc: self.ycbcr_enc as u8,
        };
        pix_mp.quantization = self.quantization as u8;
        pix_mp.xfer_func = self.xfer_func as u8;
    }

    fn colorimetry(self) -> Colorimetry {
        Colorimetry {
            colorspace: self.colorspace,
            ycbcr_enc: self.ycbcr_enc,
            quantization: self.quantization,
            xfer_func: self.xfer_func,
        }
    }
}

/// Streaming state of the two queues, following the kernel's encoder state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct StreamingState {
    output_streaming: bool,
    capture_streaming: bool,
}

impl StreamingState {
    fn running(&self) -> bool {
        self.output_streaming && self.capture_streaming
    }
}

/// Where a `V4L2_ENC_CMD_STOP` drain has got to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Drain {
    None,
    /// `STOP` issued; waiting for the backend's `LAST` buffer.
    Pending,
    /// The `LAST` buffer has been returned; new `OUTPUT` frames are held and `CAPTURE` buffers
    /// are not lent until `V4L2_ENC_CMD_START` or a `CAPTURE` restart.
    Done,
}

/// Bytes of one tightly packed NV12 frame of `width` x `height`. Odd dimensions round the chroma
/// plane up.
fn nv12_sizeimage(width: u32, height: u32) -> u32 {
    width * height + 2 * width.div_ceil(2) * height.div_ceil(2)
}

/// The bitstream buffer size the device picks for a raw size when the client does not.
fn default_bitstream_size(width: u32, height: u32) -> u32 {
    (nv12_sizeimage(width, height) / 2).max(DEFAULT_BITSTREAM_FLOOR)
}

/// Where a buffer's bytes live. Same two ownerships as `camera.rs` / `video_decoder.rs`.
enum Backing<GM> {
    /// Host-owned, from the allocator; mappable by the guest at `offset`.
    Host { buffer: HostBuffer, offset: u32 },
    /// Guest-owned; mapped from the guest's `USERPTR` SG list while the buffer is lent.
    Guest(Option<GM>),
}

struct Buffer<GM> {
    v4l2_buffer: V4l2Buffer,
    backing: Backing<GM>,
    /// Queued by the guest and not yet returned.
    queued: bool,
    /// Lent to the backend (implies `queued`).
    lent: bool,
    /// Bytes this buffer was created for: the `sizeimage` `REQBUFS`/`CREATE_BUFS` sized it with.
    size: u32,
    /// `(bytesused, length)` `PREPARE_BUF` accepted for this buffer, while it is prepared.
    prepared: Option<(u32, u32)>,
}

impl<GM: GuestMemoryRange> Buffer<GM> {
    /// Bytes the buffer can hold: the host buffer's length, or -- for a guest-owned one -- the
    /// length the guest declared, held to what the mapping actually covers.
    fn capacity(&self) -> u32 {
        match &self.backing {
            Backing::Host { buffer, .. } => buffer.len.min(u32::MAX as u64) as u32,
            Backing::Guest(mapping) => {
                let declared = *self.v4l2_buffer.get_first_plane().length;
                match mapping {
                    Some(mapping) => declared.min(mapping.len().min(u32::MAX as usize) as u32),
                    None => declared,
                }
            }
        }
    }

    fn data_ptr(&mut self) -> Option<*mut u8> {
        match &mut self.backing {
            Backing::Host { buffer, .. } => Some(buffer.as_mut_ptr()),
            Backing::Guest(Some(mapping)) => Some(mapping.as_mut_ptr()),
            Backing::Guest(None) => None,
        }
    }

    fn drop_guest_mapping(&mut self) {
        if let Backing::Guest(mapping) = &mut self.backing {
            *mapping = None;
        }
    }

    /// Return the buffer to the not-queued state, releasing any guest mapping. Only after the
    /// backend that may have been lent it has stopped touching it.
    fn unqueue(&mut self) {
        self.drop_guest_mapping();
        self.queued = false;
        self.lent = false;
        self.prepared = None;
        self.v4l2_buffer
            .clear_flags(BufferFlags::QUEUED | BufferFlags::PREPARED);
    }
}

/// One of the two queues.
struct Queue<GM> {
    /// Memory type the buffers were allocated with; `None` while there are none.
    memory: Option<MemoryType>,
    buffers: Vec<Buffer<GM>>,
    /// Indices queued but not yet lent to the backend, in queueing order. (Whether the queue is
    /// streaming lives in [`StreamingState`], shared by both queues.)
    pending: VecDeque<usize>,
}

impl<GM> Default for Queue<GM> {
    fn default() -> Self {
        Self {
            memory: None,
            buffers: Vec::new(),
            pending: VecDeque::new(),
        }
    }
}

/// Session data of [`VideoEncoder`].
pub struct VideoEncoderSession<GM, S> {
    id: u32,
    /// What the worker polls; the backend bumps it through an [`EncoderSink`].
    signal: Arc<EncoderSignal>,
    /// Declared before the queues on purpose: a session dropped without `close_session` joins
    /// the backend before any buffer it may still be touching goes away (review-m4 R1).
    backend: S,
    /// Whether [`VideoEncoderBackendSession::start`] has been called (the codec exists).
    codec_started: bool,
    state: StreamingState,
    drain: Drain,
    /// OUTPUT (raw frame) queue.
    input: Queue<GM>,
    /// CAPTURE (bitstream) queue.
    output: Queue<GM>,
    /// The coded format selected by `S_FMT(CAPTURE)`.
    coded_format: PixelFormat,
    /// The raw frame size, `S_FMT(OUTPUT)`, fitted to the coded format's range.
    coded_size: (u32, u32),
    /// The visible rectangle within the raw frame, `S_SELECTION(OUTPUT, CROP)`.
    crop: v4l2r::Rect,
    /// The frame interval `S_PARM(OUTPUT)` set, as `(numerator, denominator)` seconds.
    timeperframe: (u32, u32),
    /// The bitstream buffer size the client set through `S_FMT(CAPTURE)`, if it did.
    bitstream_size: Option<u32>,
    colorspace: V4l2FormatColorspace,
    /// Current control values, one per entry of the device's table (unused for class and
    /// button entries).
    ctrl_values: Vec<i32>,
    eos_subscribed: bool,
    /// The codec died; every ioctl that would touch it answers `ENODEV` until the guest closes.
    dead: bool,
    /// Sequence number of the next CAPTURE buffer.
    sequence: u32,
}

impl<GM, S> VirtioMediaDeviceSession for VideoEncoderSession<GM, S> {
    fn poll_fd(&self) -> Option<BorrowedFd> {
        Some(self.signal.as_fd())
    }
}

impl<GM, S> VideoEncoderSession<GM, S> {
    fn queue(&self, queue: QueueType) -> IoctlResult<&Queue<GM>> {
        match queue {
            QueueType::VideoOutputMplane => Ok(&self.input),
            QueueType::VideoCaptureMplane => Ok(&self.output),
            _ => Err(libc::EINVAL),
        }
    }

    fn queue_mut(&mut self, queue: QueueType) -> IoctlResult<&mut Queue<GM>> {
        match queue {
            QueueType::VideoOutputMplane => Ok(&mut self.input),
            QueueType::VideoCaptureMplane => Ok(&mut self.output),
            _ => Err(libc::EINVAL),
        }
    }

    fn has_buffers(&self) -> bool {
        !self.input.buffers.is_empty() || !self.output.buffers.is_empty()
    }

    fn streaming(&self, direction: QueueDirection) -> bool {
        match direction {
            QueueDirection::Output => self.state.output_streaming,
            QueueDirection::Capture => self.state.capture_streaming,
        }
    }

    /// The OUTPUT (NV12) `sizeimage` for the current raw size.
    fn raw_sizeimage(&self) -> u32 {
        nv12_sizeimage(self.coded_size.0, self.coded_size.1)
    }

    /// The CAPTURE (bitstream) `sizeimage`: the client's, or the device's default.
    fn bitstream_sizeimage(&self) -> u32 {
        self.bitstream_size
            .unwrap_or_else(|| default_bitstream_size(self.coded_size.0, self.coded_size.1))
    }

    fn sizeimage(&self, direction: QueueDirection) -> u32 {
        match direction {
            QueueDirection::Output => self.raw_sizeimage(),
            QueueDirection::Capture => self.bitstream_sizeimage(),
        }
    }

    /// The format of a queue as a single-plane multi-planar `v4l2_format`, `sizeimage` bytes per
    /// buffer (`CREATE_BUFS` may ask for more than a frame needs).
    fn format_sized(&self, direction: QueueDirection, sizeimage: u32) -> v4l2_format {
        let (pixelformat, bytesperline, queue) = match direction {
            QueueDirection::Output => (
                NV12.to_u32(),
                self.coded_size.0,
                QueueType::VideoOutputMplane,
            ),
            QueueDirection::Capture => {
                (self.coded_format.to_u32(), 0, QueueType::VideoCaptureMplane)
            }
        };
        let mut pix_mp = bindings::v4l2_pix_format_mplane {
            width: self.coded_size.0,
            height: self.coded_size.1,
            pixelformat,
            field: bindings::v4l2_field_V4L2_FIELD_NONE,
            num_planes: 1,
            ..Default::default()
        };
        self.colorspace.apply(&mut pix_mp);
        pix_mp.plane_fmt[0] = bindings::v4l2_plane_pix_format {
            sizeimage,
            bytesperline,
            ..Default::default()
        };
        v4l2_format {
            type_: queue as u32,
            fmt: bindings::v4l2_format__bindgen_ty_1 { pix_mp },
        }
    }

    fn format(&self, direction: QueueDirection) -> v4l2_format {
        self.format_sized(direction, self.sizeimage(direction))
    }

    fn full_rect(&self) -> v4l2r::Rect {
        v4l2r::Rect::new(0, 0, self.coded_size.0, self.coded_size.1)
    }

    /// `G/S_PARM(OUTPUT)`'s answer.
    fn streamparm(&self) -> v4l2_streamparm {
        v4l2_streamparm {
            type_: QueueType::VideoOutputMplane as u32,
            parm: bindings::v4l2_streamparm__bindgen_ty_1 {
                output: bindings::v4l2_outputparm {
                    capability: bindings::V4L2_CAP_TIMEPERFRAME,
                    outputmode: 0,
                    timeperframe: bindings::v4l2_fract {
                        numerator: self.timeperframe.0,
                        denominator: self.timeperframe.1,
                    },
                    extendedmode: 0,
                    writebuffers: 0,
                    reserved: [0; 4],
                },
            },
        }
    }
}

/// A stateful V4L2 video encoder over a [`VideoEncoderBackend`]. See the module documentation.
pub struct VideoEncoder<
    B: VideoEncoderBackend,
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
    /// The one session allowed to hold buffers: one encode at a time, `EBUSY` for a second
    /// (`v4l2-compliance` checks a second session is refused).
    active_session: Option<u32>,
    /// The controls, sorted by id, built from the capabilities.
    controls: Vec<CtrlDef>,
}

impl<B, Q, M, HM, A> VideoEncoder<B, Q, M, HM, A>
where
    B: VideoEncoderBackend,
    Q: VirtioMediaEventQueue,
    M: VirtioMediaGuestMemoryMapper,
    HM: VirtioMediaHostMemoryMapper,
    A: VirtioMediaBufferAllocator,
{
    pub fn new(backend: B, evt_queue: Q, mem: M, mapper: HM, allocator: A) -> Self {
        let controls = controls_for(backend.capabilities());
        Self {
            backend,
            evt_queue,
            mem,
            mmap_manager: MmapMappingManager::from(mapper),
            allocator,
            retired: RetiredBuffers::new(),
            active_session: None,
            controls,
        }
    }

    pub fn capabilities(&self) -> &EncoderCapabilities {
        self.backend.capabilities()
    }

    /// The coded format `S_FMT(CAPTURE)` selects for `fourcc`, or the first advertised format if
    /// the guest asked for one the backend does not have.
    fn adjust_coded_format(&self, fourcc: u32) -> IoctlResult<CodedFormat> {
        let caps = self.backend.capabilities();
        let asked = PixelFormat::from_u32(fourcc);
        caps.coded_format(asked)
            .or_else(|| caps.coded_formats.first())
            .cloned()
            .ok_or(libc::EINVAL)
    }

    fn current_format(
        &self,
        session: &VideoEncoderSession<M::GuestMemoryMapping, B::Session>,
    ) -> IoctlResult<&CodedFormat> {
        self.backend
            .capabilities()
            .coded_format(session.coded_format)
            .ok_or(libc::EINVAL)
    }

    // controls ---------------------------------------------------------------------------------

    fn control_index(&self, id: u32) -> Option<usize> {
        self.controls.binary_search_by_key(&id, |c| c.id).ok()
    }

    /// The control an id with query flags names: the exact one, or -- with `NEXT_CTRL` -- the
    /// first with a greater id. `NEXT_COMPOUND` alone finds nothing (there are none).
    fn query_control(&self, id: CtrlId, flags: QueryCtrlFlags) -> IoctlResult<&CtrlDef> {
        let id: u32 = id.into();
        if flags.contains(QueryCtrlFlags::NEXT) {
            self.controls.iter().find(|c| c.id > id).ok_or(libc::EINVAL)
        } else if flags.contains(QueryCtrlFlags::COMPOUND) {
            Err(libc::EINVAL)
        } else {
            self.control_index(id)
                .map(|i| &self.controls[i])
                .ok_or(libc::EINVAL)
        }
    }

    fn ctrl_value(
        &self,
        session: &VideoEncoderSession<M::GuestMemoryMapping, B::Session>,
        id: u32,
    ) -> Option<i32> {
        self.control_index(id).map(|i| session.ctrl_values[i])
    }

    /// What `G_CTRL` / `G_EXT_CTRLS` answer for a control: `EINVAL` for a class, `EACCES` for a
    /// write-only button, the value otherwise.
    fn read_control(
        &self,
        session: &VideoEncoderSession<M::GuestMemoryMapping, B::Session>,
        id: u32,
        defaults: bool,
    ) -> IoctlResult<i32> {
        let index = self.control_index(id).ok_or(libc::EINVAL)?;
        let def = &self.controls[index];
        match def.ty {
            CtrlType::Class => Err(libc::EINVAL),
            CtrlType::Button => Err(libc::EACCES),
            _ if defaults => Ok(def.default()),
            _ => Ok(session.ctrl_values[index]),
        }
    }

    /// What `value` becomes for a control, or the errno refusing it: `EACCES` for a read-only
    /// control, the type's own rule otherwise. When `applying`, the live-change rule too: a
    /// running codec takes only the controls that can change on the fly, the rest answer `EBUSY`
    /// (kernel encoder interface, "Encoding Parameter Changes").
    fn check_control(
        &self,
        session: &VideoEncoderSession<M::GuestMemoryMapping, B::Session>,
        id: u32,
        value: i32,
        applying: bool,
    ) -> IoctlResult<i32> {
        let index = self.control_index(id).ok_or(libc::EINVAL)?;
        let def = &self.controls[index];
        // A class marker is not a value at all (`EINVAL`, the kernel's `is_int` check) before
        // its read-only flag would say `EACCES`.
        if matches!(def.ty, CtrlType::Class) {
            return Err(libc::EINVAL);
        }
        if def.read_only() {
            return Err(libc::EACCES);
        }
        let value = def.validate(value)?;
        if applying
            && session.codec_started
            && !matches!(def.ty, CtrlType::Button)
            && session.ctrl_values[index] != value
            && !settable_while_running(id)
        {
            return Err(libc::EBUSY);
        }
        Ok(value)
    }

    /// Apply a value [`Self::check_control`] accepted: store it, and forward it to a running
    /// codec when it is one of the controls that change live.
    fn apply_control(
        &mut self,
        session: &mut VideoEncoderSession<M::GuestMemoryMapping, B::Session>,
        id: u32,
        value: i32,
    ) -> IoctlResult<i32> {
        let index = self.control_index(id).ok_or(libc::EINVAL)?;
        if session.dead {
            return Err(libc::ENODEV);
        }
        match self.controls[index].ty {
            CtrlType::Button => {
                if session.codec_started {
                    session.backend.force_keyframe()?;
                }
            }
            _ => {
                if session.codec_started
                    && session.ctrl_values[index] != value
                    && id == bindings::V4L2_CID_MPEG_VIDEO_BITRATE
                {
                    session.backend.set_bitrate(value.max(0) as u32)?;
                }
                session.ctrl_values[index] = value;
            }
        }
        Ok(value)
    }

    /// The read half of `G/S/TRY_EXT_CTRLS`: what every control of the array would answer or
    /// become, or the errno and the `error_idx` the kernel would report -- the failing control's
    /// index for `TRY`, `count` for `G` and `S`. `which` selects the values: current, default
    /// (`G` only) or a class every control must belong to.
    fn check_ext_ctrls(
        &self,
        session: &VideoEncoderSession<M::GuestMemoryMapping, B::Session>,
        op: ExtCtrlOp,
        which: CtrlWhich,
        ctrl_array: &[v4l2_ext_control],
    ) -> Result<Vec<i32>, (i32, u32)> {
        let count = ctrl_array.len() as u32;
        let fail_idx = |i: usize| {
            if op == ExtCtrlOp::Try {
                i as u32
            } else {
                count
            }
        };
        let (defaults, class) = match which {
            CtrlWhich::Current => (false, None),
            CtrlWhich::Default if op == ExtCtrlOp::Get => (true, None),
            CtrlWhich::Class(c) => (false, Some(c)),
            _ => return Err((libc::EINVAL, count)),
        };
        let mut values = Vec::with_capacity(ctrl_array.len());
        for (i, ctrl) in ctrl_array.iter().enumerate() {
            let id = ctrl.id;
            if class.is_some_and(|c| ctrl_class(id) != c) {
                return Err((libc::EINVAL, fail_idx(i)));
            }
            let anon = ctrl.__bindgen_anon_1;
            // SAFETY: the union's `value` member is the one a plain (payload-less) control
            // carries.
            let value = unsafe { anon.value };
            let result = match op {
                ExtCtrlOp::Get => self.read_control(session, id, defaults),
                ExtCtrlOp::Try => self.check_control(session, id, value, false),
                ExtCtrlOp::Set => self.check_control(session, id, value, true),
            };
            match result {
                Ok(v) => values.push(v),
                Err(e) => return Err((e, fail_idx(i))),
            }
        }
        Ok(values)
    }

    /// Write the values back into the array, `error_idx` cleared.
    fn write_back_ext_ctrls(
        ctrls: &mut v4l2_ext_controls,
        ctrl_array: &mut [v4l2_ext_control],
        values: &[i32],
    ) {
        for (ctrl, value) in ctrl_array.iter_mut().zip(values) {
            ctrl.__bindgen_anon_1 = bindings::v4l2_ext_control__bindgen_ty_1 { value: *value };
        }
        ctrls.error_idx = 0;
    }

    /// The `V4L2_EVENT_CTRL` event describing a control's current state.
    fn ctrl_event(
        &self,
        session: &VideoEncoderSession<M::GuestMemoryMapping, B::Session>,
        index: usize,
    ) -> bindings::v4l2_event {
        let def = &self.controls[index];
        let (minimum, maximum, step, default_value) = def.bounds();
        let value = match def.ty {
            CtrlType::Class | CtrlType::Button => 0,
            _ => session.ctrl_values[index],
        };
        bindings::v4l2_event {
            type_: bindings::V4L2_EVENT_CTRL,
            id: def.id,
            u: bindings::v4l2_event__bindgen_ty_1 {
                ctrl: bindings::v4l2_event_ctrl {
                    changes: bindings::V4L2_EVENT_CTRL_CH_VALUE,
                    type_: def.v4l2_type(),
                    __bindgen_anon_1: bindings::v4l2_event_ctrl__bindgen_ty_1 { value },
                    flags: def.flags(),
                    minimum,
                    maximum,
                    step,
                    default_value,
                },
            },
            ..Default::default()
        }
    }

    // the codec ---------------------------------------------------------------------------------

    /// What the codec is created with: the formats, the frame rate and the control values as
    /// they stand, held to what the backend declared for the selected format.
    fn config(
        &self,
        session: &VideoEncoderSession<M::GuestMemoryMapping, B::Session>,
    ) -> IoctlResult<EncoderConfig> {
        let fmt = self.current_format(session)?;
        let value = |id: u32| self.ctrl_value(session, id);
        let bitrate = fmt
            .bitrate
            .clamp(value(bindings::V4L2_CID_MPEG_VIDEO_BITRATE).unwrap_or(fmt.bitrate.default))
            .max(0) as u32;
        let gop_size = fmt
            .gop_size
            .clamp(value(bindings::V4L2_CID_MPEG_VIDEO_GOP_SIZE).unwrap_or(fmt.gop_size.default))
            .max(0) as u32;
        let bitrate_mode = value(bindings::V4L2_CID_MPEG_VIDEO_BITRATE_MODE)
            .and_then(VideoBitrateMode::n)
            .filter(|m| fmt.bitrate_modes.contains(m))
            .or_else(|| {
                log::info!(
                    "encoder: bitrate mode not offered by {}, using its default",
                    fmt.fourcc
                );
                fmt.bitrate_modes.first().copied()
            })
            .unwrap_or(VideoBitrateMode::VariableBitrate);
        let header_mode = value(bindings::V4L2_CID_MPEG_VIDEO_HEADER_MODE)
            .and_then(VideoHeaderMode::n)
            .unwrap_or(VideoHeaderMode::JoinedWith1stFrame);
        let prepend_sps_pps_to_idr =
            value(bindings::V4L2_CID_MPEG_VIDEO_PREPEND_SPSPPS_TO_IDR).unwrap_or(0) != 0;
        let (profile_id, level_id, qp_ids) = match &fmt.fourcc.to_fourcc() {
            b"H264" => (
                bindings::V4L2_CID_MPEG_VIDEO_H264_PROFILE,
                bindings::V4L2_CID_MPEG_VIDEO_H264_LEVEL,
                Some((
                    bindings::V4L2_CID_MPEG_VIDEO_H264_MIN_QP,
                    bindings::V4L2_CID_MPEG_VIDEO_H264_MAX_QP,
                )),
            ),
            b"HEVC" => (
                bindings::V4L2_CID_MPEG_VIDEO_HEVC_PROFILE,
                bindings::V4L2_CID_MPEG_VIDEO_HEVC_LEVEL,
                Some((
                    bindings::V4L2_CID_MPEG_VIDEO_HEVC_MIN_QP,
                    bindings::V4L2_CID_MPEG_VIDEO_HEVC_MAX_QP,
                )),
            ),
            b"VP80" => (bindings::V4L2_CID_MPEG_VIDEO_VP8_PROFILE, 0, None),
            b"VP90" => (bindings::V4L2_CID_MPEG_VIDEO_VP9_PROFILE, 0, None),
            _ => (0, 0, None),
        };
        let pick = |id: u32, choices: &[i32]| -> Option<i32> {
            if choices.is_empty() {
                return None;
            }
            value(id)
                .filter(|v| choices.contains(v))
                .or_else(|| choices.first().copied())
        };
        let profile = pick(profile_id, &fmt.profiles);
        let level = pick(level_id, &fmt.levels);
        let qp = match (qp_ids, fmt.qp) {
            (Some((min_id, max_id)), Some(range)) => {
                let min = value(min_id)
                    .unwrap_or(range.min)
                    .clamp(range.min, range.max);
                let max = value(max_id)
                    .unwrap_or(range.max)
                    .clamp(range.min, range.max);
                Some((min.min(max), max))
            }
            _ => None,
        };
        Ok(EncoderConfig {
            coded_format: fmt.fourcc,
            coded_size: session.coded_size,
            visible_rect: session.crop,
            frame_rate: FrameRate {
                num: session.timeperframe.1,
                den: session.timeperframe.0,
            },
            bitrate,
            bitrate_mode,
            gop_size,
            header_mode,
            prepend_sps_pps_to_idr,
            profile,
            level,
            qp,
            colorimetry: session.colorspace.colorimetry(),
        })
    }

    /// Create the codec if both queues stream and it does not exist yet, then feed it every
    /// buffer queued so far.
    fn maybe_start_codec(
        &mut self,
        session: &mut VideoEncoderSession<M::GuestMemoryMapping, B::Session>,
    ) -> IoctlResult<()> {
        if !session.state.running() {
            return Ok(());
        }
        if !session.codec_started {
            let config = self.config(session)?;
            session.backend.start(&config)?;
            session.codec_started = true;
        }
        self.feed_pending(session)
    }

    /// Lend every pending buffer of both queues, if the codec runs and no finished drain holds
    /// them back.
    fn feed_pending(
        &mut self,
        session: &mut VideoEncoderSession<M::GuestMemoryMapping, B::Session>,
    ) -> IoctlResult<()> {
        if !session.codec_started || !session.state.running() || session.drain == Drain::Done {
            return Ok(());
        }
        while let Some(index) = session.output.pending.pop_front() {
            if let Err(e) = Self::lend_output(session, index) {
                self.end_session(session, &format!("the backend refused a buffer: errno {e}"));
                return Err(libc::EIO);
            }
        }
        while let Some(index) = session.input.pending.pop_front() {
            if let Err(e) = Self::lend_input(session, index) {
                self.end_session(session, &format!("the backend refused a frame: errno {e}"));
                return Err(libc::EIO);
            }
        }
        Ok(())
    }

    /// Lend a queued OUTPUT buffer to the backend to encode.
    fn lend_input(
        session: &mut VideoEncoderSession<M::GuestMemoryMapping, B::Session>,
        index: usize,
    ) -> IoctlResult<()> {
        let entry = session.input.buffers.get_mut(index).ok_or(libc::EINVAL)?;
        let timestamp = entry.v4l2_buffer.timestamp();
        let len = entry.capacity() as usize;
        let ptr = entry.data_ptr().ok_or(libc::EIO)?;
        session.backend.encode(InputBuffer {
            index: index as u32,
            ptr: SendPtr(ptr),
            len,
            timestamp,
        })?;
        entry.lent = true;
        Ok(())
    }

    /// Lend a queued CAPTURE buffer to the backend to encode into.
    fn lend_output(
        session: &mut VideoEncoderSession<M::GuestMemoryMapping, B::Session>,
        index: usize,
    ) -> IoctlResult<()> {
        let entry = session.output.buffers.get_mut(index).ok_or(libc::EINVAL)?;
        let len = entry.capacity() as usize;
        let ptr = entry.data_ptr().ok_or(libc::EIO)?;
        session.backend.use_as_capture(OutputBuffer {
            index: index as u32,
            ptr: SendPtr(ptr),
            len,
        })?;
        entry.lent = true;
        Ok(())
    }

    /// Tear the codec down (`STREAMOFF(CAPTURE)`, `REQBUFS(0)`, close): join it, return the raw
    /// frames it finished to the guest, and re-queue the ones it did not get to, so that the
    /// next start encodes them (the kernel's "Reset").
    fn stop_codec(&mut self, session: &mut VideoEncoderSession<M::GuestMemoryMapping, B::Session>) {
        if !session.codec_started {
            return;
        }
        session.backend.stop();
        session.codec_started = false;
        for event in session.backend.take_events() {
            if let EncoderEvent::InputBufferDone(index) = event {
                self.handle_event(session, EncoderEvent::InputBufferDone(index));
            }
        }
        for (index, buffer) in session.input.buffers.iter_mut().enumerate() {
            if buffer.lent {
                buffer.lent = false;
                session.input.pending.push_back(index);
            }
        }
        for buffer in session.output.buffers.iter_mut() {
            buffer.lent = false;
        }
    }

    /// Drop every buffer of a queue, returning host buffers to the allocator (or holding them
    /// until the guest unmaps them) and releasing guest mappings. The backend must have stopped
    /// touching the queue already.
    fn free_buffers(&mut self, queue: &mut Queue<M::GuestMemoryMapping>) {
        queue.pending.clear();
        queue.memory = None;
        for buffer in queue.buffers.drain(..) {
            if let Backing::Host { buffer, offset } = buffer.backing {
                self.retired
                    .retire(&mut self.mmap_manager, &mut self.allocator, offset, buffer);
            }
        }
    }

    /// Append `count` buffers of `sizeimage` bytes to `queue`. All or nothing.
    fn add_buffers(
        &mut self,
        queue: &mut Queue<M::GuestMemoryMapping>,
        queue_type: QueueType,
        memory: MemoryType,
        count: usize,
        sizeimage: u32,
    ) -> IoctlResult<()> {
        let first = queue.buffers.len();
        let mut added: Vec<Buffer<M::GuestMemoryMapping>> = Vec::with_capacity(count);

        for index in first..first + count {
            let mut v4l2_buffer = V4l2Buffer::new(queue_type, index as u32, memory);
            v4l2_buffer.set_field(BufferField::None);
            // An encoder copies the OUTPUT timestamp to the CAPTURE buffer; both queues are
            // `TIMESTAMP_COPY` (kernel encoder interface, "Encoding").
            v4l2_buffer.set_flags(BufferFlags::TIMESTAMP_COPY);

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

        queue.buffers.extend(added);
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

    /// The codec died: unqueue every buffer, mark the session dead, tell the guest.
    fn end_session(
        &mut self,
        session: &mut VideoEncoderSession<M::GuestMemoryMapping, B::Session>,
        why: &str,
    ) {
        log::error!("encoder: session {} ends: {}", session.id, why);
        session.backend.stop();
        session.codec_started = false;
        session.input.pending.clear();
        session.output.pending.clear();
        for buffer in session
            .input
            .buffers
            .iter_mut()
            .chain(session.output.buffers.iter_mut())
        {
            buffer.unqueue();
        }
        session.dead = true;
        self.evt_queue.send_error(session.id, libc::ENODEV);
    }

    /// One backend event: turn it into V4L2 events / buffer state.
    fn handle_event(
        &mut self,
        session: &mut VideoEncoderSession<M::GuestMemoryMapping, B::Session>,
        event: EncoderEvent,
    ) {
        match event {
            EncoderEvent::InputBufferDone(index) => {
                let Some(entry) = session.input.buffers.get_mut(index as usize) else {
                    log::error!("encoder: no OUTPUT buffer {} to return", index);
                    return;
                };
                if !entry.lent {
                    return;
                }
                // The backend is done reading the frame: drop the guest mapping (§2.5) and
                // tell the guest.
                entry.unqueue();
                let event = entry.v4l2_buffer.clone();
                self.evt_queue
                    .send_event(V4l2Event::DequeueBuffer(DequeueBufferEvent::new(
                        session.id, event,
                    )));
            }
            EncoderEvent::FrameEncoded {
                index,
                bytesused,
                timestamp,
                kind,
                is_last,
            } => {
                let seq = session.sequence;
                let Some(entry) = session.output.buffers.get_mut(index as usize) else {
                    log::error!("encoder: no CAPTURE buffer {} to return", index);
                    return;
                };
                if !entry.lent {
                    return;
                }
                let capacity = entry.capacity();
                // The bitstream is written: drop the guest mapping (a shadowed one is written
                // back here) before the guest hears the buffer is done.
                entry.unqueue();
                let plane = entry.v4l2_buffer.get_first_plane_mut();
                *plane.bytesused = bytesused.min(capacity);
                entry.v4l2_buffer.set_timestamp(timestamp);
                let mut flags = BufferFlags::TIMESTAMP_COPY;
                if bytesused > 0 {
                    flags |= match kind {
                        FrameKind::Headers => BufferFlags::empty(),
                        FrameKind::Key => BufferFlags::KEYFRAME,
                        FrameKind::Inter => BufferFlags::PFRAME,
                        FrameKind::Bidirectional => BufferFlags::BFRAME,
                    };
                }
                if is_last {
                    flags |= BufferFlags::LAST;
                }
                entry.v4l2_buffer.set_flags(flags);
                entry.v4l2_buffer.set_sequence(seq);
                session.sequence = session.sequence.wrapping_add(1);
                let event = entry.v4l2_buffer.clone();
                self.evt_queue
                    .send_event(V4l2Event::DequeueBuffer(DequeueBufferEvent::new(
                        session.id, event,
                    )));
                if is_last {
                    session.drain = Drain::Done;
                    if session.eos_subscribed {
                        self.evt_queue
                            .send_event(V4l2Event::Event(SessionEvent::new(
                                session.id,
                                bindings::v4l2_event {
                                    type_: bindings::V4L2_EVENT_EOS,
                                    ..Default::default()
                                },
                            )));
                    }
                }
            }
            EncoderEvent::Error(reason) => {
                self.end_session(session, &reason);
            }
        }
    }
}

/// Which of the three ext-control ioctls is being served.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ExtCtrlOp {
    Get,
    Set,
    Try,
}

impl<B, Q, M, HM, A, Reader, Writer> VirtioMediaDevice<Reader, Writer>
    for VideoEncoder<B, Q, M, HM, A>
where
    B: VideoEncoderBackend,
    Q: VirtioMediaEventQueue,
    M: VirtioMediaGuestMemoryMapper,
    HM: VirtioMediaHostMemoryMapper,
    A: VirtioMediaBufferAllocator,
    Reader: ReadFromDescriptorChain,
    Writer: WriteToDescriptorChain,
{
    type Session = VideoEncoderSession<M::GuestMemoryMapping, B::Session>;

    fn new_session(&mut self, session_id: u32) -> Result<Self::Session, i32> {
        let caps = self.backend.capabilities();
        let first = caps.coded_formats.first().ok_or_else(|| {
            log::error!("encoder: backend has no coded format");
            libc::ENODEV
        })?;
        let coded_size = (
            first.width.fit(DEFAULT_CODED_SIZE.0),
            first.height.fit(DEFAULT_CODED_SIZE.1),
        );
        let fourcc = first.fourcc;
        let min_fps = first.frame_rate.min.max(1);
        let fps = DEFAULT_FRAME_RATE.clamp(min_fps, first.frame_rate.max.max(min_fps));
        let ctrl_values = self.controls.iter().map(|c| c.default()).collect();
        let signal = Arc::new(EncoderSignal::new()?);
        let backend = self
            .backend
            .new_session(session_id, EncoderSink(Arc::clone(&signal)))?;
        Ok(VideoEncoderSession {
            id: session_id,
            signal,
            backend,
            codec_started: false,
            state: StreamingState {
                output_streaming: false,
                capture_streaming: false,
            },
            drain: Drain::None,
            input: Queue::default(),
            output: Queue::default(),
            coded_format: fourcc,
            coded_size,
            crop: v4l2r::Rect::new(0, 0, coded_size.0, coded_size.1),
            timeperframe: (1, fps),
            bitstream_size: None,
            colorspace: Default::default(),
            ctrl_values,
            eos_subscribed: false,
            dead: false,
            sequence: 0,
        })
    }

    fn close_session(&mut self, mut session: Self::Session) {
        if self.active_session == Some(session.id) {
            self.active_session = None;
        }
        // The backend first, so no thread is touching a buffer that goes away below.
        session.backend.stop();
        session.codec_started = false;
        self.free_buffers(&mut session.input);
        self.free_buffers(&mut session.output);
        self.backend.close_session(session.backend);
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
            .input
            .buffers
            .iter()
            .chain(session.output.buffers.iter())
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

    /// The session's eventfd is readable: collect what the backend has produced.
    fn process_events(&mut self, session: &mut Self::Session) -> Result<(), i32> {
        session.signal.drain();
        if session.dead {
            return Ok(());
        }
        let events = session.backend.take_events();
        for event in events {
            self.handle_event(session, event);
            if session.dead {
                break;
            }
        }
        Ok(())
    }
}

impl<B, Q, M, HM, A> VirtioMediaIoctlHandler for VideoEncoder<B, Q, M, HM, A>
where
    B: VideoEncoderBackend,
    Q: VirtioMediaEventQueue,
    M: VirtioMediaGuestMemoryMapper,
    HM: VirtioMediaHostMemoryMapper,
    A: VirtioMediaBufferAllocator,
{
    type Session = VideoEncoderSession<M::GuestMemoryMapping, B::Session>;

    /// `CAPTURE`: the backend's coded formats; `OUTPUT`: NV12 (kernel encoder interface,
    /// "Querying Capabilities" 1-2).
    fn enum_fmt(
        &mut self,
        _session: &Self::Session,
        queue: QueueType,
        index: u32,
    ) -> IoctlResult<v4l2_fmtdesc> {
        let (fourcc, flags) = match queue {
            QueueType::VideoCaptureMplane => {
                let f = self
                    .backend
                    .capabilities()
                    .coded_formats
                    .get(index as usize)
                    .ok_or(libc::EINVAL)?;
                (f.fourcc, bindings::V4L2_FMT_FLAG_COMPRESSED)
            }
            QueueType::VideoOutputMplane => {
                if index != 0 {
                    return Err(libc::EINVAL);
                }
                (NV12, 0)
            }
            _ => return Err(libc::EINVAL),
        };
        let mut desc = v4l2_fmtdesc {
            index,
            type_: queue as u32,
            flags,
            pixelformat: fourcc.to_u32(),
            ..Default::default()
        };
        let description = fourcc_description(fourcc);
        desc.description[..description.len()].copy_from_slice(description);
        Ok(desc)
    }

    /// Stepwise frame sizes: a coded format's own; NV12's follow the coded format currently set
    /// on `CAPTURE` (kernel encoder interface, "Querying Capabilities" 3).
    fn enum_framesizes(
        &mut self,
        session: &Self::Session,
        index: u32,
        pixel_format: u32,
    ) -> IoctlResult<v4l2_frmsizeenum> {
        if index != 0 {
            return Err(libc::EINVAL);
        }
        let caps = self.backend.capabilities();
        let asked = PixelFormat::from_u32(pixel_format);
        let coded = if asked == NV12 {
            caps.coded_format(session.coded_format)
        } else {
            caps.coded_format(asked)
        }
        .ok_or(libc::EINVAL)?;
        Ok(v4l2_frmsizeenum {
            index: 0,
            pixel_format,
            type_: bindings::v4l2_frmsizetypes_V4L2_FRMSIZE_TYPE_STEPWISE,
            __bindgen_anon_1: bindings::v4l2_frmsizeenum__bindgen_ty_1 {
                stepwise: bindings::v4l2_frmsize_stepwise {
                    min_width: coded.width.min,
                    max_width: coded.width.max,
                    step_width: coded.width.step.max(1),
                    min_height: coded.height.min,
                    max_height: coded.height.max,
                    step_height: coded.height.step.max(1),
                },
            },
            ..Default::default()
        })
    }

    /// The frame-rate range of a format at a size it accepts: one continuous interval
    /// `1/max_fps .. 1/min_fps` (a single discrete one when the range is one rate). A size the
    /// format does not accept is `EINVAL`, as `v4l2-compliance` probes for.
    fn enum_frameintervals(
        &mut self,
        session: &Self::Session,
        index: u32,
        pixel_format: u32,
        width: u32,
        height: u32,
    ) -> IoctlResult<v4l2_frmivalenum> {
        if index != 0 {
            return Err(libc::EINVAL);
        }
        let caps = self.backend.capabilities();
        let asked = PixelFormat::from_u32(pixel_format);
        let coded = if asked == NV12 {
            caps.coded_format(session.coded_format)
        } else {
            caps.coded_format(asked)
        }
        .ok_or(libc::EINVAL)?;
        if !coded.width.accepts(width) || !coded.height.accepts(height) {
            return Err(libc::EINVAL);
        }
        let min_fps = coded.frame_rate.min.max(1);
        let max_fps = coded.frame_rate.max.max(min_fps);
        let fract = |den: u32| bindings::v4l2_fract {
            numerator: 1,
            denominator: den,
        };
        let (type_, anon) = if min_fps == max_fps {
            (
                bindings::v4l2_frmivaltypes_V4L2_FRMIVAL_TYPE_DISCRETE,
                bindings::v4l2_frmivalenum__bindgen_ty_1 {
                    discrete: fract(max_fps),
                },
            )
        } else {
            (
                bindings::v4l2_frmivaltypes_V4L2_FRMIVAL_TYPE_CONTINUOUS,
                bindings::v4l2_frmivalenum__bindgen_ty_1 {
                    stepwise: bindings::v4l2_frmival_stepwise {
                        min: fract(max_fps),
                        max: fract(min_fps),
                        step: fract(1),
                    },
                },
            )
        };
        Ok(v4l2_frmivalenum {
            index: 0,
            pixel_format,
            width,
            height,
            type_,
            __bindgen_anon_1: anon,
            ..Default::default()
        })
    }

    fn g_fmt(&mut self, session: &Self::Session, queue: QueueType) -> IoctlResult<v4l2_format> {
        Ok(session.format(queue.direction_or_einval()?))
    }

    fn try_fmt(
        &mut self,
        session: &Self::Session,
        queue: QueueType,
        format: v4l2_format,
    ) -> IoctlResult<v4l2_format> {
        let direction = queue.direction_or_einval()?;
        // SAFETY: both accepted queue types are multi-planar, so `pix_mp` is the live member.
        let pix_mp = unsafe { format.fmt.pix_mp };
        match direction {
            // The raw side: NV12 only, at a size the current coded format accepts, with the
            // colorimetry the guest declares.
            QueueDirection::Output => {
                let coded = self.current_format(session)?;
                let width = coded.width.fit(if pix_mp.width == 0 {
                    session.coded_size.0
                } else {
                    pix_mp.width
                });
                let height = coded.height.fit(if pix_mp.height == 0 {
                    session.coded_size.1
                } else {
                    pix_mp.height
                });
                let colorspace = V4l2FormatColorspace::from_pix_mp(&pix_mp);
                Ok(raw_output_format(width, height, colorspace))
            }
            // The coded side: the codec (an unknown one snaps to the first), the client's
            // bitstream buffer size if it is a plausible one, the raw size refitted to the
            // codec's range (read-only), the colorimetry of the raw side.
            QueueDirection::Capture => {
                let coded = self.adjust_coded_format(pix_mp.pixelformat)?;
                let width = coded.width.fit(session.coded_size.0);
                let height = coded.height.fit(session.coded_size.1);
                let asked = pix_mp.plane_fmt[0].sizeimage;
                let sizeimage = if asked >= MIN_BITSTREAM_SIZE {
                    asked
                } else {
                    default_bitstream_size(width, height)
                };
                Ok(coded_capture_format(
                    coded.fourcc,
                    width,
                    height,
                    sizeimage,
                    session.colorspace,
                ))
            }
        }
    }

    fn s_fmt(
        &mut self,
        session: &mut Self::Session,
        queue: QueueType,
        format: v4l2_format,
    ) -> IoctlResult<v4l2_format> {
        let direction = queue.direction_or_einval()?;
        if session.dead {
            return Err(libc::ENODEV);
        }
        match direction {
            QueueDirection::Capture => {
                // The CAPTURE format governs the encode: it cannot change while buffers are
                // allocated on either queue or the codec runs (kernel encoder interface,
                // "Commit Points" 5).
                if session.has_buffers() || session.codec_started {
                    return Err(libc::EBUSY);
                }
                let adjusted = self.try_fmt(session, queue, format)?;
                // SAFETY: multi-planar.
                let pix_mp = unsafe { adjusted.fmt.pix_mp };
                session.coded_format = PixelFormat::from_u32(pix_mp.pixelformat);
                let size = (pix_mp.width, pix_mp.height);
                if size != session.coded_size {
                    session.coded_size = size;
                    session.crop = session.full_rect();
                }
                session.bitstream_size = Some(pix_mp.plane_fmt[0].sizeimage);
                Ok(adjusted)
            }
            QueueDirection::Output => {
                // The raw format can change while CAPTURE buffers exist (the kernel lets it),
                // not while its own buffers do or the codec runs.
                if !session.input.buffers.is_empty() || session.codec_started {
                    return Err(libc::EBUSY);
                }
                let adjusted = self.try_fmt(session, queue, format)?;
                // SAFETY: multi-planar.
                let pix_mp = unsafe { adjusted.fmt.pix_mp };
                session.coded_size = (pix_mp.width, pix_mp.height);
                session.colorspace = V4l2FormatColorspace::from_pix_mp(&pix_mp);
                // "Setting the OUTPUT format will reset the selection rectangles to their
                // default values" (kernel encoder interface, "Initialization" 3).
                session.crop = session.full_rect();
                Ok(adjusted)
            }
        }
    }

    fn reqbufs(
        &mut self,
        session: &mut Self::Session,
        queue: QueueType,
        memory: MemoryType,
        count: u32,
    ) -> IoctlResult<v4l2_requestbuffers> {
        let direction = queue.direction_or_einval()?;
        if !matches!(memory, MemoryType::Mmap | MemoryType::UserPtr) {
            return Err(libc::EINVAL);
        }
        match self.active_session {
            Some(id) if id != session.id => return Err(libc::EBUSY),
            _ => (),
        }
        // `REQBUFS(0)` is an implicit `STREAMOFF`; any other count on a streaming queue is
        // refused, as vb2 does, because its buffers may be lent.
        if count == 0 {
            self.streamoff(session, queue)?;
        } else if session.streaming(direction) {
            return Err(libc::EBUSY);
        }
        // Old buffers go first, mappings and all, so the reply never races a stale view. The
        // backend has stopped touching them (`streamoff` above, or the queue was not streaming).
        self.free_buffers(session.queue_mut(queue)?);
        let count = (count as usize).min(MAX_BUFFERS);
        if count > 0 {
            let sizeimage = session.sizeimage(direction);
            self.add_buffers(session.queue_mut(queue)?, queue, memory, count, sizeimage)?;
            session.queue_mut(queue)?.memory = Some(memory);
            self.active_session = Some(session.id);
        } else if !session.has_buffers() {
            self.active_session = None;
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
        let direction = queue.direction_or_einval()?;
        if !matches!(memory, MemoryType::Mmap | MemoryType::UserPtr) {
            return Err(libc::EINVAL);
        }
        match self.active_session {
            Some(id) if id != session.id => return Err(libc::EBUSY),
            _ => (),
        }
        // `CREATE_BUFS` is the one call where the guest sizes the buffers itself, so the format
        // it hands over is checked rather than adjusted (D6.2/D9, as the loopback and camera do):
        // one plane, `sizeimage` at least what the queue's own format needs.
        // SAFETY: both accepted queue types are multi-planar.
        let pix_mp = unsafe { format.fmt.pix_mp };
        if pix_mp.num_planes != 1 {
            return Err(libc::EINVAL);
        }
        let asked = pix_mp.plane_fmt[0].sizeimage;
        if asked < session.sizeimage(direction) {
            return Err(libc::EINVAL);
        }
        {
            let q = session.queue_mut(queue)?;
            if let Some(existing) = q.memory {
                if existing != memory {
                    return Err(libc::EINVAL);
                }
            }
        }
        let first = session.queue(queue)?.buffers.len();
        let count = (count as usize).min(MAX_BUFFERS - first);
        if count > 0 {
            self.add_buffers(session.queue_mut(queue)?, queue, memory, count, asked)?;
            session.queue_mut(queue)?.memory = Some(memory);
            self.active_session = Some(session.id);
        }

        Ok(v4l2_create_buffers {
            index: first as u32,
            count: count as u32,
            memory: memory as u32,
            format: session.format_sized(direction, asked),
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
        let buffer = session
            .queue(queue)?
            .buffers
            .get(index as usize)
            .ok_or(libc::EINVAL)?;
        Ok(buffer.v4l2_buffer.clone())
    }

    fn qbuf(
        &mut self,
        session: &mut Self::Session,
        buffer: V4l2Buffer,
        guest_regions: Vec<Vec<SgEntry>>,
        payload: PayloadValidity,
    ) -> IoctlResult<V4l2Buffer> {
        let queue_type = buffer.queue();
        let direction = queue_type.direction_or_einval()?;
        if session.dead {
            return Err(libc::ENODEV);
        }
        let index = buffer.index() as usize;
        let sizeimage = session.sizeimage(direction);
        let q = session.queue_mut(queue_type)?;
        let entry = q.buffers.get_mut(index).ok_or(libc::EINVAL)?;
        if entry.queued || Some(buffer.memory()) != q.memory {
            return Err(libc::EINVAL);
        }
        // A prepared buffer keeps the payload description `PREPARE_BUF` accepted; V4L2 says this
        // call's own `bytesused`/`data_offset` are ignored. Otherwise it is the queue's own plane
        // count that decides which of the guest's slots are a description at all, and an `MMAP`
        // capture buffer has no guest description to check: the device reports the payload (D21,
        // `ioctl::PayloadValidity`).
        let prepared = entry.prepared;
        if prepared.is_none() && !payload.is_accepted_by(direction, buffer.memory(), NUM_PLANES) {
            return Err(libc::EINVAL);
        }
        // A guest-supplied MPLANE buffer may carry no plane at all; the first plane is asked for,
        // never assumed (this VMM aborts on panic).
        let guest_plane = buffer.planes_iter().next().ok_or(libc::EINVAL)?;
        let guest_bytesused = *guest_plane.bytesused;
        let guest_length = match prepared {
            Some((_, length)) => length,
            None => *guest_plane.length,
        };

        match &mut entry.backing {
            Backing::Host { .. } => {
                let plane = entry.v4l2_buffer.get_first_plane_mut();
                *plane.bytesused = if direction == QueueDirection::Output {
                    guest_bytesused
                } else {
                    0
                };
            }
            Backing::Guest(slot) => {
                // `length` sizes the mapping, and a raw frame or a bitstream buffer must hold
                // what the queue's format says, no more than the buffer was allocated for.
                //
                // Checked on *this* call's plane whatever `PREPARE_BUF` accepted earlier: the
                // scatter list about to be mapped was read against this call's `length`
                // (`ioctl::get_userptr_regions`), so that is the number that decides how much
                // guest memory the loan really covers (review-m4 R2). The prepared length is
                // checked too, for what it legitimately describes.
                if *guest_plane.length < sizeimage || *guest_plane.length > entry.size {
                    return Err(libc::EINVAL);
                }
                if guest_length < sizeimage || guest_length > entry.size {
                    return Err(libc::EINVAL);
                }
                let sgs = guest_regions.into_iter().next().ok_or(libc::EINVAL)?;
                // OUTPUT is read by the backend; CAPTURE is written by it.
                let writable = direction == QueueDirection::Capture;
                let mapping = self.mem.new_mapping_for(sgs, writable).map_err(|e| {
                    log::error!("failed to map USERPTR buffer: {:#}", e);
                    guest_mapping_errno(&e)
                })?;
                // The backstop: a scatter list that covers less than `length` claims.
                if mapping.len() < sizeimage as usize {
                    return Err(libc::EINVAL);
                }
                *slot = Some(mapping);
                if prepared.is_none() {
                    // The guest's own view of its buffer -- userptr and length -- is echoed back.
                    let mut v4l2_buffer = buffer.clone();
                    v4l2_buffer.set_field(BufferField::None);
                    v4l2_buffer.set_flags(BufferFlags::TIMESTAMP_COPY);
                    *v4l2_buffer.get_first_plane_mut().bytesused =
                        if direction == QueueDirection::Output {
                            guest_bytesused
                        } else {
                            0
                        };
                    entry.v4l2_buffer = v4l2_buffer;
                }
            }
        }

        if direction == QueueDirection::Output {
            entry.v4l2_buffer.set_timestamp(buffer.timestamp());
        }
        entry.queued = true;
        entry.prepared = None;
        entry.v4l2_buffer.clear_flags(
            BufferFlags::PREPARED
                | BufferFlags::LAST
                | BufferFlags::DONE
                | BufferFlags::KEYFRAME
                | BufferFlags::PFRAME
                | BufferFlags::BFRAME,
        );
        entry.v4l2_buffer.add_flags(BufferFlags::QUEUED);
        let reply = entry.v4l2_buffer.clone();

        session.queue_mut(queue_type)?.pending.push_back(index);
        self.feed_pending(session)?;
        Ok(reply)
    }

    /// `VIDIOC_PREPARE_BUF`: everything `QBUF` validates, minus the queueing (D6.1, as the
    /// loopback and camera do). No guest memory is mapped: the driver sends the SG list again
    /// with the `QBUF` that follows.
    fn prepare_buf(
        &mut self,
        session: &mut Self::Session,
        buffer: V4l2Buffer,
        _guest_regions: Vec<Vec<SgEntry>>,
        payload: PayloadValidity,
    ) -> IoctlResult<V4l2Buffer> {
        let queue_type = buffer.queue();
        let direction = queue_type.direction_or_einval()?;
        // The same rule `qbuf` applies, with no prepared description to fall back on.
        if !payload.is_accepted_by(direction, buffer.memory(), NUM_PLANES) {
            return Err(libc::EINVAL);
        }
        let sizeimage = session.sizeimage(direction);
        let q = session.queue_mut(queue_type)?;
        let entry = q
            .buffers
            .get_mut(buffer.index() as usize)
            .ok_or(libc::EINVAL)?;
        if entry.queued || entry.prepared.is_some() || Some(buffer.memory()) != q.memory {
            return Err(libc::EINVAL);
        }

        let guest_plane = buffer.planes_iter().next().ok_or(libc::EINVAL)?;
        let guest_bytesused = *guest_plane.bytesused;
        let guest_length = *guest_plane.length;

        if let Backing::Guest(_) = &entry.backing {
            if guest_length < sizeimage || guest_length > entry.size {
                return Err(libc::EINVAL);
            }
            let mut v4l2_buffer = buffer.clone();
            v4l2_buffer.set_field(BufferField::None);
            v4l2_buffer.set_flags(BufferFlags::TIMESTAMP_COPY);
            entry.v4l2_buffer = v4l2_buffer;
        }

        let bytesused = if direction == QueueDirection::Output {
            guest_bytesused
        } else {
            0
        };
        entry.v4l2_buffer.set_timestamp(Default::default());
        entry.v4l2_buffer.set_sequence(0);
        entry
            .v4l2_buffer
            .clear_flags(BufferFlags::QUEUED | BufferFlags::DONE | BufferFlags::LAST);
        *entry.v4l2_buffer.get_first_plane_mut().bytesused = bytesused;
        entry.v4l2_buffer.add_flags(BufferFlags::PREPARED);
        entry.prepared = Some((bytesused, guest_length));

        Ok(entry.v4l2_buffer.clone())
    }

    /// Streaming starts on a queue; the codec is created when both stream (kernel encoder
    /// interface, "Initialization" 8: "the actual encoding process starts when both queues
    /// start streaming"), from the formats, frame rate and controls current at that moment.
    fn streamon(&mut self, session: &mut Self::Session, queue: QueueType) -> IoctlResult<()> {
        let direction = queue.direction_or_einval()?;
        if session.dead {
            return Err(libc::ENODEV);
        }
        if session.queue(queue)?.buffers.is_empty() {
            return Err(libc::EINVAL);
        }
        if session.streaming(direction) {
            return Ok(());
        }
        let before = session.state;
        match direction {
            QueueDirection::Output => session.state.output_streaming = true,
            QueueDirection::Capture => session.state.capture_streaming = true,
        }
        if let Err(e) = self.maybe_start_codec(session) {
            // A codec that will not start leaves the queue as it was, buffers queued for a
            // retry.
            if !session.dead {
                session.state = before;
            }
            return Err(e);
        }
        Ok(())
    }

    /// `STREAMOFF(OUTPUT)` pauses the encoder (the kernel's "Stopped"): pending frames are
    /// dropped and returned, the codec is kept. `STREAMOFF(CAPTURE)` resets it (the kernel's
    /// "Reset"): the codec is torn down, so the next start produces a stream that stands on its
    /// own, headers included; raw frames it had not encoded are queued again for it.
    fn streamoff(&mut self, session: &mut Self::Session, queue: QueueType) -> IoctlResult<()> {
        let direction = queue.direction_or_einval()?;
        match direction {
            QueueDirection::Output => {
                // The backend must let go of every raw frame before we release them (§2.5).
                if session.codec_started && !session.dead {
                    session.backend.flush()?;
                }
                session.state.output_streaming = false;
                session.input.pending.clear();
                session.drain = Drain::None;
                for buffer in session.input.buffers.iter_mut() {
                    buffer.unqueue();
                }
            }
            QueueDirection::Capture => {
                if !session.dead {
                    self.stop_codec(session);
                }
                session.state.capture_streaming = false;
                session.output.pending.clear();
                session.drain = Drain::None;
                for buffer in session.output.buffers.iter_mut() {
                    buffer.unqueue();
                }
            }
        }
        Ok(())
    }

    /// Only the `OUTPUT` crop exists on an encoder: the visible part of the raw frame (kernel
    /// encoder interface, "Initialization" 6). `v4l2-compliance` requires `CAPTURE` selection
    /// and composition to be refused.
    fn g_selection(
        &mut self,
        session: &Self::Session,
        sel_type: SelectionType,
        sel_target: SelectionTarget,
    ) -> IoctlResult<bindings::v4l2_rect> {
        match (sel_type, sel_target) {
            (SelectionType::Output, SelectionTarget::CropBounds)
            | (SelectionType::Output, SelectionTarget::CropDefault) => {
                Ok(session.full_rect().into())
            }
            (SelectionType::Output, SelectionTarget::Crop) => Ok(session.crop.into()),
            _ => Err(libc::EINVAL),
        }
    }

    fn s_selection(
        &mut self,
        session: &mut Self::Session,
        sel_type: SelectionType,
        sel_target: SelectionTarget,
        sel_rect: bindings::v4l2_rect,
        _sel_flags: SelectionFlags,
    ) -> IoctlResult<bindings::v4l2_rect> {
        if !matches!(
            (sel_type, sel_target),
            (SelectionType::Output, SelectionTarget::Crop)
        ) {
            return Err(libc::EINVAL);
        }
        if session.codec_started {
            return Err(libc::EBUSY);
        }
        // Held inside the frame, never empty: an empty or oversized request means the whole
        // frame, as the default does.
        let (w, h) = session.coded_size;
        let left = sel_rect.left.clamp(0, w.saturating_sub(1) as i32);
        let top = sel_rect.top.clamp(0, h.saturating_sub(1) as i32);
        let max_w = w - left as u32;
        let max_h = h - top as u32;
        let width = if sel_rect.width == 0 {
            max_w
        } else {
            sel_rect.width.min(max_w)
        };
        let height = if sel_rect.height == 0 {
            max_h
        } else {
            sel_rect.height.min(max_h)
        };
        session.crop = v4l2r::Rect::new(left, top, width, height);
        Ok(session.crop.into())
    }

    /// `G_PARM`: the frame rate on `OUTPUT`; `ENOTTY` on `CAPTURE` (an encoder without
    /// `V4L2_FMT_FLAG_ENC_CAP_FRAME_INTERVAL` must refuse it, `v4l2-test-formats.cpp:1418-1424`,
    /// and the driver's D6.4 forwarding leaves the answer to us).
    fn g_parm(
        &mut self,
        session: &Self::Session,
        queue: QueueType,
    ) -> IoctlResult<v4l2_streamparm> {
        match queue.direction_or_einval()? {
            QueueDirection::Output => Ok(session.streamparm()),
            QueueDirection::Capture => Err(libc::ENOTTY),
        }
    }

    /// `S_PARM(OUTPUT)` sets the frame rate the codec is created with: the interval as given
    /// when the format sustains it, else the nearest end of the range; a fraction that is not a
    /// rate (`0/x`, `x/0`) asks for the default. Applies to the next codec started; a running
    /// one keeps its rate (the kernel calls the OUTPUT interval a hint).
    fn s_parm(
        &mut self,
        session: &mut Self::Session,
        parm: v4l2_streamparm,
    ) -> IoctlResult<v4l2_streamparm> {
        let queue = QueueType::n(parm.type_).ok_or(libc::EINVAL)?;
        match queue.direction_or_einval()? {
            QueueDirection::Capture => return Err(libc::ENOTTY),
            QueueDirection::Output => (),
        }
        if session.dead {
            return Err(libc::ENODEV);
        }
        let range = self.current_format(session)?.frame_rate;
        let min_fps = range.min.max(1);
        let max_fps = range.max.max(min_fps);
        // SAFETY: the type says output, so `output` is the live member.
        let asked = unsafe { parm.parm.output.timeperframe };
        session.timeperframe = if asked.numerator == 0 || asked.denominator == 0 {
            (1, DEFAULT_FRAME_RATE.clamp(min_fps, max_fps))
        } else {
            let fps = asked.denominator as f64 / asked.numerator as f64;
            if fps > max_fps as f64 {
                (1, max_fps)
            } else if fps < min_fps as f64 {
                (1, min_fps)
            } else {
                (asked.numerator, asked.denominator)
            }
        };
        Ok(session.streamparm())
    }

    fn subscribe_event(
        &mut self,
        session: &mut Self::Session,
        event: EventType,
        flags: SubscribeEventFlags,
    ) -> IoctlResult<()> {
        match event {
            EventType::Eos => {
                session.eos_subscribed = true;
                Ok(())
            }
            // A control event: only the initial one is ever sent (control values are per
            // session, so no other subscriber can see a change), which is what
            // `v4l2-compliance`'s `testEvents` asks for.
            EventType::Ctrl(id) => {
                let index = self.control_index(id).ok_or(libc::EINVAL)?;
                if flags.contains(SubscribeEventFlags::SEND_INITIAL)
                    && !matches!(self.controls[index].ty, CtrlType::Class)
                {
                    let event = self.ctrl_event(session, index);
                    self.evt_queue
                        .send_event(V4l2Event::Event(SessionEvent::new(session.id, event)));
                }
                Ok(())
            }
            // An encoder has no SOURCE_CHANGE (`v4l2-test-controls.cpp:1200-1201`).
            _ => Err(libc::EINVAL),
        }
    }

    fn unsubscribe_event(
        &mut self,
        session: &mut Self::Session,
        event: v4l2_event_subscription,
    ) -> IoctlResult<()> {
        if event.type_ == bindings::V4L2_EVENT_ALL {
            session.eos_subscribed = false;
            return Ok(());
        }
        match EventType::try_from(&event) {
            Ok(EventType::Eos) => {
                session.eos_subscribed = false;
                Ok(())
            }
            Ok(EventType::Ctrl(_)) => Ok(()),
            _ => Err(libc::EINVAL),
        }
    }

    fn queryctrl(
        &mut self,
        _session: &Self::Session,
        id: CtrlId,
        flags: QueryCtrlFlags,
    ) -> IoctlResult<v4l2_queryctrl> {
        let def = self.query_control(id, flags)?;
        let (minimum, maximum, step, default_value) = def.bounds();
        let mut out = v4l2_queryctrl {
            id: def.id,
            type_: def.v4l2_type(),
            minimum,
            maximum,
            step,
            default_value,
            flags: def.flags(),
            ..Default::default()
        };
        copy_name(&mut out.name, def.name);
        Ok(out)
    }

    fn query_ext_ctrl(
        &mut self,
        _session: &Self::Session,
        id: CtrlId,
        flags: QueryCtrlFlags,
    ) -> IoctlResult<v4l2_query_ext_ctrl> {
        let def = self.query_control(id, flags)?;
        let (minimum, maximum, step, default_value) = def.bounds();
        let mut out = v4l2_query_ext_ctrl {
            id: def.id,
            type_: def.v4l2_type(),
            minimum: minimum as i64,
            maximum: maximum as i64,
            step: step as u64,
            default_value: default_value as i64,
            flags: def.flags(),
            elem_size: std::mem::size_of::<i32>() as u32,
            elems: 1,
            ..Default::default()
        };
        copy_c_name(&mut out.name, def.name);
        Ok(out)
    }

    fn querymenu(
        &mut self,
        _session: &Self::Session,
        id: u32,
        index: u32,
    ) -> IoctlResult<v4l2_querymenu> {
        let def = self
            .control_index(id)
            .map(|i| &self.controls[i])
            .ok_or(libc::EINVAL)?;
        let name = i32::try_from(index)
            .ok()
            .and_then(|i| def.menu_name(i))
            .ok_or(libc::EINVAL)?;
        let mut out = v4l2_querymenu {
            id,
            index,
            __bindgen_anon_1: bindings::v4l2_querymenu__bindgen_ty_1 { name: [0; 32] },
            reserved: 0,
        };
        // SAFETY: `name` is the member a menu (not integer-menu) control fills.
        copy_name(unsafe { &mut out.__bindgen_anon_1.name }, name);
        Ok(out)
    }

    fn g_ctrl(&mut self, session: &Self::Session, id: u32) -> IoctlResult<v4l2_control> {
        let value = self.read_control(session, id, false)?;
        Ok(v4l2_control { id, value })
    }

    fn s_ctrl(
        &mut self,
        session: &mut Self::Session,
        id: u32,
        value: i32,
    ) -> IoctlResult<v4l2_control> {
        let value = self.check_control(session, id, value, true)?;
        let value = self.apply_control(session, id, value)?;
        Ok(v4l2_control { id, value })
    }

    fn g_ext_ctrls(
        &mut self,
        session: &Self::Session,
        which: CtrlWhich,
        ctrls: &mut v4l2_ext_controls,
        ctrl_array: &mut Vec<v4l2_ext_control>,
        _user_regions: Vec<Vec<SgEntry>>,
    ) -> IoctlResult<()> {
        match self.check_ext_ctrls(session, ExtCtrlOp::Get, which, ctrl_array) {
            Ok(values) => {
                Self::write_back_ext_ctrls(ctrls, ctrl_array, &values);
                Ok(())
            }
            Err((errno, error_idx)) => {
                ctrls.error_idx = error_idx;
                Err(errno)
            }
        }
    }

    /// Every control is validated before any is applied, so a refused set changes nothing
    /// (`error_idx = count`, as the kernel reports a validation failure of a set).
    fn s_ext_ctrls(
        &mut self,
        session: &mut Self::Session,
        which: CtrlWhich,
        ctrls: &mut v4l2_ext_controls,
        ctrl_array: &mut Vec<v4l2_ext_control>,
        _user_regions: Vec<Vec<SgEntry>>,
    ) -> IoctlResult<()> {
        let values = match self.check_ext_ctrls(session, ExtCtrlOp::Set, which, ctrl_array) {
            Ok(values) => values,
            Err((errno, error_idx)) => {
                ctrls.error_idx = error_idx;
                return Err(errno);
            }
        };
        for (ctrl, value) in ctrl_array.iter().zip(&values) {
            if let Err(e) = self.apply_control(session, ctrl.id, *value) {
                ctrls.error_idx = ctrl_array.len() as u32;
                return Err(e);
            }
        }
        Self::write_back_ext_ctrls(ctrls, ctrl_array, &values);
        Ok(())
    }

    fn try_ext_ctrls(
        &mut self,
        session: &Self::Session,
        which: CtrlWhich,
        ctrls: &mut v4l2_ext_controls,
        ctrl_array: &mut Vec<v4l2_ext_control>,
        _user_regions: Vec<Vec<SgEntry>>,
    ) -> IoctlResult<()> {
        match self.check_ext_ctrls(session, ExtCtrlOp::Try, which, ctrl_array) {
            Ok(values) => {
                Self::write_back_ext_ctrls(ctrls, ctrl_array, &values);
                Ok(())
            }
            Err((errno, error_idx)) => {
                ctrls.error_idx = error_idx;
                Err(errno)
            }
        }
    }

    fn try_encoder_cmd(
        &mut self,
        _session: &Self::Session,
        cmd: v4l2_encoder_cmd,
    ) -> IoctlResult<v4l2_encoder_cmd> {
        normalize_encoder_cmd(cmd)
    }

    /// `V4L2_ENC_CMD_STOP` drains (only if both queues stream, else a no-op success -- kernel
    /// encoder interface, "Drain" 1), `START` resumes after a finished drain; `PAUSE`/`RESUME`
    /// are `EINVAL`; either command during a drain is `EBUSY`.
    fn encoder_cmd(
        &mut self,
        session: &mut Self::Session,
        cmd: v4l2_encoder_cmd,
    ) -> IoctlResult<v4l2_encoder_cmd> {
        let cmd = normalize_encoder_cmd(cmd)?;
        if session.dead {
            return Err(libc::ENODEV);
        }
        match (cmd.cmd, session.drain) {
            (_, Drain::Pending) => return Err(libc::EBUSY),
            (bindings::V4L2_ENC_CMD_STOP, Drain::None) => {
                if session.state.running() && session.codec_started {
                    session.drain = Drain::Pending;
                    session.backend.drain()?;
                }
            }
            (bindings::V4L2_ENC_CMD_STOP, Drain::Done) => (),
            (bindings::V4L2_ENC_CMD_START, Drain::Done) => {
                // "The encoder will not be reset and will resume operation normally": the
                // backend takes input again, and what was held back is lent.
                session.backend.flush()?;
                session.drain = Drain::None;
                self.feed_pending(session)?;
            }
            (bindings::V4L2_ENC_CMD_START, Drain::None) => (),
            _ => return Err(libc::EINVAL),
        }
        Ok(cmd)
    }
}

/// A raw (OUTPUT) NV12 `v4l2_format`, tightly packed.
fn raw_output_format(width: u32, height: u32, colorspace: V4l2FormatColorspace) -> v4l2_format {
    let mut pix_mp = bindings::v4l2_pix_format_mplane {
        width,
        height,
        pixelformat: NV12.to_u32(),
        field: bindings::v4l2_field_V4L2_FIELD_NONE,
        num_planes: 1,
        ..Default::default()
    };
    colorspace.apply(&mut pix_mp);
    pix_mp.plane_fmt[0] = bindings::v4l2_plane_pix_format {
        sizeimage: nv12_sizeimage(width, height),
        bytesperline: width,
        ..Default::default()
    };
    v4l2_format {
        type_: QueueType::VideoOutputMplane as u32,
        fmt: bindings::v4l2_format__bindgen_ty_1 { pix_mp },
    }
}

/// A coded (CAPTURE) `v4l2_format`. Compressed formats carry `width`/`height` (the coded size)
/// but `bytesperline = 0`, and a single plane sized `sizeimage`.
fn coded_capture_format(
    fourcc: PixelFormat,
    width: u32,
    height: u32,
    sizeimage: u32,
    colorspace: V4l2FormatColorspace,
) -> v4l2_format {
    let mut pix_mp = bindings::v4l2_pix_format_mplane {
        width,
        height,
        pixelformat: fourcc.to_u32(),
        field: bindings::v4l2_field_V4L2_FIELD_NONE,
        num_planes: 1,
        ..Default::default()
    };
    colorspace.apply(&mut pix_mp);
    pix_mp.plane_fmt[0] = bindings::v4l2_plane_pix_format {
        sizeimage,
        bytesperline: 0,
        ..Default::default()
    };
    v4l2_format {
        type_: QueueType::VideoCaptureMplane as u32,
        fmt: bindings::v4l2_format__bindgen_ty_1 { pix_mp },
    }
}

/// `V4L2_ENC_CMD_STOP` / `START` reduced to what an encoder implements: both with their flags
/// cleared (`V4L2_ENC_CMD_STOP_AT_GOP_END` is not honoured) and `raw` zeroed, `PAUSE` /
/// `RESUME` `EINVAL`. `v4l2-compliance`'s `testEncoder` checks exactly this.
fn normalize_encoder_cmd(cmd: v4l2_encoder_cmd) -> IoctlResult<v4l2_encoder_cmd> {
    match cmd.cmd {
        bindings::V4L2_ENC_CMD_STOP | bindings::V4L2_ENC_CMD_START => Ok(v4l2_encoder_cmd {
            cmd: cmd.cmd,
            flags: 0,
            __bindgen_anon_1: bindings::v4l2_encoder_cmd__bindgen_ty_1 {
                raw: bindings::v4l2_encoder_cmd__bindgen_ty_1__bindgen_ty_1 { data: [0; 8] },
            },
        }),
        _ => Err(libc::EINVAL),
    }
}

/// Small helper: a queue type's direction, or `EINVAL` for a non-video-mplane queue.
trait QueueDirectionExt {
    fn direction_or_einval(self) -> IoctlResult<QueueDirection>;
}

impl QueueDirectionExt for QueueType {
    fn direction_or_einval(self) -> IoctlResult<QueueDirection> {
        match self {
            QueueType::VideoOutputMplane => Ok(QueueDirection::Output),
            QueueType::VideoCaptureMplane => Ok(QueueDirection::Capture),
            _ => Err(libc::EINVAL),
        }
    }
}

#[cfg(test)]
#[path = "video_encoder_tests.rs"]
mod tests;
