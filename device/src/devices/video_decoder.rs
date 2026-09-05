// Copyright 2024 The ChromiumOS Authors
// Copyright 2026 The DroidVM Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

//! A stateful V4L2 memory-to-memory video decoder over a [`VideoDecoderBackend`]
//! (`VPU_DESIGN.md` §7.2).
//!
//! This module is the V4L2 half of the decoder and knows nothing about how the bitstream is
//! actually decoded: a [`VideoDecoderBackend`] enumerates what it can decode
//! ([`DecoderCapabilities`]) and, when the guest starts streaming, opens a
//! [`VideoDecoderBackendSession`] that turns `OUTPUT` (bitstream) buffers into `CAPTURE` (NV12)
//! frames. On DroidVM the backend is `MediaCodecDecoderBackend` in crosvm over the Android
//! MediaCodec NDK, so this crate stays free of Android.
//!
//! # What the guest sees
//!
//! `V4L2_CAP_VIDEO_M2M_MPLANE | V4L2_CAP_STREAMING`, the kernel's stateful decoder interface
//! (`Documentation/userspace-api/media/v4l/dev-decoder.rst`): the `OUTPUT` queue advertises the
//! backend's coded formats (`ENUM_FMT`, `ENUM_FRAMESIZES` stepwise; a decoder has no frame rate,
//! so `ENUM_FRAMEINTERVALS` and `G/S_PARM` answer `ENOTTY`), the `CAPTURE` queue advertises
//! **NV12 single-plane, tightly packed** only (`bytesperline = width`,
//! `sizeimage = w*h*3/2` for even dimensions). `S_FMT(OUTPUT)` selects the coded format; the
//! backend session -- the actual codec -- is created at `STREAMON(OUTPUT)`. Once the backend has
//! parsed the stream it sends a `V4L2_EVENT_SOURCE_CHANGE`, from which the client sets the
//! `CAPTURE` queue up. Seek is `STREAMOFF(OUTPUT)`, drain is `V4L2_DEC_CMD_STOP` (a `LAST` buffer
//! then `V4L2_EVENT_EOS`), and a mid-stream resolution change is another `SOURCE_CHANGE` --
//! which is the kernel's *implicit* drain: the last `CAPTURE` buffer of the old resolution
//! carries `LAST` too, but **no `EOS` follows it**, and the decoder stays stopped until the
//! client restarts the `CAPTURE` queue (`dev-decoder.rst`, "Dynamic Resolution Change").
//!
//! Buffers are host-owned (`MMAP`, from the device's [`VirtioMediaBufferAllocator`] -- the
//! `media_host` pool on DroidVM) or guest-owned (`USERPTR`): an `OUTPUT` bitstream buffer is
//! guest-owned in the usual mode (`VPU_DESIGN.md` §2.1), a `CAPTURE` frame buffer is guest-owned
//! only in `driver_owned_queues=all`. Only one decoding session per device instance is allowed;
//! a second session's `REQBUFS`/`STREAMON` is refused with `EBUSY`.
//!
//! # Threads and buffers
//!
//! Frames arrive on a thread the backend owns -- a codec is a stream of async callbacks -- while
//! every ioctl runs on the device's worker thread. The two meet in three places, none of which
//! blocks the worker, mirroring `camera.rs`:
//!
//! * a bitstream buffer the guest queues is *lent* to the backend ([`InputBuffer`]) as a raw
//!   read-only pointer; the backend owns those bytes until it reports [`DecoderEvent::InputBufferDone`],
//!   and the device holds the buffer's guest mapping until then (`VPU_DESIGN.md` §2.5);
//! * a `CAPTURE` buffer the guest queues is lent to the backend ([`OutputBuffer`]) as a writable
//!   pointer -- into the `media_host` pool or into a writable guest mapping -- and the backend
//!   decodes into it and reports [`DecoderEvent::FrameDecoded`];
//! * `STREAMOFF`, `REQBUFS(0)`, a session close and a decode error all stop the backend from
//!   touching a queue's buffers *before* the device unqueues or frees them
//!   ([`VideoDecoderBackendSession::flush`] for `OUTPUT`,
//!   [`VideoDecoderBackendSession::clear_capture_buffers`] for `CAPTURE`,
//!   [`VideoDecoderBackendSession::stop`] for the whole session): a lent buffer is never released
//!   while the thread that may be writing into it is alive.
//!
//! # The frame copy contract
//!
//! An [`OutputBuffer`] is `len` bytes at `ptr`, sized for one NV12 frame of the coded size, and
//! the backend fills it as tightly packed NV12: `height` rows of `width` luma bytes, then
//! `height / 2` rows of `width` interleaved Cb/Cr bytes. `bytesused` of the returned
//! [`DecoderEvent::FrameDecoded`] is the number of bytes written, normally exactly
//! `width * height * 3 / 2`. Converting from whatever the codec produces (padded rows, tiled
//! layouts) is the backend's job; the device never looks at the pixels.

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
use v4l2r::bindings::v4l2_decoder_cmd;
use v4l2r::bindings::v4l2_event_subscription;
use v4l2r::bindings::v4l2_ext_control;
use v4l2r::bindings::v4l2_ext_controls;
use v4l2r::bindings::v4l2_fmtdesc;
use v4l2r::bindings::v4l2_format;
use v4l2r::bindings::v4l2_frmsizeenum;
use v4l2r::bindings::v4l2_query_ext_ctrl;
use v4l2r::bindings::v4l2_queryctrl;
use v4l2r::bindings::v4l2_querymenu;
use v4l2r::bindings::v4l2_requestbuffers;
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
use v4l2r::ioctl::SrcChanges;
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

/// The one CAPTURE (raw) format offered: Y plane then interleaved Cb/Cr, tightly packed.
pub const NV12: PixelFormat = PixelFormat::from_fourcc(b"NV12");
/// Most buffers on a queue, the usual V4L2 ceiling.
pub const MAX_BUFFERS: usize = 32;
/// Where a session's coded size starts before the stream is parsed: what `G_FMT`/`G_SELECTION`
/// answer so `v4l2-compliance` sees a non-empty rectangle before any `SOURCE_CHANGE`.
const DEFAULT_CODED_SIZE: (u32, u32) = (640, 480);
/// Floor for an `OUTPUT` (bitstream) buffer, when the client does not size it itself.
const MIN_BITSTREAM_SIZE: u32 = 1 << 20;

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

    /// `v` clamped into the range and rounded down to a multiple of the step.
    fn clamp(&self, v: u32) -> u32 {
        let step = self.step.max(1);
        let v = v.clamp(self.min, self.max);
        (v / step) * step
    }
}

/// One coded (compressed) format the backend accepts on the `OUTPUT` queue, in the terms
/// `ENUM_FMT(OUTPUT)` and `ENUM_FRAMESIZES` are answered with. The device advertises nothing of
/// its own: everything comes from here.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CodedFormat {
    /// The V4L2 OUTPUT fourcc: `H264`, `HEVC`, `VP80`, `VP90` or `AV01`.
    pub fourcc: PixelFormat,
    pub width: SizeRange,
    pub height: SizeRange,
    /// Whether the format carries resolution in the bitstream, i.e. whether the decoder can
    /// raise a `SOURCE_CHANGE`. Sets `V4L2_FMT_FLAG_DYN_RESOLUTION` in `ENUM_FMT`.
    pub dynamic_resolution: bool,
}

/// What a decoder can do, enumerated once at device creation (on Android by warming the
/// `AMediaCodecStore` up on a single thread, `VPU_DESIGN.md` §7.2). Controls (M5) extend this.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DecoderCapabilities {
    /// The coded formats, in `ENUM_FMT(OUTPUT)` order.
    pub coded_formats: Vec<CodedFormat>,
}

impl DecoderCapabilities {
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
        b"AV01" => b"AV1",
        b"NV12" => b"Y/UV 4:2:0",
        _ => b"Unknown",
    }
}

/// A pointer into a buffer the backend fills or reads, handed to the codec thread.
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

/// An `OUTPUT` (bitstream) buffer lent to the backend to decode. The backend reads `len` bytes at
/// `ptr` and reports [`DecoderEvent::InputBufferDone`] when done with them.
#[derive(Clone, Copy, Debug)]
pub struct InputBuffer {
    /// The V4L2 buffer index; comes back in [`DecoderEvent::InputBufferDone`].
    pub index: u32,
    /// Read-only pointer to the bitstream.
    pub ptr: SendPtr,
    /// Bytes of bitstream at `ptr` (`bytesused`).
    pub len: usize,
    /// The frame's timestamp, copied to every `CAPTURE` frame produced from it
    /// (`V4L2_BUF_FLAG_TIMESTAMP_COPY`).
    pub timestamp: bindings::timeval,
}

/// A `CAPTURE` (frame) buffer lent to the backend to decode into. See the module's copy contract.
#[derive(Clone, Copy, Debug)]
pub struct OutputBuffer {
    /// The V4L2 buffer index; comes back in [`DecoderEvent::FrameDecoded`].
    pub index: u32,
    /// Writable pointer to the frame buffer.
    pub ptr: SendPtr,
    /// Bytes available at `ptr`: at least one NV12 frame of the coded size.
    pub len: usize,
}

/// Something the backend reports on its event path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DecoderEvent {
    /// The `OUTPUT` buffer at `index` is done being read and can be returned to the guest.
    InputBufferDone(u32),
    /// The `CAPTURE` buffer at `index` holds a decoded frame.
    FrameDecoded {
        index: u32,
        /// Bytes written, from the start of the buffer.
        bytesused: u32,
        /// The `OUTPUT` timestamp the frame came from.
        timestamp: bindings::timeval,
        /// The last buffer before the decoder stops: of a `V4L2_DEC_CMD_STOP` drain, or of a
        /// mid-stream resolution change. It carries `V4L2_BUF_FLAG_LAST` and may be empty
        /// (`bytesused == 0`), which the kernel allows in both cases.
        ///
        /// `V4L2_EVENT_EOS` follows it **only when a drain is in flight**: a resolution change
        /// ends no stream (`dev-decoder.rst` lists the event under "Drain" alone, and its
        /// "Dynamic Resolution Change" step 2 asks for the flag "similarly to the Drain
        /// sequence" and for nothing else), so a backend may mark the last frame of the old
        /// resolution without ending the guest's stream -- which is what a GStreamer client
        /// waits for before it renegotiates. A backend that marks nothing is accepted too: the
        /// device stops the decoder on [`DecoderEvent::FormatChanged`] itself, only with no
        /// `LAST` buffer to show for it.
        is_last: bool,
    },
    /// The stream's format is now known, or has changed. The device raises `SOURCE_CHANGE`.
    ///
    /// The first one is the initial announcement the client waits for before it sets the
    /// `CAPTURE` queue up. Every later one is a mid-stream change and stops the decoder (the
    /// kernel's implicit drain), so it must come **after** every [`DecoderEvent::FrameDecoded`]
    /// of the old resolution.
    FormatChanged {
        /// Coded resolution of the stream.
        coded_size: (u32, u32),
        /// Visible rectangle within the coded resolution (the crop / compose rectangle).
        visible_rect: v4l2r::Rect,
        /// Minimum `CAPTURE` buffers the backend needs to decode; answered by
        /// `G_CTRL(V4L2_CID_MIN_BUFFERS_FOR_CAPTURE)`.
        min_capture_buffers: u32,
    },
    /// The session failed and produces nothing more; the string is for the log.
    Error(String),
}

/// An eventfd a session's worker polls. Bumped once per event; drained by the device before it
/// collects them, so a bump that lands in between leaves it readable. Same shape as
/// `camera.rs`'s `FrameSignal`.
pub struct DecoderSignal(OwnedFd);

impl DecoderSignal {
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

impl AsFd for DecoderSignal {
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.0.as_fd()
    }
}

/// The session's end of the wake-up path, cloned into whatever thread the backend runs. Bump it
/// after every [`DecoderEvent`] made available.
#[derive(Clone)]
pub struct DecoderSink(Arc<DecoderSignal>);

impl DecoderSink {
    pub fn signal(&self) {
        self.0.signal()
    }
}

/// A decoding session's backend: the actual codec. Every method here is called on the device's
/// worker thread and must return promptly; the codec runs on a thread of its own and reports back
/// through [`Self::take_events`], bumping the [`DecoderSink`] it was given at creation.
pub trait VideoDecoderBackendSession {
    /// Create the codec for `coded_format` at `coded_size` and start it. Called once, at the
    /// first `STREAMON(OUTPUT)`. Errors become the guest's `STREAMON` result.
    fn start(&mut self, coded_format: PixelFormat, coded_size: (u32, u32)) -> IoctlResult<()>;

    /// Lend an `OUTPUT` bitstream buffer to be decoded. The backend owns the bytes until it
    /// reports [`DecoderEvent::InputBufferDone`] for the same index.
    fn decode(&mut self, buffer: InputBuffer) -> IoctlResult<()>;

    /// Lend a `CAPTURE` buffer to be decoded into. The backend owns the bytes until it reports a
    /// [`DecoderEvent::FrameDecoded`] for the same index, or until [`Self::clear_capture_buffers`].
    fn use_as_capture(&mut self, buffer: OutputBuffer) -> IoctlResult<()>;

    /// `STREAMOFF(CAPTURE)`: stop decoding into every lent `CAPTURE` buffer and forget them. When
    /// this returns the backend touches no `CAPTURE` buffer any more, so the device may free them.
    fn clear_capture_buffers(&mut self) -> IoctlResult<()>;

    /// `STREAMOFF(OUTPUT)` = seek: drop every pending `OUTPUT` buffer and be ready to decode from
    /// a new resume point. When this returns the backend holds no `OUTPUT` buffer and reports
    /// nothing more about the ones it held: the device unqueues them itself, and a late
    /// [`DecoderEvent::InputBufferDone`] would land on a buffer the guest may have queued again
    /// (review-m6 R6-14). The `CAPTURE` queue keeps streaming. On an async codec this is
    /// flush-then-start. An error here ends the session (the device never fails `STREAMOFF`).
    fn flush(&mut self) -> IoctlResult<()>;

    /// `V4L2_DEC_CMD_STOP`: decode everything queued so far, then report the last `CAPTURE`
    /// buffer with [`DecoderEvent::FrameDecoded`] `is_last = true` (empty if there is no frame
    /// left).
    fn drain(&mut self) -> IoctlResult<()>;

    /// `V4L2_DEC_CMD_START` after the decoder stopped -- a finished drain, or a resolution change
    /// the client answers without reallocating: decode on into the `CAPTURE` buffers still lent
    /// and the ones lent from now on. A backend that stops writing after it announces a
    /// mid-stream change (so no frame of the new size lands in a buffer the client is about to
    /// take back) picks up here; the other way out of the stop, `STREAMOFF`/`STREAMON(CAPTURE)`,
    /// goes through [`Self::clear_capture_buffers`]. The default does nothing.
    fn resume(&mut self) {}

    /// Tear the codec down and join its thread. When this returns the backend touches nothing.
    ///
    /// An implementor must do the same from its `Drop`: a session can end without a `CLOSE`
    /// command -- a worker returning on its kill event, a vhost-user `stop_queue`, a device
    /// `reset` -- and then nothing calls this (review-m4 R1; `VirtioMediaDeviceRunner`'s own
    /// `Drop` covers the sessions it still holds, a bare `drop` of a session does not).
    fn stop(&mut self);

    /// Every event since the last call, oldest first.
    fn take_events(&mut self) -> Vec<DecoderEvent>;
}

/// A decoder, as a device sees it: what it can decode, and how to open a session.
pub trait VideoDecoderBackend {
    type Session: VideoDecoderBackendSession;

    /// What this decoder can do. Enumerated once; must not change over the device's life.
    fn capabilities(&self) -> &DecoderCapabilities;

    /// Prepare a session with the given `id`. `sink` is what the session bumps when an event is
    /// pending. The codec itself is not created until [`VideoDecoderBackendSession::start`].
    fn new_session(&mut self, id: u32, sink: DecoderSink) -> IoctlResult<Self::Session>;

    /// Close and destroy `session`, joining its thread.
    fn close_session(&mut self, session: Self::Session);
}

// ---------------------------------------------------------------------------------------------
// The device
// ---------------------------------------------------------------------------------------------

/// Validated colorspace information for a format, propagated `OUTPUT` -> `CAPTURE` on an m2m
/// device (`v4l2-compliance`'s `testM2MFormats`).
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
}

/// The crop rectangle. Settable by the client (`S_SELECTION`) only until the stream fixes it.
enum CropRectangle {
    Settable(v4l2r::Rect),
    FromStream(v4l2r::Rect),
}

impl CropRectangle {
    fn rect(&self) -> v4l2r::Rect {
        match self {
            CropRectangle::Settable(r) | CropRectangle::FromStream(r) => *r,
        }
    }
}

/// Streaming state of the two queues, following the kernel's decoder state machine.
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

/// Whether the decoder runs, is draining, or has stopped -- the state an explicit
/// `V4L2_DEC_CMD_STOP` drain and the implicit drain of a resolution change share.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Drain {
    None,
    /// `V4L2_DEC_CMD_STOP` issued; waiting for the backend's `LAST` buffer, which ends the drain
    /// with `V4L2_EVENT_EOS`.
    Pending,
    /// The decoder is stopped: the drain's `LAST` buffer has been returned, or a mid-stream
    /// `SOURCE_CHANGE` triggered the kernel's implicit drain ("A source change triggers an
    /// implicit decoder drain, similar to the explicit Drain sequence. The decoder is stopped
    /// after it completes", `dev-decoder.rst`, "Dynamic Resolution Change"). No `CAPTURE` buffer
    /// is lent to the backend until `STREAMON(CAPTURE)` or `V4L2_DEC_CMD_START`. (The device has
    /// no dequeue of its own: whether the guest sees `EPIPE` after the `LAST` buffer is the
    /// driver's side.)
    Stopped,
}

/// Bytes of one tightly packed NV12 frame of `width` x `height`. Odd dimensions round the chroma
/// plane up. Saturates: the sizes come from the backend's published ranges, and a store that
/// reports a dimension past ~53 000 must not turn a plain `S_FMT` into an overflow abort of the
/// helper (review-m6 R6-8); `u32::MAX` bytes is a buffer the allocator refuses.
fn nv12_sizeimage(width: u32, height: u32) -> u32 {
    let luma = width.saturating_mul(height);
    let chroma = width
        .div_ceil(2)
        .saturating_mul(2)
        .saturating_mul(height.div_ceil(2));
    luma.saturating_add(chroma)
}

/// Where a buffer's bytes live. Same two ownerships as `camera.rs` / `loopback_device.rs`.
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

/// Session data of [`VideoDecoder`].
pub struct VideoDecoderSession<GM, S> {
    id: u32,
    /// What the worker polls; the backend bumps it through a [`DecoderSink`].
    signal: Arc<DecoderSignal>,
    /// Declared before `input` and `output`, and the order is load-bearing: a session that is
    /// only dropped -- no `CLOSE`, so no [`VirtioMediaDevice::close_session`] -- drops the
    /// backend, and with it the codec's thread, before the buffers that thread may be writing
    /// into (`VPU_DESIGN.md` §2.5, review-m4 R1). [`VideoDecoderBackendSession::stop`] says an
    /// implementor must do that job from its own `Drop` too.
    backend: S,
    /// Whether [`VideoDecoderBackendSession::start`] has been called (the codec exists).
    codec_started: bool,
    state: StreamingState,
    drain: Drain,
    /// OUTPUT (bitstream) queue.
    input: Queue<GM>,
    /// CAPTURE (frame) queue.
    output: Queue<GM>,
    /// The coded format selected by `S_FMT(OUTPUT)`.
    coded_format: PixelFormat,
    /// The CAPTURE frame size: the `S_FMT(OUTPUT)` placeholder until the backend parses the
    /// stream, then the value from [`DecoderEvent::FormatChanged`].
    coded_size: (u32, u32),
    /// The OUTPUT bitstream buffer size, client-set via `S_FMT(OUTPUT)` or defaulted.
    output_sizeimage: u32,
    crop: CropRectangle,
    min_capture_buffers: u32,
    colorspace: V4l2FormatColorspace,
    src_change_subscribed: bool,
    eos_subscribed: bool,
    /// Whether the backend has announced a format: the first [`DecoderEvent::FormatChanged`] is
    /// the initial announcement, every later one is a mid-stream resolution change.
    format_announced: bool,
    /// One `warn!` per session about a `CAPTURE` buffer too small for the announced canvas.
    warned_small_capture: bool,
    /// A mid-stream `SOURCE_CHANGE` stopped the decoder while `CAPTURE` streamed and no `LAST`
    /// buffer has gone out for it yet: the client is owed one (a GStreamer client waits for
    /// exactly that buffer before it renegotiates, review-m6 R6-2). The backend supplies it from
    /// a buffer it holds when it can; otherwise the next `CAPTURE` buffer the guest queues is
    /// returned empty with `V4L2_BUF_FLAG_LAST`, as `v4l2_m2m_qbuf` does for a stopped decoder.
    /// Cleared by the `LAST` buffer, by `STREAMOFF(CAPTURE)` (the client is reallocating and
    /// needs no answer), and by the two resumes.
    last_owed: bool,
    /// The codec died; every ioctl that would touch it answers `ENODEV` until the guest closes.
    dead: bool,
    /// Sequence number of the next CAPTURE frame; restarts at 0 with `STREAMON(CAPTURE)`.
    sequence: u32,
}

impl<GM, S> VirtioMediaDeviceSession for VideoDecoderSession<GM, S> {
    fn poll_fd(&self) -> Option<BorrowedFd> {
        Some(self.signal.as_fd())
    }
}

impl<GM, S> VideoDecoderSession<GM, S> {
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

    /// The CAPTURE (NV12) `sizeimage` for the current coded size.
    fn capture_sizeimage(&self) -> u32 {
        nv12_sizeimage(self.coded_size.0, self.coded_size.1)
    }

    fn sizeimage(&self, direction: QueueDirection) -> u32 {
        match direction {
            QueueDirection::Output => self.output_sizeimage,
            QueueDirection::Capture => self.capture_sizeimage(),
        }
    }

    /// The format of a queue as a single-plane multi-planar `v4l2_format`, `sizeimage` bytes per
    /// buffer (`CREATE_BUFS` may ask for more than a frame needs).
    fn format_sized(&self, direction: QueueDirection, sizeimage: u32) -> v4l2_format {
        let (pixelformat, bytesperline, queue) = match direction {
            QueueDirection::Output => (self.coded_format.to_u32(), 0, QueueType::VideoOutputMplane),
            QueueDirection::Capture => {
                (NV12.to_u32(), self.coded_size.0, QueueType::VideoCaptureMplane)
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
}

/// A stateful V4L2 video decoder over a [`VideoDecoderBackend`]. See the module documentation.
pub struct VideoDecoder<
    B: VideoDecoderBackend,
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
    /// The one session allowed to hold buffers: one decode at a time, `EBUSY` for a second
    /// (`v4l2-compliance` checks a second session is refused).
    active_session: Option<u32>,
}

impl<B, Q, M, HM, A> VideoDecoder<B, Q, M, HM, A>
where
    B: VideoDecoderBackend,
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

    pub fn capabilities(&self) -> &DecoderCapabilities {
        self.backend.capabilities()
    }

    /// The coded format `S_FMT(OUTPUT)` selects for `fourcc`, or the first advertised format if
    /// the guest asked for one the backend does not have.
    fn adjust_coded_format(&self, fourcc: u32) -> IoctlResult<CodedFormat> {
        let caps = self.backend.capabilities();
        let asked = PixelFormat::from_u32(fourcc);
        caps.coded_format(asked)
            .or_else(|| caps.coded_formats.first())
            .cloned()
            .ok_or(libc::EINVAL)
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
            // A decoder copies the OUTPUT timestamp to the CAPTURE frame; both queues are
            // `TIMESTAMP_COPY` (kernel decoder interface, "Decoding").
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

    /// Lend a queued OUTPUT buffer to the backend to decode.
    fn lend_input(
        session: &mut VideoDecoderSession<M::GuestMemoryMapping, B::Session>,
        index: usize,
    ) -> IoctlResult<()> {
        let entry = session.input.buffers.get_mut(index).ok_or(libc::EINVAL)?;
        let timestamp = entry.v4l2_buffer.timestamp();
        let len = {
            let plane = entry.v4l2_buffer.get_first_plane();
            let used = if *plane.bytesused == 0 {
                *plane.length
            } else {
                *plane.bytesused
            };
            used.min(entry.capacity()) as usize
        };
        let ptr = entry.data_ptr().ok_or(libc::EIO)?;
        session.backend.decode(InputBuffer {
            index: index as u32,
            ptr: SendPtr(ptr),
            len,
            timestamp,
        })?;
        entry.lent = true;
        Ok(())
    }

    /// Lend a queued CAPTURE buffer to the backend to decode into.
    fn lend_output(
        session: &mut VideoDecoderSession<M::GuestMemoryMapping, B::Session>,
        index: usize,
    ) -> IoctlResult<()> {
        let sizeimage = session.capture_sizeimage();
        let entry = session.output.buffers.get_mut(index).ok_or(libc::EINVAL)?;
        let len = entry.capacity() as usize;
        let ptr = entry.data_ptr().ok_or(libc::EIO)?;
        // Lending a buffer too small for the announced canvas is not an error -- the guest may
        // simply not have acted on the `SOURCE_CHANGE` yet, and a `MMAP` queue sized before it
        // is exactly that -- but a backend can do nothing with such a buffer except hold it
        // unfilled until the queue is reallocated, so decoding stalls with nothing in the log to
        // say why (`M6-backend` §10 item 6). Say it once per session.
        if len < sizeimage as usize && !session.warned_small_capture {
            session.warned_small_capture = true;
            log::warn!(
                "decoder: session {}: CAPTURE buffer {} holds {} bytes but a {}x{} frame needs \
                 {}; the backend can only hold it until the queue is reallocated",
                session.id,
                index,
                len,
                session.coded_size.0,
                session.coded_size.1,
                sizeimage,
            );
        }
        session.backend.use_as_capture(OutputBuffer {
            index: index as u32,
            ptr: SendPtr(ptr),
            len,
        })?;
        entry.lent = true;
        Ok(())
    }

    /// Send every CAPTURE buffer queued so far to the backend, if both queues stream and the
    /// drain has not ended the queue. A buffer the backend refuses ends the session -- from
    /// every caller, the way `qbuf` always did -- and the buffer is popped only once it is lent,
    /// so a refused one is never left neither lent nor pending (review-m6 R6-6).
    fn try_send_pending_capture(
        &mut self,
        session: &mut VideoDecoderSession<M::GuestMemoryMapping, B::Session>,
    ) -> IoctlResult<()> {
        if !session.state.running() || session.drain == Drain::Stopped {
            return Ok(());
        }
        while let Some(&index) = session.output.pending.front() {
            if let Err(e) = Self::lend_output(session, index) {
                self.end_session(session, &format!("the backend refused a frame: errno {e}"));
                return Err(libc::EIO);
            }
            session.output.pending.pop_front();
        }
        Ok(())
    }

    /// Return the CAPTURE buffer `index` -- queued, not lent -- to the guest as the empty `LAST`
    /// buffer a stopped decoder owes it (`last_owed`).
    fn return_empty_last(
        &mut self,
        session: &mut VideoDecoderSession<M::GuestMemoryMapping, B::Session>,
        index: usize,
    ) {
        let seq = session.sequence;
        let Some(entry) = session.output.buffers.get_mut(index) else {
            return;
        };
        session.last_owed = false;
        session.sequence = session.sequence.wrapping_add(1);
        entry.unqueue();
        *entry.v4l2_buffer.get_first_plane_mut().bytesused = 0;
        entry.v4l2_buffer.set_timestamp(Default::default());
        entry.v4l2_buffer.set_sequence(seq);
        entry
            .v4l2_buffer
            .set_flags(BufferFlags::TIMESTAMP_COPY | BufferFlags::LAST);
        let event = entry.v4l2_buffer.clone();
        self.evt_queue
            .send_event(V4l2Event::DequeueBuffer(DequeueBufferEvent::new(
                session.id, event,
            )));
    }

    /// The codec died: unqueue every buffer, mark the session dead, tell the guest. Once per
    /// session: a second cause after the first changes nothing.
    fn end_session(
        &mut self,
        session: &mut VideoDecoderSession<M::GuestMemoryMapping, B::Session>,
        why: &str,
    ) {
        if session.dead {
            return;
        }
        log::error!("decoder: session {} ends: {}", session.id, why);
        session.backend.stop();
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
        session.last_owed = false;
        session.dead = true;
        self.evt_queue.send_error(session.id, libc::ENODEV);
    }

    /// One backend event: turn it into V4L2 events / buffer state.
    fn handle_event(
        &mut self,
        session: &mut VideoDecoderSession<M::GuestMemoryMapping, B::Session>,
        event: DecoderEvent,
    ) {
        match event {
            DecoderEvent::InputBufferDone(index) => {
                let Some(entry) = session.input.buffers.get_mut(index as usize) else {
                    log::error!("decoder: no OUTPUT buffer {} to return", index);
                    return;
                };
                if !entry.lent {
                    return;
                }
                // The backend is done reading the bitstream: drop the guest mapping (§2.5) and
                // tell the guest.
                entry.unqueue();
                let event = entry.v4l2_buffer.clone();
                self.evt_queue
                    .send_event(V4l2Event::DequeueBuffer(DequeueBufferEvent::new(
                        session.id, event,
                    )));
            }
            DecoderEvent::FrameDecoded {
                index,
                bytesused,
                timestamp,
                is_last,
            } => {
                let seq = session.sequence;
                let Some(entry) = session.output.buffers.get_mut(index as usize) else {
                    log::error!("decoder: no CAPTURE buffer {} to return", index);
                    return;
                };
                if !entry.lent {
                    return;
                }
                let capacity = entry.capacity();
                // The frame is written: drop the guest mapping (a shadowed one is written back
                // here) before the guest hears the buffer is done.
                entry.unqueue();
                let plane = entry.v4l2_buffer.get_first_plane_mut();
                *plane.bytesused = bytesused.min(capacity);
                entry.v4l2_buffer.set_timestamp(timestamp);
                entry.v4l2_buffer.set_flags(BufferFlags::TIMESTAMP_COPY);
                entry.v4l2_buffer.set_sequence(seq);
                if is_last {
                    entry.v4l2_buffer.add_flags(BufferFlags::LAST);
                }
                session.sequence = session.sequence.wrapping_add(1);
                let event = entry.v4l2_buffer.clone();
                self.evt_queue
                    .send_event(V4l2Event::DequeueBuffer(DequeueBufferEvent::new(
                        session.id, event,
                    )));
                if is_last {
                    // `LAST` says the decoder stopped; `EOS` says a *drain* finished. The
                    // resolution-change sequence marks its last buffer the same way and sends no
                    // `EOS` (`dev-decoder.rst`, "Dynamic Resolution Change" step 2, against
                    // "Drain" step 3 -- the only place the event is listed), so the event
                    // follows only a `V4L2_DEC_CMD_STOP` that is still in flight. Either way the
                    // decoder is now stopped (`Drain::Stopped`).
                    let drained = session.drain == Drain::Pending;
                    session.drain = Drain::Stopped;
                    session.last_owed = false;
                    if drained && session.eos_subscribed {
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
            DecoderEvent::FormatChanged {
                coded_size,
                visible_rect,
                min_capture_buffers,
            } => {
                // A mid-stream change is the kernel's implicit drain: the decoder is stopped
                // until the client restarts the CAPTURE queue (`STREAMON(CAPTURE)` after the
                // reallocation, or `V4L2_DEC_CMD_START`). Nothing lent is lost by this -- the
                // backend reports every frame of the old resolution before it announces the new
                // one -- and it is also what makes a backend that cannot mark its last old-size
                // buffer (`is_last`) behave: no further CAPTURE buffer is handed to it at a size
                // it has stopped decoding into. If the stop is this event's doing (no `LAST`
                // buffer has stopped the decoder already), the client is owed that `LAST`
                // buffer: the backend may still send it -- the MediaCodec backend does, right
                // behind this event, the kernel's order -- else `qbuf` supplies it.
                if session.format_announced && session.drain != Drain::Stopped {
                    session.drain = Drain::Stopped;
                    session.last_owed = session.state.capture_streaming;
                }
                session.format_announced = true;
                session.coded_size = coded_size;
                session.crop = CropRectangle::FromStream(visible_rect);
                session.min_capture_buffers = min_capture_buffers;
                if session.src_change_subscribed {
                    self.evt_queue
                        .send_event(V4l2Event::Event(SessionEvent::new(
                            session.id,
                            bindings::v4l2_event {
                                type_: bindings::V4L2_EVENT_SOURCE_CHANGE,
                                u: bindings::v4l2_event__bindgen_ty_1 {
                                    src_change: bindings::v4l2_event_src_change {
                                        changes: SrcChanges::RESOLUTION.bits(),
                                    },
                                },
                                ..Default::default()
                            },
                        )));
                }
            }
            DecoderEvent::Error(reason) => {
                self.end_session(session, &reason);
            }
        }
    }
}

impl<B, Q, M, HM, A, Reader, Writer> VirtioMediaDevice<Reader, Writer>
    for VideoDecoder<B, Q, M, HM, A>
where
    B: VideoDecoderBackend,
    Q: VirtioMediaEventQueue,
    M: VirtioMediaGuestMemoryMapper,
    HM: VirtioMediaHostMemoryMapper,
    A: VirtioMediaBufferAllocator,
    Reader: ReadFromDescriptorChain,
    Writer: WriteToDescriptorChain,
{
    type Session = VideoDecoderSession<M::GuestMemoryMapping, B::Session>;

    fn new_session(&mut self, session_id: u32) -> Result<Self::Session, i32> {
        let caps = self.backend.capabilities();
        let first = caps.coded_formats.first().ok_or_else(|| {
            log::error!("decoder: backend has no coded format");
            libc::ENODEV
        })?;
        let coded_size = (
            first.width.clamp(DEFAULT_CODED_SIZE.0),
            first.height.clamp(DEFAULT_CODED_SIZE.1),
        );
        let fourcc = first.fourcc;
        let signal = Arc::new(DecoderSignal::new()?);
        let backend = self
            .backend
            .new_session(session_id, DecoderSink(Arc::clone(&signal)))?;
        Ok(VideoDecoderSession {
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
            output_sizeimage: MIN_BITSTREAM_SIZE,
            crop: CropRectangle::Settable(v4l2r::Rect::new(0, 0, coded_size.0, coded_size.1)),
            min_capture_buffers: 1,
            colorspace: Default::default(),
            src_change_subscribed: false,
            eos_subscribed: false,
            format_announced: false,
            warned_small_capture: false,
            last_owed: false,
            dead: false,
            sequence: 0,
        })
    }

    fn close_session(&mut self, mut session: Self::Session) {
        if self.active_session == Some(session.id) {
            self.active_session = None;
        }
        // The backend first, so no thread is writing into a buffer that goes away below.
        session.backend.stop();
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

impl<B, Q, M, HM, A> VirtioMediaIoctlHandler for VideoDecoder<B, Q, M, HM, A>
where
    B: VideoDecoderBackend,
    Q: VirtioMediaEventQueue,
    M: VirtioMediaGuestMemoryMapper,
    HM: VirtioMediaHostMemoryMapper,
    A: VirtioMediaBufferAllocator,
{
    type Session = VideoDecoderSession<M::GuestMemoryMapping, B::Session>;

    fn enum_fmt(
        &mut self,
        _session: &Self::Session,
        queue: QueueType,
        index: u32,
    ) -> IoctlResult<v4l2_fmtdesc> {
        let (fourcc, flags) = match queue {
            QueueType::VideoOutputMplane => {
                let f = self
                    .backend
                    .capabilities()
                    .coded_formats
                    .get(index as usize)
                    .ok_or(libc::EINVAL)?;
                let mut flags = bindings::V4L2_FMT_FLAG_COMPRESSED;
                if f.dynamic_resolution {
                    flags |= bindings::V4L2_FMT_FLAG_DYN_RESOLUTION;
                }
                (f.fourcc, flags)
            }
            QueueType::VideoCaptureMplane => {
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

    /// Stepwise frame sizes for a coded (OUTPUT) format or for NV12 (CAPTURE). A decoder has no
    /// frame rate, so `ENUM_FRAMEINTERVALS` and `G/S_PARM` are left at their `ENOTTY` default,
    /// which `v4l2-compliance` requires of an m2m decoder.
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
        // For NV12 (CAPTURE) the sizes follow the coded format `S_FMT(OUTPUT)` selected (the
        // first offered one until then), not always the first: on 5566 that is H264's
        // `96..8192` against VP9's `96..4096` (review-m6 R6-16).
        let coded = if asked == NV12 {
            caps.coded_format(session.coded_format)
                .or_else(|| caps.coded_formats.first())
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
            QueueDirection::Output => {
                let coded = self.adjust_coded_format(pix_mp.pixelformat)?;
                let width = coded.width.clamp(if pix_mp.width == 0 {
                    session.coded_size.0
                } else {
                    pix_mp.width
                });
                let height = coded.height.clamp(if pix_mp.height == 0 {
                    session.coded_size.1
                } else {
                    pix_mp.height
                });
                let sizeimage = pix_mp.plane_fmt[0]
                    .sizeimage
                    .max(nv12_sizeimage(width, height) / 2)
                    .max(MIN_BITSTREAM_SIZE);
                let colorspace = V4l2FormatColorspace::from_pix_mp(&pix_mp);
                Ok(coded_output_format(coded.fourcc, width, height, sizeimage, colorspace))
            }
            // CAPTURE is always NV12 at the current coded size; only the size may be echoed.
            QueueDirection::Capture => Ok(session.format(QueueDirection::Capture)),
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
            QueueDirection::Output => {
                // The OUTPUT format governs the decode; changing it while buffers are allocated
                // is refused, and a running codec cannot be reconfigured (kernel decoder
                // interface, "Commit Points").
                if !session.input.buffers.is_empty()
                    || !session.output.buffers.is_empty()
                    || session.codec_started
                {
                    return Err(libc::EBUSY);
                }
                let adjusted = self.try_fmt(session, queue, format)?;
                // SAFETY: multi-planar.
                let pix_mp = unsafe { adjusted.fmt.pix_mp };
                session.coded_format = PixelFormat::from_u32(pix_mp.pixelformat);
                session.coded_size = (pix_mp.width, pix_mp.height);
                session.output_sizeimage = pix_mp.plane_fmt[0].sizeimage;
                session.colorspace = V4l2FormatColorspace::from_pix_mp(&pix_mp);
                if let CropRectangle::Settable(rect) = &mut session.crop {
                    *rect = v4l2r::Rect::new(0, 0, pix_mp.width, pix_mp.height);
                }
                Ok(adjusted)
            }
            QueueDirection::Capture => {
                // The client may set the CAPTURE format but the decoder only offers NV12 at the
                // established coded size; the pixel format is not negotiable.
                Ok(session.format(QueueDirection::Capture))
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
        // refused, as vb2 does (`vb2_core_reqbufs`, Linux 6.18.21
        // `drivers/media/common/videobuf2/videobuf2-core.c:883-886`), because the buffers it
        // would free may be lent to the backend and freeing them without a join is exactly the
        // §2.5 violation (`M7-crate` §9 item 3). A dead session may free, never allocate: pool
        // space a decode can never use (review-m6 R6-11).
        if count == 0 {
            self.streamoff(session, queue)?;
        } else if session.dead {
            return Err(libc::ENODEV);
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
        if session.dead {
            return Err(libc::ENODEV);
        }
        match self.active_session {
            Some(id) if id != session.id => return Err(libc::EBUSY),
            _ => (),
        }
        // Unlike `REQBUFS`, `CREATE_BUFS` is *not* refused on a streaming queue: vb2 does not
        // refuse it either (`vb2_core_create_bufs` has no `q->streaming` check, Linux 6.18.21
        // `videobuf2-core.c:1038-1081`), and it only appends buffers -- it frees nothing, so no
        // lent buffer can go away under the backend. GStreamer's pool grows itself this way
        // while it streams.
        //
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
                // The prepared payload wins here too (review-m6 R6-7): with this call's own
                // `bytesused` of 0 `lend_input` would hand the backend the whole buffer.
                let plane = entry.v4l2_buffer.get_first_plane_mut();
                *plane.bytesused = if direction == QueueDirection::Output {
                    prepared
                        .map(|(bytesused, _)| bytesused)
                        .unwrap_or(guest_bytesused)
                } else {
                    0
                };
            }
            Backing::Guest(slot) => {
                // `length` sizes the mapping: a bitstream buffer and a frame buffer must both
                // hold what the queue's format says, and no more than the buffer was allocated
                // for (vb2's `min_length` rule for `USERPTR`, `__prepare_userptr`, Linux 6.18.21
                // `videobuf2-core.c:1291-1299`).
                //
                // Checked on *this* call's plane whatever `PREPARE_BUF` accepted earlier: the
                // scatter list about to be mapped was read against this call's `length`
                // (`ioctl::get_userptr_regions`), so that is the number that says how much guest
                // memory the loan really covers (review-m4 R2, as `camera.rs` and
                // `video_encoder.rs` do). The prepared length is checked too, for what it
                // legitimately describes.
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
                    if direction == QueueDirection::Output {
                        v4l2_buffer.set_timestamp(buffer.timestamp());
                        *v4l2_buffer.get_first_plane_mut().bytesused = guest_bytesused;
                    } else {
                        *v4l2_buffer.get_first_plane_mut().bytesused = 0;
                    }
                    entry.v4l2_buffer = v4l2_buffer;
                }
            }
        }

        if direction == QueueDirection::Output {
            entry.v4l2_buffer.set_timestamp(buffer.timestamp());
        }
        entry.queued = true;
        entry.prepared = None;
        entry
            .v4l2_buffer
            .clear_flags(BufferFlags::PREPARED | BufferFlags::LAST | BufferFlags::DONE);
        entry.v4l2_buffer.add_flags(BufferFlags::QUEUED);
        let reply = entry.v4l2_buffer.clone();

        match direction {
            QueueDirection::Output => {
                if session.codec_started && session.state.output_streaming {
                    if let Err(e) = Self::lend_input(session, index) {
                        self.end_session(session, &format!("the backend refused input: errno {e}"));
                        return Err(libc::EIO);
                    }
                } else {
                    session.input.pending.push_back(index);
                }
            }
            QueueDirection::Capture => {
                // A decoder stopped by a resolution change that still owes its `LAST` buffer
                // answers with this one, empty: the kernel's `v4l2_m2m_qbuf` rule for a stopped
                // decoder, and what a client waiting for the flag needs (review-m6 R6-2).
                if session.drain == Drain::Stopped && session.last_owed {
                    self.return_empty_last(session, index);
                    return Ok(reply);
                }
                session.output.pending.push_back(index);
                self.try_send_pending_capture(session)?;
            }
        }

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
        if session.dead {
            return Err(libc::ENODEV);
        }
        // The same rule `qbuf` applies, with no prepared description to fall back on.
        if !payload.is_accepted_by(direction, buffer.memory(), NUM_PLANES) {
            return Err(libc::EINVAL);
        }
        let sizeimage = session.sizeimage(direction);
        let q = session.queue_mut(queue_type)?;
        let entry = q.buffers.get_mut(buffer.index() as usize).ok_or(libc::EINVAL)?;
        if entry.queued || entry.prepared.is_some() || Some(buffer.memory()) != q.memory {
            return Err(libc::EINVAL);
        }

        let guest_plane = buffer.planes_iter().next().ok_or(libc::EINVAL)?;
        let guest_bytesused = *guest_plane.bytesused;
        let guest_length = *guest_plane.length;

        if let Backing::Guest(_) = &entry.backing {
            // The rule `qbuf` applies, so a length accepted here is one a `QBUF` can use: vb2
            // runs the same `__prepare_userptr` for both (Linux 6.18.21
            // `drivers/media/common/videobuf2/videobuf2-core.c:1291-1299`).
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

    fn streamon(&mut self, session: &mut Self::Session, queue: QueueType) -> IoctlResult<()> {
        let direction = queue.direction_or_einval()?;
        if session.dead {
            return Err(libc::ENODEV);
        }
        if session.queue(queue)?.buffers.is_empty() {
            return Err(libc::EINVAL);
        }
        match direction {
            QueueDirection::Output => {
                if session.state.output_streaming {
                    return Ok(());
                }
                // The codec is created at the first OUTPUT streamon; a later streamon after a
                // seek (STREAMOFF(OUTPUT)) just resumes it.
                if !session.codec_started {
                    let (fmt, size) = (session.coded_format, session.coded_size);
                    session.backend.start(fmt, size)?;
                    session.codec_started = true;
                }
                session.state.output_streaming = true;
                // Lend every OUTPUT buffer queued before streaming.
                while let Some(index) = session.input.pending.pop_front() {
                    if let Err(e) = Self::lend_input(session, index) {
                        self.end_session(session, &format!("the backend refused input: errno {e}"));
                        return Err(libc::EIO);
                    }
                }
                self.try_send_pending_capture(session)?;
            }
            QueueDirection::Capture => {
                if session.state.capture_streaming {
                    return Ok(());
                }
                session.state.capture_streaming = true;
                // The sequence counter counts frames since this queue started streaming
                // (`v4l2_buffer.sequence`; review-m6 R6-16).
                session.sequence = 0;
                // A CAPTURE restart clears a stopped decoder, whether a drain or a resolution
                // change stopped it (kernel decoder interface, "Drain" and "Dynamic Resolution
                // Change": the sequence resumes with `STREAMOFF`/`STREAMON(CAPTURE)`).
                if session.drain == Drain::Stopped {
                    session.drain = Drain::None;
                    session.last_owed = false;
                }
                self.try_send_pending_capture(session)?;
            }
        }
        Ok(())
    }

    /// `STREAMOFF` never fails: V4L2 has it remove every buffer from the queue whatever the
    /// state of the hardware, and a guest that cannot `STREAMOFF` cannot `REQBUFS(0)` either
    /// (review-m6 R6-4). A backend that cannot flush or clear -- a wedged codec past its bound
    /// -- ends the session instead; the queue is reset and the buffers unqueued either way, so
    /// the `media_host` lease can be freed.
    fn streamoff(&mut self, session: &mut Self::Session, queue: QueueType) -> IoctlResult<()> {
        let direction = queue.direction_or_einval()?;
        match direction {
            QueueDirection::Output => {
                // Seek: the backend drops its pending input; the CAPTURE queue keeps streaming.
                if session.codec_started && !session.dead {
                    if let Err(e) = session.backend.flush() {
                        self.end_session(
                            session,
                            &format!("the backend could not flush: errno {e}"),
                        );
                    }
                }
                session.state.output_streaming = false;
                session.input.pending.clear();
                session.drain = Drain::None;
                session.last_owed = false;
                for buffer in session.input.buffers.iter_mut() {
                    buffer.unqueue();
                }
            }
            QueueDirection::Capture => {
                // The backend must stop writing CAPTURE buffers before we release them (§2.5).
                if session.codec_started && !session.dead {
                    if let Err(e) = session.backend.clear_capture_buffers() {
                        self.end_session(
                            session,
                            &format!(
                                "the backend could not release the CAPTURE buffers: errno {e}"
                            ),
                        );
                    }
                }
                session.state.capture_streaming = false;
                session.output.pending.clear();
                session.drain = Drain::None;
                session.last_owed = false;
                for buffer in session.output.buffers.iter_mut() {
                    buffer.unqueue();
                }
            }
        }
        Ok(())
    }

    fn g_selection(
        &mut self,
        session: &Self::Session,
        sel_type: SelectionType,
        sel_target: SelectionTarget,
    ) -> IoctlResult<bindings::v4l2_rect> {
        match (sel_type, sel_target) {
            // Coded resolution of the stream.
            (SelectionType::Capture, SelectionTarget::CropBounds) => {
                Ok(v4l2r::Rect::new(0, 0, session.coded_size.0, session.coded_size.1).into())
            }
            // Visible area of CAPTURE buffers.
            (
                SelectionType::Capture,
                SelectionTarget::Crop
                | SelectionTarget::CropDefault
                | SelectionTarget::ComposeDefault
                | SelectionTarget::ComposeBounds
                | SelectionTarget::Compose,
            ) => Ok(session.crop.rect().into()),
            _ => Err(libc::EINVAL),
        }
    }

    fn s_selection(
        &mut self,
        session: &mut Self::Session,
        sel_type: SelectionType,
        sel_target: SelectionTarget,
        mut sel_rect: bindings::v4l2_rect,
        _sel_flags: SelectionFlags,
    ) -> IoctlResult<bindings::v4l2_rect> {
        if !matches!(
            (sel_type, sel_target),
            (SelectionType::Capture, SelectionTarget::Compose)
        ) {
            return Err(libc::EINVAL);
        }
        // Settable only until the stream fixes the crop.
        if let CropRectangle::Settable(rect) = &mut session.crop {
            sel_rect.left = sel_rect.left.max(0);
            sel_rect.top = sel_rect.top.max(0);
            sel_rect.width = sel_rect.width.min(session.coded_size.0);
            sel_rect.height = sel_rect.height.min(session.coded_size.1);
            *rect = sel_rect.into();
        }
        self.g_selection(session, sel_type, sel_target)
    }

    fn subscribe_event(
        &mut self,
        session: &mut Self::Session,
        event: EventType,
        flags: SubscribeEventFlags,
    ) -> IoctlResult<()> {
        match event {
            EventType::SourceChange(0) => {
                session.src_change_subscribed = true;
                Ok(())
            }
            EventType::Eos => {
                session.eos_subscribed = true;
                Ok(())
            }
            // A control event (D50): D29 gave the decoder a control table, so
            // `SUBSCRIBE_EVENT(V4L2_EVENT_CTRL)` must be accepted for the class marker and each
            // exposed control (the encoder does the same at `video_encoder.rs:2895`; refusing it
            // cost the decoder a `v4l2-compliance` subtest, `testEvents`). Only the initial event
            // is ever sent -- the decoder's one real control (`MIN_BUFFERS_FOR_CAPTURE`) is
            // read-only and per session, so no other subscriber can see it change -- and the class
            // marker gets no initial value, as the kernel's `v4l2_ctrl_add_event` does.
            EventType::Ctrl(id) => {
                let def = decoder_control(id).ok_or(libc::EINVAL)?;
                if flags.contains(SubscribeEventFlags::SEND_INITIAL) && !def.is_class {
                    let event = decoder_ctrl_event(session, def);
                    self.evt_queue
                        .send_event(V4l2Event::Event(SessionEvent::new(session.id, event)));
                }
                Ok(())
            }
            _ => Err(libc::EINVAL),
        }
    }

    fn unsubscribe_event(
        &mut self,
        session: &mut Self::Session,
        event: v4l2_event_subscription,
    ) -> IoctlResult<()> {
        let all = event.type_ == bindings::V4L2_EVENT_ALL;
        let mut valid = all;
        if all || matches!(EventType::try_from(&event), Ok(EventType::SourceChange(0))) {
            session.src_change_subscribed = false;
            valid = true;
        }
        if all || matches!(EventType::try_from(&event), Ok(EventType::Eos)) {
            session.eos_subscribed = false;
            valid = true;
        }
        // A control-event subscription (D50) sends only its initial event and keeps no state, so
        // there is nothing to tear down; the kernel answers 0 for a control id whether or not it
        // was subscribed (`video_encoder.rs`'s unsubscribe does the same). `V4L2_EVENT_ALL`
        // (`all`) already succeeds above.
        if matches!(EventType::try_from(&event), Ok(EventType::Ctrl(_))) {
            valid = true;
        }
        if valid {
            Ok(())
        } else {
            Err(libc::EINVAL)
        }
    }

    /// Enumerate a control by id, walking with `V4L2_CTRL_FLAG_NEXT_CTRL`. The decoder exposes the
    /// user-control class marker and the read-only `V4L2_CID_MIN_BUFFERS_FOR_CAPTURE` (D29); the
    /// walk from the last one ends in `EINVAL`, as `v4l2-compliance` expects.
    fn queryctrl(
        &mut self,
        _session: &Self::Session,
        id: CtrlId,
        flags: QueryCtrlFlags,
    ) -> IoctlResult<v4l2_queryctrl> {
        let def = decoder_query_control(id, flags)?;
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
        let def = decoder_query_control(id, flags)?;
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

    /// The decoder has no menu control, so every `QUERYMENU` is `EINVAL` (never `ENOTTY`, which
    /// hid the whole control interface before D29).
    fn querymenu(
        &mut self,
        _session: &Self::Session,
        _id: u32,
        _index: u32,
    ) -> IoctlResult<v4l2_querymenu> {
        Err(libc::EINVAL)
    }

    /// `V4L2_CID_MIN_BUFFERS_FOR_CAPTURE`, which GStreamer reads to size its CAPTURE pool. Its
    /// value follows the last `SOURCE_CHANGE` (`min_capture_buffers`). Full codec-control
    /// enumeration (the codec control class, profile/level menus) is M5; see the report.
    ///
    /// On a 6.15+ guest kernel `VIDIOC_G_CTRL` reaches the device as `G_EXT_CTRLS`, so this is
    /// answered by [`Self::g_ext_ctrls`] there; it stays for an older kernel that forwards the
    /// legacy ioctl directly.
    fn g_ctrl(&mut self, session: &Self::Session, id: u32) -> IoctlResult<v4l2_control> {
        Ok(v4l2_control {
            id,
            value: decoder_control_value(session, id)?,
        })
    }

    /// Every decoder control is read-only (`MIN_BUFFERS_FOR_CAPTURE`) or a class marker, so a
    /// `S_CTRL` is `EACCES` for a known control and `EINVAL` otherwise -- what the kernel answers,
    /// and what `v4l2-compliance` checks of a read-only control.
    fn s_ctrl(
        &mut self,
        _session: &mut Self::Session,
        id: u32,
        _value: i32,
    ) -> IoctlResult<v4l2_control> {
        match decoder_control(id) {
            Some(def) if def.is_class => Err(libc::EINVAL),
            Some(_) => Err(libc::EACCES),
            None => Err(libc::EINVAL),
        }
    }

    fn g_ext_ctrls(
        &mut self,
        session: &Self::Session,
        which: CtrlWhich,
        ctrls: &mut v4l2_ext_controls,
        ctrl_array: &mut Vec<v4l2_ext_control>,
        _user_regions: Vec<Vec<SgEntry>>,
    ) -> IoctlResult<()> {
        match check_ext_ctrls(session, ExtCtrlOp::Get, which, ctrl_array) {
            Ok(values) => {
                write_back_ext_ctrls(ctrls, ctrl_array, &values);
                Ok(())
            }
            Err((errno, error_idx)) => {
                ctrls.error_idx = error_idx;
                Err(errno)
            }
        }
    }

    fn s_ext_ctrls(
        &mut self,
        session: &mut Self::Session,
        which: CtrlWhich,
        ctrls: &mut v4l2_ext_controls,
        ctrl_array: &mut Vec<v4l2_ext_control>,
        _user_regions: Vec<Vec<SgEntry>>,
    ) -> IoctlResult<()> {
        match check_ext_ctrls(session, ExtCtrlOp::Set, which, ctrl_array) {
            Ok(values) => {
                write_back_ext_ctrls(ctrls, ctrl_array, &values);
                Ok(())
            }
            Err((errno, error_idx)) => {
                ctrls.error_idx = error_idx;
                Err(errno)
            }
        }
    }

    fn try_ext_ctrls(
        &mut self,
        session: &Self::Session,
        which: CtrlWhich,
        ctrls: &mut v4l2_ext_controls,
        ctrl_array: &mut Vec<v4l2_ext_control>,
        _user_regions: Vec<Vec<SgEntry>>,
    ) -> IoctlResult<()> {
        match check_ext_ctrls(session, ExtCtrlOp::Try, which, ctrl_array) {
            Ok(values) => {
                write_back_ext_ctrls(ctrls, ctrl_array, &values);
                Ok(())
            }
            Err((errno, error_idx)) => {
                ctrls.error_idx = error_idx;
                Err(errno)
            }
        }
    }

    fn try_decoder_cmd(
        &mut self,
        _session: &Self::Session,
        cmd: v4l2_decoder_cmd,
    ) -> IoctlResult<v4l2_decoder_cmd> {
        normalize_decoder_cmd(cmd)
    }

    fn decoder_cmd(
        &mut self,
        session: &mut Self::Session,
        cmd: v4l2_decoder_cmd,
    ) -> IoctlResult<v4l2_decoder_cmd> {
        let cmd = normalize_decoder_cmd(cmd)?;
        if session.dead {
            return Err(libc::ENODEV);
        }
        match cmd.cmd {
            bindings::V4L2_DEC_CMD_STOP => {
                // The drain only starts if both queues stream; otherwise it is a no-op success
                // (kernel decoder interface, "Drain"). `Pending` only once the backend has
                // taken the drain: a session waiting for a `LAST` buffer no drain will produce
                // is the review-m6 R6-6 shape.
                if session.state.running() && session.drain == Drain::None {
                    session.backend.drain()?;
                    session.drain = Drain::Pending;
                }
            }
            bindings::V4L2_DEC_CMD_START => {
                // The other way out of a stopped decoder, and the one the kernel names for a
                // resolution change the client answers without reallocating. The backend hears
                // of it (`resume`): it may have stopped writing when it announced the change.
                if session.drain != Drain::None {
                    session.drain = Drain::None;
                    session.last_owed = false;
                    session.backend.resume();
                    self.try_send_pending_capture(session)?;
                }
            }
            _ => return Err(libc::EINVAL),
        }
        Ok(cmd)
    }
}

// ---------------------------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------------------------

/// One control a stateful decoder answers for. The decoder exposes only the user-control class
/// marker and the read-only `V4L2_CID_MIN_BUFFERS_FOR_CAPTURE` that GStreamer reads to size its
/// `CAPTURE` pool (`M6-crate` §9 item 1); full codec-control enumeration (profile/level menus) is
/// M5.
///
/// Answering these at all matters on a 6.15+ guest kernel: the virtio-media driver stops defining
/// `.vidioc_g_ctrl` / `.vidioc_queryctrl` there and lets the V4L2 core emulate the legacy ioctls
/// through their `EXT` forms (`driver/virtio_media_ioctls.c:1879`), so a decoder that answered only
/// `g_ctrl` was invisible to every client -- `QUERYCTRL`, `QUERY_EXT_CTRL`, `QUERYMENU` and
/// `G_CTRL` all `ENOTTY`, and GStreamer never read its pool size (defect D29). The entries are in
/// ascending id order, which is the order `V4L2_CTRL_FLAG_NEXT_CTRL` walks.
struct DecoderControl {
    id: u32,
    name: &'static str,
    /// A `V4L2_CTRL_TYPE_CTRL_CLASS` marker: neither readable nor writable.
    is_class: bool,
}

const DECODER_CONTROLS: [DecoderControl; 2] = [
    DecoderControl {
        id: bindings::V4L2_CID_USER_CLASS,
        name: "User Controls",
        is_class: true,
    },
    DecoderControl {
        id: bindings::V4L2_CID_MIN_BUFFERS_FOR_CAPTURE,
        name: "Min Number of Capture Buffers",
        is_class: false,
    },
];

impl DecoderControl {
    fn v4l2_type(&self) -> u32 {
        if self.is_class {
            bindings::v4l2_ctrl_type_V4L2_CTRL_TYPE_CTRL_CLASS
        } else {
            bindings::v4l2_ctrl_type_V4L2_CTRL_TYPE_INTEGER
        }
    }

    fn flags(&self) -> u32 {
        if self.is_class {
            // "You can neither read nor write these" (the kernel's `v4l2_ctrl_fill`).
            bindings::V4L2_CTRL_FLAG_READ_ONLY | bindings::V4L2_CTRL_FLAG_WRITE_ONLY
        } else {
            // `MIN_BUFFERS_FOR_CAPTURE` is read-only and changes with every `SOURCE_CHANGE`, so
            // it is volatile: a client must re-read it rather than trust a cached value.
            bindings::V4L2_CTRL_FLAG_READ_ONLY | bindings::V4L2_CTRL_FLAG_VOLATILE
        }
    }

    /// `(minimum, maximum, step, default)`.
    fn bounds(&self) -> (i32, i32, i32, i32) {
        if self.is_class {
            (0, 0, 0, 0)
        } else {
            // A stateful decoder never needs more than the queue's ceiling; the live value comes
            // from `G_CTRL` / `G_EXT_CTRLS`, not from this default.
            (1, MAX_BUFFERS as i32, 1, 1)
        }
    }
}

/// The control class an id belongs to (`V4L2_CTRL_ID2WHICH`).
fn ctrl_class(id: u32) -> u32 {
    id & 0x0fff_0000
}

/// The exact control an id names, if the decoder has it.
fn decoder_control(id: u32) -> Option<&'static DecoderControl> {
    DECODER_CONTROLS.iter().find(|c| c.id == id)
}

/// The control an id with query flags names: the exact one, or -- with `NEXT_CTRL` -- the first
/// with a greater id, ending in `EINVAL` past the last. `NEXT_COMPOUND` alone finds nothing (the
/// decoder has no compound control).
fn decoder_query_control(
    id: CtrlId,
    flags: QueryCtrlFlags,
) -> IoctlResult<&'static DecoderControl> {
    let id: u32 = id.into();
    if flags.contains(QueryCtrlFlags::NEXT) {
        DECODER_CONTROLS
            .iter()
            .find(|c| c.id > id)
            .ok_or(libc::EINVAL)
    } else if flags.contains(QueryCtrlFlags::COMPOUND) {
        Err(libc::EINVAL)
    } else {
        decoder_control(id).ok_or(libc::EINVAL)
    }
}

/// What `G_CTRL` / `G_EXT_CTRLS` answer for a control: `EINVAL` for a class marker (the kernel's
/// `is_int` check), the live value otherwise. `MIN_BUFFERS_FOR_CAPTURE` reads the session's value.
fn decoder_control_value<GM, S>(session: &VideoDecoderSession<GM, S>, id: u32) -> IoctlResult<i32> {
    match id {
        bindings::V4L2_CID_MIN_BUFFERS_FOR_CAPTURE => Ok(session.min_capture_buffers.max(1) as i32),
        _ => match decoder_control(id) {
            Some(def) if def.is_class => Err(libc::EINVAL),
            // No other readable control exists yet (codec controls are M5).
            Some(_) => Ok(0),
            None => Err(libc::EINVAL),
        },
    }
}

/// The `V4L2_EVENT_CTRL` event describing a control's current state, for `SUBSCRIBE_EVENT` with
/// `SEND_INITIAL` (D50). Modelled on the encoder's `ctrl_event` (`video_encoder.rs:1577`): the
/// bounds and flags come from the static [`DecoderControl`] table, the value from the session
/// (`MIN_BUFFERS_FOR_CAPTURE` follows the last `SOURCE_CHANGE`). A class marker carries no value.
fn decoder_ctrl_event<GM, S>(
    session: &VideoDecoderSession<GM, S>,
    def: &DecoderControl,
) -> bindings::v4l2_event {
    let (minimum, maximum, step, default_value) = def.bounds();
    let value = if def.is_class {
        0
    } else {
        decoder_control_value(session, def.id).unwrap_or(0)
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

/// Which of the three ext-control ioctls is being served.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ExtCtrlOp {
    Get,
    Set,
    Try,
}

/// The read/validate half of `G/S/TRY_EXT_CTRLS`: the value each control of the array would
/// answer, or the errno and the `error_idx` the kernel would report -- the failing control's
/// index for `TRY`, `count` for `G`/`S`. `which` selects the values: current, default (`G` only)
/// or a class every control must belong to. The decoder has no writable control, so every `SET`
/// or `TRY` of a real control is refused with `EACCES` (read-only), which is what the kernel does.
fn check_ext_ctrls<GM, S>(
    session: &VideoDecoderSession<GM, S>,
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
        // The access flags are checked before what the control is, as the kernel does: a
        // `WRITE_ONLY` control (the class marker carries the flag) is refused for a get, a
        // `READ_ONLY` one for a set or try, both with `EACCES`.
        let flags = match decoder_control(id) {
            Some(def) => def.flags(),
            None => return Err((libc::EINVAL, fail_idx(i))),
        };
        let refused = match op {
            ExtCtrlOp::Get => flags & bindings::V4L2_CTRL_FLAG_WRITE_ONLY != 0,
            ExtCtrlOp::Try | ExtCtrlOp::Set => flags & bindings::V4L2_CTRL_FLAG_READ_ONLY != 0,
        };
        if refused {
            return Err((libc::EACCES, fail_idx(i)));
        }
        // Only a readable, non-class control reaches here (a get of `MIN_BUFFERS_FOR_CAPTURE`).
        let value = if defaults {
            decoder_control(id).map(|def| def.bounds().3).unwrap_or(0)
        } else {
            decoder_control_value(session, id).map_err(|e| (e, fail_idx(i)))?
        };
        values.push(value);
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

/// A coded (OUTPUT) `v4l2_format`. Compressed formats carry `width`/`height` (the coded size) but
/// `bytesperline = 0`, and a single plane sized `sizeimage`.
fn coded_output_format(
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
        type_: QueueType::VideoOutputMplane as u32,
        fmt: bindings::v4l2_format__bindgen_ty_1 { pix_mp },
    }
}

/// `V4L2_DEC_CMD_STOP` / `START` reduced to what a decoder implements: `STOP` and `START` with
/// their flags and `start` parameters cleared (a decoder honours none of them), `PAUSE` / `RESUME`
/// `EINVAL`. `v4l2-compliance`'s `testDecoder` checks exactly this.
fn normalize_decoder_cmd(cmd: v4l2_decoder_cmd) -> IoctlResult<v4l2_decoder_cmd> {
    let anon = match cmd.cmd {
        bindings::V4L2_DEC_CMD_STOP => bindings::v4l2_decoder_cmd__bindgen_ty_1 {
            stop: bindings::v4l2_decoder_cmd__bindgen_ty_1__bindgen_ty_1 { pts: 0 },
        },
        bindings::V4L2_DEC_CMD_START => bindings::v4l2_decoder_cmd__bindgen_ty_1 {
            start: bindings::v4l2_decoder_cmd__bindgen_ty_1__bindgen_ty_2 {
                speed: 0,
                format: 0,
            },
        },
        _ => return Err(libc::EINVAL),
    };
    Ok(v4l2_decoder_cmd {
        cmd: cmd.cmd,
        flags: 0,
        __bindgen_anon_1: anon,
    })
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
#[path = "video_decoder_tests.rs"]
mod tests;
