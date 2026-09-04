// Copyright 2026 The DroidVM Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

//! A memory-to-memory loopback device: every OUTPUT buffer the guest queues is copied, byte for
//! byte, into the next available CAPTURE buffer, and both are returned.
//!
//! It exists to test the memory model end to end without a codec or a camera in the way
//! (`VPU_DESIGN.md` §4.2). Both queues accept `MMAP` (host-owned buffers from the device's
//! [`VirtioMediaBufferAllocator`] -- a pre-shared pool on DroidVM, a memfd apiece upstream) and
//! `USERPTR` (guest-owned pages handed over as scatter-gather lists and mapped through the
//! [`VirtioMediaGuestMemoryMapper`]), so all four combinations of who owns which side can be
//! driven from `v4l2-ctl` in the guest and checked for byte equality.
//!
//! Two formats are offered, `NV12` and `RGB3`, each as a single-plane multi-planar format
//! (`num_planes = 1`, one `v4l2_plane` per buffer), with stepwise frame sizes from 64 to 4096.
//! The CAPTURE resolution follows the OUTPUT one; the CAPTURE pixel format is free, since the
//! device does not convert anything -- it copies `bytesused` bytes, clipped to what the CAPTURE
//! buffer can hold.
//!
//! Everything happens inside the ioctls (a pair is copied as soon as both queues are streaming
//! and each has a buffer), so sessions have no poll descriptor and `process_events` is a no-op.
//! Guest mappings are held exactly as long as the buffer is queued and are released before the
//! dequeue event, and before `STREAMOFF` / `REQBUFS(0)` / close return (`VPU_DESIGN.md` §2.5).

use std::collections::VecDeque;
use std::io::Result as IoResult;
use std::os::fd::BorrowedFd;

use v4l2r::bindings;
use v4l2r::bindings::v4l2_create_buffers;
use v4l2r::bindings::v4l2_event_subscription;
use v4l2r::bindings::v4l2_fmtdesc;
use v4l2r::bindings::v4l2_format;
use v4l2r::bindings::v4l2_frmsizeenum;
use v4l2r::bindings::v4l2_requestbuffers;
use v4l2r::ioctl::BufferCapabilities;
use v4l2r::ioctl::BufferField;
use v4l2r::ioctl::BufferFlags;
use v4l2r::ioctl::EventType;
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

pub const NV12: PixelFormat = PixelFormat::from_fourcc(b"NV12");
pub const RGB3: PixelFormat = PixelFormat::from_fourcc(b"RGB3");
/// The formats offered, in `ENUM_FMT` order.
const FORMATS: [(PixelFormat, &[u8]); 2] = [(NV12, b"Y/UV 4:2:0"), (RGB3, b"24-bit RGB 8-8-8")];

pub const MIN_DIMENSION: u32 = 64;
pub const MAX_DIMENSION: u32 = 4096;
/// Dimensions are kept even so NV12's half-size chroma plane needs no rounding.
const DIMENSION_STEP: u32 = 2;
const DEFAULT_WIDTH: u32 = 640;
const DEFAULT_HEIGHT: u32 = 480;
/// Most buffers per queue, the usual V4L2 ceiling.
pub const MAX_BUFFERS: usize = 32;

/// The device-level format of one queue.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FrameFormat {
    pixelformat: PixelFormat,
    width: u32,
    height: u32,
}

impl Default for FrameFormat {
    fn default() -> Self {
        Self {
            pixelformat: NV12,
            width: DEFAULT_WIDTH,
            height: DEFAULT_HEIGHT,
        }
    }
}

impl FrameFormat {
    /// The closest format this device can do to what was asked: unknown pixel formats become
    /// NV12, dimensions are clamped to the stepwise range and rounded down to a multiple of the
    /// step.
    fn adjusted(pixelformat: u32, width: u32, height: u32) -> Self {
        let pixelformat = PixelFormat::from_u32(pixelformat);
        let pixelformat = if FORMATS.iter().any(|(f, _)| *f == pixelformat) {
            pixelformat
        } else {
            NV12
        };
        let clamp = |v: u32| v.clamp(MIN_DIMENSION, MAX_DIMENSION) / DIMENSION_STEP * DIMENSION_STEP;
        Self {
            pixelformat,
            width: clamp(width),
            height: clamp(height),
        }
    }

    fn bytesperline(&self) -> u32 {
        if self.pixelformat == RGB3 {
            self.width * 3
        } else {
            self.width
        }
    }

    fn sizeimage(&self) -> u32 {
        if self.pixelformat == RGB3 {
            self.width * self.height * 3
        } else {
            // Luma plane plus interleaved half-size chroma; dimensions are even.
            self.width * self.height + 2 * (self.width / 2) * (self.height / 2)
        }
    }

    fn colorspace(&self) -> u32 {
        if self.pixelformat == RGB3 {
            bindings::v4l2_colorspace_V4L2_COLORSPACE_SRGB
        } else {
            bindings::v4l2_colorspace_V4L2_COLORSPACE_REC709
        }
    }

    /// The format as a single-plane multi-planar `v4l2_format` for `queue`.
    fn to_v4l2(self, queue: QueueType) -> v4l2_format {
        let mut pix_mp = bindings::v4l2_pix_format_mplane {
            width: self.width,
            height: self.height,
            pixelformat: self.pixelformat.to_u32(),
            field: bindings::v4l2_field_V4L2_FIELD_NONE,
            colorspace: self.colorspace(),
            num_planes: 1,
            ..Default::default()
        };
        pix_mp.plane_fmt[0] = bindings::v4l2_plane_pix_format {
            sizeimage: self.sizeimage(),
            bytesperline: self.bytesperline(),
            ..Default::default()
        };

        v4l2_format {
            type_: queue as u32,
            fmt: bindings::v4l2_format__bindgen_ty_1 { pix_mp },
        }
    }
}

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
    /// Queued and not yet returned.
    queued: bool,
    /// Bytes this buffer was created for: the `sizeimage` `REQBUFS`/`CREATE_BUFS` sized it with.
    ///
    /// It is the ceiling a guest-owned (`USERPTR`) buffer's `length` is held to at `QBUF`, and it
    /// never changes afterwards -- unlike the plane `length` of a `Backing::Guest` buffer, which
    /// is replaced by the guest's own number on every `QBUF`.
    size: u32,
}

impl<GM: GuestMemoryRange> Buffer<GM> {
    /// Bytes the buffer can hold: the host buffer's length, or the length the guest declared.
    fn capacity(&self) -> u32 {
        match &self.backing {
            Backing::Host { buffer, .. } => buffer.len.min(u32::MAX as u64) as u32,
            Backing::Guest(_) => *self.v4l2_buffer.get_first_plane().length,
        }
    }

    /// Pointer to the first byte, for reading. `None` for a guest buffer that is not mapped.
    fn data_ptr(&self) -> Option<*const u8> {
        match &self.backing {
            Backing::Host { buffer, .. } => Some(buffer.as_ptr()),
            Backing::Guest(Some(mapping)) => Some(mapping.as_ptr()),
            Backing::Guest(None) => None,
        }
    }

    /// Pointer to the first byte, for writing. `None` for a guest buffer that is not mapped.
    fn data_mut_ptr(&mut self) -> Option<*mut u8> {
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

    /// Return the buffer to the not-queued state, releasing any guest mapping.
    fn unqueue(&mut self) {
        self.drop_guest_mapping();
        self.queued = false;
        self.v4l2_buffer.clear_flags(BufferFlags::QUEUED);
    }
}

/// One of the two queues of a session.
struct Queue<GM> {
    format: FrameFormat,
    /// Memory type the buffers were allocated with; `None` while there are none.
    memory: Option<MemoryType>,
    buffers: Vec<Buffer<GM>>,
    /// Indices of queued buffers, in queueing order.
    queued: VecDeque<usize>,
    streaming: bool,
    /// Sequence number of the next buffer returned on this queue.
    sequence: u32,
}

impl<GM> Default for Queue<GM> {
    fn default() -> Self {
        Self {
            format: Default::default(),
            memory: None,
            buffers: Vec::new(),
            queued: VecDeque::new(),
            streaming: false,
            sequence: 0,
        }
    }
}

/// Session data of [`LoopbackDevice`].
pub struct LoopbackSession<GM> {
    id: u32,
    output: Queue<GM>,
    capture: Queue<GM>,
}

impl<GM> VirtioMediaDeviceSession for LoopbackSession<GM> {
    fn poll_fd(&self) -> Option<BorrowedFd> {
        None
    }
}

impl<GM> LoopbackSession<GM> {
    fn queue(&self, queue: QueueType) -> IoctlResult<&Queue<GM>> {
        match queue {
            QueueType::VideoOutputMplane => Ok(&self.output),
            QueueType::VideoCaptureMplane => Ok(&self.capture),
            _ => Err(libc::EINVAL),
        }
    }

    fn queue_mut(&mut self, queue: QueueType) -> IoctlResult<&mut Queue<GM>> {
        match queue {
            QueueType::VideoOutputMplane => Ok(&mut self.output),
            QueueType::VideoCaptureMplane => Ok(&mut self.capture),
            _ => Err(libc::EINVAL),
        }
    }

    fn has_buffers(&self) -> bool {
        !self.output.buffers.is_empty() || !self.capture.buffers.is_empty()
    }
}

/// A memory-to-memory device copying OUTPUT buffers into CAPTURE buffers. See the module
/// documentation.
pub struct LoopbackDevice<
    Q: VirtioMediaEventQueue,
    M: VirtioMediaGuestMemoryMapper,
    HM: VirtioMediaHostMemoryMapper,
    A: VirtioMediaBufferAllocator,
> {
    evt_queue: Q,
    /// Guest memory mapper, for `USERPTR` buffers.
    mem: M,
    mmap_manager: MmapMappingManager<HM>,
    /// Where `MMAP` buffers come from.
    allocator: A,
    /// Freed `MMAP` buffers the guest still maps.
    retired: RetiredBuffers,
    /// The one session allowed to hold buffers (see `SimpleCaptureDevice`).
    active_session: Option<u32>,
}

impl<Q, M, HM, A> LoopbackDevice<Q, M, HM, A>
where
    Q: VirtioMediaEventQueue,
    M: VirtioMediaGuestMemoryMapper,
    HM: VirtioMediaHostMemoryMapper,
    A: VirtioMediaBufferAllocator,
{
    pub fn new(evt_queue: Q, mem: M, mapper: HM, allocator: A) -> Self {
        Self {
            evt_queue,
            mem,
            mmap_manager: MmapMappingManager::from(mapper),
            allocator,
            retired: RetiredBuffers::new(),
            active_session: None,
        }
    }

    /// Drop every buffer of `queue`: guest mappings now, host buffers back to the allocator (or
    /// parked until the guest unmaps them).
    fn free_buffers(&mut self, queue: &mut Queue<M::GuestMemoryMapping>) {
        queue.queued.clear();
        for buffer in queue.buffers.drain(..) {
            if let Backing::Host { buffer, offset } = buffer.backing {
                self.retired.retire(
                    &mut self.mmap_manager,
                    &mut self.allocator,
                    offset,
                    buffer,
                );
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
                _ => return Err(libc::EINVAL),
            };

            added.push(Buffer {
                v4l2_buffer,
                backing,
                queued: false,
                size: sizeimage,
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

    /// Copy every OUTPUT buffer that has a CAPTURE buffer waiting for it, while both queues
    /// stream, and return both to the guest.
    fn process(&mut self, session: &mut LoopbackSession<M::GuestMemoryMapping>) {
        while session.output.streaming && session.capture.streaming {
            let (Some(&out_idx), Some(&cap_idx)) =
                (session.output.queued.front(), session.capture.queued.front())
            else {
                break;
            };
            session.output.queued.pop_front();
            session.capture.queued.pop_front();

            let out = &mut session.output.buffers[out_idx];
            let cap = &mut session.capture.buffers[cap_idx];

            let bytes = {
                let plane = out.v4l2_buffer.get_first_plane();
                let used = if *plane.bytesused == 0 {
                    *plane.length
                } else {
                    *plane.bytesused
                };
                used.min(cap.capacity()) as usize
            };

            match (out.data_ptr(), cap.data_mut_ptr()) {
                (Some(src), Some(dst)) => {
                    // SAFETY: `src` is readable for `out`'s capacity and `dst` writable for
                    // `cap`'s, `bytes` is within both (checked above), and both mappings live
                    // until the `unqueue` calls below. `copy` rather than `copy_nonoverlapping`
                    // because a guest may legitimately hand the same pages to both queues.
                    unsafe { std::ptr::copy(src, dst, bytes) };
                }
                _ => {
                    // A queued guest buffer always has its mapping; reaching here is a bug on
                    // our side, and the guest gets an error event rather than a silent frame.
                    log::error!("loopback: queued buffer without a mapping");
                    self.evt_queue.send_error(session.id, libc::EIO);
                }
            }

            // OUTPUT buffer: consumed.
            out.unqueue();
            out.v4l2_buffer.set_sequence(session.output.sequence);
            session.output.sequence = session.output.sequence.wrapping_add(1);
            let out_event = out.v4l2_buffer.clone();

            // CAPTURE buffer: filled, with the OUTPUT buffer's timestamp.
            let timestamp = out.v4l2_buffer.timestamp();
            cap.unqueue();
            *cap.v4l2_buffer.get_first_plane_mut().bytesused = bytes as u32;
            cap.v4l2_buffer.set_sequence(session.capture.sequence);
            session.capture.sequence = session.capture.sequence.wrapping_add(1);
            cap.v4l2_buffer.set_timestamp(timestamp);
            cap.v4l2_buffer.add_flags(BufferFlags::TIMESTAMP_COPY);
            let cap_event = cap.v4l2_buffer.clone();

            // Mappings are gone (`unqueue`), so a shadowed CAPTURE buffer has been written back
            // to the guest before it hears the buffer is done.
            self.evt_queue.send_event(V4l2Event::DequeueBuffer(DequeueBufferEvent::new(
                session.id, out_event,
            )));
            self.evt_queue.send_event(V4l2Event::DequeueBuffer(DequeueBufferEvent::new(
                session.id, cap_event,
            )));
        }
    }

    /// Validate `format` for `queue` and return what the device would actually use.
    fn adjust_format(
        &self,
        session: &LoopbackSession<M::GuestMemoryMapping>,
        queue: QueueType,
        format: &v4l2_format,
    ) -> IoctlResult<FrameFormat> {
        session.queue(queue)?;
        // SAFETY: both accepted queue types are multi-planar, so `pix_mp` is the live member.
        let pix_mp = unsafe { format.fmt.pix_mp };
        let mut wanted = FrameFormat::adjusted(pix_mp.pixelformat, pix_mp.width, pix_mp.height);
        if queue.direction() == QueueDirection::Capture {
            // The CAPTURE resolution follows the OUTPUT one.
            wanted.width = session.output.format.width;
            wanted.height = session.output.format.height;
        }
        Ok(wanted)
    }
}

impl<Q, M, HM, A, Reader, Writer> VirtioMediaDevice<Reader, Writer> for LoopbackDevice<Q, M, HM, A>
where
    Q: VirtioMediaEventQueue,
    M: VirtioMediaGuestMemoryMapper,
    HM: VirtioMediaHostMemoryMapper,
    A: VirtioMediaBufferAllocator,
    Reader: ReadFromDescriptorChain,
    Writer: WriteToDescriptorChain,
{
    type Session = LoopbackSession<M::GuestMemoryMapping>;

    fn new_session(&mut self, session_id: u32) -> Result<Self::Session, i32> {
        Ok(LoopbackSession {
            id: session_id,
            output: Default::default(),
            capture: Default::default(),
        })
    }

    fn close_session(&mut self, mut session: Self::Session) {
        if self.active_session == Some(session.id) {
            self.active_session = None;
        }
        // Guest mappings die with the session's buffers; host buffers go back (or wait for the
        // guest's munmap) before this returns.
        self.free_buffers(&mut session.output);
        self.free_buffers(&mut session.capture);
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
            .output
            .buffers
            .iter()
            .chain(session.capture.buffers.iter())
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

    /// Nothing is asynchronous in this device: pairs are copied inside `QBUF`/`STREAMON`.
    fn process_events(&mut self, _session: &mut Self::Session) -> Result<(), i32> {
        Ok(())
    }
}

impl<Q, M, HM, A> VirtioMediaIoctlHandler for LoopbackDevice<Q, M, HM, A>
where
    Q: VirtioMediaEventQueue,
    M: VirtioMediaGuestMemoryMapper,
    HM: VirtioMediaHostMemoryMapper,
    A: VirtioMediaBufferAllocator,
{
    type Session = LoopbackSession<M::GuestMemoryMapping>;

    fn enum_fmt(
        &mut self,
        session: &Self::Session,
        queue: QueueType,
        index: u32,
    ) -> IoctlResult<v4l2_fmtdesc> {
        session.queue(queue)?;
        let (pixelformat, description) = FORMATS.get(index as usize).ok_or(libc::EINVAL)?;
        let mut desc = v4l2_fmtdesc {
            index,
            type_: queue as u32,
            pixelformat: pixelformat.to_u32(),
            ..Default::default()
        };
        let n = description.len().min(desc.description.len() - 1);
        desc.description[..n].copy_from_slice(&description[..n]);
        Ok(desc)
    }

    fn enum_framesizes(
        &mut self,
        _session: &Self::Session,
        index: u32,
        pixel_format: u32,
    ) -> IoctlResult<v4l2_frmsizeenum> {
        // One stepwise entry per supported format.
        if index != 0 || !FORMATS.iter().any(|(f, _)| f.to_u32() == pixel_format) {
            return Err(libc::EINVAL);
        }
        Ok(v4l2_frmsizeenum {
            index: 0,
            pixel_format,
            type_: bindings::v4l2_frmsizetypes_V4L2_FRMSIZE_TYPE_STEPWISE,
            __bindgen_anon_1: bindings::v4l2_frmsizeenum__bindgen_ty_1 {
                stepwise: bindings::v4l2_frmsize_stepwise {
                    min_width: MIN_DIMENSION,
                    max_width: MAX_DIMENSION,
                    step_width: DIMENSION_STEP,
                    min_height: MIN_DIMENSION,
                    max_height: MAX_DIMENSION,
                    step_height: DIMENSION_STEP,
                },
            },
            ..Default::default()
        })
    }

    fn g_fmt(&mut self, session: &Self::Session, queue: QueueType) -> IoctlResult<v4l2_format> {
        Ok(session.queue(queue)?.format.to_v4l2(queue))
    }

    fn try_fmt(
        &mut self,
        session: &Self::Session,
        queue: QueueType,
        format: v4l2_format,
    ) -> IoctlResult<v4l2_format> {
        Ok(self.adjust_format(session, queue, &format)?.to_v4l2(queue))
    }

    fn s_fmt(
        &mut self,
        session: &mut Self::Session,
        queue: QueueType,
        format: v4l2_format,
    ) -> IoctlResult<v4l2_format> {
        let wanted = self.adjust_format(session, queue, &format)?;
        // Buffers were sized for the old format.
        if !session.queue(queue)?.buffers.is_empty() {
            return Err(libc::EBUSY);
        }
        session.queue_mut(queue)?.format = wanted;
        if queue.direction() == QueueDirection::Output {
            session.capture.format.width = wanted.width;
            session.capture.format.height = wanted.height;
        }
        Ok(wanted.to_v4l2(queue))
    }

    fn reqbufs(
        &mut self,
        session: &mut Self::Session,
        queue_type: QueueType,
        memory: MemoryType,
        count: u32,
    ) -> IoctlResult<v4l2_requestbuffers> {
        if !matches!(memory, MemoryType::Mmap | MemoryType::UserPtr) {
            return Err(libc::EINVAL);
        }
        match self.active_session {
            Some(id) if id != session.id => return Err(libc::EBUSY),
            _ => (),
        }
        let queue = session.queue_mut(queue_type)?;
        if queue.streaming {
            return Err(libc::EBUSY);
        }

        // Old buffers go first, mappings and all, so the reply never races a stale view.
        self.free_buffers(queue);
        let count = (count as usize).min(MAX_BUFFERS);
        queue.memory = None;
        if count > 0 {
            let sizeimage = queue.format.sizeimage();
            self.add_buffers(queue, queue_type, memory, count, sizeimage)?;
            queue.memory = Some(memory);
        }

        self.active_session = if session.has_buffers() {
            Some(session.id)
        } else {
            None
        };

        Ok(v4l2_requestbuffers {
            count: count as u32,
            type_: queue_type as u32,
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
        queue_type: QueueType,
        memory: MemoryType,
        format: v4l2_format,
    ) -> IoctlResult<v4l2_create_buffers> {
        if !matches!(memory, MemoryType::Mmap | MemoryType::UserPtr) {
            return Err(libc::EINVAL);
        }
        match self.active_session {
            Some(id) if id != session.id => return Err(libc::EBUSY),
            _ => (),
        }
        let wanted = self.adjust_format(session, queue_type, &format)?;
        let queue = session.queue_mut(queue_type)?;
        // One memory type per queue.
        if let Some(existing) = queue.memory {
            if existing != memory {
                return Err(libc::EINVAL);
            }
        }

        let first = queue.buffers.len();
        let count = (count as usize).min(MAX_BUFFERS - first);
        if count > 0 {
            // At least what the queue's own format needs, so the buffer stays usable.
            let sizeimage = wanted.sizeimage().max(queue.format.sizeimage());
            self.add_buffers(queue, queue_type, memory, count, sizeimage)?;
            queue.memory = Some(memory);
        }
        self.active_session = if session.has_buffers() {
            Some(session.id)
        } else {
            None
        };

        Ok(v4l2_create_buffers {
            index: first as u32,
            count: count as u32,
            memory: memory as u32,
            format: wanted.to_v4l2(queue_type),
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
    ) -> IoctlResult<V4l2Buffer> {
        let queue_type = buffer.queue();
        let direction = queue_type.direction();
        let queue = session.queue_mut(queue_type)?;
        let entry = queue
            .buffers
            .get_mut(buffer.index() as usize)
            .ok_or(libc::EINVAL)?;
        if entry.queued || Some(buffer.memory()) != queue.memory {
            return Err(libc::EINVAL);
        }

        // A guest-supplied MPLANE buffer may legitimately carry no plane at all -- v4l2r only
        // refuses `length >= VIDEO_MAX_PLANES` -- so the first plane is asked for, never assumed
        // (`get_first_plane()` would panic, and this VMM is built with `panic = 'abort'`).
        let guest_plane = buffer.planes_iter().next().ok_or(libc::EINVAL)?;
        let (guest_bytesused, guest_length) = (*guest_plane.bytesused, *guest_plane.length);
        // What this buffer was sized for at REQBUFS/CREATE_BUFS time.
        let max_length = entry.size;

        match &mut entry.backing {
            Backing::Host { .. } => {
                // Our buffer, our description; only the guest's payload size and stamp matter.
                let plane = entry.v4l2_buffer.get_first_plane_mut();
                *plane.bytesused = if direction == QueueDirection::Output {
                    guest_bytesused
                } else {
                    0
                };
            }
            Backing::Guest(slot) => {
                // `length` is entirely the guest's number and decides both how much guest memory
                // is mapped and how many bytes `process()` copies synchronously on the device
                // thread; hold it to the size the queue allocated the buffer for.
                if guest_length == 0 || guest_length > max_length {
                    return Err(libc::EINVAL);
                }
                let sgs = guest_regions.into_iter().next().ok_or(libc::EINVAL)?;
                let mapping = self.mem.new_mapping(sgs).map_err(|e| {
                    log::error!("failed to map USERPTR buffer: {:#}", e);
                    guest_mapping_errno(&e)
                })?;
                *slot = Some(mapping);
                // The guest's view of its own buffer -- userptr and length -- is what must be
                // echoed back in the dequeue event.
                let mut v4l2_buffer = buffer.clone();
                v4l2_buffer.set_field(BufferField::None);
                v4l2_buffer.set_flags(BufferFlags::TIMESTAMP_COPY);
                *v4l2_buffer.get_first_plane_mut().bytesused =
                    if direction == QueueDirection::Output {
                        guest_bytesused.min(guest_length)
                    } else {
                        0
                    };
                entry.v4l2_buffer = v4l2_buffer;
            }
        }

        if direction == QueueDirection::Output {
            entry.v4l2_buffer.set_timestamp(buffer.timestamp());
            // "bytesused == 0 means the whole buffer" -- resolve it now, against our capacity.
            let capacity = entry.capacity();
            let plane = entry.v4l2_buffer.get_first_plane_mut();
            if *plane.bytesused == 0 || *plane.bytesused > capacity {
                *plane.bytesused = capacity;
            }
        }

        entry.queued = true;
        entry.v4l2_buffer.add_flags(BufferFlags::QUEUED);
        queue.queued.push_back(buffer.index() as usize);
        let reply = entry.v4l2_buffer.clone();

        self.process(session);

        Ok(reply)
    }

    fn streamon(&mut self, session: &mut Self::Session, queue_type: QueueType) -> IoctlResult<()> {
        let queue = session.queue_mut(queue_type)?;
        if queue.buffers.is_empty() {
            return Err(libc::EINVAL);
        }
        queue.streaming = true;
        self.process(session);
        Ok(())
    }

    fn streamoff(&mut self, session: &mut Self::Session, queue_type: QueueType) -> IoctlResult<()> {
        let queue = session.queue_mut(queue_type)?;
        queue.streaming = false;
        queue.queued.clear();
        for buffer in queue.buffers.iter_mut() {
            // Guest mappings go before we answer: the guest gives the pages back once it
            // hears from us.
            buffer.unqueue();
        }
        Ok(())
    }

    /// This device never emits `EOS` or `SOURCE_CHANGE`, but a codec-shaped client subscribes
    /// to them as a matter of course and must not be refused.
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
    use std::rc::Rc;

    use v4l2r::ioctl::UncheckedV4l2Buffer;

    use super::*;
    use crate::MemFdAllocator;

    /// Collects the events the device sends.
    #[derive(Default)]
    struct EventLog(Rc<RefCell<Vec<V4l2Event>>>);

    impl VirtioMediaEventQueue for EventLog {
        fn send_event(&mut self, event: V4l2Event) {
            self.0.borrow_mut().push(event);
        }
    }

    /// A pretend guest: one flat byte array standing in for guest-physical memory, from which
    /// SG lists are "mapped" by pointing straight into it. Live mappings are counted so the
    /// tests can check they are released when the design says they must be.
    #[derive(Clone)]
    struct FakeGuest {
        memory: Rc<RefCell<Vec<u8>>>,
        live_mappings: Rc<RefCell<usize>>,
    }

    struct FakeMapping {
        guest: FakeGuest,
        start: usize,
    }

    impl GuestMemoryRange for FakeMapping {
        fn as_ptr(&self) -> *const u8 {
            // SAFETY: nothing else resizes `memory` while the test runs.
            unsafe { self.guest.memory.as_ptr().as_ref().unwrap().as_ptr().add(self.start) }
        }

        fn as_mut_ptr(&mut self) -> *mut u8 {
            // SAFETY: as above.
            unsafe { self.guest.memory.as_ptr().as_mut().unwrap().as_mut_ptr().add(self.start) }
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
            // Only contiguous lists, which is all these tests send.
            let start = sgs.first().map(|sg| sg.start).unwrap_or(0) as usize;
            let total: usize = sgs.iter().map(|sg| sg.len as usize).sum();
            if total == 0 || start + total > self.memory.borrow().len() {
                anyhow::bail!("bad SG list");
            }
            *self.live_mappings.borrow_mut() += 1;
            Ok(FakeMapping {
                guest: self.clone(),
                start,
            })
        }
    }

    /// The mapper answers the buffer's own pool offset, or a fixed address for memfd buffers.
    struct FakeHostMapper;

    impl VirtioMediaHostMemoryMapper for FakeHostMapper {
        fn add_mapping(&mut self, buffer: &HostBuffer, offset: u64, _rw: bool) -> Result<u64, i32> {
            Ok(buffer.pool_offset.unwrap_or(0x8000_0000 + offset))
        }

        fn remove_mapping(&mut self, _shm_offset: u64) -> Result<(), i32> {
            Ok(())
        }
    }

    /// Counts what the device gives back.
    struct CountingAllocator {
        inner: MemFdAllocator,
        released: Rc<RefCell<usize>>,
    }

    impl VirtioMediaBufferAllocator for CountingAllocator {
        fn allocate(&mut self, len: u64) -> Result<HostBuffer, i32> {
            self.inner.allocate(len)
        }

        fn release(&mut self, buf: HostBuffer) {
            *self.released.borrow_mut() += 1;
            self.inner.release(buf);
        }
    }

    type Device = LoopbackDevice<EventLog, FakeGuest, FakeHostMapper, CountingAllocator>;
    type Session = LoopbackSession<FakeMapping>;

    struct Rig {
        device: Device,
        events: Rc<RefCell<Vec<V4l2Event>>>,
        guest: FakeGuest,
        released: Rc<RefCell<usize>>,
    }

    const GUEST_MEMORY: usize = 1 << 20;

    fn rig() -> Rig {
        let events = EventLog::default();
        let events_log = Rc::clone(&events.0);
        let guest = FakeGuest {
            memory: Rc::new(RefCell::new(vec![0u8; GUEST_MEMORY])),
            live_mappings: Rc::new(RefCell::new(0)),
        };
        let released = Rc::new(RefCell::new(0));
        let device = LoopbackDevice::new(
            events,
            guest.clone(),
            FakeHostMapper,
            CountingAllocator {
                inner: MemFdAllocator::new(),
                released: Rc::clone(&released),
            },
        );
        Rig {
            device,
            events: events_log,
            guest,
            released,
        }
    }

    fn session(device: &mut Device) -> Session {
        <Device as VirtioMediaDevice<&[u8], Vec<u8>>>::new_session(device, 0).unwrap()
    }

    fn close(device: &mut Device, session: Session) {
        <Device as VirtioMediaDevice<&[u8], Vec<u8>>>::close_session(device, session)
    }

    fn munmap(device: &mut Device, guest_addr: u64) -> Result<(), i32> {
        <Device as VirtioMediaDevice<&[u8], Vec<u8>>>::do_munmap(device, guest_addr)
    }

    fn mmap(device: &mut Device, session: &mut Session, offset: u32) -> Result<(u64, u64), i32> {
        <Device as VirtioMediaDevice<&[u8], Vec<u8>>>::do_mmap(
            device,
            session,
            VIRTIO_MEDIA_MMAP_FLAG_RW,
            offset,
        )
    }

    fn format(queue: QueueType, pixelformat: PixelFormat, width: u32, height: u32) -> v4l2_format {
        FrameFormat {
            pixelformat,
            width,
            height,
        }
        .to_v4l2(queue)
    }

    fn sizeimage(format: &v4l2_format) -> u32 {
        // SAFETY: every format these tests build is multi-planar.
        unsafe { format.fmt.pix_mp.plane_fmt[0].sizeimage }
    }

    /// A USERPTR buffer for `queue`, backed by `len` bytes of fake guest memory at `gpa`.
    fn userptr_buffer(queue: QueueType, index: u32, gpa: u64, len: u32) -> (V4l2Buffer, Vec<Vec<SgEntry>>) {
        let mut buffer = V4l2Buffer::new(queue, index, MemoryType::UserPtr);
        if let V4l2PlanesWithBackingMut::UserPtr(mut planes) = buffer.planes_with_backing_iter_mut()
        {
            let mut plane = planes.next().unwrap();
            plane.set_userptr(0xc000_0000 + index as u64);
            *plane.length = len;
        }
        (buffer, vec![vec![SgEntry::new(gpa, len)]])
    }

    fn mmap_buffer(queue: QueueType, index: u32, len: u32) -> V4l2Buffer {
        let mut buffer = V4l2Buffer::new(queue, index, MemoryType::Mmap);
        *buffer.get_first_plane_mut().length = len;
        buffer
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

    #[test]
    fn formats_are_single_plane_and_capture_follows_output() {
        let mut r = rig();
        let mut s = session(&mut r.device);

        let desc = r.device.enum_fmt(&s, QueueType::VideoOutputMplane, 0).unwrap();
        assert_eq!(desc.pixelformat, NV12.to_u32());
        let desc = r.device.enum_fmt(&s, QueueType::VideoCaptureMplane, 1).unwrap();
        assert_eq!(desc.pixelformat, RGB3.to_u32());
        assert_eq!(
            r.device.enum_fmt(&s, QueueType::VideoCaptureMplane, 2).err(),
            Some(libc::EINVAL)
        );
        assert_eq!(
            r.device.enum_fmt(&s, QueueType::VideoCapture, 0).err(),
            Some(libc::EINVAL)
        );

        let sizes = r.device.enum_framesizes(&s, 0, NV12.to_u32()).unwrap();
        // SAFETY: the type says stepwise.
        let stepwise = unsafe { sizes.__bindgen_anon_1.stepwise };
        assert_eq!((stepwise.min_width, stepwise.max_width), (64, 4096));
        assert_eq!(
            r.device.enum_framesizes(&s, 1, NV12.to_u32()).err(),
            Some(libc::EINVAL)
        );
        assert_eq!(
            r.device.enum_framesizes(&s, 0, 0x1234_5678).err(),
            Some(libc::EINVAL)
        );

        // Odd, out-of-range and unknown are all adjusted, never refused.
        let out = r
            .device
            .s_fmt(
                &mut s,
                QueueType::VideoOutputMplane,
                format(QueueType::VideoOutputMplane, RGB3, 8193, 33),
            )
            .unwrap();
        // SAFETY: multi-planar. (Fields are copied out because the struct is packed.)
        let pix_mp = unsafe { out.fmt.pix_mp };
        let (width, height, num_planes) = (pix_mp.width, pix_mp.height, pix_mp.num_planes);
        let (sizeimage, bytesperline) = (
            pix_mp.plane_fmt[0].sizeimage,
            pix_mp.plane_fmt[0].bytesperline,
        );
        assert_eq!((width, height), (4096, 64));
        assert_eq!(num_planes, 1);
        assert_eq!(sizeimage, 4096 * 64 * 3);
        assert_eq!(bytesperline, 4096 * 3);

        // CAPTURE keeps its own pixel format but takes OUTPUT's size.
        let cap = r
            .device
            .s_fmt(
                &mut s,
                QueueType::VideoCaptureMplane,
                format(QueueType::VideoCaptureMplane, NV12, 100, 100),
            )
            .unwrap();
        // SAFETY: multi-planar.
        let pix_mp = unsafe { cap.fmt.pix_mp };
        let (width, height, pixelformat) = (pix_mp.width, pix_mp.height, pix_mp.pixelformat);
        let sizeimage = pix_mp.plane_fmt[0].sizeimage;
        assert_eq!((width, height), (4096, 64));
        assert_eq!(pixelformat, NV12.to_u32());
        assert_eq!(sizeimage, 4096 * 64 * 3 / 2);

        let cap = r.device.g_fmt(&s, QueueType::VideoCaptureMplane).unwrap();
        // SAFETY: multi-planar.
        let width = unsafe { cap.fmt.pix_mp.width };
        assert_eq!(width, 4096);
        close(&mut r.device, s);
    }

    /// Guest-owned OUTPUT, host-owned CAPTURE: the design's default split.
    #[test]
    fn userptr_output_is_copied_into_mmap_capture() {
        let mut r = rig();
        let mut s = session(&mut r.device);
        let fmt = format(QueueType::VideoOutputMplane, RGB3, 64, 64);
        r.device.s_fmt(&mut s, QueueType::VideoOutputMplane, fmt).unwrap();
        // Same pixel format on CAPTURE, so a whole frame fits (the size already follows).
        r.device
            .s_fmt(
                &mut s,
                QueueType::VideoCaptureMplane,
                format(QueueType::VideoCaptureMplane, RGB3, 64, 64),
            )
            .unwrap();
        let size = sizeimage(&fmt);

        let reply = r
            .device
            .reqbufs(&mut s, QueueType::VideoOutputMplane, MemoryType::UserPtr, 2)
            .unwrap();
        assert_eq!(reply.count, 2);
        assert!(reply.capabilities & BufferCapabilities::SUPPORTS_USERPTR.bits() != 0);
        let reply = r
            .device
            .reqbufs(&mut s, QueueType::VideoCaptureMplane, MemoryType::Mmap, 1)
            .unwrap();
        assert_eq!(reply.count, 1);

        // Pattern in "guest memory" at page 4.
        let gpa = 4 * 0x1000u64;
        let payload = size - 100;
        {
            let mut mem = r.guest.memory.borrow_mut();
            for (i, b) in mem[gpa as usize..gpa as usize + payload as usize].iter_mut().enumerate() {
                *b = (i * 7 % 251) as u8;
            }
        }

        r.device.streamon(&mut s, QueueType::VideoOutputMplane).unwrap();
        r.device.streamon(&mut s, QueueType::VideoCaptureMplane).unwrap();

        // CAPTURE first: nothing happens until an OUTPUT buffer arrives.
        let cap = mmap_buffer(QueueType::VideoCaptureMplane, 0, size);
        let reply = r.device.qbuf(&mut s, cap, vec![]).unwrap();
        assert!(reply.flags().contains(BufferFlags::QUEUED));
        assert!(r.events.borrow().is_empty());

        let (mut out, sgs) = userptr_buffer(QueueType::VideoOutputMplane, 1, gpa, size);
        *out.get_first_plane_mut().bytesused = payload;
        out.set_timestamp(bindings::timeval {
            tv_sec: 12,
            tv_usec: 34,
        });
        let reply = r.device.qbuf(&mut s, out, sgs).unwrap();
        // The reply is our view of the queued buffer, userptr preserved.
        assert_eq!(reply.index(), 1);
        assert_eq!(reply.memory(), MemoryType::UserPtr);

        let events = dequeued(&r.events.borrow());
        assert_eq!(events.len(), 2, "one DQBUF per queue");
        let (out_ev, cap_ev) = (&events[0], &events[1]);
        assert_eq!(out_ev.queue(), QueueType::VideoOutputMplane);
        assert_eq!(out_ev.index(), 1);
        assert!(!out_ev.flags().contains(BufferFlags::QUEUED));
        assert_eq!(out_ev.sequence(), 0);
        if let v4l2r::ioctl::V4l2PlanesWithBacking::UserPtr(mut planes) =
            out_ev.planes_with_backing_iter()
        {
            assert_eq!(planes.next().unwrap().userptr(), 0xc000_0001);
        } else {
            panic!("OUTPUT buffer lost its memory type");
        }
        assert_eq!(cap_ev.queue(), QueueType::VideoCaptureMplane);
        assert_eq!(cap_ev.index(), 0);
        assert_eq!(*cap_ev.get_first_plane().bytesused, payload);
        assert_eq!(cap_ev.sequence(), 0);
        assert_eq!((cap_ev.timestamp().tv_sec, cap_ev.timestamp().tv_usec), (12, 34));
        assert!(cap_ev.flags().contains(BufferFlags::TIMESTAMP_COPY));
        assert!(!cap_ev.flags().contains(BufferFlags::QUEUED));

        // The guest mapping was let go before the event was sent.
        assert_eq!(*r.guest.live_mappings.borrow(), 0);

        // And the CAPTURE buffer holds the bytes.
        let Backing::Host { buffer, .. } = &s.capture.buffers[0].backing else {
            panic!("CAPTURE buffer is not host-owned");
        };
        let expected: Vec<u8> = (0..payload as usize).map(|i| (i * 7 % 251) as u8).collect();
        assert_eq!(&buffer.as_slice()[..payload as usize], &expected[..]);

        // Requeueing a dequeued buffer works, requeueing a queued one does not.
        let (out, sgs) = userptr_buffer(QueueType::VideoOutputMplane, 0, gpa, size);
        r.device.qbuf(&mut s, out, sgs).unwrap();
        assert_eq!(*r.guest.live_mappings.borrow(), 1, "held while queued");
        let (out, sgs) = userptr_buffer(QueueType::VideoOutputMplane, 0, gpa, size);
        assert_eq!(r.device.qbuf(&mut s, out, sgs).err(), Some(libc::EINVAL));
        // A memory type other than the one REQBUFS chose is refused.
        assert_eq!(
            r.device
                .qbuf(&mut s, mmap_buffer(QueueType::VideoOutputMplane, 1, size), vec![])
                .err(),
            Some(libc::EINVAL)
        );

        // STREAMOFF releases the pending OUTPUT buffer's mapping before returning.
        r.device.streamoff(&mut s, QueueType::VideoOutputMplane).unwrap();
        assert_eq!(*r.guest.live_mappings.borrow(), 0);
        assert!(!s.output.buffers[0].queued);

        close(&mut r.device, s);
        assert_eq!(*r.released.borrow(), 1, "the CAPTURE buffer went back");
    }

    /// The other way round: host-owned OUTPUT, guest-owned CAPTURE, with the copy clipped to
    /// what the CAPTURE side can hold, and the guest's CAPTURE mapping gone before the event.
    #[test]
    fn mmap_output_is_copied_into_userptr_capture() {
        let mut r = rig();
        let mut s = session(&mut r.device);
        let fmt = format(QueueType::VideoOutputMplane, NV12, 64, 64);
        r.device.s_fmt(&mut s, QueueType::VideoOutputMplane, fmt).unwrap();
        let size = sizeimage(&fmt);
        assert_eq!(size, 64 * 64 * 3 / 2);

        r.device
            .reqbufs(&mut s, QueueType::VideoOutputMplane, MemoryType::Mmap, 1)
            .unwrap();
        r.device
            .reqbufs(&mut s, QueueType::VideoCaptureMplane, MemoryType::UserPtr, 1)
            .unwrap();

        // Fill the host OUTPUT buffer the way the guest would through its mapping.
        let offset = if let Backing::Host { buffer, offset } = &mut s.output.buffers[0].backing {
            buffer.as_mut_slice().fill(0xab);
            *offset
        } else {
            panic!("OUTPUT buffer is not host-owned");
        };
        let (guest_addr, mapped_len) = mmap(&mut r.device, &mut s, offset).unwrap();
        assert_eq!(mapped_len, size as u64);

        r.device.streamon(&mut s, QueueType::VideoCaptureMplane).unwrap();
        r.device.streamon(&mut s, QueueType::VideoOutputMplane).unwrap();

        // bytesused == 0 means the whole buffer; the CAPTURE buffer is 100 bytes shorter.
        let out = mmap_buffer(QueueType::VideoOutputMplane, 0, size);
        r.device.qbuf(&mut s, out, vec![]).unwrap();
        assert!(r.events.borrow().is_empty());

        let gpa = 8 * 0x1000u64;
        let (cap, sgs) = userptr_buffer(QueueType::VideoCaptureMplane, 0, gpa, size - 100);
        r.device.qbuf(&mut s, cap, sgs).unwrap();

        let events = dequeued(&r.events.borrow());
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].queue(), QueueType::VideoOutputMplane);
        assert_eq!(*events[0].get_first_plane().bytesused, size);
        assert_eq!(events[1].queue(), QueueType::VideoCaptureMplane);
        assert_eq!(*events[1].get_first_plane().bytesused, size - 100);
        assert_eq!(*r.guest.live_mappings.borrow(), 0);
        {
            let mem = r.guest.memory.borrow();
            let dst = &mem[gpa as usize..gpa as usize + (size - 100) as usize];
            assert!(dst.iter().all(|&b| b == 0xab));
            // Not one byte past the CAPTURE buffer.
            assert_eq!(mem[gpa as usize + (size - 100) as usize], 0);
        }

        // REQBUFS(0) frees the host buffer, but the guest still maps it: it is held, not
        // released, until the guest's MUNMAP.
        r.device.streamoff(&mut s, QueueType::VideoOutputMplane).unwrap();
        r.device
            .reqbufs(&mut s, QueueType::VideoOutputMplane, MemoryType::Mmap, 0)
            .unwrap();
        assert!(s.output.buffers.is_empty());
        assert_eq!(*r.released.borrow(), 0);
        munmap(&mut r.device, guest_addr).unwrap();
        assert_eq!(*r.released.borrow(), 1);

        close(&mut r.device, s);
    }

    #[test]
    fn create_bufs_appends_and_rejects_mixed_memory() {
        let mut r = rig();
        let mut s = session(&mut r.device);
        let fmt = format(QueueType::VideoCaptureMplane, RGB3, 64, 64);
        r.device
            .reqbufs(&mut s, QueueType::VideoCaptureMplane, MemoryType::Mmap, 2)
            .unwrap();
        let reply = r
            .device
            .create_bufs(&mut s, 3, QueueType::VideoCaptureMplane, MemoryType::Mmap, fmt)
            .unwrap();
        assert_eq!((reply.index, reply.count), (2, 3));
        assert_eq!(s.capture.buffers.len(), 5);
        // The requested format is adjusted first (CAPTURE takes OUTPUT's 640x480, so RGB3 at
        // 64x64 becomes RGB3 at 640x480), and the new buffers are at least as large as both that
        // and the queue's current format (NV12 640x480) need.
        let adjusted = sizeimage(&reply.format);
        assert_eq!(adjusted, 640 * 480 * 3);
        let Backing::Host { buffer, .. } = &s.capture.buffers[4].backing else {
            panic!()
        };
        assert_eq!(
            buffer.len,
            adjusted.max(s.capture.format.sizeimage()) as u64
        );
        assert!(r.device.querybuf(&s, QueueType::VideoCaptureMplane, 4).is_ok());
        assert_eq!(
            r.device.querybuf(&s, QueueType::VideoCaptureMplane, 5).err(),
            Some(libc::EINVAL)
        );

        assert_eq!(
            r.device
                .create_bufs(&mut s, 1, QueueType::VideoCaptureMplane, MemoryType::UserPtr, fmt)
                .err(),
            Some(libc::EINVAL)
        );
        // S_FMT with buffers allocated is refused.
        assert_eq!(
            r.device
                .s_fmt(&mut s, QueueType::VideoCaptureMplane, fmt)
                .err(),
            Some(libc::EBUSY)
        );

        // A second session cannot take buffers while this one holds them.
        let mut other = <Device as VirtioMediaDevice<&[u8], Vec<u8>>>::new_session(&mut r.device, 1)
            .unwrap();
        assert_eq!(
            r.device
                .reqbufs(&mut other, QueueType::VideoOutputMplane, MemoryType::Mmap, 1)
                .err(),
            Some(libc::EBUSY)
        );

        close(&mut r.device, s);
        assert_eq!(*r.released.borrow(), 5);
        r.device
            .reqbufs(&mut other, QueueType::VideoOutputMplane, MemoryType::Mmap, 1)
            .unwrap();
        close(&mut r.device, other);
    }

    /// A guest can send an MPLANE `v4l2_buffer` that carries no plane at all: v4l2r only refuses
    /// `length >= VIDEO_MAX_PLANES`, so `QBUF` sees a buffer whose `planes_iter()` is empty.
    /// Reading its first plane used to panic, and this VMM aborts on panic -- the whole VM died
    /// with it. The device must answer `EINVAL` and keep working.
    #[test]
    fn qbuf_of_a_planeless_mplane_buffer_is_refused() {
        let mut r = rig();
        let mut s = session(&mut r.device);
        let size = s.output.format.sizeimage();

        r.device
            .reqbufs(&mut s, QueueType::VideoOutputMplane, MemoryType::UserPtr, 2)
            .unwrap();
        r.device
            .reqbufs(&mut s, QueueType::VideoCaptureMplane, MemoryType::Mmap, 1)
            .unwrap();
        r.device.streamon(&mut s, QueueType::VideoOutputMplane).unwrap();
        r.device.streamon(&mut s, QueueType::VideoCaptureMplane).unwrap();

        // `length == 0` means "no plane" for a multi-planar buffer, and that is what the guest
        // driver forwards unchanged.
        let mut raw = UncheckedV4l2Buffer::new_for_querybuf(QueueType::VideoOutputMplane, Some(0));
        raw.0.memory = MemoryType::UserPtr as u32;
        raw.0.length = 0;
        let planeless = V4l2Buffer::try_from(raw).expect("v4l2r accepts a zero-plane buffer");
        assert_eq!(planeless.planes_iter().count(), 0);

        assert_eq!(
            r.device.qbuf(&mut s, planeless, vec![]).err(),
            Some(libc::EINVAL)
        );
        assert!(r.events.borrow().is_empty(), "nothing was dequeued");
        assert_eq!(*r.guest.live_mappings.borrow(), 0);

        // The device is still usable afterwards: a well-formed pair goes through.
        let gpa = 4 * 0x1000u64;
        let (out, sgs) = userptr_buffer(QueueType::VideoOutputMplane, 0, gpa, size);
        let cap = mmap_buffer(QueueType::VideoCaptureMplane, 0, size);
        r.device.qbuf(&mut s, cap, vec![]).unwrap();
        r.device.qbuf(&mut s, out, sgs).unwrap();
        assert_eq!(dequeued(&r.events.borrow()).len(), 2);

        close(&mut r.device, s);
    }

    /// The `length` of a `USERPTR` plane is the guest's own number and bounds both the mapping
    /// and the synchronous copy `process()` does; a buffer larger than what the queue was sized
    /// for is refused rather than mapped.
    #[test]
    fn userptr_length_is_bounded_by_the_queue_format() {
        let mut r = rig();
        let mut s = session(&mut r.device);
        let fmt = format(QueueType::VideoOutputMplane, RGB3, 64, 64);
        r.device.s_fmt(&mut s, QueueType::VideoOutputMplane, fmt).unwrap();
        let size = sizeimage(&fmt);

        r.device
            .reqbufs(&mut s, QueueType::VideoOutputMplane, MemoryType::UserPtr, 1)
            .unwrap();

        let gpa = 4 * 0x1000u64;
        let (out, sgs) = userptr_buffer(QueueType::VideoOutputMplane, 0, gpa, size + 1);
        assert_eq!(r.device.qbuf(&mut s, out, sgs).err(), Some(libc::EINVAL));
        assert_eq!(*r.guest.live_mappings.borrow(), 0, "nothing was mapped");

        // Zero-length is refused too, and exactly `sizeimage` is accepted.
        let (out, sgs) = userptr_buffer(QueueType::VideoOutputMplane, 0, gpa, 0);
        assert_eq!(r.device.qbuf(&mut s, out, sgs).err(), Some(libc::EINVAL));
        let (mut out, sgs) = userptr_buffer(QueueType::VideoOutputMplane, 0, gpa, size);
        // `bytesused` beyond the plane's own length is clamped to it, never trusted.
        *out.get_first_plane_mut().bytesused = size;
        let reply = r.device.qbuf(&mut s, out, sgs).unwrap();
        assert_eq!(*reply.get_first_plane().bytesused, size);
        assert_eq!(*r.guest.live_mappings.borrow(), 1);

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
}
