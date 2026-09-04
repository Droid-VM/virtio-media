// Copyright 2024 The ChromiumOS Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

//! Simple example virtio-media CAPTURE device with no dependency.
//!
//! This module illustrates how to write a device for virtio-media. It exposes a capture device
//! that generates a RGB pattern on the buffers queued by the guest.
//!
//! Buffers can be host-owned (`MMAP`, allocated from the device's
//! [`VirtioMediaBufferAllocator`]) or guest-owned (`USERPTR`, in which case the pattern is
//! written through the [`VirtioMediaGuestMemoryMapper`] into the guest's own pages). The latter
//! is what a guest driver that owns its buffers exercises (`VPU_DESIGN.md` §4.2).

use std::collections::VecDeque;
use std::io::Result as IoResult;
use std::os::fd::BorrowedFd;

use v4l2r::bindings;
use v4l2r::bindings::v4l2_event_subscription;
use v4l2r::bindings::v4l2_fmtdesc;
use v4l2r::bindings::v4l2_format;
use v4l2r::bindings::v4l2_pix_format;
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

/// Current status of a buffer.
#[derive(Debug, PartialEq, Eq)]
enum BufferState {
    /// Buffer has just been created (or streamed off) and not been used yet.
    New,
    /// Buffer has been QBUF'd by the driver but not yet processed.
    Incoming,
    /// Buffer has been processed and is ready for dequeue.
    Outgoing {
        /// Sequence of the generated frame.
        sequence: u32,
    },
}

/// Where a buffer's bytes live.
enum Backing<GM> {
    /// Host-owned, from the allocator; mappable by the guest at `offset`.
    Host { buffer: HostBuffer, offset: u32 },
    /// Guest-owned; mapped from the guest's `USERPTR` SG list while the buffer is queued.
    Guest(Option<GM>),
}

/// Information about a single buffer.
struct Buffer<GM> {
    /// Current state of the buffer.
    state: BufferState,
    /// V4L2 representation of this buffer to be sent to the guest when requested.
    v4l2_buffer: V4l2Buffer,
    /// Backing storage for the buffer.
    backing: Backing<GM>,
}

impl<GM> Buffer<GM> {
    /// Update the state of the buffer as well as its V4L2 representation.
    fn set_state(&mut self, state: BufferState) {
        let mut flags = self.v4l2_buffer.flags();
        match state {
            BufferState::New => {
                *self.v4l2_buffer.get_first_plane_mut().bytesused = 0;
                flags -= BufferFlags::QUEUED;
            }
            BufferState::Incoming => {
                *self.v4l2_buffer.get_first_plane_mut().bytesused = 0;
                flags |= BufferFlags::QUEUED;
            }
            BufferState::Outgoing { sequence } => {
                *self.v4l2_buffer.get_first_plane_mut().bytesused = BUFFER_SIZE;
                self.v4l2_buffer.set_sequence(sequence);
                self.v4l2_buffer.set_timestamp(bindings::timeval {
                    tv_sec: (sequence + 1) as bindings::time_t / 1000,
                    tv_usec: (sequence + 1) as bindings::time_t % 1000,
                });
                flags -= BufferFlags::QUEUED;
            }
        }

        self.v4l2_buffer.set_flags(flags);
        self.state = state;
    }

    /// Let go of the guest mapping, if this is a guest-owned buffer with one.
    fn drop_guest_mapping(&mut self) {
        if let Backing::Guest(mapping) = &mut self.backing {
            *mapping = None;
        }
    }
}

/// Session data of [`SimpleCaptureDevice`].
pub struct SimpleCaptureDeviceSession<GM> {
    /// Id of the session.
    id: u32,
    /// Current iteration of the pattern generation cycle.
    iteration: u64,
    /// Memory type the buffers were requested with, if any.
    memory: Option<MemoryType>,
    /// Buffers currently allocated for this session.
    buffers: Vec<Buffer<GM>>,
    /// FIFO of queued buffers awaiting processing.
    queued_buffers: VecDeque<usize>,
    /// Is the session currently streaming?
    streaming: bool,
}

impl<GM> VirtioMediaDeviceSession for SimpleCaptureDeviceSession<GM> {
    fn poll_fd(&self) -> Option<BorrowedFd> {
        None
    }
}

impl<GM: GuestMemoryRange> SimpleCaptureDeviceSession<GM> {
    /// Generate the data pattern on all queued buffers and send the corresponding
    /// [`DequeueBufferEvent`] to the driver.
    fn process_queued_buffers<Q: VirtioMediaEventQueue>(
        &mut self,
        evt_queue: &mut Q,
    ) -> IoctlResult<()> {
        while let Some(buf_id) = self.queued_buffers.pop_front() {
            let buffer = self.buffers.get_mut(buf_id).ok_or(libc::EIO)?;
            let sequence = self.iteration as u32;

            let color = [
                0xffu8 * (sequence as u8 % 2),
                0x55u8 * (sequence as u8 % 3),
                0x10u8 * (sequence as u8 % 16),
            ];
            let frame: &mut [u8] = match &mut buffer.backing {
                Backing::Host { buffer, .. } => &mut buffer.as_mut_slice()[..BUFFER_SIZE as usize],
                // SAFETY: the mapping covers at least `BUFFER_SIZE` bytes (checked at QBUF) and
                // lives until we drop it below, after writing.
                Backing::Guest(Some(mapping)) => unsafe {
                    std::slice::from_raw_parts_mut(mapping.as_mut_ptr(), BUFFER_SIZE as usize)
                },
                Backing::Guest(None) => return Err(libc::EIO),
            };
            for pixel in frame.chunks_exact_mut(3) {
                pixel.copy_from_slice(&color);
            }
            // A shadowed guest mapping is only written back when it goes away, and the guest
            // must see the frame before it is told the buffer is done.
            buffer.drop_guest_mapping();

            buffer.set_state(BufferState::Outgoing { sequence });
            // TODO: should we set the DONE flag here?
            self.iteration += 1;

            let v4l2_buffer = buffer.v4l2_buffer.clone();

            evt_queue.send_event(V4l2Event::DequeueBuffer(DequeueBufferEvent::new(
                self.id,
                v4l2_buffer,
            )));
        }

        Ok(())
    }
}

/// A simplistic video capture device, used to demonstrate how device code can be written, or for
/// testing VMMs and guests without dedicated hardware support.
///
/// This device supports a single pixel format (`RGB3`) and a single resolution, and generates
/// frames of varying uniform color. Buffers can be `MMAP` (host-owned) or `USERPTR`
/// (guest-owned).
pub struct SimpleCaptureDevice<
    Q: VirtioMediaEventQueue,
    M: VirtioMediaGuestMemoryMapper,
    HM: VirtioMediaHostMemoryMapper,
    A: VirtioMediaBufferAllocator,
> {
    /// Queue used to send events to the guest.
    evt_queue: Q,
    /// Guest memory mapper, for `USERPTR` buffers.
    mem: M,
    /// Host MMAP mapping manager.
    mmap_manager: MmapMappingManager<HM>,
    /// Where `MMAP` buffers come from.
    allocator: A,
    /// Freed `MMAP` buffers the guest still maps.
    retired: RetiredBuffers,
    /// ID of the session with allocated buffers, if any.
    ///
    /// v4l2-compliance checks that only a single session can have allocated buffers at a given
    /// time, since that's how actual hardware works - no two sessions can access a camera at the
    /// same time. It will fails if we allow simultaneous sessions to be active, so we need this
    /// artificial limitation to make it pass fully.
    active_session: Option<u32>,
}

impl<Q, M, HM, A> SimpleCaptureDevice<Q, M, HM, A>
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

    /// Drop every buffer of `session`, returning host buffers to the allocator (or holding them
    /// until the guest unmaps them) and releasing guest mappings.
    fn free_buffers(&mut self, session: &mut SimpleCaptureDeviceSession<M::GuestMemoryMapping>) {
        session.queued_buffers.clear();
        for buffer in session.buffers.drain(..) {
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

    /// Build `count` buffers of the given memory type. All or nothing: if any allocation or
    /// registration fails, everything this call took is given straight back, so a `REQBUFS` that
    /// answers an error leaves no pool space and no MMAP offset behind (a leaked pool slice is
    /// gone for the life of the device).
    fn add_buffers(
        &mut self,
        count: u32,
        memory: MemoryType,
    ) -> IoctlResult<Vec<Buffer<M::GuestMemoryMapping>>> {
        let mut buffers: Vec<Buffer<M::GuestMemoryMapping>> = Vec::with_capacity(count as usize);

        for i in 0..count {
            let buffer = match memory {
                MemoryType::Mmap => {
                    let host_buffer = match self.allocator.allocate(BUFFER_SIZE as u64) {
                        Ok(b) => b,
                        Err(e) => {
                            self.undo_added(buffers);
                            return Err(e);
                        }
                    };
                    let offset = match self.mmap_manager.register_buffer(None, BUFFER_SIZE) {
                        Ok(offset) => offset,
                        Err(e) => {
                            log::error!("failed to register MMAP buffer: {:#}", e);
                            self.allocator.release(host_buffer);
                            self.undo_added(buffers);
                            return Err(libc::EINVAL);
                        }
                    };

                    let mut v4l2_buffer =
                        V4l2Buffer::new(QueueType::VideoCapture, i, MemoryType::Mmap);
                    if let V4l2PlanesWithBackingMut::Mmap(mut planes) =
                        v4l2_buffer.planes_with_backing_iter_mut()
                    {
                        // SAFETY: every buffer has at least one plane.
                        let mut plane = planes.next().unwrap();
                        plane.set_mem_offset(offset);
                        *plane.length = BUFFER_SIZE;
                    } else {
                        // SAFETY: we have just set the buffer type to MMAP. Reaching this point means a bug in
                        // the code.
                        panic!()
                    }
                    v4l2_buffer.set_field(BufferField::None);
                    v4l2_buffer.set_flags(BufferFlags::TIMESTAMP_MONOTONIC);

                    Buffer {
                        state: BufferState::New,
                        v4l2_buffer,
                        backing: Backing::Host {
                            buffer: host_buffer,
                            offset,
                        },
                    }
                }
                MemoryType::UserPtr => {
                    // The guest brings the memory at QBUF time; until then the buffer is only
                    // a slot.
                    let mut v4l2_buffer =
                        V4l2Buffer::new(QueueType::VideoCapture, i, MemoryType::UserPtr);
                    *v4l2_buffer.get_first_plane_mut().length = BUFFER_SIZE;
                    v4l2_buffer.set_field(BufferField::None);
                    v4l2_buffer.set_flags(BufferFlags::TIMESTAMP_MONOTONIC);

                    Buffer {
                        state: BufferState::New,
                        v4l2_buffer,
                        backing: Backing::Guest(None),
                    }
                }
                _ => return Err(libc::EINVAL),
            };
            buffers.push(buffer);
        }

        Ok(buffers)
    }

    /// Give back what `add_buffers` took before it failed. These buffers were never handed to
    /// the guest, so nothing can be mapping them and `RetiredBuffers` is not involved.
    fn undo_added(&mut self, buffers: Vec<Buffer<M::GuestMemoryMapping>>) {
        for buffer in buffers {
            if let Backing::Host { buffer, offset } = buffer.backing {
                self.mmap_manager.unregister_buffer(offset);
                self.allocator.release(buffer);
            }
        }
    }
}

impl<Q, M, HM, A, Reader, Writer> VirtioMediaDevice<Reader, Writer>
    for SimpleCaptureDevice<Q, M, HM, A>
where
    Q: VirtioMediaEventQueue,
    M: VirtioMediaGuestMemoryMapper,
    HM: VirtioMediaHostMemoryMapper,
    A: VirtioMediaBufferAllocator,
    Reader: ReadFromDescriptorChain,
    Writer: WriteToDescriptorChain,
{
    type Session = SimpleCaptureDeviceSession<M::GuestMemoryMapping>;

    fn new_session(&mut self, session_id: u32) -> Result<Self::Session, i32> {
        Ok(SimpleCaptureDeviceSession {
            id: session_id,
            iteration: 0,
            memory: None,
            buffers: Default::default(),
            queued_buffers: Default::default(),
            streaming: false,
        })
    }

    fn close_session(&mut self, mut session: Self::Session) {
        if self.active_session == Some(session.id) {
            self.active_session = None;
        }

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
        let (guest_addr, size) = self
            .mmap_manager
            .create_mapping(offset, host_buffer, rw)
            .map_err(|_| libc::EINVAL)?;

        // TODO: would be nice to enable this, but how do we find the buffer again during munmap?
        //
        // Maybe keep a guest_addr -> session map in the device...
        // buffer.v4l2_buffer.set_flags(buffer.v4l2_buffer.flags() | BufferFlags::MAPPED);

        Ok((guest_addr, size))
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

    /// Nothing is asynchronous in this device: frames are produced inside `QBUF`/`STREAMON`.
    fn process_events(&mut self, _session: &mut Self::Session) -> Result<(), i32> {
        Ok(())
    }
}

const PIXELFORMAT: u32 = PixelFormat::from_fourcc(b"RGB3").to_u32();
const WIDTH: u32 = 640;
const HEIGHT: u32 = 480;
const BYTES_PER_LINE: u32 = WIDTH * 3;
const BUFFER_SIZE: u32 = BYTES_PER_LINE * HEIGHT;

const INPUTS: [bindings::v4l2_input; 1] = [bindings::v4l2_input {
    index: 0,
    name: *b"Default\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0",
    type_: bindings::V4L2_INPUT_TYPE_CAMERA,
    ..unsafe { std::mem::zeroed() }
}];

fn default_fmtdesc(queue: QueueType) -> v4l2_fmtdesc {
    v4l2_fmtdesc {
        index: 0,
        type_: queue as u32,
        pixelformat: PIXELFORMAT,
        ..Default::default()
    }
}

fn default_fmt(queue: QueueType) -> v4l2_format {
    let pix = v4l2_pix_format {
        width: WIDTH,
        height: HEIGHT,
        pixelformat: PIXELFORMAT,
        field: bindings::v4l2_field_V4L2_FIELD_NONE,
        bytesperline: BYTES_PER_LINE,
        sizeimage: BUFFER_SIZE,
        colorspace: bindings::v4l2_colorspace_V4L2_COLORSPACE_SRGB,
        ..Default::default()
    };

    v4l2_format {
        type_: queue as u32,
        fmt: bindings::v4l2_format__bindgen_ty_1 { pix },
    }
}

/// Implementations of the ioctls required by a CAPTURE device.
impl<Q, M, HM, A> VirtioMediaIoctlHandler for SimpleCaptureDevice<Q, M, HM, A>
where
    Q: VirtioMediaEventQueue,
    M: VirtioMediaGuestMemoryMapper,
    HM: VirtioMediaHostMemoryMapper,
    A: VirtioMediaBufferAllocator,
{
    type Session = SimpleCaptureDeviceSession<M::GuestMemoryMapping>;

    fn enum_fmt(
        &mut self,
        _session: &Self::Session,
        queue: QueueType,
        index: u32,
    ) -> IoctlResult<v4l2_fmtdesc> {
        if queue != QueueType::VideoCapture {
            return Err(libc::EINVAL);
        }
        if index > 0 {
            return Err(libc::EINVAL);
        }

        Ok(default_fmtdesc(queue))
    }

    fn g_fmt(&mut self, _session: &Self::Session, queue: QueueType) -> IoctlResult<v4l2_format> {
        if queue != QueueType::VideoCapture {
            return Err(libc::EINVAL);
        }

        Ok(default_fmt(queue))
    }

    fn s_fmt(
        &mut self,
        _session: &mut Self::Session,
        queue: QueueType,
        _format: v4l2_format,
    ) -> IoctlResult<v4l2_format> {
        if queue != QueueType::VideoCapture {
            return Err(libc::EINVAL);
        }

        Ok(default_fmt(queue))
    }

    fn try_fmt(
        &mut self,
        _session: &Self::Session,
        queue: QueueType,
        _format: v4l2_format,
    ) -> IoctlResult<v4l2_format> {
        if queue != QueueType::VideoCapture {
            return Err(libc::EINVAL);
        }

        Ok(default_fmt(queue))
    }

    fn reqbufs(
        &mut self,
        session: &mut Self::Session,
        queue: QueueType,
        memory: MemoryType,
        count: u32,
    ) -> IoctlResult<v4l2_requestbuffers> {
        if queue != QueueType::VideoCapture {
            return Err(libc::EINVAL);
        }
        if !matches!(memory, MemoryType::Mmap | MemoryType::UserPtr) {
            return Err(libc::EINVAL);
        }
        if session.streaming {
            return Err(libc::EBUSY);
        }
        // Buffers cannot be requested on a session if there is already another session with
        // allocated buffers.
        match self.active_session {
            Some(id) if id != session.id => return Err(libc::EBUSY),
            _ => (),
        }

        // Reqbufs(0) is an implicit streamoff.
        if count == 0 {
            self.streamoff(session, queue)?;
        }

        let count = std::cmp::min(count, 32);

        // Every mapping the guest holds on the old buffers is released (or parked until the
        // guest unmaps) before we answer, so a stale mapping never sees a new buffer. From here
        // on the session owns nothing: the state below is set only once the new buffers exist,
        // so a failed allocation leaves an empty session and a free device rather than a
        // half-built queue.
        self.free_buffers(session);
        session.memory = None;
        self.active_session = None;

        session.buffers = self.add_buffers(count, memory)?;
        if count > 0 {
            session.memory = Some(memory);
            self.active_session = Some(session.id);
        }

        Ok(v4l2_requestbuffers {
            count,
            type_: queue as u32,
            memory: memory as u32,
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
    ) -> IoctlResult<v4l2r::ioctl::V4l2Buffer> {
        if queue != QueueType::VideoCapture {
            return Err(libc::EINVAL);
        }
        let buffer = session.buffers.get(index as usize).ok_or(libc::EINVAL)?;

        Ok(buffer.v4l2_buffer.clone())
    }

    fn qbuf(
        &mut self,
        session: &mut Self::Session,
        buffer: v4l2r::ioctl::V4l2Buffer,
        guest_regions: Vec<Vec<SgEntry>>,
    ) -> IoctlResult<v4l2r::ioctl::V4l2Buffer> {
        if buffer.queue() != QueueType::VideoCapture {
            return Err(libc::EINVAL);
        }
        let host_buffer = session
            .buffers
            .get_mut(buffer.index() as usize)
            .ok_or(libc::EINVAL)?;
        // The memory type is fixed by REQBUFS.
        if Some(buffer.memory()) != session.memory {
            return Err(libc::EINVAL);
        }
        // Attempt to queue already queued buffer.
        if matches!(host_buffer.state, BufferState::Incoming) {
            return Err(libc::EINVAL);
        }

        if let Backing::Guest(slot) = &mut host_buffer.backing {
            // A guest-owned buffer: the guest's pages must hold a whole frame, and no more than
            // one. `length` is the guest's own number and decides how much guest memory is
            // mapped, so it is held to the queue format's `sizeimage` in both directions.
            // The buffer's queue is `VIDEO_CAPTURE` (checked above), which is single-planar and
            // so always has exactly one plane, but the plane is still asked for rather than
            // assumed -- `get_first_plane()` panics on a plane-less multi-planar buffer, which a
            // guest can build.
            let length = *buffer.planes_iter().next().ok_or(libc::EINVAL)?.length;
            let sgs = guest_regions.into_iter().next().ok_or(libc::EINVAL)?;
            let covered: u64 = sgs.iter().map(|sg| sg.len as u64).sum();
            if length != BUFFER_SIZE || covered < BUFFER_SIZE as u64 {
                return Err(libc::EINVAL);
            }
            let mapping = self.mem.new_mapping(sgs).map_err(|e| {
                log::error!("failed to map USERPTR buffer: {:#}", e);
                guest_mapping_errno(&e)
            })?;
            *slot = Some(mapping);
            // Keep the guest's view of the buffer (its userptr and length): it is what must be
            // echoed back in the dequeue event.
            let mut v4l2_buffer = buffer.clone();
            v4l2_buffer.set_field(BufferField::None);
            v4l2_buffer.set_flags(BufferFlags::TIMESTAMP_MONOTONIC);
            host_buffer.v4l2_buffer = v4l2_buffer;
        }

        host_buffer.set_state(BufferState::Incoming);
        session.queued_buffers.push_back(buffer.index() as usize);

        let buffer = host_buffer.v4l2_buffer.clone();

        if session.streaming {
            session.process_queued_buffers(&mut self.evt_queue)?;
        }

        Ok(buffer)
    }

    fn streamon(&mut self, session: &mut Self::Session, queue: QueueType) -> IoctlResult<()> {
        if queue != QueueType::VideoCapture || session.buffers.is_empty() {
            return Err(libc::EINVAL);
        }
        session.streaming = true;

        session.process_queued_buffers(&mut self.evt_queue)?;

        Ok(())
    }

    fn streamoff(&mut self, session: &mut Self::Session, queue: QueueType) -> IoctlResult<()> {
        if queue != QueueType::VideoCapture {
            return Err(libc::EINVAL);
        }
        session.streaming = false;
        session.queued_buffers.clear();
        for buffer in session.buffers.iter_mut() {
            // Guest mappings go before we answer: the guest gives the pages back once it
            // hears from us.
            buffer.drop_guest_mapping();
            buffer.set_state(BufferState::New);
        }

        Ok(())
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

    fn enuminput(
        &mut self,
        _session: &Self::Session,
        index: u32,
    ) -> IoctlResult<bindings::v4l2_input> {
        match INPUTS.get(index as usize) {
            Some(&input) => Ok(input),
            None => Err(libc::EINVAL),
        }
    }

    /// This device never emits `EOS` or `SOURCE_CHANGE`, but subscribing to them is harmless,
    /// and a guest that asks (v4l2-compliance, GStreamer) must not be refused for it.
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

    use super::*;
    use crate::MemFdAllocator;

    /// Discards the events the device sends; these tests only look at ioctl replies and at what
    /// the allocator saw.
    struct NullEvents;

    impl VirtioMediaEventQueue for NullEvents {
        fn send_event(&mut self, _event: V4l2Event) {}
    }

    /// No guest memory is involved in these tests; `USERPTR` is not exercised here.
    struct NoGuest;

    struct NoMapping;

    impl GuestMemoryRange for NoMapping {
        fn as_ptr(&self) -> *const u8 {
            unreachable!("no guest mapping is made in these tests")
        }

        fn as_mut_ptr(&mut self) -> *mut u8 {
            unreachable!("no guest mapping is made in these tests")
        }
    }

    impl VirtioMediaGuestMemoryMapper for NoGuest {
        type GuestMemoryMapping = NoMapping;

        fn new_mapping(&self, _sgs: Vec<SgEntry>) -> anyhow::Result<NoMapping> {
            anyhow::bail!("no guest memory in this test")
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

    /// An allocator that runs out after `fail_after` buffers, the way a full `media_host` pool
    /// does, and counts what comes back.
    struct FlakyAllocator {
        inner: MemFdAllocator,
        allocated: Rc<RefCell<usize>>,
        released: Rc<RefCell<usize>>,
        fail_after: usize,
    }

    impl VirtioMediaBufferAllocator for FlakyAllocator {
        fn allocate(&mut self, len: u64) -> Result<HostBuffer, i32> {
            if *self.allocated.borrow() >= self.fail_after {
                return Err(libc::ENOMEM);
            }
            let buffer = self.inner.allocate(len)?;
            *self.allocated.borrow_mut() += 1;
            Ok(buffer)
        }

        fn release(&mut self, buf: HostBuffer) {
            *self.released.borrow_mut() += 1;
            *self.allocated.borrow_mut() -= 1;
            self.inner.release(buf);
        }
    }

    type Device = SimpleCaptureDevice<NullEvents, NoGuest, FakeHostMapper, FlakyAllocator>;
    type Session = SimpleCaptureDeviceSession<NoMapping>;

    struct Rig {
        device: Device,
        allocated: Rc<RefCell<usize>>,
        released: Rc<RefCell<usize>>,
    }

    fn rig(fail_after: usize) -> Rig {
        let allocated = Rc::new(RefCell::new(0));
        let released = Rc::new(RefCell::new(0));
        let device = SimpleCaptureDevice::new(
            NullEvents,
            NoGuest,
            FakeHostMapper,
            FlakyAllocator {
                inner: MemFdAllocator::new(),
                allocated: Rc::clone(&allocated),
                released: Rc::clone(&released),
                fail_after,
            },
        );
        Rig {
            device,
            allocated,
            released,
        }
    }

    fn session(device: &mut Device, id: u32) -> Session {
        <Device as VirtioMediaDevice<&[u8], Vec<u8>>>::new_session(device, id).unwrap()
    }

    fn close(device: &mut Device, session: Session) {
        <Device as VirtioMediaDevice<&[u8], Vec<u8>>>::close_session(device, session)
    }

    /// Review bug 1: a `REQBUFS` whose allocator gives out on the third buffer must leave
    /// nothing behind -- no pool space, no registered MMAP offset, no session claiming to own
    /// buffers it does not have.
    #[test]
    fn reqbufs_unwinds_when_an_allocation_fails() {
        let mut r = rig(2);
        let mut s = session(&mut r.device, 0);

        assert_eq!(
            r.device
                .reqbufs(&mut s, QueueType::VideoCapture, MemoryType::Mmap, 4)
                .err(),
            Some(libc::ENOMEM)
        );

        // The two buffers the allocator did hand out went straight back.
        assert_eq!(*r.released.borrow(), 2);
        assert_eq!(*r.allocated.borrow(), 0);
        // And the session owns nothing, so another one can take the device.
        assert!(s.buffers.is_empty());
        assert_eq!(s.memory, None);
        assert_eq!(r.device.active_session, None);
        let mut other = session(&mut r.device, 1);
        assert_eq!(
            r.device
                .reqbufs(&mut other, QueueType::VideoCapture, MemoryType::UserPtr, 1)
                .map(|r| r.count),
            Ok(1)
        );
        close(&mut r.device, other);

        // Every MMAP offset the failed call registered was unregistered: the manager is empty,
        // so offsets start at 0 again.
        r.device.allocator.fail_after = 8;
        r.device
            .reqbufs(&mut s, QueueType::VideoCapture, MemoryType::Mmap, 2)
            .unwrap();
        let offsets: Vec<u32> = s
            .buffers
            .iter()
            .map(|b| match b.backing {
                Backing::Host { offset, .. } => offset,
                _ => panic!("expected host-owned buffers"),
            })
            .collect();
        assert_eq!(offsets, vec![0, 0x1000]);

        close(&mut r.device, s);
        assert_eq!(*r.allocated.borrow(), 0, "everything went back at close");
    }

    /// `REQBUFS(0)` on a session that has buffers frees them and hands the device back.
    #[test]
    fn reqbufs_zero_frees_everything() {
        let mut r = rig(8);
        let mut s = session(&mut r.device, 0);

        r.device
            .reqbufs(&mut s, QueueType::VideoCapture, MemoryType::Mmap, 3)
            .unwrap();
        assert_eq!(*r.allocated.borrow(), 3);
        assert_eq!(r.device.active_session, Some(0));

        r.device
            .reqbufs(&mut s, QueueType::VideoCapture, MemoryType::Mmap, 0)
            .unwrap();
        assert_eq!(*r.allocated.borrow(), 0);
        assert_eq!(s.memory, None);
        assert_eq!(r.device.active_session, None);

        close(&mut r.device, s);
    }
}
