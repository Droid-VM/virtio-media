// Copyright 2024 The ChromiumOS Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

//! This crate contains host-side helpers to write virtio-media devices and full devices
//! implementations.
//!
//! Both helpers and devices are VMM-independent and rely on a handful of traits being implemented
//! to operate on a given VMM. This means that implementing a specific device, and adding support
//! for all virtio-media devices on a given VMM, are two completely orthogonal tasks. Adding
//! support for a VMM makes all the devices relying on this crate available. Conversely, writing a
//! new device using this crate makes it available to all supported VMMs.
//!
//! # Traits to implement by the VMM
//!
//! * Descriptor chains must implement `Read` and `Write` on their device-readable and
//!   device-writable parts, respectively. This allows devices to read commands and writes
//!   responses.
//! * The event queue must implement the `VirtioMediaEventQueue` trait to allow devices to send
//!   events to the guest.
//! * The guest memory must be made accessible through an implementation of
//!   `VirtioMediaGuestMemoryMapper`.
//! * Host-owned (`MMAP`) buffers come from an implementation of `VirtioMediaBufferAllocator`,
//!   and are made visible to the guest through an implementation of
//!   `VirtioMediaHostMemoryMapper`. The crate ships `MemFdAllocator`, one sealed memfd per
//!   buffer, as the default allocator; a VMM that serves buffers out of a pre-shared pool
//!   provides its own and hands out `HostBuffer`s with `pool_offset` set.
//!
//! These traits allow any device that implements `VirtioMediaDevice` to run on any VMM that
//! implements them.
//!
//! # Anatomy of a device
//!
//! Devices implement `VirtioMediaDevice` to provide ways to create and close sessions, and to make
//! MMAP buffers visible to the guest (if supported). They also typically implement
//! `VirtioMediaIoctlHandler` and make use of `virtio_media_dispatch_ioctl` to handle ioctls
//! simply.
//!
//! The VMM then uses `VirtioMediaDeviceRunner` in order to ask it to process a command whenever
//! one arrives on the command queue.
//!
//! By following this pattern, devices never need to care about deserializing and validating the
//! virtio-media protocol. Instead, their relevant methods are invoked when needed, on validated
//! input, while protocol errors are handled upstream in a way that is consistent for all devices.
//!
//! The devices currently in this crate are:
//!
//! * A device that proxies any host V4L2 device into the guest, in the `crate::v4l2_device_proxy`
//!   module.
//! * A pattern-generating capture device (`simple_device`) and a memory-to-memory loopback device
//!   (`loopback_device`) for exercising a guest without hardware.

pub mod devices;
pub mod io;
pub mod ioctl;
pub mod memfd;
pub mod mmap;
pub mod poll;
pub mod protocol;

use io::ReadFromDescriptorChain;
use io::WriteToDescriptorChain;
pub use memfd::MemFdAllocator;
use poll::SessionPoller;
pub use v4l2r;

use std::collections::HashMap;
use std::io::Result as IoResult;
use std::num::NonZeroUsize;
use std::os::fd::AsFd;
use std::os::fd::BorrowedFd;
use std::os::fd::OwnedFd;
use std::ptr::NonNull;

use anyhow::Context;
use log::error;
use nix::sys::mman;

use protocol::*;

/// Trait for sending V4L2 events to the driver.
pub trait VirtioMediaEventQueue {
    /// Wait until an event descriptor becomes available and send `event` to the guest.
    fn send_event(&mut self, event: V4l2Event);

    /// Wait until an event descriptor becomes available and send `errno` as an error event to the
    /// guest.
    fn send_error(&mut self, session_id: u32, errno: i32) {
        self.send_event(V4l2Event::Error(ErrorEvent::new(session_id, errno)));
    }
}

/// Trait for representing a range of guest memory that has been mapped linearly into the host's
/// address space.
pub trait GuestMemoryRange {
    fn as_ptr(&self) -> *const u8;
    fn as_mut_ptr(&mut self) -> *mut u8;
}

/// Trait enabling guest memory linear access for the device.
///
/// Although the host can access the guest memory, it sometimes need to have a linear view of
/// sparse areas. This trait provides a way to perform such mappings.
///
/// Note to devices: [`VirtioMediaGuestMemoryMapper::GuestMemoryMapping`] instances must be held
/// for as long as the device might access the memory to avoid race conditions, as some
/// implementations might e.g. write back into the guest memory at destruction time.
pub trait VirtioMediaGuestMemoryMapper {
    /// Host-side linear mapping of sparse guest memory.
    type GuestMemoryMapping: GuestMemoryRange;

    /// Maps `sgs`, which contains a list of guest-physical SG entries into a linear mapping on the
    /// host.
    ///
    /// Implementations that want the guest to see a specific error code (e.g. `EFAULT` for
    /// memory the host is not allowed to touch) wrap a [`GuestMappingError`] in the returned
    /// error; devices recover it with [`guest_mapping_errno`]. Any other error is reported as
    /// `EINVAL`.
    fn new_mapping(&self, sgs: Vec<SgEntry>) -> anyhow::Result<Self::GuestMemoryMapping>;
}

/// Error a [`VirtioMediaGuestMemoryMapper`] can return to name the errno the guest should get.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GuestMappingError(pub i32);

impl std::fmt::Display for GuestMappingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "guest memory mapping failed with errno {}", self.0)
    }
}

impl std::error::Error for GuestMappingError {}

/// The errno a failed [`VirtioMediaGuestMemoryMapper::new_mapping`] call asked for, or `EINVAL`
/// when it did not say.
pub fn guest_mapping_errno(e: &anyhow::Error) -> i32 {
    e.downcast_ref::<GuestMappingError>()
        .map(|e| e.0)
        .unwrap_or(libc::EINVAL)
}

/// Whether a [`HostBuffer`] made its own host mapping, and so must undo it, or was handed a window
/// into a mapping someone else keeps alive.
enum HostBufferMapping {
    /// `ptr` points into a mapping owned by the allocator (a pool). Nothing to undo.
    Borrowed,
    /// `ptr` is a mapping of `len` bytes this buffer created; unmapped when it is dropped.
    Owned,
}

/// A host-owned buffer that the guest can be given `MMAP` access to.
///
/// Every field a VMM needs in order to expose the buffer to the guest is here: the backing object
/// (`fd`, always a dup the buffer owns), the byte range of it the buffer occupies (`fd_offset`,
/// `len`), a host mapping of exactly that range (`ptr`), and, when the buffer is a slice of a
/// pre-shared pool the guest already maps as a whole, its offset inside that pool
/// (`pool_offset`). A `pool_offset` of `Some` tells [`VirtioMediaHostMemoryMapper::add_mapping`]
/// that no new guest mapping is needed: the offset itself is the answer.
///
/// # Ownership contract
///
/// Buffers come from a [`VirtioMediaBufferAllocator`] and **must be handed back to the same
/// allocator's `release()`** once the device is done with them. Dropping one without doing so is
/// memory-safe but leaks its backing: a pool slice stays allocated in the pool for the life of the
/// allocator, and a memfd's pages stay until the last dup of its descriptor is closed. Dropping
/// only ever tears down what the buffer itself created (its own mapping, its own descriptor); it
/// never touches memory another party -- the guest, the allocator -- may still be mapping.
pub struct HostBuffer {
    /// The backing object. A dup owned by this buffer.
    pub fd: OwnedFd,
    /// Byte offset of the buffer's first byte inside `fd`. Zero for a per-buffer memfd; the
    /// pool's own offset plus `pool_offset` for a pool slice.
    pub fd_offset: u64,
    /// Length of the buffer in bytes.
    pub len: u64,
    /// Host mapping of the `len` bytes at `fd_offset`. Valid for as long as the buffer exists.
    pub ptr: NonNull<u8>,
    /// Offset of the buffer inside the pool the guest maps as a whole, if it lives in one.
    pub pool_offset: Option<u64>,
    mapping: HostBufferMapping,
}

// SAFETY: `ptr` is a plain pointer into a shared file mapping; nothing about it is bound to the
// thread that created it, and the buffer is only ever accessed through `&self`/`&mut self`.
unsafe impl Send for HostBuffer {}

impl HostBuffer {
    /// Wraps a buffer whose host mapping is owned by someone else, typically a pool allocator that
    /// maps its whole pool once and hands out windows into it.
    ///
    /// # Safety
    ///
    /// `ptr` must be valid for reads and writes of `len` bytes and must stay valid for as long as
    /// the returned buffer exists; the caller keeps the underlying mapping alive at least that
    /// long (a pool allocator outlives every buffer it hands out, which is why `release()` takes
    /// buffers back before the allocator goes away). `(fd, fd_offset, len)` must describe the same
    /// bytes `ptr` maps.
    pub unsafe fn from_raw_parts(
        fd: OwnedFd,
        fd_offset: u64,
        len: u64,
        ptr: NonNull<u8>,
        pool_offset: Option<u64>,
    ) -> Self {
        Self {
            fd,
            fd_offset,
            len,
            ptr,
            pool_offset,
            mapping: HostBufferMapping::Borrowed,
        }
    }

    /// Creates a buffer over `len` bytes at `fd_offset` of `fd` by mapping them into the host,
    /// read-write when `rw` is set and read-only otherwise. The mapping is undone when the buffer
    /// is dropped. `pool_offset` is `None`: the guest gets its own mapping of the descriptor.
    ///
    /// `fd_offset` must be page-aligned and `len` non-zero. Errors are `libc` error codes.
    pub fn map_fd(fd: OwnedFd, fd_offset: u64, len: u64, rw: bool) -> Result<Self, i32> {
        let size = usize::try_from(len)
            .ok()
            .and_then(NonZeroUsize::new)
            .ok_or(libc::EINVAL)?;
        let offset = libc::off_t::try_from(fd_offset).map_err(|_| libc::EINVAL)?;
        let prot = if rw {
            mman::ProtFlags::PROT_READ | mman::ProtFlags::PROT_WRITE
        } else {
            mman::ProtFlags::PROT_READ
        };

        // SAFETY: `fd` is a valid descriptor we own; we ask the kernel for a fresh mapping of
        // it and only hand the pointer out through this buffer's accessors.
        let ptr = unsafe { mman::mmap(None, size, prot, mman::MapFlags::MAP_SHARED, &fd, offset) }
            .map_err(|e| e as i32)?;

        Ok(Self {
            fd,
            fd_offset,
            len,
            ptr: ptr.cast(),
            pool_offset: None,
            mapping: HostBufferMapping::Owned,
        })
    }

    /// The buffer's descriptor, borrowed.
    pub fn as_fd(&self) -> BorrowedFd<'_> {
        self.fd.as_fd()
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// The buffer's bytes as a slice.
    ///
    /// # Safety
    ///
    /// **The guest maps these pages too.** A `&[u8]`/`&mut [u8]` asserts to the compiler that
    /// nothing else touches the bytes for the life of the borrow, and that is false for every
    /// buffer the guest has `mmap`ed: the guest can write to them from another CPU at any moment,
    /// which is a data race and undefined behaviour, and it can change what a "checked" value
    /// reads as between the check and the use. This is why crosvm hands out `VolatileSlice`
    /// rather than `&[u8]` for guest-visible memory.
    ///
    /// Device code must therefore go through [`Self::as_ptr`] / [`Self::as_mut_ptr`] and raw
    /// (`ptr::copy*`, `read_volatile`, `write_volatile`) accesses instead. These accessors remain
    /// for tests and for callers that can prove no guest mapping of the buffer exists -- a buffer
    /// that has never been given a `mem_offset`, or one whose mappings the device has already
    /// taken back.
    pub unsafe fn as_slice(&self) -> &[u8] {
        // SAFETY: `ptr` is valid for `len` bytes for the life of `self` (constructor contract),
        // and the returned borrow cannot outlive `self`. The caller promises there is no
        // concurrent guest mapping.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len as usize) }
    }

    /// The buffer's bytes as a mutable slice.
    ///
    /// # Safety
    ///
    /// Same aliasing hazard as [`Self::as_slice`], which see; additionally a buffer created with
    /// `map_fd(.., rw = false)` must not be written through this.
    pub unsafe fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: as `as_slice`, and `&mut self` makes this the only borrow of the bytes on the
        // host side.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len as usize) }
    }
}

impl Drop for HostBuffer {
    fn drop(&mut self) {
        if let HostBufferMapping::Owned = self.mapping {
            // SAFETY: this mapping was created by `map_fd` with exactly this pointer and length,
            // and nothing else references it once the buffer is gone.
            if let Err(e) = unsafe { mman::munmap(self.ptr.cast(), self.len as usize) } {
                error!("error while unmapping host buffer: {:#}", e);
            }
        }
    }
}

/// Trait for allocating the host-owned buffers that back `MMAP` V4L2 buffers.
///
/// Devices call `allocate` when the guest requests buffers (`VIDIOC_REQBUFS`,
/// `VIDIOC_CREATE_BUFS`) and `release` when they are done with them, and never free a
/// [`HostBuffer`] any other way (see its ownership contract).
pub trait VirtioMediaBufferAllocator {
    /// Allocates a buffer of `len` bytes. Returns `ENOMEM` when the backing store is exhausted,
    /// or another `libc` error code.
    fn allocate(&mut self, len: u64) -> Result<HostBuffer, i32>;

    /// Takes `buf`, which came from this allocator, back.
    fn release(&mut self, buf: HostBuffer);
}

/// No-op implementation of `VirtioMediaBufferAllocator`: every allocation fails with `ENOMEM`,
/// so a device using it cannot serve `MMAP` buffers. For tests and for devices that only ever
/// use guest memory.
impl VirtioMediaBufferAllocator for () {
    fn allocate(&mut self, _len: u64) -> Result<HostBuffer, i32> {
        Err(libc::ENOMEM)
    }

    fn release(&mut self, _buf: HostBuffer) {}
}

/// Trait for mapping host buffers into the guest physical address space.
///
/// An VMM-side implementation of this trait is needed in order to map `MMAP` buffers into the
/// guest.
///
/// If the functionality is not needed, `()` can be passed in place of an implementor of this
/// trait. It will return `ENOTTY` to each `mmap` attempt, effectively disabling the ability to
/// map `MMAP` buffers into the guest.
pub trait VirtioMediaHostMemoryMapper {
    /// Makes `buffer` visible to the guest and returns the offset the guest adds to its base to
    /// reach it: for a buffer with `pool_offset` set, that offset (the guest already maps the
    /// whole pool); otherwise the offset in the guest shared memory region the VMM mapped the
    /// buffer's descriptor at.
    ///
    /// `offset` is the buffer's V4L2 `mem_offset`, useful as a stable tag. `rw` is whether the
    /// guest asked for a writable mapping. Errors are `libc` error codes.
    fn add_mapping(&mut self, buffer: &HostBuffer, offset: u64, rw: bool) -> Result<u64, i32>;

    /// Removes a guest mapping previously created at shared memory region offset `shm_offset`.
    fn remove_mapping(&mut self, shm_offset: u64) -> Result<(), i32>;
}

/// No-op implementation of `VirtioMediaHostMemoryMapper`. Can be used for testing purposes or when
/// it is not needed to map `MMAP` buffers into the guest.
impl VirtioMediaHostMemoryMapper for () {
    fn add_mapping(&mut self, _: &HostBuffer, _: u64, _: bool) -> Result<u64, i32> {
        Err(libc::ENOTTY)
    }

    fn remove_mapping(&mut self, _: u64) -> Result<(), i32> {
        Err(libc::ENOTTY)
    }
}

pub trait VirtioMediaDeviceSession {
    /// Returns the file descriptor that the client can listen to in order to know when a session
    /// event has occurred. The FD signals that it is readable when the device's `process_events`
    /// should be called.
    ///
    /// If this method returns `None`, then the session does not need to be polled by the client,
    /// and `process_events` does not need to be called either.
    fn poll_fd(&self) -> Option<BorrowedFd>;
}

/// Trait for implementing virtio-media devices.
///
/// The preferred way to use this trait is to wrap implementations in a
/// [`VirtioMediaDeviceRunner`], which takes care of reading and dispatching commands. In addition,
/// [`ioctl::VirtioMediaIoctlHandler`] should also be used to automatically parse and dispatch
/// ioctls.
pub trait VirtioMediaDevice<Reader: ReadFromDescriptorChain, Writer: WriteToDescriptorChain> {
    type Session: VirtioMediaDeviceSession;

    /// Create a new session which ID is `session_id`.
    ///
    /// The error value returned is the error code to send back to the guest.
    fn new_session(&mut self, session_id: u32) -> Result<Self::Session, i32>;
    /// Close the passed session.
    fn close_session(&mut self, session: Self::Session);

    /// Perform the IOCTL command and write the response into `writer`.
    ///
    /// The flow for performing a given `ioctl` is to read the parameters from `reader`, perform
    /// the operation, and then write the result on `writer`. Events triggered by a given ioctl can
    /// be queued on `evt_queue`.
    ///
    /// Only returns an error if the response could not be properly written ; all other errors are
    /// propagated to the guest and considered normal operation from the host's point of view.
    ///
    /// The recommended implementation of this method is to just invoke
    /// `virtio_media_dispatch_ioctl` on an implementation of `VirtioMediaIoctlHandler`, so all the
    /// details of ioctl parsing and validation are taken care of by this crate.
    fn do_ioctl(
        &mut self,
        session: &mut Self::Session,
        ioctl: V4l2Ioctl,
        reader: &mut Reader,
        writer: &mut Writer,
    ) -> IoResult<()>;

    /// Performs the MMAP command.
    ///
    /// Only returns an error if the response could not be properly written ; all other errors are
    /// propagated to the guest.
    //
    // TODO: flags should be a dedicated enum?
    fn do_mmap(
        &mut self,
        session: &mut Self::Session,
        flags: u32,
        offset: u32,
    ) -> Result<(u64, u64), i32>;
    /// Performs the MUNMAP command.
    ///
    /// Only returns an error if the response could not be properly written ; all other errors are
    /// propagated to the guest.
    fn do_munmap(&mut self, guest_addr: u64) -> Result<(), i32>;

    fn process_events(&mut self, _session: &mut Self::Session) -> Result<(), i32> {
        panic!("process_events needs to be implemented")
    }
}

/// Wrapping structure for a `VirtioMediaDevice` managing its sessions and providing methods for
/// processing its commands.
pub struct VirtioMediaDeviceRunner<Reader, Writer, Device, Poller>
where
    Reader: ReadFromDescriptorChain,
    Writer: WriteToDescriptorChain,
    Device: VirtioMediaDevice<Reader, Writer>,
    Poller: SessionPoller,
{
    pub device: Device,
    poller: Poller,
    pub sessions: HashMap<u32, Device::Session>,
    // TODO: recycle session ids...
    session_id_counter: u32,
}

impl<Reader, Writer, Device, Poller> VirtioMediaDeviceRunner<Reader, Writer, Device, Poller>
where
    Reader: ReadFromDescriptorChain,
    Writer: WriteToDescriptorChain,
    Device: VirtioMediaDevice<Reader, Writer>,
    Poller: SessionPoller,
{
    pub fn new(device: Device, poller: Poller) -> Self {
        Self {
            device,
            poller,
            sessions: Default::default(),
            session_id_counter: 0,
        }
    }
}

impl<Reader, Writer, Device, Poller> VirtioMediaDeviceRunner<Reader, Writer, Device, Poller>
where
    Reader: ReadFromDescriptorChain,
    Writer: WriteToDescriptorChain,
    Device: VirtioMediaDevice<Reader, Writer>,
    Poller: SessionPoller,
{
    /// Handle a single command from the virtio queue.
    ///
    /// `reader` and `writer` are the device-readable and device-writable sections of the
    /// descriptor chain containing the command. After this method has returned, the caller is
    /// responsible for returning the used descriptor chain to the guest.
    ///
    /// This method never returns an error, as doing so would halt the worker thread. All errors
    /// are propagated to the guest, with the exception of errors triggered while writing the
    /// response which are logged on the host side.
    pub fn handle_command(&mut self, reader: &mut Reader, writer: &mut Writer) {
        let hdr = match reader.read_obj::<CmdHeader>() {
            Ok(hdr) => hdr,
            Err(e) => {
                error!("error while reading command header: {:#}", e);
                let _ = writer.write_err_response(libc::EINVAL);
                return;
            }
        };

        let res = match hdr.cmd {
            VIRTIO_MEDIA_CMD_OPEN => {
                let session_id = self.session_id_counter;

                match self.device.new_session(session_id) {
                    Ok(session) => {
                        if let Some(fd) = session.poll_fd() {
                            match self.poller.add_session(fd, session_id) {
                                Ok(()) => {
                                    self.sessions.insert(session_id, session);
                                    self.session_id_counter += 1;
                                    writer.write_response(OpenResp::ok(session_id))
                                }
                                Err(e) => {
                                    log::error!(
                                        "failed to register poll FD for new session: {}",
                                        e
                                    );
                                    self.device.close_session(session);
                                    writer.write_err_response(e)
                                }
                            }
                        } else {
                            self.sessions.insert(session_id, session);
                            self.session_id_counter += 1;
                            writer.write_response(OpenResp::ok(session_id))
                        }
                    }
                    Err(e) => writer.write_err_response(e),
                }
                .context("while writing response for OPEN command")
            }
            .context("while writing response for OPEN command"),
            VIRTIO_MEDIA_CMD_CLOSE => reader
                .read_obj()
                .context("while reading CLOSE command")
                .map(|CloseCmd { session_id, .. }| {
                    if let Some(session) = self.sessions.remove(&session_id) {
                        if let Some(fd) = session.poll_fd() {
                            self.poller.remove_session(fd);
                        }
                        self.device.close_session(session);
                    }
                }),
            VIRTIO_MEDIA_CMD_IOCTL => reader
                .read_obj()
                .context("while reading IOCTL command")
                .and_then(|IoctlCmd { session_id, code }| {
                    match self.sessions.get_mut(&session_id) {
                        Some(session) => match V4l2Ioctl::n(code) {
                            Some(ioctl) => self.device.do_ioctl(session, ioctl, reader, writer),
                            None => {
                                error!("unknown ioctl code {}", code);
                                writer.write_err_response(libc::ENOTTY)
                            }
                        },
                        None => writer.write_err_response(libc::EINVAL),
                    }
                    .context("while writing response for IOCTL command")
                }),
            VIRTIO_MEDIA_CMD_MMAP => reader
                .read_obj()
                .context("while reading MMAP command")
                .and_then(
                    |MmapCmd {
                         session_id,
                         flags,
                         offset,
                     }| {
                        match self
                            .sessions
                            .get_mut(&session_id)
                            .ok_or(libc::EINVAL)
                            .and_then(|session| self.device.do_mmap(session, flags, offset))
                        {
                            Ok((guest_addr, size)) => {
                                writer.write_response(MmapResp::ok(guest_addr, size))
                            }
                            Err(e) => writer.write_err_response(e),
                        }
                        .context("while writing response for MMAP command")
                    },
                ),
            VIRTIO_MEDIA_CMD_MUNMAP => reader
                .read_obj()
                .context("while reading UNMMAP command")
                .and_then(
                    |MunmapCmd {
                         driver_addr: guest_addr,
                     }| {
                        match self.device.do_munmap(guest_addr) {
                            Ok(()) => writer.write_response(MunmapResp::ok()),
                            Err(e) => writer.write_err_response(e),
                        }
                        .context("while writing response for MUNMAP command")
                    },
                ),
            _ => writer
                .write_err_response(libc::ENOTTY)
                .context("while writing error response for invalid command"),
        };

        if let Err(e) = res {
            error!("error while processing command: {:#}", e);
            let _ = writer.write_err_response(libc::EINVAL);
        }
    }

    /// Returns the device this runner has been created from.
    pub fn into_device(self) -> Device {
        self.device
    }
}
