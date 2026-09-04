// Copyright 2024 The ChromiumOS Authors

use std::collections::HashMap;

use thiserror::Error;

use crate::HostBuffer;
use crate::VirtioMediaBufferAllocator;
use crate::VirtioMediaHostMemoryMapper;

#[derive(Debug, PartialEq, Eq)]
pub struct MmapBufferMapping {
    /// Number of times `mmap` has been performed for this buffer.
    num_mappings: usize,
    /// Guest address at which the buffer is currently mapped.
    guest_addr: u64,
    /// Whether the mapping of this buffer is read-only or read-write.
    rw: bool,
}

/// Information about a MMAP buffer.
#[derive(Debug, PartialEq, Eq)]
pub struct MmapBuffer {
    /// Start offset in the MMAP range of this buffer.
    offset: u32,
    /// Size of the buffer.
    size: u32,
    /// Whether this buffer is still registered, i.e. hasn't been deleted by the driver.
    /// Unregistered buffers can still have active mappings, and are kept alive until their mapping
    /// count reaches zero. However such buffers cannot be mapped anymore and take no space in the
    /// MMAP range.
    registered: bool,
    /// Mapping information about this buffer, if the buffer is currently mapped into the guest.
    mapping: Option<MmapBufferMapping>,
}

impl MmapBuffer {
    /// Returns a new instance of `MmapBuffer` with the given parameters, zero mappings and
    /// a registered status.
    fn new(offset: u32, size: u32) -> Self {
        Self {
            offset,
            size,
            registered: true,
            mapping: None,
        }
    }
}

/// Range manager for MMAP buffers, using a host memory mapper.
///
/// Devices that allocate MMAP buffers can register a buffer using [`Self::register_buffer`] and
/// unregister them with [`Self::unregister_buffer`]. Registered buffers can then be mapped into the
/// guest address space by calling [`Self::create_mapping`] on their offset. This will return the address
/// of their guest mapping, which can then be accessed or used to unmap them with [`Self::remove_mapping`].
pub struct MmapMappingManager<M: VirtioMediaHostMemoryMapper> {
    /// Sorted MMAP space of the device. Each registered MMAP buffer takes the `[offset, size - 1]`
    /// range in this space, which is sorted in order to be binary-searchable.
    ///
    /// Buffers that are unregistered but still mapped are still kept here, but do not take space
    /// in the MMAP range (i.e. they are skipped during the binary search).
    buffers: Vec<MmapBuffer>,
    /// Guest address of every live mapping, back to the offset of the buffer it maps, so that
    /// `MUNMAP` (which only carries the guest address) finds its buffer without walking
    /// `buffers`. Kept in step with `MmapBuffer::mapping`.
    mapped: HashMap<u64, u32>,
    /// Memory mapper used to create buffer mappings.
    mapper: M,
}

impl<M: VirtioMediaHostMemoryMapper> From<M> for MmapMappingManager<M> {
    fn from(mapper: M) -> Self {
        Self {
            buffers: Vec::new(),
            mapped: HashMap::new(),
            mapper,
        }
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum RegisterBufferError {
    #[error("insufficient free space in the MMAP range")]
    NoFreeSpace,
    #[error("requested offset is already occupied")]
    OffsetOccupied,
    #[error("buffers of size 0 cannot be registered")]
    EmptyBuffer,
    #[error("buffer offset must be a multiple of the memory page size")]
    UnalignedOffset,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum CreateMappingError {
    #[error("no buffer registered at the requested offset")]
    InvalidOffset,
    #[error("cannot create new mappings for unregistered buffers")]
    UnregisteredBuffer,
    #[error("requested mapping range goes outside the buffer")]
    SizeOutOfBounds,
    #[error("error while cloning the FD for the buffer")]
    FdCloneFailure(std::io::ErrorKind),
    #[error("error while mapping the buffer: {0}")]
    MappingFailure(i32),
    #[error("mapping requested with different permission from the old one")]
    NonMatchingPermissions,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum RemoveMappingError {
    #[error("no buffer registered at the requested offset")]
    InvalidOffset,
}

const PAGE_SIZE: u32 = 0x1000;
const PAGE_MASK: u32 = !(PAGE_SIZE - 1);

impl<M: VirtioMediaHostMemoryMapper> MmapMappingManager<M> {
    /// Registers a new buffer at `offset`. If `offset` if `None`, then an offset is allocated and
    /// returned.
    ///
    /// This method fails if the range is full, or if `offset` is `Some` and the requested offset
    /// if already used by some other buffer. If `offset` is `Some` and the function succeed, then
    /// the returned value is guaranteed to be the passed offset.
    ///
    /// Note that the ranges automatically allocated are of fixed size: only the offset of a
    /// buffer is relevant when mapping it, not its size. Real V4L2 drivers also use this trick of
    /// allocating ranges such that buffers appear to overlap. This is useful as the address space
    /// is technically 32-bit, and we might need to use buffers which added size would not fit.
    ///
    /// TODO: we should recycle offsets, and further type `MmapMappingManager` so that only one
    /// allocation type can be used per instance (fixed or dynamic).
    pub fn register_buffer(
        &mut self,
        offset: Option<u32>,
        size: u32,
    ) -> Result<u32, RegisterBufferError> {
        let offset = offset.unwrap_or_else(|| {
            self.buffers
                .last()
                // Align the start offset to the next page, or `register_buffer_by_offset` will
                // fail.
                .map(|b| ((b.offset + 1).next_multiple_of(PAGE_SIZE)))
                .unwrap_or(0)
        });

        self.register_buffer_by_offset(offset, size)
            .map(|()| offset)
    }

    /// Unregisters the buffer previously registered at `offset`. Returns `true` if a buffer was
    /// indeed registered as starting at `offset`, `false` otherwise.
    pub fn unregister_buffer(&mut self, offset: u32) -> bool {
        match self.buffers.binary_search_by_key(&offset, |b| b.offset) {
            Err(_) => false,
            Ok(index) => {
                let buffer = &mut self.buffers[index];

                buffer.registered = false;
                // If there is no mapping then the buffer can be removed from the MMAP range.
                if buffer.mapping.is_none() {
                    self.buffers.remove(index);
                }

                true
            }
        }
    }

    // Register a new buffer of `size` at `offset`. Returns an error if `offset` is` already
    // occupied by another buffer.
    //
    // `size` must be greater than `0` and `offset` must be a multiple of `PAGE_SIZE`.
    fn register_buffer_by_offset(
        &mut self,
        offset: u32,
        size: u32,
    ) -> Result<(), RegisterBufferError> {
        if size == 0 {
            return Err(RegisterBufferError::EmptyBuffer);
        }
        if offset & PAGE_MASK != offset {
            return Err(RegisterBufferError::UnalignedOffset);
        }

        // Check that `offset` is actually available.
        match self.buffers.binary_search_by_key(&offset, |b| b.offset) {
            // Already have a registered buffer at that very offset.
            Ok(_) => Err(RegisterBufferError::OffsetOccupied),
            Err(index) => {
                self.buffers.insert(index, MmapBuffer::new(offset, size));
                Ok(())
            }
        }
    }

    /// Size of the buffer registered at `offset`, if there is one (registered or not yet
    /// garbage-collected).
    pub fn buffer_size(&self, offset: u32) -> Option<u32> {
        self.buffers
            .binary_search_by_key(&offset, |b| b.offset)
            .ok()
            .map(|i| self.buffers[i].size)
    }

    /// Create a new mapping for the buffer registered at `offset`, backed by `buffer`. `rw`
    /// indicates whether the mapping is read-only or read-write. Returns the guest address at
    /// which the buffer is mapped, and the size of the mapping, which should be equal to the size
    /// of the buffer.
    ///
    /// This method can be called several times and will reuse the prior mapping if it exists. The
    /// mapping will also persist until an identical number of calls to [`Self::remove_mapping`]
    /// are performed.
    ///
    /// Note however that requiring the same active mapping with different `rw` permissions will
    /// result in a `EPERM` error.
    pub fn create_mapping(
        &mut self,
        offset: u32,
        buffer: &HostBuffer,
        rw: bool,
    ) -> Result<(u64, u64), CreateMappingError> {
        let entry = self
            .buffers
            .binary_search_by_key(&offset, |b| b.offset)
            .map(|i| &mut self.buffers[i])
            .map_err(|_| CreateMappingError::InvalidOffset)?;
        let last_buffer_address = entry
            .offset
            .checked_add(entry.size - 1)
            .ok_or(CreateMappingError::InvalidOffset)?;

        // Cannot create additional mappings for buffers that have been destroyed on the guest side.
        if !entry.registered {
            return Err(CreateMappingError::UnregisteredBuffer);
        }

        // Check that we are not requiring more mapping than the buffer can cover.
        if last_buffer_address > entry.offset + (entry.size - 1) {
            return Err(CreateMappingError::SizeOutOfBounds);
        }
        // The host buffer must cover the range the guest was told the buffer has.
        if buffer.len < entry.size as u64 {
            return Err(CreateMappingError::SizeOutOfBounds);
        }

        let guest_addr = match &mut entry.mapping {
            None => {
                let guest_addr = self
                    .mapper
                    .add_mapping(buffer, entry.offset as u64, rw)
                    .map_err(CreateMappingError::MappingFailure)?;

                entry.mapping = Some(MmapBufferMapping {
                    num_mappings: 1,
                    rw,
                    guest_addr,
                });
                self.mapped.insert(guest_addr, entry.offset);

                guest_addr
            }
            Some(mapping) => {
                if mapping.rw != rw {
                    return Err(CreateMappingError::NonMatchingPermissions);
                }
                mapping.num_mappings += 1;
                mapping.guest_addr
            }
        };

        Ok((guest_addr, entry.size as u64))
    }

    /// Returns `true` if the buffer still has other mappings, `false` if this was the last mapping.
    pub fn remove_mapping(&mut self, guest_addr: u64) -> Result<bool, RemoveMappingError> {
        let offset = *self
            .mapped
            .get(&guest_addr)
            .ok_or(RemoveMappingError::InvalidOffset)?;
        // Unregistered-but-mapped buffers stay in the sorted vector until their last mapping
        // goes, so a mapped offset is always found here.
        let i = self
            .buffers
            .binary_search_by_key(&offset, |b| b.offset)
            .map_err(|_| RemoveMappingError::InvalidOffset)?;
        let buffer = &mut self.buffers[i];
        let mapping = buffer
            .mapping
            .as_mut()
            .ok_or(RemoveMappingError::InvalidOffset)?;

        mapping.num_mappings -= 1;
        if mapping.num_mappings > 0 {
            return Ok(true);
        }

        if let Err(e) = self.mapper.remove_mapping(guest_addr) {
            log::error!("error while unmapping MMAP buffer: {:#}", e);
        }
        buffer.mapping = None;
        self.mapped.remove(&guest_addr);
        // If this was the last dangling mapping then the buffer can be removed from the MMAP
        // range.
        if !buffer.registered {
            self.buffers.remove(i);
        }
        Ok(false)
    }
    /// Returns `true` if the buffer registered at `offset` is already mapped.
    pub fn is_mapped(&self, offset: u64) -> bool {
        let Ok(offset) = u32::try_from(offset) else {
            return false;
        };

        match self.buffers.binary_search_by_key(&offset, |b| b.offset) {
            Err(_) => false,
            Ok(index) => self.buffers[index].mapping.is_some(),
        }
    }

    /// The mapper this manager was constructed from.
    pub fn mapper_mut(&mut self) -> &mut M {
        &mut self.mapper
    }

    /// Consume the mapping manager and return the mapper it has been constructed from.
    pub fn into_mapper(self) -> M {
        self.mapper
    }
}

/// Host buffers a device is done with but must not hand back to their allocator yet.
///
/// A guest may still have a buffer mapped when the device frees it (`REQBUFS(0)` with a mapping
/// outstanding, a session closed with `mmap`s alive). The allocator is free to hand the same
/// backing out again the moment it gets the buffer back, and for a pool-backed buffer that means
/// the guest's stale mapping would look straight into someone else's fresh frame. So the release
/// waits until [`MmapMappingManager`] reports the buffer's last guest mapping gone -- the same
/// point at which the manager itself forgets the offset (`VPU_DESIGN.md` §2.5).
///
/// Usage: `retire` instead of `allocator.release` when freeing, and `reap` after every
/// `remove_mapping`.
#[derive(Default)]
pub struct RetiredBuffers {
    pending: Vec<(u32, HostBuffer)>,
}

impl RetiredBuffers {
    pub fn new() -> Self {
        Self::default()
    }

    /// Unregisters the buffer at `offset` and releases `buffer` to `allocator` now if the guest
    /// has no mapping of it, or later, from `reap`, once it has none.
    pub fn retire<M, A>(
        &mut self,
        manager: &mut MmapMappingManager<M>,
        allocator: &mut A,
        offset: u32,
        buffer: HostBuffer,
    ) where
        M: VirtioMediaHostMemoryMapper,
        A: VirtioMediaBufferAllocator,
    {
        manager.unregister_buffer(offset);
        if manager.is_mapped(offset as u64) {
            self.pending.push((offset, buffer));
        } else {
            allocator.release(buffer);
        }
    }

    /// Releases every retired buffer whose guest mappings are all gone.
    pub fn reap<M, A>(&mut self, manager: &MmapMappingManager<M>, allocator: &mut A)
    where
        M: VirtioMediaHostMemoryMapper,
        A: VirtioMediaBufferAllocator,
    {
        let mut i = 0;
        while i < self.pending.len() {
            if manager.is_mapped(self.pending[i].0 as u64) {
                i += 1;
            } else {
                let (_, buffer) = self.pending.swap_remove(i);
                allocator.release(buffer);
            }
        }
    }

    /// Number of buffers waiting for the guest to unmap them.
    pub fn len(&self) -> usize {
        self.pending.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use crate::HostBuffer;
    use crate::MemFdAllocator;
    use crate::VirtioMediaBufferAllocator;
    use crate::VirtioMediaHostMemoryMapper;

    use super::CreateMappingError;
    use super::MmapBuffer;
    use super::MmapBufferMapping;
    use super::MmapMappingManager;
    use super::RegisterBufferError;
    use super::RemoveMappingError;
    use super::RetiredBuffers;

    struct DummyHostMemoryMapper;

    impl VirtioMediaHostMemoryMapper for DummyHostMemoryMapper {
        fn add_mapping(&mut self, _buffer: &HostBuffer, offset: u64, _rw: bool) -> Result<u64, i32> {
            Ok(offset | 0x8000_0000)
        }

        fn remove_mapping(&mut self, _guest_addr: u64) -> Result<(), i32> {
            Ok(())
        }
    }

    /// A mapper for pool-style buffers: the answer is the buffer's own pool offset, as crosvm's
    /// pool backing does it.
    struct PoolStyleMapper;

    impl VirtioMediaHostMemoryMapper for PoolStyleMapper {
        fn add_mapping(&mut self, buffer: &HostBuffer, _offset: u64, _rw: bool) -> Result<u64, i32> {
            buffer.pool_offset.ok_or(libc::EINVAL)
        }

        fn remove_mapping(&mut self, _guest_addr: u64) -> Result<(), i32> {
            Ok(())
        }
    }

    fn host_buffer(size: u64) -> HostBuffer {
        MemFdAllocator::new().allocate(size).unwrap()
    }

    /// An allocator that only counts what comes back, for the retirement tests.
    #[derive(Default)]
    struct CountingAllocator {
        released: usize,
    }

    impl VirtioMediaBufferAllocator for CountingAllocator {
        fn allocate(&mut self, len: u64) -> Result<HostBuffer, i32> {
            MemFdAllocator::new().allocate(len)
        }

        fn release(&mut self, _buf: HostBuffer) {
            self.released += 1;
        }
    }

    #[test]
    fn mmap_manager_register_by_offset() {
        let mut mm = MmapMappingManager::from(DummyHostMemoryMapper);
        assert_eq!(mm.buffers, vec![]);

        assert_eq!(mm.register_buffer_by_offset(0x0, 0x1000), Ok(()));
        assert_eq!(mm.buffers, vec![MmapBuffer::new(0x0, 0x1000)]);

        assert_eq!(mm.register_buffer_by_offset(0x1000, 0x5000), Ok(()));
        assert_eq!(
            mm.buffers,
            vec![
                MmapBuffer::new(0x0, 0x1000),
                MmapBuffer::new(0x1000, 0x5000),
            ]
        );

        assert_eq!(mm.register_buffer_by_offset(0xa000, 0x1000), Ok(()));
        assert_eq!(
            mm.buffers,
            vec![
                MmapBuffer::new(0x0, 0x1000),
                MmapBuffer::new(0x1000, 0x5000),
                MmapBuffer::new(0xa000, 0x1000),
            ]
        );

        assert_eq!(mm.register_buffer_by_offset(0x6000, 0x2000), Ok(()));
        assert_eq!(
            mm.buffers,
            vec![
                MmapBuffer::new(0x0, 0x1000),
                MmapBuffer::new(0x1000, 0x5000),
                MmapBuffer::new(0x6000, 0x2000),
                MmapBuffer::new(0xa000, 0x1000),
            ]
        );

        assert_eq!(
            mm.register_buffer_by_offset(0x8000, 0x0),
            Err(RegisterBufferError::EmptyBuffer)
        );

        assert_eq!(
            mm.register_buffer_by_offset(0x8100, 0x1000),
            Err(RegisterBufferError::UnalignedOffset)
        );

        assert_eq!(
            mm.register_buffer_by_offset(0x0, 0x1000),
            Err(RegisterBufferError::OffsetOccupied)
        );

        assert_eq!(
            mm.register_buffer_by_offset(0x1000, 0x1000),
            Err(RegisterBufferError::OffsetOccupied)
        );

        assert_eq!(mm.register_buffer_by_offset(0x2000, 0x1000), Ok(()));
        assert_eq!(
            mm.buffers,
            vec![
                MmapBuffer::new(0x0, 0x1000),
                MmapBuffer::new(0x1000, 0x5000),
                MmapBuffer::new(0x2000, 0x1000),
                MmapBuffer::new(0x6000, 0x2000),
                MmapBuffer::new(0xa000, 0x1000),
            ]
        );

        assert_eq!(mm.register_buffer_by_offset(0x7000, 0x2000), Ok(()));
        assert_eq!(
            mm.buffers,
            vec![
                MmapBuffer::new(0x0, 0x1000),
                MmapBuffer::new(0x1000, 0x5000),
                MmapBuffer::new(0x2000, 0x1000),
                MmapBuffer::new(0x6000, 0x2000),
                MmapBuffer::new(0x7000, 0x2000),
                MmapBuffer::new(0xa000, 0x1000),
            ]
        );

        assert_eq!(mm.register_buffer_by_offset(0x8000, 0x2000), Ok(()));
        assert_eq!(
            mm.buffers,
            vec![
                MmapBuffer::new(0x0, 0x1000),
                MmapBuffer::new(0x1000, 0x5000),
                MmapBuffer::new(0x2000, 0x1000),
                MmapBuffer::new(0x6000, 0x2000),
                MmapBuffer::new(0x7000, 0x2000),
                MmapBuffer::new(0x8000, 0x2000),
                MmapBuffer::new(0xa000, 0x1000),
            ]
        );

        assert_eq!(mm.register_buffer_by_offset(0xffff_f000, 0x1000), Ok(()));
        assert_eq!(
            mm.buffers,
            vec![
                MmapBuffer::new(0x0, 0x1000),
                MmapBuffer::new(0x1000, 0x5000),
                MmapBuffer::new(0x2000, 0x1000),
                MmapBuffer::new(0x6000, 0x2000),
                MmapBuffer::new(0x7000, 0x2000),
                MmapBuffer::new(0x8000, 0x2000),
                MmapBuffer::new(0xa000, 0x1000),
                MmapBuffer::new(0xffff_f000, 0x1000),
            ]
        );

        assert!(mm.unregister_buffer(0xffff_f000));
        assert_eq!(
            mm.buffers,
            vec![
                MmapBuffer::new(0x0, 0x1000),
                MmapBuffer::new(0x1000, 0x5000),
                MmapBuffer::new(0x2000, 0x1000),
                MmapBuffer::new(0x6000, 0x2000),
                MmapBuffer::new(0x7000, 0x2000),
                MmapBuffer::new(0x8000, 0x2000),
                MmapBuffer::new(0xa000, 0x1000),
            ]
        );

        assert!(mm.unregister_buffer(0x6000));
        assert_eq!(
            mm.buffers,
            vec![
                MmapBuffer::new(0x0, 0x1000),
                MmapBuffer::new(0x1000, 0x5000),
                MmapBuffer::new(0x2000, 0x1000),
                MmapBuffer::new(0x7000, 0x2000),
                MmapBuffer::new(0x8000, 0x2000),
                MmapBuffer::new(0xa000, 0x1000),
            ]
        );

        assert!(!mm.unregister_buffer(0x6000));
        assert_eq!(
            mm.buffers,
            vec![
                MmapBuffer::new(0x0, 0x1000),
                MmapBuffer::new(0x1000, 0x5000),
                MmapBuffer::new(0x2000, 0x1000),
                MmapBuffer::new(0x7000, 0x2000),
                MmapBuffer::new(0x8000, 0x2000),
                MmapBuffer::new(0xa000, 0x1000),
            ]
        );

        assert!(!mm.unregister_buffer(0x8100));
        assert_eq!(
            mm.buffers,
            vec![
                MmapBuffer::new(0x0, 0x1000),
                MmapBuffer::new(0x1000, 0x5000),
                MmapBuffer::new(0x2000, 0x1000),
                MmapBuffer::new(0x7000, 0x2000),
                MmapBuffer::new(0x8000, 0x2000),
                MmapBuffer::new(0xa000, 0x1000),
            ]
        );

        assert!(mm.unregister_buffer(0x0));
        assert_eq!(
            mm.buffers,
            vec![
                MmapBuffer::new(0x1000, 0x5000),
                MmapBuffer::new(0x2000, 0x1000),
                MmapBuffer::new(0x7000, 0x2000),
                MmapBuffer::new(0x8000, 0x2000),
                MmapBuffer::new(0xa000, 0x1000),
            ]
        );

        assert!(mm.unregister_buffer(0xa000));
        assert_eq!(
            mm.buffers,
            vec![
                MmapBuffer::new(0x1000, 0x5000),
                MmapBuffer::new(0x2000, 0x1000),
                MmapBuffer::new(0x7000, 0x2000),
                MmapBuffer::new(0x8000, 0x2000),
            ]
        );

        assert!(mm.unregister_buffer(0x1000));
        assert_eq!(
            mm.buffers,
            vec![
                MmapBuffer::new(0x2000, 0x1000),
                MmapBuffer::new(0x7000, 0x2000),
                MmapBuffer::new(0x8000, 0x2000),
            ]
        );

        assert!(mm.unregister_buffer(0x8000));
        assert_eq!(
            mm.buffers,
            vec![
                MmapBuffer::new(0x2000, 0x1000),
                MmapBuffer::new(0x7000, 0x2000),
            ]
        );

        assert_eq!(
            mm.buffers,
            vec![
                MmapBuffer::new(0x2000, 0x1000),
                MmapBuffer::new(0x7000, 0x2000),
            ]
        );
    }

    #[test]
    fn mmap_manager_register() {
        let mut mm = MmapMappingManager::from(DummyHostMemoryMapper);

        assert_eq!(mm.register_buffer(None, 0x1000), Ok(0x0));
        assert_eq!(mm.buffers, vec![MmapBuffer::new(0x0, 0x1000)]);

        assert_eq!(mm.register_buffer(None, 0x5000), Ok(0x1000));
        assert_eq!(
            mm.buffers,
            vec![
                MmapBuffer::new(0x0, 0x1000),
                MmapBuffer::new(0x1000, 0x5000),
            ]
        );

        assert_eq!(mm.register_buffer(None, 0xffff_a000), Ok(0x2000));
        assert_eq!(
            mm.buffers,
            vec![
                MmapBuffer::new(0x0, 0x1000),
                MmapBuffer::new(0x1000, 0x5000),
                MmapBuffer::new(0x2000, 0xffff_a000),
            ]
        );

        assert!(mm.unregister_buffer(0x2000));
        assert_eq!(
            mm.buffers,
            vec![
                MmapBuffer::new(0x0, 0x1000),
                MmapBuffer::new(0x1000, 0x5000),
            ]
        );

        assert_eq!(mm.register_buffer(None, 0xffff_b000), Ok(0x2000));
        assert_eq!(
            mm.buffers,
            vec![
                MmapBuffer::new(0x0, 0x1000),
                MmapBuffer::new(0x1000, 0x5000),
                MmapBuffer::new(0x2000, 0xffff_b000),
            ]
        );
    }

    #[test]
    fn mmap_manager_mapping() {
        let mut mm = MmapMappingManager::from(DummyHostMemoryMapper);

        assert_eq!(mm.register_buffer(None, 0x1000), Ok(0x0));
        assert_eq!(mm.register_buffer(None, 0x5000), Ok(0x1000));
        assert_eq!(
            mm.buffers,
            vec![
                MmapBuffer::new(0x0, 0x1000),
                MmapBuffer::new(0x1000, 0x5000),
            ]
        );

        let buffer = host_buffer(0x5000);

        // A host buffer smaller than what the guest was told is refused.
        let short = host_buffer(0x4000);
        assert_eq!(
            mm.create_mapping(0x1000, &short, false),
            Err(CreateMappingError::SizeOutOfBounds)
        );

        // Single mapping
        assert_eq!(
            mm.create_mapping(0x1000, &buffer, false),
            Ok((0x8000_1000, 0x5000))
        );
        assert!(mm.is_mapped(0x1000));
        assert_eq!(mm.remove_mapping(0x8000_1000), Ok(false));
        assert!(!mm.is_mapped(0x1000));
        assert_eq!(
            mm.remove_mapping(0x8000_1000),
            Err(RemoveMappingError::InvalidOffset)
        );

        // Multiple mappings
        assert_eq!(
            mm.create_mapping(0x1000, &buffer, false),
            Ok((0x8000_1000, 0x5000))
        );
        assert_eq!(
            mm.create_mapping(0x1000, &buffer, false),
            Ok((0x8000_1000, 0x5000))
        );
        assert_eq!(mm.remove_mapping(0x8000_1000), Ok(true));
        assert_eq!(mm.remove_mapping(0x8000_1000), Ok(false));
        assert_eq!(
            mm.remove_mapping(0x8000_1000),
            Err(RemoveMappingError::InvalidOffset)
        );

        // Mapping at non-existing offset
        assert_eq!(
            mm.create_mapping(0x2000, &buffer, false),
            Err(CreateMappingError::InvalidOffset)
        );

        // Requesting same mapping with different access
        assert_eq!(
            mm.create_mapping(0x1000, &buffer, false),
            Ok((0x8000_1000, 0x5000))
        );
        assert_eq!(
            mm.create_mapping(0x1000, &buffer, true),
            Err(CreateMappingError::NonMatchingPermissions)
        );
        assert_eq!(mm.remove_mapping(0x8000_1000), Ok(false));

        // Mappings must survive a buffer's deregistration
        assert_eq!(
            mm.create_mapping(0x1000, &buffer, false),
            Ok((0x8000_1000, 0x5000))
        );
        assert!(mm.unregister_buffer(0x1000));
        assert_eq!(
            mm.buffers,
            vec![
                MmapBuffer::new(0x0, 0x1000),
                MmapBuffer {
                    offset: 0x1000,
                    size: 0x5000,
                    registered: false,
                    mapping: Some(MmapBufferMapping {
                        num_mappings: 1,
                        guest_addr: 0x8000_1000,
                        rw: false
                    })
                }
            ]
        );
        // ... but un-registered buffers are removed alongside their last mapping.
        assert_eq!(mm.remove_mapping(0x8000_1000), Ok(false));
        assert_eq!(mm.buffers, vec![MmapBuffer::new(0x0, 0x1000),]);
        assert!(mm.mapped.is_empty());
    }

    /// Two buffers mapped at once: `MUNMAP` must find each by its own guest address, and the
    /// table must not confuse them.
    #[test]
    fn mmap_manager_addr_table() {
        let mut mm = MmapMappingManager::from(DummyHostMemoryMapper);
        let a = host_buffer(0x1000);
        let b = host_buffer(0x1000);
        assert_eq!(mm.register_buffer(None, 0x1000), Ok(0x0));
        assert_eq!(mm.register_buffer(None, 0x1000), Ok(0x1000));
        assert_eq!(mm.buffer_size(0x1000), Some(0x1000));
        assert_eq!(mm.buffer_size(0x2000), None);

        assert_eq!(mm.create_mapping(0x0, &a, true), Ok((0x8000_0000, 0x1000)));
        assert_eq!(mm.create_mapping(0x1000, &b, true), Ok((0x8000_1000, 0x1000)));
        assert_eq!(mm.mapped.len(), 2);

        // Removing the second leaves the first intact.
        assert_eq!(mm.remove_mapping(0x8000_1000), Ok(false));
        assert!(mm.is_mapped(0x0));
        assert!(!mm.is_mapped(0x1000));
        assert_eq!(
            mm.remove_mapping(0x8000_1000),
            Err(RemoveMappingError::InvalidOffset)
        );
        assert_eq!(mm.remove_mapping(0x8000_0000), Ok(false));
        assert!(mm.mapped.is_empty());
    }

    /// A pool-style mapper answers with the buffer's pool offset rather than a fresh address.
    #[test]
    fn mmap_manager_pool_offsets() {
        let mut mm = MmapMappingManager::from(PoolStyleMapper);
        let mut buffer = host_buffer(0x2000);
        buffer.pool_offset = Some(0x40_0000);
        assert_eq!(mm.register_buffer(None, 0x2000), Ok(0x0));
        assert_eq!(mm.create_mapping(0x0, &buffer, true), Ok((0x40_0000, 0x2000)));
        assert_eq!(mm.remove_mapping(0x40_0000), Ok(false));

        // A buffer that is not in a pool cannot be mapped by a pool-only mapper.
        let plain = host_buffer(0x2000);
        assert_eq!(mm.register_buffer(None, 0x2000), Ok(0x1000));
        assert_eq!(
            mm.create_mapping(0x1000, &plain, true),
            Err(CreateMappingError::MappingFailure(libc::EINVAL))
        );
    }

    /// The allocator gets a retired buffer back only once the guest has unmapped it.
    #[test]
    fn retired_buffers_wait_for_the_last_munmap() {
        let mut mm = MmapMappingManager::from(DummyHostMemoryMapper);
        let mut allocator = CountingAllocator::default();
        let mut retired = RetiredBuffers::new();

        let mapped = allocator.allocate(0x1000).unwrap();
        let unmapped = allocator.allocate(0x1000).unwrap();
        assert_eq!(mm.register_buffer(None, 0x1000), Ok(0x0));
        assert_eq!(mm.register_buffer(None, 0x1000), Ok(0x1000));
        assert_eq!(mm.create_mapping(0x0, &mapped, true), Ok((0x8000_0000, 0x1000)));
        assert_eq!(mm.create_mapping(0x0, &mapped, true), Ok((0x8000_0000, 0x1000)));

        // Never mapped: released on the spot.
        retired.retire(&mut mm, &mut allocator, 0x1000, unmapped);
        assert_eq!(allocator.released, 1);
        assert!(retired.is_empty());

        // Mapped twice: held through both munmaps.
        retired.retire(&mut mm, &mut allocator, 0x0, mapped);
        assert_eq!(allocator.released, 1);
        assert_eq!(retired.len(), 1);
        retired.reap(&mm, &mut allocator);
        assert_eq!(allocator.released, 1);

        assert_eq!(mm.remove_mapping(0x8000_0000), Ok(true));
        retired.reap(&mm, &mut allocator);
        assert_eq!(allocator.released, 1);

        assert_eq!(mm.remove_mapping(0x8000_0000), Ok(false));
        retired.reap(&mm, &mut allocator);
        assert_eq!(allocator.released, 2);
        assert!(retired.is_empty());
        // The offset is free again only now.
        assert_eq!(mm.register_buffer(None, 0x1000), Ok(0x0));
    }
}
