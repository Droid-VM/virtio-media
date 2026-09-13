// Copyright 2024 The ChromiumOS Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

//! Virtio-media host devices.
//!
//! This module contains some host-side devices implementations that any VMM can use as long as it
//! provides implementations of the required traits.
//!
//! The conditions for using these devices are as follows:
//!
//! * [`std::io::Read`] and [`std::io::Write`] implementations for the device-readable and
//!   device-writable sections of the descriptor chain,
//! * An implementation of [`crate::VirtioMediaEventQueue`], so devices can send events to the guest,
//! * For devices that need to access guest memory linearly, an implementation of
//!   [`crate::VirtioMediaGuestMemoryMapper`].
//! * For devices that need to map host memory into the guest, an implementation of
//!   [`crate::VirtioMediaHostMemoryMapper`].
//!
//! [simple_device] implements a simple capture device that generates frames in software. It can be
//! used as a reference for how to write devices, or as a way to test the guest without any
//! specific hardware on the host.
//!
//! [v4l2_device_proxy] proxies any host V4L2 device to the guest, making its functionality
//! available to the guest with minimal overhead.
//!
//! [loopback_device] is a memory-to-memory device that copies every OUTPUT buffer the guest
//! queues into the next available CAPTURE buffer. It exercises both buffer ownerships at once --
//! guest-owned `USERPTR` and host-owned `MMAP`, on either queue -- which is what a guest driver
//! with its own buffer pool needs to be tested against (`VPU_DESIGN.md` §4.2).
//!
//! [camera] is a capture device over a host camera reached through the `CameraBackend` trait,
//! so the V4L2 side is VMM- and platform-independent and the VMM supplies the camera
//! (`VPU_DESIGN.md` §7.1).
//!
//! [video_decoder] and [video_encoder] are the stateful codec devices, the V4L2 halves of a
//! decoder and an encoder over the `VideoDecoderBackend` / `VideoEncoderBackend` traits the VMM
//! implements (`VPU_DESIGN.md` §7.2, §7.3).

#[cfg(feature = "simple-device")]
pub mod simple_device;
#[cfg(feature = "simple-device")]
pub use simple_device::SimpleCaptureDevice;

#[cfg(feature = "loopback-device")]
pub mod loopback_device;
#[cfg(feature = "loopback-device")]
pub use loopback_device::LoopbackDevice;

#[cfg(feature = "camera-device")]
pub mod camera;
#[cfg(feature = "camera-device")]
pub use camera::CameraDevice;

pub mod v4l2_device_proxy;
pub use v4l2_device_proxy::V4l2ProxyDevice;

#[cfg(feature = "video-decoder-device")]
pub mod video_decoder;
#[cfg(feature = "video-decoder-device")]
pub use video_decoder::VideoDecoder;

#[cfg(feature = "video-encoder-device")]
pub mod video_encoder;
#[cfg(feature = "video-encoder-device")]
pub use video_encoder::VideoEncoder;

/// A byte budget the device tests' allocators consult, standing in for the VMM's fixed-size
/// `media_host` pool.
///
/// [`crate::MemFdAllocator`] never runs out, so before D73 no test could tell "the pool granted
/// what it could" from "the pool granted everything": the partial `REQBUFS`/`CREATE_BUFS` rule
/// needs an allocator that refuses the seventh buffer and keeps counting bytes. Each device's
/// test allocator wraps the memfd one and asks this first; the default is unlimited, so every
/// test that predates the rule allocates exactly as it always did.
#[cfg(test)]
pub(crate) mod test_pool {
    use std::cell::Cell;
    use std::rc::Rc;

    /// `capacity` bytes, `used` of them handed out. Shared (`Rc`) between the test and the
    /// allocator the device holds, because the test sizes it and reads it back.
    pub(crate) struct PoolBudget {
        capacity: Cell<u64>,
        used: Cell<u64>,
    }

    impl PoolBudget {
        /// A pool nothing can exhaust: what every test that does not size one gets.
        pub(crate) fn unlimited() -> Rc<Self> {
            Rc::new(PoolBudget {
                capacity: Cell::new(u64::MAX),
                used: Cell::new(0),
            })
        }

        /// Size the pool to hold exactly `n` buffers of `sizeimage` bytes and not one more.
        pub(crate) fn holds(&self, n: u64, sizeimage: u64) {
            self.capacity.set(n * sizeimage);
        }

        /// Leave room for exactly `n` more buffers of `sizeimage` bytes on top of what is
        /// already out -- for a queue allocated against a pool another queue is already using.
        pub(crate) fn holds_more(&self, n: u64, sizeimage: u64) {
            self.capacity.set(self.used.get() + n * sizeimage);
        }

        /// Bytes handed out and not yet given back -- the `pool used` of the VMM's accounting
        /// line, which a partial answer must match buffer for buffer.
        pub(crate) fn used(&self) -> u64 {
            self.used.get()
        }

        /// Take `len` bytes, or `ENOMEM` if they do not fit -- the errno the VMM's own pool
        /// answers an exhausted `media_host` with.
        pub(crate) fn take(&self, len: u64) -> Result<(), i32> {
            let used = self.used.get();
            match used.checked_add(len) {
                Some(after) if after <= self.capacity.get() => {
                    self.used.set(after);
                    Ok(())
                }
                _ => Err(libc::ENOMEM),
            }
        }

        /// Give `len` bytes back.
        pub(crate) fn give_back(&self, len: u64) {
            self.used.set(self.used.get().saturating_sub(len));
        }
    }
}
