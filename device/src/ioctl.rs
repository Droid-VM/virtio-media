// Copyright 2024 The ChromiumOS Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

use std::io::Result as IoResult;

use v4l2r::bindings::v4l2_audio;
use v4l2r::bindings::v4l2_audioout;
use v4l2r::bindings::v4l2_buffer;
use v4l2r::bindings::v4l2_control;
use v4l2r::bindings::v4l2_create_buffers;
use v4l2r::bindings::v4l2_decoder_cmd;
use v4l2r::bindings::v4l2_dv_timings;
use v4l2r::bindings::v4l2_dv_timings_cap;
use v4l2r::bindings::v4l2_enc_idx;
use v4l2r::bindings::v4l2_encoder_cmd;
use v4l2r::bindings::v4l2_enum_dv_timings;
use v4l2r::bindings::v4l2_event_subscription;
use v4l2r::bindings::v4l2_ext_control;
use v4l2r::bindings::v4l2_ext_controls;
use v4l2r::bindings::v4l2_fmtdesc;
use v4l2r::bindings::v4l2_format;
use v4l2r::bindings::v4l2_frequency;
use v4l2r::bindings::v4l2_frequency_band;
use v4l2r::bindings::v4l2_frmivalenum;
use v4l2r::bindings::v4l2_frmsizeenum;
use v4l2r::bindings::v4l2_input;
use v4l2r::bindings::v4l2_modulator;
use v4l2r::bindings::v4l2_output;
use v4l2r::bindings::v4l2_plane;
use v4l2r::bindings::v4l2_query_ext_ctrl;
use v4l2r::bindings::v4l2_queryctrl;
use v4l2r::bindings::v4l2_querymenu;
use v4l2r::bindings::v4l2_rect;
use v4l2r::bindings::v4l2_requestbuffers;
use v4l2r::bindings::v4l2_selection;
use v4l2r::bindings::v4l2_standard;
use v4l2r::bindings::v4l2_std_id;
use v4l2r::bindings::v4l2_streamparm;
use v4l2r::bindings::v4l2_tuner;
use v4l2r::ioctl::AudioMode;
use v4l2r::ioctl::CtrlId;
use v4l2r::ioctl::CtrlWhich;
use v4l2r::ioctl::EventType as V4l2EventType;
use v4l2r::ioctl::QueryCtrlFlags;
use v4l2r::ioctl::SelectionFlags;
use v4l2r::ioctl::SelectionTarget;
use v4l2r::ioctl::SelectionType;
use v4l2r::ioctl::SubscribeEventFlags;
use v4l2r::ioctl::TunerMode;
use v4l2r::ioctl::TunerTransmissionFlags;
use v4l2r::ioctl::TunerType;
use v4l2r::ioctl::UncheckedV4l2Buffer;
use v4l2r::ioctl::V4l2Buffer;
use v4l2r::ioctl::V4l2BufferFromError;
use v4l2r::ioctl::V4l2PlanesWithBacking;
use v4l2r::memory::MemoryType;
use v4l2r::QueueDirection;
use v4l2r::QueueType;

use crate::io::ReadFromDescriptorChain;
use crate::io::VmediaType;
use crate::io::WriteToDescriptorChain;
use crate::protocol::RespHeader;
use crate::protocol::SgEntry;
use crate::protocol::V4l2Ioctl;

/// Most entries a single SG list may carry.
///
/// The guest chooses the entry count, and every entry it sends is another `read_obj` and, later,
/// another host mapping. 4096 entries of 4 KiB pages is 16 MiB, more than a 4K frame, and four
/// times what the guest driver's own shadow buffer can hold at 16 bytes per entry (64 KiB, see
/// `VPU_DESIGN.md` §5.2), so nothing that is expected to work is refused.
pub const MAX_SG_ENTRIES: usize = 4096;

/// Reads a SG list of guest physical addresses passed from the driver and returns it.
///
/// The list is bounded by `MAX_SG_ENTRIES`, and an entry of length 0 is refused: it would never
/// advance `bytes_taken`, so a guest could keep this loop reading entries for as long as the
/// descriptor chain holds out (`VPU_DESIGN.md` §4.3).
fn get_userptr_regions<R: ReadFromDescriptorChain>(
    r: &mut R,
    size: usize,
) -> anyhow::Result<Vec<SgEntry>> {
    let mut bytes_taken = 0;
    let mut res = Vec::new();

    while bytes_taken < size {
        if res.len() >= MAX_SG_ENTRIES {
            anyhow::bail!(
                "SG list exceeds the {} entry limit before covering {} bytes",
                MAX_SG_ENTRIES,
                size
            );
        }
        let sg_entry = r.read_obj::<SgEntry>()?;
        if sg_entry.len == 0 {
            anyhow::bail!("SG entry of length 0 at guest address {:#x}", sg_entry.start);
        }
        bytes_taken += sg_entry.len as usize;
        res.push(sg_entry);
    }

    Ok(res)
}

/// Local trait for reading simple or complex objects from a reader, e.g. the device-readable
/// section of a descriptor chain.
trait FromDescriptorChain {
    fn read_from_chain<R: ReadFromDescriptorChain>(reader: &mut R) -> std::io::Result<Self>
    where
        Self: Sized;
}

/// Implementation for simple objects that can be returned as-is after their endianness is
/// fixed.
impl<T> FromDescriptorChain for T
where
    T: VmediaType,
{
    fn read_from_chain<R: ReadFromDescriptorChain>(reader: &mut R) -> std::io::Result<Self> {
        reader.read_obj()
    }
}

/// Which slots of the plane array a guest sent describe a payload -- `bytesused` and
/// `data_offset` -- that the buffer can actually hold.
///
/// They are the two fields of a queued buffer a guest can make nonsensical without making the
/// buffer itself unusable, and the reader cannot decide what that means: it does not know how
/// many planes the queue's format has, and V4L2 does not treat the pair the same way everywhere.
/// So the reader zeroes the slots it could not represent, records them here, and the device
/// applies its own queue's rule.
///
/// The rule is vb2's `__verify_length`
/// (`GKI_6.18-2026-06_r11/drivers/media/common/videobuf2/videobuf2-v4l2.c:95-129`), and it has
/// three parts:
///
/// * it looks at `vb->num_planes` entries only, never at `b->length` -- `length` is the size of
///   the caller's plane *array*, and the slots past the format's plane count are scratch space a
///   caller may legally leave dirty. ffmpeg does exactly that: it sends
///   `length = VIDEO_MAX_PLANES` with an uninitialised `v4l2_plane[8]` on the stack and fills
///   only `planes[0]` from `QUERYBUF`, so judging all eight slots refused every `QBUF` it made
///   and `ffmpeg -f v4l2 -i /dev/videoN` could not capture a single frame (defect D21,
///   `logs/vpu_wp/B5-acceptance.md` §4.3 and §6);
/// * on a **capture** queue it returns 0 before looking at anything (`:101-102`): the payload is
///   the device's to report, not the caller's to declare;
/// * on an **output** queue every one of those planes must satisfy `bytesused <= length` and
///   `data_offset < bytesused`, which for these devices is the bitstream length.
///
/// [`Self::is_accepted_by`] is that rule. The devices here take the capture half only for `MMAP`
/// buffers, which is D21's case; on a `USERPTR` capture buffer the guest's own description is
/// the only thing the device has to check, and nothing needs it relaxed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PayloadValidity(u8);

// One bit per plane slot, so the mask must have as many bits as a `v4l2_buffer` has slots.
const _: () = assert!(v4l2r::bindings::VIDEO_MAX_PLANES == u8::BITS);

impl PayloadValidity {
    /// Every slot describes a payload the buffer can hold -- nothing was sanitised.
    pub const ALL: Self = Self(u8::MAX);

    /// The same, with the payload of `plane` marked as zeroed by the reader.
    pub const fn without(self, plane: usize) -> Self {
        if plane >= u8::BITS as usize {
            self
        } else {
            Self(self.0 & !(1u8 << plane))
        }
    }

    /// Whether the guest's payload description for `plane` was taken as sent.
    pub const fn plane_is_valid(self, plane: usize) -> bool {
        plane < u8::BITS as usize && (self.0 & (1u8 << plane)) != 0
    }

    /// Whether the first `num_planes` slots -- the ones a format with `num_planes` planes
    /// actually uses -- were all taken as sent. A count larger than a plane array is never
    /// valid.
    pub fn planes_are_valid(self, num_planes: usize) -> bool {
        (0..num_planes).all(|plane| self.plane_is_valid(plane))
    }

    /// Whether a queue of `num_planes` planes, in `direction` and `memory` mode, may take this
    /// payload description: vb2's `__verify_length` rule, as the type doc describes it.
    pub fn is_accepted_by(
        self,
        direction: QueueDirection,
        memory: MemoryType,
        num_planes: usize,
    ) -> bool {
        match (direction, memory) {
            // The device fills the buffer and reports what it wrote, so V4L2 ignores whatever
            // the guest declared here -- including the dirty stack ffmpeg sends (D21).
            (QueueDirection::Capture, MemoryType::Mmap) => true,
            _ => self.planes_are_valid(num_planes),
        }
    }
}

/// A `v4l2_buffer` as the guest sent it, together with the SG lists of its `USERPTR` planes.
///
/// `payload` says which of the plane slots' `bytesused` / `data_offset` the reader could take as
/// sent; the rest have been zeroed here so that the buffer -- index, memory type, plane backing,
/// SG lists -- is still there to be used. `QBUF` **ignores** both fields once `PREPARE_BUF` has
/// taken the buffer, and a capture queue ignores them always, so only the device knows whether
/// an inconsistent pair is an error: see [`PayloadValidity`].
pub struct GuestV4l2Buffer {
    pub buffer: V4l2Buffer,
    pub guest_regions: Vec<Vec<SgEntry>>,
    pub payload: PayloadValidity,
}

/// Implementation to easily read a `v4l2_buffer` of `USERPTR` memory type and its associated
/// guest-side buffers from a descriptor chain.
impl FromDescriptorChain for GuestV4l2Buffer {
    fn read_from_chain<R: ReadFromDescriptorChain>(reader: &mut R) -> IoResult<Self>
    where
        Self: Sized,
    {
        let v4l2_buffer = reader.read_obj::<v4l2_buffer>()?;
        let queue = match QueueType::n(v4l2_buffer.type_) {
            Some(queue) => queue,
            None => return Err(std::io::ErrorKind::InvalidData.into()),
        };

        let v4l2_planes = if queue.is_multiplanar() {
            // `length` is the size of the caller's plane array, and vb2 takes
            // `num_planes <= length <= VB2_MAX_PLANES` (`__verify_planes_array`,
            // `videobuf2-v4l2.c:76`). A format has at least one plane, so an array of none is
            // not a buffer any queue could use -- and `V4l2Buffer::get_first_plane` would
            // `unwrap` an empty iterator on it, which aborts this VMM. Refused here, where the
            // buffer is built, rather than left for each device to trip over.
            if v4l2_buffer.length == 0 || v4l2_buffer.length > v4l2r::bindings::VIDEO_MAX_PLANES {
                return Err(std::io::ErrorKind::InvalidData.into());
            }

            let planes: [v4l2r::bindings::v4l2_plane; v4l2r::bindings::VIDEO_MAX_PLANES as usize] =
                (0..v4l2_buffer.length as usize)
                    .map(|_| reader.read_obj::<v4l2_plane>())
                    .collect::<IoResult<Vec<_>>>()?
                    .into_iter()
                    .chain(std::iter::repeat(Default::default()))
                    .take(v4l2r::bindings::VIDEO_MAX_PLANES as usize)
                    .collect::<Vec<_>>()
                    .try_into()
                    .map_err(|_| std::io::Error::from(std::io::ErrorKind::InvalidData))?;
            Some(planes)
        } else {
            None
        };

        // A `v4l2_buffer` whose `bytesused` overflows its `length`, or whose `data_offset` is not
        // inside its payload, is refused by `V4l2Buffer`'s invariants -- but it is a legal thing
        // for a guest to send: V4L2 ignores both fields on a buffer `PREPARE_BUF` has taken and
        // on every capture queue, and the slots past the format's plane count are scratch space
        // a caller may leave dirty. Zero the offending slot, remember which one it was, and let
        // the device apply its own queue's rule (`PayloadValidity`).
        //
        // One slot at a time, not the whole array: judging all eight of them at once is what
        // refused every `QBUF` ffmpeg made, which sends `length = VIDEO_MAX_PLANES` with planes
        // 1..7 uninitialised (defect D21, `logs/vpu_wp/B5-acceptance.md` §4.3). Zeroing the
        // whole array would also have destroyed plane 0's `bytesused` -- the bitstream length an
        // output queue must still validate -- on a buffer whose only fault was in the scratch.
        //
        // The third way this conversion used to fail was `InvalidNumberOfPlanes` for a `length`
        // of exactly `VIDEO_MAX_PLANES`, which is the plane array size ffmpeg puts on every
        // multiplanar buffer ioctl (`libavdevice/v4l2.c`) and which V4L2 allows -- the kernel's
        // `__verify_planes_array` takes `num_planes <= length <= VB2_MAX_PLANES`. It is fixed in
        // the v4l2r fork (Droid-VM/v4l2r wip/vpu, `lib/src/ioctl.rs`), not forgiven here: unlike
        // the two above it is not a field a device could reinterpret, and letting it through
        // with the plane array clamped would have hidden a real out-of-range `length` too. It is
        // why `ffmpeg -f v4l2 -i /dev/video0` could not queue a single buffer (D17).
        let mut unchecked_buffer = v4l2_buffer;
        let mut unchecked_planes = v4l2_planes;
        let mut payload = PayloadValidity::ALL;
        let v4l2_buffer = loop {
            let attempt =
                V4l2Buffer::try_from(UncheckedV4l2Buffer(unchecked_buffer, unchecked_planes));
            match attempt {
                Ok(buffer) => break buffer,
                Err(V4l2BufferFromError::PlaneSizeOverflow(plane, ..))
                | Err(V4l2BufferFromError::InvalidDataOffset(plane, ..)) => {
                    // Zeroing a slot makes it representable, so the same slot cannot come back:
                    // if it does, the conversion is failing for a reason this loop cannot fix
                    // and the buffer is not addressable at all.
                    if !payload.plane_is_valid(plane) {
                        return Err(std::io::ErrorKind::InvalidData.into());
                    }
                    payload = payload.without(plane);
                    match unchecked_planes.as_mut() {
                        Some(planes) => match planes.get_mut(plane) {
                            Some(slot) => {
                                slot.bytesused = 0;
                                slot.data_offset = 0;
                            }
                            None => return Err(std::io::ErrorKind::InvalidData.into()),
                        },
                        // Single-planar: the payload is on the buffer itself, and the only
                        // plane v4l2r can name is 0.
                        None => unchecked_buffer.bytesused = 0,
                    }
                }
                Err(_) => return Err(std::io::ErrorKind::InvalidData.into()),
            }
        };

        // Read the `MemRegion`s of all planes if the buffer is `USERPTR`.
        let guest_regions = if let V4l2PlanesWithBacking::UserPtr(planes) =
            v4l2_buffer.planes_with_backing_iter()
        {
            planes
                .filter(|p| *p.length > 0)
                .map(|p| {
                    get_userptr_regions(reader, *p.length as usize)
                        .map_err(|_| std::io::ErrorKind::InvalidData.into())
                })
                .collect::<IoResult<Vec<_>>>()?
        } else {
            vec![]
        };

        Ok(GuestV4l2Buffer {
            buffer: v4l2_buffer,
            guest_regions,
            payload,
        })
    }
}

/// Implementation to easily read a `v4l2_ext_controls` struct, its array of controls, and the SG
/// list of the buffers pointed to by the controls from a descriptor chain.
impl FromDescriptorChain for (v4l2_ext_controls, Vec<v4l2_ext_control>, Vec<Vec<SgEntry>>) {
    fn read_from_chain<R: ReadFromDescriptorChain>(reader: &mut R) -> std::io::Result<Self>
    where
        Self: Sized,
    {
        let ctrls = reader.read_obj::<v4l2_ext_controls>()?;

        let ctrl_array = (0..ctrls.count)
            .map(|_| reader.read_obj::<v4l2_ext_control>())
            .collect::<IoResult<Vec<_>>>()?;

        // Read all the payloads.
        let mem_regions = ctrl_array
            .iter()
            .filter(|ctrl| ctrl.size > 0)
            .map(|ctrl| {
                get_userptr_regions(reader, ctrl.size as usize)
                    .map_err(|_| std::io::Error::from(std::io::ErrorKind::InvalidData))
            })
            .collect::<IoResult<Vec<_>>>()?;

        Ok((ctrls, ctrl_array, mem_regions))
    }
}

/// Local trait for writing simple or complex objects to a writer, e.g. the device-writable section
/// of a descriptor chain.
trait ToDescriptorChain {
    fn write_to_chain<W: WriteToDescriptorChain>(self, writer: &mut W) -> std::io::Result<()>;
}

/// Implementation for simple objects that can be written as-is after their endianness is
/// fixed.
impl<T> ToDescriptorChain for T
where
    T: VmediaType,
{
    fn write_to_chain<W: WriteToDescriptorChain>(self, writer: &mut W) -> std::io::Result<()> {
        writer.write_obj(self)
    }
}

/// Implementation to easily write a `v4l2_buffer` to a descriptor chain, while ensuring the number
/// of planes written is not larger than a limit (i.e. the maximum number of planes that the
/// descriptor chain can receive).
impl ToDescriptorChain for (V4l2Buffer, usize) {
    fn write_to_chain<W: WriteToDescriptorChain>(self, writer: &mut W) -> std::io::Result<()> {
        let mut v4l2_buffer = *self.0.as_v4l2_buffer();
        // If the buffer is multiplanar, nullify the `planes` pointer to avoid leaking host
        // addresses.
        if self.0.queue().is_multiplanar() {
            v4l2_buffer.m.planes = std::ptr::null_mut();
        }
        writer.write_obj(v4l2_buffer)?;

        // Write plane information if the buffer is multiplanar. Limit the number of planes to the
        // upper bound we were given.
        for plane in self.0.as_v4l2_planes().iter().take(self.1) {
            writer.write_obj(*plane)?;
        }

        Ok(())
    }
}

/// Implementation to easily write a `v4l2_ext_controls` struct and its array of controls to a
/// descriptor chain.
impl ToDescriptorChain for (v4l2_ext_controls, Vec<v4l2_ext_control>) {
    fn write_to_chain<W: WriteToDescriptorChain>(self, writer: &mut W) -> std::io::Result<()> {
        let (ctrls, ctrl_array) = self;
        let mut ctrls = ctrls;

        // Nullify the control pointer to avoid leaking host addresses.
        ctrls.controls = std::ptr::null_mut();
        writer.write_obj(ctrls)?;

        for ctrl in ctrl_array {
            writer.write_obj(ctrl)?;
        }

        Ok(())
    }
}

/// Returns `ENOTTY` to signal that an ioctl is not handled by this device.
macro_rules! unhandled_ioctl {
    () => {
        Err(libc::ENOTTY)
    };
}

pub type IoctlResult<T> = Result<T, i32>;

/// Trait for implementing ioctls supported by a device.
///
/// It provides a default implementation for all ioctls that returns the error code for an
/// unsupported ioctl (`ENOTTY`) to the driver. This means that a device just needs to implement
/// this trait and override the ioctls it supports in order to provide the expected behavior. All
/// parsing and input validation is done by the companion function [`virtio_media_dispatch_ioctl`].
#[allow(unused_variables)]
pub trait VirtioMediaIoctlHandler {
    type Session;

    fn enum_fmt(
        &mut self,
        session: &Self::Session,
        queue: QueueType,
        index: u32,
    ) -> IoctlResult<v4l2_fmtdesc> {
        unhandled_ioctl!()
    }
    fn g_fmt(&mut self, session: &Self::Session, queue: QueueType) -> IoctlResult<v4l2_format> {
        unhandled_ioctl!()
    }
    /// Hook for the `VIDIOC_S_FMT` ioctl.
    ///
    /// `queue` is guaranteed to match `format.type_`.
    fn s_fmt(
        &mut self,
        session: &mut Self::Session,
        queue: QueueType,
        format: v4l2_format,
    ) -> IoctlResult<v4l2_format> {
        unhandled_ioctl!()
    }
    fn reqbufs(
        &mut self,
        session: &mut Self::Session,
        queue: QueueType,
        memory: MemoryType,
        count: u32,
    ) -> IoctlResult<v4l2_requestbuffers> {
        unhandled_ioctl!()
    }
    fn querybuf(
        &mut self,
        session: &Self::Session,
        queue: QueueType,
        index: u32,
    ) -> IoctlResult<V4l2Buffer> {
        unhandled_ioctl!()
    }

    /// `payload` names the plane slots whose `bytesused` / `data_offset` the guest described in a
    /// way the buffer can hold; the rest were zeroed by the dispatcher. Only the device knows
    /// which slots its queue's format uses and which are the caller's scratch, so it is the
    /// device that judges them -- [`PayloadValidity::is_accepted_by`] is the rule vb2 applies.
    /// V4L2 ignores both fields on a buffer `PREPARE_BUF` has already taken, so a device that
    /// implements `prepare_buf` may accept any description on a prepared buffer.
    fn qbuf(
        &mut self,
        session: &mut Self::Session,
        buffer: V4l2Buffer,
        guest_regions: Vec<Vec<SgEntry>>,
        payload: PayloadValidity,
    ) -> IoctlResult<V4l2Buffer> {
        unhandled_ioctl!()
    }

    // TODO expbuf

    fn streamon(&mut self, session: &mut Self::Session, queue: QueueType) -> IoctlResult<()> {
        unhandled_ioctl!()
    }
    fn streamoff(&mut self, session: &mut Self::Session, queue: QueueType) -> IoctlResult<()> {
        unhandled_ioctl!()
    }

    fn g_parm(
        &mut self,
        session: &Self::Session,
        queue: QueueType,
    ) -> IoctlResult<v4l2_streamparm> {
        unhandled_ioctl!()
    }
    fn s_parm(
        &mut self,
        session: &mut Self::Session,
        parm: v4l2_streamparm,
    ) -> IoctlResult<v4l2_streamparm> {
        unhandled_ioctl!()
    }

    fn g_std(&mut self, session: &Self::Session) -> IoctlResult<v4l2_std_id> {
        unhandled_ioctl!()
    }

    fn s_std(&mut self, session: &mut Self::Session, std: v4l2_std_id) -> IoctlResult<()> {
        unhandled_ioctl!()
    }

    fn enumstd(&mut self, session: &Self::Session, index: u32) -> IoctlResult<v4l2_standard> {
        unhandled_ioctl!()
    }

    fn enuminput(&mut self, session: &Self::Session, index: u32) -> IoctlResult<v4l2_input> {
        unhandled_ioctl!()
    }

    fn g_ctrl(&mut self, session: &Self::Session, id: u32) -> IoctlResult<v4l2_control> {
        unhandled_ioctl!()
    }

    fn s_ctrl(
        &mut self,
        session: &mut Self::Session,
        id: u32,
        value: i32,
    ) -> IoctlResult<v4l2_control> {
        unhandled_ioctl!()
    }

    fn g_tuner(&mut self, session: &Self::Session, index: u32) -> IoctlResult<v4l2_tuner> {
        unhandled_ioctl!()
    }

    fn s_tuner(
        &mut self,
        session: &mut Self::Session,
        index: u32,
        mode: TunerMode,
    ) -> IoctlResult<()> {
        unhandled_ioctl!()
    }

    fn g_audio(&mut self, session: &Self::Session) -> IoctlResult<v4l2_audio> {
        unhandled_ioctl!()
    }

    fn s_audio(
        &mut self,
        session: &mut Self::Session,
        index: u32,
        mode: Option<AudioMode>,
    ) -> IoctlResult<()> {
        unhandled_ioctl!()
    }

    fn queryctrl(
        &mut self,
        session: &Self::Session,
        id: CtrlId,
        flags: QueryCtrlFlags,
    ) -> IoctlResult<v4l2_queryctrl> {
        unhandled_ioctl!()
    }

    fn querymenu(
        &mut self,
        session: &Self::Session,
        id: u32,
        index: u32,
    ) -> IoctlResult<v4l2_querymenu> {
        unhandled_ioctl!()
    }

    fn g_input(&mut self, session: &Self::Session) -> IoctlResult<i32> {
        unhandled_ioctl!()
    }

    fn s_input(&mut self, session: &mut Self::Session, input: i32) -> IoctlResult<i32> {
        unhandled_ioctl!()
    }

    fn g_output(&mut self, session: &Self::Session) -> IoctlResult<i32> {
        unhandled_ioctl!()
    }

    fn s_output(&mut self, session: &mut Self::Session, output: i32) -> IoctlResult<i32> {
        unhandled_ioctl!()
    }

    fn enumoutput(&mut self, session: &Self::Session, index: u32) -> IoctlResult<v4l2_output> {
        unhandled_ioctl!()
    }

    fn g_audout(&mut self, session: &Self::Session) -> IoctlResult<v4l2_audioout> {
        unhandled_ioctl!()
    }

    fn s_audout(&mut self, session: &mut Self::Session, index: u32) -> IoctlResult<()> {
        unhandled_ioctl!()
    }

    fn g_modulator(&mut self, session: &Self::Session, index: u32) -> IoctlResult<v4l2_modulator> {
        unhandled_ioctl!()
    }

    fn s_modulator(
        &mut self,
        session: &mut Self::Session,
        index: u32,
        flags: TunerTransmissionFlags,
    ) -> IoctlResult<()> {
        unhandled_ioctl!()
    }

    fn g_frequency(&mut self, session: &Self::Session, tuner: u32) -> IoctlResult<v4l2_frequency> {
        unhandled_ioctl!()
    }

    fn s_frequency(
        &mut self,
        session: &mut Self::Session,
        tuner: u32,
        type_: TunerType,
        frequency: u32,
    ) -> IoctlResult<()> {
        unhandled_ioctl!()
    }

    fn querystd(&mut self, session: &Self::Session) -> IoctlResult<v4l2_std_id> {
        unhandled_ioctl!()
    }

    /// Hook for the `VIDIOC_TRY_FMT` ioctl.
    ///
    /// `queue` is guaranteed to match `format.type_`.
    fn try_fmt(
        &mut self,
        session: &Self::Session,
        queue: QueueType,
        format: v4l2_format,
    ) -> IoctlResult<v4l2_format> {
        unhandled_ioctl!()
    }

    fn enumaudio(&mut self, session: &Self::Session, index: u32) -> IoctlResult<v4l2_audio> {
        unhandled_ioctl!()
    }

    fn enumaudout(&mut self, session: &Self::Session, index: u32) -> IoctlResult<v4l2_audioout> {
        unhandled_ioctl!()
    }

    /// Ext control ioctls modify `ctrls` and `ctrl_array` in place instead of returning them.
    fn g_ext_ctrls(
        &mut self,
        session: &Self::Session,
        which: CtrlWhich,
        ctrls: &mut v4l2_ext_controls,
        ctrl_array: &mut Vec<v4l2_ext_control>,
        user_regions: Vec<Vec<SgEntry>>,
    ) -> IoctlResult<()> {
        unhandled_ioctl!()
    }
    /// Ext control ioctls modify `ctrls` and `ctrl_array` in place instead of returning them.
    fn s_ext_ctrls(
        &mut self,
        session: &mut Self::Session,
        which: CtrlWhich,
        ctrls: &mut v4l2_ext_controls,
        ctrl_array: &mut Vec<v4l2_ext_control>,
        user_regions: Vec<Vec<SgEntry>>,
    ) -> IoctlResult<()> {
        unhandled_ioctl!()
    }
    /// Ext control ioctls modify `ctrls` and `ctrl_array` in place instead of returning them.
    fn try_ext_ctrls(
        &mut self,
        session: &Self::Session,
        which: CtrlWhich,
        ctrls: &mut v4l2_ext_controls,
        ctrl_array: &mut Vec<v4l2_ext_control>,
        user_regions: Vec<Vec<SgEntry>>,
    ) -> IoctlResult<()> {
        unhandled_ioctl!()
    }

    fn enum_framesizes(
        &mut self,
        session: &Self::Session,
        index: u32,
        pixel_format: u32,
    ) -> IoctlResult<v4l2_frmsizeenum> {
        unhandled_ioctl!()
    }

    fn enum_frameintervals(
        &mut self,
        session: &Self::Session,
        index: u32,
        pixel_format: u32,
        width: u32,
        height: u32,
    ) -> IoctlResult<v4l2_frmivalenum> {
        unhandled_ioctl!()
    }

    fn g_enc_index(&mut self, session: &Self::Session) -> IoctlResult<v4l2_enc_idx> {
        unhandled_ioctl!()
    }

    fn encoder_cmd(
        &mut self,
        session: &mut Self::Session,
        cmd: v4l2_encoder_cmd,
    ) -> IoctlResult<v4l2_encoder_cmd> {
        unhandled_ioctl!()
    }

    fn try_encoder_cmd(
        &mut self,
        session: &Self::Session,
        cmd: v4l2_encoder_cmd,
    ) -> IoctlResult<v4l2_encoder_cmd> {
        unhandled_ioctl!()
    }

    fn s_dv_timings(
        &mut self,
        session: &mut Self::Session,
        timings: v4l2_dv_timings,
    ) -> IoctlResult<v4l2_dv_timings> {
        unhandled_ioctl!()
    }

    fn g_dv_timings(&mut self, session: &Self::Session) -> IoctlResult<v4l2_dv_timings> {
        unhandled_ioctl!()
    }

    fn subscribe_event(
        &mut self,
        session: &mut Self::Session,
        event: V4l2EventType,
        flags: SubscribeEventFlags,
    ) -> IoctlResult<()> {
        unhandled_ioctl!()
    }

    fn unsubscribe_event(
        &mut self,
        session: &mut Self::Session,
        event: v4l2_event_subscription,
    ) -> IoctlResult<()> {
        unhandled_ioctl!()
    }

    /// `queue` and `memory` are validated versions of the information in `create_buffers`.
    ///
    /// `create_buffers` is modified in place and returned to the guest event in case of error.
    fn create_bufs(
        &mut self,
        session: &mut Self::Session,
        count: u32,
        queue: QueueType,
        memory: MemoryType,
        format: v4l2_format,
    ) -> IoctlResult<v4l2_create_buffers> {
        unhandled_ioctl!()
    }

    /// `payload` is as in [`Self::qbuf`], except that `PREPARE_BUF` is the ioctl that
    /// *validates* the payload, so an implementation must refuse a description its queue cannot
    /// take -- there is no earlier call to have accepted one.
    ///
    /// The default is `ENOTTY` whatever the guest sent, which is what makes "this device has no
    /// `PREPARE_BUF`" a single answer: the dispatcher used to validate the payload first, so a
    /// device without the ioctl answered `EINVAL` to a malformed buffer and `ENOTTY` to a
    /// well-formed one, and `v4l2-compliance` reads the first answer as "the ioctl exists".
    fn prepare_buf(
        &mut self,
        session: &mut Self::Session,
        buffer: V4l2Buffer,
        guest_regions: Vec<Vec<SgEntry>>,
        payload: PayloadValidity,
    ) -> IoctlResult<V4l2Buffer> {
        unhandled_ioctl!()
    }

    fn g_selection(
        &mut self,
        session: &Self::Session,
        sel_type: SelectionType,
        sel_target: SelectionTarget,
    ) -> IoctlResult<v4l2_rect> {
        unhandled_ioctl!()
    }

    fn s_selection(
        &mut self,
        session: &mut Self::Session,
        sel_type: SelectionType,
        sel_target: SelectionTarget,
        sel_rect: v4l2_rect,
        sel_flags: SelectionFlags,
    ) -> IoctlResult<v4l2_rect> {
        unhandled_ioctl!()
    }

    fn decoder_cmd(
        &mut self,
        session: &mut Self::Session,
        cmd: v4l2_decoder_cmd,
    ) -> IoctlResult<v4l2_decoder_cmd> {
        unhandled_ioctl!()
    }

    fn try_decoder_cmd(
        &mut self,
        session: &Self::Session,
        cmd: v4l2_decoder_cmd,
    ) -> IoctlResult<v4l2_decoder_cmd> {
        unhandled_ioctl!()
    }

    fn enum_dv_timings(
        &mut self,
        session: &Self::Session,
        index: u32,
    ) -> IoctlResult<v4l2_dv_timings> {
        unhandled_ioctl!()
    }

    fn query_dv_timings(&mut self, session: &Self::Session) -> IoctlResult<v4l2_dv_timings> {
        unhandled_ioctl!()
    }

    fn dv_timings_cap(&self, session: &Self::Session) -> IoctlResult<v4l2_dv_timings_cap> {
        unhandled_ioctl!()
    }

    fn enum_freq_bands(
        &self,
        session: &Self::Session,
        tuner: u32,
        type_: TunerType,
        index: u32,
    ) -> IoctlResult<v4l2_frequency_band> {
        unhandled_ioctl!()
    }

    fn query_ext_ctrl(
        &mut self,
        session: &Self::Session,
        id: CtrlId,
        flags: QueryCtrlFlags,
    ) -> IoctlResult<v4l2_query_ext_ctrl> {
        unhandled_ioctl!()
    }

    /// `VIDIOC_QUERYCTRL` with the `id` word as the guest sent it: the control id with the
    /// `V4L2_CTRL_FLAG_NEXT_*` bits still in it. The default splits it and calls
    /// [`Self::queryctrl`], which is what a device forwarding to a host V4L2 node wants
    /// (`CtrlId` is v4l2r's newtype for its own ioctl wrappers and keeps the number to itself).
    /// A device that answers from a control table of its own, and so must compare and order
    /// ids, implements this one instead.
    fn queryctrl_raw(&mut self, session: &Self::Session, id: u32) -> IoctlResult<v4l2_queryctrl> {
        let (id, flags) = v4l2r::ioctl::parse_ctrl_id_and_flags(id);
        self.queryctrl(session, id, flags)
    }

    /// `VIDIOC_QUERY_EXT_CTRL` with the `id` word as the guest sent it; see
    /// [`Self::queryctrl_raw`].
    fn query_ext_ctrl_raw(
        &mut self,
        session: &Self::Session,
        id: u32,
    ) -> IoctlResult<v4l2_query_ext_ctrl> {
        let (id, flags) = v4l2r::ioctl::parse_ctrl_id_and_flags(id);
        self.query_ext_ctrl(session, id, flags)
    }
}

/// Writes a `ENOTTY` error response into `writer` to signal that an ioctl is not implemented by
/// the device.
fn invalid_ioctl<W: WriteToDescriptorChain>(code: V4l2Ioctl, writer: &mut W) -> IoResult<()> {
    writer.write_err_response(libc::ENOTTY).map_err(|e| {
        log::error!(
            "failed to write error response for invalid ioctl {:?}: {:#}",
            code,
            e
        );
        e
    })
}

/// The response of `VIDIOC_G/S/TRY_EXT_CTRLS`, success or failure.
///
/// The `v4l2_ext_controls` header and its array of controls are written back **on both paths**.
/// V4L2 requires it: the header carries `error_idx`, "the index of the control causing the
/// error" (`vidioc-g-ext-ctrls.rst`), and it is only ever set on a failure. The driver reads
/// the header back out of the response only when the device wrote enough of one --
/// `virtio_media_send_ext_controls_ioctl()` guards its error branch with
/// `resp_len >= sizeof(resp_ioctl) + sizeof(*ctrls)` -- so a short error response silently
/// loses `error_idx`, which is defect D37. Every error of these three ioctls therefore goes
/// through here, including the ones raised before the device is called at all.
fn ext_ctrls_response(
    ctrls: v4l2_ext_controls,
    ctrl_array: Vec<v4l2_ext_control>,
    result: IoctlResult<()>,
) -> Result<
    (v4l2_ext_controls, Vec<v4l2_ext_control>),
    (i32, Option<(v4l2_ext_controls, Vec<v4l2_ext_control>)>),
> {
    match result {
        Ok(()) => Ok((ctrls, ctrl_array)),
        Err(e) => {
            // The D37 instrument (B9-acceptance §8, follow-up 5). Every error reply of these
            // three ioctls passes here, so this line is the byte count the driver's guard is
            // about to be compared against, next to the `error_idx` the guest is supposed to
            // read out of it. `wr_ioctl_with_err_payload` writes a `RespHeader` and then this
            // pair, and `ToDescriptorChain` writes the header and one `v4l2_ext_control` per
            // entry, so the reply is exactly the sum below -- the arithmetic
            // `a_failed_ext_ctrls_writes_the_header_back_with_error_idx` pins end to end
            // (M5b §4.2). A guest that reads `error_idx` back as the value it sent, against a
            // line that says the reply carried the device's value in >= threshold bytes, has
            // lost it after the response, not in it.
            log::debug!(
                "ext-controls error reply: errno {}, count {}, error_idx {}, {} control(s), \
                 {} + {} + {}x{} = {} bytes (the driver keeps the header only from {} bytes up)",
                e,
                ctrls.count,
                ctrls.error_idx,
                ctrl_array.len(),
                std::mem::size_of::<RespHeader>(),
                std::mem::size_of::<v4l2_ext_controls>(),
                ctrl_array.len(),
                std::mem::size_of::<v4l2_ext_control>(),
                std::mem::size_of::<RespHeader>()
                    + std::mem::size_of::<v4l2_ext_controls>()
                    + ctrl_array.len() * std::mem::size_of::<v4l2_ext_control>(),
                std::mem::size_of::<RespHeader>() + std::mem::size_of::<v4l2_ext_controls>(),
            );
            Err((e, Some((ctrls, ctrl_array))))
        }
    }
}

/// Implements a `WR` ioctl for which errors may also carry a payload.
///
/// * `Reader` is the reader to the device-readable part of the descriptor chain,
/// * `Writer` is the writer to the device-writable part of the descriptor chain,
/// * `I` is the data to be read from the descriptor chain,
/// * `O` is the type of response to be written to the descriptor chain for both success and
///   failure,
/// * `X` processes the input and produces a result. In case of failure, an error code and optional
///   payload to write along with it are returned.
fn wr_ioctl_with_err_payload<Reader, Writer, I, O, X>(
    ioctl: V4l2Ioctl,
    reader: &mut Reader,
    writer: &mut Writer,
    process: X,
) -> IoResult<()>
where
    Reader: ReadFromDescriptorChain,
    Writer: WriteToDescriptorChain,
    I: FromDescriptorChain,
    O: ToDescriptorChain,
    X: FnOnce(I) -> Result<O, (i32, Option<O>)>,
{
    let input = match I::read_from_chain(reader) {
        Ok(input) => input,
        Err(e) => {
            log::error!("error while reading input for {:?} ioctl: {:#}", ioctl, e);
            return writer.write_err_response(libc::EINVAL);
        }
    };

    let (resp_header, output) = match process(input) {
        Ok(output) => (RespHeader::ok(), Some(output)),
        Err((errno, output)) => (RespHeader::err(errno), output),
    };

    writer.write_response(resp_header)?;
    if let Some(output) = output {
        output.write_to_chain(writer)?;
    }

    Ok(())
}

/// Implements a `WR` ioctl for which errors do not carry a payload.
///
/// * `Reader` is the reader to the device-readable part of the descriptor chain,
/// * `Writer` is the writer to the device-writable part of the descriptor chain,
/// * `I` is the data to be read from the descriptor chain,
/// * `O` is the type of response to be written to the descriptor chain in case of success,
/// * `X` processes the input and produces a result. In case of failure, an error code to transmit
///   to the guest is returned.
fn wr_ioctl<Reader, Writer, I, O, X>(
    ioctl: V4l2Ioctl,
    reader: &mut Reader,
    writer: &mut Writer,
    process: X,
) -> IoResult<()>
where
    Reader: ReadFromDescriptorChain,
    Writer: WriteToDescriptorChain,
    I: FromDescriptorChain,
    O: ToDescriptorChain,
    X: FnOnce(I) -> Result<O, i32>,
{
    wr_ioctl_with_err_payload(ioctl, reader, writer, |input| {
        process(input).map_err(|err| (err, None))
    })
}

/// Implements a `W` ioctl.
///
/// * `Reader` is the reader to the device-readable part of the descriptor chain,
/// * `I` is the data to be read from the descriptor chain,
/// * `X` processes the input. In case of failure, an error code to transmit to the guest is
///   returned.
fn w_ioctl<Reader, Writer, I, X>(
    ioctl: V4l2Ioctl,
    reader: &mut Reader,
    writer: &mut Writer,
    process: X,
) -> IoResult<()>
where
    I: FromDescriptorChain,
    Reader: ReadFromDescriptorChain,
    Writer: WriteToDescriptorChain,
    X: FnOnce(I) -> Result<(), i32>,
{
    wr_ioctl(ioctl, reader, writer, process)
}

/// Implements a `R` ioctl.
///
/// * `Writer` is the writer to the device-writable part of the descriptor chain,
/// * `O` is the type of response to be written to the descriptor chain in case of success,
/// * `X` runs the ioctl and produces a result. In case of failure, an error code to transmit to
///   the guest is returned.
fn r_ioctl<Writer, O, X>(ioctl: V4l2Ioctl, writer: &mut Writer, process: X) -> IoResult<()>
where
    Writer: WriteToDescriptorChain,
    O: ToDescriptorChain,
    X: FnOnce() -> Result<O, i32>,
{
    wr_ioctl(ioctl, &mut std::io::empty(), writer, |()| process())
}

/// Ensures that the `readbuffers` and `writebuffers` members of a `v4l2_streamparm` are zero since
/// we do not expose the `READWRITE` capability.
fn patch_streamparm(mut parm: v4l2_streamparm) -> v4l2_streamparm {
    match QueueType::n(parm.type_)
        .unwrap_or(QueueType::VideoCapture)
        .direction()
    {
        QueueDirection::Output => parm.parm.output.writebuffers = 0,
        QueueDirection::Capture => parm.parm.capture.readbuffers = 0,
    }

    parm
}

/// IOCTL dispatcher for implementors of [`VirtioMediaIoctlHandler`].
///
/// This function takes care of reading and validating IOCTL inputs and writing outputs or errors
/// back to the driver, invoking the relevant method of the handler in the middle.
///
/// Implementors of [`VirtioMediaIoctlHandler`] can thus just focus on writing the desired behavior
/// for their device, and let the more tedious parsing and validation to this function.
pub fn virtio_media_dispatch_ioctl<S, H, Reader, Writer>(
    handler: &mut H,
    session: &mut S,
    ioctl: V4l2Ioctl,
    reader: &mut Reader,
    writer: &mut Writer,
) -> IoResult<()>
where
    H: VirtioMediaIoctlHandler<Session = S>,
    Reader: ReadFromDescriptorChain,
    Writer: WriteToDescriptorChain,
{
    use V4l2Ioctl::*;

    match ioctl {
        VIDIOC_QUERYCAP => invalid_ioctl(ioctl, writer),
        VIDIOC_ENUM_FMT => wr_ioctl(ioctl, reader, writer, |format: v4l2_fmtdesc| {
            let queue = QueueType::n(format.type_).ok_or(libc::EINVAL)?;
            handler.enum_fmt(session, queue, format.index)
        }),
        VIDIOC_G_FMT => wr_ioctl(ioctl, reader, writer, |format: v4l2_format| {
            let queue = QueueType::n(format.type_).ok_or(libc::EINVAL)?;
            handler.g_fmt(session, queue)
        }),
        VIDIOC_S_FMT => wr_ioctl(ioctl, reader, writer, |format: v4l2_format| {
            let queue = QueueType::n(format.type_).ok_or(libc::EINVAL)?;
            handler.s_fmt(session, queue, format)
        }),
        VIDIOC_REQBUFS => wr_ioctl(ioctl, reader, writer, |reqbufs: v4l2_requestbuffers| {
            let queue = QueueType::n(reqbufs.type_).ok_or(libc::EINVAL)?;
            let memory = MemoryType::n(reqbufs.memory).ok_or(libc::EINVAL)?;

            match memory {
                MemoryType::Mmap | MemoryType::UserPtr => (),
                t => {
                    log::error!(
                        "VIDIOC_REQBUFS: memory type {:?} is currently unsupported",
                        t
                    );
                    return Err(libc::EINVAL);
                }
            }

            handler.reqbufs(session, queue, memory, reqbufs.count)
        }),
        VIDIOC_QUERYBUF => {
            wr_ioctl(ioctl, reader, writer, |buffer: v4l2_buffer| {
                let queue = QueueType::n(buffer.type_).ok_or(libc::EINVAL)?;
                // Maximum number of planes we can write back to the driver.
                let num_planes = if queue.is_multiplanar() {
                    buffer.length as usize
                } else {
                    0
                };

                handler
                    .querybuf(session, queue, buffer.index)
                    .map(|guest_buffer| (guest_buffer, num_planes))
            })
        }
        VIDIOC_G_FBUF => invalid_ioctl(ioctl, writer),
        VIDIOC_S_FBUF => invalid_ioctl(ioctl, writer),
        VIDIOC_OVERLAY => invalid_ioctl(ioctl, writer),
        VIDIOC_QBUF => wr_ioctl(ioctl, reader, writer, |input: GuestV4l2Buffer| {
            let num_planes = input.buffer.num_planes();

            handler
                .qbuf(session, input.buffer, input.guest_regions, input.payload)
                .map(|guest_buffer| (guest_buffer, num_planes))
        }),
        // TODO implement EXPBUF.
        VIDIOC_EXPBUF => invalid_ioctl(ioctl, writer),
        VIDIOC_DQBUF => invalid_ioctl(ioctl, writer),
        VIDIOC_STREAMON => w_ioctl(ioctl, reader, writer, |input: u32| {
            let queue = QueueType::n(input).ok_or(libc::EINVAL)?;

            handler.streamon(session, queue)
        }),
        VIDIOC_STREAMOFF => w_ioctl(ioctl, reader, writer, |input: u32| {
            let queue = QueueType::n(input).ok_or(libc::EINVAL)?;

            handler.streamoff(session, queue)
        }),
        VIDIOC_G_PARM => wr_ioctl(ioctl, reader, writer, |parm: v4l2_streamparm| {
            let queue = QueueType::n(parm.type_).ok_or(libc::EINVAL)?;

            handler.g_parm(session, queue).map(patch_streamparm)
        }),
        VIDIOC_S_PARM => wr_ioctl(ioctl, reader, writer, |parm: v4l2_streamparm| {
            handler
                .s_parm(session, patch_streamparm(parm))
                .map(patch_streamparm)
        }),
        VIDIOC_G_STD => r_ioctl(ioctl, writer, || handler.g_std(session)),
        VIDIOC_S_STD => w_ioctl(ioctl, reader, writer, |id: v4l2_std_id| {
            handler.s_std(session, id)
        }),
        VIDIOC_ENUMSTD => wr_ioctl(ioctl, reader, writer, |std: v4l2_standard| {
            handler.enumstd(session, std.index)
        }),
        VIDIOC_ENUMINPUT => wr_ioctl(ioctl, reader, writer, |input: v4l2_input| {
            handler.enuminput(session, input.index)
        }),
        VIDIOC_G_CTRL => wr_ioctl(ioctl, reader, writer, |ctrl: v4l2_control| {
            handler.g_ctrl(session, ctrl.id)
        }),
        VIDIOC_S_CTRL => wr_ioctl(ioctl, reader, writer, |ctrl: v4l2_control| {
            handler.s_ctrl(session, ctrl.id, ctrl.value)
        }),
        VIDIOC_G_TUNER => wr_ioctl(ioctl, reader, writer, |tuner: v4l2_tuner| {
            handler.g_tuner(session, tuner.index)
        }),
        VIDIOC_S_TUNER => w_ioctl(ioctl, reader, writer, |tuner: v4l2_tuner| {
            let mode = TunerMode::n(tuner.audmode).ok_or(libc::EINVAL)?;
            handler.s_tuner(session, tuner.index, mode)
        }),
        VIDIOC_G_AUDIO => r_ioctl(ioctl, writer, || handler.g_audio(session)),
        VIDIOC_S_AUDIO => w_ioctl(ioctl, reader, writer, |input: v4l2_audio| {
            handler.s_audio(session, input.index, AudioMode::n(input.mode))
        }),
        VIDIOC_QUERYCTRL => wr_ioctl(ioctl, reader, writer, |input: v4l2_queryctrl| {
            handler.queryctrl_raw(session, input.id)
        }),
        VIDIOC_QUERYMENU => wr_ioctl(ioctl, reader, writer, |input: v4l2_querymenu| {
            handler.querymenu(session, input.id, input.index)
        }),
        VIDIOC_G_INPUT => r_ioctl(ioctl, writer, || handler.g_input(session)),
        VIDIOC_S_INPUT => wr_ioctl(ioctl, reader, writer, |input: i32| {
            handler.s_input(session, input)
        }),
        VIDIOC_G_EDID => invalid_ioctl(ioctl, writer),
        VIDIOC_S_EDID => invalid_ioctl(ioctl, writer),
        VIDIOC_G_OUTPUT => r_ioctl(ioctl, writer, || handler.g_output(session)),
        VIDIOC_S_OUTPUT => wr_ioctl(ioctl, reader, writer, |output: i32| {
            handler.s_output(session, output)
        }),
        VIDIOC_ENUMOUTPUT => wr_ioctl(ioctl, reader, writer, |output: v4l2_output| {
            handler.enumoutput(session, output.index)
        }),
        VIDIOC_G_AUDOUT => r_ioctl(ioctl, writer, || handler.g_audout(session)),
        VIDIOC_S_AUDOUT => w_ioctl(ioctl, reader, writer, |audout: v4l2_audioout| {
            handler.s_audout(session, audout.index)
        }),
        VIDIOC_G_MODULATOR => wr_ioctl(ioctl, reader, writer, |modulator: v4l2_modulator| {
            handler.g_modulator(session, modulator.index)
        }),
        VIDIOC_S_MODULATOR => w_ioctl(ioctl, reader, writer, |modulator: v4l2_modulator| {
            let flags =
                TunerTransmissionFlags::from_bits(modulator.txsubchans).ok_or(libc::EINVAL)?;
            handler.s_modulator(session, modulator.index, flags)
        }),
        VIDIOC_G_FREQUENCY => wr_ioctl(ioctl, reader, writer, |freq: v4l2_frequency| {
            handler.g_frequency(session, freq.tuner)
        }),
        VIDIOC_S_FREQUENCY => w_ioctl(ioctl, reader, writer, |freq: v4l2_frequency| {
            let type_ = TunerType::n(freq.type_).ok_or(libc::EINVAL)?;

            handler.s_frequency(session, freq.tuner, type_, freq.frequency)
        }),
        // TODO do these 3 need to be supported?
        VIDIOC_CROPCAP => invalid_ioctl(ioctl, writer),
        VIDIOC_G_CROP => invalid_ioctl(ioctl, writer),
        VIDIOC_S_CROP => invalid_ioctl(ioctl, writer),
        // Deprecated in V4L2.
        VIDIOC_G_JPEGCOMP => invalid_ioctl(ioctl, writer),
        // Deprecated in V4L2.
        VIDIOC_S_JPEGCOMP => invalid_ioctl(ioctl, writer),
        VIDIOC_QUERYSTD => r_ioctl(ioctl, writer, || handler.querystd(session)),
        VIDIOC_TRY_FMT => wr_ioctl(ioctl, reader, writer, |format: v4l2_format| {
            let queue = QueueType::n(format.type_).ok_or(libc::EINVAL)?;
            handler.try_fmt(session, queue, format)
        }),
        VIDIOC_ENUMAUDIO => wr_ioctl(ioctl, reader, writer, |audio: v4l2_audio| {
            handler.enumaudio(session, audio.index)
        }),
        VIDIOC_ENUMAUDOUT => wr_ioctl(ioctl, reader, writer, |audio: v4l2_audioout| {
            handler.enumaudout(session, audio.index)
        }),
        VIDIOC_G_PRIORITY => invalid_ioctl(ioctl, writer),
        VIDIOC_S_PRIORITY => invalid_ioctl(ioctl, writer),
        // TODO support this, although it's marginal.
        VIDIOC_G_SLICED_VBI_CAP => invalid_ioctl(ioctl, writer),
        // Doesn't make sense in a virtual context.
        VIDIOC_LOG_STATUS => invalid_ioctl(ioctl, writer),
        VIDIOC_G_EXT_CTRLS => wr_ioctl_with_err_payload(
            ioctl,
            reader,
            writer,
            |(mut ctrls, mut ctrl_array, user_regions)| {
                let result = match CtrlWhich::try_from(&ctrls) {
                    Ok(which) => handler.g_ext_ctrls(
                        session,
                        which,
                        &mut ctrls,
                        &mut ctrl_array,
                        user_regions,
                    ),
                    // A `which` no control set can be read under: `error_idx` is `count` for a
                    // get, as `v4l2_g_ext_ctrls_common` leaves it after `prepare_ext_ctrls`.
                    Err(()) => {
                        ctrls.error_idx = ctrls.count;
                        Err(libc::EINVAL)
                    }
                };
                ext_ctrls_response(ctrls, ctrl_array, result)
            },
        ),
        VIDIOC_S_EXT_CTRLS => wr_ioctl_with_err_payload(
            ioctl,
            reader,
            writer,
            |(mut ctrls, mut ctrl_array, user_regions)| {
                let result = match CtrlWhich::try_from(&ctrls) {
                    Ok(which) => handler.s_ext_ctrls(
                        session,
                        which,
                        &mut ctrls,
                        &mut ctrl_array,
                        user_regions,
                    ),
                    // `error_idx` is `count` for a refused set (`try_set_ext_ctrls_common`
                    // puts it back to `count` whenever `set` failed).
                    Err(()) => {
                        ctrls.error_idx = ctrls.count;
                        Err(libc::EINVAL)
                    }
                };
                ext_ctrls_response(ctrls, ctrl_array, result)
            },
        ),
        VIDIOC_TRY_EXT_CTRLS => wr_ioctl_with_err_payload(
            ioctl,
            reader,
            writer,
            |(mut ctrls, mut ctrl_array, user_regions)| {
                let result = match CtrlWhich::try_from(&ctrls) {
                    Ok(which) => handler.try_ext_ctrls(
                        session,
                        which,
                        &mut ctrls,
                        &mut ctrl_array,
                        user_regions,
                    ),
                    // A try names the control it stopped at, and it stopped at the first.
                    Err(()) => {
                        ctrls.error_idx = 0;
                        Err(libc::EINVAL)
                    }
                };
                ext_ctrls_response(ctrls, ctrl_array, result)
            },
        ),
        VIDIOC_ENUM_FRAMESIZES => {
            wr_ioctl(ioctl, reader, writer, |frmsizeenum: v4l2_frmsizeenum| {
                handler.enum_framesizes(session, frmsizeenum.index, frmsizeenum.pixel_format)
            })
        }
        VIDIOC_ENUM_FRAMEINTERVALS => {
            wr_ioctl(ioctl, reader, writer, |frmivalenum: v4l2_frmivalenum| {
                handler.enum_frameintervals(
                    session,
                    frmivalenum.index,
                    frmivalenum.pixel_format,
                    frmivalenum.width,
                    frmivalenum.height,
                )
            })
        }
        VIDIOC_G_ENC_INDEX => r_ioctl(ioctl, writer, || handler.g_enc_index(session)),
        VIDIOC_ENCODER_CMD => wr_ioctl(ioctl, reader, writer, |cmd: v4l2_encoder_cmd| {
            handler.encoder_cmd(session, cmd)
        }),
        VIDIOC_TRY_ENCODER_CMD => wr_ioctl(ioctl, reader, writer, |cmd: v4l2_encoder_cmd| {
            handler.try_encoder_cmd(session, cmd)
        }),
        // Doesn't make sense in a virtual context.
        VIDIOC_DBG_G_REGISTER => invalid_ioctl(ioctl, writer),
        // Doesn't make sense in a virtual context.
        VIDIOC_DBG_S_REGISTER => invalid_ioctl(ioctl, writer),
        VIDIOC_S_HW_FREQ_SEEK => invalid_ioctl(ioctl, writer),
        VIDIOC_S_DV_TIMINGS => wr_ioctl(ioctl, reader, writer, |timings: v4l2_dv_timings| {
            handler.s_dv_timings(session, timings)
        }),
        VIDIOC_G_DV_TIMINGS => wr_ioctl(
            ioctl,
            reader,
            writer,
            // We are not using the input - this should probably have been a R ioctl?
            |_: v4l2_dv_timings| handler.g_dv_timings(session),
        ),
        // Supported by an event.
        VIDIOC_DQEVENT => invalid_ioctl(ioctl, writer),
        VIDIOC_SUBSCRIBE_EVENT => {
            w_ioctl(ioctl, reader, writer, |input: v4l2_event_subscription| {
                // Both come straight from the guest: an event type v4l2r does not know, or a
                // flag bit it does not define, is the guest's mistake and gets `EINVAL`, not a
                // panic of the device thread (`VPU_DESIGN.md` §1.10).
                let event = V4l2EventType::try_from(&input).map_err(|_| libc::EINVAL)?;
                let flags = SubscribeEventFlags::from_bits(input.flags).ok_or(libc::EINVAL)?;

                handler.subscribe_event(session, event, flags)
            })?;

            Ok(())
        }
        VIDIOC_UNSUBSCRIBE_EVENT => {
            w_ioctl(ioctl, reader, writer, |event: v4l2_event_subscription| {
                handler.unsubscribe_event(session, event)
            })
        }
        VIDIOC_CREATE_BUFS => wr_ioctl(ioctl, reader, writer, |input: v4l2_create_buffers| {
            let queue = QueueType::n(input.format.type_).ok_or(libc::EINVAL)?;
            let memory = MemoryType::n(input.memory).ok_or(libc::EINVAL)?;

            handler.create_bufs(session, input.count, queue, memory, input.format)
        }),
        VIDIOC_PREPARE_BUF => wr_ioctl(ioctl, reader, writer, |input: GuestV4l2Buffer| {
            let num_planes = input.buffer.num_planes();

            handler
                .prepare_buf(session, input.buffer, input.guest_regions, input.payload)
                .map(|out_buffer| (out_buffer, num_planes))
        }),
        VIDIOC_G_SELECTION => wr_ioctl(ioctl, reader, writer, |mut selection: v4l2_selection| {
            let sel_type = SelectionType::n(selection.type_).ok_or(libc::EINVAL)?;
            let sel_target = SelectionTarget::n(selection.target).ok_or(libc::EINVAL)?;

            handler
                .g_selection(session, sel_type, sel_target)
                .map(|rect| {
                    selection.r = rect;
                    selection
                })
        }),
        VIDIOC_S_SELECTION => wr_ioctl(ioctl, reader, writer, |mut selection: v4l2_selection| {
            let sel_type = SelectionType::n(selection.type_).ok_or(libc::EINVAL)?;
            let sel_target = SelectionTarget::n(selection.target).ok_or(libc::EINVAL)?;
            let sel_flags = SelectionFlags::from_bits(selection.flags).ok_or(libc::EINVAL)?;

            handler
                .s_selection(session, sel_type, sel_target, selection.r, sel_flags)
                .map(|rect| {
                    selection.r = rect;
                    selection
                })
        }),
        VIDIOC_DECODER_CMD => wr_ioctl(ioctl, reader, writer, |cmd: v4l2_decoder_cmd| {
            handler.decoder_cmd(session, cmd)
        }),
        VIDIOC_TRY_DECODER_CMD => wr_ioctl(ioctl, reader, writer, |cmd: v4l2_decoder_cmd| {
            handler.try_decoder_cmd(session, cmd)
        }),
        VIDIOC_ENUM_DV_TIMINGS => wr_ioctl(
            ioctl,
            reader,
            writer,
            |mut enum_timings: v4l2_enum_dv_timings| {
                handler
                    .enum_dv_timings(session, enum_timings.index)
                    .map(|timings| {
                        enum_timings.timings = timings;
                        enum_timings
                    })
            },
        ),
        VIDIOC_QUERY_DV_TIMINGS => r_ioctl(ioctl, writer, || handler.query_dv_timings(session)),
        VIDIOC_DV_TIMINGS_CAP => wr_ioctl(ioctl, reader, writer, |_: v4l2_dv_timings_cap| {
            handler.dv_timings_cap(session)
        }),
        VIDIOC_ENUM_FREQ_BANDS => {
            wr_ioctl(ioctl, reader, writer, |freq_band: v4l2_frequency_band| {
                let type_ = TunerType::n(freq_band.type_).ok_or(libc::EINVAL)?;

                handler.enum_freq_bands(session, freq_band.tuner, type_, freq_band.index)
            })
        }
        // Doesn't make sense in a virtual context.
        VIDIOC_DBG_G_CHIP_INFO => invalid_ioctl(ioctl, writer),
        VIDIOC_QUERY_EXT_CTRL => wr_ioctl(ioctl, reader, writer, |ctrl: v4l2_query_ext_ctrl| {
            handler.query_ext_ctrl_raw(session, ctrl.id)
        }),
    }
}

/// The wire bytes an ffmpeg-shaped client puts on a `QBUF`, shared by every device's tests.
///
/// It is one dump, replayed in five places, because the bug it pins (D21) was in the shared
/// reader and hit all of them at once.
#[cfg(test)]
pub(crate) mod ffmpeg_wire {
    use zerocopy::AsBytes;

    use super::*;
    use crate::io::VmediaType;

    /// The dirty stack `ffmpeg -f v4l2` leaves in `planes[1..8]`, captured from the guest by an
    /// `LD_PRELOAD` shim on 5566 (`logs/vpu_wp/B5-acceptance.md` §4.3, `scratch-b5/qbufspy.c`).
    /// ffmpeg declares `length = VIDEO_MAX_PLANES` and fills `planes[0]` from `QUERYBUF`; the
    /// rest is whatever was on its stack. The dump elided slots 5..7 as "likewise", so they
    /// repeat the shape of 3 and 4.
    ///
    /// `(bytesused, length, data_offset)`. Slot 2's triple happens to be self-consistent, which
    /// is exactly why the reader must judge the slots one at a time rather than as a block.
    const FFMPEG_DIRTY_TAIL: [(u32, u32, u32); 7] = [
        (1, 3425573041, 119),
        (119, 3750202224, 1),
        (3019899000, 912, 3133457408),
        (2626962592, 44344, 4294967295),
        (3019899000, 912, 3133457408),
        (2626962592, 44344, 4294967295),
        (3019899000, 912, 3133457408),
    ];

    /// One `QBUF` on the wire in ffmpeg's shape: `length = VIDEO_MAX_PLANES`, `planes[0]` as
    /// `QUERYBUF` reported it, `planes[1..8]` uninitialised.
    pub(crate) fn ffmpeg_qbuf_bytes(
        queue: QueueType,
        memory: MemoryType,
        index: u32,
        plane0: (u32, u32),
    ) -> Vec<u8> {
        let buffer = v4l2_buffer {
            index,
            type_: queue as u32,
            memory: memory as u32,
            length: v4l2r::bindings::VIDEO_MAX_PLANES,
            ..Default::default()
        };
        let mut out = Vec::new();
        out.extend_from_slice(buffer.to_le().as_bytes());
        let (bytesused, length) = plane0;
        let mut planes = vec![v4l2_plane {
            bytesused,
            length,
            ..Default::default()
        }];
        for &(bytesused, length, data_offset) in FFMPEG_DIRTY_TAIL.iter() {
            planes.push(v4l2_plane {
                bytesused,
                length,
                data_offset,
                ..Default::default()
            });
        }
        for plane in planes {
            out.extend_from_slice(plane.to_le().as_bytes());
        }
        out
    }

    /// Runs one `QBUF` of those bytes through the shared reader into `handler`, and returns the
    /// errno the guest would see (0 on success).
    pub(crate) fn dispatch_qbuf<H: VirtioMediaIoctlHandler>(
        handler: &mut H,
        session: &mut H::Session,
        bytes: &[u8],
    ) -> i32 {
        let mut out = Vec::new();
        virtio_media_dispatch_ioctl(
            handler,
            session,
            V4l2Ioctl::VIDIOC_QBUF,
            &mut &bytes[..],
            &mut out,
        )
        .unwrap();
        i32::from_le_bytes(out[0..4].try_into().unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::VmediaType;

    /// Serialize SG entries the way the guest puts them on the descriptor chain.
    fn sg_bytes(entries: &[(u64, u32)]) -> Vec<u8> {
        use zerocopy::AsBytes;
        let mut out = Vec::new();
        for &(start, len) in entries {
            out.extend_from_slice(SgEntry::new(start, len).to_le().as_bytes());
        }
        out
    }

    #[test]
    fn userptr_regions_are_read_up_to_size() {
        let bytes = sg_bytes(&[(0x1000, 0x1000), (0x3000, 0x800)]);
        let regions = get_userptr_regions(&mut &bytes[..], 0x1800).unwrap();
        assert_eq!(regions.len(), 2);
        assert_eq!((regions[0].start, regions[0].len), (0x1000, 0x1000));
        assert_eq!((regions[1].start, regions[1].len), (0x3000, 0x800));
    }

    #[test]
    fn zero_length_sg_entry_is_refused_instead_of_spinning() {
        let bytes = sg_bytes(&[(0x1000, 0)]);
        assert!(get_userptr_regions(&mut &bytes[..], 0x1000).is_err());
    }

    /// A device that implements no ioctl at all: everything falls through to the trait's
    /// defaults, as `simple_device`, `video_decoder` and a future camera device do for
    /// `PREPARE_BUF`.
    struct NoopHandler;

    impl VirtioMediaIoctlHandler for NoopHandler {
        type Session = ();
    }

    /// One `v4l2_buffer` on the wire: MPLANE OUTPUT, MMAP, one plane with `bytesused`/`length`.
    fn mplane_buffer_bytes(bytesused: u32, length: u32) -> Vec<u8> {
        use zerocopy::AsBytes;
        let buffer = v4l2_buffer {
            index: 0,
            type_: QueueType::VideoOutputMplane as u32,
            memory: MemoryType::Mmap as u32,
            length: 1,
            ..Default::default()
        };
        let plane = v4l2_plane {
            bytesused,
            length,
            ..Default::default()
        };
        let mut out = Vec::new();
        out.extend_from_slice(buffer.to_le().as_bytes());
        out.extend_from_slice(plane.to_le().as_bytes());
        out
    }

    fn dispatch_prepare_buf(bytes: &[u8]) -> i32 {
        let mut out = Vec::new();
        virtio_media_dispatch_ioctl(
            &mut NoopHandler,
            &mut (),
            V4l2Ioctl::VIDIOC_PREPARE_BUF,
            &mut &bytes[..],
            &mut out,
        )
        .unwrap();
        i32::from_le_bytes(out[0..4].try_into().unwrap())
    }

    /// An ioctl a device does not implement answers `ENOTTY` whatever the guest sent.
    ///
    /// The payload used to be validated before the handler was asked, so a buffer whose
    /// `bytesused` overflows its `length` came back `EINVAL` from a device that has no
    /// `PREPARE_BUF` at all -- and `v4l2-compliance` reads that first `EINVAL` as "the ioctl
    /// exists", then fails when the well-formed call answers `ENOTTY` (defect D6).
    #[test]
    fn an_unimplemented_ioctl_is_enotty_for_a_malformed_payload_too() {
        assert_eq!(
            dispatch_prepare_buf(&mplane_buffer_bytes(0, 4096)),
            libc::ENOTTY
        );
        // `bytesused > length`: refused by `V4l2Buffer`'s invariants, sanitised by the reader.
        assert_eq!(
            dispatch_prepare_buf(&mplane_buffer_bytes(4097, 4096)),
            libc::ENOTTY
        );
        // A buffer that is not addressable at all is still `EINVAL`: there is no ioctl to run.
        let mut broken = mplane_buffer_bytes(0, 4096);
        broken[4..8].copy_from_slice(&0xdead_beefu32.to_le_bytes()); // `type_`
        assert_eq!(dispatch_prepare_buf(&broken), libc::EINVAL);
    }

    /// One `v4l2_buffer` on the wire declaring a plane array of `length` entries: the first a
    /// real 4096-byte plane, the rest the zeroed tail a multiplanar client leaves behind it.
    fn mplane_buffer_with_plane_array(length: u32) -> Vec<u8> {
        use zerocopy::AsBytes;
        let buffer = v4l2_buffer {
            index: 0,
            type_: QueueType::VideoOutputMplane as u32,
            memory: MemoryType::Mmap as u32,
            length,
            ..Default::default()
        };
        let mut out = Vec::new();
        out.extend_from_slice(buffer.to_le().as_bytes());
        for i in 0..length {
            let plane = v4l2_plane {
                length: if i == 0 { 4096 } else { 0 },
                ..Default::default()
            };
            out.extend_from_slice(plane.to_le().as_bytes());
        }
        out
    }

    /// `length` is the size of the guest's plane array, not the number of planes the format
    /// uses, and V4L2 allows exactly `VIDEO_MAX_PLANES` entries -- which is what ffmpeg puts on
    /// every multiplanar buffer ioctl (`libavdevice/v4l2.c`). The vendored v4l2r refused that
    /// value (`>=` where the kernel's `__verify_planes_array` has `<=`), so a legal `QBUF` came
    /// back `EINVAL` from the shared reader and `ffmpeg -f v4l2` could not capture at all
    /// (defect D17). The reader here has always allowed it; this pins the pair, since the check
    /// that broke lives in the dependency and every device inherits it.
    #[test]
    fn a_plane_array_of_video_max_planes_reaches_the_device() {
        for length in 1..=v4l2r::bindings::VIDEO_MAX_PLANES {
            assert_eq!(
                dispatch_prepare_buf(&mplane_buffer_with_plane_array(length)),
                libc::ENOTTY,
                "a plane array of {} entries never reached the handler",
                length
            );
        }
        // One entry more than a `V4l2Buffer` can hold is still refused by the reader, before any
        // handler is asked: there is no buffer to run an ioctl on.
        assert_eq!(
            dispatch_prepare_buf(&mplane_buffer_with_plane_array(
                v4l2r::bindings::VIDEO_MAX_PLANES + 1
            )),
            libc::EINVAL
        );
    }

    /// The reader judges each plane slot on its own, so a device can apply its queue's plane
    /// count to the answer.
    ///
    /// This is defect D21: ffmpeg sends `length = VIDEO_MAX_PLANES` with `planes[1..8]`
    /// uninitialised, the reader called the whole payload invalid, and every device refused the
    /// buffer -- so `ffmpeg -f v4l2 -i /dev/video0` could not queue one frame on a queue whose
    /// format has a single plane (`logs/vpu_wp/B5-acceptance.md` §4.3 and §6). vb2 looks at
    /// `vb->num_planes` entries and no further (`__verify_length`,
    /// `GKI_6.18-2026-06_r11/drivers/media/common/videobuf2/videobuf2-v4l2.c:105`).
    #[test]
    fn each_plane_slot_is_judged_on_its_own() {
        let bytes = ffmpeg_wire::ffmpeg_qbuf_bytes(
            QueueType::VideoCaptureMplane,
            MemoryType::Mmap,
            0,
            (0, 1382400),
        );
        let read = GuestV4l2Buffer::read_from_chain(&mut &bytes[..]).unwrap();

        // The plane the format actually has was taken as sent, so a one-plane queue accepts it.
        assert!(read.payload.plane_is_valid(0));
        assert!(read.payload.planes_are_valid(1));
        assert_eq!(*read.buffer.get_first_plane().length, 1382400);
        // The dirty tail: slots 1, 3 and 4 (and 5..7, which repeat them) are not descriptions
        // this buffer could hold, slot 2's triple happens to be self-consistent.
        for plane in [1, 3, 4, 5, 6, 7] {
            assert!(!read.payload.plane_is_valid(plane), "slot {plane}");
        }
        assert!(read.payload.plane_is_valid(2));
        assert!(
            !read.payload.planes_are_valid(2),
            "slot 1 is in the first two"
        );
        assert_ne!(read.payload, PayloadValidity::ALL);

        // Only the slots it could not represent were zeroed; plane 0's numbers survive, which is
        // what an output queue validates as the bitstream length.
        let planes = read.buffer.as_v4l2_planes();
        assert_eq!((planes[0].bytesused, planes[0].length), (0, 1382400));
        assert_eq!((planes[1].bytesused, planes[1].data_offset), (0, 0));
        assert_eq!((planes[2].bytesused, planes[2].data_offset), (119, 1));
        assert_eq!((planes[3].bytesused, planes[3].data_offset), (0, 0));

        // Garbage in plane 0 is the one an output queue must still refuse.
        let bad = ffmpeg_wire::ffmpeg_qbuf_bytes(
            QueueType::VideoOutputMplane,
            MemoryType::Mmap,
            0,
            (4097, 4096),
        );
        let read = GuestV4l2Buffer::read_from_chain(&mut &bad[..]).unwrap();
        assert!(!read.payload.plane_is_valid(0));
        assert!(!read.payload.planes_are_valid(1));
        assert_eq!(*read.buffer.get_first_plane().bytesused, 0);
    }

    /// The rule the devices apply to what the reader reports, both halves of it.
    #[test]
    fn payload_validity_follows_verify_length() {
        let dirty_tail = PayloadValidity::ALL.without(1).without(3);
        // A one-plane format never looks past slot 0.
        assert!(dirty_tail.is_accepted_by(QueueDirection::Output, MemoryType::UserPtr, 1));
        assert!(!dirty_tail.is_accepted_by(QueueDirection::Output, MemoryType::UserPtr, 2));
        // Plane 0 is the bitstream length on an output queue, whatever the memory type.
        let bad_plane_0 = PayloadValidity::ALL.without(0);
        for memory in [MemoryType::Mmap, MemoryType::UserPtr] {
            assert!(!bad_plane_0.is_accepted_by(QueueDirection::Output, memory, 1));
        }
        // A capture `MMAP` buffer's payload is the device's to report, so the guest's is ignored
        // (`__verify_length` returns before it looks, `videobuf2-v4l2.c:101`).
        assert!(bad_plane_0.is_accepted_by(QueueDirection::Capture, MemoryType::Mmap, 1));
        // A `USERPTR` capture buffer is held to the stricter rule these devices keep.
        assert!(!bad_plane_0.is_accepted_by(QueueDirection::Capture, MemoryType::UserPtr, 1));
        // A count no plane array could satisfy is never valid.
        assert!(!PayloadValidity::ALL.is_accepted_by(QueueDirection::Output, MemoryType::Mmap, 9));
    }

    /// A multiplanar buffer with an empty plane array is not a buffer: vb2 wants
    /// `num_planes <= length` and a format has at least one plane, and `V4l2Buffer`'s
    /// `get_first_plane` would `unwrap` an empty iterator on it -- an abort, in a VMM built with
    /// `panic = 'abort'` (`logs/vpu_wp/F6.md` §10). The reader refuses it before any device can
    /// reach for a plane.
    #[test]
    fn a_multiplanar_buffer_with_no_plane_is_refused_by_the_reader() {
        use zerocopy::AsBytes;
        let buffer = v4l2_buffer {
            index: 0,
            type_: QueueType::VideoCaptureMplane as u32,
            memory: MemoryType::Mmap as u32,
            length: 0,
            ..Default::default()
        };
        let bytes = buffer.to_le().as_bytes().to_vec();
        assert!(GuestV4l2Buffer::read_from_chain(&mut &bytes[..]).is_err());
        assert_eq!(dispatch_prepare_buf(&bytes), libc::EINVAL);

        // A single-planar buffer has no plane array to be empty, and is unaffected.
        let single = v4l2_buffer {
            index: 0,
            type_: QueueType::VideoCapture as u32,
            memory: MemoryType::Mmap as u32,
            length: 0,
            ..Default::default()
        };
        let bytes = single.to_le().as_bytes().to_vec();
        assert!(GuestV4l2Buffer::read_from_chain(&mut &bytes[..]).is_ok());
    }

    #[test]
    fn sg_list_is_capped() {
        let entries: Vec<(u64, u32)> = (0..(MAX_SG_ENTRIES as u64 + 1))
            .map(|i| (i * 0x1000, 0x1000))
            .collect();
        let bytes = sg_bytes(&entries);
        let too_much = (MAX_SG_ENTRIES + 1) * 0x1000;
        assert!(get_userptr_regions(&mut &bytes[..], too_much).is_err());
        let just_enough = MAX_SG_ENTRIES * 0x1000;
        assert_eq!(
            get_userptr_regions(&mut &bytes[..], just_enough)
                .unwrap()
                .len(),
            MAX_SG_ENTRIES
        );
    }
}
