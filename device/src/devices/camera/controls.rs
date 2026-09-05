// Copyright 2026 The DroidVM Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

//! The camera device's V4L2 controls (`VPU_DESIGN.md` §7.1, `VIRTIO_MEDIA_PLAN.md` §3.1).
//!
//! A control exists only when the camera's characteristics say the feature does: [`Controls::new`]
//! builds the table from a [`CameraInfo`] once per device, in ascending id order, which is the
//! order `QUERY_EXT_CTRL` with `V4L2_CTRL_FLAG_NEXT_*` walks it in. The *values* are the
//! device's, not a session's: V4L2 controls belong to the video node, `v4l2-ctl --set-ctrl` from
//! one process must reach a stream another process runs, and a value set with no stream open is
//! applied when one opens. What is per session is the `V4L2_EVENT_CTRL` subscription, which the
//! device keeps by session id.
//!
//! # Numeric conventions (guest-visible ABI, design §7.1)
//!
//! * `V4L2_CID_ZOOM_ABSOLUTE` is the zoom ratio in hundredths: 67 is 0.67x, 100 is 1x, 2000 is
//!   20x. On a logical multi-camera the ratio is what makes the platform switch lenses.
//! * `V4L2_CID_EXPOSURE_ABSOLUTE` is in V4L2's standard unit, 100 µs; a camera whose shortest
//!   exposure is under that reports a minimum of 1.
//! * `V4L2_CID_ISO_SENSITIVITY` is an `INTEGER_MENU` synthesised from the camera's continuous
//!   sensitivity range: the range's minimum, then 100, 200, 400, ... doubling while inside the
//!   range, then the range's maximum. The control's value is the menu *index*; `QUERYMENU` gives
//!   the ISO number.
//! * `V4L2_CID_AUTO_EXPOSURE_BIAS` is an `INTEGER_MENU` with one item per exposure-compensation
//!   step the camera offers, each item being that step's value in 0.001 EV as V4L2 defines.
//! * Manual exposure is one switch in Camera2 (`AE_MODE` off) and two in V4L2
//!   (`EXPOSURE_AUTO = Manual`, `ISO_SENSITIVITY_AUTO = Manual`): either one turns the camera's
//!   auto-exposure off, and then *both* `EXPOSURE_ABSOLUTE` and `ISO_SENSITIVITY` apply. While
//!   neither is manual the two carry `V4L2_CTRL_FLAG_INACTIVE`, as a V4L2 auto cluster does.
//! * `V4L2_CID_FLASH_LED_MODE` offers `Off` and `Torch`; `Flash` (fire on strobe) has no Camera2
//!   equivalent that fits a repeating request and is a skipped menu item (plan §3.1).
//! * `V4L2_CID_FOCUS_AUTO` set is continuous autofocus; clear is the camera's single-shot mode, in
//!   which `V4L2_CID_AUTO_FOCUS_START` scans once and `V4L2_CID_AUTO_FOCUS_STOP` cancels.
//!   `V4L2_CID_AUTO_FOCUS_STATUS` (read-only, volatile) is the scan's state.
//!
//! # The private block: `V4L2_CID_USER_BASE + 0x1200`
//!
//! Private controls live in the USER class at `V4L2_CID_USER_BASE + 0x1200`, a block of sixteen
//! in the convention of `v4l2-controls.h` (one block per driver, MEYE at +0x1000 up); the ids
//! are `V4L2_CTRL_DRIVER_PRIV` and the guest's V4L2 core also reaches the integer-valued ones
//! through the old `V4L2_CID_PRIVATE_BASE + n` aliases, which `QUERY_EXT_CTRL` answers as the
//! kernel does.
//!
//! * `+0` [`VCAM_CID_ACTIVE_PHYSICAL_ID`] (read-only, volatile, `INTEGER_MENU`): which physical
//!   lens of a logical multi-camera is serving the stream. The value is an index into the
//!   camera's physical-id list; each menu item is that id as a number (or the index, for an id
//!   that is not one). Changes arrive as `V4L2_EVENT_CTRL`.
//! * `+1` [`VCAM_CID_AE_STATE`] (read-only, volatile, `MENU`): the auto-exposure state --
//!   Inactive, Searching, Converged, Locked, Flash Required, Precapture. Changes arrive as
//!   `V4L2_EVENT_CTRL`.
//! * `+2` [`VCAM_CID_AE_REGIONS`], `+3` [`VCAM_CID_AF_REGIONS`], `+4` [`VCAM_CID_AWB_REGIONS`]
//!   (`V4L2_CTRL_TYPE_U32`, two dimensions `[regions][5]`): the metering, focus and white-balance
//!   regions -- tap-to-meter and tap-to-focus, which V4L2 has no standard control for. Each is
//!   present only when the camera supports at least one region of that kind. The payload is an
//!   array of [`Region`]: see there for the wire format, which is ABI.
//!
//! # Events
//!
//! `V4L2_EVENT_CTRL` is produced as the kernel's control framework produces it: on
//! `SUBSCRIBE_EVENT` with `V4L2_EVENT_SUB_FL_SEND_INITIAL` (the current value and flags, for
//! every control but a class), when a session sets a control to a *different* value (to every
//! other subscribed session, and to the setting one only with `V4L2_EVENT_SUB_FL_ALLOW_FEEDBACK`),
//! when a flag changes (`INACTIVE` on the manual exposure controls, to every subscriber), and when
//! the backend reports a change it observed -- the autofocus state, the auto-exposure state, the
//! active physical lens -- from its capture-result callback. The backend already reports only
//! changes; the device compares against the value it holds anyway, so a repeated report is not a
//! repeated event. The event carries the value, the flags and the range, as `v4l2_event_ctrl`
//! defines; the guest driver forwards it to the file handle that subscribed
//! (`virtio_media_driver.c`, `VIRTIO_MEDIA_EVT_EVENT`), and the V4L2 core there only delivers an
//! event to a handle subscribed to that `(type, id)`.
//!
//! Note for guest software: the V4L2 documentation says a volatile control does not generate
//! change events; here `AUTO_FOCUS_STATUS`, `AE_STATE` and the active physical id do, because the
//! events do not come from the guest's control framework at all (plan §3.1). A program that
//! polls still works.

use std::collections::BTreeMap;

use enumn::N;
use v4l2r::bindings;
use v4l2r::bindings::v4l2_event;
use v4l2r::bindings::v4l2_query_ext_ctrl;
use v4l2r::bindings::v4l2_queryctrl;
use v4l2r::bindings::v4l2_querymenu;

use super::CameraInfo;

/// `V4L2_CID_PRIVATE_BASE` (`videodev2.h`): the old per-driver private control ids, which the
/// kernel maps onto a driver's private USER-class controls in `QUERY_EXT_CTRL`.
pub const V4L2_CID_PRIVATE_BASE: u32 = 0x0800_0000;

/// `V4L2_CTRL_ID2WHICH(id)`: the class bits of a control id.
pub const fn ctrl_class(id: u32) -> u32 {
    id & 0x0fff_0000
}

/// `V4L2_CTRL_DRIVER_PRIV(id)`: whether an id is in a driver's private range.
pub const fn is_driver_private(id: u32) -> bool {
    (id & 0xffff) >= 0x1000
}

/// The base of this device's private controls; see the module documentation.
pub const VCAM_CID_BASE: u32 = bindings::V4L2_CID_USER_BASE + 0x1200;
/// Which physical lens serves the stream (read-only, volatile, `INTEGER_MENU` over the
/// physical ids).
pub const VCAM_CID_ACTIVE_PHYSICAL_ID: u32 = VCAM_CID_BASE;
/// The auto-exposure state (read-only, volatile, `MENU`; see [`AeState`]).
pub const VCAM_CID_AE_STATE: u32 = VCAM_CID_BASE + 1;
/// Auto-exposure metering regions (`U32[regions][5]`, [`Region`]).
pub const VCAM_CID_AE_REGIONS: u32 = VCAM_CID_BASE + 2;
/// Autofocus regions (`U32[regions][5]`, [`Region`]).
pub const VCAM_CID_AF_REGIONS: u32 = VCAM_CID_BASE + 3;
/// Auto-white-balance regions (`U32[regions][5]`, [`Region`]).
pub const VCAM_CID_AWB_REGIONS: u32 = VCAM_CID_BASE + 4;

/// One metering or focus region, as it crosses the wire in the [`VCAM_CID_AE_REGIONS`],
/// [`VCAM_CID_AF_REGIONS`] and [`VCAM_CID_AWB_REGIONS`] payloads. **This layout is ABI.**
///
/// Five little-endian `u32` words in this order: `x`, `y`, `width`, `height`, `weight`. The
/// payload of the control is `regions` of these back to back (`elem_size = 4`, `elems = 5 *
/// regions`, `dims = [regions, 5]`), so `v4l2-ctl --subset` and any `V4L2_CTRL_TYPE_U32` array
/// client can address a field as `[region][field]`.
///
/// The rectangle is in *stream-relative* coordinates: `x`, `y`, `width` and `height` are in
/// units of [`REGION_SCALE`]ths (1/10000) of the frame the guest receives, so `(0, 0, 10000,
/// 10000)` is the whole frame and the centre quarter is `(2500, 2500, 5000, 5000)`, whatever the
/// format's size. The guest only ever knows its own frame; it is the host that knows the
/// sensor, the crop that gives the frame its aspect ratio and the zoom, and it converts the
/// rectangle into the camera's sensor coordinates (design §7.1, plan §3.1). `x + width` and
/// `y + height` may not exceed [`REGION_SCALE`].
///
/// `weight` is the region's weight, `0..=`[`REGION_MAX_WEIGHT`], as Camera2 defines it: regions
/// weigh against each other, a single region works with any non-zero weight, and a region of
/// weight 0 -- or of zero width or height -- is not a region: setting every entry to zero
/// clears the regions and returns the camera to its own metering.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Region {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    pub weight: u32,
}

/// The unit of a [`Region`]'s coordinates: 1/10000 of the frame.
pub const REGION_SCALE: u32 = 10_000;
/// The largest [`Region::weight`] (Camera2's).
pub const REGION_MAX_WEIGHT: u32 = 1000;
/// Words in one [`Region`] on the wire.
pub const REGION_WORDS: usize = 5;

impl Region {
    /// Whether this entry describes a region at all.
    pub fn is_set(&self) -> bool {
        self.weight > 0 && self.width > 0 && self.height > 0
    }

    fn from_words(words: &[u32]) -> Self {
        Self {
            x: words[0],
            y: words[1],
            width: words[2],
            height: words[3],
            weight: words[4],
        }
    }

    fn to_words(self) -> [u32; REGION_WORDS] {
        [self.x, self.y, self.width, self.height, self.weight]
    }

    /// Whether the rectangle fits the frame and the weight is in range.
    fn is_valid(&self) -> bool {
        self.x
            .checked_add(self.width)
            .is_some_and(|r| r <= REGION_SCALE)
            && self
                .y
                .checked_add(self.height)
                .is_some_and(|b| b <= REGION_SCALE)
            && self.weight <= REGION_MAX_WEIGHT
    }
}

// ---------------------------------------------------------------------------------------------
// The values, as the crate names them: each enum's discriminants are the V4L2 menu values.
// ---------------------------------------------------------------------------------------------

/// `V4L2_CID_POWER_LINE_FREQUENCY` (`V4L2_CID_POWER_LINE_FREQUENCY_*`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, N)]
#[repr(u32)]
pub enum PowerLine {
    Disabled = 0,
    Hz50 = 1,
    Hz60 = 2,
    Auto = 3,
}

/// `V4L2_CID_COLORFX` (`V4L2_COLORFX_*`; the two that take a colour argument are left out).
#[derive(Clone, Copy, Debug, PartialEq, Eq, N)]
#[repr(u32)]
pub enum ColorEffect {
    None = 0,
    BlackWhite = 1,
    Sepia = 2,
    Negative = 3,
    Emboss = 4,
    Sketch = 5,
    SkyBlue = 6,
    GrassGreen = 7,
    SkinWhiten = 8,
    Vivid = 9,
    Aqua = 10,
    ArtFreeze = 11,
    Silhouette = 12,
    Solarization = 13,
    Antique = 14,
}

/// `V4L2_CID_EXPOSURE_AUTO` (`V4L2_EXPOSURE_*`; the two priority modes are not offered).
#[derive(Clone, Copy, Debug, PartialEq, Eq, N)]
#[repr(u32)]
pub enum ExposureMode {
    Auto = 0,
    Manual = 1,
}

/// `V4L2_CID_ISO_SENSITIVITY_AUTO` (`V4L2_ISO_SENSITIVITY_*`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, N)]
#[repr(u32)]
pub enum IsoMode {
    Manual = 0,
    Auto = 1,
}

/// `V4L2_CID_AUTO_N_PRESET_WHITE_BALANCE` (`V4L2_WHITE_BALANCE_*`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, N)]
#[repr(u32)]
pub enum WhiteBalance {
    Manual = 0,
    Auto = 1,
    Incandescent = 2,
    Fluorescent = 3,
    FluorescentH = 4,
    Horizon = 5,
    Daylight = 6,
    Flash = 7,
    Cloudy = 8,
    Shade = 9,
}

/// `V4L2_CID_SCENE_MODE` (`V4L2_SCENE_MODE_*`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, N)]
#[repr(u32)]
pub enum SceneMode {
    None = 0,
    Backlight = 1,
    BeachSnow = 2,
    CandleLight = 3,
    DawnDusk = 4,
    FallColors = 5,
    Fireworks = 6,
    Landscape = 7,
    Night = 8,
    PartyIndoor = 9,
    Portrait = 10,
    Sports = 11,
    Sunset = 12,
    Text = 13,
}

/// `V4L2_CID_FLASH_LED_MODE` (`V4L2_FLASH_LED_MODE_*`). `Flash` is never offered (module
/// documentation).
#[derive(Clone, Copy, Debug, PartialEq, Eq, N)]
#[repr(u32)]
pub enum FlashLed {
    Off = 0,
    Flash = 1,
    Torch = 2,
}

/// The autofocus modes a camera offers (a capability, not a V4L2 value): what `FOCUS_AUTO` and
/// the trigger buttons can be built from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AfMode {
    Off,
    Auto,
    Macro,
    ContinuousVideo,
    ContinuousPicture,
    Edof,
}

/// The auto-exposure modes a camera offers (a capability): `Off` is what makes manual exposure
/// possible at all.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AeMode {
    Off,
    On,
    OnAutoFlash,
    OnAlwaysFlash,
    OnAutoFlashRedeye,
    OnExternalFlash,
}

/// `V4L2_CID_AUTO_FOCUS_START` / `V4L2_CID_AUTO_FOCUS_STOP`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AfTrigger {
    Start,
    Cancel,
}

/// The values of [`VCAM_CID_AE_STATE`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, N)]
#[repr(u32)]
pub enum AeState {
    Inactive = 0,
    Searching = 1,
    Converged = 2,
    Locked = 3,
    FlashRequired = 4,
    Precapture = 5,
}

/// `V4L2_AUTO_FOCUS_STATUS_IDLE`: no scan.
pub const AF_STATUS_IDLE: u32 = 0;
/// `V4L2_AUTO_FOCUS_STATUS_BUSY`: scanning.
pub const AF_STATUS_BUSY: u32 = 1 << 0;
/// `V4L2_AUTO_FOCUS_STATUS_REACHED`: in focus.
pub const AF_STATUS_REACHED: u32 = 1 << 1;
/// `V4L2_AUTO_FOCUS_STATUS_FAILED`: the scan ended out of focus.
pub const AF_STATUS_FAILED: u32 = 1 << 2;

/// The exposure-compensation range a camera offers: `min..=max` steps of `step_num/step_den` EV
/// (Camera2's `AE_COMPENSATION_RANGE` and `AE_COMPENSATION_STEP`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExposureBias {
    pub min: i32,
    pub max: i32,
    pub step_num: i32,
    pub step_den: i32,
}

/// How many regions of each kind a camera takes (Camera2's `CONTROL_MAX_REGIONS`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MaxRegions {
    pub ae: u32,
    pub awb: u32,
    pub af: u32,
}

/// A control applied to the camera, in the crate's units (see the module documentation), or --
/// for the last three -- one the backend reports having observed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CameraControl {
    /// `CONTROL_AE_TARGET_FPS_RANGE`: what `S_PARM` during streaming asks for.
    FpsRange(u32, u32),
    /// `V4L2_CID_ZOOM_ABSOLUTE`: the zoom ratio in hundredths.
    Zoom(u32),
    /// `V4L2_CID_EXPOSURE_AUTO`.
    ExposureMode(ExposureMode),
    /// `V4L2_CID_EXPOSURE_ABSOLUTE`: the exposure time in 100 µs units.
    ExposureTime(u32),
    /// `V4L2_CID_ISO_SENSITIVITY_AUTO`.
    IsoMode(IsoMode),
    /// `V4L2_CID_ISO_SENSITIVITY`: the ISO number (the menu item, not its index).
    Iso(u32),
    /// `V4L2_CID_AUTO_EXPOSURE_BIAS`: in the camera's own compensation steps
    /// ([`ExposureBias`]), not in the menu's 0.001 EV.
    ExposureBias(i32),
    /// `V4L2_CID_AUTO_N_PRESET_WHITE_BALANCE`.
    WhiteBalance(WhiteBalance),
    /// `V4L2_CID_POWER_LINE_FREQUENCY`.
    PowerLine(PowerLine),
    /// `V4L2_CID_COLORFX`.
    ColorEffect(ColorEffect),
    /// `V4L2_CID_SCENE_MODE`.
    SceneMode(SceneMode),
    /// `V4L2_CID_IMAGE_STABILIZATION`.
    Stabilization(bool),
    /// `V4L2_CID_FOCUS_AUTO`: continuous autofocus on, or the single-shot mode off.
    FocusAuto(bool),
    /// `V4L2_CID_AUTO_FOCUS_START` / `_STOP`.
    AfTrigger(AfTrigger),
    /// `V4L2_CID_FLASH_LED_MODE`.
    FlashLed(FlashLed),
    /// [`VCAM_CID_AE_REGIONS`], every entry, set or not.
    AeRegions(Vec<Region>),
    /// [`VCAM_CID_AF_REGIONS`].
    AfRegions(Vec<Region>),
    /// [`VCAM_CID_AWB_REGIONS`].
    AwbRegions(Vec<Region>),
    /// Reported by the backend: `V4L2_CID_AUTO_FOCUS_STATUS`, an `AF_STATUS_*` mask.
    AfStatus(u32),
    /// Reported by the backend: [`VCAM_CID_AE_STATE`].
    AeState(AeState),
    /// Reported by the backend: the physical lens now serving the stream, by its id.
    ActivePhysicalId(String),
}

// ---------------------------------------------------------------------------------------------
// The table
// ---------------------------------------------------------------------------------------------

/// What kind of control a [`ControlDesc`] is, with its range.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Kind {
    /// `V4L2_CTRL_TYPE_CTRL_CLASS`.
    Class,
    Boolean {
        default: bool,
    },
    /// Step 1.
    Integer {
        min: i64,
        max: i64,
        default: i64,
    },
    /// `V4L2_CTRL_TYPE_MENU`: `items[i]` names item `i`, `None` for one the camera lacks.
    Menu {
        items: Vec<Option<&'static str>>,
        default: u32,
    },
    /// `V4L2_CTRL_TYPE_INTEGER_MENU`: the value is an index into `items`.
    IntegerMenu {
        items: Vec<i64>,
        default: u32,
    },
    /// `V4L2_CTRL_TYPE_BITMASK`, read-only here.
    Bitmask {
        max: u32,
    },
    Button,
    /// `V4L2_CTRL_TYPE_U32` in `[regions][REGION_WORDS]`.
    Regions {
        regions: u32,
    },
}

impl Kind {
    fn v4l2_type(&self) -> u32 {
        match self {
            Kind::Class => bindings::v4l2_ctrl_type_V4L2_CTRL_TYPE_CTRL_CLASS,
            Kind::Boolean { .. } => bindings::v4l2_ctrl_type_V4L2_CTRL_TYPE_BOOLEAN,
            Kind::Integer { .. } => bindings::v4l2_ctrl_type_V4L2_CTRL_TYPE_INTEGER,
            Kind::Menu { .. } => bindings::v4l2_ctrl_type_V4L2_CTRL_TYPE_MENU,
            Kind::IntegerMenu { .. } => bindings::v4l2_ctrl_type_V4L2_CTRL_TYPE_INTEGER_MENU,
            Kind::Bitmask { .. } => bindings::v4l2_ctrl_type_V4L2_CTRL_TYPE_BITMASK,
            Kind::Button => bindings::v4l2_ctrl_type_V4L2_CTRL_TYPE_BUTTON,
            Kind::Regions { .. } => bindings::v4l2_ctrl_type_V4L2_CTRL_TYPE_U32,
        }
    }

    /// `(minimum, maximum, step, default)` as `QUERY_EXT_CTRL` reports them.
    fn range(&self) -> (i64, i64, u64, i64) {
        match self {
            Kind::Class | Kind::Button => (0, 0, 0, 0),
            Kind::Boolean { default } => (0, 1, 1, *default as i64),
            Kind::Integer { min, max, default } => (*min, *max, 1, *default),
            Kind::Menu { items, default } => (0, items.len() as i64 - 1, 1, *default as i64),
            Kind::IntegerMenu { items, default } => (0, items.len() as i64 - 1, 1, *default as i64),
            Kind::Bitmask { max } => (0, *max as i64, 0, 0),
            Kind::Regions { .. } => (0, REGION_SCALE as i64, 1, 0),
        }
    }

    /// Whether `G_CTRL`/`S_CTRL` may carry this control (the kernel's `is_int`).
    fn is_int(&self) -> bool {
        matches!(
            self,
            Kind::Boolean { .. }
                | Kind::Integer { .. }
                | Kind::Menu { .. }
                | Kind::IntegerMenu { .. }
                | Kind::Bitmask { .. }
                | Kind::Button
        )
    }

    /// A compound control (walked by `V4L2_CTRL_FLAG_NEXT_COMPOUND`, carried as a payload).
    fn is_compound(&self) -> bool {
        matches!(self, Kind::Regions { .. })
    }

    /// The default value.
    pub fn default_value(&self) -> Value {
        match self {
            Kind::Regions { regions } => Value::Array(vec![0; *regions as usize * REGION_WORDS]),
            other => Value::Int(other.range().3),
        }
    }
}

/// One control of the table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ControlDesc {
    pub id: u32,
    pub name: &'static str,
    pub kind: Kind,
    /// The flags that never change; [`Controls::flags`] adds the ones that do.
    pub flags: u32,
}

impl ControlDesc {
    fn new(id: u32, name: &'static str, kind: Kind, flags: u32) -> Self {
        Self {
            id,
            name,
            kind,
            flags,
        }
    }

    fn class(id: u32, name: &'static str) -> Self {
        Self::new(
            id,
            name,
            Kind::Class,
            bindings::V4L2_CTRL_FLAG_READ_ONLY | bindings::V4L2_CTRL_FLAG_WRITE_ONLY,
        )
    }

    pub fn is_read_only(&self) -> bool {
        self.flags & bindings::V4L2_CTRL_FLAG_READ_ONLY != 0
    }

    pub fn is_write_only(&self) -> bool {
        self.flags & bindings::V4L2_CTRL_FLAG_WRITE_ONLY != 0
    }

    pub fn is_compound(&self) -> bool {
        self.kind.is_compound()
    }

    pub fn is_int(&self) -> bool {
        self.kind.is_int()
    }

    /// Bytes of the payload a compound control carries; 0 for a plain one.
    pub fn payload_len(&self) -> u32 {
        match &self.kind {
            Kind::Regions { regions } => regions * REGION_WORDS as u32 * 4,
            _ => 0,
        }
    }
}

/// A control's value: one number, or the words of a compound payload.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Value {
    Int(i64),
    Array(Vec<u32>),
}

impl Value {
    pub fn int(&self) -> i64 {
        match self {
            Value::Int(v) => *v,
            Value::Array(_) => 0,
        }
    }
}

/// A menu over the values a camera offers: `items[v]` is named when `v` is offered, and the
/// default is `preferred` when offered, else the lowest offered value.
fn menu<T: Copy + Into<u32>>(offered: &[T], names: &[&'static str], preferred: T) -> Kind {
    let mut values: Vec<u32> = offered.iter().map(|&v| v.into()).collect();
    values.sort_unstable();
    values.dedup();
    let max = values.last().copied().unwrap_or(0);
    let items = (0..=max)
        .map(|v| {
            (values.contains(&v))
                .then(|| names.get(v as usize).copied())
                .flatten()
        })
        .collect();
    let preferred: u32 = preferred.into();
    let default = if values.contains(&preferred) {
        preferred
    } else {
        values.first().copied().unwrap_or(0)
    };
    Kind::Menu { items, default }
}

const POWER_LINE_NAMES: [&str; 4] = ["Disabled", "50 Hz", "60 Hz", "Auto"];
const COLORFX_NAMES: [&str; 15] = [
    "None",
    "Black & White",
    "Sepia",
    "Negative",
    "Emboss",
    "Sketch",
    "Sky Blue",
    "Grass Green",
    "Skin Whiten",
    "Vivid",
    "Aqua",
    "Art Freeze",
    "Silhouette",
    "Solarization",
    "Antique",
];
const EXPOSURE_AUTO_NAMES: [&str; 2] = ["Auto Mode", "Manual Mode"];
const ISO_AUTO_NAMES: [&str; 2] = ["Manual", "Auto"];
const WHITE_BALANCE_NAMES: [&str; 10] = [
    "Manual",
    "Auto",
    "Incandescent",
    "Fluorescent",
    "Fluorescent H",
    "Horizon",
    "Daylight",
    "Flash",
    "Cloudy",
    "Shade",
];
const SCENE_MODE_NAMES: [&str; 14] = [
    "None",
    "Backlight",
    "Beach/Snow",
    "Candle Light",
    "Dusk/Dawn",
    "Fall Colors",
    "Fireworks",
    "Landscape",
    "Night",
    "Party/Indoor",
    "Portrait",
    "Sports",
    "Sunset",
    "Text",
];
const FLASH_LED_NAMES: [&str; 3] = ["Off", "Flash", "Torch"];
const AE_STATE_NAMES: [&str; 6] = [
    "Inactive",
    "Searching",
    "Converged",
    "Locked",
    "Flash Required",
    "Precapture",
];

impl From<PowerLine> for u32 {
    fn from(v: PowerLine) -> u32 {
        v as u32
    }
}
impl From<ColorEffect> for u32 {
    fn from(v: ColorEffect) -> u32 {
        v as u32
    }
}
impl From<ExposureMode> for u32 {
    fn from(v: ExposureMode) -> u32 {
        v as u32
    }
}
impl From<IsoMode> for u32 {
    fn from(v: IsoMode) -> u32 {
        v as u32
    }
}
impl From<WhiteBalance> for u32 {
    fn from(v: WhiteBalance) -> u32 {
        v as u32
    }
}
impl From<SceneMode> for u32 {
    fn from(v: SceneMode) -> u32 {
        v as u32
    }
}
impl From<FlashLed> for u32 {
    fn from(v: FlashLed) -> u32 {
        v as u32
    }
}
impl From<AeState> for u32 {
    fn from(v: AeState) -> u32 {
        v as u32
    }
}

/// The ISO menu: the range's ends and the 100 x 2^n values strictly between them.
fn iso_ladder(min: u32, max: u32) -> Vec<i64> {
    let mut items = vec![min as i64];
    let mut v: u64 = 100;
    while v < max as u64 {
        if v > min as u64 {
            items.push(v as i64);
        }
        v *= 2;
    }
    if max > min {
        items.push(max as i64);
    }
    items
}

/// The exposure-compensation menu: every step from `min` to `max`, in 0.001 EV.
fn bias_items(bias: &ExposureBias) -> Vec<i64> {
    (bias.min..=bias.max)
        .map(|step| step as i64 * bias.step_num as i64 * 1000 / bias.step_den as i64)
        .collect()
}

/// The controls of one camera device: the table and the current values.
#[derive(Clone, Debug)]
pub struct Controls {
    /// Ascending by id.
    descs: Vec<ControlDesc>,
    values: BTreeMap<u32, Value>,
    /// What the active-physical-id menu indexes.
    physical_ids: Vec<String>,
    /// The first exposure-compensation step, which menu index 0 stands for.
    bias_min: i32,
}

impl Controls {
    /// The table for `info`: each control only when the camera has the feature.
    pub fn new(info: &CameraInfo) -> Self {
        let mut descs: Vec<ControlDesc> = Vec::new();

        // USER class.
        let mut user = Vec::new();
        if !info.antibanding.is_empty() {
            user.push(ControlDesc::new(
                bindings::V4L2_CID_POWER_LINE_FREQUENCY,
                "Power Line Frequency",
                menu(&info.antibanding, &POWER_LINE_NAMES, PowerLine::Auto),
                0,
            ));
        }
        if info.effects.iter().any(|&e| e != ColorEffect::None) {
            let mut effects = info.effects.clone();
            effects.push(ColorEffect::None);
            user.push(ControlDesc::new(
                bindings::V4L2_CID_COLORFX,
                "Color Effects",
                menu(&effects, &COLORFX_NAMES, ColorEffect::None),
                0,
            ));
        }
        if !info.physical_ids.is_empty() {
            let items = info
                .physical_ids
                .iter()
                .enumerate()
                .map(|(i, id)| id.parse::<i64>().unwrap_or(i as i64))
                .collect();
            user.push(ControlDesc::new(
                VCAM_CID_ACTIVE_PHYSICAL_ID,
                "Active Physical Camera",
                Kind::IntegerMenu { items, default: 0 },
                bindings::V4L2_CTRL_FLAG_READ_ONLY | bindings::V4L2_CTRL_FLAG_VOLATILE,
            ));
        }
        user.push(ControlDesc::new(
            VCAM_CID_AE_STATE,
            "Auto Exposure, State",
            menu(
                &[
                    AeState::Inactive,
                    AeState::Searching,
                    AeState::Converged,
                    AeState::Locked,
                    AeState::FlashRequired,
                    AeState::Precapture,
                ],
                &AE_STATE_NAMES,
                AeState::Inactive,
            ),
            bindings::V4L2_CTRL_FLAG_READ_ONLY | bindings::V4L2_CTRL_FLAG_VOLATILE,
        ));
        for (id, name, regions) in [
            (
                VCAM_CID_AE_REGIONS,
                "Auto Exposure, Regions",
                info.max_regions.ae,
            ),
            (
                VCAM_CID_AF_REGIONS,
                "Auto Focus, Regions",
                info.max_regions.af,
            ),
            (
                VCAM_CID_AWB_REGIONS,
                "White Balance, Regions",
                info.max_regions.awb,
            ),
        ] {
            if regions > 0 {
                user.push(ControlDesc::new(
                    id,
                    name,
                    Kind::Regions { regions },
                    bindings::V4L2_CTRL_FLAG_HAS_PAYLOAD,
                ));
            }
        }
        if !user.is_empty() {
            descs.push(ControlDesc::class(
                bindings::V4L2_CID_USER_CLASS,
                "User Controls",
            ));
            descs.extend(user);
        }

        // CAMERA class.
        let mut camera = Vec::new();
        let manual_ae = info.ae_modes.contains(&AeMode::Off) && info.ae_modes.contains(&AeMode::On);
        if manual_ae {
            camera.push(ControlDesc::new(
                bindings::V4L2_CID_EXPOSURE_AUTO,
                "Exposure, Auto",
                menu(
                    &[ExposureMode::Auto, ExposureMode::Manual],
                    &EXPOSURE_AUTO_NAMES,
                    ExposureMode::Auto,
                ),
                0,
            ));
            if let Some((min_ns, max_ns)) = info.exposure_range_ns {
                // 100 µs units; a camera faster than that reports 1.
                let min = min_ns.div_ceil(100_000).max(1) as i64;
                let max = (max_ns / 100_000).max(min as u64) as i64;
                camera.push(ControlDesc::new(
                    bindings::V4L2_CID_EXPOSURE_ABSOLUTE,
                    "Exposure Time, Absolute",
                    Kind::Integer {
                        min,
                        max,
                        default: 333.clamp(min, max),
                    },
                    0,
                ));
            }
        }
        let continuous = info
            .af_modes
            .iter()
            .any(|m| matches!(m, AfMode::ContinuousVideo | AfMode::ContinuousPicture));
        let focuses = info.af_modes.iter().any(|&m| m != AfMode::Off);
        if continuous {
            camera.push(ControlDesc::new(
                bindings::V4L2_CID_FOCUS_AUTO,
                "Focus, Automatic Continuous",
                Kind::Boolean { default: true },
                0,
            ));
        }
        if let Some((min, max)) = info.zoom_range.filter(|(min, max)| min <= max) {
            camera.push(ControlDesc::new(
                bindings::V4L2_CID_ZOOM_ABSOLUTE,
                "Zoom, Absolute",
                Kind::Integer {
                    min: min as i64,
                    max: max as i64,
                    default: 100.clamp(min as i64, max as i64),
                },
                0,
            ));
        }
        let mut bias_min = 0;
        if let Some(bias) = info
            .exposure_bias
            .filter(|b| b.max > b.min && b.step_den > 0 && b.step_num != 0)
        {
            bias_min = bias.min;
            let items = bias_items(&bias);
            // Index 0 is `min` steps; no compensation is `-min` items in.
            let default = (0i64 - bias.min as i64).clamp(0, items.len() as i64 - 1) as u32;
            camera.push(ControlDesc::new(
                bindings::V4L2_CID_AUTO_EXPOSURE_BIAS,
                "Auto Exposure, Bias",
                Kind::IntegerMenu { items, default },
                0,
            ));
        }
        let presets: Vec<WhiteBalance> = info
            .awb_modes
            .iter()
            .copied()
            .filter(|&m| m != WhiteBalance::Manual)
            .collect();
        if presets.len() >= 2 {
            camera.push(ControlDesc::new(
                bindings::V4L2_CID_AUTO_N_PRESET_WHITE_BALANCE,
                "White Balance, Auto & Preset",
                menu(&presets, &WHITE_BALANCE_NAMES, WhiteBalance::Auto),
                0,
            ));
        }
        if info.stabilization {
            camera.push(ControlDesc::new(
                bindings::V4L2_CID_IMAGE_STABILIZATION,
                "Image Stabilization",
                Kind::Boolean { default: false },
                0,
            ));
        }
        if manual_ae {
            if let Some((min, max)) = info.iso_range.filter(|(min, max)| *min > 0 && min <= max) {
                let items = iso_ladder(min, max);
                let default = items.iter().position(|&v| v == 100).unwrap_or(0) as u32;
                camera.push(ControlDesc::new(
                    bindings::V4L2_CID_ISO_SENSITIVITY,
                    "ISO Sensitivity",
                    Kind::IntegerMenu { items, default },
                    0,
                ));
            }
            camera.push(ControlDesc::new(
                bindings::V4L2_CID_ISO_SENSITIVITY_AUTO,
                "ISO Sensitivity, Auto",
                menu(
                    &[IsoMode::Manual, IsoMode::Auto],
                    &ISO_AUTO_NAMES,
                    IsoMode::Auto,
                ),
                0,
            ));
        }
        if info.scenes.iter().any(|&s| s != SceneMode::None) {
            let mut scenes = info.scenes.clone();
            scenes.push(SceneMode::None);
            camera.push(ControlDesc::new(
                bindings::V4L2_CID_SCENE_MODE,
                "Scene Mode",
                menu(&scenes, &SCENE_MODE_NAMES, SceneMode::None),
                0,
            ));
        }
        if focuses {
            let button =
                bindings::V4L2_CTRL_FLAG_WRITE_ONLY | bindings::V4L2_CTRL_FLAG_EXECUTE_ON_WRITE;
            camera.push(ControlDesc::new(
                bindings::V4L2_CID_AUTO_FOCUS_START,
                "Auto Focus, Start",
                Kind::Button,
                button,
            ));
            camera.push(ControlDesc::new(
                bindings::V4L2_CID_AUTO_FOCUS_STOP,
                "Auto Focus, Stop",
                Kind::Button,
                button,
            ));
            camera.push(ControlDesc::new(
                bindings::V4L2_CID_AUTO_FOCUS_STATUS,
                "Auto Focus, Status",
                Kind::Bitmask {
                    max: AF_STATUS_BUSY | AF_STATUS_REACHED | AF_STATUS_FAILED,
                },
                bindings::V4L2_CTRL_FLAG_READ_ONLY | bindings::V4L2_CTRL_FLAG_VOLATILE,
            ));
        }
        if !camera.is_empty() {
            descs.push(ControlDesc::class(
                bindings::V4L2_CID_CAMERA_CLASS,
                "Camera Controls",
            ));
            descs.extend(camera);
        }

        // FLASH class.
        if info.flash {
            descs.push(ControlDesc::class(
                bindings::V4L2_CID_FLASH_CLASS,
                "Flash Controls",
            ));
            descs.push(ControlDesc::new(
                bindings::V4L2_CID_FLASH_LED_MODE,
                "Flash LED Mode",
                menu(
                    &[FlashLed::Off, FlashLed::Torch],
                    &FLASH_LED_NAMES,
                    FlashLed::Off,
                ),
                0,
            ));
        }

        descs.sort_by_key(|d| d.id);
        let values = descs
            .iter()
            .filter(|d| !matches!(d.kind, Kind::Class | Kind::Button))
            .map(|d| (d.id, d.kind.default_value()))
            .collect();
        Self {
            descs,
            values,
            physical_ids: info.physical_ids.clone(),
            bias_min,
        }
    }

    pub fn descs(&self) -> &[ControlDesc] {
        &self.descs
    }

    pub fn find(&self, id: u32) -> Option<&ControlDesc> {
        self.descs.iter().find(|d| d.id == id)
    }

    /// Whether the class of `class_id` (an `id & 0x0fff0000`) has a class control, which is what
    /// an ext-controls call with `count == 0` and `which = class` asks.
    pub fn has_class(&self, class: u32) -> bool {
        self.find(class | 1).is_some_and(|d| d.kind == Kind::Class)
    }

    /// The `V4L2_CTRL_FLAG_NEXT_*` walk: the first control whose id is above `after` and whose
    /// kind the flags ask for (`regular`: plain controls; `compound`: payload controls).
    pub fn next(&self, after: u32, regular: bool, compound: bool) -> Option<&ControlDesc> {
        self.descs
            .iter()
            .find(|d| d.id > after && if d.is_compound() { compound } else { regular })
    }

    /// The control an id names for the four ioctls the kernel resolves the old-style
    /// `V4L2_CID_PRIVATE_BASE + n` alias in -- `QUERYCTRL`, `QUERYMENU`, `G_CTRL` and `S_CTRL`,
    /// the four `find_private_ref` names (`v4l2-ctrls-core.c`). `find_ref` branches to it
    /// before it hashes, so all four agree; the extended controls do not (`prepare_ext_ctrls`
    /// refuses a private id outright).
    pub fn find_legacy(&self, id: u32) -> Option<&ControlDesc> {
        if id >= V4L2_CID_PRIVATE_BASE {
            self.private_alias(id)
        } else {
            self.find(id)
        }
    }

    /// The old-style `V4L2_CID_PRIVATE_BASE + n` alias: the n-th private USER-class control that
    /// `G_CTRL` could carry, as the kernel resolves it.
    pub fn private_alias(&self, id: u32) -> Option<&ControlDesc> {
        let n = id.checked_sub(V4L2_CID_PRIVATE_BASE)? as usize;
        self.descs
            .iter()
            .filter(|d| {
                ctrl_class(d.id) == bindings::V4L2_CTRL_CLASS_USER
                    && is_driver_private(d.id)
                    && d.is_int()
            })
            .nth(n)
    }

    /// Whether the camera's auto-exposure is off, i.e. the manual exposure controls apply.
    pub fn manual_exposure(&self) -> bool {
        let exposure = self
            .values
            .get(&bindings::V4L2_CID_EXPOSURE_AUTO)
            .is_some_and(|v| v.int() == ExposureMode::Manual as i64);
        let iso = self
            .values
            .get(&bindings::V4L2_CID_ISO_SENSITIVITY_AUTO)
            .is_some_and(|v| v.int() == IsoMode::Manual as i64);
        exposure || iso
    }

    /// The flags `QUERY_EXT_CTRL` and the events report now: the static ones, and `INACTIVE` on
    /// the manual exposure controls while the camera exposes automatically.
    pub fn flags(&self, desc: &ControlDesc) -> u32 {
        let manual_only = matches!(
            desc.id,
            bindings::V4L2_CID_EXPOSURE_ABSOLUTE | bindings::V4L2_CID_ISO_SENSITIVITY
        );
        if manual_only && !self.manual_exposure() {
            desc.flags | bindings::V4L2_CTRL_FLAG_INACTIVE
        } else {
            desc.flags
        }
    }

    /// `QUERY_EXT_CTRL`'s answer, with `id` as the id to report (the alias, when one was used).
    pub fn query_ext(&self, desc: &ControlDesc, id: u32) -> v4l2_query_ext_ctrl {
        let (minimum, maximum, step, default_value) = desc.kind.range();
        let mut qc = v4l2_query_ext_ctrl {
            id,
            type_: desc.kind.v4l2_type(),
            minimum,
            maximum,
            step,
            default_value,
            flags: self.flags(desc),
            ..Default::default()
        };
        for (dst, src) in qc.name.iter_mut().zip(desc.name.bytes()) {
            *dst = src as std::os::raw::c_char;
        }
        match &desc.kind {
            Kind::Regions { regions } => {
                qc.elem_size = 4;
                qc.elems = regions * REGION_WORDS as u32;
                qc.nr_of_dims = 2;
                qc.dims[0] = *regions;
                qc.dims[1] = REGION_WORDS as u32;
            }
            _ => {
                qc.elem_size = 4;
                qc.elems = 1;
            }
        }
        qc
    }

    /// `QUERYCTRL`'s answer: `QUERY_EXT_CTRL`'s, with the range zeroed for every type the old
    /// structure cannot carry, as `v4l2_query_ext_ctrl_to_v4l2_queryctrl` does.
    pub fn query(&self, desc: &ControlDesc, id: u32) -> v4l2_queryctrl {
        let ext = self.query_ext(desc, id);
        let mut qc = v4l2_queryctrl {
            id: ext.id,
            type_: ext.type_,
            flags: ext.flags,
            ..Default::default()
        };
        for (dst, src) in qc.name.iter_mut().zip(ext.name.iter()) {
            *dst = *src as u8;
        }
        if matches!(
            desc.kind,
            Kind::Boolean { .. }
                | Kind::Integer { .. }
                | Kind::Menu { .. }
                | Kind::IntegerMenu { .. }
                | Kind::Bitmask { .. }
        ) {
            qc.minimum = ext.minimum as i32;
            qc.maximum = ext.maximum as i32;
            qc.step = ext.step as i32;
            qc.default_value = ext.default_value as i32;
        }
        qc
    }

    /// `QUERYMENU`: item `index` of menu `id`; `EINVAL` for a non-menu, an index out of range, or
    /// an item the camera lacks.
    /// `id` may be an old-style private alias; the answer carries the id as asked, the way
    /// `v4l2_querymenu` leaves `qm->id` alone (D33).
    pub fn menu_item(&self, id: u32, index: u32) -> Result<v4l2_querymenu, i32> {
        let desc = self.find_legacy(id).ok_or(libc::EINVAL)?;
        let mut qm = v4l2_querymenu {
            id,
            index,
            ..Default::default()
        };
        match &desc.kind {
            Kind::Menu { items, .. } => {
                let name = items
                    .get(index as usize)
                    .copied()
                    .flatten()
                    .ok_or(libc::EINVAL)?;
                let mut bytes = [0u8; 32];
                for (dst, src) in bytes.iter_mut().zip(name.bytes()) {
                    *dst = src;
                }
                qm.__bindgen_anon_1.name = bytes;
            }
            Kind::IntegerMenu { items, .. } => {
                let value = *items.get(index as usize).ok_or(libc::EINVAL)?;
                qm.__bindgen_anon_1.value = value;
            }
            _ => return Err(libc::EINVAL),
        }
        Ok(qm)
    }

    /// The current value of `id`, for a control that has one.
    pub fn current(&self, id: u32) -> Option<&Value> {
        self.values.get(&id)
    }

    /// Check `value` for `desc`: `ERANGE` outside the range, `EINVAL` for a menu item the camera
    /// lacks or a payload of the wrong length; booleans are normalised, a button is 0. The
    /// kernel would clamp an integer; refusing is what lets `TRY_EXT_CTRLS` say no.
    pub fn validate(&self, desc: &ControlDesc, value: &Value) -> Result<Value, i32> {
        match (&desc.kind, value) {
            (Kind::Class, _) => Err(libc::EACCES),
            (Kind::Button, _) => Ok(Value::Int(0)),
            (Kind::Boolean { .. }, Value::Int(v)) => Ok(Value::Int((*v != 0) as i64)),
            (Kind::Integer { min, max, .. }, Value::Int(v)) => {
                if v < min || v > max {
                    Err(libc::ERANGE)
                } else {
                    Ok(Value::Int(*v))
                }
            }
            (Kind::Menu { items, .. }, Value::Int(v)) => {
                if *v < 0 || *v >= items.len() as i64 {
                    Err(libc::ERANGE)
                } else if items[*v as usize].is_none() {
                    Err(libc::EINVAL)
                } else {
                    Ok(Value::Int(*v))
                }
            }
            (Kind::IntegerMenu { items, .. }, Value::Int(v)) => {
                if *v < 0 || *v >= items.len() as i64 {
                    Err(libc::ERANGE)
                } else {
                    Ok(Value::Int(*v))
                }
            }
            (Kind::Bitmask { max }, Value::Int(v)) => Ok(Value::Int(*v & *max as i64)),
            (Kind::Regions { regions }, Value::Array(words)) => {
                if words.len() != *regions as usize * REGION_WORDS {
                    return Err(libc::EINVAL);
                }
                if words
                    .chunks_exact(REGION_WORDS)
                    .any(|w| !Region::from_words(w).is_valid())
                {
                    return Err(libc::ERANGE);
                }
                Ok(Value::Array(words.clone()))
            }
            _ => Err(libc::EINVAL),
        }
    }

    /// Store `value` (already validated) for `id`; whether it differs from what was there. A
    /// button stores nothing and is always "changed": it executes on write.
    pub fn set(&mut self, id: u32, value: Value) -> bool {
        match self.values.get_mut(&id) {
            Some(slot) if *slot != value => {
                *slot = value;
                true
            }
            Some(_) => false,
            None => self.find(id).is_some_and(|d| d.kind == Kind::Button),
        }
    }

    /// `value` of control `id` as the backend is told it, in the crate's units; `None` for a
    /// control the backend is never sent (the read-only ones).
    pub fn control_of(&self, id: u32, value: &Value) -> Option<CameraControl> {
        let v = value.int();
        let regions = |value: &Value| match value {
            Value::Array(words) => words
                .chunks_exact(REGION_WORDS)
                .map(Region::from_words)
                .collect(),
            Value::Int(_) => Vec::new(),
        };
        Some(match id {
            bindings::V4L2_CID_ZOOM_ABSOLUTE => CameraControl::Zoom(v as u32),
            bindings::V4L2_CID_EXPOSURE_AUTO => {
                CameraControl::ExposureMode(ExposureMode::n(v as u32)?)
            }
            bindings::V4L2_CID_EXPOSURE_ABSOLUTE => CameraControl::ExposureTime(v as u32),
            bindings::V4L2_CID_ISO_SENSITIVITY_AUTO => {
                CameraControl::IsoMode(IsoMode::n(v as u32)?)
            }
            bindings::V4L2_CID_ISO_SENSITIVITY => match &self.find(id)?.kind {
                Kind::IntegerMenu { items, .. } => {
                    CameraControl::Iso(*items.get(v as usize)? as u32)
                }
                _ => return None,
            },
            bindings::V4L2_CID_AUTO_EXPOSURE_BIAS => {
                CameraControl::ExposureBias(self.bias_min.saturating_add(v as i32))
            }
            bindings::V4L2_CID_AUTO_N_PRESET_WHITE_BALANCE => {
                CameraControl::WhiteBalance(WhiteBalance::n(v as u32)?)
            }
            bindings::V4L2_CID_POWER_LINE_FREQUENCY => {
                CameraControl::PowerLine(PowerLine::n(v as u32)?)
            }
            bindings::V4L2_CID_COLORFX => CameraControl::ColorEffect(ColorEffect::n(v as u32)?),
            bindings::V4L2_CID_SCENE_MODE => CameraControl::SceneMode(SceneMode::n(v as u32)?),
            bindings::V4L2_CID_IMAGE_STABILIZATION => CameraControl::Stabilization(v != 0),
            bindings::V4L2_CID_FOCUS_AUTO => CameraControl::FocusAuto(v != 0),
            bindings::V4L2_CID_AUTO_FOCUS_START => CameraControl::AfTrigger(AfTrigger::Start),
            bindings::V4L2_CID_AUTO_FOCUS_STOP => CameraControl::AfTrigger(AfTrigger::Cancel),
            bindings::V4L2_CID_FLASH_LED_MODE => CameraControl::FlashLed(FlashLed::n(v as u32)?),
            VCAM_CID_AE_REGIONS => CameraControl::AeRegions(regions(value)),
            VCAM_CID_AF_REGIONS => CameraControl::AfRegions(regions(value)),
            VCAM_CID_AWB_REGIONS => CameraControl::AwbRegions(regions(value)),
            _ => return None,
        })
    }

    /// Every settable control at its current value: what a stream is opened with.
    pub fn all_settable(&self) -> Vec<CameraControl> {
        self.descs
            .iter()
            .filter(|d| !d.is_read_only() && d.kind != Kind::Button)
            .filter_map(|d| self.control_of(d.id, self.values.get(&d.id)?))
            .collect()
    }

    /// A value the backend observed: store it and say which control changed, or `None` when
    /// nothing did (the same value again, a control this camera does not have, a physical id
    /// the camera never listed).
    pub fn observe(&mut self, control: &CameraControl) -> Option<u32> {
        let (id, value) = match control {
            CameraControl::AfStatus(status) => (
                bindings::V4L2_CID_AUTO_FOCUS_STATUS,
                Value::Int(
                    (*status & (AF_STATUS_BUSY | AF_STATUS_REACHED | AF_STATUS_FAILED)) as i64,
                ),
            ),
            CameraControl::AeState(state) => (VCAM_CID_AE_STATE, Value::Int(*state as i64)),
            CameraControl::ActivePhysicalId(id) => {
                let index = self.physical_ids.iter().position(|p| p == id)?;
                (VCAM_CID_ACTIVE_PHYSICAL_ID, Value::Int(index as i64))
            }
            // The backend reports the value of a control it changed itself (a clamp): stored
            // as a guest's set would be, minus the validation the guest gets.
            other => {
                let id = self.id_of(other)?;
                let desc = self.find(id)?;
                (id, self.value_of(desc, other)?)
            }
        };
        if !self.values.contains_key(&id) {
            return None;
        }
        self.set(id, value).then_some(id)
    }

    /// The control id a backend-side value belongs to.
    fn id_of(&self, control: &CameraControl) -> Option<u32> {
        Some(match control {
            CameraControl::Zoom(_) => bindings::V4L2_CID_ZOOM_ABSOLUTE,
            CameraControl::ExposureMode(_) => bindings::V4L2_CID_EXPOSURE_AUTO,
            CameraControl::ExposureTime(_) => bindings::V4L2_CID_EXPOSURE_ABSOLUTE,
            CameraControl::IsoMode(_) => bindings::V4L2_CID_ISO_SENSITIVITY_AUTO,
            CameraControl::Iso(_) => bindings::V4L2_CID_ISO_SENSITIVITY,
            CameraControl::ExposureBias(_) => bindings::V4L2_CID_AUTO_EXPOSURE_BIAS,
            CameraControl::WhiteBalance(_) => bindings::V4L2_CID_AUTO_N_PRESET_WHITE_BALANCE,
            CameraControl::PowerLine(_) => bindings::V4L2_CID_POWER_LINE_FREQUENCY,
            CameraControl::ColorEffect(_) => bindings::V4L2_CID_COLORFX,
            CameraControl::SceneMode(_) => bindings::V4L2_CID_SCENE_MODE,
            CameraControl::Stabilization(_) => bindings::V4L2_CID_IMAGE_STABILIZATION,
            CameraControl::FocusAuto(_) => bindings::V4L2_CID_FOCUS_AUTO,
            CameraControl::FlashLed(_) => bindings::V4L2_CID_FLASH_LED_MODE,
            CameraControl::AeRegions(_) => VCAM_CID_AE_REGIONS,
            CameraControl::AfRegions(_) => VCAM_CID_AF_REGIONS,
            CameraControl::AwbRegions(_) => VCAM_CID_AWB_REGIONS,
            _ => return None,
        })
    }

    /// The stored form of a backend-side value, clamped into the control's range.
    fn value_of(&self, desc: &ControlDesc, control: &CameraControl) -> Option<Value> {
        let clamp_int = |v: i64| match desc.kind.range() {
            (min, max, _, _) => v.clamp(min, max),
        };
        Some(match control {
            CameraControl::Zoom(v) | CameraControl::ExposureTime(v) => {
                Value::Int(clamp_int(*v as i64))
            }
            CameraControl::ExposureMode(m) => Value::Int(*m as i64),
            CameraControl::IsoMode(m) => Value::Int(*m as i64),
            CameraControl::Iso(iso) => match &desc.kind {
                Kind::IntegerMenu { items, .. } => {
                    // The nearest item.
                    let index = items
                        .iter()
                        .enumerate()
                        .min_by_key(|(_, &v)| (v - *iso as i64).abs())?
                        .0;
                    Value::Int(index as i64)
                }
                _ => return None,
            },
            CameraControl::ExposureBias(steps) => {
                Value::Int(clamp_int(*steps as i64 - self.bias_min as i64))
            }
            CameraControl::WhiteBalance(v) => Value::Int(*v as i64),
            CameraControl::PowerLine(v) => Value::Int(*v as i64),
            CameraControl::ColorEffect(v) => Value::Int(*v as i64),
            CameraControl::SceneMode(v) => Value::Int(*v as i64),
            CameraControl::Stabilization(v) | CameraControl::FocusAuto(v) => Value::Int(*v as i64),
            CameraControl::FlashLed(v) => Value::Int(*v as i64),
            CameraControl::AeRegions(r)
            | CameraControl::AfRegions(r)
            | CameraControl::AwbRegions(r) => {
                let Kind::Regions { regions } = desc.kind else {
                    return None;
                };
                let mut words: Vec<u32> = r.iter().flat_map(|r| r.to_words()).collect();
                words.resize(regions as usize * REGION_WORDS, 0);
                Value::Array(words)
            }
            _ => return None,
        })
    }

    /// A `V4L2_EVENT_CTRL` for `desc` with `changes` (`V4L2_EVENT_CTRL_CH_*`), carrying the
    /// current value, flags and range as the kernel's `fill_event` does.
    pub fn event(&self, desc: &ControlDesc, changes: u32) -> v4l2_event {
        let (minimum, maximum, step, default_value) = desc.kind.range();
        let value = match self.values.get(&desc.id) {
            Some(Value::Int(v)) => *v,
            _ => 0,
        };
        let mut ev = v4l2_event {
            type_: bindings::V4L2_EVENT_CTRL,
            id: desc.id,
            ..Default::default()
        };
        ev.u.ctrl = bindings::v4l2_event_ctrl {
            changes,
            type_: desc.kind.v4l2_type(),
            __bindgen_anon_1: bindings::v4l2_event_ctrl__bindgen_ty_1 { value64: value },
            flags: self.flags(desc),
            minimum: minimum as i32,
            maximum: maximum as i32,
            step: step as i32,
            default_value: default_value as i32,
        };
        ev
    }
}
