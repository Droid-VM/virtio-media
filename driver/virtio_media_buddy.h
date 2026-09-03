// SPDX-License-Identifier: BSD-3-Clause OR GPL-2.0+

/*
 * drm_buddy compatibility shims for the virtio-media driver.
 *
 * The media_guest pool (VPU_DESIGN.md 5.1) is carved with the DRM buddy
 * allocator. Its API is stable but its spelling is not: 6.10 gave
 * drm_buddy_free_list() a flags argument, and 7.1 moved the allocator to the
 * generic <linux/gpu_buddy.h> library, renaming every drm_buddy_* symbol and
 * DRM_BUDDY_* flag to gpu_buddy_* / GPU_BUDDY_* 1:1 with unchanged
 * signatures. Keep the pre-7.1 spellings in the driver and map them here.
 *
 * Vendored from droidvm-guest-additions/virtio_gpu/virtgpu_drv.h so the DKMS
 * copy of this driver builds against the same range of guest kernels as the
 * virtio-gpu driver next to it.
 *
 * Copyright (c) 2026 DroidVM contributors.
 */

#ifndef __VIRTIO_MEDIA_BUDDY_H
#define __VIRTIO_MEDIA_BUDDY_H

#include <linux/version.h>

#if LINUX_VERSION_CODE >= KERNEL_VERSION(7, 1, 0)
#include <linux/gpu_buddy.h>
#endif
#include <drm/drm_buddy.h>

#if LINUX_VERSION_CODE >= KERNEL_VERSION(7, 1, 0)
#define drm_buddy			gpu_buddy
#define drm_buddy_block			gpu_buddy_block
#define drm_buddy_init			gpu_buddy_init
#define drm_buddy_fini			gpu_buddy_fini
#define drm_buddy_alloc_blocks		gpu_buddy_alloc_blocks
#define drm_buddy_free_list		gpu_buddy_free_list
#define drm_buddy_block_offset		gpu_buddy_block_offset
#define drm_buddy_block_size		gpu_buddy_block_size
#define DRM_BUDDY_RANGE_ALLOCATION	GPU_BUDDY_RANGE_ALLOCATION
#define DRM_BUDDY_CONTIGUOUS_ALLOCATION	GPU_BUDDY_CONTIGUOUS_ALLOCATION
#define DRM_BUDDY_CLEAR_ALLOCATION	GPU_BUDDY_CLEAR_ALLOCATION
#endif

/* drm_buddy_free_list() gained the flags argument with clear-page tracking in 6.10. */
#ifdef DRM_BUDDY_CLEAR_ALLOCATION
#define vmedia_drm_buddy_free_list(mm, objects) \
	drm_buddy_free_list((mm), (objects), 0)
#else
#define vmedia_drm_buddy_free_list(mm, objects) \
	drm_buddy_free_list((mm), (objects))
#endif

#endif // __VIRTIO_MEDIA_BUDDY_H
