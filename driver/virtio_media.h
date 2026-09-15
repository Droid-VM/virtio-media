// SPDX-License-Identifier: BSD-3-Clause OR GPL-2.0+

/*
 * Virtio-media structures & functions declarations.
 *
 * Copyright (c) 2023-2024 Google LLC.
 */

#ifndef __VIRTIO_MEDIA_H
#define __VIRTIO_MEDIA_H

#include <linux/mutex.h>
#include <linux/virtio_config.h>
#include <media/v4l2-device.h>

#include "protocol.h"
#include "virtio_media_buddy.h"

#define DESC_CHAIN_MAX_LEN SG_MAX_SINGLE_ALLOC

#define VIRTIO_MEDIA_DEFAULT_DRIVER_NAME "virtio_media"

extern char *driver_name;
extern char *driver_owned_queues;

/**
 * Virtio-media device.
 */
struct virtio_media {
	struct v4l2_device v4l2_dev;
	struct video_device video_dev;

	struct virtio_device *virtio_dev;
	struct virtqueue *commandq;
	struct virtqueue *eventq;
	struct work_struct eventq_work;

	/*
	 * Region into which MMAP buffers are mapped by the host: the media_host
	 * pool from /reserved-memory when the VMM built one, else virtio shm
	 * region 0. len == 0 means neither exists and host-owned MMAP buffers
	 * cannot be mapped at all (VPU_DESIGN.md 2.4, 5.1).
	 */
	struct virtio_shm_region mmap_region;

	/*
	 * Device driver-owned buffers are DMA-allocated for when there is no
	 * media_guest pool: the transport's parent, which is what the
	 * virtqueues DMA through as well (the virtio_device itself carries no
	 * DMA mask).
	 */
	struct device *dma_dev;

	/*
	 * dma-buf resolver: a bare, driver-owned struct device DMABUF imports
	 * attach to instead of @dma_dev. In a protected VM @dma_dev is bound
	 * to a restricted DMA pool (swiotlb force-bounce), and such a device
	 * cannot map a resource address at all -- dma_map_phys(DMA_ATTR_MMIO)
	 * returns DMA_MAPPING_ERROR, so importing a virtio-gpu vram dma-buf
	 * failed -EIO (D90). The resolver has no OF node, no bus and no
	 * driver, so nothing ever runs of_dma_configure()/arch_setup_dma_ops()
	 * on it: dma_ops stays NULL (dma-direct), dma_range_map stays NULL
	 * (identity), and its swiotlb is the default, non-force-bounce pool --
	 * dma_map_resource()/dma_map_sgtable() on it return the physical
	 * address unchanged, which on this IOMMU-less transport is the
	 * guest-physical address the host-facing SG list needs. NULL when its
	 * creation failed at probe; imports then fall back to @dma_dev.
	 * Created in probe, destroyed with the final teardown so an import
	 * held by an open fd across an unbind can still detach (D66 lifetime).
	 */
	struct device *import_dev;

	/*
	 * media_guest pool: guest-physical range the host SHARE'd for buffers
	 * the guest fills (OUTPUT queues by default), carved with drm_buddy.
	 * guest_pool_ready is cleared under guest_pool_lock at the final
	 * teardown (the v4l2_dev release, once the last file handle and
	 * mapping are gone -- not at remove, D66) so a late free sees the
	 * closed gate instead of a dead allocator.
	 */
	bool guest_pool_ready;
	phys_addr_t guest_pool_base;
	u64 guest_pool_size;
	struct drm_buddy guest_pool_mm;
	struct mutex guest_pool_lock;

	/* Buffer for event descriptors. */
	void *event_buffer;

	/* List of active decoding sessions */
	struct list_head sessions;
	/* Protects `sessions` */
	struct mutex sessions_lock;

	/* Make sure we don't have two threads processing events at the same time */
	struct mutex events_process_lock;

	union {
		struct virtio_media_cmd_open open;
		struct virtio_media_cmd_munmap munmap;
	} cmd;

	union {
		struct virtio_media_resp_open open;
		struct virtio_media_resp_munmap munmap;
	} resp;

	/* Protects `cmd_buf` and `resp_buf` */
	struct mutex bufs_lock;

	/* Used to serialize all virtio commands */
	struct mutex vlock;

	/* Waitqueue for host responses on the command queue */
	wait_queue_head_t wq;

	/*
	 * Set under vlock by virtio_media_remove() when the virtio device is
	 * unbound (sysfs unbind or hot unplug). Every command sender checks
	 * it before touching the virtqueues, so nothing reaches a reset or
	 * deleted queue; the struct itself, the sessions and the pool
	 * allocator stay allocated until the v4l2_dev refcount drops to zero
	 * -- the last file handle or mapping -- so a late ioctl dereferences
	 * live memory and fails with -ENODEV instead of oopsing (D66,
	 * B12-acceptance section 15).
	 */
	bool disconnected;
};

static inline struct virtio_media *
to_virtio_media(struct video_device *video_dev)
{
	return container_of(video_dev, struct virtio_media, video_dev);
}

/* virtio_media_driver.c */

int virtio_media_send_command(struct virtio_media *vv, struct scatterlist **sgs,
			      const size_t out_sgs, const size_t in_sgs,
			      size_t minimum_resp_len, size_t *resp_len);
void virtio_media_process_events(struct virtio_media *vv);

/* virtio_media_ioctls.c */

long virtio_media_device_ioctl(struct file *file, unsigned int cmd,
			       unsigned long arg);
extern const struct v4l2_ioctl_ops virtio_media_ioctl_ops;

#endif // __VIRTIO_MEDIA_H
