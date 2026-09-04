// SPDX-License-Identifier: BSD-3-Clause OR GPL-2.0+

/*
 * Driver-owned buffer allocator for virtio-media (VPU_DESIGN.md 5.2).
 *
 * A driver-owned buffer is what user-space sees as a V4L2_MEMORY_MMAP buffer
 * on a queue the guest fills (OUTPUT queues by default): the driver allocates
 * the memory, maps it into user-space itself, and hands it to the host as a
 * USERPTR buffer followed by a guest-physical scatter-gather list. The memory
 * comes from the media_guest pool when the VMM built one (drm_buddy), else
 * from dma_alloc_pages().
 *
 * Copyright (c) 2026 DroidVM contributors.
 */

#ifndef __VIRTIO_MEDIA_ALLOC_H
#define __VIRTIO_MEDIA_ALLOC_H

#include <linux/list.h>
#include <linux/refcount.h>
#include <linux/types.h>

#include "session.h"

struct virtio_media;
struct vm_area_struct;
struct v4l2_format;

/*
 * mmap offsets of driver-owned buffers start here, one page per buffer. The
 * host's MMAP offsets start at 0 and grow one page per buffer as well, so the
 * two spaces cannot collide short of 786432 host buffers. Stays below 4 GiB
 * because v4l2_buffer.m.offset is 32 bits wide.
 */
#define VMEDIA_DBUF_COOKIE_BASE 0xC0000000ULL

/**
 * struct vmedia_dbuf_chunk - One dma_alloc_pages() allocation.
 * @pages: first page of the allocation.
 * @dma: DMA handle, only needed to free.
 * @size: size of the allocation in bytes, page aligned.
 */
struct vmedia_dbuf_chunk {
	struct page *pages;
	dma_addr_t dma;
	size_t size;
};

/**
 * struct vmedia_dbuf - Backing of one driver-owned plane.
 *
 * @vv: device the buffer was allocated on.
 * @blocks: drm_buddy blocks, pool mode.
 * @chunks: dma_alloc_pages() allocations, no-pool mode. One entry when the
 *	whole buffer could be allocated at once, which is the normal case; more
 *	when it could not (a single allocation is bounded by MAX_PAGE_ORDER
 *	without CMA, and by the swiotlb segment behind a restricted-dma-pool).
 * @nchunks: number of entries in @chunks.
 * @cookie: the mmap offset user-space sees in m.offset, >= VMEDIA_DBUF_COOKIE_BASE.
 * @sg: precomputed guest-physical scatter-gather list sent to the host with
 *	every QBUF / PREPARE_BUF; entries sum to @len exactly.
 * @nents: number of entries in @sg.
 * @maps: the owning queue's reference plus one per VMA mapping the buffer;
 *	the memory is released when it drops to zero, so a mapping that
 *	outlives REQBUFS(0) keeps its pages (vm_ops .open/.close).
 * @len: usable length of the buffer, what user-space asked for.
 * @pool: true when @blocks is used, false for @chunks.
 * @contiguous: pool mode only, whether the allocation is one physical run.
 */
struct vmedia_dbuf {
	struct virtio_media *vv;
	struct list_head blocks;
	struct vmedia_dbuf_chunk *chunks;
	u32 nchunks;
	u64 cookie;
	struct virtio_media_sg_entry *sg;
	u32 nents;
	refcount_t maps;
	size_t len;
	bool pool;
	bool contiguous;
};

/**
 * vmedia_dbuf_type_supported - Whether buffers of a queue type can be
 * driver-owned, i.e. whether the type's format tells their size.
 */
bool vmedia_dbuf_type_supported(u32 type);

/**
 * vmedia_dbuf_buffer_to_host - Rewrite @b for the host: memory USERPTR,
 * m.userptr the cookie (opaque, echoed back) and length the buffer size, per
 * plane for multiplanar buffers. Planes without a dbuf are left untouched.
 */
void vmedia_dbuf_buffer_to_host(struct v4l2_buffer *b,
				struct vmedia_dbuf *const *dbufs);

/**
 * vmedia_dbuf_buffer_from_host - Rewrite @b for user-space: memory MMAP,
 * m.offset the cookie and length the buffer size, per plane for multiplanar
 * buffers.
 * @planes: the plane array to patch (b->m.planes may be a dangling pointer in
 *	the buffer state kept for DQBUF, so it is passed explicitly).
 * @max_planes: number of entries in @planes.
 */
void vmedia_dbuf_buffer_from_host(struct v4l2_buffer *b,
				  struct v4l2_plane *planes, u32 max_planes,
				  struct vmedia_dbuf *const *dbufs);

/**
 * vmedia_dbuf_plane_sizes - Plane sizes a queue's current format implies.
 * @f: the format.
 * @sizes: filled with the size of each plane.
 * @num_planes: filled with the number of planes.
 *
 * Returns 0, or -EINVAL for a type that cannot be sized (VBI, overlay) or a
 * format with a zero-sized plane.
 */
int vmedia_dbuf_plane_sizes(const struct v4l2_format *f,
			    size_t sizes[VIDEO_MAX_PLANES], u32 *num_planes);

/**
 * vmedia_dbuf_alloc - Allocate a driver-owned buffer of @len bytes.
 *
 * Returns the buffer with one (owner) reference, or an ERR_PTR: -ENOMEM when
 * neither the pool nor dma_alloc_pages() could provide the memory, -ENOSPC
 * when the session ran out of cookies.
 */
struct vmedia_dbuf *vmedia_dbuf_alloc(struct virtio_media *vv,
				      struct virtio_media_session *session,
				      size_t len);

/**
 * vmedia_dbuf_put - Drop one reference; frees the memory on the last one.
 */
void vmedia_dbuf_put(struct vmedia_dbuf *dbuf);

/**
 * vmedia_dbuf_lookup - Find the driver-owned buffer behind an mmap cookie.
 *
 * Walks the session's queues; called with the device's vlock held so it
 * cannot race a REQBUFS dropping the owner reference.
 */
struct vmedia_dbuf *vmedia_dbuf_lookup(struct virtio_media_session *session,
				       u64 cookie);

/**
 * vmedia_dbuf_mmap - Map a driver-owned buffer into @vma.
 *
 * @vma covers at most PAGE_ALIGN(len). Installs vm_ops that keep the buffer
 * alive for as long as any derived VMA exists.
 */
int vmedia_dbuf_mmap(struct vmedia_dbuf *dbuf, struct vm_area_struct *vma);

/**
 * vmedia_queue_alloc_dbufs - Give buffers [@first, @first + @count) of @queue
 * driver-owned backing for @num_planes planes of @sizes bytes each.
 *
 * On failure every buffer allocated by this call is released again.
 */
int vmedia_queue_alloc_dbufs(struct virtio_media *vv,
			     struct virtio_media_session *session,
			     struct virtio_media_queue_state *queue, u32 first,
			     u32 count, const size_t sizes[VIDEO_MAX_PLANES],
			     u32 num_planes);

/**
 * vmedia_queue_put_dbufs - Drop the queue's reference on every driver-owned
 * buffer it holds. Mappings still held by user-space keep their memory.
 */
void vmedia_queue_put_dbufs(struct virtio_media_queue_state *queue);

#endif // __VIRTIO_MEDIA_ALLOC_H
