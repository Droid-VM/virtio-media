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

/*
 * DMABUF import (VPU_DESIGN.md 7.7). A V4L2_MEMORY_DMABUF buffer is the third
 * flavour of the driver-owned substitution: like a driver-owned MMAP buffer it
 * travels to the host as USERPTR with a guest-physical SG list, but the memory
 * is not allocated by the driver -- it is a dma-buf the guest handed in at
 * QBUF time (a GBM/virtio-gpu surface, another V4L2 node, udmabuf, ...). The
 * import is attached to the per-device dma-buf resolver (virtio_media.h
 * @import_dev) and its sg_table is read through sg_dma_address/sg_dma_len
 * only: the virtio-gpu vram exporter's sgt has no struct pages
 * (virtgpu_vram.c virtio_gpu_vram_map_dma_buf uses sg_set_page(sg, NULL,
 * ...)), so sg_phys()/sg_page() would fault.
 *
 * Why a resolver device and not the transport's DMA device: in a protected VM
 * every virtio device is bound to a restricted DMA pool (swiotlb
 * force-bounce), and dma_direct_map_phys() refuses DMA_ATTR_MMIO on a
 * force-bounce device -- an MMIO range cannot be bounced -- so attaching to
 * the virtio device made every virtio-gpu vram import fail -EIO (D90,
 * B21-acceptance section 3). The resolver has direct DMA ops and no
 * restricted pool, so mapping through it returns the physical address
 * unchanged: sg_dma_address == the guest-physical address, exactly the wire
 * form the driver-owned USERPTR path builds from sg_phys()
 * (scatterlist_filler.c prepare_userptr_to_host). Whether the host may
 * actually touch a range is not the guest's DMA layer's question: the host
 * checks the SG list against its SHARE'd access windows and answers EFAULT
 * for anything outside them. A udmabuf over plain RAM pages therefore now
 * maps fine in the guest and is refused by the host in a pVM -- the correct
 * split (the exporter is not this driver's business, the windows are the
 * host's).
 */

/**
 * vmedia_import_dev_create - Create the dma-buf resolver device for @vv.
 *
 * Initializes a bare struct device (never added to sysfs or a bus) with a
 * 64-bit DMA mask and direct DMA ops, prints one line naming it and its DMA
 * mode, and stores it in @vv->import_dev. On failure @vv->import_dev stays
 * NULL and imports fall back to @vv->dma_dev (the r24 behaviour: RAM-backed
 * dma-bufs import, vram exporters fail).
 */
int vmedia_import_dev_create(struct virtio_media *vv);

/**
 * vmedia_import_dev_destroy - Drop the resolver device (NULL is allowed).
 * Called from the final teardown, after the last session -- and so the last
 * import -- is gone.
 */
void vmedia_import_dev_destroy(struct virtio_media *vv);

/**
 * struct vmedia_dmabuf - One imported DMABUF plane held on behalf of a queued
 * V4L2_MEMORY_DMABUF buffer.
 *
 * @vv: device the import is attached to (through its resolver device).
 * @dmabuf: the dma-buf the fd resolved to.
 * @attach: attachment to @vv's resolver device (@vv->dma_dev when the
 *	resolver could not be created at probe).
 * @sgt: mapped scatter-gather table (DMA_BIDIRECTIONAL).
 * @sg: guest-physical SG list sent to the host, trimmed to @size starting at
 *	the plane's @data_offset; built from sg_dma_address/sg_dma_len.
 * @nents: number of entries in @sg.
 * @size: usable plane size the host is told (the queue's sizeimage), what the
 *	USERPTR length carries on the wire.
 * @fd: the fd the guest submitted, restored into m.fd on the reply/dequeue.
 */
struct vmedia_dmabuf {
	struct virtio_media *vv;
	struct dma_buf *dmabuf;
	struct dma_buf_attachment *attach;
	struct sg_table *sgt;
	struct virtio_media_sg_entry *sg;
	u32 nents;
	size_t size;
	int fd;
};

/**
 * vmedia_dmabuf_import - Import plane @fd for a buffer of @size usable bytes
 * starting at @data_offset within the dma-buf.
 *
 * Returns the import (attachment + mapping + SG list held) or an ERR_PTR:
 * -EINVAL for a non-dma-buf fd or a dma-buf smaller than @data_offset + @size,
 * or the attach/map errno. A dma-buf larger than needed is accepted, the extra
 * ignored. On success the caller owns the import until vmedia_dmabuf_release().
 */
struct vmedia_dmabuf *vmedia_dmabuf_import(struct virtio_media *vv, int fd,
					   size_t size, u32 data_offset);

/**
 * vmedia_dmabuf_release - Unmap, detach and put an import (NULL is allowed).
 */
void vmedia_dmabuf_release(struct vmedia_dmabuf *import);

/**
 * vmedia_buffer_put_dmabufs - Release every imported plane of @buffer and NULL
 * the slots. Called on DQBUF, queue teardown and session close.
 */
void vmedia_buffer_put_dmabufs(struct virtio_media_buffer *buffer);

/**
 * vmedia_queue_put_dmabufs - Release the imports of every buffer the queue
 * still holds (a STREAMOFF/REQBUFS(0) with buffers queued but not dequeued).
 */
void vmedia_queue_put_dmabufs(struct virtio_media_queue_state *queue);

/**
 * vmedia_dmabuf_buffer_to_host - Rewrite @b for the host: memory USERPTR, a
 * non-zero opaque m.userptr (so the USERPTR fixup keeps the plane length) and
 * length the queue's sizeimage, per plane for multiplanar buffers. data_offset
 * is zeroed because the SG list already begins at it. Planes without an import
 * are left untouched.
 */
void vmedia_dmabuf_buffer_to_host(struct v4l2_buffer *b,
				  struct vmedia_dmabuf *const *imports);

/**
 * vmedia_dmabuf_buffer_from_host - Rewrite @b back for user-space: memory
 * DMABUF and m.fd the fd the guest submitted, per plane for multiplanar.
 * @planes: the plane array to patch (b->m.planes may dangle in the kept state).
 * @max_planes: number of entries in @planes.
 */
void vmedia_dmabuf_buffer_from_host(struct v4l2_buffer *b,
				    struct v4l2_plane *planes, u32 max_planes,
				    struct vmedia_dmabuf *const *imports);

/**
 * vmedia_dmabuf_warn_host_refused - Rate-limited line naming each imported
 * plane's exporter and physical range when the host answered EFAULT to a
 * QBUF/PREPARE_BUF carrying them: the range is outside the host's SHARE'd
 * access windows (in a pVM this is the expected answer for e.g. a udmabuf
 * over plain RAM pages, which the guest maps fine but the host may not
 * touch).
 */
void vmedia_dmabuf_warn_host_refused(const struct virtio_media_buffer *buffer,
				     int err);

/*
 * Upper bound on one bounced ioctl payload (defect D34). Compound-control
 * payloads are tens of bytes to a few KiB; anything approaching this is a
 * corrupt size and is refused rather than allowed to drain the pool.
 */
#define VMEDIA_BOUNCE_MAX_SIZE (1 << 20)

/**
 * struct vmedia_bounce - Driver-owned bounce buffer for an ioctl payload.
 *
 * A pool-mode helper can only touch guest memory inside its access windows
 * (the media pools); an arbitrary guest page -- like the user-space payload a
 * compound v4l2_ext_control points to -- is EFAULT for it in both directions
 * (defect D34). A bounce is one physically contiguous, driver-owned run the
 * host may always touch: media_guest pool memory when the VMM built the pool,
 * else a plain kernel buffer (without a pool the device runs in-VMM and reads
 * all guest RAM).
 *
 * @vv: device the bounce was allocated on.
 * @blocks: drm_buddy blocks, pool mode.
 * @vaddr: kernel mapping, for copying user data in and out.
 * @phys: guest-physical start of the run, what the host is told.
 * @size: allocated size, page aligned in pool mode.
 * @len: requested length, what goes on the wire.
 * @pool: true when @blocks back the memory, false for a kernel buffer.
 */
struct vmedia_bounce {
	struct virtio_media *vv;
	struct list_head blocks;
	void *vaddr;
	phys_addr_t phys;
	size_t size;
	size_t len;
	bool pool;
};

/**
 * vmedia_bounce_alloc - Allocate a bounce buffer of @len bytes.
 *
 * Returns the buffer, or an ERR_PTR: -EINVAL for a zero or absurd @len,
 * -ENOMEM when the memory is not there.
 */
struct vmedia_bounce *vmedia_bounce_alloc(struct virtio_media *vv, size_t len);

/**
 * vmedia_bounce_free - Release @bounce (NULL is allowed).
 */
void vmedia_bounce_free(struct vmedia_bounce *bounce);

#endif // __VIRTIO_MEDIA_ALLOC_H
