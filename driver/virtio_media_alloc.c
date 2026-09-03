// SPDX-License-Identifier: BSD-3-Clause OR GPL-2.0+

/*
 * Driver-owned buffer allocator for virtio-media (VPU_DESIGN.md 5.2).
 *
 * See virtio_media_alloc.h for what a driver-owned buffer is. Two backends:
 *
 *  - media_guest pool: the VMM SHARE'd a guest-physical range with the host
 *    and described it in /reserved-memory. drm_buddy carves it; a contiguous
 *    allocation is asked for first (one SG entry, and the host can build one
 *    udmabuf from it), falling back to a fragmented one.
 *  - no pool: dma_alloc_pages() on the transport's DMA device. On a plain KVM
 *    guest that is system RAM the host reads directly; on a VM that lends its
 *    RAM it lands in the restricted-dma-pool, the one piece of system memory
 *    the host can reach. The whole buffer is tried in one allocation, then in
 *    smaller power-of-two chunks, because one allocation is bounded by
 *    MAX_PAGE_ORDER without CMA and by the swiotlb segment behind a
 *    restricted pool.
 *
 * Either way the result is a precomputed list of guest-physical runs the
 * QBUF path sends to the host verbatim, and a cookie user-space mmaps.
 *
 * Copyright (c) 2026 DroidVM contributors.
 */

#include <linux/dma-mapping.h>
#include <linux/err.h>
#include <linux/mm.h>
#include <linux/module.h>
#include <linux/moduleparam.h>
#include <linux/slab.h>
#include <linux/sort.h>
#include <linux/videodev2.h>
#include <linux/vmalloc.h>

#include "virtio_media.h"
#include "virtio_media_alloc.h"

/*
 * Print one line per driver-owned allocation and release: which backend,
 * cookie, size, how many physical runs. What a bringup reads to see whether
 * OUTPUT buffers really come from the media_guest pool.
 */
static bool pool_debug;
module_param(pool_debug, bool, 0660);
MODULE_PARM_DESC(pool_debug,
		 "log every driver-owned buffer allocation and release");

/*
 * Upper bound on the physical runs of one buffer. Mirrors the host's
 * MAX_SG_ENTRIES (VPU_DESIGN.md 4.3): a buffer more fragmented than this is
 * refused rather than sent as a command the host would reject.
 */
#define VMEDIA_DBUF_MAX_ENTS 4096

int vmedia_dbuf_plane_sizes(const struct v4l2_format *f,
			    size_t sizes[VIDEO_MAX_PLANES], u32 *num_planes)
{
	u32 i;

	switch (f->type) {
	case V4L2_BUF_TYPE_VIDEO_CAPTURE:
	case V4L2_BUF_TYPE_VIDEO_OUTPUT:
		*num_planes = 1;
		sizes[0] = f->fmt.pix.sizeimage;
		break;
	case V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE:
	case V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE:
		*num_planes = f->fmt.pix_mp.num_planes;
		if (*num_planes == 0 || *num_planes > VIDEO_MAX_PLANES)
			return -EINVAL;
		for (i = 0; i < *num_planes; i++)
			sizes[i] = f->fmt.pix_mp.plane_fmt[i].sizeimage;
		break;
	case V4L2_BUF_TYPE_SDR_CAPTURE:
	case V4L2_BUF_TYPE_SDR_OUTPUT:
		*num_planes = 1;
		sizes[0] = f->fmt.sdr.buffersize;
		break;
	case V4L2_BUF_TYPE_META_CAPTURE:
	case V4L2_BUF_TYPE_META_OUTPUT:
		*num_planes = 1;
		sizes[0] = f->fmt.meta.buffersize;
		break;
	default:
		return -EINVAL;
	}

	for (i = 0; i < *num_planes; i++)
		if (sizes[i] == 0)
			return -EINVAL;

	return 0;
}

/* A physical run while the SG list is being assembled. */
struct vmedia_run {
	u64 start;
	u64 len;
};

static int vmedia_run_cmp(const void *a, const void *b)
{
	const struct vmedia_run *ra = a, *rb = b;

	if (ra->start < rb->start)
		return -1;
	return ra->start > rb->start;
}

/**
 * Turn @nruns physical runs into the buffer's SG list: sort them, merge the
 * adjacent ones, and cut the total down to exactly @dbuf->len the way
 * sg_alloc_table_from_pages() does for a real USERPTR buffer.
 */
static int vmedia_dbuf_build_sg(struct vmedia_dbuf *dbuf,
				struct vmedia_run *runs, u32 nruns)
{
	u64 remaining = dbuf->len;
	u32 i, n = 0;

	sort(runs, nruns, sizeof(*runs), vmedia_run_cmp, NULL);

	/* Merge adjacent runs in place. */
	for (i = 0; i < nruns; i++) {
		if (n > 0 && runs[n - 1].start + runs[n - 1].len == runs[i].start)
			runs[n - 1].len += runs[i].len;
		else
			runs[n++] = runs[i];
	}

	/* Trim to the usable length. */
	nruns = n;
	n = 0;
	for (i = 0; i < nruns && remaining > 0; i++) {
		if (runs[i].len > remaining)
			runs[i].len = remaining;
		remaining -= runs[i].len;
		n++;
	}
	if (remaining > 0 || n == 0)
		return -EINVAL;
	if (n > VMEDIA_DBUF_MAX_ENTS)
		return -ENOMEM;

	dbuf->sg = kvmalloc_array(n, sizeof(*dbuf->sg), GFP_KERNEL | __GFP_ZERO);
	if (!dbuf->sg)
		return -ENOMEM;
	for (i = 0; i < n; i++) {
		if (runs[i].len > U32_MAX)
			return -EINVAL;
		dbuf->sg[i].start = runs[i].start;
		dbuf->sg[i].len = runs[i].len;
	}
	dbuf->nents = n;

	return 0;
}

/*
 * Pool mode: drm_buddy over the media_guest range, contiguous first.
 */
static int vmedia_dbuf_alloc_pool(struct virtio_media *vv,
				  struct vmedia_dbuf *dbuf, size_t size)
{
	struct drm_buddy_block *block;
	struct vmedia_run *runs;
	u32 nruns = 0;
	int ret;

	mutex_lock(&vv->guest_pool_lock);
	if (!vv->guest_pool_ready) {
		mutex_unlock(&vv->guest_pool_lock);
		return -ENODEV;
	}
	ret = drm_buddy_alloc_blocks(&vv->guest_pool_mm, 0, vv->guest_pool_size,
				     size, PAGE_SIZE, &dbuf->blocks,
				     DRM_BUDDY_CONTIGUOUS_ALLOCATION);
	if (ret == 0) {
		dbuf->contiguous = true;
	} else {
		ret = drm_buddy_alloc_blocks(&vv->guest_pool_mm, 0,
					     vv->guest_pool_size, size,
					     PAGE_SIZE, &dbuf->blocks, 0);
		dbuf->contiguous = false;
	}
	mutex_unlock(&vv->guest_pool_lock);
	if (ret)
		return -ENOMEM;
	dbuf->pool = true;

	list_for_each_entry(block, &dbuf->blocks, link)
		nruns++;
	runs = kvmalloc_array(nruns, sizeof(*runs), GFP_KERNEL);
	if (!runs)
		return -ENOMEM;
	nruns = 0;
	list_for_each_entry(block, &dbuf->blocks, link) {
		runs[nruns].start = vv->guest_pool_base +
				    drm_buddy_block_offset(block);
		runs[nruns].len = drm_buddy_block_size(&vv->guest_pool_mm,
						       block);
		nruns++;
	}
	ret = vmedia_dbuf_build_sg(dbuf, runs, nruns);
	kvfree(runs);
	return ret;
}

/*
 * No-pool mode: dma_alloc_pages(), whole buffer first, then halving chunks.
 */
static int vmedia_dbuf_alloc_dma(struct virtio_media *vv,
				 struct vmedia_dbuf *dbuf, size_t size)
{
	u32 max_chunks = min_t(size_t, size >> PAGE_SHIFT, VMEDIA_DBUF_MAX_ENTS);
	struct vmedia_dbuf_chunk *chunks;
	struct vmedia_run *runs;
	size_t remaining = size;
	size_t chunk = size;
	u32 i, n = 0;
	int ret;

	chunks = kvmalloc_array(max_chunks, sizeof(*chunks), GFP_KERNEL);
	if (!chunks)
		return -ENOMEM;

	while (remaining > 0) {
		size_t this = min(chunk, remaining);
		dma_addr_t dma;
		struct page *page;

		if (n == max_chunks) {
			ret = -ENOMEM;
			goto err_free;
		}
		page = dma_alloc_pages(vv->dma_dev, this, &dma,
				       DMA_BIDIRECTIONAL,
				       GFP_KERNEL | __GFP_NOWARN);
		if (page) {
			chunks[n].pages = page;
			chunks[n].dma = dma;
			chunks[n].size = this;
			n++;
			remaining -= this;
			continue;
		}
		if (chunk <= PAGE_SIZE) {
			ret = -ENOMEM;
			goto err_free;
		}
		/* Next power of two below what just failed. */
		chunk = rounddown_pow_of_two(chunk - 1);
	}

	/* Keep an exact-size array; max_chunks was sized for the worst case. */
	dbuf->chunks = kvmalloc_array(n, sizeof(*chunks), GFP_KERNEL);
	if (!dbuf->chunks) {
		ret = -ENOMEM;
		goto err_free;
	}
	memcpy(dbuf->chunks, chunks, n * sizeof(*chunks));
	dbuf->nchunks = n;
	kvfree(chunks);
	dbuf->pool = false;
	dbuf->contiguous = n == 1;

	runs = kvmalloc_array(n, sizeof(*runs), GFP_KERNEL);
	if (!runs)
		return -ENOMEM;
	for (i = 0; i < n; i++) {
		runs[i].start = page_to_phys(dbuf->chunks[i].pages);
		runs[i].len = dbuf->chunks[i].size;
	}
	ret = vmedia_dbuf_build_sg(dbuf, runs, n);
	kvfree(runs);
	return ret;

err_free:
	for (i = 0; i < n; i++)
		dma_free_pages(vv->dma_dev, chunks[i].size, chunks[i].pages,
			       chunks[i].dma, DMA_BIDIRECTIONAL);
	kvfree(chunks);
	return ret;
}

static void vmedia_dbuf_release(struct vmedia_dbuf *dbuf)
{
	struct virtio_media *vv = dbuf->vv;
	u32 i;

	if (pool_debug)
		pr_info("virtio-media: dbuf free cookie %#llx len %zu (%s)\n",
			dbuf->cookie, dbuf->len,
			dbuf->pool ? "media_guest" : "dma_alloc_pages");

	if (dbuf->pool) {
		mutex_lock(&vv->guest_pool_lock);
		/*
		 * A mapping that outlived device removal finds the gate
		 * closed: leave the block list alone rather than touch a
		 * dead allocator.
		 */
		if (vv->guest_pool_ready)
			vmedia_drm_buddy_free_list(&vv->guest_pool_mm,
						   &dbuf->blocks);
		mutex_unlock(&vv->guest_pool_lock);
	} else {
		for (i = 0; i < dbuf->nchunks; i++)
			dma_free_pages(vv->dma_dev, dbuf->chunks[i].size,
				       dbuf->chunks[i].pages,
				       dbuf->chunks[i].dma, DMA_BIDIRECTIONAL);
		kvfree(dbuf->chunks);
	}
	kvfree(dbuf->sg);
	kfree(dbuf);
}

struct vmedia_dbuf *vmedia_dbuf_alloc(struct virtio_media *vv,
				      struct virtio_media_session *session,
				      size_t len)
{
	struct vmedia_dbuf *dbuf;
	size_t size;
	int ret;

	if (len == 0)
		return ERR_PTR(-EINVAL);
	size = PAGE_ALIGN(len);
	if (size < len)
		return ERR_PTR(-EINVAL);
	/* Cookies are 32-bit m.offset values. */
	if (session->next_dbuf_cookie + PAGE_SIZE - 1 > U32_MAX)
		return ERR_PTR(-ENOSPC);

	dbuf = kzalloc(sizeof(*dbuf), GFP_KERNEL);
	if (!dbuf)
		return ERR_PTR(-ENOMEM);
	dbuf->vv = vv;
	dbuf->len = len;
	INIT_LIST_HEAD(&dbuf->blocks);
	refcount_set(&dbuf->maps, 1);

	if (vv->guest_pool_ready)
		ret = vmedia_dbuf_alloc_pool(vv, dbuf, size);
	else
		ret = vmedia_dbuf_alloc_dma(vv, dbuf, size);
	if (ret) {
		/* Partial state is released by the same path as a full buffer. */
		vmedia_dbuf_release(dbuf);
		return ERR_PTR(ret);
	}

	dbuf->cookie = session->next_dbuf_cookie;
	session->next_dbuf_cookie += PAGE_SIZE;

	if (pool_debug)
		pr_info("virtio-media: dbuf alloc session %u cookie %#llx len %zu from %s%s, %u run%s, first %#llx\n",
			session->id, dbuf->cookie, dbuf->len,
			dbuf->pool ? "media_guest" : "dma_alloc_pages",
			dbuf->contiguous ? " (contiguous)" : " (fragmented)",
			dbuf->nents, dbuf->nents == 1 ? "" : "s",
			dbuf->sg[0].start);

	return dbuf;
}

void vmedia_dbuf_put(struct vmedia_dbuf *dbuf)
{
	if (refcount_dec_and_test(&dbuf->maps))
		vmedia_dbuf_release(dbuf);
}

struct vmedia_dbuf *vmedia_dbuf_lookup(struct virtio_media_session *session,
				       u64 cookie)
{
	int q, i, p;

	for (q = 0; q <= VIRTIO_MEDIA_LAST_QUEUE; q++) {
		struct virtio_media_queue_state *queue = &session->queues[q];

		for (i = 0; i < queue->allocated_bufs; i++) {
			for (p = 0; p < VIDEO_MAX_PLANES; p++) {
				struct vmedia_dbuf *dbuf =
					queue->buffers[i].dbuf[p];

				if (dbuf && dbuf->cookie == cookie)
					return dbuf;
			}
		}
	}

	return NULL;
}

static void vmedia_dbuf_vma_open(struct vm_area_struct *vma)
{
	struct vmedia_dbuf *dbuf = vma->vm_private_data;

	refcount_inc(&dbuf->maps);
}

static void vmedia_dbuf_vma_close(struct vm_area_struct *vma)
{
	vmedia_dbuf_put(vma->vm_private_data);
}

static const struct vm_operations_struct vmedia_dbuf_vm_ops = {
	.open = vmedia_dbuf_vma_open,
	.close = vmedia_dbuf_vma_close,
};

int vmedia_dbuf_mmap(struct vmedia_dbuf *dbuf, struct vm_area_struct *vma)
{
	unsigned long remaining = vma->vm_end - vma->vm_start;
	unsigned long uaddr = vma->vm_start;
	u32 i;
	int ret;

	if (remaining > PAGE_ALIGN(dbuf->len))
		return -EINVAL;

	/*
	 * Every SG entry starts page aligned and only the last one is cut
	 * short of a page multiple, so the entries rounded up cover exactly
	 * the allocation. The pages are cacheable, like the media_host pool
	 * and virtio-gpu's guest pool: the host is a CPU reading the same
	 * memory, coherent with these caches.
	 */
	for (i = 0; i < dbuf->nents && remaining > 0; i++) {
		unsigned long run = min_t(unsigned long,
					  PAGE_ALIGN(dbuf->sg[i].len),
					  remaining);

		if (dbuf->pool)
			ret = io_remap_pfn_range(vma, uaddr,
						 dbuf->sg[i].start >> PAGE_SHIFT,
						 run, vma->vm_page_prot);
		else
			ret = remap_pfn_range(vma, uaddr,
					      dbuf->sg[i].start >> PAGE_SHIFT,
					      run, vma->vm_page_prot);
		if (ret)
			return ret;
		uaddr += run;
		remaining -= run;
	}

	/*
	 * Only now: a failure above leaves vm_ops unset so the core does not
	 * call .close for a reference that was never taken.
	 */
	refcount_inc(&dbuf->maps);
	vma->vm_private_data = dbuf;
	vma->vm_ops = &vmedia_dbuf_vm_ops;

	return 0;
}

static void vmedia_buffer_put_dbufs(struct virtio_media_buffer *buffer)
{
	u32 p;

	for (p = 0; p < VIDEO_MAX_PLANES; p++) {
		if (buffer->dbuf[p]) {
			vmedia_dbuf_put(buffer->dbuf[p]);
			buffer->dbuf[p] = NULL;
		}
	}
}

int vmedia_queue_alloc_dbufs(struct virtio_media *vv,
			     struct virtio_media_session *session,
			     struct virtio_media_queue_state *queue, u32 first,
			     u32 count, const size_t sizes[VIDEO_MAX_PLANES],
			     u32 num_planes)
{
	u32 i, p;

	for (i = first; i < first + count; i++) {
		struct virtio_media_buffer *buffer = &queue->buffers[i];

		for (p = 0; p < num_planes; p++) {
			struct vmedia_dbuf *dbuf =
				vmedia_dbuf_alloc(vv, session, sizes[p]);

			if (IS_ERR(dbuf)) {
				int ret = PTR_ERR(dbuf);

				pr_warn("virtio-media: driver-owned buffer allocation of %zu bytes (buffer %u plane %u) failed: %d\n",
					sizes[p], i, p, ret);
				/*
				 * Release what this call allocated: the buffer
				 * array is zeroed, so every non-NULL plane of
				 * buffers first..i is ours.
				 */
				while (i >= first) {
					vmedia_buffer_put_dbufs(&queue->buffers[i]);
					if (i == first)
						break;
					i--;
				}
				return ret;
			}
			buffer->dbuf[p] = dbuf;
		}
	}

	return 0;
}

void vmedia_queue_put_dbufs(struct virtio_media_queue_state *queue)
{
	size_t i;

	for (i = 0; i < queue->allocated_bufs; i++)
		vmedia_buffer_put_dbufs(&queue->buffers[i]);
}
