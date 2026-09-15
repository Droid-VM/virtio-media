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

#include <linux/device.h>
#include <linux/dma-buf.h>
#include <linux/dma-map-ops.h>
#include <linux/dma-mapping.h>
#include <linux/err.h>
#include <linux/io.h>
#include <linux/mm.h>
#include <linux/module.h>
#include <linux/moduleparam.h>
#include <linux/slab.h>
#include <linux/sort.h>
#include <linux/swiotlb.h>
#include <linux/videodev2.h>
#include <linux/vmalloc.h>

#include "virtio_media.h"
#include "virtio_media_alloc.h"

/* The DMABUF import path (VPU_DESIGN.md 7.7) calls dma_buf_* symbols. */
MODULE_IMPORT_NS("DMA_BUF");

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

bool vmedia_dbuf_type_supported(u32 type)
{
	switch (type) {
	case V4L2_BUF_TYPE_VIDEO_CAPTURE:
	case V4L2_BUF_TYPE_VIDEO_OUTPUT:
	case V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE:
	case V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE:
	case V4L2_BUF_TYPE_SDR_CAPTURE:
	case V4L2_BUF_TYPE_SDR_OUTPUT:
	case V4L2_BUF_TYPE_META_CAPTURE:
	case V4L2_BUF_TYPE_META_OUTPUT:
		return true;
	default:
		return false;
	}
}

void vmedia_dbuf_buffer_to_host(struct v4l2_buffer *b,
				struct vmedia_dbuf *const *dbufs)
{
	u32 i;

	b->memory = V4L2_MEMORY_USERPTR;
	if (V4L2_TYPE_IS_MULTIPLANAR(b->type)) {
		for (i = 0; i < b->length && i < VIDEO_MAX_PLANES; i++) {
			if (!dbufs[i])
				continue;
			b->m.planes[i].m.userptr = dbufs[i]->cookie;
			b->m.planes[i].length = dbufs[i]->len;
		}
	} else if (dbufs[0]) {
		b->m.userptr = dbufs[0]->cookie;
		b->length = dbufs[0]->len;
	}
}

void vmedia_dbuf_buffer_from_host(struct v4l2_buffer *b,
				  struct v4l2_plane *planes, u32 max_planes,
				  struct vmedia_dbuf *const *dbufs)
{
	u32 i;

	b->memory = V4L2_MEMORY_MMAP;
	if (V4L2_TYPE_IS_MULTIPLANAR(b->type)) {
		for (i = 0; i < b->length && i < max_planes &&
			    i < VIDEO_MAX_PLANES;
		     i++) {
			if (!dbufs[i])
				continue;
			planes[i].m.mem_offset = dbufs[i]->cookie;
			planes[i].length = dbufs[i]->len;
		}
	} else if (dbufs[0]) {
		b->m.offset = dbufs[0]->cookie;
		b->length = dbufs[0]->len;
	}
}

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
	/* May free @vv itself: nothing above this touches it afterwards. */
	v4l2_device_put(&vv->v4l2_dev);
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
	/*
	 * A dbuf can outlive its session's file handle (a VMA holds a map
	 * reference) and even the driver binding (sysfs unbind, D66): pin
	 * the device so @vv, the guest pool allocator and @dma_dev are
	 * still there when the last reference drops. Released at the end of
	 * vmedia_dbuf_release(), which every path out of here goes through.
	 */
	v4l2_device_get(&vv->v4l2_dev);

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

				/*
				 * Rate-limited: REQBUFS is a client-driven
				 * loop, and an exhausted pool answers every
				 * turn of it. An unrated line here printed
				 * 2 520 copies of itself in 2.3 s under the
				 * D46 stress and pushed the rest of the run
				 * out of the kernel ring (D51, B9-acceptance
				 * §12). One line still says what failed;
				 * printk's suppression counter says how many
				 * more there were.
				 */
				pr_warn_ratelimited(
					"virtio-media: driver-owned buffer allocation of %zu bytes (buffer %u plane %u) failed: %d\n",
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

/*
 * DMABUF import: the V4L2_MEMORY_DMABUF flavour of the driver-owned
 * substitution (VPU_DESIGN.md 7.7). See virtio_media_alloc.h for why the SG
 * list is built from sg_dma_address/sg_dma_len and not from sg_page, and why
 * the attachment goes to the resolver device below and not to the virtio
 * device (D90: the protected VM's restricted DMA pool refuses resource
 * mappings).
 */

/**
 * struct vmedia_import_dev - The dma-buf resolver device.
 *
 * A bare struct device, device_initialize()d and never device_add()ed: it
 * exists only as a DMA mapping identity for dma_buf_attach(), so it needs
 * refcounting, a name and DMA fields, not sysfs, PM or a bus. Staying
 * unregistered is also what guarantees its DMA mode: no OF node, no bus, no
 * driver bind means no code path -- of_dma_configure(),
 * arch_setup_dma_ops(), of_reserved_mem device init -- ever assigns it
 * dma_ops, a dma_range_map or a restricted swiotlb pool.
 * device_initialize() leaves it on dma-direct with the default io_tlb_mem
 * (not force-bounce), and dma_map_resource()/dma_map_sgtable() on it return
 * physical addresses unchanged. It also takes no parent: an import held by
 * an open fd outlives a D66 unbind, so the resolver outlives the virtio
 * device, and with no sysfs entry and no parent pointer there is no teardown
 * ordering against the transport to get wrong. @dma_parms only backs
 * dma_set_max_seg_size(): an exporter must not have to split its sgt for us.
 */
struct vmedia_import_dev {
	struct device dev;
	struct device_dma_parameters dma_parms;
};

static void vmedia_import_dev_release(struct device *dev)
{
	kfree(container_of(dev, struct vmedia_import_dev, dev));
}

int vmedia_import_dev_create(struct virtio_media *vv)
{
	struct vmedia_import_dev *idev;
	struct device *dev;
	int ret;

	idev = kzalloc(sizeof(*idev), GFP_KERNEL);
	if (!idev)
		return -ENOMEM;

	dev = &idev->dev;
	device_initialize(dev);
	dev->release = vmedia_import_dev_release;
	dev->dma_parms = &idev->dma_parms;

	ret = dev_set_name(dev, "%s-dmabuf", dev_name(&vv->virtio_dev->dev));
	if (ret)
		goto err_put;

	ret = dma_coerce_mask_and_coherent(dev, DMA_BIT_MASK(64));
	if (ret)
		goto err_put;
	dma_set_max_seg_size(dev, UINT_MAX);

	/*
	 * Both must hold or D90 is back: dma_ops NULL means dma-direct, and a
	 * non-force-bounce device is what lets dma_direct_map_phys() pass a
	 * DMA_ATTR_MMIO (resource) address through as-is.
	 */
	if (get_dma_ops(dev) || is_swiotlb_force_bounce(dev))
		pr_warn("virtio-media: dma-buf resolver %s is NOT dma-direct (dma_ops %s, swiotlb force-bounce %d): DMABUF imports of non-RAM exporters will fail\n",
			dev_name(dev), get_dma_ops(dev) ? "set" : "null",
			is_swiotlb_force_bounce(dev));
	else
		pr_info("virtio-media: dma-buf resolver %s: dma-direct, no swiotlb force-bounce; imported SG addresses are guest-physical\n",
			dev_name(dev));

	vv->import_dev = dev;
	return 0;

err_put:
	/* Runs the release; frees @idev. */
	put_device(dev);
	return ret;
}

void vmedia_import_dev_destroy(struct virtio_media *vv)
{
	if (!vv->import_dev)
		return;

	/* Never device_add()ed, so the last put is the whole teardown. */
	put_device(vv->import_dev);
	vv->import_dev = NULL;
}

struct vmedia_dmabuf *vmedia_dmabuf_import(struct virtio_media *vv, int fd,
					   size_t size, u32 data_offset)
{
	struct vmedia_dmabuf *import;
	struct virtio_media_sg_entry *ents;
	struct scatterlist *sg;
	size_t skip = data_offset;
	size_t need = size;
	u32 nents = 0, out = 0;
	int ret, i;

	if (size == 0)
		return ERR_PTR(-EINVAL);

	import = kzalloc(sizeof(*import), GFP_KERNEL);
	if (!import)
		return ERR_PTR(-ENOMEM);
	import->vv = vv;
	import->size = size;
	import->fd = fd;

	/* A non-dma-buf fd is a client error, reported as such before the host. */
	import->dmabuf = dma_buf_get(fd);
	if (IS_ERR(import->dmabuf)) {
		import->dmabuf = NULL;
		ret = -EINVAL;
		goto err_free;
	}

	/*
	 * The dma-buf must cover the plane; a larger one is fine, the extra is
	 * ignored. A smaller one cannot back a buffer of the format's size.
	 */
	if (import->dmabuf->size < (u64)data_offset + size) {
		ret = -EINVAL;
		goto err_put;
	}

	/*
	 * Attach to the resolver device (direct DMA ops, so the mapping below
	 * yields guest-physical addresses); the transport's DMA device is
	 * only the fallback for a probe that could not create the resolver.
	 */
	import->attach = dma_buf_attach(import->dmabuf,
					vv->import_dev ?: vv->dma_dev);
	if (IS_ERR(import->attach)) {
		ret = PTR_ERR(import->attach);
		import->attach = NULL;
		goto err_put;
	}

	import->sgt = dma_buf_map_attachment(import->attach, DMA_BIDIRECTIONAL);
	if (IS_ERR(import->sgt)) {
		ret = PTR_ERR(import->sgt);
		import->sgt = NULL;
		goto err_detach;
	}

	for_each_sgtable_dma_sg(import->sgt, sg, i)
		nents++;
	if (nents == 0 || nents > VMEDIA_DBUF_MAX_ENTS) {
		ret = nents == 0 ? -EINVAL : -ENOMEM;
		goto err_unmap;
	}

	ents = kvmalloc_array(nents, sizeof(*ents), GFP_KERNEL | __GFP_ZERO);
	if (!ents) {
		ret = -ENOMEM;
		goto err_unmap;
	}

	/*
	 * Trim the DMA runs to [@data_offset, @data_offset + @size): skip the
	 * offset, take the size. The address is the DMA address, which the
	 * resolver device's direct, identity mapping makes the guest-physical
	 * address -- the same wire form the driver-owned USERPTR path builds
	 * from sg_phys().
	 */
	for_each_sgtable_dma_sg(import->sgt, sg, i) {
		dma_addr_t addr = sg_dma_address(sg);
		size_t len = sg_dma_len(sg);

		if (skip >= len) {
			skip -= len;
			continue;
		}
		addr += skip;
		len -= skip;
		skip = 0;
		if (len > need)
			len = need;
		ents[out].start = addr;
		ents[out].len = len;
		out++;
		need -= len;
		if (need == 0)
			break;
	}
	if (need > 0) {
		/* Fragmented shorter than expected despite the size check. */
		kvfree(ents);
		ret = -EINVAL;
		goto err_unmap;
	}

	import->sg = ents;
	import->nents = out;
	return import;

err_unmap:
	dma_buf_unmap_attachment(import->attach, import->sgt,
				 DMA_BIDIRECTIONAL);
err_detach:
	dma_buf_detach(import->dmabuf, import->attach);
err_put:
	dma_buf_put(import->dmabuf);
err_free:
	kfree(import);
	return ERR_PTR(ret);
}

void vmedia_dmabuf_release(struct vmedia_dmabuf *import)
{
	if (!import)
		return;

	kvfree(import->sg);
	if (import->sgt)
		dma_buf_unmap_attachment(import->attach, import->sgt,
					 DMA_BIDIRECTIONAL);
	if (import->attach)
		dma_buf_detach(import->dmabuf, import->attach);
	if (import->dmabuf)
		dma_buf_put(import->dmabuf);
	kfree(import);
}

void vmedia_buffer_put_dmabufs(struct virtio_media_buffer *buffer)
{
	u32 p;

	for (p = 0; p < VIDEO_MAX_PLANES; p++) {
		if (buffer->dmabuf[p]) {
			vmedia_dmabuf_release(buffer->dmabuf[p]);
			buffer->dmabuf[p] = NULL;
		}
	}
}

void vmedia_queue_put_dmabufs(struct virtio_media_queue_state *queue)
{
	size_t i;

	for (i = 0; i < queue->allocated_bufs; i++)
		vmedia_buffer_put_dmabufs(&queue->buffers[i]);
}

void vmedia_dmabuf_buffer_to_host(struct v4l2_buffer *b,
				  struct vmedia_dmabuf *const *imports)
{
	u32 i;

	b->memory = V4L2_MEMORY_USERPTR;
	if (V4L2_TYPE_IS_MULTIPLANAR(b->type)) {
		for (i = 0; i < b->length && i < VIDEO_MAX_PLANES; i++) {
			if (!imports[i])
				continue;
			/*
			 * Non-zero so scatterlist_filler_add_buffer()'s USERPTR
			 * fixup keeps the plane length (a zero userptr zeroes
			 * it); the value itself is opaque, the host uses the SG
			 * list. data_offset is zeroed: the SG already starts at
			 * it.
			 */
			b->m.planes[i].m.userptr = imports[i]->size;
			b->m.planes[i].length = imports[i]->size;
			b->m.planes[i].data_offset = 0;
		}
	} else if (imports[0]) {
		b->m.userptr = imports[0]->size;
		b->length = imports[0]->size;
	}
}

void vmedia_dmabuf_buffer_from_host(struct v4l2_buffer *b,
				    struct v4l2_plane *planes, u32 max_planes,
				    struct vmedia_dmabuf *const *imports)
{
	u32 i;

	b->memory = V4L2_MEMORY_DMABUF;
	if (V4L2_TYPE_IS_MULTIPLANAR(b->type)) {
		for (i = 0; i < b->length && i < max_planes &&
			    i < VIDEO_MAX_PLANES;
		     i++) {
			if (!imports[i])
				continue;
			planes[i].m.fd = imports[i]->fd;
			planes[i].length = imports[i]->size;
		}
	} else if (imports[0]) {
		b->m.fd = imports[0]->fd;
		b->length = imports[0]->size;
	}
}

void vmedia_dmabuf_warn_host_refused(const struct virtio_media_buffer *buffer,
				     int err)
{
	u32 p;

	for (p = 0; p < VIDEO_MAX_PLANES; p++) {
		const struct vmedia_dmabuf *import = buffer->dmabuf[p];
		u64 start, end;

		if (!import || import->nents == 0)
			continue;
		start = import->sg[0].start;
		end = import->sg[import->nents - 1].start +
		      import->sg[import->nents - 1].len;
		/*
		 * Rate-limited (D51): a client that keeps re-queueing an
		 * unreachable buffer answers every turn of its loop with this
		 * line. The guest mapped the range fine; the host refused it
		 * because it lies outside its SHARE'd windows -- in a pVM the
		 * expected answer for plain RAM pages (e.g. a udmabuf over a
		 * memfd), while SHARE'd pool memory (a GBM bo in the
		 * gpu-guest pool) is accepted.
		 */
		pr_warn_ratelimited(
			"virtio-media: host refused DMABUF plane %u (exporter %s, phys %#llx-%#llx, %u runs): %d -- range outside the host's SHARE'd windows\n",
			p, import->dmabuf->exp_name ?: "?", start, end,
			import->nents, err);
	}
}

/*
 * Bounce buffers for ioctl payloads the host must touch directly (D34).
 *
 * Pool mode wants one physically contiguous run so a single SG entry
 * describes it, and a kernel mapping to copy the user data through;
 * memremap() gives the same cacheable view of the pool that
 * vmedia_dbuf_mmap() gives user-space. Without a pool the device is in-VMM
 * and reads all guest RAM, so a plain physically contiguous kernel buffer is
 * enough.
 */
struct vmedia_bounce *vmedia_bounce_alloc(struct virtio_media *vv, size_t len)
{
	struct vmedia_bounce *bounce;
	size_t size;
	int ret;

	if (len == 0 || len > VMEDIA_BOUNCE_MAX_SIZE)
		return ERR_PTR(-EINVAL);

	bounce = kzalloc(sizeof(*bounce), GFP_KERNEL);
	if (!bounce)
		return ERR_PTR(-ENOMEM);
	bounce->vv = vv;
	bounce->len = len;
	INIT_LIST_HEAD(&bounce->blocks);

	mutex_lock(&vv->guest_pool_lock);
	if (vv->guest_pool_ready) {
		struct drm_buddy_block *block;
		u64 offset = U64_MAX;

		size = PAGE_ALIGN(len);
		ret = drm_buddy_alloc_blocks(&vv->guest_pool_mm, 0,
					     vv->guest_pool_size, size,
					     PAGE_SIZE, &bounce->blocks,
					     DRM_BUDDY_CONTIGUOUS_ALLOCATION);
		if (ret) {
			mutex_unlock(&vv->guest_pool_lock);
			kfree(bounce);
			return ERR_PTR(-ENOMEM);
		}
		/*
		 * A contiguous allocation may still come as several adjacent
		 * blocks; the run starts at the lowest one.
		 */
		list_for_each_entry(block, &bounce->blocks, link)
			offset = min(offset, drm_buddy_block_offset(block));
		bounce->pool = true;
		bounce->size = size;
		bounce->phys = vv->guest_pool_base + offset;
	}
	mutex_unlock(&vv->guest_pool_lock);

	if (bounce->pool) {
		bounce->vaddr = memremap(bounce->phys, bounce->size,
					 MEMREMAP_WB);
		if (!bounce->vaddr)
			goto err_backing;
	} else {
		bounce->size = len;
		bounce->vaddr = kmalloc(len, GFP_KERNEL);
		if (!bounce->vaddr)
			goto err_backing;
		bounce->phys = virt_to_phys(bounce->vaddr);
	}

	return bounce;

err_backing:
	if (bounce->pool) {
		mutex_lock(&vv->guest_pool_lock);
		if (vv->guest_pool_ready)
			vmedia_drm_buddy_free_list(&vv->guest_pool_mm,
						   &bounce->blocks);
		mutex_unlock(&vv->guest_pool_lock);
	}
	kfree(bounce);
	return ERR_PTR(-ENOMEM);
}

void vmedia_bounce_free(struct vmedia_bounce *bounce)
{
	struct virtio_media *vv;

	if (!bounce)
		return;
	vv = bounce->vv;

	if (bounce->pool) {
		memunmap(bounce->vaddr);
		mutex_lock(&vv->guest_pool_lock);
		/* Same closed-gate rule as vmedia_dbuf_release(). */
		if (vv->guest_pool_ready)
			vmedia_drm_buddy_free_list(&vv->guest_pool_mm,
						   &bounce->blocks);
		mutex_unlock(&vv->guest_pool_lock);
	} else {
		kfree(bounce->vaddr);
	}
	kfree(bounce);
}
