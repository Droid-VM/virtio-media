// SPDX-License-Identifier: BSD-3-Clause OR GPL-2.0+

/*
 * Ioctls implementations for the virtio-media driver.
 *
 * Copyright (c) 2023-2024 Google LLC.
 */

#include <linux/overflow.h>
#include <linux/virtio_config.h>
#include <linux/vmalloc.h>
#include <media/v4l2-event.h>
#include <media/v4l2-ioctl.h>

#include "scatterlist_filler.h"
#include "virtio_media.h"
#include "virtio_media_alloc.h"

#include <linux/version.h>

/**
 * Send an ioctl that has no driver payload, but expects a reponse from the host (i.e. an
 * ioctl specified with _IOR).
 *
 * Returns 0 in case of success, or a negative error code.
 */
static int virtio_media_send_r_ioctl(struct v4l2_fh *fh, u32 ioctl,
				     void *ioctl_data, size_t ioctl_data_len)
{
	struct video_device *video_dev = fh->vdev;
	struct virtio_media *vv = to_virtio_media(video_dev);
	struct virtio_media_session *session = fh_to_session(fh);
	struct scatterlist *sgs[3];
	struct scatterlist_filler filler = {
		.descs = session->command_sgs.sgl,
		.num_descs = DESC_CHAIN_MAX_LEN,
		.cur_desc = 0,
		.shadow_buffer = session->shadow_buf,
		.shadow_buffer_size = VIRTIO_SHADOW_BUF_SIZE,
		.shadow_buffer_pos = 0,
		.sgs = sgs,
		.num_sgs = ARRAY_SIZE(sgs),
		.cur_sg = 0,
	};
	int ret;

	/* Command descriptor */
	ret = scatterlist_filler_add_ioctl_cmd(&filler, session, ioctl);
	if (ret)
		return ret;

	/* Response descriptor */
	ret = scatterlist_filler_add_ioctl_resp(&filler, session);
	if (ret)
		return ret;

	/* Response payload */
	ret = scatterlist_filler_add_data(&filler, ioctl_data, ioctl_data_len);
	if (ret) {
		v4l2_err(&vv->v4l2_dev,
			 "failed to prepare command descriptor chain\n");
		return ret;
	}

	ret = virtio_media_send_command(
		vv, sgs, 1, 2,
		sizeof(struct virtio_media_resp_ioctl) + ioctl_data_len, NULL);
	if (ret < 0)
		return ret;

	ret = scatterlist_filler_retrieve_data(session, filler.sgs[2],
					       ioctl_data, ioctl_data_len);
	if (ret) {
		v4l2_err(&vv->v4l2_dev,
			 "failed to retrieve response descriptor chain\n");
		return ret;
	}

	return 0;
}
/**
 * Send an ioctl that does not expect a reply beyond an error status (i.e. an
 * ioctl specified with _IOW) to the host.
 *
 * Returns 0 in case of success, or a negative error code.
 */
static int virtio_media_send_w_ioctl(struct v4l2_fh *fh, u32 ioctl,
				     const void *ioctl_data,
				     size_t ioctl_data_len)
{
	struct video_device *video_dev = fh->vdev;
	struct virtio_media *vv = to_virtio_media(video_dev);
	struct virtio_media_session *session = fh_to_session(fh);
	struct scatterlist *sgs[3];
	struct scatterlist_filler filler = {
		.descs = session->command_sgs.sgl,
		.num_descs = DESC_CHAIN_MAX_LEN,
		.cur_desc = 0,
		.shadow_buffer = session->shadow_buf,
		.shadow_buffer_size = VIRTIO_SHADOW_BUF_SIZE,
		.shadow_buffer_pos = 0,
		.sgs = sgs,
		.num_sgs = ARRAY_SIZE(sgs),
		.cur_sg = 0,
	};
	int ret;

	/* Command descriptor */
	ret = scatterlist_filler_add_ioctl_cmd(&filler, session, ioctl);
	if (ret)
		return ret;

	/* Command payload */
	ret = scatterlist_filler_add_data(&filler, (void *)ioctl_data,
					  ioctl_data_len);
	if (ret) {
		v4l2_err(&vv->v4l2_dev,
			 "failed to prepare command descriptor chain\n");
		return ret;
	}

	/* Response descriptor */
	ret = scatterlist_filler_add_ioctl_resp(&filler, session);
	if (ret)
		return ret;

	ret = virtio_media_send_command(
		vv, sgs, 2, 1, sizeof(struct virtio_media_resp_ioctl), NULL);
	if (ret < 0)
		return ret;

	return 0;
}

/**
 * Sends an ioctl that expects a response of exactly the same size as the
 * input (i.e. an ioctl specified with _IOWR) to the host.
 *
 * This corresponds to what most V4L2 ioctls do. For instance VIDIOC_ENUM_FMT
 * takes a partially-initialized struct v4l2_fmtdesc and returns its filled
 * version.
 *
 * Ioctls specified with _IOR can also use this, since the host will simply
 * ignore the extra input data provided.
 *
 * Returns 0 in case of success, or a negative error code.
 */
static int virtio_media_send_wr_ioctl(struct v4l2_fh *fh, u32 ioctl,
				      void *ioctl_data, size_t ioctl_data_len,
				      size_t min_resp_payload)
{
	struct video_device *video_dev = fh->vdev;
	struct virtio_media *vv = to_virtio_media(video_dev);
	struct virtio_media_session *session = fh_to_session(fh);
	struct scatterlist *sgs[4];
	struct scatterlist_filler filler = {
		.descs = session->command_sgs.sgl,
		.num_descs = DESC_CHAIN_MAX_LEN,
		.cur_desc = 0,
		.shadow_buffer = session->shadow_buf,
		.shadow_buffer_size = VIRTIO_SHADOW_BUF_SIZE,
		.shadow_buffer_pos = 0,
		.sgs = sgs,
		.num_sgs = ARRAY_SIZE(sgs),
		.cur_sg = 0,
	};
	int ret;

	/* Command descriptor */
	ret = scatterlist_filler_add_ioctl_cmd(&filler, session, ioctl);
	if (ret)
		return ret;

	/* Command payload */
	ret = scatterlist_filler_add_data(&filler, ioctl_data, ioctl_data_len);
	if (ret) {
		v4l2_err(&vv->v4l2_dev,
			 "failed to prepare command descriptor chain\n");
		return ret;
	}

	/* Response descriptor */
	ret = scatterlist_filler_add_ioctl_resp(&filler, session);
	if (ret)
		return ret;

	/* Response payload, same as command */
	ret = scatterlist_filler_add_sg(&filler, filler.sgs[1]);
	if (ret)
		return ret;

	ret = virtio_media_send_command(vv, sgs, 2, 2,
					sizeof(struct virtio_media_resp_ioctl) +
						min_resp_payload,
					NULL);
	if (ret < 0)
		return ret;

	ret = scatterlist_filler_retrieve_data(session, filler.sgs[3],
					       ioctl_data, ioctl_data_len);
	if (ret) {
		v4l2_err(&vv->v4l2_dev,
			 "failed to retrieve response descriptor chain\n");
		return ret;
	}

	return 0;
}

/**
 * Send an ioctl carrying a v4l2_buffer (QUERYBUF, PREPARE_BUF, QBUF).
 *
 * @vbuf: the driver's state for that buffer, or NULL. When its planes are
 * driver-owned the buffer travels as USERPTR with the cookie as opaque
 * pointer and the precomputed SG list appended (VPU_DESIGN.md 5.3 items 3
 * and 4), and comes back as MMAP with the cookie in m.offset.
 */
static int virtio_media_send_buffer_ioctl(struct v4l2_fh *fh, u32 ioctl_code,
					  struct v4l2_buffer *b,
					  struct virtio_media_buffer *vbuf)
{
	struct video_device *video_dev = fh->vdev;
	struct virtio_media *vv = to_virtio_media(video_dev);
	struct virtio_media_session *session = fh_to_session(fh);
	const bool driver_owned = vbuf && vbuf->dbuf[0];
	struct v4l2_plane *planes_backup = NULL;
	u32 length_backup = 0;
	struct scatterlist *sgs[64];
	/* End of the device-readable buffer SGs, to reuse in device-writable section. */
	size_t num_cmd_sgs;
	size_t end_buf_sg;
	struct scatterlist_filler filler = {
		.descs = session->command_sgs.sgl,
		.num_descs = DESC_CHAIN_MAX_LEN,
		.cur_desc = 0,
		.shadow_buffer = session->shadow_buf,
		.shadow_buffer_size = VIRTIO_SHADOW_BUF_SIZE,
		.shadow_buffer_pos = 0,
		.sgs = sgs,
		.num_sgs = ARRAY_SIZE(sgs),
		.cur_sg = 0,
	};
	size_t resp_len;
	int ret;
	int i;

	if (b->type > VIRTIO_MEDIA_LAST_QUEUE)
		return -EINVAL;

	if (V4L2_TYPE_IS_MULTIPLANAR(b->type)) {
		planes_backup = b->m.planes;
		length_backup = b->length;
	}

	/* Command descriptor */
	ret = scatterlist_filler_add_ioctl_cmd(&filler, session, ioctl_code);
	if (ret)
		return ret;

	/* Driver-owned planes: USERPTR + cookie towards the host. */
	if (driver_owned)
		vmedia_dbuf_buffer_to_host(b, vbuf->dbuf);

	/* Command payload (struct v4l2_buffer) */
	ret = scatterlist_filler_add_buffer(&filler, b);
	if (ret < 0)
		goto out;

	end_buf_sg = filler.cur_sg;

	/*
	 * Payload of USERPTR buffers, if relevant. Driver-owned buffers carry
	 * their precomputed SG lists on QBUF/PREPARE_BUF only: QUERYBUF has
	 * no payload to hand over.
	 */
	if (driver_owned) {
		if (ioctl_code != VIDIOC_QUERYBUF)
			ret = scatterlist_filler_add_buffer_dbuf(&filler, b,
								 vbuf->dbuf);
	} else {
		ret = scatterlist_filler_add_buffer_userptr(&filler, b);
	}
	if (ret < 0)
		goto out;

	num_cmd_sgs = filler.cur_sg;

	/* Response descriptor */
	ret = scatterlist_filler_add_ioctl_resp(&filler, session);
	if (ret)
		goto out;

	/* Response payload (same as input, but no userptr mapping) */
	for (i = 1; i < end_buf_sg; i++) {
		ret = scatterlist_filler_add_sg(&filler, filler.sgs[i]);
		if (ret < 0)
			goto out;
	}

	ret = virtio_media_send_command(
		vv, filler.sgs, num_cmd_sgs, filler.cur_sg - num_cmd_sgs,
		sizeof(struct virtio_media_resp_ioctl) + sizeof(*b), &resp_len);

	if (V4L2_TYPE_IS_MULTIPLANAR(b->type)) {
		b->m.planes = planes_backup;
		if (b->length > length_backup) {
			ret = -ENOSPC;
			goto out;
		}
	}

	if (ret < 0)
		goto out;

	resp_len -= sizeof(struct virtio_media_resp_ioctl);

	/* Make sure that the reply's length covers our v4l2_buffer */
	if (resp_len < sizeof(*b)) {
		ret = -EINVAL;
		goto out;
	}

	ret = scatterlist_filler_retrieve_buffer(session, &sgs[num_cmd_sgs + 1],
						 b, length_backup);
	if (ret) {
		v4l2_err(&vv->v4l2_dev,
			 "failed to retrieve response descriptor chain\n");
		goto out;
	}

	/* TODO ideally we should not be doing this twice, but the scatterlist may screw us up here? */
	if (V4L2_TYPE_IS_MULTIPLANAR(b->type)) {
		b->m.planes = planes_backup;
		if (b->length > length_backup) {
			ret = -ENOSPC;
			goto out;
		}
	}

	ret = 0;

out:
	/*
	 * Back to what user-space knows: MMAP, cookie, length. b->m.planes is
	 * the kernel copy again (restored above, or never touched on an early
	 * exit) and holds length_backup entries.
	 */
	if (driver_owned)
		vmedia_dbuf_buffer_from_host(b, b->m.planes, length_backup,
					     vbuf->dbuf);
	return ret;
}

/**
 * struct virtio_media_ctrl_bounce - One control payload routed through a
 * driver-owned bounce buffer (defect D34, B7-controls §9).
 *
 * The payload of a compound control is an ordinary guest user-space page:
 * outside every access window a pool-mode host helper may touch, so handing
 * its pinned pages to the host EFAULTs in both directions. Instead the
 * payload travels through @bounce -- media_guest pool memory when there is a
 * pool, a plain kernel buffer otherwise -- copied from user space before the
 * ioctl and back after it.
 *
 * @bounce: the driver-owned memory the host sees.
 * @uptr: the user-space payload pointer as submitted, kept here because the
 *	control array is overwritten with the host's echo of it before the
 *	copy-out.
 */
struct virtio_media_ctrl_bounce {
	struct vmedia_bounce *bounce;
	u64 uptr;
};

/**
 * Allocate and fill a bounce buffer for every control of @ctrls that carries
 * a payload. Returns NULL when no control does, an ERR_PTR on failure, and
 * the bounce array (indexed per control, of @ctrls->count entries) otherwise.
 */
static struct virtio_media_ctrl_bounce *
virtio_media_bounce_in_ext_ctrls(struct virtio_media *vv,
				 const struct v4l2_ext_controls *ctrls)
{
	struct virtio_media_ctrl_bounce *bounces;
	bool any = false;
	int ret;
	u32 i;

	for (i = 0; i < ctrls->count; i++)
		any |= ctrls->controls[i].size > 0;
	if (!any)
		return NULL;

	bounces = kvcalloc(ctrls->count, sizeof(*bounces), GFP_KERNEL);
	if (!bounces)
		return ERR_PTR(-ENOMEM);

	for (i = 0; i < ctrls->count; i++) {
		const struct v4l2_ext_control *ctrl = &ctrls->controls[i];
		struct vmedia_bounce *bounce;

		if (ctrl->size == 0)
			continue;

		bounce = vmedia_bounce_alloc(vv, ctrl->size);
		if (IS_ERR(bounce)) {
			ret = PTR_ERR(bounce);
			goto err_free;
		}
		bounces[i].bounce = bounce;
		bounces[i].uptr = (u64)(uintptr_t)ctrl->ptr;

		/*
		 * Copied in whatever the direction: S/TRY need the data, G
		 * overwrites it, and an unreadable pointer is -EFAULT for
		 * all three, exactly as if the host had been handed the
		 * pages themselves.
		 */
		if (copy_from_user(bounce->vaddr, (void __user *)ctrl->ptr,
				   ctrl->size)) {
			ret = -EFAULT;
			goto err_free;
		}
	}

	return bounces;

err_free:
	for (i = 0; i < ctrls->count; i++)
		vmedia_bounce_free(bounces[i].bounce);
	kvfree(bounces);
	return ERR_PTR(ret);
}

/**
 * Copy every bounced payload back to user space (the host writes updated or
 * requested values into the bounce, whatever the ioctl direction) and free
 * the array. @copy_back is false when the host was never reached, in which
 * case user memory is left untouched.
 *
 * Returns 0 or -EFAULT.
 */
static int virtio_media_bounce_out_ext_ctrls(
	struct virtio_media_ctrl_bounce *bounces, u32 count, bool copy_back)
{
	int ret = 0;
	u32 i;

	if (!bounces)
		return 0;

	for (i = 0; i < count; i++) {
		struct vmedia_bounce *bounce = bounces[i].bounce;

		if (!bounce)
			continue;
		if (copy_back &&
		    copy_to_user((void __user *)(uintptr_t)bounces[i].uptr,
				 bounce->vaddr, bounce->len))
			ret = -EFAULT;
		vmedia_bounce_free(bounce);
	}
	kvfree(bounces);

	return ret;
}

/**
 * Queues an ioctl that sends a v4l2_ext_controls to the host and receives an updated version.
 *
 * v4l2_ext_controls has a pointer to an array of v4l2_ext_control, and also
 * potentially pointers to user-space memory that we need to map properly --
 * through bounce buffers, see above -- hence the dedicated function.
 */
static int virtio_media_send_ext_controls_ioctl(struct v4l2_fh *fh,
						u32 ioctl_code,
						struct v4l2_ext_controls *ctrls)
{
	struct video_device *video_dev = fh->vdev;
	struct virtio_media *vv = to_virtio_media(video_dev);
	struct virtio_media_session *session = fh_to_session(fh);
	size_t num_cmd_sgs;
	struct v4l2_ext_control *controls_backup = ctrls->controls;
	const u32 num_ctrls = ctrls->count;
	struct virtio_media_ctrl_bounce *bounces = NULL;
	struct vmedia_bounce **bounce_ptrs = NULL;
	bool sent = false;
	struct scatterlist *sgs[64];
	struct scatterlist_filler filler = {
		.descs = session->command_sgs.sgl,
		.num_descs = DESC_CHAIN_MAX_LEN,
		.cur_desc = 0,
		.shadow_buffer = session->shadow_buf,
		.shadow_buffer_size = VIRTIO_SHADOW_BUF_SIZE,
		.shadow_buffer_pos = 0,
		.sgs = sgs,
		.num_sgs = ARRAY_SIZE(sgs),
		.cur_sg = 0,
	};
	size_t resp_len = 0;
	int bounce_ret;
	int ret;
	u32 i;

	/* Control payloads travel through driver-owned bounces (D34). */
	if (num_ctrls > 0 && ctrls->controls) {
		bounces = virtio_media_bounce_in_ext_ctrls(vv, ctrls);
		if (IS_ERR(bounces))
			return PTR_ERR(bounces);
		if (bounces) {
			bounce_ptrs = kvcalloc(num_ctrls,
					       sizeof(*bounce_ptrs),
					       GFP_KERNEL);
			if (!bounce_ptrs) {
				ret = -ENOMEM;
				goto out;
			}
			for (i = 0; i < num_ctrls; i++)
				bounce_ptrs[i] = bounces[i].bounce;
		}
	}

	/* Command descriptor */
	ret = scatterlist_filler_add_ioctl_cmd(&filler, session, ioctl_code);
	if (ret)
		goto out;

	/* v4l2_controls and the bounced payloads they point to */
	ret = scatterlist_filler_add_ext_ctrls(&filler, ctrls, true,
					       bounce_ptrs);
	if (ret)
		goto out;

	num_cmd_sgs = filler.cur_sg;

	/* Response descriptor */
	ret = scatterlist_filler_add_ioctl_resp(&filler, session);
	if (ret)
		goto out;

	/*
	 * Response payload (same as input but without payloads: the host
	 * writes those into the bounces directly)
	 */
	ret = scatterlist_filler_add_ext_ctrls(&filler, ctrls, false, NULL);
	if (ret)
		goto out;

	ret = virtio_media_send_command(
		vv, filler.sgs, num_cmd_sgs, filler.cur_sg - num_cmd_sgs,
		sizeof(struct virtio_media_resp_ioctl) + sizeof(*ctrls),
		&resp_len);
	sent = true;

	/* Just in case the host touched these. */
	ctrls->controls = controls_backup;
	if (ctrls->count != num_ctrls) {
		v4l2_err(
			&vv->v4l2_dev,
			"device returned a number of extended controls different than submitted\n");
	}
	if (ctrls->count > num_ctrls) {
		ret = -ENOSPC;
		goto out;
	}

	/* Event if we have received an error, we may need to read our payload back */
	if (ret < 0 && resp_len >= sizeof(struct virtio_media_resp_ioctl) +
					   sizeof(*ctrls)) {
		/* Deliberately ignore the error here as we want to return the previous one */
		scatterlist_filler_retrieve_ext_ctrls(
			session, &sgs[num_cmd_sgs + 1],
			filler.cur_sg - (num_cmd_sgs + 1), ctrls);
		goto out;
	}

	if (ret < 0)
		goto out;

	resp_len -= sizeof(struct virtio_media_resp_ioctl);

	/* Make sure that the reply's length covers our v4l2_ext_controls */
	if (resp_len < sizeof(*ctrls)) {
		ret = -EINVAL;
		goto out;
	}

	ret = scatterlist_filler_retrieve_ext_ctrls(
		session, &sgs[num_cmd_sgs + 1],
		filler.cur_sg - (num_cmd_sgs + 1), ctrls);

out:
	/*
	 * Once the host has answered, the bounces carry the payloads user
	 * space must see -- on success and on error alike, since an EXT_CTRLS
	 * error reply still updates the input (error_idx, adjusted values).
	 */
	bounce_ret = virtio_media_bounce_out_ext_ctrls(bounces, num_ctrls,
						       sent);
	kvfree(bounce_ptrs);
	ctrls->controls = controls_backup;
	if (bounce_ret && !ret)
		ret = bounce_ret;
	return ret;
}

/**
 * Helper function to clear the list of buffers waiting to be dequeued on a
 * queue that has just been streamed off.
 */
static void virtio_media_clear_queue(struct virtio_media *vv,
				     struct virtio_media_session *session,
				     struct virtio_media_queue_state *queue)
{
	struct list_head *p, *n;
	int i;

	mutex_lock(&session->dqbufs_lock);

	list_for_each_safe(p, n, &queue->pending_dqbufs) {
		struct virtio_media_buffer *dqbuf =
			list_entry(p, struct virtio_media_buffer, list);

		list_del(&dqbuf->list);
	}

	/*
	 * The flags walk and the counters stay under dqbufs_lock too: the
	 * event work reads buffer flags and decrements @queued_bufs under
	 * that lock (D46), and clearing the flags here is exactly what makes
	 * it drop an event still in flight for a streamed-off buffer.
	 */

	/* All buffers are now dequeued. */
	for (i = 0; i < queue->allocated_bufs; i++) {
		queue->buffers[i].buffer.flags = 0;
	}

	queue->queued_bufs = 0;
	queue->streaming = false;
	queue->is_capture_last = false;

	mutex_unlock(&session->dqbufs_lock);
}

/*
 * Macros suitable for defining ioctls with a constant size payload.
 *
 * Since Linux 6.18 the V4L2 core calls every driver op as
 * `ops->vidioc_xxx(file, NULL, arg)`: all of the call sites in
 * drivers/media/v4l2-core/v4l2-ioctl.c pass NULL for the `priv` argument,
 * where 6.17 and earlier passed the file's `v4l2_fh`. A driver must now take
 * its file handle from the file itself with `file_to_v4l2_fh(file)`, which is
 * literally `file->private_data` (include/media/v4l2-fh.h). That is exactly
 * what the core used to hand over, so deriving it here is correct on every
 * kernel version, old and new. Every op below therefore ignores the `priv`
 * argument -- it is named `priv_unused` so that a stray use fails to
 * compile -- and derives `fh` from `file`.
 */

#define SIMPLE_WR_IOCTL(name, ioctl, type)                            \
	static int virtio_media_##name(struct file *file,             \
				       void *priv_unused,             \
				       type *payload)                 \
	{                                                             \
		struct v4l2_fh *fh = file->private_data;              \
                                                                      \
		if (!fh)                                              \
			return -ENODEV;                               \
		return virtio_media_send_wr_ioctl(fh, ioctl, payload, \
						  sizeof(*payload),   \
						  sizeof(*payload));  \
	}
#define SIMPLE_R_IOCTL(name, ioctl, type)                            \
	static int virtio_media_##name(struct file *file,            \
				       void *priv_unused,            \
				       type *payload)                \
	{                                                            \
		struct v4l2_fh *fh = file->private_data;             \
                                                                     \
		if (!fh)                                             \
			return -ENODEV;                              \
		return virtio_media_send_r_ioctl(fh, ioctl, payload, \
						 sizeof(*payload));  \
	}
#define SIMPLE_W_IOCTL(name, ioctl, type)                            \
	static int virtio_media_##name(struct file *file,            \
				       void *priv_unused,            \
				       type *payload)                \
	{                                                            \
		struct v4l2_fh *fh = file->private_data;             \
                                                                     \
		if (!fh)                                             \
			return -ENODEV;                              \
		return virtio_media_send_w_ioctl(fh, ioctl, payload, \
						 sizeof(*payload));  \
	}

/*
 * V4L2 ioctl handlers.
 *
 * Most of these functions just forward the ioctl to the host, for these we can
 * use one of the SIMPLE_*_IOCTL macros. Exceptions that have their own
 * standalone function follow.
 */

SIMPLE_WR_IOCTL(enum_fmt, VIDIOC_ENUM_FMT, struct v4l2_fmtdesc)
SIMPLE_WR_IOCTL(g_fmt, VIDIOC_G_FMT, struct v4l2_format)
SIMPLE_WR_IOCTL(s_fmt, VIDIOC_S_FMT, struct v4l2_format)
SIMPLE_WR_IOCTL(try_fmt, VIDIOC_TRY_FMT, struct v4l2_format)
SIMPLE_WR_IOCTL(enum_framesizes, VIDIOC_ENUM_FRAMESIZES,
		struct v4l2_frmsizeenum)
SIMPLE_WR_IOCTL(enum_frameintervals, VIDIOC_ENUM_FRAMEINTERVALS,
		struct v4l2_frmivalenum)
#if LINUX_VERSION_CODE < KERNEL_VERSION(6,15,0)
SIMPLE_WR_IOCTL(queryctrl, VIDIOC_QUERYCTRL, struct v4l2_queryctrl)
SIMPLE_WR_IOCTL(g_ctrl, VIDIOC_G_CTRL, struct v4l2_control)
SIMPLE_WR_IOCTL(s_ctrl, VIDIOC_S_CTRL, struct v4l2_control)
#endif
SIMPLE_WR_IOCTL(query_ext_ctrl, VIDIOC_QUERY_EXT_CTRL,
		struct v4l2_query_ext_ctrl)
SIMPLE_WR_IOCTL(s_dv_timings, VIDIOC_S_DV_TIMINGS, struct v4l2_dv_timings)
SIMPLE_WR_IOCTL(g_dv_timings, VIDIOC_G_DV_TIMINGS, struct v4l2_dv_timings)
SIMPLE_R_IOCTL(query_dv_timings, VIDIOC_QUERY_DV_TIMINGS,
	       struct v4l2_dv_timings)
SIMPLE_WR_IOCTL(enum_dv_timings, VIDIOC_ENUM_DV_TIMINGS,
		struct v4l2_enum_dv_timings)
SIMPLE_WR_IOCTL(dv_timings_cap, VIDIOC_DV_TIMINGS_CAP,
		struct v4l2_dv_timings_cap)
SIMPLE_WR_IOCTL(enuminput, VIDIOC_ENUMINPUT, struct v4l2_input)
SIMPLE_WR_IOCTL(querymenu, VIDIOC_QUERYMENU, struct v4l2_querymenu)
SIMPLE_WR_IOCTL(enumoutput, VIDIOC_ENUMOUTPUT, struct v4l2_output)
SIMPLE_WR_IOCTL(enumaudio, VIDIOC_ENUMAUDIO, struct v4l2_audio)
SIMPLE_R_IOCTL(g_audio, VIDIOC_G_AUDIO, struct v4l2_audio)
SIMPLE_W_IOCTL(s_audio, VIDIOC_S_AUDIO, const struct v4l2_audio)
SIMPLE_WR_IOCTL(enumaudout, VIDIOC_ENUMAUDOUT, struct v4l2_audioout)
SIMPLE_R_IOCTL(g_audout, VIDIOC_G_AUDOUT, struct v4l2_audioout)
SIMPLE_W_IOCTL(s_audout, VIDIOC_S_AUDOUT, const struct v4l2_audioout)
SIMPLE_WR_IOCTL(g_modulator, VIDIOC_G_MODULATOR, struct v4l2_modulator)
SIMPLE_W_IOCTL(s_modulator, VIDIOC_S_MODULATOR, const struct v4l2_modulator)
SIMPLE_WR_IOCTL(g_selection, VIDIOC_G_SELECTION, struct v4l2_selection)
SIMPLE_WR_IOCTL(s_selection, VIDIOC_S_SELECTION, struct v4l2_selection)
SIMPLE_R_IOCTL(g_enc_index, VIDIOC_G_ENC_INDEX, struct v4l2_enc_idx)
SIMPLE_WR_IOCTL(try_encoder_cmd, VIDIOC_TRY_ENCODER_CMD,
		struct v4l2_encoder_cmd)
SIMPLE_WR_IOCTL(try_decoder_cmd, VIDIOC_TRY_DECODER_CMD,
		struct v4l2_decoder_cmd)
SIMPLE_WR_IOCTL(g_parm, VIDIOC_G_PARM, struct v4l2_streamparm)
SIMPLE_WR_IOCTL(s_parm, VIDIOC_S_PARM, struct v4l2_streamparm)
SIMPLE_R_IOCTL(g_std, VIDIOC_G_STD, v4l2_std_id)
SIMPLE_R_IOCTL(querystd, VIDIOC_QUERYSTD, v4l2_std_id)
SIMPLE_WR_IOCTL(enumstd, VIDIOC_ENUMSTD, struct v4l2_standard)
SIMPLE_WR_IOCTL(g_tuner, VIDIOC_G_TUNER, struct v4l2_tuner)
SIMPLE_W_IOCTL(s_tuner, VIDIOC_S_TUNER, const struct v4l2_tuner)
SIMPLE_WR_IOCTL(g_frequency, VIDIOC_G_FREQUENCY, struct v4l2_frequency)
SIMPLE_W_IOCTL(s_frequency, VIDIOC_S_FREQUENCY, const struct v4l2_frequency)
SIMPLE_WR_IOCTL(enum_freq_bands, VIDIOC_ENUM_FREQ_BANDS,
		struct v4l2_frequency_band)
SIMPLE_WR_IOCTL(g_sliced_vbi_cap, VIDIOC_G_SLICED_VBI_CAP,
		struct v4l2_sliced_vbi_cap)
SIMPLE_W_IOCTL(s_hw_freq_seek, VIDIOC_S_HW_FREQ_SEEK,
	       const struct v4l2_hw_freq_seek)

/*
 * QUERYCAP is handled by reading the configuration area.
 *
 */

static int virtio_media_querycap(struct file *file, void *priv_unused,
				 struct v4l2_capability *cap)
{
	struct video_device *video_dev = video_devdata(file);
	struct virtio_media *vv = to_virtio_media(video_dev);

	/* TODO add proper number? */
	strncpy(cap->bus_info, "platform:virtio-media0", sizeof(cap->bus_info));

	if (!driver_name) {
		strncpy(cap->driver, VIRTIO_MEDIA_DEFAULT_DRIVER_NAME,
			sizeof(cap->driver));
	} else {
		strncpy(cap->driver, driver_name, sizeof(cap->driver));
	}
	virtio_cread_bytes(vv->virtio_dev, 8, cap->card, sizeof(cap->card));

	cap->capabilities = video_dev->device_caps | V4L2_CAP_DEVICE_CAPS;
	cap->device_caps = video_dev->device_caps;

	return 0;
}

/*
 * Extended control ioctls are handled mostly identically.
 */

static int virtio_media_g_ext_ctrls(struct file *file, void *priv_unused,
				    struct v4l2_ext_controls *ctrls)
{
	struct v4l2_fh *fh = file->private_data;

	if (!fh)
		return -ENODEV;

	return virtio_media_send_ext_controls_ioctl(fh, VIDIOC_G_EXT_CTRLS,
						    ctrls);
}

static int virtio_media_s_ext_ctrls(struct file *file, void *priv_unused,
				    struct v4l2_ext_controls *ctrls)
{
	struct v4l2_fh *fh = file->private_data;

	if (!fh)
		return -ENODEV;

	return virtio_media_send_ext_controls_ioctl(fh, VIDIOC_S_EXT_CTRLS,
						    ctrls);
}

static int virtio_media_try_ext_ctrls(struct file *file, void *priv_unused,
				      struct v4l2_ext_controls *ctrls)
{
	struct v4l2_fh *fh = file->private_data;

	if (!fh)
		return -ENODEV;

	return virtio_media_send_ext_controls_ioctl(fh, VIDIOC_TRY_EXT_CTRLS,
						    ctrls);
}

/*
 * Subscribe/unsubscribe from an event.
 */

static int
virtio_media_subscribe_event(struct v4l2_fh *fh,
			     const struct v4l2_event_subscription *sub)
{
	struct video_device *video_dev;
	struct virtio_media *vv;
	int ret;

	/*
	 * v4l_subscribe_event() resolves the handle with file_to_v4l2_fh()
	 * and hands it to us directly, so this is only a guard.
	 */
	if (!fh)
		return -ENODEV;

	video_dev = fh->vdev;
	vv = to_virtio_media(video_dev);

	/* First subscribe to the event in the guest. */
	switch (sub->type) {
	case V4L2_EVENT_SOURCE_CHANGE:
		ret = v4l2_src_change_event_subscribe(fh, sub);
		break;
	default:
		ret = v4l2_event_subscribe(fh, sub, 1, NULL);
		break;
	}
	if (ret)
		return ret;

	/* Then ask the host to signal us these events. */
	ret = virtio_media_send_w_ioctl(fh, VIDIOC_SUBSCRIBE_EVENT, sub,
					sizeof(*sub));
	if (ret < 0) {
		v4l2_event_unsubscribe(fh, sub);
		return ret;
	}

	/*
	 * Subscribing to an event may result in that event being signaled
	 * immediately. Process all pending events to make sure we don't miss it.
	 */
	if (sub->flags & V4L2_EVENT_SUB_FL_SEND_INITIAL) {
		virtio_media_process_events(vv);
	}

	return 0;
}

static int
virtio_media_unsubscribe_event(struct v4l2_fh *fh,
			       const struct v4l2_event_subscription *sub)
{
	int ret;

	if (!fh)
		return -ENODEV;

	ret = virtio_media_send_w_ioctl(fh, VIDIOC_UNSUBSCRIBE_EVENT, sub,
					sizeof(*sub));
	if (ret < 0)
		return ret;

	ret = v4l2_event_unsubscribe(fh, sub);
	if (ret)
		return ret;

	return 0;
}

/*
 * Streamon/off affect the local queue state.
 */

static int virtio_media_streamon(struct file *file, void *priv_unused,
				 enum v4l2_buf_type i)
{
	struct v4l2_fh *fh = file->private_data;
	struct virtio_media_session *session;
	int ret;

	if (!fh)
		return -ENODEV;
	session = fh_to_session(fh);

	if (i > VIRTIO_MEDIA_LAST_QUEUE)
		return -EINVAL;

	ret = virtio_media_send_w_ioctl(fh, VIDIOC_STREAMON, &i, sizeof(i));
	if (ret < 0)
		return ret;

	session->queues[i].streaming = true;
	/*
	 * STREAMON ends the EPIPE-after-LAST drain state alongside
	 * STREAMOFF and the *_CMD_START commands (dev-decoder.rst "Drain":
	 * "until the client issues any of the following operations"), D27b.
	 */
	session->queues[i].is_capture_last = false;

	return 0;
}

static int virtio_media_streamoff(struct file *file, void *priv_unused,
				  enum v4l2_buf_type i)
{
	struct video_device *video_dev = video_devdata(file);
	struct virtio_media *vv = to_virtio_media(video_dev);
	struct v4l2_fh *fh = file->private_data;
	struct virtio_media_session *session;
	int ret;

	if (!fh)
		return -ENODEV;
	session = fh_to_session(fh);

	if (i > VIRTIO_MEDIA_LAST_QUEUE) {
		return -EINVAL;
	}

	ret = virtio_media_send_w_ioctl(fh, VIDIOC_STREAMOFF, &i, sizeof(i));
	if (ret < 0)
		return ret;

	virtio_media_clear_queue(vv, session, &session->queues[i]);

	return 0;
}

/*
 * Buffer creation/queuing functions deal with the local driver state.
 */

/**
 * Whether host-owned MMAP buffers can be mapped into the guest at all: only
 * with a media_host pool or a virtio shm region (VPU_DESIGN.md 2.4). Without
 * either, fail the request that would allocate them rather than let mmap map
 * physical page 0 later.
 */
static bool virtio_media_host_mmap_available(struct virtio_media *vv)
{
	if (vv->mmap_region.len > 0)
		return true;

	pr_warn_once("virtio-media: host-owned MMAP buffers requested but there is neither a media_host pool nor a virtio shm region, failing with -ENOMEM\n");
	return false;
}

/**
 * Whether MMAP buffers on a queue of @type are allocated by the driver
 * (VPU_DESIGN.md 2.1): the driver_owned_queues module parameter decides,
 * evaluated against V4L2_TYPE_IS_OUTPUT(). Types whose format does not tell
 * the buffer size (VBI, overlay) stay host-owned.
 */
static bool virtio_media_type_is_driver_owned(u32 type)
{
	const char *mode = driver_owned_queues;
	bool wanted;

	/*
	 * driver_owned_queues_set() rejects anything else, so an unexpected
	 * value can only be a NULL default; treat it as "output".
	 */
	if (mode && sysfs_streq(mode, "all"))
		wanted = true;
	else if (mode && sysfs_streq(mode, "none"))
		wanted = false;
	else
		wanted = V4L2_TYPE_IS_OUTPUT(type);

	if (wanted && !vmedia_dbuf_type_supported(type)) {
		pr_info_once("virtio-media: queue type %u carries no buffer size in its format (VBI/overlay), its MMAP buffers stay host-owned\n",
			     type);
		return false;
	}

	return wanted;
}

/**
 * Ask the host for the queue's current format and derive the plane sizes
 * driver-owned buffers need. REQBUFS does not carry a format, so this is the
 * same source a vb2 driver's queue_setup() uses.
 */
static int virtio_media_queue_plane_sizes(struct v4l2_fh *fh, u32 type,
					  size_t sizes[VIDEO_MAX_PLANES],
					  u32 *num_planes)
{
	struct v4l2_format f = { .type = type };
	int ret;

	ret = virtio_media_send_wr_ioctl(fh, VIDIOC_G_FMT, &f, sizeof(f),
					 sizeof(f));
	if (ret)
		return ret;

	return vmedia_dbuf_plane_sizes(&f, sizes, num_planes);
}

/**
 * The reply of a REQBUFS/CREATE_BUFS that went out as USERPTR on behalf of
 * driver-owned MMAP buffers: user-space gets @user_memory back, the value it
 * asked with, and the capabilities say what this driver serves on the queue.
 *
 * @user_memory is not always MMAP: a REQBUFS(0) that tears such a queue down
 * has to name the type the host holds (USERPTR), whatever user-space asked
 * with. The capability rewrite only makes sense for the MMAP answer.
 *
 * ORPHANED_BUFS is true of driver-owned buffers: vmedia_dbuf keeps its memory
 * for as long as a VMA maps it, so REQBUFS(0) with buffers still mapped
 * succeeds instead of returning -EBUSY (VPU_DESIGN.md 2.5).
 */
static void virtio_media_fixup_driver_owned_reply(u32 *memory,
						  u32 *capabilities,
						  u32 user_memory)
{
	*memory = user_memory;

	if (user_memory != V4L2_MEMORY_MMAP)
		return;

	*capabilities |= V4L2_BUF_CAP_SUPPORTS_MMAP |
			 V4L2_BUF_CAP_SUPPORTS_ORPHANED_BUFS;
	*capabilities &= ~V4L2_BUF_CAP_SUPPORTS_USERPTR;
}

/**
 * A buffer ioctl must name the memory type its queue was set up with. The
 * driver-owned substitution means the host sees USERPTR on a queue user-space
 * knows as MMAP, so the check cannot be made against what goes on the wire or
 * against the presence of a dbuf: only the queue's recorded type says what
 * user-space agreed to (review bug 4).
 */
static int virtio_media_check_buffer_memory(struct virtio_media_queue_state *queue,
					    const struct v4l2_buffer *b)
{
	if (b->memory != queue->memory)
		return -EINVAL;

	return 0;
}

/**
 * Drop everything the queue holds for its current buffers: the driver-owned
 * backing (kept alive by any VMA still mapping it) and the buffer state array.
 * Leaves the queue in the shape it has before its first REQBUFS.
 *
 * virtio_media_clear_queue() must run first when there may be queued or
 * pending buffers: it walks the array this frees.
 *
 * The caller must hold session->dqbufs_lock: the event work dereferences
 * @buffers under that lock (D46), and freeing the array under anything less
 * (the ioctl path's vv->vlock, which the work never takes) was the B8
 * use-after-free. @pending_dqbufs is re-initialized because its nodes live
 * inside the array being freed, and @queued_bufs counted buffers that no
 * longer exist.
 */
static void
virtio_media_release_queue_bufs(struct virtio_media_queue_state *queue)
{
	vmedia_queue_put_dbufs(queue);
	vfree(queue->buffers);
	queue->buffers = NULL;
	queue->allocated_bufs = 0;
	queue->driver_owned = false;
	queue->memory = 0;
	queue->queued_bufs = 0;
	INIT_LIST_HEAD(&queue->pending_dqbufs);
}

/* Bound on what the host may claim it allocated; vb2 itself stops at 1024. */
#define VIRTIO_MEDIA_MAX_BUFFERS 1024

static int virtio_media_reqbufs(struct file *file, void *priv_unused,
				struct v4l2_requestbuffers *b)
{
	struct video_device *video_dev = video_devdata(file);
	struct virtio_media *vv = to_virtio_media(video_dev);
	struct v4l2_fh *fh = file->private_data;
	struct virtio_media_session *session;
	struct virtio_media_queue_state *queue;
	size_t sizes[VIDEO_MAX_PLANES];
	u32 num_planes = 0;
	const u32 user_memory = b->memory;
	bool driver_owned = false;
	bool as_userptr;
	int ret;

	if (!fh)
		return -ENODEV;
	session = fh_to_session(fh);

	if (b->type > VIRTIO_MEDIA_LAST_QUEUE)
		return -EINVAL;

	queue = &session->queues[b->type];

	/*
	 * MMAP on a queue the guest fills: the driver allocates the buffers
	 * and the host sees a USERPTR queue (VPU_DESIGN.md 5.3 item 1). The
	 * sizes come from the current format, fetched before anything on the
	 * host changes so a failure here leaves both sides untouched.
	 */
	if (user_memory == V4L2_MEMORY_MMAP && b->count > 0) {
		driver_owned = virtio_media_type_is_driver_owned(b->type);
		if (driver_owned) {
			ret = virtio_media_queue_plane_sizes(fh, b->type, sizes,
							     &num_planes);
			if (ret)
				return ret;
		} else if (!virtio_media_host_mmap_available(vv)) {
			return -ENOMEM;
		}
	}

	/*
	 * REQBUFS(0) frees whatever the queue holds, and the type it must name
	 * is the one the host has: USERPTR for a queue whose MMAP buffers this
	 * driver owns, whatever user-space passed. A backend that checks
	 * q->memory (v4l2-proxy does) returns -EINVAL otherwise (review
	 * contract 3).
	 */
	as_userptr = b->count > 0 ? driver_owned : queue->driver_owned;

	if (as_userptr)
		b->memory = V4L2_MEMORY_USERPTR;
	ret = virtio_media_send_wr_ioctl(fh, VIDIOC_REQBUFS, b, sizeof(*b),
					 sizeof(*b));
	if (as_userptr)
		virtio_media_fixup_driver_owned_reply(&b->memory,
						      &b->capabilities,
						      user_memory);
	if (ret)
		return ret;

	if (b->count > VIRTIO_MEDIA_MAX_BUFFERS) {
		v4l2_err(&vv->v4l2_dev,
			 "host allocated %u buffers, more than the %u supported\n",
			 b->count, VIRTIO_MEDIA_MAX_BUFFERS);
		ret = -EINVAL;
		goto err_release_host;
	}

	/* REQBUFS(0) is an implicit STREAMOFF. */
	if (b->count == 0) {
		virtio_media_clear_queue(vv, session, queue);
	}

	/*
	 * The host has answered, so it holds no mapping of the previous
	 * buffers any more and their memory can go back (VPU_DESIGN.md 2.5).
	 *
	 * The teardown and the publication of the new array happen under
	 * dqbufs_lock: the event work dereferences @buffers/@allocated_bufs
	 * under that lock and nothing else (D46). Lock order:
	 * vlock (held by the ioctl dispatcher) -> dqbufs_lock ->
	 * guest_pool_lock (taken inside vmedia_dbuf_put for pool-backed
	 * dbufs); the event work orders events_process_lock -> dqbufs_lock
	 * and never takes vlock, so the two chains cannot cross.
	 */
	mutex_lock(&session->dqbufs_lock);
	virtio_media_release_queue_bufs(queue);

	if (b->count > 0) {
		queue->buffers =
			vzalloc(sizeof(struct virtio_media_buffer) * b->count);
		if (!queue->buffers) {
			mutex_unlock(&session->dqbufs_lock);
			ret = -ENOMEM;
			goto err_release_host;
		}
		queue->allocated_bufs = b->count;
		queue->memory = user_memory;
	}
	mutex_unlock(&session->dqbufs_lock);

	/*
	 * The dbuf backing can be filled in outside the lock: none of the
	 * new buffers carries V4L2_BUF_FLAG_QUEUED yet, so the event work
	 * will not touch them (it drops events for unqueued buffers), and
	 * QBUF cannot run before this ioctl returns -- both hold vlock.
	 * This keeps dqbufs_lock off the allocator's guest_pool_lock path.
	 */
	if (b->count > 0 && driver_owned) {
		ret = vmedia_queue_alloc_dbufs(vv, session, queue, 0, b->count,
					       sizes, num_planes);
		if (ret)
			goto err_release_host;
		queue->driver_owned = true;
	}

	/*
	 * If a multiplanar queue is successfully used here, this means
	 * we are using the multiplanar interface.
	 */
	if (V4L2_TYPE_IS_MULTIPLANAR(b->type)) {
		session->uses_mplane = true;
	}

	/* TODO remove once we support DMABUFs */
	b->capabilities &= ~V4L2_BUF_CAP_SUPPORTS_DMABUF;

	return 0;

err_release_host:
	/*
	 * The host allocated buffers we cannot track: put it back to zero so
	 * both sides agree, and report zero buffers. Our own side has to go
	 * away too -- whichever of the failures above we came from, the buffer
	 * array and the dbufs it may still hold describe buffers the host no
	 * longer has (review bug 3).
	 */
	{
		struct v4l2_requestbuffers zero = {
			.count = 0,
			.type = b->type,
			.memory = as_userptr ? V4L2_MEMORY_USERPTR :
					       user_memory,
		};

		virtio_media_send_wr_ioctl(fh, VIDIOC_REQBUFS, &zero,
					   sizeof(zero), sizeof(zero));
	}
	virtio_media_clear_queue(vv, session, queue);
	mutex_lock(&session->dqbufs_lock);
	virtio_media_release_queue_bufs(queue);
	mutex_unlock(&session->dqbufs_lock);
	b->count = 0;
	return ret;
}

static int virtio_media_querybuf(struct file *file, void *priv_unused,
				 struct v4l2_buffer *b)
{
	struct v4l2_fh *fh = file->private_data;
	struct virtio_media_session *session;
	struct virtio_media_queue_state *queue;
	struct virtio_media_buffer *buffer;
	int ret;

	if (!fh)
		return -ENODEV;
	session = fh_to_session(fh);

	if (b->type > VIRTIO_MEDIA_LAST_QUEUE) {
		return -EINVAL;
	}
	queue = &session->queues[b->type];
	if (b->index >= queue->allocated_bufs) {
		return -EINVAL;
	}
	buffer = &queue->buffers[b->index];

	/*
	 * The host answers with its view (flags, timestamps, ...); for a
	 * driver-owned buffer the memory type, offset and length are then
	 * replaced with ours on the way back (VPU_DESIGN.md 5.3 item 2).
	 */
	ret = virtio_media_send_buffer_ioctl(fh, VIDIOC_QUERYBUF, b, buffer);
	if (ret)
		return ret;

	/* Set the DONE flag if the buffer is waiting in our own dequeue queue. */
	b->flags |= (buffer->buffer.flags & V4L2_BUF_FLAG_DONE);

	return 0;
}

static int virtio_media_create_bufs(struct file *file, void *priv_unused,
				    struct v4l2_create_buffers *b)
{
	struct video_device *video_dev = video_devdata(file);
	struct virtio_media *vv = to_virtio_media(video_dev);
	struct v4l2_fh *fh = file->private_data;
	struct virtio_media_session *session;
	struct virtio_media_queue_state *queue;
	struct virtio_media_buffer *buffers;
	struct virtio_media_buffer *new_buffers;
	LIST_HEAD(old_pending);
	size_t sizes[VIDEO_MAX_PLANES];
	u32 num_planes = 0;
	u32 type = b->format.type;
	const u32 user_memory = b->memory;
	bool driver_owned = false;
	u32 last_buf;
	int ret;

	if (!fh)
		return -ENODEV;
	session = fh_to_session(fh);

	if (type > VIRTIO_MEDIA_LAST_QUEUE)
		return -EINVAL;

	queue = &session->queues[type];

	/*
	 * Same rule as REQBUFS, except that a queue that already has buffers
	 * keeps whatever ownership they have: the host sees one memory type
	 * per queue. count == 0 is only a format probe, but it still has to
	 * name the type the host holds on an existing queue (review
	 * contract 3).
	 */
	if (user_memory == V4L2_MEMORY_MMAP) {
		driver_owned = queue->allocated_bufs > 0 ?
				       queue->driver_owned :
				       virtio_media_type_is_driver_owned(type);
		if (!driver_owned && b->count > 0 &&
		    !virtio_media_host_mmap_available(vv))
			return -ENOMEM;
	}

	if (driver_owned)
		b->memory = V4L2_MEMORY_USERPTR;
	ret = virtio_media_send_wr_ioctl(fh, VIDIOC_CREATE_BUFS, b, sizeof(*b),
					 sizeof(*b));
	if (driver_owned)
		virtio_media_fixup_driver_owned_reply(&b->memory,
						      &b->capabilities,
						      user_memory);
	if (ret)
		return ret;

	/* If count is zero, we were just checking for format. */
	if (b->count == 0)
		return 0;

	/*
	 * The host's index and count are u32 and only the host's word for it:
	 * add them with an overflow check before anything sizes an allocation
	 * with the sum (review bug 3).
	 */
	if (check_add_overflow(b->index, b->count, &last_buf) ||
	    b->index != queue->allocated_bufs ||
	    last_buf > VIRTIO_MEDIA_MAX_BUFFERS) {
		v4l2_err(&vv->v4l2_dev,
			 "host created %u buffers at index %u, expected index %zu and at most %u buffers\n",
			 b->count, b->index, queue->allocated_bufs,
			 VIRTIO_MEDIA_MAX_BUFFERS);
		return -EINVAL;
	}

	new_buffers = vzalloc(sizeof(struct virtio_media_buffer) * last_buf);
	if (!new_buffers)
		return -ENOMEM;

	/*
	 * Swap the array under dqbufs_lock, like REQBUFS does: the event
	 * work dereferences @buffers under that lock and nothing else
	 * (D46). CREATE_BUFS is legal on a streaming queue, so buffers may
	 * sit on @pending_dqbufs right now -- their list nodes live inside
	 * the old array and must be re-threaded onto their copies in the
	 * new one, or the list would walk into freed memory (the memcpy
	 * copies stale prev/next pointers, it does not move list
	 * membership).
	 */
	mutex_lock(&session->dqbufs_lock);
	buffers = queue->buffers;
	if (buffers)
		memcpy(new_buffers, buffers,
		       sizeof(*buffers) * queue->allocated_bufs);
	list_replace_init(&queue->pending_dqbufs, &old_pending);
	while (!list_empty(&old_pending)) {
		struct virtio_media_buffer *old = list_first_entry(
			&old_pending, struct virtio_media_buffer, list);

		list_del(&old->list);
		list_add_tail(&new_buffers[old - buffers].list,
			      &queue->pending_dqbufs);
	}
	queue->buffers = new_buffers;
	queue->allocated_bufs = last_buf;
	queue->memory = user_memory;
	mutex_unlock(&session->dqbufs_lock);

	vfree(buffers);

	if (driver_owned) {
		/* The host may have adjusted the format; size from its reply. */
		ret = vmedia_dbuf_plane_sizes(&b->format, sizes, &num_planes);
		if (!ret)
			ret = vmedia_queue_alloc_dbufs(vv, session, queue,
						       b->index, b->count,
						       sizes, num_planes);
		if (ret) {
			/*
			 * The host keeps the buffers it created; there is no
			 * per-buffer undo short of REQBUFS(0). Leave them
			 * without backing: QBUF on one of them fails.
			 */
			v4l2_err(&vv->v4l2_dev,
				 "no backing for %u created buffers: %d\n",
				 b->count, ret);
			return ret;
		}
		queue->driver_owned = true;
	}

	return 0;
}

static int virtio_media_prepare_buf(struct file *file, void *priv_unused,
				    struct v4l2_buffer *b)
{
	struct v4l2_fh *fh = file->private_data;
	struct virtio_media_session *session;
	struct virtio_media_queue_state *queue;
	struct virtio_media_buffer *buffer;
	int i, ret;

	if (!fh)
		return -ENODEV;
	session = fh_to_session(fh);

	if (b->type > VIRTIO_MEDIA_LAST_QUEUE)
		return -EINVAL;
	queue = &session->queues[b->type];
	if (b->index >= queue->allocated_bufs)
		return -EINVAL;
	ret = virtio_media_check_buffer_memory(queue, b);
	if (ret)
		return ret;
	buffer = &queue->buffers[b->index];

	buffer->buffer.m = b->m;
	if (V4L2_TYPE_IS_MULTIPLANAR(b->type)) {
		if (b->length > VIDEO_MAX_PLANES)
			return -EINVAL;
		for (i = 0; i < b->length; i++)
			buffer->planes[i].m = b->m.planes[i].m;
	}

	ret = virtio_media_send_buffer_ioctl(fh, VIDIOC_PREPARE_BUF, b, buffer);
	if (ret)
		return ret;

	buffer->buffer.flags = V4L2_BUF_FLAG_PREPARED;

	return 0;
}

static int virtio_media_qbuf(struct file *file, void *priv_unused,
			     struct v4l2_buffer *b)
{
	struct v4l2_fh *fh = file->private_data;
	struct virtio_media_session *session;
	struct virtio_media_queue_state *queue;
	struct virtio_media_buffer *buffer;
	bool prepared;
	u32 old_flags;
	int i, ret;

	if (!fh)
		return -ENODEV;
	session = fh_to_session(fh);

	if (b->type > VIRTIO_MEDIA_LAST_QUEUE)
		return -EINVAL;
	queue = &session->queues[b->type];
	if (b->index >= queue->allocated_bufs)
		return -EINVAL;
	ret = virtio_media_check_buffer_memory(queue, b);
	if (ret)
		return ret;
	buffer = &queue->buffers[b->index];
	prepared = buffer->buffer.flags & V4L2_BUF_FLAG_PREPARED;

	/*
	 * Store the buffer and plane `m` information so we can retrieve it again
	 * when DQBUF occurs.
	 */
	if (!prepared) {
		buffer->buffer.m = b->m;
		if (V4L2_TYPE_IS_MULTIPLANAR(b->type)) {
			if (b->length > VIDEO_MAX_PLANES)
				return -EINVAL;
			for (i = 0; i < b->length; i++)
				buffer->planes[i].m = b->m.planes[i].m;
		}
	}
	/*
	 * Flag and count the buffer as queued under dqbufs_lock *before* the
	 * host sees the QBUF: the completion event can arrive before this
	 * ioctl's response, and the event work (which reads the flag and
	 * decrements the count under the same lock, D46) must then find the
	 * buffer already accounted for -- counting it afterwards left a
	 * window where the decrement was lost and @queued_bufs drifted.
	 */
	old_flags = buffer->buffer.flags;
	mutex_lock(&session->dqbufs_lock);
	buffer->buffer.flags = V4L2_BUF_FLAG_QUEUED;
	queue->queued_bufs += 1;
	mutex_unlock(&session->dqbufs_lock);

	ret = virtio_media_send_buffer_ioctl(fh, VIDIOC_QBUF, b, buffer);
	if (ret) {
		/* Rollback the previous flags as the buffer is not queued. */
		mutex_lock(&session->dqbufs_lock);
		buffer->buffer.flags = old_flags;
		if (queue->queued_bufs > 0)
			queue->queued_bufs -= 1;
		mutex_unlock(&session->dqbufs_lock);
		return ret;
	}

	return 0;
}

static int virtio_media_dqbuf(struct file *file, void *priv_unused,
			      struct v4l2_buffer *b)
{
	struct v4l2_fh *fh = file->private_data;
	struct virtio_media_session *session;
	struct video_device *video_dev = video_devdata(file);
	struct virtio_media *vv = to_virtio_media(video_dev);
	struct virtio_media_buffer *dqbuf;
	struct virtio_media_queue_state *queue;
	struct list_head *buffer_queue;
	struct v4l2_plane *planes_backup = NULL;
	const bool is_multiplanar = V4L2_TYPE_IS_MULTIPLANAR(b->type);
	int ret;

	if (!fh)
		return -ENODEV;
	session = fh_to_session(fh);

	if (b->type > VIRTIO_MEDIA_LAST_QUEUE)
		return -EINVAL;

	queue = &session->queues[b->type];

	/*
	 * If a buffer with the LAST flag has been returned, subsequent calls to DQBUF
	 * must return -EPIPE until the queue is cleared.
	 */
	if (queue->is_capture_last)
		return -EPIPE;

	buffer_queue = &queue->pending_dqbufs;

	if (session->nonblocking_dequeue) {
		if (list_empty(buffer_queue))
			return -EAGAIN;
	} else if (queue->allocated_bufs == 0) {
		return -EINVAL;
	} else if (!queue->streaming) {
		return -EINVAL;
	} else {
		mutex_unlock(&vv->vlock);
		ret = wait_event_interruptible(session->dqbufs_wait,
					       !list_empty(buffer_queue) ||
						       READ_ONCE(session->dead));
		mutex_lock(&vv->vlock);
		if (ret)
			return -EINTR;
		if (READ_ONCE(session->dead))
			return -ENODEV;
	}

	mutex_lock(&session->dqbufs_lock);
	dqbuf = list_first_entry(buffer_queue, struct virtio_media_buffer,
				 list);
	list_del(&dqbuf->list);
	/*
	 * Clear the DONE flag as the buffer is now being dequeued -- under
	 * dqbufs_lock, since the event work reads these flags under it to
	 * decide whether an event is deliverable (D46).
	 */
	dqbuf->buffer.flags &= ~V4L2_BUF_FLAG_DONE;
	mutex_unlock(&session->dqbufs_lock);

	if (is_multiplanar) {
		size_t nb_planes = min(b->length, (u32)VIDEO_MAX_PLANES);
		memcpy(b->m.planes, dqbuf->planes,
		       nb_planes * sizeof(struct v4l2_plane));
		planes_backup = b->m.planes;
	}

	memcpy(b, &dqbuf->buffer, sizeof(*b));

	if (is_multiplanar) {
		b->m.planes = planes_backup;
	}

	if (V4L2_TYPE_IS_CAPTURE(b->type) && b->flags & V4L2_BUF_FLAG_LAST) {
		queue->is_capture_last = true;
	}

	return 0;
}

/*
 * s/g_input/output work with an unsigned int - recast this to a u32 so the
 * size is unambiguous.
 */

static int virtio_media_g_input(struct file *file, void *priv_unused,
				unsigned int *i)
{
	struct v4l2_fh *fh = file->private_data;
	u32 input;
	int ret;

	if (!fh)
		return -ENODEV;

	ret = virtio_media_send_wr_ioctl(fh, VIDIOC_G_INPUT, &input,
					 sizeof(input), sizeof(input));
	if (ret)
		return ret;

	*i = input;

	return 0;
}

static int virtio_media_s_input(struct file *file, void *priv_unused,
				unsigned int i)
{
	struct v4l2_fh *fh = file->private_data;
	u32 input = i;

	if (!fh)
		return -ENODEV;

	return virtio_media_send_wr_ioctl(fh, VIDIOC_S_INPUT, &input,
					  sizeof(input), sizeof(input));
}

static int virtio_media_g_output(struct file *file, void *priv_unused,
				 unsigned int *o)
{
	struct v4l2_fh *fh = file->private_data;
	u32 output;
	int ret;

	if (!fh)
		return -ENODEV;

	ret = virtio_media_send_wr_ioctl(fh, VIDIOC_G_OUTPUT, &output,
					 sizeof(output), sizeof(output));
	if (ret)
		return ret;

	*o = output;

	return 0;
}

static int virtio_media_s_output(struct file *file, void *priv_unused,
				 unsigned int o)
{
	struct v4l2_fh *fh = file->private_data;
	u32 output = o;

	if (!fh)
		return -ENODEV;

	return virtio_media_send_wr_ioctl(fh, VIDIOC_S_OUTPUT, &output,
					  sizeof(output), sizeof(output));
}

/*
 * decoder_cmd can affect the state of the CAPTURE queue.
 */

static int virtio_media_decoder_cmd(struct file *file, void *priv_unused,
				    struct v4l2_decoder_cmd *cmd)
{
	struct v4l2_fh *fh = file->private_data;
	struct virtio_media_session *session;
	int ret;

	if (!fh)
		return -ENODEV;
	session = fh_to_session(fh);

	ret = virtio_media_send_wr_ioctl(fh, VIDIOC_DECODER_CMD, cmd,
					 sizeof(*cmd), sizeof(*cmd));
	if (ret)
		return ret;

	/* A START command makes the CAPTURE queue able to dequeue again. */
	if (cmd->cmd == V4L2_DEC_CMD_START) {
		session->queues[V4L2_BUF_TYPE_VIDEO_CAPTURE].is_capture_last =
			false;
		session->queues[V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE]
			.is_capture_last = false;
	}

	return 0;
}

/*
 * encoder_cmd affects the CAPTURE queue the same way (dev-encoder.rst
 * "Drain": ENC_CMD_START resumes a queue parked by a dequeued LAST buffer);
 * without this the driver kept answering -EPIPE after an encoder drain until
 * STREAMOFF, stalling clients that restart with ENC_CMD_START (D27b's
 * encoder-side twin, D40/D43 root).
 */
static int virtio_media_encoder_cmd(struct file *file, void *priv_unused,
				    struct v4l2_encoder_cmd *cmd)
{
	struct v4l2_fh *fh = file->private_data;
	struct virtio_media_session *session;
	int ret;

	if (!fh)
		return -ENODEV;
	session = fh_to_session(fh);

	ret = virtio_media_send_wr_ioctl(fh, VIDIOC_ENCODER_CMD, cmd,
					 sizeof(*cmd), sizeof(*cmd));
	if (ret)
		return ret;

	/* A START command makes the CAPTURE queue able to dequeue again. */
	if (cmd->cmd == V4L2_ENC_CMD_START) {
		session->queues[V4L2_BUF_TYPE_VIDEO_CAPTURE].is_capture_last =
			false;
		session->queues[V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE]
			.is_capture_last = false;
	}

	return 0;
}

/*
 * s_std doesn't work with a pointer, so we cannot use SIMPLE_W_IOCTL.
 */

static int virtio_media_s_std(struct file *file, void *priv_unused,
			      v4l2_std_id s)
{
	struct v4l2_fh *fh = file->private_data;
	int ret;

	if (!fh)
		return -ENODEV;

	ret = virtio_media_send_w_ioctl(fh, VIDIOC_S_STD, &s, sizeof(s));
	if (ret)
		return ret;

	return 0;
}

const struct v4l2_ioctl_ops virtio_media_ioctl_ops = {
	/* VIDIOC_QUERYCAP handler */
	.vidioc_querycap = virtio_media_querycap,

	/* VIDIOC_ENUM_FMT handlers */
	.vidioc_enum_fmt_vid_cap = virtio_media_enum_fmt,
	.vidioc_enum_fmt_vid_overlay = virtio_media_enum_fmt,
	.vidioc_enum_fmt_vid_out = virtio_media_enum_fmt,
	.vidioc_enum_fmt_sdr_cap = virtio_media_enum_fmt,
	.vidioc_enum_fmt_sdr_out = virtio_media_enum_fmt,
	.vidioc_enum_fmt_meta_cap = virtio_media_enum_fmt,
	.vidioc_enum_fmt_meta_out = virtio_media_enum_fmt,

	/* VIDIOC_G_FMT handlers */
	.vidioc_g_fmt_vid_cap = virtio_media_g_fmt,
	.vidioc_g_fmt_vid_overlay = virtio_media_g_fmt,
	.vidioc_g_fmt_vid_out = virtio_media_g_fmt,
	.vidioc_g_fmt_vid_out_overlay = virtio_media_g_fmt,
	.vidioc_g_fmt_vbi_cap = virtio_media_g_fmt,
	.vidioc_g_fmt_vbi_out = virtio_media_g_fmt,
	.vidioc_g_fmt_sliced_vbi_cap = virtio_media_g_fmt,
	.vidioc_g_fmt_sliced_vbi_out = virtio_media_g_fmt,
	.vidioc_g_fmt_vid_cap_mplane = virtio_media_g_fmt,
	.vidioc_g_fmt_vid_out_mplane = virtio_media_g_fmt,
	.vidioc_g_fmt_sdr_cap = virtio_media_g_fmt,
	.vidioc_g_fmt_sdr_out = virtio_media_g_fmt,
	.vidioc_g_fmt_meta_cap = virtio_media_g_fmt,
	.vidioc_g_fmt_meta_out = virtio_media_g_fmt,

	/* VIDIOC_S_FMT handlers */
	.vidioc_s_fmt_vid_cap = virtio_media_s_fmt,
	.vidioc_s_fmt_vid_overlay = virtio_media_s_fmt,
	.vidioc_s_fmt_vid_out = virtio_media_s_fmt,
	.vidioc_s_fmt_vid_out_overlay = virtio_media_s_fmt,
	.vidioc_s_fmt_vbi_cap = virtio_media_s_fmt,
	.vidioc_s_fmt_vbi_out = virtio_media_s_fmt,
	.vidioc_s_fmt_sliced_vbi_cap = virtio_media_s_fmt,
	.vidioc_s_fmt_sliced_vbi_out = virtio_media_s_fmt,
	.vidioc_s_fmt_vid_cap_mplane = virtio_media_s_fmt,
	.vidioc_s_fmt_vid_out_mplane = virtio_media_s_fmt,
	.vidioc_s_fmt_sdr_cap = virtio_media_s_fmt,
	.vidioc_s_fmt_sdr_out = virtio_media_s_fmt,
	.vidioc_s_fmt_meta_cap = virtio_media_s_fmt,
	.vidioc_s_fmt_meta_out = virtio_media_s_fmt,

	/* VIDIOC_TRY_FMT handlers */
	.vidioc_try_fmt_vid_cap = virtio_media_try_fmt,
	.vidioc_try_fmt_vid_overlay = virtio_media_try_fmt,
	.vidioc_try_fmt_vid_out = virtio_media_try_fmt,
	.vidioc_try_fmt_vid_out_overlay = virtio_media_try_fmt,
	.vidioc_try_fmt_vbi_cap = virtio_media_try_fmt,
	.vidioc_try_fmt_vbi_out = virtio_media_try_fmt,
	.vidioc_try_fmt_sliced_vbi_cap = virtio_media_try_fmt,
	.vidioc_try_fmt_sliced_vbi_out = virtio_media_try_fmt,
	.vidioc_try_fmt_vid_cap_mplane = virtio_media_try_fmt,
	.vidioc_try_fmt_vid_out_mplane = virtio_media_try_fmt,
	.vidioc_try_fmt_sdr_cap = virtio_media_try_fmt,
	.vidioc_try_fmt_sdr_out = virtio_media_try_fmt,
	.vidioc_try_fmt_meta_cap = virtio_media_try_fmt,
	.vidioc_try_fmt_meta_out = virtio_media_try_fmt,

	/* Buffer handlers */
	.vidioc_reqbufs = virtio_media_reqbufs,
	.vidioc_querybuf = virtio_media_querybuf,
	.vidioc_qbuf = virtio_media_qbuf,
	.vidioc_expbuf = NULL,
	.vidioc_dqbuf = virtio_media_dqbuf,
	.vidioc_create_bufs = virtio_media_create_bufs,
	.vidioc_prepare_buf = virtio_media_prepare_buf,
	/* Overlay interface not supported yet */
	.vidioc_overlay = NULL,
	/* Overlay interface not supported yet */
	.vidioc_g_fbuf = NULL,
	/* Overlay interface not supported yet */
	.vidioc_s_fbuf = NULL,

	/* Stream on/off */
	.vidioc_streamon = virtio_media_streamon,
	.vidioc_streamoff = virtio_media_streamoff,

	/* Standard handling */
	.vidioc_g_std = virtio_media_g_std,
	.vidioc_s_std = virtio_media_s_std,
	.vidioc_querystd = virtio_media_querystd,

	/* Input handling */
	.vidioc_enum_input = virtio_media_enuminput,
	.vidioc_g_input = virtio_media_g_input,
	.vidioc_s_input = virtio_media_s_input,

	/* Output handling */
	.vidioc_enum_output = virtio_media_enumoutput,
	.vidioc_g_output = virtio_media_g_output,
	.vidioc_s_output = virtio_media_s_output,

	/* Control handling */
#if LINUX_VERSION_CODE < KERNEL_VERSION(6,15,0)
	.vidioc_queryctrl = virtio_media_queryctrl,
	.vidioc_g_ctrl = virtio_media_g_ctrl,
	.vidioc_s_ctrl = virtio_media_s_ctrl,
#endif
	.vidioc_query_ext_ctrl = virtio_media_query_ext_ctrl,
	.vidioc_g_ext_ctrls = virtio_media_g_ext_ctrls,
	.vidioc_s_ext_ctrls = virtio_media_s_ext_ctrls,
	.vidioc_try_ext_ctrls = virtio_media_try_ext_ctrls,
	.vidioc_querymenu = virtio_media_querymenu,

	/* Audio ioctls */
	.vidioc_enumaudio = virtio_media_enumaudio,
	.vidioc_g_audio = virtio_media_g_audio,
	.vidioc_s_audio = virtio_media_s_audio,

	/* Audio out ioctls */
	.vidioc_enumaudout = virtio_media_enumaudout,
	.vidioc_g_audout = virtio_media_g_audout,
	.vidioc_s_audout = virtio_media_s_audout,
	.vidioc_g_modulator = virtio_media_g_modulator,
	.vidioc_s_modulator = virtio_media_s_modulator,

	/* Crop ioctls */
	/* Not directly an ioctl (part of VIDIOC_CROPCAP), so no need to implement */
	.vidioc_g_pixelaspect = NULL,
	.vidioc_g_selection = virtio_media_g_selection,
	.vidioc_s_selection = virtio_media_s_selection,

	/* Compression ioctls */
	/* Deprecated in V4L2. */
	.vidioc_g_jpegcomp = NULL,
	/* Deprecated in V4L2. */
	.vidioc_s_jpegcomp = NULL,
	.vidioc_g_enc_index = virtio_media_g_enc_index,
	.vidioc_encoder_cmd = virtio_media_encoder_cmd,
	.vidioc_try_encoder_cmd = virtio_media_try_encoder_cmd,
	.vidioc_decoder_cmd = virtio_media_decoder_cmd,
	.vidioc_try_decoder_cmd = virtio_media_try_decoder_cmd,

	/*
	 * Stream type-dependent parameter ioctls.
	 *
	 * v4l2-compliance fails these on an m2m device that is not a stateful
	 * encoder (v4l2-test-formats.cpp:1445; defect D6.4 of the B3
	 * acceptance run): a non-encoder m2m driver must not offer G/S_PARM at
	 * all, i.e. the video device would have to v4l2_disable_ioctl() them.
	 * Deferred by decision, not an oversight: this table is shared by every
	 * virtio-media device, and the driver cannot tell an m2m decoder from a
	 * camera or from a device that legitimately implements G/S_PARM until
	 * the host says which ioctls its session supports. Revisit together
	 * with the codec device, when that contract exists.
	 */
	.vidioc_g_parm = virtio_media_g_parm,
	.vidioc_s_parm = virtio_media_s_parm,

	/* Tuner ioctls */
	.vidioc_g_tuner = virtio_media_g_tuner,
	.vidioc_s_tuner = virtio_media_s_tuner,
	.vidioc_g_frequency = virtio_media_g_frequency,
	.vidioc_s_frequency = virtio_media_s_frequency,
	.vidioc_enum_freq_bands = virtio_media_enum_freq_bands,

	/* Sliced VBI cap */
	.vidioc_g_sliced_vbi_cap = virtio_media_g_sliced_vbi_cap,

	/* Log status ioctl */
	/* Guest-only operation */
	.vidioc_log_status = NULL,

	.vidioc_s_hw_freq_seek = virtio_media_s_hw_freq_seek,

	.vidioc_enum_framesizes = virtio_media_enum_framesizes,
	.vidioc_enum_frameintervals = virtio_media_enum_frameintervals,

	/* DV Timings IOCTLs */
	.vidioc_s_dv_timings = virtio_media_s_dv_timings,
	.vidioc_g_dv_timings = virtio_media_g_dv_timings,
	.vidioc_query_dv_timings = virtio_media_query_dv_timings,
	.vidioc_enum_dv_timings = virtio_media_enum_dv_timings,
	.vidioc_dv_timings_cap = virtio_media_dv_timings_cap,
	.vidioc_g_edid = NULL,
	.vidioc_s_edid = NULL,

	.vidioc_subscribe_event = virtio_media_subscribe_event,
	.vidioc_unsubscribe_event = virtio_media_unsubscribe_event,

	/* For other private ioctls */
	.vidioc_default = NULL,
};

long virtio_media_device_ioctl(struct file *file, unsigned int cmd,
			       unsigned long arg)
{
	struct video_device *video_dev = video_devdata(file);
	struct virtio_media *vv = to_virtio_media(video_dev);
	struct v4l2_fh *vfh = file->private_data;
	struct v4l2_standard standard;
	v4l2_std_id std_id = 0;
	int ret;

	/*
	 * virtio_media_device_open() sets file->private_data (and so does
	 * v4l2_fh_add()), so this only fires if the session went away.
	 */
	if (!vfh)
		return -ENODEV;

	/* The host closed this session after an error: nothing can go through. */
	if (READ_ONCE(fh_to_session(vfh)->dead))
		return -ENODEV;

	/*
	 * A blocking VIDIOC_DQEVENT sleeps in v4l2_event_dequeue() until an
	 * event arrives. It never talks to the host -- it only reads the
	 * v4l2 event queue under vdev->fh_lock, filled by
	 * virtio_media_process_events() from the event work, which does not
	 * take vlock either. Dispatch it without the device lock, the way
	 * virtio_media_dqbuf() already drops vlock around its own wait:
	 * holding vlock across the sleep wedged every other ioctl and every
	 * open() of the node behind one waiter, including the very setter
	 * that would have produced the awaited event (defect D35,
	 * B7-controls §12.1). SUBSCRIBE/UNSUBSCRIBE_EVENT stay under vlock:
	 * they only wait for the host's bounded command response, never for
	 * guest-side activity.
	 *
	 * The core's own blocking wait only ever ends when an event arrives
	 * (v4l2_event_dequeue()'s condition is fh->navailable alone), so a
	 * waiter inside it would sleep through a device removal
	 * indefinitely. Wait here instead, where the disconnect path's
	 * wake-up can end the sleep -- virtio_media_remove() sets
	 * session->dead before waking fh->wait (D66) -- and only then let
	 * the core dequeue what is pending, which it does without blocking.
	 * (If a second reader steals the pending event first the core blocks
	 * again until the next one; a multi-reader race the plain dispatch
	 * had as well.)
	 */
	if (cmd == VIDIOC_DQEVENT) {
		if (!(file->f_flags & O_NONBLOCK)) {
			ret = wait_event_interruptible(
				vfh->wait,
				v4l2_event_pending(vfh) ||
					READ_ONCE(fh_to_session(vfh)->dead));
			if (ret)
				return ret;
			if (READ_ONCE(fh_to_session(vfh)->dead))
				return -ENODEV;
		}
		return video_ioctl2(file, cmd, arg);
	}

	mutex_lock(&vv->vlock);

	/*
	 * We need to handle a few ioctls manually because their result rely on
	 * vfd->tvnorms, which is normally updated by the driver as S_INPUT is
	 * called. Since we want to just pass these ioctls through, we have to hijack
	 * them from here.
	 */
	switch (cmd) {
	case VIDIOC_S_STD:
		ret = copy_from_user(&std_id, (void __user *)arg,
				     sizeof(std_id));
		if (ret) {
			ret = -EINVAL;
			break;
		}
		ret = virtio_media_s_std(file, NULL, std_id);
		break;
	case VIDIOC_ENUMSTD:
		ret = copy_from_user(&standard, (void __user *)arg,
				     sizeof(standard));
		if (ret) {
			ret = -EINVAL;
			break;
		}
		ret = virtio_media_enumstd(file, NULL, &standard);
		if (ret)
			break;
		ret = copy_to_user((void __user *)arg, &standard,
				   sizeof(standard));
		if (ret)
			ret = -EINVAL;
		break;
	case VIDIOC_QUERYSTD:
		ret = virtio_media_querystd(file, NULL, &std_id);
		if (ret)
			break;
		ret = copy_to_user((void __user *)arg, &std_id, sizeof(std_id));
		if (ret)
			ret = -EINVAL;
		break;
	default:
		ret = video_ioctl2(file, cmd, arg);
		break;
	}

	mutex_unlock(&vv->vlock);

	return ret;
}
