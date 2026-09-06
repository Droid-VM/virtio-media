// SPDX-License-Identifier: BSD-3-Clause OR GPL-2.0+

/*
 * Virtio-media driver.
 *
 * Copyright (c) 2023-2024 Google LLC.
 */

#include <linux/delay.h>
#include <linux/device.h>
#include <linux/ioport.h>
#include <linux/mm.h>
#include <linux/mutex.h>
#include <linux/of.h>
#include <linux/of_address.h>
#include <linux/scatterlist.h>
#include <linux/types.h>
#include <linux/videodev2.h>
#include <linux/vmalloc.h>
#include <linux/wait.h>
#include <linux/workqueue.h>
#include <linux/module.h>
#include <linux/moduleparam.h>
#include <linux/version.h>
#include <linux/virtio.h>
#include <linux/virtio_config.h>
#include <linux/virtio_ids.h>

#include <media/frame_vector.h>
#include <media/v4l2-dev.h>
#include <media/v4l2-event.h>
#include <media/videobuf2-memops.h>
#include <media/v4l2-device.h>
#include <media/v4l2-ioctl.h>

#include "protocol.h"
#include "session.h"
#include "virtio_media.h"
#include "virtio_media_alloc.h"

#define VIRTIO_MEDIA_NUM_EVENT_BUFS 16

#ifndef VIRTIO_ID_MEDIA
#define VIRTIO_ID_MEDIA 48
#endif

/* ID of the SHM region into which MMAP buffer will be mapped. */
#define VIRTIO_MEDIA_SHM_MMAP 0

/*
 * Name of the driver to expose to user-space.
 *
 * This is configurable because v4l2-compliance has workarounds specific to
 * some drivers. When proxying these directly from the host, this allows it to
 * apply them as needed.
 */
char *driver_name = NULL;
module_param(driver_name, charp, 0660);

/*
 * Which queues get driver-owned buffers when user-space asks for MMAP
 * (VPU_DESIGN.md 2.1): "output" (default) = the queues the guest fills, i.e.
 * V4L2_TYPE_IS_OUTPUT; "all" = every queue, so no media_host pool is needed;
 * "none" = every MMAP buffer is host-owned, the upstream behaviour, for A/B
 * comparison. Evaluated at REQBUFS/CREATE_BUFS time.
 */
char *driver_owned_queues = "output";

/*
 * Only the three values above are accepted, at insmod time and through sysfs:
 * a typo used to be logged once and silently behave as "output", which made a
 * mistyped A/B comparison look like a driver bug (review nit on WP G1).
 */
static int driver_owned_queues_set(const char *val,
				   const struct kernel_param *kp)
{
	if (!val)
		return -EINVAL;

	if (!sysfs_streq(val, "output") && !sysfs_streq(val, "all") &&
	    !sysfs_streq(val, "none"))
		return -EINVAL;

	return param_set_charp(val, kp);
}

static const struct kernel_param_ops driver_owned_queues_ops = {
	.set = driver_owned_queues_set,
	.get = param_get_charp,
	.free = param_free_charp,
};

module_param_cb(driver_owned_queues, &driver_owned_queues_ops,
		&driver_owned_queues, 0660);
MODULE_PARM_DESC(driver_owned_queues,
		 "queues whose MMAP buffers the driver allocates: output (default), all, none");

/**
 * Allocate a new session. The id and list fields must still be set by the
 * caller.
 */
static struct virtio_media_session *
virtio_media_session_alloc(struct virtio_media *vv, u32 id,
			   bool nonblocking_dequeue, struct file *file)
{
	struct virtio_media_session *session;
	int i;
	int ret;

	session = kzalloc(sizeof(*session), GFP_KERNEL);
	if (!session)
		goto err_session;

	session->shadow_buf = kzalloc(VIRTIO_SHADOW_BUF_SIZE, GFP_KERNEL);
	if (!session->shadow_buf)
		goto err_shadow_buf;

	ret = sg_alloc_table(&session->command_sgs, DESC_CHAIN_MAX_LEN,
			     GFP_KERNEL);
	if (ret) {
		goto err_payload_sgs;
	}

	session->id = id;
	session->nonblocking_dequeue = nonblocking_dequeue;
	session->file = file;
	session->next_dbuf_cookie = VMEDIA_DBUF_COOKIE_BASE;

	INIT_LIST_HEAD(&session->list);
	v4l2_fh_init(&session->fh, &vv->video_dev);
#if LINUX_VERSION_CODE >= KERNEL_VERSION(6, 17, 0)
	/* Also sets file->private_data = &session->fh. */
	v4l2_fh_add(&session->fh, file);
#else
	v4l2_fh_add(&session->fh);
#endif

	for (i = 0; i <= VIRTIO_MEDIA_LAST_QUEUE; i++)
		INIT_LIST_HEAD(&session->queues[i].pending_dqbufs);
	mutex_init(&session->dqbufs_lock);

	init_waitqueue_head(&session->dqbufs_wait);

	mutex_lock(&vv->sessions_lock);
	list_add_tail(&session->list, &vv->sessions);
	/*
	 * A remove() that ran between CMD_OPEN and this list_add marked every
	 * listed session dead but could not see this one: inherit the verdict
	 * so no session on a disconnected device ever looks alive (D66).
	 */
	if (READ_ONCE(vv->disconnected))
		WRITE_ONCE(session->dead, true);
	mutex_unlock(&vv->sessions_lock);

	return session;

err_payload_sgs:
	kfree(session->shadow_buf);
err_shadow_buf:
	kfree(session);
err_session:
	return ERR_PTR(-ENOMEM);
}

/**
 * Close and destroy `session`.
 */
static void virtio_media_session_close(struct virtio_media *vv,
				       struct virtio_media_session *session)
{
	int i;

	mutex_lock(&vv->sessions_lock);
	list_del(&session->list);
	mutex_unlock(&vv->sessions_lock);

	/*
	 * The event work may have looked this session up before the
	 * list_del and still be delivering an event into its queues: wait
	 * that run out before freeing anything it touches (same D46 family
	 * as the REQBUFS race). A run that starts after the flush cannot
	 * find the session any more and drops its events. No lock is held
	 * here, and the work only takes sessions_lock and dqbufs_lock, so
	 * this cannot deadlock; after a device removal the work was already
	 * cancelled and this is a no-op.
	 */
	flush_work(&vv->eventq_work);

#if LINUX_VERSION_CODE >= KERNEL_VERSION(6, 17, 0)
	v4l2_fh_del(&session->fh, session->file);
#else
	v4l2_fh_del(&session->fh);
#endif
	v4l2_fh_exit(&session->fh);

	sg_free_table(&session->command_sgs);

	/*
	 * The host has let go of the buffers by now -- it processed
	 * CMD_CLOSE, or the whole device was reset at the unbind (D66) --
	 * so its mappings of driver-owned buffers are gone and the memory
	 * can go back to the pool (VPU_DESIGN.md 2.5).
	 */
	for (i = 0; i <= VIRTIO_MEDIA_LAST_QUEUE; i++)
		if (session->queues[i].buffers) {
			vmedia_queue_put_dbufs(&session->queues[i]);
			vfree(session->queues[i].buffers);
		}

	kfree(session->shadow_buf);
	kfree(session);
}

/**
 * Lookup the session with `id`.
 */
static struct virtio_media_session *
virtio_media_find_session(struct virtio_media *vv, u32 id)
{
	struct list_head *p;
	struct virtio_media_session *session = NULL;

	mutex_lock(&vv->sessions_lock);
	list_for_each(p, &vv->sessions) {
		struct virtio_media_session *s =
			list_entry(p, struct virtio_media_session, list);
		if (s->id == id) {
			session = s;
			break;
		}
	}
	mutex_unlock(&vv->sessions_lock);

	return session;
}

/**
 * Callback parameters to the virtio command queue.
 */
struct virtio_media_cmd_callback_param {
	struct virtio_media *vv;
	/* Flag to switch once the command is completed */
	bool done_flag;
	/* Size of the received response */
	size_t resp_len;
};

/**
 * Callback for the command queue. This just wakes up the thread that was
 * waiting on the command to complete.
 */
static void commandq_callback(struct virtqueue *queue)
{
	unsigned int len;
	struct virtio_media_cmd_callback_param *param;

	while ((param = virtqueue_get_buf(queue, &len))) {
		param->done_flag = true;
		param->resp_len = len;
		wake_up(&param->vv->wq);
	}

	virtqueue_enable_cb(queue);
}

/**
 * Returns 0 in case of success, or a negative error code.
 */
static int virtio_media_kick_command(struct virtio_media *vv,
				     struct scatterlist **sgs,
				     const size_t out_sgs, const size_t in_sgs,
				     size_t *resp_len)
{
	struct virtio_media_cmd_callback_param cb_param = {
		.vv = vv,
		.done_flag = false,
		.resp_len = 0,
	};
	struct virtio_media_resp_header *resp_header;
	int ret;

	/*
	 * Every sender holds vv->vlock and virtio_media_remove() sets the
	 * flag under it, so a disconnect never catches a command half-added:
	 * either the command completed before remove() could take the lock,
	 * or the sender sees the flag here, before touching a virtqueue that
	 * is about to be reset and deleted (D66).
	 */
	if (vv->disconnected)
		return -ENODEV;

	ret = virtqueue_add_sgs(vv->commandq, sgs, out_sgs, in_sgs, &cb_param,
				GFP_ATOMIC);
	if (ret) {
		v4l2_err(&vv->v4l2_dev,
			 "failed to add sgs to command virtqueue\n");
		return ret;
	}

	if (!virtqueue_kick(vv->commandq)) {
		v4l2_err(&vv->v4l2_dev, "failed to kick command virtqueue\n");
		return -EINVAL;
	}

	/* Wait for the response. */
	ret = wait_event_timeout(vv->wq, cb_param.done_flag, 5 * HZ);
	if (ret == 0) {
		v4l2_err(&vv->v4l2_dev,
			 "timed out waiting for response to command\n");
		return -ETIMEDOUT;
	}

	if (resp_len)
		*resp_len = cb_param.resp_len;

	if (in_sgs > 0) {
		/* 
		 * If we expect a response, make sure we have at least a response header - anything shorter is
		 * invalid.
		 */
		if (cb_param.resp_len < sizeof(*resp_header)) {
			v4l2_err(&vv->v4l2_dev,
				 "received response header is too short\n");
			return -EINVAL;
		}

		resp_header = sg_virt(sgs[out_sgs]);
		if (resp_header->status)
			/* Host returns a positive error code. */
			return -resp_header->status;
	}

	return 0;
}

/**
 * Send a command to the host and wait for its response.
 * @vv: the virtio_media device to communicate with.
 * @minimum_resp_len: the minimum length of the response expected by the caller
 * in case the command succeeded. Anything shorter than that will result in an
 * error.
 *
 * Returns 0 in case of success or an error code. If an error is returned,
 * resp_len might not have been updated.
 */
int virtio_media_send_command(struct virtio_media *vv, struct scatterlist **sgs,
			      const size_t out_sgs, const size_t in_sgs,
			      size_t minimum_resp_len, size_t *resp_len)
{
	size_t local_resp_len = resp_len ? *resp_len : 0;
	int ret = virtio_media_kick_command(vv, sgs, out_sgs, in_sgs,
					    &local_resp_len);
	if (resp_len)
		*resp_len = local_resp_len;

	/* If the host could not process the command, there is no valid response */
	if (ret < 0)
		return ret;

	/* Make sure the host wrote a complete reply. */
	if (local_resp_len < minimum_resp_len) {
		v4l2_err(
			&vv->v4l2_dev,
			"received response is too short: received %zu, expected at least %zu\n",
			local_resp_len, minimum_resp_len);
		return -EINVAL;
	}

	return 0;
}

/**
 * Send the event buffer to the host so it can return it back to us filled with
 * the next event that occurred.
 */
static int virtio_media_send_event_buffer(struct virtio_media *vv,
					  void *event_buffer)
{
	struct scatterlist *sgs[1], vresp;
	int ret;

	sg_init_one(&vresp, event_buffer, VIRTIO_MEDIA_EVENT_MAX_SIZE);
	sgs[0] = &vresp;

	ret = virtqueue_add_sgs(vv->eventq, sgs, 0, 1, event_buffer,
				GFP_ATOMIC);
	if (ret) {
		v4l2_err(&vv->v4l2_dev,
			 "failed to add sgs to event virtqueue\n");
		return ret;
	}

	if (!virtqueue_kick(vv->eventq)) {
		v4l2_err(&vv->v4l2_dev, "failed to kick event virtqueue\n");
		return -EINVAL;
	}

	return 0;
}

static void eventq_callback(struct virtqueue *queue)
{
	struct virtio_media *vv = queue->vdev->priv;

	schedule_work(&vv->eventq_work);
}

static void
virtio_media_process_dqbuf_event(struct virtio_media *vv,
				 struct virtio_media_session *session,
				 struct virtio_media_event_dqbuf *dqbuf_evt)
{
	struct virtio_media_buffer *dqbuf;
	const enum v4l2_buf_type queue_type = dqbuf_evt->buffer.type;
	struct virtio_media_queue_state *queue;
	typeof(dqbuf->buffer.m) buffer_m;
	typeof(dqbuf->buffer.m.planes[0].m) plane_m;
	int i;

	if (queue_type >= ARRAY_SIZE(session->queues)) {
		v4l2_err(&vv->v4l2_dev,
			 "unmanaged queue %d passed to dqbuf event",
			 dqbuf_evt->buffer.type);
		return;
	}
	queue = &session->queues[queue_type];

	/*
	 * dqbufs_lock guards the queue's buffer array itself, not only
	 * @pending_dqbufs: VIDIOC_REQBUFS/CREATE_BUFS vfree() and swap
	 * @buffers and @allocated_bufs under it, and this work runs under
	 * events_process_lock only, which the ioctl path never takes.
	 * Dereferencing the array without the lock was a use-after-free
	 * whenever a REQBUFS raced an event in flight (D46, B8 §10).
	 * Lock order: events_process_lock -> dqbufs_lock, the order the
	 * pending_dqbufs insertion below has always used; never take
	 * vv->vlock or events_process_lock from under dqbufs_lock.
	 */
	mutex_lock(&session->dqbufs_lock);

	if (dqbuf_evt->buffer.index >= queue->allocated_bufs) {
		/*
		 * With no buffers at all this is not host misbehavior: a
		 * REQBUFS(0) tore the queue down while this event was in
		 * flight (the normal STREAMOFF + REQBUFS(0) shutdown), and
		 * the buffer it names no longer exists. Drop it quietly.
		 */
		if (queue->allocated_bufs == 0)
			pr_debug("virtio-media: dropping dqbuf event for buffer %u of freed queue %d\n",
				 dqbuf_evt->buffer.index,
				 dqbuf_evt->buffer.type);
		else
			v4l2_err(&vv->v4l2_dev,
				 "invalid buffer ID %d for queue %d in dqbuf event",
				 dqbuf_evt->buffer.index,
				 dqbuf_evt->buffer.type);
		goto out_unlock;
	}

	dqbuf = &queue->buffers[dqbuf_evt->buffer.index];

	/*
	 * Only a buffer the guest queued can be dequeued. A freshly
	 * reallocated array is zeroed, so a stale event that raced a
	 * REQBUFS(n) and indexes into the new allocation lands here instead
	 * of corrupting a buffer that was never queued; so does a host that
	 * completes the same buffer twice (the first event replaced QUEUED
	 * with DONE).
	 */
	if (!(dqbuf->buffer.flags & V4L2_BUF_FLAG_QUEUED) ||
	    (dqbuf->buffer.flags & V4L2_BUF_FLAG_DONE)) {
		pr_debug("virtio-media: dropping dqbuf event for buffer %u of queue %d that is not queued\n",
			 dqbuf_evt->buffer.index, dqbuf_evt->buffer.type);
		goto out_unlock;
	}

	/*
	 * Preserve the 'm' union that was passed to us during QBUF so userspace
	 * gets back the information it submitted.
	 */
	buffer_m = dqbuf->buffer.m;
	memcpy(&dqbuf->buffer, &dqbuf_evt->buffer, sizeof(dqbuf->buffer));
	dqbuf->buffer.m = buffer_m;
	if (V4L2_TYPE_IS_MULTIPLANAR(dqbuf->buffer.type)) {
		if (dqbuf->buffer.length > VIDEO_MAX_PLANES) {
			v4l2_err(
				&vv->v4l2_dev,
				"invalid number of planes received from host for "
				"a multiplanar buffer\n");
			goto out_unlock;
		}
		for (i = 0; i < dqbuf->buffer.length; i++) {
			plane_m = dqbuf->planes[i].m;
			memcpy(&dqbuf->planes[i], &dqbuf_evt->planes[i],
			       sizeof(struct v4l2_plane));
			dqbuf->planes[i].m = plane_m;
		}
	}

	/*
	 * A driver-owned buffer went to the host as USERPTR: give user-space
	 * back the MMAP memory type, its cookie and its length
	 * (VPU_DESIGN.md 5.3 item 5).
	 */
	if (dqbuf->dbuf[0])
		vmedia_dbuf_buffer_from_host(&dqbuf->buffer, dqbuf->planes,
					     VIDEO_MAX_PLANES, dqbuf->dbuf);

	/* Set the DONE flag as the buffer is waiting for being dequeued. */
	dqbuf->buffer.flags |= V4L2_BUF_FLAG_DONE;

	list_add_tail(&dqbuf->list, &queue->pending_dqbufs);
	/*
	 * Guarded: VIDIOC_QBUF counts the buffer before the host sees it,
	 * so a legitimate completion always observes the increment; only a
	 * host that dequeues something it was never given lands on zero.
	 */
	if (queue->queued_bufs > 0)
		queue->queued_bufs -= 1;

out_unlock:
	mutex_unlock(&session->dqbufs_lock);
	wake_up(&session->dqbufs_wait);
}

void virtio_media_process_events(struct virtio_media *vv)
{
	struct virtio_media_event_error *error_evt;
	struct virtio_media_event_dqbuf *dqbuf_evt;
	struct virtio_media_event_event *event_evt;
	struct virtio_media_session *session;
	struct virtio_media_event_header *evt;
	unsigned int len;

	mutex_lock(&vv->events_process_lock);

	/*
	 * A work run scheduled by a last interrupt can land here after
	 * remove() reset the device and deleted the virtqueues; the flag is
	 * set before the reset, so leave without touching them (D66).
	 */
	if (READ_ONCE(vv->disconnected)) {
		mutex_unlock(&vv->events_process_lock);
		return;
	}

	while ((evt = virtqueue_get_buf(vv->eventq, &len))) {
		/* Make sure we received enough data */
		if (len < sizeof(*evt)) {
			v4l2_err(
				&vv->v4l2_dev,
				"event is too short: got %u, expected at least %zu\n",
				len, sizeof(*evt));
			goto end_of_event;
		}

		session = virtio_media_find_session(vv, evt->session_id);
		if (session == NULL) {
			v4l2_err(&vv->v4l2_dev, "cannot find session %d\n",
				 evt->session_id);
			goto end_of_event;
		}

		switch (evt->event) {
		case VIRTIO_MEDIA_EVT_ERROR:
			if (len < sizeof(*error_evt)) {
				v4l2_err(
					&vv->v4l2_dev,
					"error event is too short: got %u, expected %zu\n",
					len, sizeof(*error_evt));
				break;
			}
			error_evt = (struct virtio_media_event_error *)evt;
			v4l2_err(&vv->v4l2_dev,
				 "received error %d for session %d, marking it dead\n",
				 error_evt->errno, error_evt->hdr.session_id);
			/*
			 * The host considers the session corrupted and closed
			 * (protocol.h). Fail every further ioctl with -ENODEV,
			 * make poll report EPOLLERR and release anyone blocked
			 * in DQBUF, so a camera or codec reclaimed on the host
			 * gives the guest process a clean exit instead of a
			 * hang.
			 */
			WRITE_ONCE(session->dead, true);
			wake_up(&session->dqbufs_wait);
			break;

		/*
		 * Dequeued buffer: put it into the right queue so user-space can dequeue
		 * it.
		 */
		case VIRTIO_MEDIA_EVT_DQBUF:
			if (len < sizeof(*dqbuf_evt)) {
				v4l2_err(
					&vv->v4l2_dev,
					"dqbuf event is too short: got %u, expected %zu\n",
					len, sizeof(*dqbuf_evt));
				break;
			}
			dqbuf_evt = (struct virtio_media_event_dqbuf *)evt;
			virtio_media_process_dqbuf_event(vv, session,
							 dqbuf_evt);
			break;

		case VIRTIO_MEDIA_EVT_EVENT:
			if (len < sizeof(*event_evt)) {
				v4l2_err(
					&vv->v4l2_dev,
					"session event is too short: got %u expected %zu\n",
					len, sizeof(*event_evt));
				break;
			}

			event_evt = (struct virtio_media_event_event *)evt;
			v4l2_event_queue_fh(&session->fh, &event_evt->event);
			break;

		default:
			v4l2_err(&vv->v4l2_dev, "unknown event type %d\n",
				 evt->event);
			break;
		}

end_of_event:
		virtio_media_send_event_buffer(vv, evt);
	}

	virtqueue_enable_cb(vv->eventq);

	mutex_unlock(&vv->events_process_lock);
}

/**
 * Event callback. This processes the returned event buffer and immediately
 * sends it again to the host so it can send us the next event without ever
 * starving.
 */
static void virtio_media_event_work(struct work_struct *work)
{
	struct virtio_media *vv =
		container_of(work, struct virtio_media, eventq_work);

	virtio_media_process_events(vv);
}

/**
 * Opens the device and create a new session.
 */
static int virtio_media_device_open(struct file *file)
{
	struct video_device *video_dev = video_devdata(file);
	struct virtio_media *vv = to_virtio_media(video_dev);
	struct virtio_media_cmd_open *cmd_open = &vv->cmd.open;
	struct virtio_media_resp_open *resp_open = &vv->resp.open;
	struct scatterlist cmd_sg = {}, resp_sg = {};
	struct scatterlist *sgs[2] = { &cmd_sg, &resp_sg };
	struct virtio_media_session *session;
	u32 session_id;
	int ret;

	mutex_lock(&vv->vlock);

	sg_set_buf(&cmd_sg, cmd_open, sizeof(*cmd_open));
	sg_mark_end(&cmd_sg);

	sg_set_buf(&resp_sg, resp_open, sizeof(*resp_open));
	sg_mark_end(&resp_sg);

	mutex_lock(&vv->bufs_lock);
	cmd_open->hdr.cmd = VIRTIO_MEDIA_CMD_OPEN;
	ret = virtio_media_send_command(vv, sgs, 1, 1, sizeof(*resp_open),
					NULL);
	session_id = resp_open->session_id;
	mutex_unlock(&vv->bufs_lock);
	mutex_unlock(&vv->vlock);
	if (ret < 0)
		return ret;

	session = virtio_media_session_alloc(vv, session_id,
					     (file->f_flags & O_NONBLOCK), file);
	if (IS_ERR(session))
		return PTR_ERR(session);

	file->private_data = &session->fh;

	return 0;
}

/**
 * Close a previously opened session.
 */
static int virtio_media_device_close(struct file *file)
{
	struct video_device *video_dev = video_devdata(file);
	struct virtio_media *vv = to_virtio_media(video_dev);
	struct v4l2_fh *fh = file->private_data;
	struct virtio_media_session *session;
	struct virtio_media_cmd_close *cmd_close;
	struct scatterlist cmd_sg = {};
	struct scatterlist *sgs[1] = { &cmd_sg };
	int ret;

	if (!fh)
		return -ENODEV;
	session = fh_to_session(fh);
	cmd_close = &session->cmd.close;

	mutex_lock(&vv->vlock);
	if (!vv->disconnected) {
		cmd_close->hdr.cmd = VIRTIO_MEDIA_CMD_CLOSE;
		cmd_close->session_id = session->id;

		sg_set_buf(&cmd_sg, cmd_close, sizeof(*cmd_close));
		sg_mark_end(&cmd_sg);

		ret = virtio_media_send_command(vv, sgs, 1, 0, 0, NULL);
		if (ret < 0)
			v4l2_err(&vv->v4l2_dev,
				 "failed to close session %u: %d; freeing it anyway\n",
				 session->id, ret);
	}
	mutex_unlock(&vv->vlock);

	/*
	 * Freed unconditionally: the file handle is gone whatever CMD_CLOSE
	 * said, and on a disconnected device (D66) there is nobody to tell --
	 * the VMM already reclaimed the session when the device reset
	 * (B12-acceptance section 5 measured exactly that). Returning early
	 * on a send error used to leak the session and leave it on
	 * vv->sessions.
	 */
	virtio_media_session_close(vv, session);

	return 0;
}

/**
 * Implements poll logic for a virtio-media device.
 */
static __poll_t virtio_media_device_poll(struct file *file, poll_table *wait)
{
	struct v4l2_fh *fh = file->private_data;
	struct virtio_media_session *session;
	enum v4l2_buf_type capture_type;
	enum v4l2_buf_type output_type;
	struct virtio_media_queue_state *capture_queue;
	struct virtio_media_queue_state *output_queue;
	__poll_t req_events = poll_requested_events(wait);
	__poll_t rc = 0;

	if (!fh)
		return EPOLLERR;
	session = fh_to_session(fh);
	capture_type = session->uses_mplane ?
			       V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE :
			       V4L2_BUF_TYPE_VIDEO_CAPTURE;
	output_type = session->uses_mplane ? V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE :
					     V4L2_BUF_TYPE_VIDEO_OUTPUT;
	capture_queue = &session->queues[capture_type];
	output_queue = &session->queues[output_type];

	if (READ_ONCE(session->dead))
		return EPOLLERR;

	poll_wait(file, &session->dqbufs_wait, wait);
	poll_wait(file, &session->fh.wait, wait);

	mutex_lock(&session->dqbufs_lock);
	if (fh->vdev->vfl_dir == VFL_DIR_M2M &&
	    (req_events &
	     (EPOLLIN | EPOLLRDNORM | EPOLLOUT | EPOLLWRNORM))) {
		/*
		 * m2m nodes follow v4l2_m2m_poll_for_data()
		 * (v4l2-mem2mem.c:912-949), not vb2_core_poll(): EPOLLERR
		 * only when *neither* queue can make progress, because a
		 * client is expected to poll the CAPTURE side while only
		 * OUTPUT is streaming yet (GStreamer prerolls exactly like
		 * that, waiting for SOURCE_CHANGE before it allocates
		 * CAPTURE buffers -- reporting EPOLLERR there was defect
		 * D26/D40). A queue can make progress when it is streaming
		 * and holds a buffer either at the host (queued_bufs, vb2's
		 * @queued_list) or ready to dequeue (@pending_dqbufs, vb2's
		 * @done_list); a CAPTURE queue whose LAST buffer was
		 * dequeued reports EPOLLIN so the client hears the -EPIPE
		 * (v4l2-mem2mem.c:940-945).
		 */
		bool out_usable = output_queue->streaming &&
				  (output_queue->queued_bufs > 0 ||
				   !list_empty(&output_queue->pending_dqbufs));
		bool cap_usable = capture_queue->streaming &&
				  (capture_queue->queued_bufs > 0 ||
				   !list_empty(&capture_queue->pending_dqbufs) ||
				   capture_queue->is_capture_last);

		if (!out_usable && !cap_usable) {
			rc |= EPOLLERR;
		} else {
			if (!list_empty(&output_queue->pending_dqbufs))
				rc |= EPOLLOUT | EPOLLWRNORM;
			if (!list_empty(&capture_queue->pending_dqbufs) ||
			    capture_queue->is_capture_last)
				rc |= EPOLLIN | EPOLLRDNORM;
		}
	} else if (fh->vdev->vfl_dir != VFL_DIR_M2M) {
		if (req_events & (EPOLLIN | EPOLLRDNORM)) {
			/*
			 * Mirror vb2_core_poll(): not streaming or nothing to
			 * wait for -> EPOLLERR (the waiting_for_buffers
			 * quirk); an empty done list whose LAST buffer was
			 * already dequeued -> EPOLLIN, so the client issues
			 * the DQBUF that answers -EPIPE and ends its drain
			 * (videobuf2-core.c:2769-2776).
			 */
			if (!capture_queue->streaming ||
			    (capture_queue->queued_bufs == 0 &&
			     list_empty(&capture_queue->pending_dqbufs) &&
			     !capture_queue->is_capture_last))
				rc |= EPOLLERR;
			else if (!list_empty(&capture_queue->pending_dqbufs) ||
				 capture_queue->is_capture_last)
				rc |= EPOLLIN | EPOLLRDNORM;
		}
		/*
		 * EPOLLOUT on an OUTPUT queue means a buffer is ready to be
		 * *dequeued*, not that a free slot is ready to be queued
		 * into: vb2_core_poll() reports a free slot only for queues
		 * that emulate write() (VB2_WRITE), which virtio-media does
		 * not offer. Reporting a free slot invited a blocking DQBUF
		 * that could never complete (defect D10, B3 acceptance
		 * §5.3). Mirror vb2: not streaming or no buffers ->
		 * EPOLLERR, otherwise EPOLLOUT only for a buffer waiting in
		 * @pending_dqbufs (vb2's @done_list); vb2's
		 * waiting_for_buffers quirk is capture-only, so an
		 * idle-but-streaming OUTPUT queue reports nothing at all.
		 */
		if (req_events & (EPOLLOUT | EPOLLWRNORM)) {
			if (!output_queue->streaming ||
			    output_queue->allocated_bufs == 0)
				rc |= EPOLLERR;
			else if (!list_empty(&output_queue->pending_dqbufs))
				rc |= EPOLLOUT | EPOLLWRNORM;
		}
	}
	mutex_unlock(&session->dqbufs_lock);

	if (v4l2_event_pending(&session->fh))
		rc |= EPOLLPRI;

	return rc;
}

/**
 * struct virtio_media_hostmap - One host MMAP mapping and the VMAs using it.
 *
 * @vv: device the mapping belongs to.
 * @driver_addr: offset the host returned in VIRTIO_MEDIA_CMD_MMAP, i.e. what
 *	VIRTIO_MEDIA_CMD_MUNMAP must be sent with.
 * @vmas: number of VMAs currently referencing this mapping.
 *
 * Upstream kept no state per mapping and derived the MUNMAP offset from
 * vm_pgoff in the sole .close callback. A fork() duplicates the VMA and a
 * partial munmap() splits it, so the host then received one MUNMAP per
 * resulting VMA -- and the tail of a split computed a wrong offset because
 * the split shifts vm_pgoff. Keeping the host offset here and counting the
 * VMAs through .open/.close sends exactly one MUNMAP, once the last VMA is
 * gone (VPU_DESIGN.md 2.5, 5.4). The map also holds a v4l2_dev reference:
 * a VMA can outlive both the file handle and the driver binding (D66), and
 * @vv must still be there for the vlock and the disconnected gate.
 */
struct virtio_media_hostmap {
	struct virtio_media *vv;
	u64 driver_addr;
	refcount_t vmas;
};

/**
 * Inform the host that a previously created MMAP mapping is no longer needed
 * and can be removed. Called with vv->vlock held.
 */
static void virtio_media_host_munmap_locked(struct virtio_media *vv,
					    u64 driver_addr)
{
	struct virtio_media_cmd_munmap *cmd_munmap = &vv->cmd.munmap;
	struct virtio_media_resp_munmap *resp_munmap = &vv->resp.munmap;
	struct scatterlist cmd_sg = {}, resp_sg = {};
	struct scatterlist *sgs[2] = { &cmd_sg, &resp_sg };
	int ret;

	sg_set_buf(&cmd_sg, cmd_munmap, sizeof(*cmd_munmap));
	sg_mark_end(&cmd_sg);

	sg_set_buf(&resp_sg, resp_munmap, sizeof(*resp_munmap));
	sg_mark_end(&resp_sg);

	mutex_lock(&vv->bufs_lock);
	cmd_munmap->hdr.cmd = VIRTIO_MEDIA_CMD_MUNMAP;
	cmd_munmap->driver_addr = driver_addr;
	ret = virtio_media_send_command(vv, sgs, 1, 1, sizeof(*resp_munmap),
					NULL);
	mutex_unlock(&vv->bufs_lock);
	if (ret < 0) {
		v4l2_err(&vv->v4l2_dev, "host failed to unmap buffer: %d\n",
			 ret);
	}
}

static void virtio_media_vma_open(struct vm_area_struct *vma)
{
	struct virtio_media_hostmap *map = vma->vm_private_data;

	refcount_inc(&map->vmas);
}

static void virtio_media_vma_close(struct vm_area_struct *vma)
{
	struct virtio_media_hostmap *map = vma->vm_private_data;
	struct virtio_media *vv = map->vv;

	if (!refcount_dec_and_test(&map->vmas))
		return;

	mutex_lock(&vv->vlock);
	/*
	 * After a disconnect there is nobody to tell: the host's mappings
	 * died with the device (D66).
	 */
	if (!vv->disconnected)
		virtio_media_host_munmap_locked(vv, map->driver_addr);
	mutex_unlock(&vv->vlock);
	kfree(map);
	v4l2_device_put(&vv->v4l2_dev);
}

static const struct vm_operations_struct virtio_media_vm_ops = {
	.open = virtio_media_vma_open,
	.close = virtio_media_vma_close,
};

/**
 * Perform a mmap request from the guest.
 *
 * For a driver-owned buffer (cookie >= VMEDIA_DBUF_COOKIE_BASE) the pages are
 * ours and get mapped directly; the host is not involved. Otherwise this
 * requests the host to map a MMAP buffer for us, so we can make that mapping
 * visible into the user-space address space.
 */
static int virtio_media_device_mmap(struct file *file,
				    struct vm_area_struct *vma)
{
	struct video_device *video_dev = video_devdata(file);
	struct virtio_media *vv = to_virtio_media(video_dev);
	struct v4l2_fh *fh = file->private_data;
	struct virtio_media_session *session;
	struct virtio_media_cmd_mmap *cmd_mmap;
	struct virtio_media_resp_mmap *resp_mmap;
	struct scatterlist cmd_sg = {}, resp_sg = {};
	struct scatterlist *sgs[2] = { &cmd_sg, &resp_sg };
	struct virtio_media_hostmap *map;
	const u64 cookie = (u64)vma->vm_pgoff << PAGE_SHIFT;
	u64 driver_addr;
	u64 len;
	int ret;

	if (!fh)
		return -ENODEV;
	session = fh_to_session(fh);
	cmd_mmap = &session->cmd.mmap;
	resp_mmap = &session->resp.mmap;

	if (!(vma->vm_flags & VM_SHARED))
		return -EINVAL;
	if (!(vma->vm_flags & (VM_READ | VM_WRITE)))
		return -EINVAL;
	if (READ_ONCE(session->dead))
		return -ENODEV;

	mutex_lock(&vv->vlock);

	if (cookie >= VMEDIA_DBUF_COOKIE_BASE) {
		struct vmedia_dbuf *dbuf = vmedia_dbuf_lookup(session, cookie);

		ret = dbuf ? vmedia_dbuf_mmap(dbuf, vma) : -EINVAL;
		goto end;
	}

	cmd_mmap->hdr.cmd = VIRTIO_MEDIA_CMD_MMAP;
	cmd_mmap->session_id = session->id;
	cmd_mmap->flags =
		(vma->vm_flags & VM_WRITE) ? VIRTIO_MEDIA_MMAP_FLAG_RW : 0;
	cmd_mmap->offset = vma->vm_pgoff << PAGE_SHIFT;

	sg_set_buf(&cmd_sg, cmd_mmap, sizeof(*cmd_mmap));
	sg_mark_end(&cmd_sg);

	sg_set_buf(&resp_sg, resp_mmap, sizeof(*resp_mmap));
	sg_mark_end(&resp_sg);

	/*
	 * The host performs reference counting and is smart enough to return the
	 * same guest physical address if this is called several times on the same
	 * buffer.
	 */
	ret = virtio_media_send_command(vv, sgs, 1, 1, sizeof(*resp_mmap),
					NULL);
	if (ret < 0)
		goto end;

	driver_addr = resp_mmap->driver_addr;
	len = resp_mmap->len;

	if (vma->vm_end - vma->vm_start > PAGE_ALIGN(len)) {
		ret = -EINVAL;
		goto unmap;
	}

	/*
	 * The host offset must stay inside the region we map from; with no
	 * region at all (len 0) this would otherwise map physical page 0.
	 */
	if (driver_addr > vv->mmap_region.len ||
	    PAGE_ALIGN(len) > vv->mmap_region.len - driver_addr) {
		v4l2_err(&vv->v4l2_dev,
			 "host MMAP offset %#llx len %llu exceeds the MMAP region (len %llu)\n",
			 driver_addr, len, vv->mmap_region.len);
		ret = -EINVAL;
		goto unmap;
	}

	map = kzalloc(sizeof(*map), GFP_KERNEL);
	if (!map) {
		ret = -ENOMEM;
		goto unmap;
	}
	map->vv = vv;
	map->driver_addr = driver_addr;
	refcount_set(&map->vmas, 1);

	/*
	 * Guest PFN of the mapping: host offset relative to the MMAP region.
	 * vm_pgoff keeps the cookie user-space passed to mmap() -- that is
	 * what /proc/self/maps and mremap() show, and MUNMAP uses
	 * map->driver_addr, so nothing needs the PFN to be stored there.
	 * remap_pfn_range() only rewrites vm_pgoff for COW mappings, and this
	 * one is VM_SHARED.
	 */
	ret = io_remap_pfn_range(vma, vma->vm_start,
				 (driver_addr + vv->mmap_region.addr) >>
					 PAGE_SHIFT,
				 vma->vm_end - vma->vm_start,
				 vma->vm_page_prot);
	if (ret) {
		kfree(map);
		goto unmap;
	}

	/*
	 * Only now: on a failure after this point the core would call
	 * .close, which would send a second MUNMAP.
	 */
	vma->vm_private_data = map;
	vma->vm_ops = &virtio_media_vm_ops;
	v4l2_device_get(&vv->v4l2_dev);
	goto end;

unmap:
	virtio_media_host_munmap_locked(vv, driver_addr);
end:
	mutex_unlock(&vv->vlock);
	return ret;
}

/**
 * vmedia_find_pool - Look up a DroidVM pool node under /reserved-memory.
 * @prefix: node name before the unit address, e.g. "media_host" for
 *	media_host@<gpa>.
 * @base: on success, the guest-physical base of the pool.
 * @len: on success, the length of the node's reg property.
 *
 * Same walk as virtio_gpu_find_pool_base_named() in droidvm-guest-additions
 * (virtio_gpu/virtgpu_kms.c), except that the length is returned too, so the
 * mmap path can bounds-check the offsets the host hands out (VPU_DESIGN.md
 * 5.1).
 *
 * Returns true when the node exists and carries a usable reg.
 */
static bool vmedia_find_pool(const char *prefix, phys_addr_t *base, u64 *len)
{
	struct device_node *rmem, *child;
	bool found = false;

	rmem = of_find_node_by_path("/reserved-memory");
	if (!rmem)
		return false;
	for_each_child_of_node(rmem, child) {
		struct resource res;

		if (!of_node_name_prefix(child, prefix))
			continue;
		if (of_address_to_resource(child, 0, &res) == 0) {
			*base = res.start;
			*len = resource_size(&res);
			found = true;
			of_node_put(child);
			break;
		}
	}
	of_node_put(rmem);
	return found;
}

/**
 * Decide where host-owned MMAP buffers get mapped from (VPU_DESIGN.md 2.4):
 * the media_host pool if the VMM built one, else virtio shm region 0, else
 * nowhere -- in which case REQBUFS(MMAP) on a host-owned queue fails.
 */
static void virtio_media_setup_host_pool(struct virtio_media *vv)
{
	struct virtio_device *virtio_dev = vv->virtio_dev;
	phys_addr_t base;
	u64 len;

	if (vmedia_find_pool("media_host", &base, &len)) {
		vv->mmap_region.addr = base;
		vv->mmap_region.len = len;
		pr_info("virtio-media: media_host pool base %pa len %llu\n",
			&base, len);
		return;
	}

	if (virtio_get_shm_region(virtio_dev, &vv->mmap_region,
				  VIRTIO_MEDIA_SHM_MMAP) &&
	    vv->mmap_region.len > 0) {
		pr_info("virtio-media: no media_host pool, host MMAP buffers use virtio shm region %d base %#llx len %llu\n",
			VIRTIO_MEDIA_SHM_MMAP, vv->mmap_region.addr,
			vv->mmap_region.len);
		return;
	}

	vv->mmap_region.addr = 0;
	vv->mmap_region.len = 0;
	pr_info("virtio-media: no media_host pool and no virtio shm region, host-owned MMAP buffers are unavailable\n");
}

/**
 * Set up the media_guest pool (VPU_DESIGN.md 5.1): drm_buddy over the range
 * the host SHARE'd, or nothing, in which case driver-owned buffers come from
 * dma_alloc_pages() on the transport's DMA device.
 */
static void virtio_media_setup_guest_pool(struct virtio_media *vv)
{
	phys_addr_t base;
	u64 len, size;
	int ret;

	mutex_init(&vv->guest_pool_lock);

	if (!vmedia_find_pool("media_guest", &base, &len)) {
		/*
		 * A restricted-dma-pool is the fingerprint of a VM whose RAM
		 * is lent rather than shared: dma_alloc_pages() then lands in
		 * that bounce pool, which is the only system memory the host
		 * can read, so say where the buffers will go.
		 */
		struct device_node *rdma = of_find_compatible_node(
			NULL, NULL, "restricted-dma-pool");

		if (rdma) {
			pr_info("virtio-media: no media_guest pool, driver-owned buffers come from dma_alloc_pages (restricted-dma-pool present: they land in the bounce pool)\n");
			of_node_put(rdma);
		} else {
			pr_info("virtio-media: no media_guest pool, driver-owned buffers come from dma_alloc_pages\n");
		}
		return;
	}

	/* drm_buddy wants a chunk-aligned size; trim, the tail is not ours. */
	size = ALIGN_DOWN(len, PAGE_SIZE);
	if (!size) {
		pr_warn("virtio-media: media_guest pool at %pa is smaller than a page (%llu), ignoring it\n",
			&base, len);
		return;
	}

	ret = drm_buddy_init(&vv->guest_pool_mm, size, PAGE_SIZE);
	if (ret) {
		pr_warn("virtio-media: media_guest pool init failed: %d, driver-owned buffers come from dma_alloc_pages\n",
			ret);
		return;
	}

	vv->guest_pool_base = base;
	vv->guest_pool_size = size;
	vv->guest_pool_ready = true;
	pr_info("virtio-media: media_guest pool base %pa size %llu MiB (drm_buddy)\n",
		&base, size >> 20);
}

static void virtio_media_guest_pool_fini(struct virtio_media *vv)
{
	if (!vv->guest_pool_ready)
		return;

	mutex_lock(&vv->guest_pool_lock);
	vv->guest_pool_ready = false;
	if (vv->guest_pool_mm.avail != vv->guest_pool_mm.size)
		pr_warn("virtio-media: media_guest pool still has %llu bytes allocated at teardown\n",
			vv->guest_pool_mm.size - vv->guest_pool_mm.avail);
	drm_buddy_fini(&vv->guest_pool_mm);
	mutex_unlock(&vv->guest_pool_lock);
}

/**
 * Final teardown: runs when the last reference on the v4l2_device drops.
 *
 * That is the driver binding's reference (dropped at the end of
 * virtio_media_remove()), the video device's (dropped by the core once the
 * device is unregistered and the last file handle is closed -- each open fd
 * pins the video device, v4l2-dev.c v4l2_open/v4l2_release), and one per
 * live driver-owned buffer and host mapping (a VMA can outlive its fd).
 * Only here is it safe to tear the pool allocator down and free the device:
 * freeing any of it at remove() time is what oopsed a client that still had
 * a session open across a sysfs unbind (D66, B12-acceptance section 15) --
 * the same reason mainline hotpluggable V4L2 drivers free their state from
 * a release callback, not from disconnect.
 */
static void virtio_media_v4l2_release(struct v4l2_device *v4l2_dev)
{
	struct virtio_media *vv =
		container_of(v4l2_dev, struct virtio_media, v4l2_dev);

	virtio_media_guest_pool_fini(vv);
	put_device(vv->dma_dev);
	kfree(vv->event_buffer);
	kfree(vv);
}

static const struct v4l2_file_operations virtio_media_fops = {
	.owner = THIS_MODULE,
	.open = virtio_media_device_open,
	.release = virtio_media_device_close,
	.poll = virtio_media_device_poll,
	.unlocked_ioctl = virtio_media_device_ioctl,
	.mmap = virtio_media_device_mmap,
};

static int virtio_media_probe(struct virtio_device *virtio_dev)
{
	struct device *dev = &virtio_dev->dev;
	struct virtqueue *vqs[2];
#if LINUX_VERSION_CODE >= KERNEL_VERSION(6, 11, 0)
	static struct virtqueue_info vq_info[2] = {
		{
			.name = "command",
			.callback = commandq_callback,
		},
		{
			.name = "event",
			.callback = eventq_callback,
		},
	};
#else
	static vq_callback_t *vq_callbacks[] = {
		commandq_callback,
		eventq_callback,
	};
	static const char *const vq_names[] = { "command", "event" };
#endif
	struct virtio_media *vv;
	struct video_device *vd;
	int i;
	int ret;

	/*
	 * Not devm: a devm allocation dies when the driver unbinds, while
	 * open file handles and mappings legitimately outlive a sysfs unbind
	 * and keep dereferencing this memory (D66). vv -- and the video_dev
	 * and v4l2_dev embedded in it -- is freed by
	 * virtio_media_v4l2_release() once the last reference is gone.
	 */
	vv = kzalloc(sizeof(*vv), GFP_KERNEL);
	if (!vv)
		return -ENOMEM;

	vv->event_buffer = kzalloc(
		VIRTIO_MEDIA_EVENT_MAX_SIZE * VIRTIO_MEDIA_NUM_EVENT_BUFS,
		GFP_KERNEL);
	if (!vv->event_buffer) {
		kfree(vv);
		return -ENOMEM;
	}

	mutex_init(&vv->bufs_lock);

	INIT_LIST_HEAD(&vv->sessions);
	mutex_init(&vv->sessions_lock);
	mutex_init(&vv->events_process_lock);
	mutex_init(&vv->vlock);

	vv->virtio_dev = virtio_dev;
	virtio_dev->priv = vv;
	vv->dma_dev = virtio_dev->dev.parent ? virtio_dev->dev.parent :
					       &virtio_dev->dev;
	/*
	 * A driver-owned buffer in no-pool mode frees its DMA pages when its
	 * last mapping goes away, which can be after the unbind: pin the DMA
	 * device until the final teardown.
	 */
	get_device(vv->dma_dev);

	init_waitqueue_head(&vv->wq);

	ret = v4l2_device_register(dev, &vv->v4l2_dev);
	if (ret)
		goto err_v4l2_register;
	/*
	 * From here on everything is freed through the v4l2_dev refcount:
	 * the core takes a reference per registered video device and per
	 * open file, so the release only runs once the node is unregistered
	 * AND the last fd is closed -- the disconnected-device lifetime D66
	 * requires. Must be set before video_register_device(), whose
	 * matching put in v4l2_device_release() is conditional on it.
	 */
	vv->v4l2_dev.release = virtio_media_v4l2_release;

#if LINUX_VERSION_CODE >= KERNEL_VERSION(6, 11, 0)
	ret = virtio_find_vqs(virtio_dev, 2, vqs, vq_info, NULL);
#else
	ret = virtio_find_vqs(virtio_dev, 2, vqs, vq_callbacks, vq_names, NULL);
#endif
	if (ret)
		goto err_find_vqs;

	vv->commandq = vqs[0];
	vv->eventq = vqs[1];
	INIT_WORK(&vv->eventq_work, virtio_media_event_work);

	/* Where host-owned MMAP buffers come from, and where guest-owned go. */
	virtio_media_setup_host_pool(vv);
	virtio_media_setup_guest_pool(vv);

	virtio_device_ready(virtio_dev);

	vd = &vv->video_dev;

	vd->v4l2_dev = &vv->v4l2_dev;
	vd->vfl_type = VFL_TYPE_VIDEO;
	vd->ioctl_ops = &virtio_media_ioctl_ops;
	vd->fops = &virtio_media_fops;
	vd->device_caps = virtio_cread32(virtio_dev, 0);
	if (vd->device_caps & (V4L2_CAP_VIDEO_M2M | V4L2_CAP_VIDEO_M2M_MPLANE))
		vd->vfl_dir = VFL_DIR_M2M;
	else if (vd->device_caps &
		 (V4L2_CAP_VIDEO_OUTPUT | V4L2_CAP_VIDEO_OUTPUT_MPLANE))
		vd->vfl_dir = VFL_DIR_TX;
	else
		vd->vfl_dir = VFL_DIR_RX;
	vd->release = video_device_release_empty;
	strscpy(vd->name, "virtio-media", sizeof(vd->name));

	video_set_drvdata(vd, vv);

	/* TODO find out when we should enable this ioctl? */
	v4l2_disable_ioctl(vd, VIDIOC_S_HW_FREQ_SEEK);

	ret = video_register_device(vd, virtio_cread32(virtio_dev, 4), 0);
	if (ret)
		goto err_register;

	for (i = 0; i < VIRTIO_MEDIA_NUM_EVENT_BUFS; i++) {
		ret = virtio_media_send_event_buffer(
			vv, vv->event_buffer + VIRTIO_MEDIA_EVENT_MAX_SIZE * i);
		if (ret) {
			goto send_event_buffer;
		}
	}

	return 0;

send_event_buffer:
	/*
	 * The node existed for an instant: mirror remove() so a file handle
	 * that slipped in cannot reach the dying virtqueues.
	 */
	mutex_lock(&vv->vlock);
	vv->disconnected = true;
	mutex_unlock(&vv->vlock);
	video_unregister_device(&vv->video_dev);
err_register:
	virtio_reset_device(virtio_dev);
	cancel_work_sync(&vv->eventq_work);
	virtio_dev->config->del_vqs(virtio_dev);
err_find_vqs:
	v4l2_device_disconnect(&vv->v4l2_dev);
	/*
	 * Drops the probe reference; virtio_media_v4l2_release() then frees
	 * everything (now, or after a straggling fd from the
	 * send_event_buffer path closes).
	 */
	v4l2_device_put(&vv->v4l2_dev);

	return ret;

err_v4l2_register:
	put_device(vv->dma_dev);
	kfree(vv->event_buffer);
	kfree(vv);

	return ret;
}

/*
 * Driver unbind (sysfs unbind, module unload, hot unplug), following the
 * video_unregister_device + disconnect pattern mainline hotpluggable V4L2
 * drivers use: the device node disappears and every new entry point answers
 * -ENODEV, open file handles keep working against allocated (dead) state,
 * sleepers are woken, and nothing per-device is freed here -- sessions go
 * away with their file's release(), and the device itself with the last
 * v4l2_dev reference (virtio_media_v4l2_release()).
 *
 * The old order -- close every session and tear the pool down right here --
 * is defect D66 (B12-acceptance sections 5 and 15): a client that held
 * buffers across the unbind dereferenced its freed queue array on the next
 * QBUF (level-3 translation fault in vmedia_dbuf_buffer_from_host) and was
 * left as an unreapable zombie only a VM restart could clear.
 */
static void virtio_media_remove(struct virtio_device *virtio_dev)
{
	struct virtio_media *vv = virtio_dev->priv;
	struct virtio_media_session *s;

	/*
	 * Close the command gate first, under vlock: after this no thread
	 * can add to a virtqueue (virtio_media_kick_command() checks the
	 * flag under the same lock), and taking the lock waited out any
	 * command in flight, so the reset below never yanks a live command.
	 */
	mutex_lock(&vv->vlock);
	vv->disconnected = true;
	mutex_unlock(&vv->vlock);

	/*
	 * Give every open session the dead-session treatment the host-error
	 * event path already gets right: ioctls answer -ENODEV, poll answers
	 * EPOLLERR, and the wake-ups release anyone sleeping in DQBUF, in
	 * poll, or in the DQEVENT pre-wait -- no thread is left in D state
	 * over a vanished device (D66 requirement).
	 */
	mutex_lock(&vv->sessions_lock);
	list_for_each_entry(s, &vv->sessions, list) {
		WRITE_ONCE(s->dead, true);
		wake_up(&s->dqbufs_wait);
		wake_up_all(&s->fh.wait);
	}
	mutex_unlock(&vv->sessions_lock);

	/*
	 * Unregister before touching the transport: the core then fails
	 * every new open/ioctl/poll/mmap with -ENODEV or EPOLLERR on its
	 * own (v4l2-dev.c checks video_is_registered on each entry).
	 */
	video_unregister_device(&vv->video_dev);

	/* No interrupts after the reset, so nothing re-schedules the work. */
	virtio_reset_device(virtio_dev);
	cancel_work_sync(&vv->eventq_work);
	virtio_dev->config->del_vqs(virtio_dev);

	/* Sever the parent struct device; vv itself stays until the last put. */
	v4l2_device_disconnect(&vv->v4l2_dev);
	v4l2_device_put(&vv->v4l2_dev);
}

static struct virtio_device_id id_table[] = {
	{ VIRTIO_ID_MEDIA, VIRTIO_DEV_ANY_ID },
	{ 0 },
};

static unsigned int features[] = {};

static struct virtio_driver virtio_media_driver = {
	.feature_table = features,
	.feature_table_size = ARRAY_SIZE(features),
	.driver.name = VIRTIO_MEDIA_DEFAULT_DRIVER_NAME,
	.driver.owner = THIS_MODULE,
	.id_table = id_table,
	.probe = virtio_media_probe,
	.remove = virtio_media_remove,
};

module_virtio_driver(virtio_media_driver);

MODULE_DEVICE_TABLE(virtio, id_table);
MODULE_DESCRIPTION("virtio media driver");
MODULE_AUTHOR("Alexandre Courbot <acourbot@google.com>");
MODULE_LICENSE("Dual BSD/GPL");
