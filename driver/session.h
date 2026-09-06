// SPDX-License-Identifier: BSD-3-Clause OR GPL-2.0+

/*
 * Definitions of virtio-media session related structures.
 *
 * Copyright (c) 2023-2024 Google LLC.
 */

#ifndef __VIRTIO_MEDIA_SESSION_H
#define __VIRTIO_MEDIA_SESSION_H

#include <linux/scatterlist.h>
#include <media/v4l2-fh.h>

#include "protocol.h"

#define VIRTIO_MEDIA_LAST_QUEUE (V4L2_BUF_TYPE_META_OUTPUT)

/**
 * Size of our virtio shadow buffer. 64K: a fragmented 4K frame sent as a
 * USERPTR SG list needs about 3100 16-byte entries (VPU_DESIGN.md 5.2).
 */
#define VIRTIO_SHADOW_BUF_SIZE 0x10000

struct virtio_media_sg_entry {
	u64 start;
	u32 len;
	u32 __padding;
};

struct vmedia_dbuf;

/**
 * struct virtio_media_buffer - Current state of a given buffer.
 *
 * @buffer: struct v4l2_buffer with current information about the buffer.
 * @planes: backing planes array for @buffer.
 * @list: link into the list of buffers pending dequeue.
 * @dbuf: driver-owned backing of each plane (VPU_DESIGN.md 5.2), NULL for
 *	host-owned MMAP buffers and real USERPTR buffers.
 */
struct virtio_media_buffer {
	struct v4l2_buffer buffer;
	struct v4l2_plane planes[VIDEO_MAX_PLANES];
	struct list_head list;
	struct vmedia_dbuf *dbuf[VIDEO_MAX_PLANES];
};

/**
 * struct virtio_media_queue_state - Represents the state of a V4L2 queue.
 *
 * @streaming: Whether the queue is currently streaming.
 * @allocated_bufs: How many buffers are currently allocated.
 * @is_capture_last: set to true when the last buffer has been received on a
 * 	capture queue, so we can return -EPIPE on subsequent DQBUF requests.
 * @buffers: Buffer state array of size @allocated_bufs.
 * @queued_bufs: How many buffers are currently queued at the host.
 * @pending_dqbufs: Buffers that are available for being dequeued.
 * @driver_owned: the buffers user-space requested as MMAP are allocated by
 *	the driver and presented to the host as USERPTR (VPU_DESIGN.md 2.1).
 * @memory: the V4L2_MEMORY_* type user-space set the queue up with, valid
 *	while @allocated_bufs is non-zero. Buffer ioctls must name it; it is
 *	not always what the host sees, which is USERPTR whenever
 *	@driver_owned.
 */
struct virtio_media_queue_state {
	bool streaming;
	size_t allocated_bufs;
	bool is_capture_last;

	struct virtio_media_buffer *buffers;
	size_t queued_bufs;
	struct list_head pending_dqbufs;
	bool driver_owned;
	u32 memory;
};

/**
 * struct virtio_media_session - A session on a virtio_media device, created whenever the device is opened.
 *
 * @fh: file handler for the session.
 * @file: the struct file @fh is attached to (v4l2_fh_add/del need it since 6.17).
 * @id: session ID used to communicate with the device.
 * @nonblocking_dequeue: whether dequeue should block or not (nonblocking if file opened with O_NONBLOCK).
 * @uses_mplane: whether the queues for this session use the MPLANE API or not.
 * @cmd: union of session-related commands. Each session can have one command currently running.
 * @resp: union of session-related responses.
 * @shadow_buf: shadow buffer where commandq data can be staged before being sent to the device.
 * @command_sg: SG table gathering descriptors for a given command and its response.
 * @queues: state of all the queues for this session.
 * @dqbufs_lock: protects the queue state that is read outside vv->vlock:
 *	the @buffers array and the @allocated_bufs it is sized by, the buffer
 *	flags the event work tests, @queued_bufs and @pending_dqbufs. The
 *	ioctl path (REQBUFS/CREATE_BUFS/QBUF/DQBUF/STREAMOFF) writes them
 *	under vlock + dqbufs_lock; the event work and poll read them under
 *	dqbufs_lock alone (D46, B8 §10). Lock order: vlock or
 *	events_process_lock -> dqbufs_lock -> guest_pool_lock, never the
 *	reverse.
 * @dqbufs_wait: waitqueue for dequeued buffers, if VIDIOC_DQBUF needs to block or when polling.
 * @dead: the host reported an error event for this session; it is unusable.
 * @next_dbuf_cookie: mmap offset handed to the next driver-owned buffer.
 * @list: link into the list of sessions for the device.
 */
struct virtio_media_session {
	struct v4l2_fh fh;
	struct file *file;
	u32 id;
	bool nonblocking_dequeue;
	bool uses_mplane;

	union {
		struct virtio_media_cmd_close close;
		struct virtio_media_cmd_ioctl ioctl;
		struct virtio_media_cmd_mmap mmap;
	} cmd;

	union {
		struct virtio_media_resp_ioctl ioctl;
		struct virtio_media_resp_mmap mmap;
	} resp;

	void *shadow_buf;

	struct sg_table command_sgs;

	struct virtio_media_queue_state queues[VIRTIO_MEDIA_LAST_QUEUE + 1];
	struct mutex dqbufs_lock;
	wait_queue_head_t dqbufs_wait;

	/*
	 * Set once the device sent VIRTIO_MEDIA_EVT_ERROR for this session. The
	 * protocol says the session is then corrupted and closed on the host,
	 * so every further ioctl fails with -ENODEV and poll reports EPOLLERR
	 * until user-space closes the file (VPU_DESIGN.md 5.4). Also set for
	 * every session when the device itself is unbound (D66): the same
	 * dead-session paths give a disconnected device's clients their clean
	 * -ENODEV.
	 */
	bool dead;

	u64 next_dbuf_cookie;

	struct list_head list;
};

static inline struct virtio_media_session *fh_to_session(struct v4l2_fh *fh)
{
	return container_of(fh, struct virtio_media_session, fh);
}

#endif // __VIRTIO_MEDIA_SESSION_H
