import numpy as np
import os
import pickle


class RingBuffer:
    def __init__(self, capacity, element_dim, dtype=np.float32):
        self._use_disk = False
        self._ringbuffer = np.zeros(np.hstack([capacity, element_dim]), dtype=dtype)
        self._head = 0
        self.size = 0
        self._elements_inserted = 0

    def append(self, elem):
        self._ringbuffer[self._head, :] = elem
        self._head = (self._head + 1) % self._ringbuffer.shape[0]
        self.size = min(self.size + 1, self._ringbuffer.shape[0])
        self._elements_inserted += 1

    def extend(self, elements):
        for element in elements:
            self.append(element)

    def sample(self, num_samples):
        assert(num_samples <= self.size)
        indices = np.random.choice(self.size, num_samples, replace=False)
        if self._use_disk:
            indices = sorted(indices.tolist())
        return self._ringbuffer[indices, :]

    def to_array(self):
        return self._ringbuffer[:self.size]

    def save(self, dir_path, name):
        if self._use_disk:
            return

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        with open(dir_path + '/ringbuffers_meta_' + name, 'wb') as f:
            pickle.dump([self.size, self._head, self._use_disk, self._elements_inserted, self._ringbuffer.dtype,
                         self._ringbuffer.shape], f)
        with open(dir_path + '/ringbuffers_data_' + name, 'wb') as f:
            self._ringbuffer.tofile(f)

    @classmethod
    def load(cls, dir_path, name):
        ring_buffer = cls.__new__(cls)
        with open(dir_path + '/ringbuffers_meta_' + name, 'rb') as f:
            ring_buffer.size, ring_buffer._head, ring_buffer._use_disk, ring_buffer._elements_inserted, dtype, shape = \
                pickle.load(f)
        with open(dir_path + '/ringbuffers_data_' + name, 'rb') as f:
            ring_buffer._ringbuffer = np.fromfile(f, dtype=dtype).reshape(shape)

        return ring_buffer

    def __getitem__(self, item):
        if type(item[0]) is slice:
            start, end, step = item[0].indices(self._elements_inserted)
            wrapped_start = start % self.size
            wrapped_end = ((end - 1) % self.size) + 1
            if end > start and wrapped_start >= wrapped_end:
                return np.vstack([self._ringbuffer[wrapped_start::step, item[1]],
                                  self._ringbuffer[:wrapped_end:step, item[1]]])
            else:
                return self._ringbuffer[wrapped_start:wrapped_end:step, item[1]]
        else:
            return self._ringbuffer[item[0] % self.size, item[1]]

    def __repr__(self):
        return self._ringbuffer.__repr__()


class RingBufferCollection:
    def __init__(self, capacity, element_dims, dtypes=None):
        if dtypes is None:
            dtypes = [np.float32] * len(element_dims)
        self._buffers = [RingBuffer(capacity, element_dim, dtype=dtype)
                         for dtype, element_dim in zip(dtypes, element_dims)]
        self.size = 0
        self._capacity = capacity

    def append(self, *elements):
        for element, buffer in zip(elements, self._buffers):
            buffer.append(element)
        self.size = min(self.size + 1, self._capacity)

    def sample(self, num_samples):
        indices = np.random.choice(self.size, num_samples, replace=False)
        return (buffer[indices, :] for buffer in self._buffers)

    def save(self, dir_path, name):
        with open(dir_path + "/ringbuffercollection_meta_" + name, 'wb') as f:
            pickle.dump([len(self._buffers), self._capacity], f)

        for i, buffer in enumerate(self._buffers):
            buffer.save(dir_path, '%s_%d' % (name, i))

    @classmethod
    def load(cls, dir_path, name):
        ring_buffer_collection = cls.__new__(cls)

        with open(dir_path + "/ringbuffercollection_meta_" + name, 'rb') as f:
            num_buffers, ring_buffer_collection._capacity = pickle.load(f)

        ring_buffer_collection._buffers = [RingBuffer.load(dir_path, '%s_%d' % (name, i)) for i in range(num_buffers)]
        ring_buffer_collection.size = ring_buffer_collection._buffers[0].size
        return ring_buffer_collection

    def __repr__(self):
        return '\n'.join([buffer.__repr__() for buffer in self._buffers])