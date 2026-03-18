#pragma once
#include <atomic>
#include <cstring>
#include <vector>

template <typename T>
class RingBuffer
{
public:
    explicit RingBuffer(size_t capacity)
        : capacity_(capacity), buffer_(capacity)
    {
    }

    bool push(const T* data, size_t count)
    {
        const size_t writePos = writePos_.load(std::memory_order_relaxed);
        const size_t readPos = readPos_.load(std::memory_order_acquire);

        size_t available = capacity_ - (writePos - readPos);
        if (count > available)
            return false;

        for (size_t i = 0; i < count; ++i)
            buffer_[(writePos + i) % capacity_] = data[i];

        writePos_.store(writePos + count, std::memory_order_release);
        return true;
    }

    bool pop(T* data, size_t count)
    {
        const size_t readPos = readPos_.load(std::memory_order_relaxed);
        const size_t writePos = writePos_.load(std::memory_order_acquire);

        size_t available = writePos - readPos;
        if (count > available)
            return false;

        for (size_t i = 0; i < count; ++i)
            data[i] = buffer_[(readPos + i) % capacity_];

        readPos_.store(readPos + count, std::memory_order_release);
        return true;
    }

    size_t availableToRead() const
    {
        return writePos_.load(std::memory_order_acquire)
             - readPos_.load(std::memory_order_acquire);
    }

    void reset()
    {
        readPos_.store(0, std::memory_order_relaxed);
        writePos_.store(0, std::memory_order_relaxed);
    }

private:
    size_t capacity_;
    std::vector<T> buffer_;
    std::atomic<size_t> readPos_{0};
    std::atomic<size_t> writePos_{0};
};
