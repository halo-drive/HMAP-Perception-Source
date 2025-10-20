/////////////////////////////////////////////////////////////////////////////////////////
// Livox Mid-360 Plugin for NVIDIA DriveWorks
//
// Copyright (c) 2025 - All rights reserved.
/////////////////////////////////////////////////////////////////////////////////////////

#ifndef BUFFER_POOL_HPP_
#define BUFFER_POOL_HPP_

#include <queue>
#include <mutex>
#include <chrono>
#include <condition_variable>
#include <iostream>

namespace dw
{
namespace plugins
{
namespace common
{

/**
 * @brief Template class providing memory pooling mechanism to avoid re-allocation.
 *
 * This class is used to create/hold a pool of fixed sized elements.
 * Applications can request a pointer to an element (with get()),
 * and release the element back to the pool (with put()).
 *
 * Elements are of fixed size, thus avoid re-allocations when requests arrive.
 * The element type T needs to be provided as template parameter.
 */
template <typename T>
class BufferPool
{
private:
    T* m_pool;
    std::queue<T*> m_freeList;
    std::mutex m_mutex;
    std::condition_variable m_condition;
    bool m_valid;
    size_t m_count;

public:
    /**
     * @brief Create memory pool of n elements.
     *
     * @param[in] count Number of elements to create.
     */
    explicit BufferPool(size_t count)
        : m_valid(true)
        , m_count(count)
    {
        m_pool = new T[count];
        for (size_t i = 0; i < count; i++)
        {
            m_freeList.push(&m_pool[i]);
        }
    }

    ~BufferPool()
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_valid = false;
        m_condition.notify_all();
        lock.unlock();

        if (m_pool)
        {
            delete[] m_pool;
            m_pool = nullptr;
        }
    }

    /**
     * @brief Get element from pool.
     *
     * This function will block until an element is available in pool.
     *
     * @param[out] element Pointer to the element.
     * @return true Element is valid and can be used.
     * @return false Pool has been invalidated, do not use the element.
     */
    bool get(T*& element)
    {
        return get(element, 0);
    }

    /**
     * @brief Get element from pool with timeout.
     *
     * This function will block until an element is available in pool or timeout reached.
     *
     * @param[out] element Pointer to the element.
     * @param[in] timeout_us Timeout in microseconds (0 = wait forever).
     * @return true Element is valid and can be used.
     * @return false Pool has been invalidated or timeout reached, do not use the element.
     */
    bool get(T*& element, uint64_t timeout_us)
    {
        std::unique_lock<std::mutex> lock(m_mutex);

        if (!m_valid)
        {
            return false;
        }

        if (m_freeList.empty())
        {
            if (timeout_us == 0)
            {
                m_condition.wait(lock, [this]() { return !m_freeList.empty() || !m_valid; });
            }
            else
            {
                auto waitDuration = std::chrono::microseconds(timeout_us);
                auto status = m_condition.wait_for(lock, waitDuration, [this]() { return !m_freeList.empty() || !m_valid; });
                if (!status && m_freeList.empty())
                {
                    // Timeout reached and still no elements available
                    return false;
                }
            }

            if (!m_valid)
            {
                return false;
            }
        }

        element = m_freeList.front();
        m_freeList.pop();
        return true;
    }

    /**
     * @brief Return element to pool.
     *
     * @param[in] element Element to return.
     * @return true Element was successfully returned to the pool.
     * @return false Element was not returned because it is invalid or not from the pool.
     */
    bool put(T* element)
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        if (!m_valid)
        {
            return false;
        }

        if (!element)
        {
            return false;
        }

        // Check if element is part of the pool
        if (element < m_pool || element >= m_pool + m_count)
        {
            return false;
        }

        m_freeList.push(element);
        m_condition.notify_one();
        return true;
    }

    /**
     * @brief Get number of free elements in pool.
     *
     * @return size_t Number of free elements.
     */
    size_t getFreeCount() const
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        return m_freeList.size();
    }

    /**
     * @brief Get total number of elements managed by pool.
     *
     * @return size_t Total number of elements.
     */
    size_t getTotalCount() const
    {
        return m_count;
    }
};

} // namespace common
} // namespace plugins
} // namespace dw

#endif // BUFFER_POOL_HPP_