#pragma once

#include <dw/core/context/Context.h>
#include <dw/dnn/tensor/Tensor.h>
#include <dw/image/Image.h>

#include <memory>
#include <queue>
#include <mutex>
#include <atomic>

namespace depth_pipeline {

/**
 * @brief Memory pool manager for DNN tensors and images
 * 
 * Responsibilities:
 * - Pre-allocate tensor pools to avoid runtime allocations
 * - Manage tensor lifecycle (acquire/release pattern)
 * - Track memory usage statistics
 */
class MemoryManager {
public:
    struct PoolConfig {
        dwDNNTensorProperties tensorProps;
        uint32_t poolSize;
        const char* poolName;
    };
    
    struct MemoryStatistics {
        uint64_t totalAllocatedBytes;
        uint64_t currentUsedBytes;
        uint32_t totalTensors;
        uint32_t availableTensors;
        uint32_t acquisitionFailures;
    };

    /**
     * @brief Constructor
     * @param context DriveWorks context handle
     */
    explicit MemoryManager(dwContextHandle_t context);
    
    /**
     * @brief Destructor - releases all pools
     */
    ~MemoryManager();
    
    /**
     * @brief Create tensor pool
     * @param poolId Output pool identifier
     * @param config Pool configuration
     * @return DW_SUCCESS on success
     */
    dwStatus createTensorPool(uint32_t& poolId, const PoolConfig& config);
    
    /**
     * @brief Acquire tensor from pool
     * @param tensor Output tensor handle
     * @param poolId Pool identifier
     * @return DW_SUCCESS if available, DW_BUFFER_FULL if pool exhausted
     */
    dwStatus acquireTensor(dwDNNTensorHandle_t& tensor, uint32_t poolId);
    
    /**
     * @brief Release tensor back to pool
     * @param tensor Tensor to release
     * @param poolId Pool identifier
     * @return DW_SUCCESS on success
     */
    dwStatus releaseTensor(dwDNNTensorHandle_t tensor, uint32_t poolId);
    
    /**
     * @brief Get memory statistics for pool
     */
    MemoryStatistics getStatistics(uint32_t poolId) const;
    
    /**
     * @brief Get total memory usage across all pools
     */
    uint64_t getTotalMemoryUsage() const;

private:
    struct TensorPool {
        std::queue<dwDNNTensorHandle_t> available;
        dwDNNTensorProperties properties;
        uint32_t totalSize;
        std::atomic<uint32_t> inUseCount;
        std::atomic<uint32_t> acquisitionFailures;
        std::string name;
        std::mutex mutex;
    };
    
    dwContextHandle_t m_context;
    std::vector<std::unique_ptr<TensorPool>> m_pools;
    std::atomic<uint64_t> m_totalAllocatedBytes;
    
    // Non-copyable
    MemoryManager(const MemoryManager&) = delete;
    MemoryManager& operator=(const MemoryManager&) = delete;
};

} // namespace depth_pipeline