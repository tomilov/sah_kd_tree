#pragma once

#include <vulkan/vulkan.hpp>

#include <limits>
#include <new>

#include <cstddef>

#include <engine/engine_export.h>

namespace engine
{

struct ENGINE_EXPORT AllocationCallbacks
{
    const vk::AllocationCallbacks allocationCallbacks = [this]
    {
        vk::AllocationCallbacks allocationCallbacks;

        allocationCallbacks.pUserData = this;

        allocationCallbacks.pfnAllocation = [](void * pUserData, size_t size, size_t alignment, VkSystemAllocationScope allocationScope) -> void *
        {
            return static_cast<AllocationCallbacks *>(pUserData)->allocation(size, alignment, vk::SystemAllocationScope(allocationScope));
        };
        allocationCallbacks.pfnReallocation = nullptr;
        allocationCallbacks.pfnFree = [](void * pUserData, void * pMemory)
        {
            return static_cast<AllocationCallbacks *>(pUserData)->free(pMemory);
        };
        allocationCallbacks.pfnInternalAllocation = [](void * pUserData, size_t size, VkInternalAllocationType allocationType, VkSystemAllocationScope allocationScope)
        {
            return static_cast<AllocationCallbacks *>(pUserData)->internalAllocation(size, vk::InternalAllocationType(allocationType), vk::SystemAllocationScope(allocationScope));
        };
        allocationCallbacks.pfnInternalFree = [](void * pUserData, size_t size, VkInternalAllocationType allocationType, VkSystemAllocationScope allocationScope)
        {
            return static_cast<AllocationCallbacks *>(pUserData)->internalFreeNotification(size, vk::InternalAllocationType(allocationType), vk::SystemAllocationScope(allocationScope));
        };

        return allocationCallbacks;
    }();

    [[nodiscard]] void * allocation(size_t size, size_t alignment, vk::SystemAllocationScope allocationScope);
    void free(void * pMemory);
    void internalAllocation(size_t size, vk::InternalAllocationType allocationType, vk::SystemAllocationScope allocationScope);
    void internalFreeNotification(size_t size, vk::InternalAllocationType allocationType, vk::SystemAllocationScope allocationScope);
};

template<typename T, vk::SystemAllocationScope systemAllocationScope>
class Allocator
{
public:
    using value_type = T;

    template<typename R>
    struct rebind
    {
        using other = Allocator<R, systemAllocationScope>;
    };

    Allocator(vk::Optional<const vk::AllocationCallbacks> allocationCallbacks) noexcept : allocationCallbacks{allocationCallbacks}
    {}

    template<typename R>
    Allocator(const Allocator<R, systemAllocationScope> & rhs) noexcept : allocationCallbacks{rhs.allocationCallbacks}
    {}

    [[nodiscard]] T * allocate(size_t n) const
    {
        if (n == 0) {
            return nullptr;
        }
        if (std::numeric_limits<size_t>::max() / sizeof(T) < n) {
            throw std::bad_array_new_length{};
        }
        if (!allocationCallbacks) {
            return static_cast<T *>(::operator new(sizeof(T) * n, std::align_val_t(alignof(T))));
        }
        auto p = allocationCallbacks->pfnAllocation(allocationCallbacks->pUserData, sizeof(T) * n, alignof(T), VkSystemAllocationScope(systemAllocationScope));
        if (!p) {
            throw std::bad_alloc{};
        }
        return static_cast<T *>(p);
    }

    void deallocate(T * p, [[maybe_unused]] size_t n) const noexcept
    {
        if (!p) {
            return;
        }
        if (!allocationCallbacks) {
            return ::operator delete(static_cast<void *>(p));
        }
        return allocationCallbacks->pfnFree(allocationCallbacks->pUserData, p);
    }

    template<typename R>
    bool operator==(const Allocator<R, systemAllocationScope> & rhs) noexcept
    {
        return allocationCallbacks == rhs.allocationCallbacks;
    }

    template<typename R>
    bool operator!=(const Allocator<R, systemAllocationScope> & rhs) noexcept
    {
        return !operator==(rhs);
    }

private:
    const vk::Optional<const vk::AllocationCallbacks> allocationCallbacks;
};

}  // namespace engine
