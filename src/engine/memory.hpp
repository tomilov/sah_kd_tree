#pragma once

#include <utils/assert.hpp>
#include <utils/auto_cast.hpp>
#include <utils/noncopyable.hpp>

#include <vulkan/vulkan.hpp>

#include <limits>
#include <new>

#include <cstddef>

#include <engine/engine_export.h>

namespace engine
{

struct ENGINE_EXPORT AllocationCallbacks final : utils::NonCopyable
{
    const vk::AllocationCallbacks allocationCallbacks = [this]
    {
        vk::AllocationCallbacks allocationCallbacksOut;

        allocationCallbacksOut.pUserData = this;

        allocationCallbacksOut.pfnAllocation = [](void * pUserData, size_t size, size_t alignment, vk::SystemAllocationScope allocationScope) -> void *
        {
            return static_cast<AllocationCallbacks *>(pUserData)->allocation(size, alignment, allocationScope);
        };
        allocationCallbacksOut.pfnReallocation = nullptr;
        allocationCallbacksOut.pfnFree = [](void * pUserData, void * pMemory)
        {
            return static_cast<AllocationCallbacks *>(pUserData)->free(pMemory);
        };
        allocationCallbacksOut.pfnInternalAllocation = [](void * pUserData, size_t size, vk::InternalAllocationType allocationType, vk::SystemAllocationScope allocationScope)
        {
            return static_cast<AllocationCallbacks *>(pUserData)->internalAllocation(size, allocationType, allocationScope);
        };
        allocationCallbacksOut.pfnInternalFree = [](void * pUserData, size_t size, vk::InternalAllocationType allocationType, vk::SystemAllocationScope allocationScope)
        {
            return static_cast<AllocationCallbacks *>(pUserData)->internalFreeNotification(size, allocationType, allocationScope);
        };

        return allocationCallbacksOut;
    }();

    [[nodiscard]] static void * allocation(size_t size, size_t alignment, vk::SystemAllocationScope allocationScope);
    static void free(void * pMemory);
    static void internalAllocation(size_t size, vk::InternalAllocationType allocationType, vk::SystemAllocationScope allocationScope);
    static void internalFreeNotification(size_t size, vk::InternalAllocationType allocationType, vk::SystemAllocationScope allocationScope);
};

template<typename T, vk::SystemAllocationScope systemAllocationScope>
class Allocator
{
public:
    using value_type = T;

    using propagate_on_container_copy_assignment = std::true_type;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_swap = std::true_type;

    template<typename R>
    struct rebind
    {
        using other = Allocator<R, systemAllocationScope>;
    };

    explicit Allocator(vk::Optional<const vk::AllocationCallbacks> allocationCallbacksIn) noexcept
        : allocationCallbacks{allocationCallbacksIn}
    {}

    template<typename R>
    explicit Allocator(const Allocator<R, systemAllocationScope> & rhs) noexcept
        : allocationCallbacks{rhs.allocationCallbacks}
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
            return static_cast<T *>(::operator new(sizeof(T) * n, std::align_val_t{alignof(T)}));
        }
        auto p = allocationCallbacks->pfnAllocation(allocationCallbacks->pUserData, sizeof(T) * n, alignof(T), systemAllocationScope);
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

    const Allocator & select_on_container_copy_construction() const &
    {
        return *this;
    }

private:
    vk::Optional<const vk::AllocationCallbacks> allocationCallbacks;
};

}  // namespace engine
