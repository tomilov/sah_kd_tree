#include <engine/format.hpp>
#include <engine/memory.hpp>

#include <fmt/format.h>
#include <spdlog/spdlog.h>

#include <cstddef>

namespace engine
{

void * AllocationCallbacks::allocation(size_t size, size_t alignment, vk::SystemAllocationScope allocationScope)
{
    auto pMemory = ::operator new (size, std::align_val_t{alignment});
    if ((false)) {
        SPDLOG_DEBUG("Allocation mem size {}, alignment {}, system allocation scope: {}, address {}", size, alignment, allocationScope, fmt::ptr(pMemory));
    }
    return pMemory;
}

void AllocationCallbacks::free(void * pMemory)
{
    if ((false)) {
        SPDLOG_DEBUG("Free mem  address {}", fmt::ptr(pMemory));
    }
    return ::operator delete(static_cast<void *>(pMemory));
}

void AllocationCallbacks::internalAllocation(size_t size, vk::InternalAllocationType allocationType, vk::SystemAllocationScope allocationScope)
{
    SPDLOG_DEBUG("Internal allocation mem size {}, allocation type: {}, system allocation scope: {}", size, allocationType, allocationScope);
}

void AllocationCallbacks::internalFreeNotification(size_t size, vk::InternalAllocationType allocationType, vk::SystemAllocationScope allocationScope)
{
    SPDLOG_DEBUG("Internal free mem size {}, allocation type: {}, system allocation scope: {}", size, allocationType, allocationScope);
}

}  // namespace engine
