#include <renderer/context.hpp>
#include <renderer/renderer.hpp>

#include <common/version.hpp>

#include <vulkan/vulkan.hpp>

#include <new>

#include <cstddef>

namespace
{

struct AllocationCallbacks
{
    vk::AllocationCallbacks allocationCallbacks;

    AllocationCallbacks()
    {
        allocationCallbacks.pUserData = this;
        allocationCallbacks.pfnAllocation = [](void * pUserData, size_t size, size_t alignment, VkSystemAllocationScope allocationScope) -> void * {
            return static_cast<AllocationCallbacks *>(pUserData)->allocation(size, alignment, vk::SystemAllocationScope(allocationScope));
        };
        allocationCallbacks.pfnReallocation = nullptr;
        allocationCallbacks.pfnFree = [](void * pUserData, void * pMemory) { return static_cast<AllocationCallbacks *>(pUserData)->free(pMemory); };
        allocationCallbacks.pfnInternalAllocation = [](void * pUserData, size_t size, VkInternalAllocationType allocationType, VkSystemAllocationScope allocationScope) {
            return static_cast<AllocationCallbacks *>(pUserData)->internalAllocation(size, vk::InternalAllocationType(allocationType), vk::SystemAllocationScope(allocationScope));
        };
        allocationCallbacks.pfnInternalFree = [](void * pUserData, size_t size, VkInternalAllocationType allocationType, VkSystemAllocationScope allocationScope) {
            return static_cast<AllocationCallbacks *>(pUserData)->internalFreeNotification(size, vk::InternalAllocationType(allocationType), vk::SystemAllocationScope(allocationScope));
        };
    }

    void * allocation(size_t size, size_t alignment, [[maybe_unused]] vk::SystemAllocationScope allocationScope)
    {
        return new (std::align_val_t(alignment)) std::byte[size];
    }

    void free(void * pMemory)
    {
        return delete[] static_cast<std::byte *>(pMemory);
    }

    void internalAllocation([[maybe_unused]] size_t size, [[maybe_unused]] vk::InternalAllocationType allocationType, [[maybe_unused]] vk::SystemAllocationScope allocationScope)
    {}

    void internalFreeNotification([[maybe_unused]] size_t size, [[maybe_unused]] vk::InternalAllocationType allocationType, [[maybe_unused]] vk::SystemAllocationScope allocationScope)
    {}
};

template<typename T>
class CppAllocator
{
public:
    using value_type = T;

    CppAllocator(vk::SystemAllocationScope systemAllocationScope, vk::Optional<const vk::AllocationCallbacks> allocationCallbacks) : systemAllocationScope{systemAllocationScope}, allocationCallbacks{allocationCallbacks}
    {}

    template<typename R>
    constexpr CppAllocator(const CppAllocator<R> & rhs) noexcept : systemAllocationScope{rhs.systemAllocationScope}, allocationCallbacks{rhs.allocationCallbacks}
    {}

    [[nodiscard]] T * allocate(std::size_t n) const
    {
        if (n == 0) {
            return nullptr;
        }
        if (std::numeric_limits<std::size_t>::max() / sizeof(T) < n) {
            throw std::bad_array_new_length{};
        }
        if (!allocationCallbacks) {
            auto p = new std::byte[sizeof(T) * n];
            return reinterpret_cast<T *>(p);
        }
        auto p = allocationCallbacks->pfnAllocation(allocationCallbacks->pUserData, sizeof(T) * n, alignof(T), VkSystemAllocationScope(systemAllocationScope));
        if (!p) {
            throw std::bad_alloc{};
        }
        return static_cast<T *>(p);
    }

    void deallocate(T * p, [[maybe_unused]] std::size_t n) const
    {
        if (!p) {
            return;
        }
        if (!allocationCallbacks) {
            return delete[] reinterpret_cast<std::byte *>(p);
        }
        return allocationCallbacks->pfnFree(allocationCallbacks->pUserData, p);
    }

    template<typename L, typename R>
    friend constexpr bool operator==(const CppAllocator<L> & lhs, const CppAllocator<R> & rhs) noexcept
    {
        return std::tie(lhs.systemAllocationScope, lhs.allocationCallbacks) == std::tie(rhs.systemAllocationScope, rhs.allocationCallbacks);
    }

    template<typename L, typename R>
    friend constexpr bool operator!=(const CppAllocator<L> & lhs, const CppAllocator<R> & rhs) noexcept
    {
        return !(lhs == rhs);
    }

private:
    vk::SystemAllocationScope systemAllocationScope;
    vk::Optional<const vk::AllocationCallbacks> allocationCallbacks;
};

}  // namespace

int main(int /*argc*/, char * /*argv*/[])
{
    renderer::Renderer renderer;

    renderer::Context context;
    constexpr auto kApplicationVersion = VK_MAKE_VERSION(sah_kd_tree::kProjectVersionMajor, sah_kd_tree::kProjectVersionMinor, sah_kd_tree::kProjectVersionPatch);
    AllocationCallbacks allocationCallbacks;
    std::vector<int, CppAllocator<int>> v{CppAllocator<int>{vk::SystemAllocationScope::eCommand, allocationCallbacks.allocationCallbacks}};
    v.resize(10000);
    context.createInstance(APPLICATION_NAME, kApplicationVersion, allocationCallbacks.allocationCallbacks);
    context.createDevice();
}
