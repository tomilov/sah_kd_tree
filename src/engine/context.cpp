#include <engine/context.hpp>
#include <engine/device.hpp>
#include <engine/instance.hpp>
#include <engine/library.hpp>
#include <engine/physical_device.hpp>
#include <engine/vma.hpp>

#include <spdlog/spdlog.h>

#include <initializer_list>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <string_view>
#include <unordered_set>
#include <vector>

#include <cstdint>

namespace engine
{

struct Context::DebugUtilsMessageMuteGuard::Impl
{
    enum class Action
    {
        kMute,
        kUnmute,
    };

    std::mutex & mutex;
    std::unordered_multiset<uint32_t> & mutedMessageIdNumbers;
    const Action action;
    const std::vector<uint32_t> messageIdNumbers;

    Impl(std::mutex & mutex, std::unordered_multiset<uint32_t> & mutedMessageIdNumbers, Action action, std::initializer_list<uint32_t> messageIdNumbers);
    ~Impl();

    void mute();
    void unmute();
};

Context::DebugUtilsMessageMuteGuard::~DebugUtilsMessageMuteGuard() = default;

template<typename... Args>
Context::DebugUtilsMessageMuteGuard::DebugUtilsMessageMuteGuard(Args &&... args) : impl_{std::forward<Args>(args)...}
{}

Context::DebugUtilsMessageMuteGuard::Impl::~Impl()
{
    switch (action) {
    case Action::kMute: {
        unmute();
        break;
    }
    case Action::kUnmute: {
        mute();
        break;
    }
    }
}

Context::DebugUtilsMessageMuteGuard::Impl::Impl(std::mutex & mutex, std::unordered_multiset<uint32_t> & mutedMessageIdNumbers, Action action, std::initializer_list<uint32_t> messageIdNumbers)
    : mutex{mutex}, mutedMessageIdNumbers{mutedMessageIdNumbers}, action{action}, messageIdNumbers{messageIdNumbers}
{
    switch (action) {
    case Action::kMute: {
        mute();
        break;
    }
    case Action::kUnmute: {
        unmute();
        break;
    }
    }
}

void Context::DebugUtilsMessageMuteGuard::Impl::mute()
{
    if (std::empty(messageIdNumbers)) {
        return;
    }
    std::lock_guard<std::mutex> lock{mutex};
    mutedMessageIdNumbers.insert(std::cbegin(messageIdNumbers), std::cend(messageIdNumbers));
}

void Context::DebugUtilsMessageMuteGuard::Impl::unmute()
{
    if (std::empty(messageIdNumbers)) {
        return;
    }
    std::lock_guard<std::mutex> lock{mutex};
    for (auto messageIdNumber : messageIdNumbers) {
        auto unmutedMessageIdNumber = mutedMessageIdNumbers.find(messageIdNumber);
        INVARIANT(unmutedMessageIdNumber != std::end(mutedMessageIdNumbers), "messageId {:#x} of muted message is not found", messageIdNumber);
        mutedMessageIdNumbers.erase(unmutedMessageIdNumber);
    }
}

Context::Context(std::initializer_list<uint32_t> mutedMessageIdNumbers, bool mute) : debugUtilsMessageMuteGuard{muteDebugUtilsMessages(mutedMessageIdNumbers, mute)}
{}

Context::~Context() = default;

auto Context::muteDebugUtilsMessages(std::initializer_list<uint32_t> messageIdNumbers, bool enabled) const -> DebugUtilsMessageMuteGuard
{
    return {mutex, mutedMessageIdNumbers, DebugUtilsMessageMuteGuard::Impl::Action::kMute, enabled ? messageIdNumbers : decltype(messageIdNumbers){}};
}

auto Context::unmuteDebugUtilsMessages(std::initializer_list<uint32_t> messageIdNumbers, bool enabled) const -> DebugUtilsMessageMuteGuard
{
    return {mutex, mutedMessageIdNumbers, DebugUtilsMessageMuteGuard::Impl::Action::kUnmute, enabled ? messageIdNumbers : decltype(messageIdNumbers){}};
}

bool Context::shouldMuteDebugUtilsMessage(uint32_t messageIdNumber) const
{
    std::lock_guard<std::mutex> lock{mutex};
    return mutedMessageIdNumbers.contains(messageIdNumber);
}

void Context::createInstance(std::string_view applicationName, uint32_t applicationVersion, std::optional<std::string_view> libraryName, vk::Optional<const vk::AllocationCallbacks> allocationCallbacks)
{
    library = std::make_unique<Library>(libraryName, allocationCallbacks, *this);
    instance = std::make_unique<Instance>(applicationName, applicationVersion, *this, *library);
    physicalDevices = std::make_unique<PhysicalDevices>(*this);
}

vk::Instance Context::getVulkanInstance() const
{
    return instance->instance;
}

void Context::createDevice(vk::SurfaceKHR surface)
{
    auto & physicalDevice = physicalDevices->pickPhisicalDevice(surface);
    device = std::make_unique<Device>(physicalDevice.getDeviceName(), *this, *library, physicalDevice);
    vma = std::make_unique<MemoryAllocator>(*this);
}

vk::PhysicalDevice Context::getVulkanPhysicalDevice() const
{
    return device->physicalDevice.physicalDevice;
}

vk::Device Context::getVulkanDevice() const
{
    return device->device;
}

uint32_t Context::getVulkanGraphicsQueueFamilyIndex() const
{
    return device->physicalDevice.externalGraphicsQueueCreateInfo.familyIndex;
}

uint32_t Context::getVulkanGraphicsQueueIndex() const
{
    return device->physicalDevice.externalGraphicsQueueCreateInfo.index;
}

const Library & Context::getLibrary() const
{
    return *library;
}

const Instance & Context::getInstance() const
{
    return *instance;
}

const PhysicalDevices & Context::getPhysicalDevices() const
{
    return *physicalDevices;
}

const Device & Context::getDevice() const
{
    return *device;
}

const MemoryAllocator & Context::getMemoryAllocator() const
{
    return *vma;
}

}  // namespace engine
