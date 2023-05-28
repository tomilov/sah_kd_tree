#include <engine/device.hpp>
#include <engine/engine.hpp>
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

struct Engine::DebugUtilsMessageMuteGuard::Impl
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
    ~Impl() noexcept(false);

    void mute() noexcept(false);
    void unmute() noexcept(false);
};

Engine::DebugUtilsMessageMuteGuard::~DebugUtilsMessageMuteGuard() noexcept(false) = default;

template<typename... Args>
Engine::DebugUtilsMessageMuteGuard::DebugUtilsMessageMuteGuard(Args &&... args) : impl_{std::forward<Args>(args)...}
{}

Engine::DebugUtilsMessageMuteGuard::Impl::~Impl() noexcept(false)
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

Engine::DebugUtilsMessageMuteGuard::Impl::Impl(std::mutex & mutex, std::unordered_multiset<uint32_t> & mutedMessageIdNumbers, Action action, std::initializer_list<uint32_t> messageIdNumbers)
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

void Engine::DebugUtilsMessageMuteGuard::Impl::mute() noexcept(false)
{
    if (std::empty(messageIdNumbers)) {
        return;
    }
    std::lock_guard<std::mutex> lock{mutex};
    mutedMessageIdNumbers.insert(std::cbegin(messageIdNumbers), std::cend(messageIdNumbers));
}

void Engine::DebugUtilsMessageMuteGuard::Impl::unmute() noexcept(false)
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

Engine::Engine(std::initializer_list<uint32_t> mutedMessageIdNumbers, bool mute) : debugUtilsMessageMuteGuard{muteDebugUtilsMessages(mutedMessageIdNumbers, mute)}
{}

Engine::~Engine() = default;

auto Engine::muteDebugUtilsMessages(std::initializer_list<uint32_t> messageIdNumbers, bool enabled) const -> DebugUtilsMessageMuteGuard
{
    if (!enabled) {
        return {mutex, mutedMessageIdNumbers, DebugUtilsMessageMuteGuard::Impl::Action::kMute, decltype(messageIdNumbers){}};
    }
    return {mutex, mutedMessageIdNumbers, DebugUtilsMessageMuteGuard::Impl::Action::kMute, std::move(messageIdNumbers)};
}

auto Engine::unmuteDebugUtilsMessages(std::initializer_list<uint32_t> messageIdNumbers, bool enabled) const -> DebugUtilsMessageMuteGuard
{
    if (!enabled) {
        return {mutex, mutedMessageIdNumbers, DebugUtilsMessageMuteGuard::Impl::Action::kUnmute, decltype(messageIdNumbers){}};
    }
    return {mutex, mutedMessageIdNumbers, DebugUtilsMessageMuteGuard::Impl::Action::kUnmute, std::move(messageIdNumbers)};
}

bool Engine::shouldMuteDebugUtilsMessage(uint32_t messageIdNumber) const
{
    std::lock_guard<std::mutex> lock{mutex};
    return mutedMessageIdNumbers.contains(messageIdNumber);
}

void Engine::createInstance(std::string_view applicationName, uint32_t applicationVersion, std::optional<std::string_view> libraryName, vk::Optional<const vk::AllocationCallbacks> allocationCallbacks)
{
    library = std::make_unique<Library>(libraryName, allocationCallbacks, *this);
    instance = std::make_unique<Instance>(applicationName, applicationVersion, *this, *library);
    physicalDevices = std::make_unique<PhysicalDevices>(*this);
}

vk::Instance Engine::getVulkanInstance() const
{
    return instance->instance;
}

void Engine::createDevice(vk::SurfaceKHR surface)
{
    auto & physicalDevice = physicalDevices->pickPhisicalDevice(surface);
    device = std::make_unique<Device>(physicalDevice.getDeviceName(), *this, *library, physicalDevice);
    vma = std::make_unique<MemoryAllocator>(*this);
}

vk::PhysicalDevice Engine::getVulkanPhysicalDevice() const
{
    return device->physicalDevice.physicalDevice;
}

vk::Device Engine::getVulkanDevice() const
{
    return device->device;
}

uint32_t Engine::getVulkanGraphicsQueueFamilyIndex() const
{
    return device->physicalDevice.externalGraphicsQueueCreateInfo.familyIndex;
}

uint32_t Engine::getVulkanGraphicsQueueIndex() const
{
    return device->physicalDevice.externalGraphicsQueueCreateInfo.index;
}

const Library & Engine::getLibrary() const
{
    return *library;
}

const Instance & Engine::getInstance() const
{
    return *instance;
}

const PhysicalDevices & Engine::getPhysicalDevices() const
{
    return *physicalDevices;
}

const Device & Engine::getDevice() const
{
    return *device;
}

const MemoryAllocator & Engine::getMemoryAllocator() const
{
    return *vma;
}

}  // namespace engine
