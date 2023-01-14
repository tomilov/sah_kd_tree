#include <engine/device.hpp>
#include <engine/engine.hpp>
#include <engine/instance.hpp>
#include <engine/library.hpp>
#include <engine/physical_device.hpp>
#include <engine/vma.hpp>
#include <utils/assert.hpp>

#include <spdlog/spdlog.h>
#include <vulkan/vulkan.hpp>

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

void Engine::DebugUtilsMessageMuteGuard::unmute() noexcept(false)
{
    if (std::empty(messageIdNumbers)) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock{mutex};
        while (!std::empty(messageIdNumbers)) {
            auto messageIdNumber = messageIdNumbers.back();
            auto unmutedMessageIdNumber = mutedMessageIdNumbers.find(messageIdNumber);
            INVARIANT(unmutedMessageIdNumber != std::end(mutedMessageIdNumbers), "messageId {:#x} of muted message is not found", messageIdNumber);
            mutedMessageIdNumbers.erase(unmutedMessageIdNumber);
            messageIdNumbers.pop_back();
        }
    }
}

bool Engine::DebugUtilsMessageMuteGuard::empty() const
{
    return std::empty(messageIdNumbers);
}

Engine::DebugUtilsMessageMuteGuard::~DebugUtilsMessageMuteGuard() noexcept(false)
{
    unmute();
}

void Engine::DebugUtilsMessageMuteGuard::mute()
{
    if (!std::empty(messageIdNumbers)) {
        std::lock_guard<std::mutex> lock{mutex};
        mutedMessageIdNumbers.insert(std::cbegin(messageIdNumbers), std::cend(messageIdNumbers));
    }
}

Engine::DebugUtilsMessageMuteGuard::DebugUtilsMessageMuteGuard(std::mutex & mutex, std::unordered_multiset<uint32_t> & mutedMessageIdNumbers, std::initializer_list<uint32_t> messageIdNumbers)
    : mutex{mutex}, mutedMessageIdNumbers{mutedMessageIdNumbers}, messageIdNumbers{messageIdNumbers}
{
    mute();
}

Engine::Engine(std::initializer_list<uint32_t> mutedMessageIdNumbers, bool mute) : debugUtilsMessageMuteGuard{muteDebugUtilsMessages(mutedMessageIdNumbers, mute)}
{}

Engine::~Engine() = default;

auto Engine::muteDebugUtilsMessages(std::initializer_list<uint32_t> messageIdNumbers, bool enabled) const -> DebugUtilsMessageMuteGuard
{
    if (!enabled) {
        return {mutex, mutedMessageIdNumbers, {}};
    }
    return {mutex, mutedMessageIdNumbers, std::move(messageIdNumbers)};
}

bool Engine::shouldMuteDebugUtilsMessage(uint32_t messageIdNumber) const
{
    std::lock_guard<std::mutex> lock{mutex};
    return mutedMessageIdNumbers.contains(messageIdNumber);
}

void Engine::createInstance(std::string_view applicationName, uint32_t applicationVersion, std::optional<std::string_view> libraryName, vk::Optional<const vk::AllocationCallbacks> allocationCallbacks)
{
    library = std::make_unique<Library>(libraryName, allocationCallbacks, *this);
    instance = std::make_unique<Instance>(applicationName, applicationVersion, *this);
    physicalDevices = std::make_unique<PhysicalDevices>(*this);
}

vk::Instance Engine::getInstance() const
{
    return instance->instance;
}

void Engine::createDevice(vk::SurfaceKHR surface)
{
    auto & physicalDevice = physicalDevices->pickPhisicalDevice(surface);
    device = std::make_unique<Device>(physicalDevice.getDeviceName(), *this, physicalDevice);
    vma = std::make_unique<MemoryAllocator>(*this);
}

vk::PhysicalDevice Engine::getPhysicalDevice() const
{
    return device->physicalDevice.physicalDevice;
}

vk::Device Engine::getDevice() const
{
    return device->device;
}

uint32_t Engine::getGraphicsQueueFamilyIndex() const
{
    return device->physicalDevice.externalGraphicsQueueCreateInfo.familyIndex;
}

uint32_t Engine::getGraphicsQueueIndex() const
{
    return device->physicalDevice.externalGraphicsQueueCreateInfo.index;
}

}  // namespace engine
