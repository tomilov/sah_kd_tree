#pragma once

namespace engine
{
class Context;
class FileIo;
struct Library;
struct Instance;
struct PhysicalDevice;
struct PhysicalDevices;

enum class DescriptorManagementKind
{
    Sets,
    Buffer,
    Heap,
};

struct Device;
class MemoryAllocator;
struct QueueCreateInfo;
struct CommandBuffers;
struct CommandPool;
struct PipelineLayout;
struct SpecializationInfo;
struct GraphicsPipeline;
struct ComputePipeline;
struct PipelineCache;
struct Fences;
struct VertexInputState;
struct ShaderStages;
struct RenderPass;

template<typename T>
class MappedMemory;

template<typename T>
class Buffer;

class Image;
}  // namespace engine
