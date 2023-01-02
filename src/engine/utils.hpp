#pragma once

#include <vulkan/vulkan.hpp>

#include <vector>

namespace engine
{

template<typename ChainHead, typename... ChainTail>
std::vector<ChainHead> toChainHeads(const std::vector<vk::StructureChain<ChainHead, ChainTail...>> & chains)
{
    std::vector<ChainHead> chainHeads;
    chainHeads.reserve(std::size(chains));
    for (const auto & chain : chains) {
        chainHeads.push_back(chain.template get<ChainHead>());
    }
    return chainHeads;
}

}  // namespace engine
