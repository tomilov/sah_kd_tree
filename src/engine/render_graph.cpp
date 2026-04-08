#include <engine/render_graph.hpp>

namespace engine::render_graph
{

Builder::Builder(
    utils::Name nameIn,
    const Context & contextIn,
    CommandList & /*commandList*/,
    BuilderFlags /*flags*/)
    : name{std::move(nameIn)}
    , context{contextIn}
{}

}  // namespace engine::render_graph
