#include <engine/render_graph.hpp>

namespace engine::render_graph
{

Builder::Builder(std::string_view nameIn, const Context & contextIn, CommandList & /*commandList*/, BuilderFlags /*flags*/)
    : name{nameIn}
    , context{contextIn}
{}

}  // namespace engine::render_graph
