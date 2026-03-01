#include <engine/render_graph.hpp>

namespace engine::render_graph
{

Builder::Builder(std::string_view name, const Context & context, CommandList & /*commandList*/, BuilderFlags /*flags*/)
    : name{name}
    , context{context}
{}

}  // namespace engine::render_graph
