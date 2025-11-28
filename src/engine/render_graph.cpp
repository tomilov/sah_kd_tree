#include <engine/render_graph.hpp>

namespace engine::render_graph
{

Builder::Builder(std::string_view name, const Context & context)
    : name{name}
    , context{context}
{
    (void)this->context;
}

}  // namespace engine::render_graph
