#include <engine/render_graph.hpp>

namespace engine::render_graph
{

Builder::Builder(std::string_view name, const Context & context, CommandListImmediate & commandList, BuilderFlags flags)
    : name{name}
    , context{context}
{
    (void)this->context;
}

void Builder::AddPassDependency(Pass * producer, Pass * consumer)
{

}

void Builder::AddDispatchHint()
{

}

}  // namespace engine::render_graph
