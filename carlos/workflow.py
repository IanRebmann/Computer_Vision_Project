# Missing imports

# Internal Models

# State
class State(MessagesState):
    initial_image: Image #need to define the Image type (and import it)
    processed_image: array #numpy array sent between nodes? image?
    processed_image: Image

# Structured Output Bindings

# Nodes
def denoiser(state: State):
    """Remove noise from images while preserving important details."""

    # fine-tuned model is loaded
    # fine-tuned model is fed the image
    # image is saved and moves on to the next node

    msg = llm.invoke(f"Write a joke about {state['topic']}")
    return {"processed_image": msg.content}

def super_resolutioner(state: State):
    """Enhance the resolution of low-resolution images."""

    # fine-tuned model is loaded
    # fine-tuned model is fed the image
    # image is saved and moves on to the next node

    #msg = llm.invoke(f"Write a story about {state['topic']}")
    return {"processed_image": msg.content}

def colorizer(state: State):
    """Automatically colorize black and white or grayscale images."""

    # fine-tuned model is loaded
    # fine-tuned model is fed the image
    # image is saved and moves on to the next node

    #msg = llm.invoke(f"Write a poem about {state['topic']}")
    return {"processed_image": msg.content}

def inpainter(state: State):
    """Fill in missing or damaged parts of images seamlessly."""

    # fine-tuned model is loaded
    # fine-tuned model is fed the image
    # image is saved and moves on to the next node

    #msg = llm.invoke(f"Write a poem about {state['topic']}")
    return {"processed_image": msg.content}

# Compiler
graph_builder = StateGraph(MessagesState)

# add nodes
graph_builder.add_node("denoiser", denoiser)
graph_builder.add_node("super_resolutioner", super_resolutioner)
graph_builder.add_node("colorizer", colorizer)
graph_builder.add_node("inpainter", inpainter)

# add edges
graph_builder.add_edge(START, "denoiser")
graph_builder.add_edge("denoiser", "super_resolutioner")
graph_builder.add_edge("super_resolutioner", "colorizer")
graph_builder.add_edge("colorizer", "inpainter")
graph_builder.add_edge("inpainter", END)

# compile
graph = graph_builder.compile()

#v2 - Compiler

nodes = [denoiser, super_resolutioner, colorizer, inpainter]
edges = [[START, "denoiser"],["denoiser", "super_resolutioner"],
["super_resolutioner", "colorizer"],["colorizer", "inpainter"],["inpainter", END]]

def graph_compiler(state, nodes, edges):

    graph_builder = StateGraph(state)

    for node in nodes:
        graph_builder.add_node(node[__name__], node) # __name__ might not work

    for edge in edges:
        graph_builder.add_edge(edge[0], edge[1])

    graph = graph_builder.compile()

    return graph

# Invoke, doesn't go here but keeping it for now

messages = [HumanMessage(content="Add 3 and 4.")]
messages = agent.invoke({"messages": messages})
for m in messages["messages"]:
    m.pretty_print()


# Rendering, show workflow as png
def graph_rendering(graph):
    display(Image(graph.get_graph().draw_mermaid_png())) # missing path, we want to write it to file
