from langgraph.constants import START
from langgraph.graph import StateGraph
from langgraph.graph import END
from langgraph.graph import MessagesState
import numpy as np
import FinalProject.graphs.models as models

from typing import Annotated, Literal
import operator
from PIL import Image
from pydantic import Field, BaseModel
import os

# State
class State(MessagesState):
    initial_image: np.ndarray
    processed_image: np.ndarray

# Structured Output Bindings

# Nodes
#def denoiser(state: State):
    #"""Remove noise from images while preserving important details."""

    # fine-tuned model is loaded
    # fine-tuned model is fed the image
    # image is saved and moves on to the next node

    #msg = llm.invoke(f"Write a joke about {state['topic']}")
    #return {"processed_image": msg.content}

#def super_resolutioner(state: State):
    #"""Enhance the resolution of low-resolution images."""

    # fine-tuned model is loaded
    # fine-tuned model is fed the image
    # image is saved and moves on to the next node

    #msg = llm.invoke(f"Write a story about {state['topic']}")
    #return {"processed_image": msg.content}

def colorizer(state: State):
    """Automatically colorize black and white or grayscale images."""

    try:
        #print(f"Image shape: {state['initial_image'].shape}\n")

        # model is fed the image
        image = models.initialized_models.colorization_model.colorize(state['initial_image'])
        #models.initialized_models.colorization_model.display(image)

        return {"processed_image": image}

    except Exception as e:
        print(f"Error during colorization {e}.")

#def inpainter(state: State):
    #"""Fill in missing or damaged parts of images seamlessly."""

    # fine-tuned model is loaded
    # fine-tuned model is fed the image
    # image is saved and moves on to the next node

    #msg = llm.invoke(f"Write a poem about {state['topic']}")
    #return {"processed_image": msg.content}

#nodes = [denoiser, super_resolutioner, colorizer, inpainter]
#edges = [[START, "denoiser"],["denoiser", "super_resolutioner"],
#["super_resolutioner", "colorizer"],["colorizer", "inpainter"],["inpainter", END]]

def graph_compiler(state, nodes, edges):
    try:

        graph_builder = StateGraph(state)
        [graph_builder.add_node(node.__name__, node) for node in nodes]
        [graph_builder.add_edge(edge[0], edge[1]) for edge in edges]

        graph = graph_builder.compile()

        return graph

    except Exception as e:
        print(f"Error during graph compilation {e}.")

# Rendering, show workflow as png
def graph_rendering(graph):
    from IPython.display import Image

    try:

        path = "/home/alberto/Documents/MSAAI/CV/final_project/Computer_Vision_Project/carlos/graph.png" # hard coded path
        graph = Image(graph.get_graph().draw_mermaid_png(output_file_path=(path)))

    except Exception as e:
        print(f"Error during graph rendering {e}.")

def graph_spawner(state=State):

    try:

        # v2 - Compiler
        nodes = [colorizer] # hard coded
        edges = [[START, "colorizer"], ["colorizer", END]] # hard coded

        graph = graph_compiler(state, nodes, edges)
        graph_rendering(graph)

        return graph

    except Exception as e:
        print(f"Error during graph spawner {e}.")