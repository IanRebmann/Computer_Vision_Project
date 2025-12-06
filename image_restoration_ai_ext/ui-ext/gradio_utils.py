import cv2

from workflow import graph_spawner
from PIL import Image
import numpy as np

def image_preprocessing(image):
    if image is None:
        return None
    try:

        # https://stackoverflow.com/questions/14134892/convert-image-from-pil-to-opencv-format#14140796
        #image = cv2.imread(f"{image}")
        #print(image.shape)
        #im_pil = Image.fromarray(image)

        #pil_image = Image.open(image)
        #open_cv_image = np.array(pil_image)
        #open_cv_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # room for resizing, cropping, etc
        print(f"Shape: {image.shape}\n Instance: {isinstance(image, (np.ndarray))}")

        return image

    except Exception as e:
        print(f"Error {e}.")
        return None


def output_print(image):
    """
     Generate AI response using the loaded LLM model

     Args:
         user_input (str): User's input message
         history (list): Conversation history for context

     Returns:
         str: Generated AI response
     """
    response = ""

    try:
        graph = graph_spawner()

        for step in graph.stream(
                {"initial_image": image},
                stream_mode="updates",
        ):
            key = next(iter(step))
            #print(f"Outside the graph print: {key}\n")
            response = step[key]['processed_image']
            yield response

        #print(f"Streaming response: {response}.")

    except Exception as e:
        print(f"Error during graph execution {e}.")
        response = image

    yield response

# Create Gradio interface
def chat_interface(image):
    """
    Handle chat interactions with conversation history
    """

    try:

        preprocessed_image = image_preprocessing(image)# returns PIL

        response = output_print(preprocessed_image)
        for step in response:
            #history.append((message, response))
            yield step

    except Exception as e:
        print(f"Error during chat_interface printout {e}.")