import gradio as gr
from gradio_utils import chat_interface

def exit_gradio():
    demo.close()
do_rescale=False
with gr.Blocks(title="Image Restoration") as demo:
    gr.Markdown("<center><h1>Image restoration.</h1></center>")
    with gr.Row():
        with gr.Column(scale=5):
            gr.Markdown("Greetings! Please upload your image.")
        with gr.Column(scale=1):
            exit_btn = gr.Button("Exit")

    with gr.Row():
        msg = gr.Interface(chat_interface, gr.Image(), "image", api_name="predict")

    exit_btn.click(exit_gradio)

demo.launch(theme=gr.themes.Soft())