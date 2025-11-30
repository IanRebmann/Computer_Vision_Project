import gradio as gr
from PIL import Image
import numpy as np
from super_image import HanModel, ImageLoader

# Orchestrator for different AI modules
class ProjectOrchestrator:
    def __init__(self):
        # Load the SR model just once
        self.sr_models = {
            '2x': HanModel.from_pretrained('eugenesiow/han', scale=2),
            '3x': HanModel.from_pretrained('eugenesiow/han', scale=3),
            '4x': HanModel.from_pretrained('eugenesiow/han', scale=4),
        }

    def run_super_resolution(self, input_image, scale_factor):
        # Convert input to model format
        img_loaded = ImageLoader.load_image(input_image)
        model = self.sr_models[scale_factor]
        # Run model prediction
        preds = model(img_loaded)
        # Convert to PIL image
        sr_image = ImageLoader.save_image(preds, './output.png', return_pil=True)
        # Gather dimensions
        in_w, in_h = input_image.size
        out_w, out_h = sr_image.size
        info = (f"Input Resolution: {in_w}x{in_h} | "
                f"Super-Resolved: {out_w}x{out_h} ({scale_factor} upscaling)")
        return [input_image, sr_image], info

    # --- Stubs for orchestrating additional tasks ---
    # def run_denoising(self, input_image, params):
    #     # Add your denoising module code here
    #     pass
    # def run_colorization(self, input_image, params):
    #     # Add your colorization module code here
    #     pass

# Instantiate orchestrator
orchestrator = ProjectOrchestrator()

def sr_interface(input_image, scale_factor):
    # Run only the SR branch for now
    [orig, upscaled], info = orchestrator.run_super_resolution(input_image, scale_factor)
    # Return images side by side, and dimension info text
    # Stack input/output images horizontally for easier comparison
    combined = Image.new('RGB', (orig.width + upscaled.width, max(orig.height, upscaled.height)))
    combined.paste(orig, (0, 0))
    combined.paste(upscaled, (orig.width, 0))
    return combined, info

with gr.Blocks() as demo:
    gr.Markdown("## Image Super-Resolution Demo\nUpload an image, pick scale, and see results side-by-side.")
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="pil", label="Upload low-res image")
            up_factor = gr.Radio(choices=['2x', '3x', '4x'], value='2x', label="Upscale factor")
            run_button = gr.Button("Run Super-Resolution")
        with gr.Column():
            output_img = gr.Image(type="pil", label="Input / Output Side by Side")
            info_box = gr.Textbox(label="Image Info")

    run_button.click(fn=sr_interface, inputs=[input_img, up_factor], outputs=[output_img, info_box])
    # --- Add UI for other orchestrated tasks here in future ---
    # gr.Markdown("### Denoising (coming soon)")
    # ...
if __name__ == '__main__':
    demo.launch()
