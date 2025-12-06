from flask import Flask, request, render_template_string, send_file
from io import BytesIO
from PIL import Image

from src.config import load_config
from src.enhancer import ImageEnhancer

app = Flask(__name__)
cfg = load_config("configs/default.yaml")
enhancer = ImageEnhancer(cfg)

HTML = """
<!doctype html>
<title>Image Restoration AI</title>
<h2>Upload an image</h2>
<form method=post enctype=multipart/form-data>
  <label>Task:</label>
  <select name="task">
    <option>denoise</option>
    <option>superres</option>
    <option>colorize</option>
    <option>inpaint</option>
  </select><br><br>
  <label>Image:</label>
  <input type=file name=image required><br><br>
  <label>Mask (for inpaint):</label>
  <input type=file name=mask><br><br>
  <label>Prompt:</label>
  <input type=text name=prompt><br><br>
  <button type=submit>Run</button>
</form>
"""

@app.get("/")
def index():
    return render_template_string(HTML)

@app.post("/")
def run():
    task = request.form.get("task", "denoise")
    prompt = request.form.get("prompt", "")

    img_file = request.files["image"]
    image = Image.open(img_file.stream).convert("RGB")

    if task == "denoise":
        out = enhancer.run_denoise(image, prompt=prompt or cfg.defaults["denoise"]["prompt"])
    elif task == "superres":
        out = enhancer.run_superres(image)
    elif task == "colorize":
        out = enhancer.run_colorize(image, prompt=prompt or cfg.defaults["colorize"]["prompt"])
    elif task == "inpaint":
        mask_file = request.files.get("mask")
        if not mask_file:
            return "Mask required for inpainting.", 400
        mask = Image.open(mask_file.stream).convert("RGB")
        out = enhancer.run_inpaint(image, mask, prompt=prompt)
    else:
        return "Unknown task", 400

    buf = BytesIO()
    out.save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")

if __name__ == "__main__":
    app.run(debug=True, port=7861)
