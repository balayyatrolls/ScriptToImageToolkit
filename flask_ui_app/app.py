from flask import Flask, request, render_template, send_file
from prompt_engine import generate_prompt
from diffusers import StableDiffusionPipeline
import torch
import uuid
import os

app = Flask(__name__)
pipe = StableDiffusionPipeline.from_pretrained("Lykon/dreamshaper-8", torch_dtype=torch.float16).to("cuda")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        script = request.form["script"]
        prompt = generate_prompt(script)
        image = pipe(prompt).images[0]
        filename = f"output_{uuid.uuid4().hex}.png"
        image.save(filename)
        return send_file(filename, as_attachment=True)
    return render_template("index.html")
