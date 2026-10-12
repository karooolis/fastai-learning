# Copy of the Hugging Face Space code. model.pkl lives in the Space, not in this repo.

import gradio as gr
from fastai.vision.all import load_learner

learn = load_learner("model.pkl")
labels = learn.dls.vocab


def classify_image(img):
    _, _, probs = learn.predict(img)
    return {label: float(probs[i]) for i, label in enumerate(labels)}


demo = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(),
    outputs=gr.Label(),
    title="Lithuanian vs Swedish Folk Embroidery",
)

if __name__ == "__main__":
    demo.launch()
