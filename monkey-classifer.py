from fastai.vision.all import *
import gradio as gr

learn = load_learner("model.pkl")

categories = ["Bald uakari monkey", "Mandrill monkey", 
                "Panamanian White-faced Capuchin monkey",
                "Golden snub-nosed monkey", 
                "Black howler monkey", "Guinea baboon", 
                "Bonnet macaque", "Japanese macaque"]

def classify_image(image):
    pred, i, prob = learn.predict(image)
    return dict(zip(categories, map(float, prob)))

image = gr.inputs.Image(shape=(192, 192))
label = gr.outputs.Label()

interface = gr.Interface(fn=classify_image, inputs=image, outputs=label)
interface.launch(inline=False)