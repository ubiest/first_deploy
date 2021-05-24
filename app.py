import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import fastai
from fastai.vision.all import *
import time
fig = plt.figure()

st.set_page_config(page_title='VR Photos Classifier', page_icon=None, layout='centered', initial_sidebar_state='auto')
with open("custom.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title('Vacation Rentals Photos Classifier')

st.markdown("Our web application classifies rooms in a vacation rental property. \
    Currently, the rooms or locations are Bathroom, Bedroom, Kitchen, Living Area, Dining Area, \
    Terrace, Pool & Garden, Exteriors & Views, and Others (office, stairwells, corridors, etc).\n \
    Upload an image and try it out!")

def get_x_cv(r):
    '''## Get the x values in the Cross-Validated scenario'''
    return r['fname']
def get_y(r): return r['labels'].split(' ')

def main():
    file_uploaded = st.file_uploader("Choose File", type=["png","jpg","jpeg","webp" ])
    class_btn = st.button("Classify")
    if file_uploaded is not None:
        image = PILImage.create(file_uploaded)
        st.image(image, caption='Uploaded Image', use_column_width=True)

    if class_btn:
        if file_uploaded is None:
            st.write("Invalid command, please upload an image")
        else:
            with st.spinner('Model working....'):
                predictions = predict(image)
                time.sleep(1)
                st.success('Classified')
                st.write(predictions)
                st.pyplot(fig)


def predict(image):
    model_path = Path('../')
    classifier_model = "model_T19-F44-LR0.003.pkl"
    IMAGE_SHAPE = (224, 224,3)
    model_inference = load_learner(model_path/classifier_model)
    print(image)

    predictions = model_inference.predict(image)
    x_mask = predictions[1].numpy()
    percents = predictions[2].numpy()
    predicts = [str.title(x.replace('_', ' ')) for x in predictions[0]]
    weights = [percents[element] for element in (np.nonzero(x_mask))[0]]
    output = ' \n '.join([f'{pred} with a probability of {weight:.2%}.' for pred, weight in zip(predicts, weights)])
    return output #predicts, predictions[2].numpy()











if __name__ == "__main__":
    main()
