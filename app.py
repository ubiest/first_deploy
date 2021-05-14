import streamlit as st
#from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import fastai
from fastai.vision.all import *
import time
fig = plt.figure()

with open("custom.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title('Bag Classifier')

st.markdown("Welcome to this simple web application that classifies bags. The bags are classified into six different classes namely: Backpack, Briefcase, Duffle, Handbag and Purse.")

def get_x_cv(r):
    '''## Get the x values in the Cross-Validated scenario'''
    return r['fname']
def get_y(r): return r['labels'].split(' ')

def main():
    file_uploaded = st.file_uploader("Choose File", type=["png","jpg","jpeg"])
    class_btn = st.button("Classify")
    if file_uploaded is not None:
        image = PILImage.create(file_uploaded)
        st.image(image, caption='Uploaded Image', use_column_width=True)

    if class_btn:
        if file_uploaded is None:
            st.write("Invalid command, please upload an image")
        else:
            with st.spinner('Model working....'):
                plt.imshow(image)
                plt.axis("off")
                predictions = predict(image)
                #predictions = predict(file_uploaded)
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

    #result = f"{class_names[np.argmax(scores)]} with a { (100 * np.max(scores)).round(2) } % confidence."
    return predictions











if __name__ == "__main__":
    main()
