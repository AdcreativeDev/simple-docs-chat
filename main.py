import streamlit as st
import pandas as pd
from pdf2image import convert_from_path
import pytesseract
from PyPDF2 import PdfReader
from langchain.agents import create_csv_agent
from langchain.llms import OpenAI
import os
from apikey import apikey
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
import time
from PIL import Image

st.set_page_config(
    page_title="Document Chat",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)
image = Image.open("docchat-banner.jpg")
st.image(image, caption='created by MJ')





system_openai_api_key = os.environ.get('OPENAI_API_KEY')
system_openai_api_key = st.text_input(":key: OpenAI Key :", value=system_openai_api_key)
os.environ["OPENAI_API_KEY"] = system_openai_api_key


llm = OpenAI(temperature=0.2)

def pdf_to_text(pdf_path):
    # Step 1: Convert PDF to images
    images = convert_from_path(pdf_path)

    with open('output.txt', 'w') as f:  # Open the text file in write mode
        for i, image in enumerate(images):
            # Save pages as images in the pdf
            image_file = f'page{i}.jpg'
            image.save(image_file, 'JPEG')

            # Step 2: Use OCR to extract text from images
            text = pytesseract.image_to_string(image_file)

            f.write(text + '\n')  # Write the text to the file and add a newline for each page

def load_csv_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df.to_csv("uploaded_file.csv")
    return df

def load_txt_data(uploaded_file):
    with open('uploaded_file.txt', 'w') as f:
        f.write(uploaded_file.getvalue().decode())
    return uploaded_file.getvalue().decode()

def load_pdf_data(uploaded_file):
    with open('uploaded_file.pdf', 'wb') as f:
        f.write(uploaded_file.getbuffer())
    pdf = PdfReader('uploaded_file.pdf')
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    pdf_to_text('uploaded_file.pdf')
    return text

def main():
    st.title("😀 :blue[Simple Chat With Documents]")
    st.subheader('You can select file type : ( txt, csv , pdf ) ')
    file = st.file_uploader("**Step 1 : Upload a file**", type=["csv", "txt", "pdf"])


    if file is not None:
        if file.type == "text/csv":
            doc = "csv"
            data = load_csv_data(file)
            st.write(data)
            agent = create_csv_agent(OpenAI(temperature=0), 'uploaded_file.csv', verbose=True)
            st.dataframe(data)

        elif file.type == "text/plain":
            doc = "text"
            data = load_txt_data(file)
            st.caption("File Contnet:")
            st.info(data)
            loader = TextLoader('uploaded_file.txt')
            index = VectorstoreIndexCreator().from_loaders([loader])

        elif file.type == "application/pdf":
            doc = "text"
            data = load_pdf_data(file)
            loader = TextLoader('output.txt')
            index = VectorstoreIndexCreator().from_loaders([loader])

        # do something with the data


        question = st.text_input("**Step 2 - Enter your question here:**")
        submit_button = st.button('Submit')

        if submit_button:
            if doc == "text":
                response = index.query(question)
            else:
                response = agent.run(question)

            if response:
                st.write(response)


if __name__ == "__main__":
    main()