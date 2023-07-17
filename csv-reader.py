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
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)
image = Image.open("csv_reader.jpg")
st.image(image, caption='created by MJ')


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
    st.title("😀 :blue[Ask your CSV Document]")

    system_openai_api_key = os.environ.get('OPENAI_API_KEY')
    system_openai_api_key = st.text_input(":key: OpenAI Key :", value=system_openai_api_key)
    os.environ["OPENAI_API_KEY"] = system_openai_api_key

    with st.expander("Sample, click here to view"):
        image = Image.open("csv-sample-1.jpg")
        st.image(image, caption='sample')

        image = Image.open("csv-sample-2.jpg")
        st.image(image, caption='sample')

        image = Image.open("csv-sample-3.jpg")
        st.image(image, caption='sample')

    st.subheader('You can select CSV file: ')
    file = st.file_uploader("**Step 1 : Upload a file**", type=["csv"])


    if file is not None:
        if file.type == "text/csv":
            with st.spinner('Process Start ...'):
                llm = OpenAI(temperature=0.2)
                doc = "csv"
                st.write('✔️ Start to upload file')
                data = load_csv_data(file)
                st.write('✔️ Upload Completed.')
                agent = create_csv_agent(OpenAI(temperature=0), 'uploaded_file.csv', verbose=True)
                st.write('✔️ Csv agent Created.')
                st.write('✔️ Load file content.')
                with st.expander("File Content, Click to View"):
                    st.dataframe(data)


        elif file.type == "text/plain":
            with st.spinner('Process Start ...'):
                doc = "text"
                st.write('✔️ Start to upload file')
                data = load_txt_data(file)
                st.write('✔️ Upload Completed.')
                st.write('✔️ Load file content.')
                with st.expander("File Content, Click to View"):
                    st.info(data)
                loader = TextLoader('uploaded_file.txt')
                index = VectorstoreIndexCreator().from_loaders([loader])
                st.write('✔️ Vector Store Created.')


        # elif file.type == "application/pdf":
        #     with st.spinner('Process Start ...'):
        #         doc = "text"
        #         data = load_pdf_data(file)
        #         loader = TextLoader('output.txt')
        #         index = VectorstoreIndexCreator().from_loaders([loader])

        # do something with the data


        question = st.text_input("**Step 2 - Enter your question here:**")
        submit_button = st.button('Submit', type='primary')

        if submit_button:
            if doc == "text":
                with st.spinner('Query the file  ...'):
                    response = index.query(question)
                    st.write('✔️ Query Completed.')


            else:
                with st.spinner('Query the file  ...'):
                    response = agent.run(question)
                    st.write('✔️ Query Completed.')

            if response:
                st.subheader('Query Results:')
                st.info(response)


if __name__ == "__main__":
    main()




log = """


Qurey : How many client ?
> Entering new  chain...
Thought: I need to count the number of unique clients
Action: python_repl_ast
Action Input: len(df['Client'].unique())
Observation: 4
Thought: I now know the final answer
Final Answer: There are 4 clients in this file.

> Finished chain.



Query : he sum of total amount by distinct client  ?
Entering new  chain...
Thought: I need to group the data by client and sum the total amount
Action: python_repl_ast
Action Input: df.groupby('Client')['Total'].sum()
Observation: Client
Candy's Beauty Supply    $1,906.18$9,132.61$1,659.45$2,957.04$1,753.46$...
Elizabethtown Supply     $1,758.68$795.24$5,047.95$9,611.82$3,036.88$2,...
Rockland's               $2,202.48$397.00$9,392.50$1,281.02$202.94$8,10...
Rudiger Pharmacy         $5,108.92$8,307.25$4,597.25$3,227.00$6,062.14$...
Name: Total, dtype: object
Thought: I now know the final answer
Final Answer: The sum of total amount by distinct client is: 
Candy's Beauty Supply: $1,906.18
Elizabethtown Supply: $1,758.68
Rockland's: $2,202.48
Rudiger Pharmacy: $5,108.92





Query : How many distinct client ?
> Entering new  chain...
Thought: I need to find out how many unique clients there are and what their names are
Action: python_repl_ast
Action Input: df['Client'].nunique()
Observation: 4
Thought: I need to find out what the names of the clients are
Action: python_repl_ast
Action Input: df['Client'].unique()
Observation: ["Candy's Beauty Supply" "Rockland's" 'Rudiger Pharmacy'
 'Elizabethtown Supply']
Thought: I now know the final answer
Final Answer: There are 4 distinct clients: Candy's Beauty Supply, Rockland's, Rudiger Pharmacy, and Elizabethtown Supply.





Query : Sum of total amount purchased by client ?
> Entering new  chain...
Thought: I need to group the data by client code and sum the total amount
Action: python_repl_ast
Action Input: df.groupby('Client Code')['Total'].sum()
Observation: Client Code
ARLVA283     $2,202.48
ARLVA425       $397.00
ARLVA519       $202.94
ARLVA727     $8,105.78
ARLVA851     $9,392.50
ARLVA876     $5,695.95
ARLVA924     $1,281.02
CHEMD323     $5,780.04
CHEMD339     $8,307.25
CHEMD365     $6,062.14
CHEMD584    $15,671.28
CHEMD763     $5,108.92
CHEMD887     $3,227.00
CHEMD913     $4,597.25
COLSC133     $3,455.64
COLSC201       $974.25
COLSC481    $12,762.36
COLSC533     $6,967.62
COLSC649       $795.24
COLSC741     $3,036.88
COLSC761     $1,758.68
COLSC813     $2,233.98
COLSC887     $9,611.82
COLSC970     $5,047.95
PINNC496     $1,659.45
PINNC547     $6,132.28
PINNC615     $1,753.46
PINNC674     $2,957.04
PINNC939     $9,132.61
PINNC980     $1,906.18
Name: Total, dtype: object
Thought: I now know the final answer
Final Answer: ARLVA283: $2,202.48, ARLVA425: $397.00, ARLVA519: $202.94, ARLVA727: $8,105.78, ARLVA851: $9,392.50, ARLVA876: $5,695.95, ARLVA924: $1,281.02, CHEMD323: $5,780.04, CHEMD339: $8,307.25, CHEMD365: $6,062.14, CHEMD584: $15,671.28, CHEMD763: $5,108.92, CHEMD887: $3,227.00, CHEMD913: $4,597.25, COLSC133: $3,455.64, COLSC201: $974.25, COLSC481: $12,762.36, COLSC533: $6,967.62, COLSC649: $795.24, COLSC741: $3,036.88, COLSC761: $1,758.68,

> Finished chain.




"""
with st.expander("Log"):
    st.code(log)
    

