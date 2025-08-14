import streamlit as st
import os
import json
from langchain_groq import ChatGroq
from router import RouterAgent
from dotenv import load_dotenv
import tempfile
import pandas as pd

load_dotenv()

st.title("Router Agent Sistemi")

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("GROQ_API_KEY bulunamadı.")
    st.stop()

st.subheader("JSON Konfigürasyon")
default_json = {
    'db_params': {
        "host": "localhost",
        "database": "musteri_db",
        "user": "postgres",
        "password": "123",
        "port": "5432"
    }
}

json_input = st.text_area(
    "JSON config girin:",
    value=json.dumps(default_json, indent=2),
    height=200
)

try:
    config = json.loads(json_input)
    st.success("JSON config başarıyla okundu!")
except json.JSONDecodeError as e:
    st.error(f"JSON format hatası: {e}")
    config = default_json

st.subheader("CSV Dosyası")
uploaded_file = st.file_uploader("Veri dosyasını yükleyin (.csv)", type=["csv"])

if uploaded_file:
    csv_filename = f"temp_{uploaded_file.name}"
    with open(csv_filename, "wb") as f:
        f.write(uploaded_file.read())
    
    config['csv_path'] = csv_filename
    st.success(f"Veri başarıyla yüklendi: {uploaded_file.name}")
    df = pd.read_csv(csv_filename)
    st.dataframe(df.head())

st.subheader("Column Info PDF")
uploaded_column_info = st.file_uploader("Column info PDF dosyası yükle", type=["pdf"], key="column_info")

if uploaded_column_info:
    column_filename = f"temp_column_{uploaded_column_info.name}"
    with open(column_filename, "wb") as f:
        f.write(uploaded_column_info.read())
    
    config['column_info'] = column_filename
    st.success(f"Column info PDF yüklendi: {uploaded_column_info.name}")

st.subheader("Context Info PDF")
uploaded_context = st.file_uploader("Context PDF dosyası yükle", type=["pdf"], key="doc_path")

if uploaded_context:
    context_filename = f"temp_context_{uploaded_context.name}"
    with open(context_filename, "wb") as f:
        f.write(uploaded_context.read())
    
    config['doc_path'] = context_filename
    st.success(f"Context PDF yüklendi: {uploaded_context.name}")

llm = ChatGroq(
    model_name="openai/gpt-oss-120b",
    api_key=groq_api_key,
    temperature=0.1
)

router = RouterAgent(llm=llm, config=config)

query = st.text_input("Sorgunuzu girin")

if st.button("İşle"):
    if not query.strip():
        st.warning("Lütfen geçerli bir sorgu girin.")
    else:
        with st.spinner("Sorgu işleniyor..."):
            result = router.process(query)
            if result.success:
                output = result.data.get('output', result.data) if isinstance(result.data, dict) else result.data
                st.success("İşlem başarılı!")
                st.text_area("Sonuç:", value=str(output), height=300)

                if result.metadata.get("sql_dataframe") is not None:
                        sql_df = result.metadata["sql_dataframe"] 
                        st.dataframe(sql_df)

                    
                if result.metadata.get("csv_dataframe") is not None:
                    csv_df = result.metadata["csv_dataframe"]
                    st.dataframe(csv_df)


                
            else:
                st.error(f"Hata: {result.error}")