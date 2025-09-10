import streamlit as st
import os
import json
from langchain_groq import ChatGroq
from router import RouterAgent
from oldagents.plot_agent import VisualizationAgent  
from dotenv import load_dotenv
import tempfile
import pandas as pd
from PIL import Image

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
    model_name="llama-3.3-70b-versatile",
    api_key=groq_api_key,
    temperature=0.1
)

router = RouterAgent(llm=llm, config=config)

query = st.text_input("Sorgunuzu girin")

# Store result dataframe in session state to persist it
if 'result_dataframe' not in st.session_state:
    st.session_state.result_dataframe = None
if 'dataframe_source' not in st.session_state:
    st.session_state.dataframe_source = None

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
                    st.subheader("SQL Sorgu Sonucu:")
                    st.dataframe(sql_df)
                    st.session_state.result_dataframe = sql_df
                    st.session_state.dataframe_source = "SQL"

                if result.metadata.get("csv_dataframe") is not None:
                    csv_df = result.metadata["csv_dataframe"]
                    st.subheader("CSV Analiz Sonucu:")
                    st.dataframe(csv_df)
                    st.session_state.result_dataframe = csv_df
                    st.session_state.dataframe_source = "CSV"
                
            else:
                st.error(f"Hata: {result.error}")

if st.session_state.result_dataframe is not None:
    st.markdown("---")
    st.subheader(f"{st.session_state.dataframe_source} Verisi Görselleştirme")
    
    df_info = st.session_state.result_dataframe
    st.write(f"**Veri Boyutu:** {df_info.shape[0]} satır x {df_info.shape[1]} sütun")
    st.write(f"**Sütunlar:** {', '.join(df_info.columns.tolist())}")
    
    viz_query = st.text_input(
        "Görselleştirme talebinizi girin:", 
        placeholder="Örnek: Yaş dağılımını histogram olarak göster veya Satış verilerini aylara göre çizgi grafiği yap"
    )
    
    if st.button("Görselleştir"):
        if not viz_query.strip():
            st.warning("Lütfen görselleştirme talebinizi girin.")
        else:
            with st.spinner("Görselleştirme oluşturuluyor..."):
                try:
                    viz_agent = VisualizationAgent(
                        llm=llm, 
                        df=st.session_state.result_dataframe,
                        doc_path=config.get('doc_path'),
                        column_info_path=config.get('column_info'),
                        plots_dir="streamlit_plots"
                    )
                    
                    viz_result = viz_agent.process(viz_query)
                    
                    if viz_result.success:
                        output = viz_result.data.get('output', viz_result.data) if isinstance(viz_result.data, dict) else viz_result.data
                        st.success("Görselleştirme başarıyla oluşturuldu!")
                        
                        st.text_area("Görselleştirme Açıklaması:", value=str(output), height=150)
                        
                        if viz_result.metadata.get("plot_path"):
                            plot_path = viz_result.metadata["plot_path"]
                            
                            if os.path.exists(plot_path):
                                try:
                                    image = Image.open(plot_path)
                                    st.image(image, caption=f"Görselleştirme: {os.path.basename(plot_path)}", use_column_width=True)
                                    
                                    with open(plot_path, "rb") as file:
                                        st.download_button(
                                            label="Görseli İndir",
                                            data=file.read(),
                                            file_name=os.path.basename(plot_path),
                                            mime="image/png"
                                        )
                                except Exception as e:
                                    st.error(f"Görsel yüklenirken hata: {e}")
                                    st.write(f"Görsel dosya yolu: {plot_path}")
                            else:
                                st.warning("Görsel dosyası bulunamadı, ancak işlem tamamlandı.")
                                st.write(f"Beklenen dosya yolu: {plot_path}")
                        else:
                            st.warning("Görsel oluşturuldu ancak dosya yolu alınamadı.")
                            
                    else:
                        st.error(f"Görselleştirme hatası: {viz_result.error}")
                        
                except Exception as e:
                    st.error(f"Görselleştirme agent'ı başlatılırken hata: {e}")
    
    if st.button("Yeni Sorgu İçin Temizle"):
        st.session_state.result_dataframe = None
        st.session_state.dataframe_source = None
        st.experimental_rerun()

