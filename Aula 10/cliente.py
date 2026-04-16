import streamlit as st
import seaborn as sb
import joblib
import numpy as np
import matplotlib.pyplot
import pickle


st.set_page_config(page_title="Análise por KDE", layout="wide")
@st.cache_resource

def loadModel():
    return joblib.load('modelo_total.pkl')

def conteudoPickle():
    with open('modelo_total.pkl', 'rb') as f:
        conteudo = pickle.load(f)

    return conteudo

try:
    data = loadModel()
    #pklcontent = conteudoPickle()
    kde = data['kde']
    scaler = data['scaler']
    thres = data['thres']
    caracteristicas = data['caracteristicas']
    trainScores = data.get('trainScores', [])
    st.title("Validação do Compressor")
    st.info("Configuração do modelo - KDE")
    st.subheader("Entrada de Dados")

    col1, col2, col3 = st.columns(3)
    with col1:
        tp3 = st.number_input("TP3", value=9.35, format="%.4f")

    with col2:
        h1 = st.number_input("H1", value=9.35, format="%.4f")

    with col3:
        motor = st.number_input("Corrente", value=0.04, format="%.4f")                

    st.divider()    

    if st.button("Validar", use_container_width=True):
        entrada = np.array([[tp3, h1, motor]])
        entrada.scaler.transform(entrada)
        score = kde.trainScores(entrada)[0]
        saude = score - thres

        c1, c2 = st.columns(2)

        with c1:
            st.metric(label="Score", value=score)

        with c2:
            status = "OK" if score >= thres else "NOK"
            if score >= thres: st.success(status)
            else: st.error(status)
except:
    st.error(f"Erro ao carregar o modelo")
finally:
    st.stop()