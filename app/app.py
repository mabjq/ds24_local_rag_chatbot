# Streamlit-webbapp som tillhandahåller ett användargränssnitt för att ställa frågor
# till RAG-pipelinen (via rag_pipeline.py) och visa svar samt källor från vård- och
# omsorgsnämndens protokoll. Använder konfiguration från config.yaml.

import streamlit as st
import os
import sys
import yaml
from PIL import Image

# Lägg till scripts/ i sökvägen för att importera run_rag_query
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from rag_pipeline import run_rag_query 

def load_config():
    """
    Läser YAML-konfiguration från config/config.yaml för RAG-pipeline-inställningar.
    Returnerar en dictionary med konfigurationsdata.
    """
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    """
    Konfigurerar och kör Streamlit-webbgränssnittet för att interagera med RAG-pipelinen.
    Hanterar användarfrågor, visar svar och källor, samt formaterar layouten.
    """
    st.set_page_config(page_title="AI sök i Protokoll (RAG-demo)", layout="wide")

    # Anpassa gränssnittets stil med CSS
    st.markdown("""
        <style>
        .main {
            background-color: #ffffff;
            color: #003366;
        }
        h1, h2, h3, h4 {
            color: #003366;
        }
        .stButton>button {
            background-color: #FF6600;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            padding: 0.5em 1em;
        }
        .stTextInput>div>div>input {
            border: 2px solid #003366;
        }
        .stExpanderHeader {
            background-color: #f0f0f0;
        }
        </style>
    """, unsafe_allow_html=True)

    # Visa Sölvesborgs logotyp om den finns
    logo_path = os.path.join(os.path.dirname(__file__), "static", "solvesborg_logotyp.png")
    if os.path.exists(logo_path):
        logo = Image.open(logo_path)
        st.image(logo, width=150)

    # Sätt sidans titel
    st.title("AI sök i Protokoll   (RAG-demo)")

    # Visa projektbeskrivning i gränssnittet
    st.markdown("""
    Detta är en **teknikdemonstration** som visar hur AI kan användas för att söka i nämndsprotokoll med hjälp av RAG (Retrieval-Augmented Generation).  
    Lösningen kan enkelt anpassas till andra typer av dokument och informationsmängder.
    """)

    # Ladda konfiguration för RAG-pipelinen
    config = load_config()
    
    # Skapa kolumner för fråga och källor
    col1, col2 = st.columns([3, 2])

    with col1:
        # Textfält för användarens fråga
        query = st.text_input("📥 Din fråga:", placeholder="Vad diskuterades den 27 februari 2025?")
        
        # Kör RAG-fråga vid knapptryck
        if st.button("📨 Skicka fråga") and query:
            with st.spinner("🔎 Genererar svar..."):
                try:
                    response, docs = run_rag_query(query, config)
                    st.subheader("📤 Svar:")
                    st.write(response)
                    st.session_state["docs"] = docs   # Spara källor i session_state
                except Exception as e:
                    st.error(f"Ett fel uppstod: {str(e)}")

    with col2:
        # Visa källor för genererat svar
        st.subheader("📚 Källor:")
        if "docs" in st.session_state and st.session_state["docs"]:
            for i, doc in enumerate(st.session_state["docs"], 1):
                with st.expander(f"📄 Dokument {i}: {doc.metadata['filename']}"):
                    st.write(f"**Chunk-index**: {doc.metadata['chunk_index']}")
                    st.write(f"**Innehåll**: {doc.page_content[:200]}...")
        else:
            st.write("Inga källor att visa ännu. Ställ en fråga!")

if __name__ == "__main__":
    main()
