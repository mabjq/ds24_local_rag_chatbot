# Läser in rengjorda textfiler, segmenterar protokoll i chunks, behåller
# ärendelistor och förstasidor som odelade dokument, skapar embeddings med HuggingFace-modell
# och lagrar dem i en Chroma-vektordatabas för vektorsökning i RAG-pipelinen.
# Använder text_cleaner.py och förbereder data för rag_query.py.

import os
import re
import yaml
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from text_cleaner import clean_text

def load_config():
    """
    Läser YAML-konfiguration från config/config.yaml för inställningar av chunkning,
    embeddings och sökvägar. Returnerar en dictionary med konfigurationsdata.
    """
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config", "config.yaml"
    )
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_texts(data_dir):
    """
    Läser och rengör textfiler från data-mappen, exkluderar ärendelistor och förstasidor.
    Returnerar en lista med tuples: (filnamn, rengjord text).
    """
    texts = []
    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)
        if os.path.isfile(filepath) and filename.endswith(".txt") and "_agenda.txt" not in filename and "_frontpage.txt" not in filename:
            with open(filepath, "r", encoding="utf-8") as f:
                raw_text = f.read()
                cleaned_text = clean_text(raw_text)[0]   # Använd rengjord text från text_cleaner
                texts.append((filename, cleaned_text))
    return texts

def load_agenda_texts(agenda_dir):
    """
    Läser ärendelistor från agenda-mappen. Returnerar en lista med tuples:
    (ursprungligt filnamn, agendatext, is_agenda=True).
    """
    agendas = []
    for filename in os.listdir(agenda_dir):
        filepath = os.path.join(agenda_dir, filename)
        if os.path.isfile(filepath) and filename.endswith("_agenda.txt"):
            with open(filepath, "r", encoding="utf-8") as f:
                agenda_text = f.read()
                original_filename = filename.replace("_agenda.txt", ".txt")
                agendas.append((original_filename, agenda_text, True))
    return agendas

def load_frontpage_texts(frontpage_dir):
    """
    Läser förstasidor från frontpage-mappen. Returnerar en lista med tuples:
    (ursprungligt filnamn, förstasidestext, is_frontpage=True).
    """
    frontpages = []
    for filename in os.listdir(frontpage_dir):
        filepath = os.path.join(frontpage_dir, filename)
        if os.path.isfile(filepath) and filename.endswith("_frontpage.txt"):
            with open(filepath, "r", encoding="utf-8") as f:
                frontpage_text = f.read()
                original_filename = filename.replace("_frontpage.txt", ".txt")
                frontpages.append((original_filename, frontpage_text, True))
    return frontpages

def create_chunks_and_embeddings(texts, agenda_texts, frontpage_texts, config):
    """
    Segmenterar protokolltexter i chunks, behåller ärendelistor och förstasidor som odelade
    dokument, skapar embeddings och lagrar dem i en Chroma-vektordatabas för vektorsökning.
    """
    # Definiera separatorer för textdelning i prioritetsordning
    separators = ["\n\n", "\n", " ", ""]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["chunking"]["chunk_size"],
        chunk_overlap=config["chunking"]["chunk_overlap"],
        length_function=len,
        separators=separators
    )

    # Skapa embedding-modell för vektorrepresentationer
    embedding_model = HuggingFaceEmbeddings(
        model_name=config["embeddings"]["model_name"],
        model_kwargs={"device": config["embeddings"]["device"]}
    )

    documents = []
    metadatas = []

    # Segmentera protokolltexter i chunks och lägg till metadata
    for filename, text in texts:
        date_match = re.search(r"(\d{6})", filename.replace(".txt", ""))
        date = date_match.group(1) if date_match else "unknown"
        chunks = text_splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            if chunk.strip():
                documents.append(chunk)
                metadatas.append({
                    "filename": filename,
                    "chunk_index": i,
                    "date": date,
                    "is_agenda": False,
                    "is_frontpage": False
                })

    # Lägg till ärendelistor som odelade dokument
    for filename, agenda_text, is_agenda_flag in agenda_texts:
        date_match = re.search(r"(\d{6})", filename.replace(".txt", ""))
        date = date_match.group(1) if date_match else "unknown"
        if agenda_text.strip():
            documents.append(agenda_text)
            metadatas.append({
                "filename": filename,
                "chunk_index": -1,
                "date": date,
                "is_agenda": is_agenda_flag,
                "is_frontpage": False
            })

    # Lägg till förstasidor som odelade dokument
    for filename, frontpage_text, is_frontpage_flag in frontpage_texts:
        date_match = re.search(r"(\d{6})", filename.replace(".txt", ""))
        date = date_match.group(1) if date_match else "unknown"
        if frontpage_text.strip():
            documents.append(frontpage_text)
            metadatas.append({
                "filename": filename,
                "chunk_index": -1,
                "date": date,
                "is_agenda": False,
                "is_frontpage": is_frontpage_flag
            })

    # Skapa och spara Chroma-vektordatabas med embeddings och metadata
    if documents:
        vectorstore = Chroma.from_texts(
            texts=documents,
            embedding=embedding_model,
            metadatas=metadatas,
            persist_directory=config["paths"]["chroma_db_dir"]
        )
        print(f"Skapat {len(documents)} chunks och lagrat embeddings i {config['paths']['chroma_db_dir']}")
    else:
        print("Inga dokument skapades för att lagra i ChromaDB.")

if __name__ == "__main__":
    # Kör pipeline för att ladda texter, skapa chunks och embeddings
    config = load_config()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, config["paths"]["data_dir"])
    agenda_dir = os.path.join(data_dir, "agendas")
    frontpage_dir = os.path.join(data_dir, "frontpages")

    texts = load_texts(data_dir)
    agenda_texts = load_agenda_texts(agenda_dir)
    frontpage_texts = load_frontpage_texts(frontpage_dir)

    if not texts and not agenda_texts and not frontpage_texts:
        print("Inga textfiler hittades.")
    else:
        print(f"Läste in {len(texts)} protokoll, {len(agenda_texts)} agendor och {len(frontpage_texts)} förstasidor")
        create_chunks_and_embeddings(texts, agenda_texts, frontpage_texts, config)
