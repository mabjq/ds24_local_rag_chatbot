# Hanterar RAG-frågor genom att analysera frågor, hämta relevanta dokument
# från Chroma-vektordatabasen (skapad i chunking.py), och generera svar med språkmodellen
# via Langchain/Ollama baserat på kontext och systemprompt från config.yaml.

import os
import yaml
import re
import datetime
from langchain_ollama import OllamaLLM
from langchain.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def load_config():
    """
    Läser YAML-konfiguration från config/config.yaml för inställningar av embeddings,
    språkmodell och RAG-parametrar. Returnerar en dictionary med konfigurationsdata.
    """
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config", "config.yaml"
    )
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def is_agenda_query(query: str) -> bool:
    """
    Identifierar om frågan avser ärendelistor baserat på nyckelord.
    Returnerar True om nyckelord som "ärenden" eller "dagordning" finns.
    """
    agenda_keywords = ["ärenden", "ärendelista", "dagordning", "vilka ärenden"]
    return any(word in query.lower() for word in agenda_keywords)

def is_frontpage_query(query: str) -> bool:
    """
    Identifierar om frågan avser förstasidor ( deltagare eller mötesinformation)
    baserat på nyckelord. Returnerar True om relevanta nyckelord hittas.
    """
    frontpage_keywords = [
        "deltagare", "närvarande", "frånvarande", "ordförande",
        "sekreterare", "justerare", "protokollförare", "sammanträdesdatum",
        "mötestid", "mötesplats", "ledamöter", "ersättare"
    ]
    return any(word in query.lower() for word in frontpage_keywords)

def extract_date_from_query(query: str) -> str | None:
    """
    Extraherar ett datum från frågan i formatet YYMMDD.
    Stödjer format som "27 februari 2025", "20250227" och "250227".
    Returnerar None om inget giltigt datum hittas.
    """
    query = query.lower()

    # Matcha textformat, t.ex. "27 februari 2025"
    match = re.search(r"(\d{1,2}) (\w+) (\d{4})", query)
    if match:
        day, month_str, year = match.groups()
        months = {
            "januari": 1, "februari": 2, "mars": 3, "april": 4, "maj": 5,
            "juni": 6, "juli": 7, "augusti": 8, "september": 9,
            "oktober": 10, "november": 11, "december": 12
        }
        if month_str in months:
            try:
                dt = datetime.date(int(year), months[month_str], int(day))
                return dt.strftime("%y%m%d")
            except ValueError:
                return None

    # Matcha numeriskt format, t.ex. "20250227"
    match = re.search(r"\b(\d{4})(\d{2})(\d{2})\b", query)
    if match:
        year, month, day = match.groups()
        try:
            dt = datetime.date(int(year), int(month), int(day))
            return dt.strftime("%y%m%d")
        except ValueError:
            return None

    # Matcha kort numeriskt format, t.ex. "250227"
    match = re.search(r"\b(\d{2})(\d{2})(\d{2})\b", query)
    if match:
        year, month, day = match.groups()
        try:
            dt = datetime.date(int("20" + year), int(month), int(day))
            return dt.strftime("%y%m%d")
        except ValueError:
            return None

    return None

def run_rag_query(query, config):
    """
    Kör en RAG-fråga genom att identifiera frågetyp (ärendelista, förstasida, övrigt),
    extrahera datum, hämta relevanta dokument från Chroma och generera svar med Ollama.
    Returnerar en tuple: (svar från språkmodellen, hämtade dokument).
    """
    # Skapa embedding-modell och ladda Chroma-vektordatabas
    embedding_model = HuggingFaceEmbeddings(
        model_name=config["embeddings"]["model_name"],
        model_kwargs={"device": config["embeddings"]["device"]}
    )

    vectorstore = Chroma(
        persist_directory=config["paths"]["chroma_db_dir"],
        embedding_function=embedding_model
    )

    date_filter = extract_date_from_query(query)

    # Filtrera dokument baserat på frågetyp och datum
    if is_agenda_query(query):
        docs = vectorstore.similarity_search(
            query,
            k=config["rag"]["top_k"],
            filter={"is_agenda": True}   # Endast ärendelistor
        )
        if date_filter:
            docs = [doc for doc in docs if doc.metadata.get("date") == date_filter]

    elif is_frontpage_query(query):
        docs = vectorstore.similarity_search(
            query,
            k=config["rag"]["top_k"],
            filter={"is_frontpage": True}   # Endast förstasidor
        )
        if date_filter:
            docs = [doc for doc in docs if doc.metadata.get("date") == date_filter]

    else:
        docs = vectorstore.similarity_search(
            query,
            k=config["rag"]["top_k"]   # Ingen filtrering för generella frågor
        )

    # Bygg kontext med dokument och metadata
    context_with_metadata = []
    for doc in docs:
        metadata_str = (
            f"Fil: {doc.metadata.get('filename', 'okänd')}, "
            f"Datum: {doc.metadata.get('date', 'okänt')}, "
            f"Agenda: {doc.metadata.get('is_agenda', False)}, "
            f"Förstasida: {doc.metadata.get('is_frontpage', False)}"
        )
        context_with_metadata.append(f"[{metadata_str}]\n{doc.page_content}")

    context = "\n\n---\n\n".join(context_with_metadata)

    # Generera svar med språkmodell
    llm = OllamaLLM(
        model=config["llm"]["model_name"],
        temperature=config["llm"]["temperature"],
        max_tokens=config["llm"]["max_tokens"]
    )

    prompt = f"{config['rag']['system_prompt']}\n\nKontext (inkluderar metadata): {context}\n\nFråga: {query}"
    response = llm.invoke(prompt)

    return response, docs

if __name__ == "__main__":
    # Testkör RAG-pipelinen med exempel frågor
    config = load_config()

    queries = [
        "Vilka ärenden behandlades den 27 februari 2025?",
        "Vilka deltagare fanns med vid mötet den 30 januari 2025?",
        "Hur ser demografin ut i Sölvesborg kommande åren?"
    ]

    for query in queries:
        print(f"\n--- Testar fråga: '{query}' ---")
        response, docs = run_rag_query(query, config)
        print("\nSvar från chattboten:")
        print(response)
        print("\nHämtade dokument:")
        for i, doc in enumerate(docs, 1):
            print(f"Dokument {i}: Fil: {doc.metadata.get('filename')}, Datum: {doc.metadata.get('date')}, Agenda: {doc.metadata.get('is_agenda')}, Förstasida: {doc.metadata.get('is_frontpage')}")
            print(f"Innehåll: {doc.page_content[:100]}...")
