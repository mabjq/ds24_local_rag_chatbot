# Rengör textdata från protokoll genom att ta bort oväsentligt innehåll
# (t.ex. sidnummer, vissa rubriker) och extraherar ärendelistor. Förbereder text för chunking
# och vektorsökning i RAG-pipelinen. Används efter preprocess.py och anropas via chunking.py.

import re

def extract_agenda(text):
    """
    Extraherar en strukturerad ärendelista från text baserat på formatet "Ärendelista"
    följt av rader med §-nummer, titel, diarienummer och sidnummer.
    Returnerar ärendelistan som sträng eller None om ingen hittas.
    """
    agenda_match = re.search(
        r'Ärendelista\n'
        r'(§\s*\d+\s+[^\n]+\s+\d{4}/\d+\s+\d+\n'
        r'(?:§\s*\d+\s+[^\n]+\s+\d{4}/\d+\s+\d+\n)*)',
        text
    )
    if agenda_match:
        return agenda_match.group(1).strip()
    return None

def clean_text(text):
    """
    Rengör text från oväsentliga element (sidnummer, vissa rubriker, tomma rader) och
    extraherar ärendelistan. Returnerar en tuple: (rengjord text, ärendelista eller None).
    Används för att förbereda textdata för vektorsökning i RAG-pipelinen.
    """
    # Extrahera ärendelistan innan rengöring
    agenda = extract_agenda(text)

    # Ta bort metadata och upprepade rubriker
    text = re.sub(r"^\s*Sid\s*\d+\s*$", "", text, flags=re.MULTILINE)                               # Sidnummer
    text = re.sub(r"Vård- och omsorgsnämnden\s*\n", "", text, flags=re.IGNORECASE)                  # Nämndrubriker
    text = re.sub(r"Beslut fattat\s*\n", "", text, flags=re.IGNORECASE)                             # Beslutsmarkörer
    text = re.sub(r"Ärendet tas till handlingarna\s*\n", "", text, flags=re.IGNORECASE)             # Standardfraser
    text = re.sub(r"Protokoll nr\s*\d+/\d+\s*\n", "", text, flags=re.IGNORECASE)                    # Protokollnummer
    text = re.sub(r"Sammanträdesdatum\s*\d{4}-\d{2}-\d{2}\s*\n", "", text, flags=re.IGNORECASE)     # Datumrader

    # Normalisera format för numrerade rader och beslut
    text = re.sub(r"^\s*(\d+\.\s*.*)$", r"\1", text, flags=re.MULTILINE)                            # Bevara numrerade punkter
    text = re.sub(r"^\s*(BESLUT.*)$", r"\1", text, flags=re.MULTILINE | re.IGNORECASE)              # Bevara beslutsrader

    # Ta bort ensamma paragrafnummer
    text = re.sub(r"^\s*§\s*\d+\s*$", "", text, flags=re.MULTILINE)

    # Normalisera radbrytningar och ta bort tomma rader
    text = re.sub(r"\n\s*\n+", "\n", text)
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())

    # Ta bort ärendelistan från huvudtexten om den hittades
    if agenda:
        text = text.replace(agenda, "").replace("Ärendelista", "").strip()

    return text, agenda

def process_file(input_path, output_path):
    """
    Läser en textfil, rengör innehållet med clean_text och skriver resultatet till en ny fil.
    Används för att rengöra enskilda protokolltexter i en batchprocess över flera filer.
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    cleaned_text, agenda = clean_text(text)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(cleaned_text)
