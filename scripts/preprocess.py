# Extraherar text från PDF-protokoll, separerar ärendelistor och förstasidor,
# och sparar resultaten som textfiler för vidare bearbetning i RAG-pipelinen.
# Använder PyMuPDF för PDF-hantering och organiserar utdata i mappar för protokoll, agendor och förstasidor.


import fitz  # PyMuPDF för att läsa och extrahera text från PDF-filer
import os

# Definiera projektets sökvägar relativt rotmappen
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Rotmapp för projektet
pdf_dir = os.path.join(PROJECT_ROOT, "documents")                           # Mapp med ursprungliga PDF-protokoll
output_dir = os.path.join(PROJECT_ROOT, "data")                             # Mapp för extraherade textfiler
agenda_dir = os.path.join(output_dir, "agendas")                            # Mapp för ärendelistor
frontpage_dir = os.path.join(output_dir, "frontpages")                      # Mapp för förstasidor (t.ex. närvarolistor)

# Skapa utdatamappar om de inte finns
os.makedirs(output_dir, exist_ok=True)
os.makedirs(agenda_dir, exist_ok=True)
os.makedirs(frontpage_dir, exist_ok=True)

def extract_page_text(pdf_path, page_number):
    """
    Extraherar text från en specifik sida i en PDF-fil med PyMuPDF.
    Används för att hämta ärendelistor (sida 3) och förstasidor (sida 1).
    Returnerar None vid ogiltigt sidnummer eller fel.
    """
    try:
        doc = fitz.open(pdf_path)
        if 0 <= page_number < doc.page_count:
            page = doc.load_page(page_number)
            text = page.get_text()
            doc.close()
            return text.strip()   # Ta bort ledande/avslutande blanksteg
        else:
            doc.close()
            return None
    except Exception as e:
        print(f"Fel vid extrahering av sida {page_number} från {pdf_path}: {e}")
        return None

def is_probable_agenda(text):
    """
    Identifierar om en text är en ärendelista baserat på nyckelord.
    Returnerar True om texten innehåller ord som "§", "Dnr", "ärende", "förslag" eller "beslut".
    """
    if not text:
        return False
    return any(keyword in text for keyword in ["§", "Dnr", "ärende", "förslag", "beslut"])

# Bearbeta alla PDF-filer i dokumentmappen
for pdf_file in os.listdir(pdf_dir):
    if pdf_file.endswith(".pdf"):
        pdf_path = os.path.join(pdf_dir, pdf_file)

        # 1. Extrahera hela textinnehållet från PDF:en
        try:
            doc = fitz.open(pdf_path)
            full_text = ""
            for page in doc:
                full_text += page.get_text()   # Sammanfoga text från alla sidor
            doc.close()

            # Spara fulltext till en .txt-fil i output_dir
            text_file = os.path.join(output_dir, pdf_file.replace(".pdf", ".txt"))
            with open(text_file, "w", encoding="utf-8") as f:
                f.write(full_text)
        except Exception as e:
            print(f"Fel vid extrahering av all text från {pdf_path}: {e}")

        # 2. Extrahera och spara ärendelista från sida 3 om den är en agenda
        agenda_text = extract_page_text(pdf_path, 2)  # Sida 3 (index 2)
        if agenda_text and is_probable_agenda(agenda_text):
            agenda_file = os.path.join(agenda_dir, pdf_file.replace(".pdf", "_agenda.txt"))
            with open(agenda_file, "w", encoding="utf-8") as f:
                f.write(agenda_text)
            print(f"Sparade ärendelista från {pdf_file} till {agenda_file}")
        else:
            print(f"Inget agendainnehåll hittades i sida 3 av {pdf_file} – ingen agenda sparades.")

        # 3. Extrahera och spara första sidan ( närvarolista)
        frontpage_text = extract_page_text(pdf_path, 0)  # Sida 1 (index 0)
        if frontpage_text:
            frontpage_file = os.path.join(frontpage_dir, pdf_file.replace(".pdf", "_frontpage.txt"))
            with open(frontpage_file, "w", encoding="utf-8") as f:
                f.write(frontpage_text)
            print(f"Sparade första sidan från {pdf_file} till {frontpage_file}")
        else:
            print(f"Kunde inte extrahera första sidan från {pdf_file}")