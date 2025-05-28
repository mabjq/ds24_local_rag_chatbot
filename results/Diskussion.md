# Diskussion: Chattbot för vård- och omsorgsnämndens protokoll

Denna diskussion analyserar chattbotens potential och utmaningar ur affärsmässiga, etiska och tekniska perspektiv, med fokus på dess användning för att söka information i vård- och omsorgsnämndens protokoll i Sölvesborgs kommun.

**Bakgrund och Utvecklingsprocess**

Projektets mål har varit att undersöka hur en AI-driven chattbot, även i denna enklare form, kan göra det lättare att snabbt hitta information i offentliga dokument, som protokoll från Sölvesborgs kommuns vård- och omsorgsnämnd. 
Syftet har också varit att visa hur tekniken även skulle kunna användas för att hämta information från andra källor och dokument i framtiden. Lösningen utvecklades medvetet med lokal, icke molnbaserad, öppen källkodsteknik för att kunna installeras och köras innanför kommunens brandväggar, vilket gör den säker för känslig data, kostnadseffektiv och transparent.

Till en början testades att använda Pythons `pypdf`, `numpy`, `polars` och Gemini API-modell för att hantera innehåll från PDF-filer som hämtades direkt via URL:er på kommunens hemsida. Modellen var riktigt bra på att förstå och svara på frågor, men hade andra problem, som prestandan med Polars parquet filer och VectorStore-klassen samt att lösningen var beroende av en extern molntjänst och inte lämplig för känslig data i framtiden. Detta ledde till en RAG-pipeline byggt på LangChain, ChromaDB och en lokal språkmodell (Gemma 3 via Ollama) istället och lösningen blev mer skalbar och säker. Den lokala modellens prestanda och kvalitet behövde mer finjustering däremot för att nå samma nivå som Gemini-modellens svar.

## Affärsmässigt perspektiv

### Möjligheter
Chattboten kan göra det mycket enklare för kommunanställda inom vård- och omsorgsförvaltningen att hitta information. Istället för att läsa långa protokoll själva kan de ställa frågor som "Vad beslutades om budgeten 2025?" och få snabba, tydliga svar. Detta sparar tid och resurser, vilket är viktigt i en stressig sektor. Chattboten kan användas för att:
- Snabbt hitta beslut och riktlinjer.
- Underlätta utbildning av nyanställda genom att ge enkel tillgång till historiska protokoll.
- Göra protokollen mer tillgängliga för anställda och kanske även medborgare, om chattboten blir publik, vilket ökar transparensen.

### Utmaningar
Att implementera och använda chattboten innebär flera utmaningar som behöver hanteras:
- Att sätta upp och sköta chattboten kräver teknisk utrustning, som servrar för Ollama och ChromaDB. 
- Personalen måste också lära sig använda chattboten och den ska passa in i det befintliga arbetsflödet. 
- Kommunens begränsade budget kan göra det svårt att prioritera detta framför andra viktiga uppgifter. 
- Om svaren inte är korrekta eller relevanta kan personalen sluta lita på den, så kvaliteten behöver kontinuerligt säkerställas.

## Etiskt perspektiv

### Möjligheter
Chattboten kan göra informationsåtkomst mer öppen och effektiv:
- Genom att göra protokollens innehåll lättillgängligt kan den stödja demokratiska processer, så att anställda och kanske även medborgare kan granska beslut utan tekniska hinder. 
- En tydlig systemprompt som håller svaren inom rätt kontext minskar risken för att fel information sprids.

### Utmaningar
- Om chattboten ger fel eller ofullständiga svar kan det leda till missförstånd som påverkar hur patienter behandlas i slutändan, vilket är extra känsligt i en sektor där etik är viktigt. 
- Även om datan är offentlig måste GDPR-regler följas. Anonymisering och säker lagring är superviktigt för att skydda människors integritet, även om datan just nu bara består av offentliga dokument.

## Tekniskt perspektiv

### Möjligheter
Chattboten bygger på en RAG-pipeline med LangChain, ChromaDB och en lokal LLM. Modellen använder opensource teknik och ett användarvänligt gränssnitt i Streamlit-appen. Tekniska förbättringar har inkluderat:
- **Datumhantering**: `extract_date_from_query` för att filtrera protokoll baserat på datum som extraheras från frågan (t.ex. "27 februari 2025" -> `{'date': '250227'}`).
- **Separerad ärende- och förstasideshantering**: För att hitta och lagra ärendelistan (sidan 3) och förstasidor (med deltagarlistor) separat under chunking. Vid frågor om "ärenden" eller "deltagare" prioriteras sökning i dessa separata delar.
- **Text preprocessing**: `text_cleaner.py` för att ta bort sidnummer, rubriker (t.ex. "Vård- och omsorgsnämnden"), upprepade fraser mm. för minskat brus och förbättrade embeddings.
- **Systemprompt**: Förfinad för att prioritera nyckelord och instruera LLM:en att använda separata ärende- och närvarolistorna vid relevanta frågor.
- **Streamlit-gränssnitt**: Enkel app för att ställa frågor och visa svar.

### Utmaningar och tester
Textutvinning från signerade PDF:er var svårt till en början eftersom varken `PyMuPDF`, `pdfplumber` eller `pypdf` kunde extrahera läsbar text från digitalt signerade och skyddade dokument. `Pdf2image` ihop med `pytesseracr` testades också för att via OCR få ut text men skapade istället tolkningsproblem. Genom att använda osignerade PDF:er löstes detta, men kräver tillgång till rätt dokumentformat ( innan signering sker).  

Till en början gav RAG-pipelinen svar som inte alltid var korrekta, vilket ledde till justeringar som:

- **Chunking-strategi**: `RecursiveCharacterTextSplitter` visade sig fungera väl för denna typ av innehåll och bevarade det semantiska sammanhanget bra. Efter tester valdes en `chunk_size` på `500` med `overlap=150`, vilket gav en bra balans.
- **Embedding-modeller**: Efter tester med flera modeller, däribland `all-MiniLM-L6-v2` och `all-mpnet-base-v2`, valdes **`paraphrase-multilingual-mpnet-base-v2`** eftersom den fungerade bättre för svenska texter.
- **LLM-inställningar och modellval**: Modeller som Llama 3.1 och Gemma 2 utvärderades, men **Gemma 3 12B** valdes till slut på grund av högre svarskvalitet, trots en ökad svarstid. `temperature` justerades till `0.1` och `max_tokens` till `500` för att få precisa svar.
- **Datumfiltrering**: Behövdes för att automatiskt plocka ut och filtrera protokoll baserat på datum i frågorna, vilket gjorde svaren mer träffsäkra för tidsrelaterade frågor.
- **Retriever-precision (`k` parameter)**: Efter att ha testat olika värden för `k` (antalet hämtade dokument), landade det i `k=8` som den bästa kompromissen. Det gav tillräckligt med information utan att skicka för mycket onödig data till modellen.

Valideringsresultaten har blivit mycket bättre genom justeringarna, särskilt för frågor om ärenden och deltagare. 
Men det finns fortfarande problem med specifika frågor som kräver kontext över flera chunks eller frågor som involverar komplexa tidsmässiga relationer där händelsedatum och protokolldatum inte är samma (t.ex. "beslut som togs under 2024 om budgeten för 2025"). Det är svårt att få retrievern att välja rätt protokoll för ett specifikt datum utan att blanda in liknande information från andra protokoll.
En ytterligare utmaning har varit att systemet ibland missar information som finns i inbäddade dokument eller bilagor inom protokoll-PDF:en, trots att texterna har extraherats på ett korrekt sätt. Detta beror sannolikt på skillnader i strukturen mellan inbäddad text och protokolltexten.

### Framtida förbättringar
- Testa och finjustera `chunk_size` och `overlap` ytterligare.
- Förbättra systemprompten så att den kan hantera mer komplexa frågor.
- Förbättrad hantering av inbäddade dokument/bilagor inom PDF:erna, så som policydokument och presentationer.
- Göra så att ChromaDB automatiskt uppdateras när nya protokoll kommer in.
- Testa större embedding-modeller på kraftfullare servrar för att se om det kan ge ännu bättre precision.
- Genomföra tester med användare för att hitta fler förbättringsområden och säkerställa att chattboten verkligen är användbar.

## Slutsats
Chattboten har visat sig vara bra på att extrahera och presentera information om ärenden och svara på enklare frågor. Den separata hanteringen av ärende- och närvarolistan var absolut nödvändig då dessa är vanlig och önskad information att söka fram. Framöver behöver arbetet fokusera på att hantera mer komplexa frågor, inbäddade dokument samt se till att informationssökningen är pålitlig och robust för alla typer av frågor som kan ställas om vård- och omsorgsnämndens protokoll.

