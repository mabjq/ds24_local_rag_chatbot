# Utvärderar RAG-pipelinen genom att köra testfrågor från validation_data.json,
# jämföra genererade svar med förväntade svar, och spara manuellt poängsatta resultat
# till evaluation_results.json. Använder run_rag_query från rag_pipeline.py.

import os
import sys
import json
import yaml

# Lägg till scripts/ i sökvägen för att importera run_rag_query
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from rag_pipeline import run_rag_query

def load_config():
    """
    Läser YAML-konfiguration från config/config.yaml för RAG-pipeline-inställningar.
    Returnerar en dictionary med konfigurationsdata.
    """
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "config", "config.yaml"
    )
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_test_data():
    """
    Läser testdata från validation_data.json.
    Returnerar en lista med testfall (frågor, förväntade svar, protokoll, kategori).
    """
    test_data_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "validation_data.json"
    )
    with open(test_data_path, "r", encoding="utf-8") as f:
        return json.load(f)

def evaluate_question(question, expected_answer, config):
    """
   Kör en testfråga genom RAG-pipelinen och hämtar genererat svar samt dokument.
    Args:
        question (str): Testfrågan.
        expected_answer (str): Förväntat svar för jämförelse.
        config (dict): Konfiguration från config.yaml.
    Returns:
        tuple: (genererat svar, lista med hämtade dokument).
    """
    response, docs = run_rag_query(question, config)
    return response, docs

def main():
    """
    Utvärderar RAG-pipelinen genom att köra alla testfrågor, visa resultat,
    samla in manuella poäng (0, 0.5, 1), och spara resultaten till evaluation_results.json.
    """
    config = load_config()
    test_data = load_test_data()

    results = []
    total_score = 0
    max_score = len(test_data)

    # Bearbeta varje testfall
    for i, test_case in enumerate(test_data, 1):
        question = test_case["question"]
        expected_answer = test_case["expected_answer"]
        protocol = test_case["protocol"]
        category = test_case["category"]

        print(f"\nTestfall {i}:")
        print(f"Fråga: {question}")
        print(f"Förväntat svar: {expected_answer}")
        print(f"Protokoll: {protocol}")
        print(f"Kategori: {category}")

        # Kör fråga och hämta resultat
        try:
            response, docs = evaluate_question(question, expected_answer, config)
            print(f"\nGenererat svar: {response}")
            print("\nHämtade dokument:")
            for doc in docs:
                print(f"  {doc.metadata['filename']}, chunk {doc.metadata['chunk_index']}")

            # Samla in manuell poängsättning
            print("\nPoängsätt svaret (0 = fel, 0.5 = delvis korrekt, 1 = korrekt):")
            score = float(input("Ange poäng (0, 0.5, 1): "))
            while score not in [0, 0.5, 1]:
                print("Ogiltig poäng. Ange 0, 0.5 eller 1.")
                score = float(input("Ange poäng (0, 0.5, 1): "))

            # Spara testresultat
            results.append({
                "question": question,
                "expected_answer": expected_answer,
                "generated_answer": response,
                "score": score,
                "protocol": protocol,
                "category": category,
                "documents": [doc.metadata for doc in docs]
            })
            total_score += score

        except Exception as e:
            print(f"Fel vid utvärdering av fråga: {str(e)}")
            results.append({
                "question": question,
                "expected_answer": expected_answer,
                "generated_answer": "Fel vid körning",
                "score": 0,
                "protocol": protocol,
                "category": category,
                "documents": []
            })

    # Sammanställ och visa resultat
    print("\nUtvärderingsresultat:")
    print(f"Total poäng: {total_score} av {max_score}")
    print(f"Genomsnittlig poäng: {total_score / max_score:.2f}")
    for i, result in enumerate(results, 1):
        print(f"\nTestfall {i}:")
        print(f"Fråga: {result['question']}")
        print(f"Poäng: {result['score']}")
        print(f"Genererat svar: {result['generated_answer'][:100]}...")

    # Spara resultat till JSON
    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "evaluation_results.json"
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResultat sparade till {output_path}")

if __name__ == "__main__":
    main()