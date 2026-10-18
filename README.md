# LLM Course Project

## Cel projektu
Celem projektu jest implementacja i przetestowanie podstawowych komponentów systemów opartych na modelach językowych (LLM):
1. **Agenta AI** obsługującego wywołania narzędzi wraz ze strukturalną walidacją odpowiedzi.
2. **Systemu Mini-RAG** opartego na lokalnym indeksowaniu i wyszukiwaniu wektorowym (FAISS) oraz wyszukiwaniu leksykalnym.
3. Modułu ewaluacji i zestawu testów jednostkowych/integracyjnych (pytest).

---

## Wymagania systemowe
- **Python**: wersja `>= 3.11`
- **Wirtualne środowisko**: `venv`

---

## Instalacja i przygotowanie środowiska

1. **Klonowanie repozytorium / wejście do katalogu:**
   ```bash
   cd llmy 
   ```
   

## Laboratorium 1

### Wywołanie modelu
Wykonano testowe wywołanie modelu za pomocą skryptu `backends.py`. Metryki:
- **Model:** Qwen/Qwen2.5-0.5B-Instruct
- **Tryb:** local
- **Tokeny wejściowe:** 45 
- **Tokeny wyjściowe:** 116 
- **Czas:** 5.36 s 
*(Dane pochodzą z pliku events.jsonl)*

### Dlaczego liczby tokenów nie można wiarygodnie oszacować liczbą słów?
Tokenizatory dzielą tekst na mniejsze fragmenty ,
dlatego jedno słowo (szczególnie w języku polskim), emotikony czy kod
bez spacji mogą składać się z wielu tokenów, co uniemożliwia proste przeliczanie słów na tokeny.

### Rozwój projektu
Repozytorium będzie rozwijane w kolejnych laboratoriach