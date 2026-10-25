import os
import csv
from collections import Counter
import dotenv
from backends import make_backend


dotenv.load_dotenv()
os.environ["LLM_MODE"] = "local"
backend = make_backend()


configs = {
    "A": {"temperature": 0.0, "top_p": 1.0, "do_sample": False},
    "B": {"temperature": 0.3, "top_p": 1.0, "do_sample": True},
    "C": {"temperature": 0.8, "top_p": 0.9, "do_sample": True},
    "D": {"temperature": 1.2, "top_p": 0.9, "do_sample": True}
}

prompt = "Podaj nazwę i jednozdaniowy opis aplikacji pomagającej studentowi planować naukę."
rows = []

for cfg_name, params in configs.items():
    for run in range(1, 6):

        response = backend.generate(
            system="Odpowiadaj krótko i na temat.",
            user=prompt,
            max_tokens=160,
            temperature=params["temperature"],
            top_p=params["top_p"],
            do_sample=params["do_sample"]
        )

        rows.append({
            "configuration": cfg_name,
            "run": run,
            "text": response.text,
            "output_tokens": response.output_tokens,
            "latency_s": round(response.latency_s, 2)
        })
        print(f"Zakończono: {cfg_name} - próba {run}/5")


def normalized(text: str) -> str:
    return " ".join(text.lower().split())


counts = Counter(normalized(row["text"]) for row in rows)
print("Unikalne odpowiedzi:", len(counts))

with open("lab01b_sampling.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)