import os
import time

from typing import Protocol
from dataclasses import dataclass


@dataclass
class Generation:
    text: str
    input_tokens: int | None
    output_tokens: int | None
    latency_s: float
    model: str
    mode: str


class LLMBackend(Protocol):
    def generate(self, system: str, user: str, max_tokens: int = 160,
                 temperature: float = 0.0, top_p: float = 1.0, do_sample: bool = False) -> Generation:
        ...


class ApiBackend:
    def __init__(self) -> None:
        from google import genai

        self.model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def generate(self, system: str, user: str, max_tokens: int = 160,
                 temperature: float = 0.0, top_p: float = 1.0, do_sample: bool = False) -> Generation:
        from google.genai import types

        started = time.perf_counter()
        response = self.client.models.generate_content(
            model=self.model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            ),
        )
        latency_s = time.perf_counter() - started

        generated_text = response.text
        if not generated_text:
            raise RuntimeError("API nie zwróciło tekstu odpowiedzi")

        usage = response.usage_metadata
        return Generation(
            text=generated_text,
            input_tokens=getattr(usage, "prompt_token_count", None),
            output_tokens=getattr(usage, "candidates_token_count", None),
            latency_s=latency_s,
            model=self.model,
            mode="api",
        )


class LocalBackend:
    def __init__(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.model_name = os.getenv("LOCAL_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype="auto",
        )
        self.model.to(self.device).eval()

    def generate(self, system: str, user: str, max_tokens: int = 160,
                 temperature: float = 0.0, top_p: float = 1.0, do_sample: bool = False) -> Generation:
        started = time.perf_counter()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)

        # Dynamiczne budowanie parametrów generacji dla Hugging Face
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
            "do_sample": do_sample,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                **gen_kwargs
            )

        input_tokens = int(inputs["input_ids"].shape[1])
        new_ids = output[0, input_tokens:]
        generated_text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        latency_s = time.perf_counter() - started

        return Generation(
            text=generated_text,
            input_tokens=input_tokens,
            output_tokens=int(new_ids.shape[0]),
            latency_s=latency_s,
            model=self.model_name,
            mode="local",
        )


def make_backend() -> LLMBackend:
    mode = os.getenv("LLM_MODE", "local")
    if mode == "local":
        return LocalBackend()
    if mode == "api":
        return ApiBackend()
    raise ValueError("LLM_MODE musi mieć wartość 'local' albo 'api'")


if __name__ == "__main__":
    import dotenv

    dotenv.load_dotenv()
    backend = make_backend()
    result = backend.generate(
        system="Odpowiadaj krótko i jasno po polsku.",
        user="Wyjaśnij, czym jest token w modelu językowym.",
        max_tokens=160,
    )

    print(result.text)
    print(
        f"\nModel: {result.model} | tryb: {result.mode} | "
        f"wejście: {result.input_tokens} tokenów | "
        f"wyjście: {result.output_tokens} tokenów | "
        f"czas: {result.latency_s:.2f} s"
    )