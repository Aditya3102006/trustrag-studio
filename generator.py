"""
generator.py — Multi-Engine LLM Generation for TrustRAG.

Supports:
1. Local Offline Model: Google Flan-T5 Base (with exact logit & token probability access)
2. Google Gemini Free API Tier (gemini-1.5-flash, gemini-2.0-flash via Google AI Studio)
3. Groq Free Tier API (ultra-fast Llama-3.3-70B, Mixtral)
4. OpenAI API (GPT-4o, GPT-4o-mini)
"""
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import requests

MODEL_NAME = "google/flan-t5-base"

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
model.eval()


# ── 1. Local Flan-T5 with Logit Extraction ─────────────────────────────────────

def generate_with_logits(prompt: str, max_new_tokens: int = 400):
    """Run local Flan-T5 with token logits for Shannon Entropy calculation."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.2,
            return_dict_in_generate=True,
            output_scores=True,
        )

    generated_ids = output.sequences[0]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    scores = output.scores  # one tensor per generated token

    return text, scores, generated_ids


# ── 2. Google Gemini API (Free Tier via Google AI Studio) ──────────────────────

def generate_with_gemini(prompt: str, api_key: str, model_name: str = "gemini-1.5-flash") -> str:
    """
    Call Google Gemini Free API directly via official REST endpoint.
    Automatically tries supported model candidates (gemini-1.5-flash, gemini-2.0-flash, gemini-2.5-flash, gemini-pro).
    """
    if not api_key:
        raise ValueError("Google Gemini API Key is required. Get a free key at https://aistudio.google.com/")

    candidate_models = ["gemini-3.6-flash", "gemini-flash-latest", "gemini-2.5-flash", "gemma-4-26b-a4b-it", "gemini-1.5-flash", "gemini-pro"]
    seen = set()
    models_to_try = [m for m in candidate_models if not (m in seen or seen.add(m))]

    last_err = ""
    for m in models_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={api_key}"
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 1024,
            }
        }
        headers = {"Content-Type": "application/json"}
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=45)
            if response.status_code == 200:
                data = response.json()
                if "candidates" in data and data["candidates"]:
                    return data["candidates"][0]["content"]["parts"][0]["text"].strip()
            else:
                last_err = f"{m} returned {response.status_code}: {response.text[:200]}"
        except Exception as e:
            last_err = str(e)

    raise RuntimeError(f"Gemini API Error: {last_err}")


# ── 3. Groq API (Free Tier — High Speed Llama-3.3 70B) ─────────────────────────

def generate_with_groq(prompt: str, api_key: str, model_name: str = "llama-3.3-70b-versatile") -> str:
    """
    Call Groq Cloud free tier API (OpenAI-compatible) for ultra-fast Llama-3 reasoning.
    """
    if not api_key:
        raise ValueError("Groq API Key is required. Get a free key at https://console.groq.com/")

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are an expert AI research assistant. Provide thorough, grounded, and detailed technical insights based on the provided context."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 1024,
    }

    response = requests.post(url, json=payload, headers=headers, timeout=45)
    if response.status_code != 200:
        raise RuntimeError(f"Groq API Error ({response.status_code}): {response.text}")

    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


# ── 4. OpenAI API (GPT-4o / GPT-4o-mini) ────────────────────────────────────────

def generate_with_openai(prompt: str, api_key: str, model_name: str = "gpt-4o-mini") -> str:
    """
    Call OpenAI standard Chat Completions endpoint.
    """
    if not api_key:
        raise ValueError("OpenAI API Key is required.")

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are an expert AI research assistant. Provide thorough, grounded, and structured technical insights based on the provided context."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 1024,
    }

    response = requests.post(url, json=payload, headers=headers, timeout=45)
    if response.status_code != 200:
        raise RuntimeError(f"OpenAI API Error ({response.status_code}): {response.text}")

    data = response.json()
    return data["choices"][0]["message"]["content"].strip()