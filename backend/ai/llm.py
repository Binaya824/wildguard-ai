"""
OpenAI connection and configuration
Environment-variable based (no Django)
"""

import os
from dotenv import load_dotenv
from typing import Optional
from openai import OpenAI

# Load environment variables
load_dotenv()

# ===== Load Config From .env =====
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

# ===== Configure OpenAI Client =====
try:
    if API_KEY:
        openai_client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        print("OpenAI client configured")
    else:
        print("Warning: OPENAI_API_KEY not found in environment variables")
        openai_client = None
except Exception as e:
    print(f"Warning: OpenAI client init failed: {e}")
    openai_client = None


def get_model(model_name: str = MODEL_NAME):
    """
    Dummy wrapper kept for compatibility.
    """
    if not API_KEY or not openai_client:
        raise ValueError("API_KEY not configured or OpenAI client unavailable")
    return model_name


async def generate_text(
    prompt: str,
    model_name: str = MODEL_NAME,
    temperature: float = 0.7,
    max_tokens: int = 1000
) -> str:
    """
    Generate text using OpenAI model
    """
    try:
        model = get_model(model_name)
        response = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content

    except Exception as e:
        print(f"Error generating text: {e}")
        raise


async def generate_text_with_json(
    prompt: str,
    model_name: str = MODEL_NAME,
    temperature: float = 0.3
) -> str:
    """
    Generate JSON output using OpenAI response_format
    """
    try:
        model = get_model(model_name)
        response = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content

    except Exception as e:
        print(f"Error generating JSON: {e}")
        raise


async def analyze_text_batch(
    texts: list[str],
    prompt_template: str,
    model_name: str = MODEL_NAME
) -> list[str]:
    """
    Analyze multiple texts using same prompt template
    """
    results = []

    for text in texts:
        try:
            prompt = prompt_template.format(text=text)
            result = await generate_text(prompt, model_name)
            results.append(result)
        except Exception as e:
            print(f"⚠️ Error analyzing text: {e}")
            results.append(None)

    return results


def check_api_key() -> bool:
    """
    Check if API key is configured
    """
    return API_KEY is not None and API_KEY != ""


# ===== Prompt Templates (unchanged) =====

ENTITY_EXTRACTION_PROMPT = """
You are an AI assistant specialized in wildlife smuggling analysis.

Extract the following information from the text below:
1. Animal species mentioned (only actual animal species, not products like skin, scales, horn, etc.)
2. Specific location details (city, state, country)
3. Key entities (people, organizations, vehicles)
4. Important keywords

Text: {text}

Return the information in this JSON format:
{{
    "animals": ["animal1", "animal2"],
    "location": "specific location",
    "entities": ["entity1", "entity2"],
    "keywords": ["keyword1", "keyword2"]
}}
"""

SUMMARY_PROMPT = """
You are an AI assistant specialized in wildlife smuggling analysis.

Write a brief 2-3 sentence summary of the following wildlife smuggling incident.
Focus on: what was seized, where, and the outcome.

Text: {text}

Summary:
"""

PATTERN_ANALYSIS_PROMPT = """
You are an AI assistant specialized in wildlife smuggling analysis.

Analyze the following incidents and identify patterns:
- Common smuggling routes
- Frequently smuggled animals/products
- Typical modus operandi
- Temporal patterns

Incidents:
{incidents}

Provide a detailed analysis:
"""
