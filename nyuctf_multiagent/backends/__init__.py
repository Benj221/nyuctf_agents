from .openai_backend import OpenAIBackend
from .anthropic_backend import AnthropicBackend
from .together_backend import TogetherBackend
from .gemini_backend import GeminiBackend
from .backend import Role

BACKENDS = [OpenAIBackend, AnthropicBackend, TogetherBackend, GeminiBackend]
MODELS = {m: b for b in BACKENDS for m in b.MODELS}

# Add llamaswap models
LLAMASWAP_MODELS = [
    "qwen-rag",
    "qwen-abliterated",
    "qwen-fast",
    "qwen-expert",
    "glm4-air",
]
for model in LLAMASWAP_MODELS:
    MODELS[model] = OpenAIBackend
