"""Service for Ollama model discovery and validation."""

import logging

import httpx

logger = logging.getLogger(__name__)

# Display name mapping: model_name -> (display_name, parameters, recommended)
MODEL_DISPLAY_NAMES: dict[str, tuple[str, str, bool]] = {
    "qwen3:8b": ("Qwen3 8B (Recommended)", "8B", True),
    "qwen3:latest": ("Qwen3 Latest", "8B", True),
    "llama3.2:latest": ("Llama 3.2", "3B", False),
    "llama3.2:3b": ("Llama 3.2 3B", "3B", False),
    "llama3.1:8b": ("Llama 3.1 8B", "8B", False),
    "deepseek-r1:32b": ("DeepSeek R1 32B", "32B", False),
    "deepseek-r1:14b": ("DeepSeek R1 14B", "14B", False),
    "deepseek-r1:8b": ("DeepSeek R1 8B", "8B", False),
    "mistral:7b": ("Mistral 7B", "7B", False),
    "mistral:latest": ("Mistral", "7B", False),
    "gemma2:9b": ("Gemma2 9B", "9B", False),
    "gemma2:2b": ("Gemma2 2B", "2B", False),
}


class ModelServiceError(Exception):
    """Base exception for model service errors."""

    pass


class OllamaUnavailableError(ModelServiceError):
    """Raised when Ollama is not reachable."""

    pass


class ModelService:
    """Service for discovering and validating Ollama models."""

    def __init__(self, ollama_url: str):
        """Initialize model service.

        Args:
            ollama_url: Base URL for Ollama API (e.g., http://localhost:11434)
        """
        self.ollama_url = ollama_url.rstrip("/")

    async def list_models(self) -> list[dict]:
        """Query Ollama for available models.

        Returns:
            List of model info dicts with keys:
            - name: Model name (e.g., "qwen3:8b")
            - display_name: Human-friendly name
            - size_gb: Size in gigabytes
            - parameters: Parameter count (e.g., "8B")
            - quantization: Quantization type if known
            - recommended: Whether this model is recommended

        Raises:
            OllamaUnavailableError: If Ollama is not reachable
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.ollama_url}/api/tags")
                response.raise_for_status()
        except httpx.ConnectError as e:
            logger.error(f"Cannot connect to Ollama at {self.ollama_url}: {e}")
            raise OllamaUnavailableError(f"Cannot connect to Ollama at {self.ollama_url}") from e
        except httpx.TimeoutException as e:
            logger.error(f"Timeout connecting to Ollama: {e}")
            raise OllamaUnavailableError("Ollama connection timed out") from e
        except httpx.HTTPStatusError as e:
            logger.error(f"Ollama returned error: {e}")
            raise OllamaUnavailableError(f"Ollama error: {e.response.status_code}") from e

        models_data = response.json().get("models", [])
        result = []

        for model in models_data:
            name = model.get("name", "")
            size_bytes = model.get("size", 0)
            size_gb = round(size_bytes / (1024**3), 2)

            # Look up display info or generate from name
            if name in MODEL_DISPLAY_NAMES:
                display_name, parameters, recommended = MODEL_DISPLAY_NAMES[name]
            else:
                # Generate display name from model name
                display_name = name.replace(":", " ").replace("-", " ").title()
                parameters = self._extract_parameters(name)
                recommended = False

            # Try to extract quantization from model details
            quantization = model.get("details", {}).get("quantization_level")

            result.append(
                {
                    "name": name,
                    "display_name": display_name,
                    "size_gb": size_gb,
                    "parameters": parameters,
                    "quantization": quantization,
                    "recommended": recommended,
                }
            )

        return result

    async def validate_model(self, model_name: str) -> bool:
        """Check if a model exists in Ollama.

        Args:
            model_name: Name of the model to validate

        Returns:
            True if model exists, False otherwise

        Raises:
            OllamaUnavailableError: If Ollama is not reachable
        """
        models = await self.list_models()
        available_names = {m["name"] for m in models}

        # Check exact match
        if model_name in available_names:
            return True

        # Check base name match (e.g., "qwen3" matches "qwen3:8b")
        for available in available_names:
            if available.startswith(model_name + ":") or available == model_name:
                return True

        return False

    def _extract_parameters(self, model_name: str) -> str | None:
        """Extract parameter count from model name.

        Args:
            model_name: Model name like "llama3.2:3b" or "mistral:7b"

        Returns:
            Parameter string like "3B" or "7B", or None if not found
        """
        # Common patterns: :3b, :7b, :8b, :14b, :32b
        import re

        match = re.search(r":(\d+)b", model_name.lower())
        if match:
            return f"{match.group(1)}B"
        return None

    @staticmethod
    def is_embedding_model(model_name: str) -> bool:
        """Check if model name indicates an embedding model.

        Args:
            model_name: Name of the model to check

        Returns:
            True if model is an embedding model (contains 'embed' in name)
        """
        return "embed" in model_name.lower()

    async def list_embedding_models(self) -> list[dict]:
        """Query Ollama for available embedding models only.

        Returns:
            List of embedding model info dicts (models with 'embed' in name)

        Raises:
            OllamaUnavailableError: If Ollama is not reachable
        """
        models = await self.list_models()
        return [m for m in models if self.is_embedding_model(m["name"])]

    async def list_chat_models(self) -> list[dict]:
        """Query Ollama for available chat/LLM models only.

        Returns:
            List of chat model info dicts (excludes embedding models)

        Raises:
            OllamaUnavailableError: If Ollama is not reachable
        """
        models = await self.list_models()
        return [m for m in models if not self.is_embedding_model(m["name"])]
