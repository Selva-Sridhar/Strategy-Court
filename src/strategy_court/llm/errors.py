class LLMError(Exception):
    """Base class for LLM-layer failures."""


class CacheMissError(LLMError):
    """Replay mode was requested but the request was never recorded."""


class BudgetExceededError(LLMError):
    """The configured spend limit has been reached."""


class UnpricedModelError(LLMError):
    """A live call was made to a model with no configured price (would break the cost meter)."""


class StructuredOutputError(LLMError):
    """The model failed to return schema-valid JSON within the allowed attempts."""
