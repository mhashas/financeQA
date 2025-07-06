from typing import Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class BaseInferenceClient(Protocol):

    async def get_completions(
        self,
        messages: list[dict[str, str]],
        **kwargs,
    ):
        """Get completions from an inference API

        Args:
            messages: List of message dictionaries containing the conversation history
            kwargs: Additional arguments to pass to the inference API
        """
        pass
