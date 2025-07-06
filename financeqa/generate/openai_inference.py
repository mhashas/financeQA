from typing import Optional, Type, TypeVar, Union

from openai import AzureOpenAI, NotGiven, OpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from pydantic import BaseModel

from financeqa.constants import MessageType, OpenAIProvider

T = TypeVar("T", bound=BaseModel)


class OpenAIChatCompletion:
    def __init__(self, client: OpenAI):
        self.client = client

    def _transform_messages_to_oai_format(self, messages: list[dict[str, str]]) -> list[ChatCompletionMessageParam]:
        """Transform messages to OpenAI format

        Args:
            messages: List of message dictionaries containing the conversation history

        Raises:
            ValueError: If role is not found in the message

        Returns:
            List of messages in OpenAI format
        """
        new_messages = []

        role_to_class_dictionary = {
            MessageType.SYSTEM.value: ChatCompletionSystemMessageParam,
            MessageType.USER.value: ChatCompletionUserMessageParam,
        }

        for message in messages:
            role = message.get("role")

            if role in role_to_class_dictionary:
                new_messages.append(role_to_class_dictionary[role](content=message["content"], role=role))
            else:
                raise ValueError(f"Unknown role in message: {message}")

        return new_messages

    async def get_completions(
        self,
        messages: list[dict[str, str]],
        *,
        model_name: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        response_pydantic_type: Optional[Type[T]] = None,
    ) -> Union[str, T]:
        """Get completions from OpenAI's chat API

        Args:
            messages: List of message dictionaries containing the conversation history
            model_name: Name of the model to use for completion
            temperature: Controls randomness in the response
            max_tokens: Maximum number of tokens in the response
            response_pydantic_type: Pydantic model class to validate and parse the response

        Raises:
            ValueError: If an error occurs while getting the completion

        Returns:
            Parsed and validated response matching the provided Pydantic model type
        """
        oai_messages = self._transform_messages_to_oai_format(messages)

        try:
            response = self.client.beta.chat.completions.parse(
                model=model_name,
                messages=oai_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_pydantic_type if response_pydantic_type is not None else NotGiven(),
            )

            raw_text = response.choices[0].message.content

            if not isinstance(raw_text, str):
                raise ValueError(f"Expected string response, got {type(raw_text)}")

            if response_pydantic_type is None:
                return raw_text

            return response_pydantic_type.model_validate_json(raw_text)
        except Exception as e:
            raise ValueError(f"Error getting completion from {model_name}: {str(e)}") from e


def build_openai_chat_completion(provider: OpenAIProvider) -> OpenAIChatCompletion:
    """Build OpenAI chat completion object

    Args:
        provider: OpenAI provider to use

    Returns:
        OpenAI chat completion object
    """

    def handle_openai():
        from financeqa.settings import openai_settings

        client = OpenAI(api_key=openai_settings.api_key.get_secret_value())

        return client

    def handle_openai_azure():
        from financeqa.settings import openai_azure_settings

        client = AzureOpenAI(
            azure_endpoint=openai_azure_settings.endpoint,
            api_key=openai_azure_settings.api_key.get_secret_value(),
            api_version=openai_azure_settings.api_version,
        )

        return client

    switch = {
        OpenAIProvider.OPENAI: handle_openai,
        OpenAIProvider.OPENAI_AZURE: handle_openai_azure,
    }

    client = switch.get(provider, lambda: "Unknown provider.")()

    return OpenAIChatCompletion(client)
