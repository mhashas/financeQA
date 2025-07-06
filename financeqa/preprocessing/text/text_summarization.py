import asyncio
from typing import Protocol

import torch
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

from financeqa.constants import MessageType
from financeqa.generate.hf_inference import HFChatCompletion
from financeqa.settings import hf_settings


class Summarizer(Protocol):
    def summarize(self, text: str) -> str:
        """Summarizes the given text

        Args:
            text: The text to summarize

        Returns:
            The summarized texta
        """
        ...


class HFSummarizer(Summarizer):
    def __init__(self, model_id: str = "google/flan-t5-base"):
        """Initializes the T5 summarizer

        Args:
            model_id: The huggingface model ID to use for the summarizer
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        summarizer_pipeline = pipeline(
            "summarization",
            model=model,
            tokenizer=tokenizer,
            framework="pt",
            max_length=150,
            min_length=50,
            device=device,
        )

        self.pipeline = summarizer_pipeline

    def summarize(self, text):
        """Summarizes the given text using the T5 model

        Args:
            text: the text to summarize

        Returns:
            summarized text
        """
        return self.pipeline(text)[0]["summary_text"]  # type: ignore


class HFLLamaSummarizer(Summarizer):

    class Summary(BaseModel):
        summary: str

    def __init__(self):
        """Initializes the HF Llama summarizer

        Args:
            model_id: The huggingface model ID to use for the summarizer
        """
        self.chat_completion = HFChatCompletion(hf_settings.api_key.get_secret_value())

    def summarize(self, text: str, *, model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct") -> str:
        """Summarizes the given text using the HF Llama model

        Args:
            text: the text to summarize
            model_id: model ID to use for the summarizer

        Returns:
            summarized text
        """
        messages = [
            {
                "role": MessageType.SYSTEM.value,
                "content": "Read the following text carefully and generate a concise summary that captures all the essential points, including key facts, arguments, and conclusions. The summary should include all relevant information and leave out unnecessary details or repetition. Return only the summarized version, with no additional commentary or explanations.",
            },
            {"role": MessageType.USER.value, "content": text},
        ]

        result = asyncio.run(
            self.chat_completion.get_completions(messages, model_name=model_id, response_pydantic_type=self.Summary)
        )

        return result.summary  # type: ignore
