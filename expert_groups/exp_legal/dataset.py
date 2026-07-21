from typing import Any

from transformers import PreTrainedTokenizerBase

from connito.shared.dataloader import DefaultStreamingTorchDataset, tokenize_windowed


# -------------------------------------------------------------
# Customer Extension Point: Customize how your dataset is loaded
# make sure this class was pointed to in the config through config.task.exp.data.dataset_class
# -------------------------------------------------------------
class StreamingTorchDataset(DefaultStreamingTorchDataset):
    @staticmethod
    def tokenize_and_format(
        example: dict[str, Any],
        tokenizer: PreTrainedTokenizerBase,
        sequence_length: int,
    ) -> dict[str, Any]:
        """
        Processes raw text for Continuous Pre-Training (CPT).
        Bypasses the chat template since Multi_Legal_Pile and C4 are raw
        text corpora, not conversational data.

        Long documents get a content-hash-derived window instead of the
        prefix: legal filings front-load their most templated text
        (headers, procedural boilerplate), so always scoring the first
        `sequence_length` tokens let template-memorizing miners reach
        near-zero eval loss. See `tokenize_windowed` for the mechanism
        and its consensus properties.
        """
        text = str(example.get("text", ""))
        toks = tokenize_windowed(text, tokenizer, sequence_length)
        return {
            "input_ids": toks["input_ids"],
            "attention_mask": toks["attention_mask"],
        }
