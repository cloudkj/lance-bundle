import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

_ONNX_OPSET_VERSION = 18

class SentenceTransformerONNXWrapper(nn.Module):
    """
    Custom wrapper for SentenceTransformer to support custom ONNX export that includes pooling and normalization
    See: https://github.com/huggingface/sentence-transformers/issues/3258
    """
    def __init__(self, model: SentenceTransformer):
        super().__init__()
        self._model = model

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # TODO: validate these are the appropriate steps, and add proper
        # documentation pointers for these "magic" structs

        # 1. Reconstruct the dictionary format SentenceTransformers expects
        features = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            features["token_type_ids"] = token_type_ids

        # 2. Pass through the official pipeline.
        # This executes their exact Pooling and Normalization layers.
        features = self._model(features)

        # 3. Return only the final fused vector payload
        return features["sentence_embedding"]

def save_model(model: SentenceTransformer, model_path: str, model_onnx_path: str):
    # Instantiate and set wrapper to evaluation mode
    wrapper = SentenceTransformerONNXWrapper(model)
    wrapper.eval()

    # Save the tokenizer directly to the ONNX directory for runtime use
    tokenizer = wrapper._model.tokenizer
    tokenizer.save_pretrained(model_path)

    # Generate dummy text to capture token shapes for tracing
    inputs = tokenizer(["DUMMY INPUT"], return_tensors="pt", padding=True, truncation=True)

    # Setup input and dynamic axes configuration
    input_names = ["input_ids", "attention_mask"]
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "embeddings": {0: "batch_size"}
    }

    # Prepare arguments dynamically based on whether architecture uses token_type_ids
    input_args = (inputs["input_ids"], inputs["attention_mask"])
    if "token_type_ids" in inputs:
        input_args = (*input_args, inputs["token_type_ids"])
        input_names.append("token_type_ids")
        dynamic_axes["token_type_ids"] = {0: "batch_size", 1: "sequence_length"}

    # Export the fused computational graph
    torch.onnx.export(
        wrapper,
        input_args,
        model_onnx_path,
        input_names=input_names,
        output_names=["embeddings"],
        dynamic_axes=dynamic_axes,
        # TODO: what is this?
        opset_version=_ONNX_OPSET_VERSION
    )
