from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
import torch

def load_model(model_id: str = "llava-hf/llava-1.5-7b-hf",
               quant_4bit: bool = True,
               dtype: str = "float16"):
    quant_config = None
    if quant_4bit:
        quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype)

    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor
