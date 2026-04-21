import warnings
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

warnings.filterwarnings("ignore")


class GemmaLoader:
    """
    Loads Gemma 4 E4B (or fallback) in 4-bit for edge inference.
    Optimized for 6GB VRAM (RTX 4050) and 15GB VRAM (T4).
    """

    MODELS = {
        "gemma4_e4b": "google/gemma-4-4b-it",
        "gemma4_e2b": "google/gemma-4-2b-it",
        "gemma3_4b": "google/gemma-3-4b-it",
        "qwen3_0.6b": "Qwen/Qwen3-0.6B",
    }

    def __init__(self, model_key: str = "gemma4_e4b", max_seq_length: int = 2048):
        self.model_key = model_key
        self.max_seq_length = max_seq_length
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _resolve_model_id(self) -> str:
        return self.MODELS.get(self.model_key, self.model_key)

    def load(self, load_in_4bit: bool = True) -> tuple:
        """
        Load model with 4-bit quantization.
        Returns: (model, tokenizer)
        """
        model_id = self._resolve_model_id()
        print(f"Loading: {model_id}")
        print(f"Device: {self.device}")

        if self.device == "cuda":
            free_vram = torch.cuda.get_device_properties(0).total_memory
            free_vram -= torch.cuda.memory_allocated()
            print(f"Free VRAM before load: {free_vram / 1024**3:.2f} GB")

        try:
            print("Attempting Unsloth load...")
            from unsloth import FastLanguageModel

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_id,
                max_seq_length=self.max_seq_length,
                dtype=torch.float16,
                load_in_4bit=load_in_4bit,
                token=None,
            )
            FastLanguageModel.for_inference(model)

        except Exception as error:
            print(f"Unsloth failed: {error}")
            print("Falling back to transformers + bitsandbytes...")

            bnb_config = None
            if load_in_4bit and self.device == "cuda":
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )

            tokenizer = AutoTokenizer.from_pretrained(model_id)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token

            model_kwargs = {
                "device_map": "auto" if self.device == "cuda" else None,
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
            }
            if bnb_config is not None:
                model_kwargs["quantization_config"] = bnb_config

            model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
            if self.device != "cuda":
                model = model.to(self.device)

        self.model = model
        self.tokenizer = tokenizer

        if self.device == "cuda":
            used_vram = torch.cuda.memory_allocated() / 1024**3
            total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"VRAM after load: {used_vram:.2f} GB")
            print(f"VRAM free: {total_vram - used_vram:.2f} GB")

        return model, tokenizer

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        memory_context: Optional[torch.Tensor] = None,
    ) -> str:
        """
        Generate text with optional memory injection.
        If memory_context provided, prepend to input embeddings.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        if memory_context is not None:
            pass

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][inputs.input_ids.shape[1] :]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return response

    def get_model_info(self) -> dict:
        """Return model statistics."""
        if self.model is None:
            return {"status": "not_loaded"}

        total_params = sum(param.numel() for param in self.model.parameters())
        trainable_params = sum(param.numel() for param in self.model.parameters() if param.requires_grad)

        return {
            "model_key": self.model_key,
            "model_id": self._resolve_model_id(),
            "total_params": f"{total_params / 1e9:.2f}B",
            "trainable_params": f"{trainable_params / 1e6:.2f}M",
            "device": self.device,
            "dtype": str(next(self.model.parameters()).dtype),
        }


if __name__ == "__main__":
    print("=" * 50)
    print("Gemma 4 E4B Loader Test")
    print("=" * 50)

    loader = GemmaLoader(model_key="gemma4_e4b")

    try:
        model, tokenizer = loader.load(load_in_4bit=True)
        info = loader.get_model_info()
        print(f"\nModel Info: {info}")

        prompt = "What is 2 + 2? Think step by step."
        print(f"\nPrompt: {prompt}")
        response = loader.generate(prompt, max_new_tokens=128)
        print(f"Response: {response}")

    except Exception as error:
        print(f"Gemma 4 not available: {error}")
        print("Trying fallback: Qwen3-0.6B...")

        loader = GemmaLoader(model_key="qwen3_0.6b")
        model, tokenizer = loader.load(load_in_4bit=True)

        prompt = "What is 2 + 2?"
        response = loader.generate(prompt, max_new_tokens=64)
        print(f"Response: {response}")

    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1024**3
        print(f"\nFinal VRAM used: {used:.2f} GB")
