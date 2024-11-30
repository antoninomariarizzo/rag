import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from typing import List, Union


class LargeLanguageModel:
    """ Language Model to encode and decode text. """

    def __init__(self,
                 model_id,
                 use_fp16: bool = True):
        """
        Initialize the LanguageModel with a given model ID.

        Parameters:
        - model_id: Hugging Face model identifier.
        - use_fp16: Use half-precision (float16) if supported by the device.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        torch_dtype = torch.float16 if use_fp16 and self.device == "cuda" else torch.float32
        # Load models for encoding and decoding
        self.encoder_model = AutoModel.from_pretrained(model_id,
                                                       torch_dtype=torch_dtype
                                                       ).to(self.device)

        self.decoder_model = AutoModelForCausalLM.from_pretrained(model_id,
                                                                  torch_dtype=torch_dtype).to(self.device)

    def encode(self,
               texts: Union[str, List[str]],
               max_length: int = 512
               ) -> torch.Tensor:
        """
        Encode text(s) into vector embeddings.

        Parameters:
        - texts: String(s) to encode.
        - max_length: Maximum token length for truncation.

        Returns:
        - embedding: A tensor representing the embeddings of the sentence.
        """

        # Tokenize the input texts
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(self.device)

        # Inference (forward pass of the network)
        with torch.no_grad():
            outputs = self.encoder_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )

        # Extract embedding of the last hidden state
        # Shape: (batch_size, seq_len, hidden_size)
        embeddings = outputs.last_hidden_state

        # Mask that indicates which tokens in the input are real tokens and which are padding tokens
        mask = inputs["attention_mask"].unsqueeze(-1)

        # Average embedding along the seq_len dimension,
        # filtering out the padding tokens
        # Shape: (batch_size, hidden_size)
        embeddings = embeddings * mask
        embedding = embeddings.sum(dim=1) / mask.sum(dim=1)

        return embedding

    def decode(self,
               prompt: Union[str, List[str]],
               max_new_tokens: int = 300,
               temperature: float = 0.3
               ) -> str:
        """
        Generate text based on a given prompt.

        Parameters:
        - prompt: The input prompt(s) for text generation.
        - max_new_tokens: Maximum number of tokens to generate.
        - temperature: Scaling factor to control the randomness of the model's predictions.

        Returns:
        - answer: The generated text as a string.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        outputs = self.decoder_model.generate(
            **inputs,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id
        )

        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return answer
