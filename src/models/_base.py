from typing import Optional, Union

import torch
from sentence_transformers import SparseEncoder


class BaseSparseModel:
    """Abstract base class for sparse embedding model"""

    def __init__(
            self,
            model_name_or_path: str = NameError,
            device: Optional[Union[str, torch.device]] = None,
            eval: bool = True,
            **kwargs
        ):
        if not device:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)
        
        self.device = device
        self.eval_mode = eval

        self.model_name_or_path = model_name_or_path

        if self.eval_mode:
            self.model = SparseEncoder.from_pretrained(self.model_name_or_path)
        else:
            self.model = self.build_model()

        self.model.to(self.device)

        if self.eval_mode:
            self.model.eval()

    def build_model(self):
        """Implement model architec"""
        raise NotImplementedError("You must implement model architect in function build_model().")
    
    def encode_batch(self, sentences: list[str], batch_size: int = 16, convert_to_sparse_tensor: bool = True, **kwargs):
        return self.model.encode(sentences, batch_size=batch_size, convert_to_sparse_tensor=convert_to_sparse_tensor)
    
    def encode(self, sentence: str, convert_to_sparse_tensor: bool = True, **kwargs):
        results = self.model.encode([sentence], convert_to_sparse_tensor=convert_to_sparse_tensor)
        return results[0]
    
    def save(self, model_path: str, model_name: str = None, **kwargs):
        self.model.save_pretrained(model_path, model_name=model_name)

    def load(self, model_path: str, **kwargs):
        """
        Load weights from a checkpoint into an already initialized model.
        """
        if self.model is None:
            # Safety check: ensure the architecture has been built before loading weights
            raise RuntimeError("Model is not initialized. Please build or create the model first.")

        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)

        if self.eval_mode:
            self.model.eval()

        return self
