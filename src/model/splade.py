import torch
from typing import Optional, Union, Literal
from sentence_transformers import models, SparseEncoder, SparseEncoderModelCardData
from sentence_transformers.models import Router

from sentence_transformers.sparse_encoder.models import MLMTransformer, SpladePooling, SparseStaticEmbedding, SparseAutoEncoder
from ._base import BaseSparseModel


class SpladeModel(BaseSparseModel):
    def __init__(
            self,
            model_name_or_path: str,
            device: Optional[Union[str, torch.device]] = None,
            model_type: Literal["splade", "inference-free", "csr"] = "splade",
            pooling_strategy: Literal["max", "sum"] = "max",
            language: str = "vi",
            license: str = "apache-2.0",
            **kwargs
        ):
        # Set instance attributes BEFORE calling super().__init__()
        # because build_model() is called in parent __init__ and needs these attributes
        self.model_type = model_type
        self.pooling_strategy = pooling_strategy
        self.language = language
        self.license = license
        super().__init__(model_name_or_path=model_name_or_path, device=device, **kwargs)

    def _build_splade_model(self):
        """
        Build the default SPLADE model architecture using SparseEncoder.
        """
        # Initialize MLM Transformer (use a fill-mask model)
        mlm_transformer = MLMTransformer(self.model_name_or_path)

        # Initialize SpladePooling module
        splade_pooling = SpladePooling(pooling_strategy=self.pooling_strategy)

        model_card = self._get_model_card()
        # Create the Splade model
        model = SparseEncoder(modules=[mlm_transformer, splade_pooling], model_card_data=model_card)
        
        # Set appropriate max_seq_length for Vietnamese models
        if "phobert" in self.model_name_or_path.lower():
            model.max_seq_length = 256  # PhoBERT typically supports up to 512, but 256 is safer
        else:
            model.max_seq_length = 512  # Default for other models

        return model
    
    def _build_ifsplade_model(self):
        """
        Build the Inference-free SPLADE model architecture using SparseEncoder.
        """
        # Initialize MLM Transformer for document encoding
        doc_encoder = MLMTransformer(self.model_name_or_path)

        # Create a router model with different paths for queries and documents
        router = Router.for_query_document(
            query_modules=[SparseStaticEmbedding(tokenizer=doc_encoder.tokenizer, frozen=False)],
            # Document path: full MLM transformer + pooling
            document_modules=[doc_encoder, SpladePooling(self.pooling_strategy)],
        )

        model_card = self._get_model_card()

        # Create the inference-free model
        model = SparseEncoder(modules=[router], similarity_fn_name="dot", model_card_data=model_card)
        
        # Set appropriate max_seq_length for Vietnamese models
        if "phobert" in self.model_name_or_path.lower():
            model.max_seq_length = 256  # PhoBERT typically supports up to 512, but 256 is safer
        else:
            model.max_seq_length = 512  # Default for other models

        return model
    
    def _build_csr_splade_model(self):
        """
        Build the CSR SPLADE model architecture using SparseEncoder.
        """
        # Initialize MLM Transformer for document encoding
        doc_encoder = MLMTransformer(self.model_name_or_path)

        # Initialize pooling
        pooling = models.Pooling(doc_encoder.get_word_embedding_dimension(), pooling_mode="mean")

        # Initialize SparseAutoEncoder module
        sparse_auto_encoder = SparseAutoEncoder(
            input_dim=doc_encoder.get_word_embedding_dimension(),
            hidden_dim=4 * doc_encoder.get_word_embedding_dimension(),
            k=256,  # Number of top values to keep
            k_aux=512,  # Number of top values for auxiliary loss
        )

        model_card = self._get_model_card()
        # Create the CSR model
        model = SparseEncoder(modules=[doc_encoder, pooling, sparse_auto_encoder], model_card_data=model_card)
        
        # Set appropriate max_seq_length for Vietnamese models
        if "phobert" in self.model_name_or_path.lower():
            model.max_seq_length = 256  # PhoBERT typically supports up to 512, but 256 is safer
        else:
            model.max_seq_length = 512  # Default for other models

        return model
    
    def build_model(self) -> SparseEncoder:
        if self.model_type == "splade":
            return self._build_splade_model()

        if self.model_type == "inference-free":
            return self._build_ifsplade_model()
        
        if self.model_type == "csr":
            return self._build_csr_splade_model()
    
        return self._build_splade_model()
    
    def _get_model_card(self):
        """
        Setup model card data for documentation.
        
        Args:
            model: The SparseEncoder model to setup card data for
        """

        model_card_data = SparseEncoderModelCardData(
            language=self.language,
            license=self.license,
            model_name=f"Vietnamese Legal SPLADE {self.model_name_or_path} ({self.model_type}) trained on VN Legal Documents",
        )

        return model_card_data