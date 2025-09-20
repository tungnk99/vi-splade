import torch
from typing import Optional, Union, Literal
from sentence_transformers import models, SparseEncoder
from sentence_transformers.models import Router

from sentence_transformers.sparse_encoder.models import MLMTransformer, SpladePooling, SparseStaticEmbedding, SparseAutoEncoder
from ._base import BaseSparseModel


class SpladeModel(BaseSparseModel):
    def __init__(
            self,
            model_name_or_path: str = NameError,
            device: Optional[Union[str, torch.device]] = None,
            architecture: Literal["splade", "inference-free", "csr"] = "splade",
            pooling_strategy: Literal["max", "sum"] = "max",
            **kwargs
        ):
        super().__init__(model_name_or_path=model_name_or_path, device=device, **kwargs)
        self.architecture = architecture
        self.pooling_strategy = pooling_strategy

    def _build_splade_model(self):
        """
        Build the default SPLADE model architecture using SparseEncoder.
        """
        # Initialize MLM Transformer (use a fill-mask model)
        mlm_transformer = MLMTransformer(self.model_name_or_path)

        # Initialize SpladePooling module
        splade_pooling = SpladePooling(pooling_strategy=self.pooling_strategy)

        # Create the Splade model
        model = SparseEncoder(modules=[mlm_transformer, splade_pooling])
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

        # Create the inference-free model
        model = SparseEncoder(modules=[router], similarity_fn_name="dot")
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
        # Create the CSR model
        model = SparseEncoder(modules=[doc_encoder, pooling, sparse_auto_encoder])

        return model
    
    def build_model(self) -> SparseEncoder:
        if self.architecture == "splade":
            return self._build_splade_model()

        if self.architecture == "inference-free":
            return self._build_ifsplade_model()
        
        if self.architecture == "csr":
            return self._build_csr_splade_model()
    
        return self._build_splade_model()