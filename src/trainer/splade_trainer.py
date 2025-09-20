"""
Vietnamese SPLADE Trainer

A wrapper class for training SPLADE models on Vietnamese legal documents.
Integrates with VNLegalDataset and provides easy-to-use training interface.
"""

import logging
from typing import Optional, Dict, Any, Literal, List
from pathlib import Path

from datasets import Dataset
from sentence_transformers import (
    SparseEncoder,
    SparseEncoderModelCardData,
    SparseEncoderTrainer,
    SparseEncoderTrainingArguments,
)
from sentence_transformers.sparse_encoder.evaluation import SparseNanoBEIREvaluator
from sentence_transformers.sparse_encoder.losses import SparseMultipleNegativesRankingLoss, SpladeLoss
from sentence_transformers.training_args import BatchSamplers

from ..dataset.vn_legal_dataset import VNLegalDataset
from ..models.splade import SpladeModel


class SpladeTrainer:
    """
    Vietnamese SPLADE Trainer wrapper class.
    
    Simplifies SPLADE model training on Vietnamese legal documents with sensible defaults
    and integration with VNLegalDataset.
    """
    
    def __init__(
        self,
        model: SpladeModel,
        output_dir: str = "models/vn-legal-splade",
        run_name: Optional[str] = None,
        language: str = "vi",
        license: str = "apache-2.0"
    ):
        """
        Initialize SpladeTrainer.
        
        Args:
            model: SpladeModel instance to train
            output_dir: Directory to save trained model
            run_name: Experiment name for tracking (default: auto-generated)
            language: Model language code
            license: Model license
        """
        self.splade_model = model
        self.model = model.model  # Get the actual SparseEncoder
        self.output_dir = Path(output_dir)
        self.run_name = run_name or f"vn-legal-splade-{model.model_name_or_path.split('/')[-1]}"
        self.language = language
        self.license = license
        
        # Initialize components
        self.trainer = None
        self.dataset = None
        
        # Set up logging
        logging.basicConfig(
            format="%(asctime)s - %(message)s", 
            datefmt="%Y-%m-%d %H:%M:%S", 
            level=logging.INFO
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_model_card(self):
        """
        Setup model card data for documentation.
        """
        if hasattr(self.model, 'model_card_data'):
            self.model.model_card_data = SparseEncoderModelCardData(
                language=self.language,
                license=self.license,
                model_name=f"Vietnamese Legal SPLADE {self.splade_model.model_name_or_path} ({self.splade_model.architecture}) trained on VN Legal Documents",
            )
            self.logger.info("✅ Model card data configured")
    
    def load_dataset(
        self,
        corpus_path: str,
        test_path: Optional[str] = None,
        train_path: Optional[str] = None,
        train_split_ratio: float = 0.8,
        eval_samples: int = 1000
    ) -> Dict[str, Dataset]:
        """
        Load and prepare Vietnamese legal dataset.
        
        Args:
            corpus_path: Path to corpus CSV file
            test_path: Path to test CSV file
            train_path: Path to train CSV file  
            train_split_ratio: Ratio for train/eval split if no separate eval data
            eval_samples: Number of samples for evaluation
            
        Returns:
            Dict with 'train' and 'eval' datasets
        """
        self.logger.info("📚 Loading VN Legal Dataset...")
        
        # Initialize dataset
        self.dataset = VNLegalDataset(
            corpus_path=corpus_path,
            test_path=test_path,
            train_path=train_path
        )
        
        # Prepare training data
        datasets = {}
        
        # Use test data for training if available, otherwise use train data
        if test_path and Path(test_path).exists():
            full_dataset = self.dataset["test"]
            self.logger.info(f"📖 Using test data for training: {len(full_dataset):,} samples")
        elif train_path and Path(train_path).exists():
            full_dataset = self.dataset["train"]
            self.logger.info(f"📖 Using train data: {len(full_dataset):,} samples")
        else:
            raise ValueError("❌ No training data found! Provide either test_path or train_path")
        
        # Split into train/eval
        if len(full_dataset) > eval_samples:
            dataset_split = full_dataset.train_test_split(
                test_size=min(eval_samples, int(len(full_dataset) * (1 - train_split_ratio))),
                seed=42
            )
            datasets["train"] = dataset_split["train"]
            datasets["eval"] = dataset_split["test"]
        else:
            # Small dataset - use all for training, subset for eval
            datasets["train"] = full_dataset
            datasets["eval"] = full_dataset.select(range(min(len(full_dataset), eval_samples)))
        
        self.logger.info(f"✅ Dataset prepared:")
        self.logger.info(f"   - Train: {len(datasets['train']):,} samples")
        self.logger.info(f"   - Eval: {len(datasets['eval']):,} samples")
        
        return datasets
    
    def setup_training_args(
        self,
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 8,
        per_device_eval_batch_size: int = 16,
        learning_rate: float = 2e-5,
        sparse_embedding_lr: float = 1e-3,
        warmup_ratio: float = 0.1,
        query_regularizer_weight: float = 0,
        document_regularizer_weight: float = 3e-3,
        fp16: bool = True,
        bf16: bool = False,
        eval_steps: int = 500,
        save_steps: int = 500,
        logging_steps: int = 100,
        **kwargs
    ) -> SparseEncoderTrainingArguments:
        """
        Setup training arguments with Vietnamese legal domain defaults.
        
        Args:
            num_train_epochs: Number of training epochs
            per_device_train_batch_size: Batch size for training
            per_device_eval_batch_size: Batch size for evaluation
            learning_rate: Base learning rate
            sparse_embedding_lr: Learning rate for sparse embeddings
            warmup_ratio: Warmup ratio
            query_regularizer_weight: Query regularization weight
            document_regularizer_weight: Document regularization weight
            fp16: Use FP16 precision
            bf16: Use BF16 precision (if supported)
            eval_steps: Evaluation frequency
            save_steps: Save frequency
            logging_steps: Logging frequency
            **kwargs: Additional training arguments
            
        Returns:
            SparseEncoderTrainingArguments: Configured training arguments
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        args = SparseEncoderTrainingArguments(
            # Required parameter
            output_dir=str(self.output_dir),
            # Training parameters
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            learning_rate=learning_rate,
            learning_rate_mapping={
                r"SparseStaticEmbedding\.weight": sparse_embedding_lr
            },
            warmup_ratio=warmup_ratio,
            fp16=fp16,
            bf16=bf16,
            batch_sampler=BatchSamplers.NO_DUPLICATES,
            router_mapping={"query": "query", "document": "document"},
            # Evaluation and saving
            eval_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="steps", 
            save_steps=save_steps,
            save_total_limit=2,
            logging_steps=logging_steps,
            run_name=self.run_name,
            **kwargs
        )
        
        self.logger.info(f"⚙️ Training arguments configured for {num_train_epochs} epochs")
        return args
    
    def setup_loss(
        self,
        query_regularizer_weight: float = 0,
        document_regularizer_weight: float = 3e-3
    ) -> SpladeLoss:
        """
        Setup SPLADE loss function.
        
        Args:
            query_regularizer_weight: Query regularization weight
            document_regularizer_weight: Document regularization weight
            
        Returns:
            SpladeLoss: Configured loss function
        """
        if self.model is None:
            raise ValueError("❌ Model not built! Call build_model() first")
        
        loss = SpladeLoss(
            model=self.model,
            loss=SparseMultipleNegativesRankingLoss(model=self.model),
            query_regularizer_weight=query_regularizer_weight,
            document_regularizer_weight=document_regularizer_weight,
        )
        
        self.logger.info("💥 SPLADE loss function configured")
        return loss
    
    def create_evaluator(
        self,
        dataset_names: list = None,
        batch_size: int = 16
    ) -> Optional[SparseNanoBEIREvaluator]:
        """
        Create evaluator for model performance tracking.
        
        Args:
            dataset_names: List of evaluation datasets
            batch_size: Evaluation batch size
            
        Returns:
            SparseNanoBEIREvaluator: Evaluator or None if datasets not available
        """
        if dataset_names is None:
            # For Vietnamese legal domain, we might not have standard BEIR datasets
            # Could be extended to include Vietnamese evaluation datasets
            self.logger.warning("⚠️ No evaluation datasets specified - skipping evaluator")
            return None
        
        try:
            evaluator = SparseNanoBEIREvaluator(
                dataset_names=dataset_names,
                batch_size=batch_size
            )
            self.logger.info(f"📊 Evaluator created with datasets: {dataset_names}")
            return evaluator
        except Exception as e:
            self.logger.warning(f"⚠️ Could not create evaluator: {e}")
            return None
    
    def train(
        self,
        corpus_path: str,
        test_path: Optional[str] = None,
        train_path: Optional[str] = None,
        training_args: Optional[Dict[str, Any]] = None,
        loss_args: Optional[Dict[str, Any]] = None,
        evaluator_args: Optional[Dict[str, Any]] = None,
        push_to_hub: bool = False
    ) -> SparseEncoder:
        """
        Complete training pipeline.
        
        Args:
            corpus_path: Path to corpus CSV file
            test_path: Path to test CSV file
            train_path: Path to train CSV file
            training_args: Training arguments override
            loss_args: Loss function arguments override
            evaluator_args: Evaluator arguments override
            push_to_hub: Whether to push trained model to Hugging Face Hub
            
        Returns:
            SparseEncoder: Trained model
        """
        self.logger.info(f"🚀 Starting Vietnamese Legal SPLADE training: {self.run_name}")
        self.logger.info(f"   - Model: {self.splade_model.model_name_or_path}")
        self.logger.info(f"   - Architecture: {self.splade_model.architecture}")
        
        # 1. Setup model card
        self.setup_model_card()
        
        # 2. Load dataset
        datasets = self.load_dataset(corpus_path, test_path, train_path)
        
        # 3. Setup training arguments
        args_config = training_args or {}
        args = self.setup_training_args(**args_config)
        
        # 4. Setup loss function
        loss_config = loss_args or {}
        loss = self.setup_loss(**loss_config)
        
        # 5. Create evaluator (optional)
        eval_config = evaluator_args or {}
        evaluator = self.create_evaluator(**eval_config)
        
        # 6. Create trainer
        self.trainer = SparseEncoderTrainer(
            model=self.model,
            args=args,
            train_dataset=datasets["train"],
            eval_dataset=datasets["eval"],
            loss=loss,
            evaluator=evaluator,
        )
        
        # 7. Start training
        self.logger.info("🏃‍♂️ Starting training...")
        
        # Handle checkpoint resuming - use trainer's built-in support
        train_kwargs = {}
        if resume_checkpoint:
            train_kwargs["resume_from_checkpoint"] = resume_checkpoint
        elif auto_resume:
            # Let trainer auto-detect last checkpoint in output_dir
            train_kwargs["resume_from_checkpoint"] = True
        
        self.trainer.train(**train_kwargs)
        
        # 8. Evaluate final model
        if evaluator:
            self.logger.info("📊 Evaluating final model...")
            try:
                evaluator(self.model)
            except Exception as e:
                self.logger.warning(f"⚠️ Evaluation failed: {e}")
        
        # 9. Save final model
        final_path = self.output_dir / "final"
        self.logger.info(f"💾 Saving final model to: {final_path}")
        self.model.save_pretrained(str(final_path))
        
        # 10. Cleanup old checkpoints (optional)
        if cleanup_old_checkpoints:
            try:
                self.logger.info("🧹 Cleaning up old checkpoints...")
                self.clean_old_checkpoints(keep_last_n=2)
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to cleanup checkpoints: {e}")
        
        # 11. Push to hub (optional)
        if push_to_hub:
            try:
                self.logger.info(f"🚀 Pushing model to Hugging Face Hub: {self.run_name}")
                self.model.push_to_hub(self.run_name)
            except Exception as e:
                self.logger.error(f"❌ Failed to push to hub: {e}")
        
        self.logger.info("🎉 Training completed successfully!")
        return self.model
    
    def save_model(self, path: Optional[str] = None) -> str:
        """
        Save trained model.
        
        Args:
            path: Custom save path (optional)
            
        Returns:
            str: Path where model was saved
        """
        if self.model is None:
            raise ValueError("❌ No model to save! Train a model first")
        
        save_path = path or str(self.output_dir / "final")
        self.model.save_pretrained(save_path)
        self.logger.info(f"💾 Model saved to: {save_path}")
        return save_path
    
    def load_model(self, path: str) -> SparseEncoder:
        """
        Load trained model.
        
        Args:
            path: Path to saved model
            
        Returns:
            SparseEncoder: Loaded model
        """
        self.model = SparseEncoder.from_pretrained(path)
        self.logger.info(f"📂 Model loaded from: {path}")
        return self.model
    
    def resume_training(
        self,
        corpus_path: str,
        test_path: Optional[str] = None,
        train_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        **kwargs
    ) -> SparseEncoder:
        """
        Resume training from a specific checkpoint or the latest one.
        
        Args:
            corpus_path: Path to corpus CSV file
            test_path: Path to test CSV file
            train_path: Path to train CSV file
            checkpoint_path: Specific checkpoint path (if None, uses latest)
            **kwargs: Additional arguments for train()
            
        Returns:
            SparseEncoder: Trained model
        """
        if checkpoint_path is None:
            checkpoint_path = self.find_latest_checkpoint()
            if checkpoint_path is None:
                raise ValueError("❌ No checkpoint found to resume from!")
        
        self.logger.info(f"🔄 Resuming training from: {checkpoint_path}")
        
        # Override auto_resume to use specific checkpoint
        kwargs['auto_resume'] = False
        kwargs['training_args'] = kwargs.get('training_args', {})
        kwargs['training_args']['resume_from_checkpoint'] = checkpoint_path
        
        return self.train(
            corpus_path=corpus_path,
            test_path=test_path,
            train_path=train_path,
            **kwargs
        )
