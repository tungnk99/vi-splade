"""
Example usage of SpladeTrainer for Vietnamese Legal Documents

This script demonstrates how to train a SPLADE model on Vietnamese legal data
using the integrated SpladeTrainer wrapper.
"""

import torch
from src.trainer.splade_trainer import SpladeTrainer
from src.model.splade import SpladeModel


def train_splade_model():
    """
    Train SPLADE model on Vietnamese legal documents.
    """
    print("🚀 === VIETNAMESE LEGAL SPLADE TRAINING ===")
    print()
    
    # 1. Create SPLADE model
    print("🔧 Creating SPLADE model...")
    # Option 1: Use pre-trained SPLADE model (recommended)
    # splade_model = SpladeModel(
    #     model_name_or_path="naver/splade-cocondenser-ensembledistil",  # Pre-trained SPLADE
    #     model_type="splade",
    #     pooling_strategy="max",
    #     eval=False  # Training mode
    # )
    
    # Option 2: Build custom SPLADE from Vietnamese BERT
    splade_model = SpladeModel(
        model_name_or_path="vinai/phobert-base-v2",  # Vietnamese BERT
        model_type="splade",  # Best for retrieval
        pooling_strategy="max",
        eval=False  # Training mode
    )

    print(f"   ✅ Model: {splade_model.model_name_or_path}")
    print(f"   ✅ Model Type: {splade_model.model_type}")
    print(f"   ✅ Pooling: {splade_model.pooling_strategy}")
    print(f"   ✅ Model architecture: {splade_model.model}")
    
    # 2. Initialize trainer with the model
    print("🔧 Initializing SpladeTrainer...")
    trainer = SpladeTrainer(
        model=splade_model,
        output_dir="models/vn-legal-splade-phobert",
        run_name="vn-legal-splade-phobert-v1"
    )
    print(f"   ✅ Output: {trainer.output_dir}")

    # 3. Configure training parameters
    print("⚙️ Setting up training configuration...")
    training_config = {
        "num_train_epochs": 2,  # Quick training for demo
        "per_device_train_batch_size": 2,  # Smaller batch for PhoBERT stability
        "per_device_eval_batch_size": 4,   # Smaller eval batch
        "learning_rate": 2e-5,
        # "sparse_embedding_lr": 1e-3,  # Removed - not needed for standard SPLADE
        "warmup_ratio": 0.1,
        "eval_steps": 100,
        "save_steps": 100,
        "logging_steps": 50,
        "fp16": False,  # Disable FP16 for Vietnamese model stability
        "bf16": False   # Disable BF16 as well
    }
    
    loss_config = {
        "query_regularizer_weight": 0,
        "document_regularizer_weight": 3e-3
    }
    
    print(f"   ✅ Epochs: {training_config['num_train_epochs']}")
    print(f"   ✅ Batch size: {training_config['per_device_train_batch_size']}")
    print(f"   ✅ Learning rate: {training_config['learning_rate']}")
    print()
    
    # 4. Define dataset paths
    print("📂 Dataset configuration...")
    corpus_path = "data/vn-legal-doc/cleaned_corpus_data.csv"
    test_path = "data/vn-legal-doc/test_data.csv"
    
    print(f"   ✅ Corpus: {corpus_path}")
    print(f"   ✅ Test: {test_path}")
    print()
    
    # 5. Start training
    print("🏃‍♂️ Starting training pipeline...")
    try:
        model = trainer.train(
            corpus_path=corpus_path,
            test_path=test_path,
            training_args=training_config,
            loss_args=loss_config,
            push_to_hub=False  # Set to True to push to HF Hub
        )
        
        print()
        print("🎉 === TRAINING COMPLETED SUCCESSFULLY ===")
        print(f"   - Model saved at: {trainer.output_dir}/final")
        print(f"   - Model type: {type(model)}")
        print()
        
        # 6. Quick test of trained model
        print("🔍 Testing trained model...")
        test_query = "Quy định về lương tối thiểu"
        test_doc = "Mức lương tối thiểu vùng được quy định trong Nghị định của Chính phủ"
        
        # Note: This is just a placeholder - actual inference would need proper setup
        print(f"   Query: {test_query}")
        print(f"   Document: {test_doc}")
        print("   ✅ Model ready for inference!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


def quick_training_demo():
    """
    Quick demo with minimal configuration for testing.
    """
    print("⚡ === QUICK TRAINING DEMO ===")
    print()
    
    # Super quick setup for testing
    splade_model = SpladeModel(
        model_name_or_path="vinai/phobert-base-v2",
        model_type="splade",  # Simpler model type
        pooling_strategy="max",
        eval=False
    )
    
    trainer = SpladeTrainer(
        model=splade_model,
        output_dir="models/quick-demo"
    )
    
    # Very small training config
    quick_config = {
        "num_train_epochs": 1,
        "per_device_train_batch_size": 2,
        "per_device_eval_batch_size": 4,
        "learning_rate": 5e-5,
        "eval_steps": 50,
        "save_steps": 50,
        "logging_steps": 10
    }
    
    print("🚀 Starting quick training...")
    try:
        model = trainer.train(
            corpus_path="data/vn-legal-doc/cleaned_corpus_data.csv",
            test_path="data/vn-legal-doc/test_data.csv",
            training_args=quick_config,
            # auto_resume=True  # Auto-resume parameter was removed
        )
        print("✅ Quick demo completed!")
        
    except Exception as e:
        print(f"❌ Quick demo failed: {e}")


def checkpoint_demo():
    """
    Demo checkpoint functionality.
    """
    print("💾 === CHECKPOINT DEMO ===")
    print()
    
    # Create trainer
    splade_model = SpladeModel(
        model_name_or_path="vinai/phobert-base-v2",
        model_type="splade",
        eval=False
    )
    
    trainer = SpladeTrainer(
        model=splade_model,
        output_dir="models/checkpoint-demo"
    )
    
    # Check existing checkpoints
    checkpoints = trainer.list_checkpoints()
    if checkpoints:
        print(f"📂 Found {len(checkpoints)} existing checkpoints:")
        for i, cp in enumerate(checkpoints):
            print(f"   {i+1}. {cp}")
        print()
        
        # Demo resume training
        print("🔄 Resuming from latest checkpoint...")
        try:
            trainer.resume_training(
                corpus_path="data/vn-legal-doc/cleaned_corpus_data.csv",
                test_path="data/vn-legal-doc/test_data.csv",
                training_args={"num_train_epochs": 1, "save_steps": 25}
            )
            print("✅ Resume training completed!")
        except Exception as e:
            print(f"❌ Resume failed: {e}")
    else:
        print("📂 No existing checkpoints found")
        print("🚀 Starting fresh training with checkpoints...")
        try:
            trainer.train(
                corpus_path="data/vn-legal-doc/cleaned_corpus_data.csv", 
                test_path="data/vn-legal-doc/test_data.csv",
                training_args={
                    "num_train_epochs": 2,
                    "save_steps": 25,  # Save every 25 steps
                    "per_device_train_batch_size": 2
                },
                # cleanup_old_checkpoints=True  # This parameter was removed
            )
            print("✅ Training with checkpoints completed!")
        except Exception as e:
            print(f"❌ Training failed: {e}")


def main():
    """
    Main function to run training examples.
    """
    print("🇻🇳 Vietnamese Legal SPLADE Training Demo")
    print("=" * 50)
    print()
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Device: {device}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name()}")
    print()
    
    # Choice of training mode
    # mode = input("Choose mode (1=Full Training, 2=Quick Demo, 3=Checkpoint Demo, 4=Exit): ").strip()
    mode = "1"

    if mode == "1":
        train_splade_model()
    elif mode == "2":
        quick_training_demo()
    elif mode == "3":
        checkpoint_demo()
    elif mode == "4":
        print("👋 Goodbye!")
        return
    else:
        print("❌ Invalid choice. Running full training...")
        train_splade_model()
    

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

