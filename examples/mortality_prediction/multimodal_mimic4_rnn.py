"""
Example of using RNN for mortality prediction on MIMIC-IV.

This example demonstrates:
1. Loading MIMIC-IV data with relevant EHR tables
2. Applying the MortalityPredictionMIMIC4 task
3. Creating a SampleDataset with fitted processors
4. Training an RNN model for mortality prediction
5. Evaluating on a held-out test set
"""

import os
import torch

from pyhealth.datasets import (
    MIMIC4Dataset,
    get_dataloader,
    split_by_patient,
)
from pyhealth.datasets.utils import save_processors, load_processors
from pyhealth.models import RNN
from pyhealth.tasks import MortalityPredictionMIMIC4
from pyhealth.trainer import Trainer


def main():
    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------
    ENVIRONMENT = "Local"  # "Local" or "Cluster"

    if ENVIRONMENT == "Local":
        pyhealth_repo_root = "/Users/wpang/Desktop/PyHealth"
        MIMIC4_ROOT = os.path.join(
            pyhealth_repo_root,
            "local_data/local/data/physionet.org/files/mimiciv/2.2",
        )
        CACHE_DIR = os.path.join(
            pyhealth_repo_root, "local_data/local/data/wp/pyhealth_cache"
        )
        PROCESSOR_DIR = os.path.join(
            pyhealth_repo_root, "local_data/local/data/wp/processors/rnn_mortality_mimic4"
        )
    elif ENVIRONMENT == "Cluster":
        MIMIC4_ROOT = "/projects/illinois/eng/cs/jimeng/physionet.org/files/mimiciv/2.2"
        CACHE_DIR = "/u/wp14/pyhealth_cache"
        PROCESSOR_DIR = "/u/wp14/processors/rnn_mortality_mimic4"

    DEV_MODE = True  # Set to False for full dataset
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("MORTALITY PREDICTION WITH RNN ON MIMIC-IV")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # Step 1: Load MIMIC-IV base dataset
    # -------------------------------------------------------------------------
    print("\n=== Step 1: Loading MIMIC-IV Dataset ===")
    base_dataset = MIMIC4Dataset(
        ehr_root=MIMIC4_ROOT,
        ehr_tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
        cache_dir=CACHE_DIR,
        dev=DEV_MODE,
    )
    print("Dataset loaded.")

    # -------------------------------------------------------------------------
    # Step 2: Apply mortality prediction task
    # -------------------------------------------------------------------------
    print("\n=== Step 2: Applying Mortality Prediction Task ===")
    task = MortalityPredictionMIMIC4()
    print(f"Task: {task.task_name}")
    print(f"Input schema:  {task.input_schema}")
    print(f"Output schema: {task.output_schema}")

    if os.path.exists(os.path.join(PROCESSOR_DIR, "input_processors.pkl")):
        print("\nLoading pre-fitted processors...")
        input_processors, output_processors = load_processors(PROCESSOR_DIR)
        sample_dataset = base_dataset.set_task(
            task,
            input_processors=input_processors,
            output_processors=output_processors,
        )
    else:
        print("\nFitting new processors...")
        sample_dataset = base_dataset.set_task(task)
        os.makedirs(PROCESSOR_DIR, exist_ok=True)
        save_processors(sample_dataset, PROCESSOR_DIR)
        print(f"Processors saved to {PROCESSOR_DIR}")

    print(f"\nTotal samples: {len(sample_dataset)}")

    # Label distribution
    label_counts = {0: 0, 1: 0}
    for s in sample_dataset:
        label_counts[int(s["mortality"].item())] += 1
    print(f"Label distribution:")
    print(f"  Survived (0): {label_counts[0]} ({100*label_counts[0]/len(sample_dataset):.1f}%)")
    print(f"  Died     (1): {label_counts[1]} ({100*label_counts[1]/len(sample_dataset):.1f}%)")

    # Sample inspection
    sample = sample_dataset[0]
    print(f"\nSample structure (patient {sample['patient_id']}):")
    for k, v in sample.items():
        if hasattr(v, "shape"):
            print(f"  {k}: tensor {v.shape}")
        elif isinstance(v, tuple):
            print(f"  {k}: tuple of {len(v)} tensors, shapes {[t.shape for t in v]}")
        else:
            print(f"  {k}: {v}")

    # -------------------------------------------------------------------------
    # Step 3: Split and build dataloaders
    # -------------------------------------------------------------------------
    print("\n=== Step 3: Splitting Dataset ===")
    train_dataset, val_dataset, test_dataset = split_by_patient(
        sample_dataset, [0.8, 0.1, 0.1]
    )
    print(f"Train: {len(train_dataset)}  Val: {len(val_dataset)}  Test: {len(test_dataset)}")

    train_loader = get_dataloader(train_dataset, batch_size=64, shuffle=True)
    val_loader = get_dataloader(val_dataset, batch_size=64, shuffle=False)
    test_loader = get_dataloader(test_dataset, batch_size=64, shuffle=False)

    # -------------------------------------------------------------------------
    # Step 4: Initialize RNN model
    # -------------------------------------------------------------------------
    print("\n=== Step 4: Initializing RNN Model ===")
    model = RNN(
        dataset=sample_dataset,
        embedding_dim=128,
        hidden_dim=128,
        rnn_type="GRU",  # "GRU", "LSTM", or "RNN"
        num_layers=2,
        dropout=0.3,
    )
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # -------------------------------------------------------------------------
    # Step 5: Train
    # -------------------------------------------------------------------------
    print("\n=== Step 5: Training ===")
    trainer = Trainer(
        model=model,
        device=DEVICE,
        metrics=["pr_auc", "roc_auc", "accuracy", "f1"],
    )
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=50,
        monitor="roc_auc",
        optimizer_params={"lr": 1e-3},
    )

    # -------------------------------------------------------------------------
    # Step 6: Evaluate
    # -------------------------------------------------------------------------
    print("\n=== Step 6: Evaluation ===")
    results = trainer.evaluate(test_loader)
    print("\nTest Results:")
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")

    # -------------------------------------------------------------------------
    # Step 7: Sample predictions
    # -------------------------------------------------------------------------
    print("\n=== Step 7: Sample Predictions ===")
    sample_batch = next(iter(test_loader))
    with torch.no_grad():
        output = model(**sample_batch)
    print(f"Predicted probabilities: {output['y_prob'][:5]}")
    print(f"True labels:             {output['y_true'][:5]}")

    print("\n" + "=" * 60)
    print("MORTALITY PREDICTION WITH RNN COMPLETED!")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
