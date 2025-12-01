import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, roc_auc_score

from src.dataloaders.multiomics_dataset import MultiOmicsDataset
from src.models.vae_encoder import MultiOmicsVAE
from src.models.multiomics_classifier import MultiOmicsClassifier
from src.utils.preprocessing import (
    load_omics_from_csv,
    align_omics_by_samples,
    zscore_per_sample,
    scale_omics,
    load_labels,
    align_labels_with_omics,
)
from src.utils.integration import train_val_test_split_indices, split_arrays_by_indices


def parse_args():
    parser = argparse.ArgumentParser(description="Train Multi-Omics VAE + Classifier")
    parser.add_argument("--rna_csv", type=str, required=True)
    parser.add_argument("--cnv_csv", type=str, required=True)
    parser.add_argument("--meth_csv", type=str, required=True)
    parser.add_argument("--labels_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results")

    parser.add_argument("--epochs_vae", type=int, default=100)
    parser.add_argument("--epochs_clf", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr_vae", type=float, default=1e-3)
    parser.add_argument("--lr_clf", type=float, default=1e-3)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--beta", type=float, default=1.0)  # beta-VAE coefficient
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()


def train_vae(model, loader, epochs, lr, beta, device):
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for batch in loader:
            x_rna = batch["rna"].to(device)
            x_cnv = batch["cnv"].to(device)
            x_meth = batch["meth"].to(device)

            optimizer.zero_grad()
            out = model(x_rna, x_cnv, x_meth)
            loss, components = model.loss_function(
                x_rna,
                x_cnv,
                x_meth,
                out["x_rna_hat"],
                out["x_cnv_hat"],
                out["x_meth_hat"],
                out["mu"],
                out["logvar"],
                beta=beta,
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x_rna.size(0)

        avg_loss = total_loss / len(loader.dataset)
        print(f"[VAE] Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}")


def extract_latent(model, loader, device):
    model.to(device)
    model.eval()

    zs = []
    ys = []

    with torch.no_grad():
        for batch in loader:
            x_rna = batch["rna"].to(device)
            x_cnv = batch["cnv"].to(device)
            x_meth = batch["meth"].to(device)

            out = model(x_rna, x_cnv, x_meth)
            z = out["z"].cpu().numpy()
            zs.append(z)

            if "label" in batch:
                ys.append(batch["label"].cpu().numpy())

    Z = np.concatenate(zs, axis=0)
    y = np.concatenate(ys, axis=0) if ys else None
    return Z, y


def train_classifier(Z_train, y_train, Z_val, y_val, num_classes, epochs, lr, device):
    Z_train_t = torch.tensor(Z_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.long, device=device)

    Z_val_t = torch.tensor(Z_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.long, device=device)

    clf = MultiOmicsClassifier(input_dim=Z_train.shape[1], num_classes=num_classes)
    clf.to(device)
    optimizer = torch.optim.Adam(clf.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        clf.train()
        optimizer.zero_grad()
        logits = clf(Z_train_t)
        loss = clf.loss_function(logits, y_train_t)
        loss.backward()
        optimizer.step()

        clf.eval()
        with torch.no_grad():
            logits_val = clf(Z_val_t)
            preds_val = torch.argmax(logits_val, dim=1)
            acc_val = (preds_val == y_val_t).float().mean().item()

        print(f"[CLF] Epoch {epoch}/{epochs} - Train loss: {loss.item():.4f} - Val acc: {acc_val:.4f}")

    return clf


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = args.device
    print(f"Using device: {device}")

    # 1. Load omics
    rna, cnv, meth = load_omics_from_csv(args.rna_csv, args.cnv_csv, args.meth_csv)
    rna, cnv, meth = align_omics_by_samples(rna, cnv, meth)

    # 2. Normalize / z-score if you want
    rna = zscore_per_sample(rna, winsorize=True)
    cnv = zscore_per_sample(cnv, winsorize=True)
    meth = zscore_per_sample(meth, winsorize=True)

    # 3. Scale each omics (samples x features)
    X_rna, X_cnv, X_meth, scalers = scale_omics(rna, cnv, meth)

    # 4. Labels
    labels = load_labels(args.labels_csv)
    y = align_labels_with_omics(labels, rna)

    n_samples = X_rna.shape[0]
    print(f"Total samples (after alignment): {n_samples}")

    # 5. Train/val/test split
    splits = train_val_test_split_indices(n_samples=n_samples)
    data_splits = split_arrays_by_indices(X_rna, X_cnv, X_meth, y, splits)

    # 6. DataLoaders for VAE (we use only train split for unsupervised training)
    ds_train_vae = MultiOmicsDataset(
        data_splits["train"]["rna"],
        data_splits["train"]["cnv"],
        data_splits["train"]["meth"],
        y=None,
        device=device,
    )
    loader_train_vae = DataLoader(ds_train_vae, batch_size=args.batch_size, shuffle=True)

    # 7. Initialize VAE
    dim_rna = X_rna.shape[1]
    dim_cnv = X_cnv.shape[1]
    dim_meth = X_meth.shape[1]

    vae = MultiOmicsVAE(
        dim_rna=dim_rna,
        dim_cnv=dim_cnv,
        dim_meth=dim_meth,
        latent_dim=args.latent_dim,
    )

    # 8. Train VAE
    train_vae(vae, loader_train_vae, epochs=args.epochs_vae, lr=args.lr_vae, beta=args.beta, device=device)

    # Save VAE
    torch.save(vae.state_dict(), os.path.join(args.output_dir, "vae.pth"))

    # 9. Extract latent embeddings for train/val/test
    loaders = {}
    Z = {}
    y_latent = {}

    for split_name in ["train", "val", "test"]:
        ds = MultiOmicsDataset(
            data_splits[split_name]["rna"],
            data_splits[split_name]["cnv"],
            data_splits[split_name]["meth"],
            y=data_splits[split_name]["y"],
            device=device,
        )
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
        loaders[split_name] = loader

        Z_split, y_split = extract_latent(vae, loader, device)
        Z[split_name] = Z_split
        y_latent[split_name] = y_split

        np.save(os.path.join(args.output_dir, f"Z_{split_name}.npy"), Z_split)
        np.save(os.path.join(args.output_dir, f"y_{split_name}.npy"), y_split)

    # 10. Train classifier on latent space
    num_classes = len(np.unique(y_latent["train"]))
    clf = train_classifier(
        Z_train=Z["train"],
        y_train=y_latent["train"],
        Z_val=Z["val"],
        y_val=y_latent["val"],
        num_classes=num_classes,
        epochs=args.epochs_clf,
        lr=args.lr_clf,
        device=device,
    )

    torch.save(clf.state_dict(), os.path.join(args.output_dir, "classifier.pth"))

    # 11. Evaluate on test set
    clf.eval()
    Z_test_t = torch.tensor(Z["test"], dtype=torch.float32, device=device)
    y_test_t = torch.tensor(y_latent["test"], dtype=torch.long, device=device)

    with torch.no_grad():
        logits_test = clf(Z_test_t)
        preds_test = torch.argmax(logits_test, dim=1)
        acc_test = (preds_test == y_test_t).float().mean().item()

        y_true = y_latent["test"]
        y_prob = torch.softmax(logits_test, dim=1).cpu().numpy()

        print(f"Test accuracy: {acc_test:.4f}")

        # AUC (only if binary)
        if num_classes == 2:
            auc = roc_auc_score(y_true, y_prob[:, 1])
            print(f"Test AUC: {auc:.4f}")


if __name__ == "__main__":
    main()
