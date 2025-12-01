# TCGA-Ovarian-MultiOmics-DeepLearning
Deep learning pipeline for **multi-omics integration** on **TCGA ovarian cancer (OV)**, inspired by 
[UNet-LIDC-Segmentation](https://github.com/carloshachi777/UNet-LIDC-Segmentation).

This repository provides:
- Multi-omics harmonization (RNA-Seq, CNV, methylation)
- Variational Autoencoder (VAE) for dimensionality reduction
- Supervised deep classifier for survival / subtype prediction
- Latent-space visualization (UMAP / t-SNE)
- Fully reproducible **Google Colab notebook**

---

## 📌 Project Overview

We integrate **three TCGA omics layers**:

| Omics Layer | Features |
|-------------|----------|
| RNA-Seq / expression | ~20,000 genes |
| CNV | segment means |
| DNA methylation | CpG sites |

Steps:
1. Download from **UCSC Xena**
2. Match samples by TCGA barcode
3. Normalize + scale each omics type
4. Build VAE for dimensionality reduction
5. Concatenate latent embeddings
6. Train classifier (MLP or transformer)
7. Evaluate model performance + visualizations

---

## 📁 Repository Structure


