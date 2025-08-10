# Self-Supervised Out-of-Distribution Detection with MoCo, Energy & EGS

This repository demonstrates a complete pipeline for **self-supervised pretraining** using MoCo on CIFAR-10, **fine-tuning**, and robust **out-of-distribution (OOD) detection** on SVHN.  
We benchmark three popular OOD scoring techniques—**Mahalanobis Distance, Energy Score, and Energy-Gradient Score (EGS)**—and visualize their effectiveness.

---

## 🏆 Key Results

|                | AUROC  | Detection Rate @90th | FPR@95TPR |
|:---------------|:------:|:-------------------:|:---------:|
| **Mahalanobis**| 0.258  | 0.029               | 0.997     |
| **Energy Score**| 0.911 | 0.724               | 0.345     |
| **EGS**        | 0.951  | 0.856               | 0.228     |

- **Energy/EGS scores** dramatically outperform Mahalanobis on this task.

---

## 🚀 Pipeline Overview

1. **Self-Supervised Pretraining**  
   MoCo (Momentum Contrast) pretrains a ResNet encoder on CIFAR-10 via contrastive learning.

2. **Fine-Tuning**  
   The encoder is fine-tuned for classification with a lightweight MLP head.

3. **OOD Detection**  
   Evaluate on SVHN. Compute Mahalanobis, Energy, and EGS scores for each test sample.

4. **Visualization & Analysis**  
   - Score histograms with OOD regions  
   - ROC/PR curves  
   - t-SNE feature embeddings  
   - Calibration plots  
   - Example “hard” OOD images

---

## 📊 Methods

### 1. Mahalanobis Distance  
Measures distance in feature space to the in-distribution mean:

D_M(x) = sqrt((f - μ)^T Σ⁻¹ (f - μ))


### 2. Energy Score  
Uses logsumexp of logits as an uncalibrated “confidence”:

E(x) = -T * log( ∑_c exp(z_c / T) )


### 3. Energy-Gradient Score (EGS)  
Blends normalized energy and gradient-norm information:

EGS(x) = α * Ē(x) + (1 - α) * Ĝ(x)

---

## 🖼️ Example Visualizations

- **Histograms**: Show clear OOD region separation for Energy/EGS  
- **ROC & PR Curves**: EGS > Energy > Mahalanobis  
- **t-SNE**: Visualizes ID/OOD clustering  
- **Calibration**: Model is well-calibrated on ID  
- **"Hard" OOD Cases**: Intuitive grid of most ambiguous samples

---

## 🤔 Why Mahalanobis Fails Here

- Mahalanobis assumes features are Gaussian and class-separable, but **SSL features from MoCo are not**—especially across domains (CIFAR-10 vs SVHN).
- Energy-based methods exploit **logit-space information**, capturing uncertainty more effectively for OOD detection on this dataset.
- **EGS** leverages both energy and gradient cues, giving the best separation.

---

## 📝 Final Takeaways

- **Energy and EGS** should be the default choices for OOD detection with modern self-supervised or fine-tuned encoders.
- **Mahalanobis** may fail when feature space is not Gaussian or not well-separated—especially with domain shifts.
- Visualization is critical: always inspect the score distributions and ROC/PR curves.

---

## 🔗 Citation & References

If you use this repo or find it helpful, please **cite**:

```bibtex
@software{bandyopadhyay_2025_moco_energy_egs_ood,
  author  = {Bandyopadhyay, Subhasish},
  title   = {Self-Supervised Out-of-Distribution Detection with MoCo, Energy \& EGS},
  year    = {2025},
  version = {v1.0},
  publisher = {GitHub},
  url     = {REPO_URL},        % ← replace with your repository URL
  note    = {GitHub repository},
}
