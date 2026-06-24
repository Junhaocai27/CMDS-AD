# CMDS-AD: Cross-Modal Dual-Stream Decoupling for Few-Shot Anomaly Detection

**[ECCV 2026]** Official implementation.

## Project Page
https://cmds-ad.github.io/

## 🚨 Code Coming Soon

The source code, including training and evaluation scripts, is currently being cleaned up and **will be released in a few days**. Please stay tuned!

## Abstract

Few-shot anomaly detection remains challenging due to limited training data. Multi-modal anomaly detection (MAD) offers a viable solution, leveraging 3D geometric cues to enrich 2D RGB representations and compensate for this scarcity. However, existing MAD methods apply spatially uniform feature processing, conflating stable macroscopic structures with high-frequency localized defect signals, exacerbating cross-modal misalignment and inflating false-positive rates. To overcome this, we present CMDS-AD, a Cross-Modal Dual-Stream Anomaly Detection framework. A LoRA-guided diffusion model generates diverse RGB samples to mitigate extreme data scarcity. For 3D normal augmentation, we employ a pre-trained diffusion model as a normal estimator. Crucially, this estimator inherently acts as a non-linear low-pass filter, directly extracting low-frequency normal representations from RGB inputs. This establishes an auxiliary estimated stream of purely low-frequency information, anchoring robust structural templates and assisting the uncompressed real stream, containing coupled high- and low-frequency components, to precisely isolate micro-defects. A Coordinate-Aware Hierarchical Feature Mapper adaptively aligns cross-modal semantics, while a multiplicative scoring mechanism filters modality-specific noise. Under the extreme 1-shot setting, CMDS-AD achieves absolute performance gains of **5.7%** (I-AUROC) and **2.0%** (AUPRO) on MVTec 3D-AD, alongside **7.7%** and **5.6%** improvements on EyeCandies, establishing a new state-of-the-art.
