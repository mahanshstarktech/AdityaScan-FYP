# AdityaScan — Viva Voce & Defense Master Guide
## Candidate: Mahansh Gaur (Team Lead — Core AI/ML & MLOps Pipeline)
**Project Title:** *AdityaScan: Multi-Modal Solar Flare Forecasting using Aditya-L1 Telemetry*  
**Project ID:** `SKIT/IOT/2023-2027/P-04`  
**Institution:** Swami Keshvanand Institute of Technology, Management & Gramothan (SKIT), Jaipur  
**Department:** Computer Science & Engineering (IoT)  
**Academic Year:** 2026–2027  

---

## TABLE OF CONTENTS
1. [The 30-Second Elevator Pitch & 3-Minute Project Defense Introduction](#1-the-30-second-elevator-pitch--3-minute-project-defense-introduction)
2. [Candidate Ownership & Boundary Matrix (What Mahansh Gaur Built)](#2-candidate-ownership--boundary-matrix-what-mahansh-gaur-built)
3. [High-Level Project & Domain Viva Questions](#3-high-level-project--domain-viva-questions)
4. [Every Formula Used: Mathematical Formulations, Derivations & Rationale](#4-every-formula-used-mathematical-formulations-derivations--rationale)
5. [Neural Architecture Deep-Dive (Code Line-by-Line Viva)](#5-neural-architecture-deep-dive-code-line-by-line-viva)
6. [Code Written & Committed by Mahansh Gaur (File-by-File Defense)](#6-code-written--committed-by-mahansh-gaur-file-by-file-defense)
7. [Solar Flare Physics & Space Weather Domain Questions](#7-solar-flare-physics--space-weather-domain-questions)
8. [MLOps, Hardware Optimization & Production Engineering Questions](#8-mlops-hardware-optimization--production-engineering-questions)
9. [Professor "Trap Questions" & Bulletproof Rebuttals](#9-professor-trap-questions--bulletproof-rebuttals)
10. [Quick-Recall Formula & Numbers Cheat Sheet](#10-quick-recall-formula--numbers-cheat-sheet)

---

## 1. The 30-Second Elevator Pitch & 3-Minute Project Defense Introduction

### The 30-Second Pitch (When the Professor says: "Briefly introduce your project")
> *"Good morning/afternoon, Sir/Madam. My name is Mahansh Gaur. I am the Team Lead and ML/MLOps architect for **AdityaScan**.  
> Solar flares and coronal mass ejections can cause trillions of dollars of damage to orbital satellite constellations like India's NavIC and GSAT, and knock out terrestrial power grids like the 1989 Quebec blackout. Currently, NOAA SWPC human forecasters issue warnings with only a 20-to-30 minute horizon and a True Skill Statistic of around 0.50.  
> **AdityaScan** is India's first physics-informed, multi-modal deep learning system that ingests live telemetry from ISRO's Aditya-L1 spacecraft (SoLEXS, HEL1OS, MAG, ASPEX) combined with NOAA GOES and SDO SHARP data. My neural architecture uses a Dilated Causal TCN, Bidirectional LSTM, and Cross-Modal Attention with calibrated conformal prediction to forecast solar flares across **5 simultaneous time horizons** (from 30 minutes up to 24 hours) with an inference latency of **42.6 milliseconds** and an industry-leading **TSS of 0.887**."*

---

### The 3-Minute Comprehensive Defense Walkthrough (When asked for a detailed presentation)
1. **The Problem:**
   * Severe space weather storms (Carrington-class events) cause prompt solar radiation damage, single-event upsets (SEUs) in satellite electronics, and GPS degradation.
   * Existing international systems (NOAA SWPC) give short 20–30 min lead times or binary single-horizon forecasts. India operates critical orbital assets (NavIC, GSAT, INSAT, EOS, and soon Gaganyaan) without a sovereign, domestic predictive space weather system.
2. **The Innovation (Why AI + Aditya-L1?):**
   * Aditya-L1 sits at the Sun-Earth Lagrange Point 1 (1.5 million km from Earth) with an uninterrupted 24/7 view of the Sun.
   * We do not treat deep learning as a pure black box. We encode fundamental solar physics: the **Neupert Effect** ($dI_{\text{SXR}}/dt \propto I_{\text{HXR}}$), Continuous Morlet Wavelet transforms for **Quasi-Periodic Pulsations (QPP)**, and active region magnetic helicity and flux gradients.
3. **My Core Technical Contribution (Mahansh Gaur):**
   * Designed the **Multi-Modal Fusion Neural Backbone** (`fusion.py`) fusing high-cadence X-ray time series, magnetic field vectors, and in-situ plasma parameters.
   * Solved the catastrophic **1:1200 flare-to-quiet class imbalance** using Effective Number of Samples sample weighting (`balanced_sampler.py`) and Dynamic Focal Loss.
   * Engineered the **Chronological Splitter** (`chronological_split.py`) with 24-hour quarantine buffers to eliminate temporal data leakage.
   * Built a **Memory-Bounded Distributed PyTorch DataLoader** (`distributed_loader.py`) allowing 15 months of gigabyte-scale FITS telemetry to train on constrained GPU clusters (<5 GB RAM).
   * Exported the trained PyTorch network to **ONNX Opset 17 with dynamic INT8 quantization**, cutting latency from 180 ms to 42.6 ms for CPU-based real-time deployment.
4. **The Results:**
   * True Skill Statistic (TSS): **0.887** (vs. NOAA human forecaster benchmark of ~0.50).
   * Area Under ROC Curve (AUC-ROC): **0.894**.
   * Brier Score: **0.089**.
   * Conformal Uncertainty Coverage: **90% mathematical guarantee**.
   * Production Deployment: Live FastAPI backend on Render, React 18 / Vite Mission Control on Cloudflare Pages.

---

## 2. Candidate Ownership & Boundary Matrix (What Mahansh Gaur Built)

| Domain Track | Mahansh Gaur (**Me — Team Lead**) | Sanskriti Raj (Teammate) | Somya Roy (Teammate) |
| :--- | :--- | :--- | :--- |
| **Primary Responsibility** | **Core AI/ML & MLOps Pipeline** | **Satellite Ingestion & Solar Physics** | **Mission UI & Full-Stack DevOps** |
| **Form-2 Stories** | `US-ML-01` to `US-ML-06` | `US-ING-01` to `US-PHY-06` | `US-FS-01` to `US-OPS-06` |
| **Code Modules Owned** | • `backend/pipeline/ml/fusion.py`<br>• `backend/pipeline/ml/chronological_split.py`<br>• `backend/pipeline/ml/balanced_sampler.py`<br>• `backend/pipeline/ml/distributed_loader.py`<br>• `backend/pipeline/ml/uncertainty.py` (modeling)<br>• `backend/utils/metrics.py`<br>• `scripts/validate_incremental_checkpoints.py`<br>• `run_training.sh` & ONNX quantization<br>• `generate_report.py` (Form-3 automation) | • `backend/pipeline/ingestion/pradan_downloader.py`<br>• `backend/pipeline/ingestion/solexs_loader.py`<br>• `backend/pipeline/ingestion/helios_loader.py`<br>• `backend/pipeline/ingestion/goes_fetcher.py`<br>• `backend/pipeline/ingestion/sdo_sharp_fetcher.py`<br>• `backend/pipeline/ingestion/sync_validator.py`<br>• `backend/pipeline/ingestion/telemetry_health_daemon.py`<br>• `backend/pipeline/physics/hope_detector.py` | • `frontend/src/` (React 18, Vite, Tailwind)<br>• `frontend/src/routes/index.tsx`<br>• `frontend/src/components/dashboard/`<br>• `backend/pipeline/api/main.py`<br>• `backend/models/db_schema.sql`<br>• Dockerfile & docker-compose<br>• Cloudflare Pages & Render CI/CD |

> **Viva Strategy:** If asked about UI or satellite downloading APIs, confidently say:  
> *"That ingestion/frontend component was handled by my team members Sanskriti and Somya, while I designed the end-to-end Machine Learning backbone, mathematical loss formulations, data partitioning, PyTorch dataloaders, and ONNX deployment pipeline."*

---

## 3. High-Level Project & Domain Viva Questions

### Q1: What is Aditya-L1 and why is it preferred over Earth-orbiting satellites for space weather?
**Answer:**  
Aditya-L1 is the Indian Space Research Organisation's (ISRO) first dedicated solar observatory, launched in September 2023 and inserted into a halo orbit around the **Sun-Earth Lagrange Point 1 (L1)**, approximately 1.5 million kilometers from Earth.  
* **Uninterrupted Observation:** Unlike Low Earth Orbit (LEO) satellites (e.g., SDO or GOES) that experience orbital occultation (Earth eclipses the Sun) and pass through the South Atlantic Anomaly (radiation noise), Aditya-L1 has a 100% continuous, uninhibited line of sight to the Sun.
* **Early Warning Time:** Located 1.5 million km upstream towards the Sun, solar wind and coronal mass ejections pass Aditya-L1 **30 to 60 minutes before reaching Earth**, providing critical in-situ advance detection.

### Q2: What Aditya-L1 payloads are used in AdityaScan?
**Answer:**  
AdityaScan utilizes 4 primary instruments:
1. **SoLEXS (Solar Low Energy X-ray Spectrometer):** Monitors soft X-ray flux (1 keV to 22 keV) at high cadence, detecting thermal flare initiation.
2. **HEL1OS (High Energy L1 Orbiting X-ray Spectrometer):** Observes hard X-rays (10 keV to 150 keV) from non-thermal electron beam collisions during impulsive reconnection phases.
3. **MAG (Magnetometer):** Triaxial fluxgate magnetometer measuring the interplanetary magnetic field (IMF) vector ($B_x, B_y, B_z$), especially southward $B_z$ which triggers geomagnetic reconnection.
4. **ASPEX-SWIS (Aditya Solar wind Particle Experiment - Solar Wind Ion Spectrometer):** Measures low- and high-energy solar wind protons and alpha particles across 50 energy bins.

### Q3: How is AdityaScan fundamentally unique compared to NOAA SWPC and NASA models?
**Answer:**
1. **Multi-Sensor Physics Fusion vs Single-Instrument:** NOAA SWPC primarily relies on GOES X-ray flux. AdityaScan fuses soft X-rays, hard X-rays, vector magnetic fields, and solar wind in a single unified deep neural network.
2. **Physics-Encoded Features:** We encode explicit solar physics (the Neupert Effect, CWT Quasi-Periodic Pulsations, and SHARP magnetic shear deltas) as first-class inputs rather than relying purely on naive feature learning.
3. **5 Simultaneous Forecast Horizons:** NOAA publishes single 24-hour bulletins. AdityaScan provides graduated multi-horizon predictions: **30-min, 1-hour, 3-hour, 6-hour, and 24-hour** risk profiles.
4. **3-Layer Uncertainty Quantification:** Standard models output uncalibrated probabilities (e.g., "70% chance"). AdityaScan passes raw outputs through Platt Temperature Scaling (ECE < 3%) and Conformal Prediction, giving mathematically guaranteed **90% confidence intervals** (`[p_low, p_high]`).
5. **Edge-Optimized Inference:** Quantized ONNX model running under **42.6 ms** on standard CPU hardware without needing an expensive GPU server.

---

## 4. Every Formula Used: Mathematical Formulations, Derivations & Rationale

### 1. Dilated Causal Convolution (TCN Backbone)
#### Formula:
For a 1D sequence input $\mathbf{x} \in \mathbb{R}^T$ and a filter $f: \{0, \dots, K-1\} \to \mathbb{R}$, the dilated causal convolution at time step $t$ is:
$$\mathbf{y}(t) = (x *_d f)(t) = \sum_{k=0}^{K-1} f(k) \cdot \mathbf{x}(t - d \cdot k)$$
Where:
* $K$ = kernel size (in our model, $K=3$).
* $d$ = dilation factor (exponentially increasing: $d \in \{1, 2, 4, 8, 16, 32\}$).
* $t - d \cdot k$ ensures that **only past and present time steps** are accessed (causal padding = $(K-1) \cdot d$).

#### Effective Receptive Field ($RF$) Formula:
$$RF = 1 + \sum_{l=0}^{L-1} (K_l - 1) \cdot d_l$$
For our 6-layer TCN ($L=6, K=3, d_l = 2^l$):
$$RF = 1 + 2 \cdot (1 + 2 + 4 + 8 + 16 + 32) = 1 + 2 \cdot 63 = 127 \text{ time steps}$$
At 1-minute cadence, this captures over **2.1 hours** of temporal context without any recurrent vanishing gradient!

#### Why used?
Standard convolutions look into the future (violating causality). Standard RNNs suffer from vanishing gradients and cannot be parallelized across GPUs. Dilated Causal TCNs provide **exponentially large receptive fields with zero lookahead bias and parallel $O(1)$ training step time**.

---

### 2. Bidirectional LSTM Equations (Recurrent Branch)
#### Formulas:
For active region magnetic vectors $\mathbf{x}_t$ at step $t$:
1. **Forget Gate:** $f_t = \sigma(W_f \mathbf{x}_t + U_f h_{t-1} + b_f)$
2. **Input Gate:** $i_t = \sigma(W_i \mathbf{x}_t + U_i h_{t-1} + b_i)$
3. **Candidate Memory Cell:** $\tilde{c}_t = \tanh(W_c \mathbf{x}_t + U_c h_{t-1} + b_c)$
4. **Updated Cell State:** $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$
5. **Output Gate:** $o_t = \sigma(W_o \mathbf{x}_t + U_o h_{t-1} + b_o)$
6. **Hidden State:** $h_t = o_t \odot \tanh(c_t)$

For Bidirectional LSTM, forward $\overrightarrow{h}_t$ and backward $\overleftarrow{h}_t$ are concatenated:
$$H_t = [\overrightarrow{h}_t \,\|\, \overleftarrow{h}_t] \in \mathbb{R}^{2 \cdot d_{\text{hidden}}}$$

#### Why used?
Active region magnetic parameters (SDO SHARP 12-minute cadence) have non-linear evolutionary histories. BiLSTM processes the temporal buildup and decay of active region magnetic stress.

---

### 3. Cross-Modal Scaled Dot-Product Attention
#### Formula:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
Where:
* $Q = W_Q \cdot H_{\text{xray}} \in \mathbb{R}^{B \times 1 \times d}$ (Query derived from high-cadence X-ray TCN).
* $K = W_K \cdot [H_{\text{xray}} \,\|\, H_{\text{sharp}} \,\|\, H_{\text{insitu}}] \in \mathbb{R}^{B \times M \times d}$ (Keys from all modalities).
* $V = W_V \cdot [H_{\text{xray}} \,\|\, H_{\text{sharp}} \,\|\, H_{\text{insitu}}] \in \mathbb{R}^{B \times M \times d}$ (Values).
* Multi-head attention with $h=4$ heads:
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W_O$$
$$\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$$

#### Why used?
Different modalities have different cadences (X-ray is 1-minute; SHARP is 12-minutes; solar wind is 1-second). Cross-modal attention allows the network to **dynamically attend to magnetic shear or solar wind shocks only when an X-ray thermal spike occurs**. The scaling factor $\frac{1}{\sqrt{d_k}}$ prevents dot products from growing excessively large in high dimensions, which would push softmax into regions with vanishing gradients.

---

### 4. Adaptive Focal Loss (Class Imbalance Solution)
#### Standard Binary Cross-Entropy (BCE):
$$\text{BCE}(p_t) = -\log(p_t) \quad \text{where } p_t = \begin{cases} p & \text{if } y = 1 \\ 1 - p & \text{if } y = 0 \end{cases}$$
#### Lin et al. Focal Loss:
$$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
#### Mahansh's Adaptive Focal Loss (`backend/pipeline/ml/fusion.py`):
$$\gamma_h = \gamma_{\text{base}} + (h \times 0.2) \quad \text{for horizon } h \in \{0, 1, 2, 3, 4\}$$
$$\mathcal{L}_{\text{AdaptiveFL}} = -\frac{1}{N} \sum_{i=1}^N \alpha_{y_i} (1 - p_{t,i})^{\gamma_h} \log(p_{t,i})$$
Where $\alpha = 0.25$, $\gamma_{\text{base}} = 2.0$.

#### Why used?
In solar physics, flares are extremely rare (1:1200 ratio). With standard BCE, millions of easy "Quiet Sun" negative samples dominate the gradient, drowning out the signal from rare X-class flares. The modulating factor $(1 - p_t)^\gamma$ automatically downweights well-classified examples ($p_t > 0.9 \implies (1 - p_t)^2 < 0.01$), forcing the network to concentrate gradients on hard, misclassified flare eruptions.

---

### 5. Effective Number of Samples Reweighting (Cui et al., CVPR 2019)
#### Formula (`backend/pipeline/ml/balanced_sampler.py`):
$$E_n = \frac{1 - \beta^n}{1 - \beta}$$
$$W_c = \frac{1}{E_{n_c}} = \frac{1 - \beta}{1 - \beta^{n_c}}$$
Normalized weight:
$$\tilde{W}_c = \frac{W_c}{\sum_{k=1}^C W_k} \times C$$
Where:
* $n_c$ = number of samples in class $c$ (Quiet, C-class, M-class, X-class).
* $\beta \in [0, 1)$ = hyperparameter representing the probability that a new sample is covered by the existing feature volume. We set $\beta = 0.9999$.

#### Mathematical Rationale:
Naive inverse frequency weighting ($1/n_c$) over-penalizes majority classes when datasets grow very large because newly added quiet-Sun samples contain redundant information. The *Effective Number* formulation accounts for **diminishing marginal returns in sample space volume**, yielding smooth, stable weights that prevent gradient explosions during distributed training.

---

### 6. Platt Temperature Scaling (Calibration Layer)
#### Formula (`backend/pipeline/ml/uncertainty.py`):
Given uncalibrated logit vector $z$, calibrated probability $\hat{p}$ is:
$$\hat{p} = \sigma\left(\frac{z}{T}\right) = \frac{1}{1 + e^{-z / T}}$$
Where $T > 0$ is a single learnable temperature scalar optimized on the validation set by minimizing Negative Log-Likelihood (NLL):
$$\min_{T} -\sum_{i=1}^{N_{\text{val}}} \left[ y_i \log \sigma\left(\frac{z_i}{T}\right) + (1 - y_i) \log \left(1 - \sigma\left(\frac{z_i}{T}\right)\right) \right]$$
* If $T > 1$: the model was overconfident; probabilities are smoothed towards 0.5.
* If $T < 1$: the model was underconfident.
* In our model: optimal $T^* = 0.863$ (from `models/model_summary.json`).

---

### 7. Expected Calibration Error (ECE)
#### Formula:
Partition $N$ predictions into $M$ equal-width probability bins $B_1, \dots, B_M \subset [0, 1]$ (we use $M=10$):
$$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \Big| \text{acc}(B_m) - \text{conf}(B_m) \Big|$$
Where:
$$\text{acc}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} y_i, \quad \text{conf}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \hat{p}_i$$
* **Before calibration:** raw model $\text{ECE} \approx 11.4\%$.
* **After temperature scaling:** $\text{ECE} < 2.8\%$.

---

### 8. Conformal Prediction Quantile Calibration
#### Theory & Formula (`backend/pipeline/ml/uncertainty.py`):
For a user-specified coverage $1 - \alpha = 0.90$ (90% confidence):
1. Compute nonconformity scores on holdout calibration set:
$$s_i = |y_i - \hat{p}_i| \quad \text{or} \quad s_i = \frac{|y_i - \hat{p}_i|}{\hat{\sigma}_i}$$
2. Compute the $(1-\alpha)$ empirical quantile $\hat{q}$ with finite-sample correction:
$$\hat{q} = \text{Quantile}\left(\{s_1, \dots, s_n\}, \frac{\lceil (n+1)(1-\alpha) \rceil}{n}\right)$$
3. The prediction interval for a new test instance $X_{n+1}$ is:
$$C(X_{n+1}) = \left[ \hat{p}(X_{n+1}) - \hat{q} \cdot \hat{\sigma}, \; \hat{p}(X_{n+1}) + \hat{q} \cdot \hat{\sigma} \right]$$
#### Mathematical Guarantee Theorem (Vovk et al.):
$$\mathbb{P}\left(Y_{n+1} \in C(X_{n+1})\right) \ge 1 - \alpha$$
This guarantee holds **under any data distribution, without assuming Gaussianity**, provided data samples are exchangeable!

---

### 9. True Skill Statistic (TSS / Peirce's Skill Score)
#### Formula (`backend/utils/metrics.py`):
From a $2 \times 2$ contingency table:
$$\text{TSS} = \text{POD} - \text{POFD} = \frac{\text{TP}}{\text{TP} + \text{FN}} - \frac{\text{FP}}{\text{FP} + \text{TN}} = \text{TPR} - \text{FPR}$$
* Range: $[-1, +1]$. Perfect forecast = $+1$, random guess = $0$, always wrong = $-1$.
* **Why TSS is the Gold Standard in Space Weather:**
  Accuracy is given by $\frac{\text{TP} + \text{TN}}{\text{TP} + \text{FP} + \text{FN} + \text{TN}}$. Because non-flare hours account for 99.9% of the year, a trivial model that always predicts "NO FLARE" achieves **99.9% accuracy**, but its TSS is **$0.0$**! TSS is **unaffected by class prevalence (marginal event frequency)**, making it impossible to game.

---

### 10. Heidke Skill Score (HSS)
#### Formula (`backend/utils/metrics.py`):
$$\text{HSS} = \frac{2(\text{TP} \cdot \text{TN} - \text{FP} \cdot \text{FN})}{(\text{TP} + \text{FN})(\text{FN} + \text{TN}) + (\text{TP} + \text{FP})(\text{FP} + \text{TN})}$$
Measures accuracy relative to a random chance forecast that matches marginal probabilities. Range: $[-\infty, 1]$.

---

### 11. Brier Score (BS)
#### Formula:
$$\text{BS} = \frac{1}{N} \sum_{t=1}^N (f_t - o_t)^2$$
Where $f_t \in [0, 1]$ is forecasted probability and $o_t \in \{0, 1\}$ is actual outcome.
AdityaScan achieves **$\text{BS} = 0.089$**, significantly outperforming climatological baselines ($0.22$).

---

### 12. Physics Equations: Neupert Effect & CWT Wavelets
* **Neupert Effect:**
  $$\frac{d}{dt} I_{\text{SXR}}(t) \propto I_{\text{HXR}}(t)$$
  *Physical principle:* The time derivative of thermal soft X-ray flux (SoLEXS) is directly proportional to instantaneous non-thermal hard X-ray emission (HEL1OS) produced by thick-target bremsstrahlung.
* **Continuous Wavelet Transform (CWT) for QPP:**
  $$W(s, \tau) = \frac{1}{\sqrt{s}} \int_{-\infty}^{\infty} x(t) \cdot \psi^*\left(\frac{t - \tau}{s}\right) dt$$
  Using complex Morlet mother wavelet $\psi(t) = \pi^{-1/4} e^{i \omega_0 t} e^{-t^2 / 2}$ with scale $s$ tuned to periods of $10 \le T \le 300$ seconds to isolate magnetohydrodynamic (MHD) sausage and kink mode oscillations prior to flare reconnection.

---

## 5. Neural Architecture Deep-Dive (Code Line-by-Line Viva)

```
                            ┌────────────────────────────────────────┐
                            │    AdityaScan v4 Neural Backbone       │
                            └────────────────────────────────────────┘
                                                │
       ┌────────────────────────────────────────┼────────────────────────────────────────┐
       ▼                                        ▼                                        ▼
┌───────────────────────────┐        ┌───────────────────────────┐        ┌───────────────────────────┐
│     X-Ray TCN Branch      │        │     SHARP LSTM Branch     │        │     In-Situ MLP Branch    │
│ (SoLEXS + HEL1OS + GOES)  │        │   (SDO CEA Magnetograms)  │        │   (MAG L2 + ASPEX-SWIS)   │
│  Input: [B, 1800, 11]     │        │    Input: [B, 120, 21]    │        │      Input: [B, 14]       │
└─────────────┬─────────────┘        └─────────────┬─────────────┘        └─────────────┬─────────────┘
              │                                    │                                    │
    6× CausalConv1d (d=1..32)               2× BiLSTM (dim=128)                  2× Linear + LayerNorm
              │                                    │                                    │
     Embedding: [B, 256]                  Embedding: [B, 128]                  Embedding: [B, 64]
              │                                    │                                    │
              └────────────────────────┬───────────┴────────────────────────────────────┘
                                       ▼
                       ┌───────────────────────────────┐
                       │  Cross-Modal Attention Fusion │
                       │    (4 Heads, Query = X-ray)   │
                       └───────────────┬───────────────┘
                                       ▼
                             Fused Vector: [B, 256]
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
       ┌────────────────────────┐            ┌────────────────────────┐
       │   5-Horizon Forecast   │            │   Sympathetic GNN      │
       │   (30m, 1h, 3h, 6h, 24h)│            │ Active Region Topology │
       └────────────┬───────────┘            └────────────────────────┘
                    ▼
       ┌────────────────────────┐
       │ Platt Temp Scale (T)   │
       │  + Conformal [p_l, p_h]│
       └────────────────────────┘
```

### Questions on `backend/pipeline/ml/fusion.py`

#### Q4: Walk me through lines 44–57 of `fusion.py`: How is `CausalConv1d` implemented?
**Answer:**
```python
class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation  # causal padding
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=self.padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)[:, :, :-self.padding] if self.padding > 0 else self.conv(x)
```
* **Explanation:**
  * In standard PyTorch `nn.Conv1d`, padding is symmetrical (half on the left, half on the right), which would leak future time steps into past predictions.
  * In `CausalConv1d`, we add total padding of `(kernel_size - 1) * dilation` to both sides, and then in the `forward` function, we slice off `:-self.padding` from the trailing end!
  * This guarantees that output at index $t$ depends **strictly on inputs at or before index $t$**.

#### Q5: What are the exact input tensor shapes fed into the network?
**Answer:**
*(Reference: `models/adityscan_v4_onnx_spec.json`)*
1. `xray`: `[Batch_Size, 1800, 11]` — 1800 time steps (30 minutes at 1-sec cadence), 11 channels (SoLEXS bands, HEL1OS channels, GOES primary/secondary).
2. `sharp`: `[Batch_Size, 120, 21]` — 120 time steps (24 hours at 12-min cadence), 21 magnetic parameters (18 SDO SHARP parameters + 3 Aditya MAG components).
3. `insitu`: `[Batch_Size, 14]` — 14 static/rolling plasma parameters (solar wind speed, proton density, alpha ratio, $B_z$ southward component).
4. `output`: `[Batch_Size, 5]` — Calibrated flare probabilities for `[30m, 1h, 3h, 6h, 24h]`.

#### Q6: Why did you make X-ray the Query ($Q$) in the Cross-Modal Attention?
**Answer:**  
In solar physics, the soft X-ray spectrometer (SoLEXS/GOES) is the **primary operational indicator** of a flare occurring right now. Active region magnetic fields (SHARP) evolve slowly over days. By setting $Q = H_{\text{xray}}$ and $K, V = [H_{\text{xray}}, H_{\text{sharp}}, H_{\text{insitu}}]$, the network asks: *"Given the current high-frequency thermal activity in the X-ray curve, which magnetic stresses and solar wind conditions should I attend to?"* This aligns model attention with astrophysical causality.

---

## 6. Code Written & Committed by Mahansh Gaur (File-by-File Defense)

### 1. `backend/pipeline/ml/chronological_split.py`
#### Q7: Why can't we use standard `sklearn.model_selection.train_test_split` or standard K-Fold CV?
**Answer:**  
Using random `train_test_split` on time-series telemetry causes catastrophic **lookahead leakage**. If time $t-1$ is in the training set and time $t$ is in the test set, the model simply memorizes autocorrelation rather than learning true predictive physics. The model will show 99% test accuracy in the lab and completely fail in real production.

#### Q8: How does your `ChronologicalSplitter` work?
**Answer:**
1. Sorts all timestamps in strict ascending order.
2. Allocates chronological blocks: first 70% for Training, next 15% for Validation, final 15% for Testing.
3. Inserts a **24-hour quarantine buffer** (`quarantine_buffer_hours = 24.0`):
   ```python
   val_start_time = train_end_time + pd.Timedelta(hours=self.quarantine_buffer_hours)
   val_start_idx = np.searchsorted(sorted_times, val_start_time)
   ```
4. Purges all samples within the quarantine window so that rolling 24-hour autoregressive feature windows from the training set do not overlap into validation targets.

---

### 2. `backend/pipeline/ml/balanced_sampler.py`
#### Q9: How does `StratifiedSolarSampler` work in your code?
**Answer:**
```python
class StratifiedSolarSampler(Sampler[int]):
    def __init__(self, labels: List[str], oversample_flare_ratio: float = 4.0, beta: float = 0.9999):
```
* Computes class frequencies: $N_{\text{quiet}} \gg N_C > N_M > N_X$.
* Calculates effective number weights: $W_c = \frac{1 - \beta}{1 - \beta^{n_c}}$ with $\beta = 0.9999$.
* Applies an additional multiplier `oversample_flare_ratio = 4.0` to positive flare events (X, M, C classes).
* Uses `torch.multinomial(self.weights, self.num_samples, replacement=True)` to draw balanced mini-batches on each training epoch.

---

### 3. `backend/pipeline/ml/distributed_loader.py`
#### Q10: How does `distributed_loader.py` prevent Out-Of-Memory (OOM) errors on 16 GB RAM or GPU nodes?
**Answer:**
* Standard `torch.utils.data.Dataset` attempts to load all samples into memory at once, which crashes with 15 months of multi-gigabyte FITS satellite telemetry.
* I inherited from `torch.utils.data.IterableDataset` to build `ChunkedSolarDataset`:
  ```python
  class ChunkedSolarDataset(IterableDataset):
      def __iter__(self):
          for chunk_file in self.chunk_files:
              yield from self._stream_chunk(chunk_file)
  ```
* It streams month-by-month chunk files sequentially, discards raw arrays after batching, and sets `pin_memory=True` with `prefetch_factor=2` to stream pinned tensor memory directly into GPU VRAM asynchronously. Total memory never exceeds **5 GB RAM**.

---

### 4. `scripts/validate_incremental_checkpoints.py` & Model Serialization
#### Q11: What does your checkpoint validation script verify?
**Answer:**  
When new telemetry arrives from ISRO PRADAN or NOAA, we perform incremental fine-tuning (`run_balanced_update.sh`). `validate_incremental_checkpoints.py` parses `models/training_report.json` and evaluates the updated weights against historical regression benchmarks:
```python
assert tss >= 0.80, f"Regression detected: TSS {tss} < 0.80 threshold"
```
If the new weights cause True Skill Statistic to drop below 0.80, the script exits with code 1, aborting deployment and preserving the previous checkpoint.

---

### 5. `generate_report.py` (Form-3 Automation)
#### Q12: How does `generate_report.py` generate weekly reports?
**Answer:**  
I built `generate_report.py` using Python's `subprocess` and `reportlab` library:
* Executes `git log --no-merges --pretty=format:COMMIT|||%h|||%an|||%ad|||%s --date=short --numstat --since=... --until=...`.
* Parses line additions, deletions, and daily commit frequencies per student.
* Uses `matplotlib` to render a 2-panel chart: (1) Chronological Commit Timeline, and (2) Net Lines of Code Written bar chart.
* Embeds institutional headers for SKIT Jaipur Department of Computer Science & Engineering, generates student contribution tables, and prints mentor evaluation mark sheets (`_____ / 10`).

---

## 7. Solar Flare Physics & Space Weather Domain Questions

### Q13: Explain the GOES Solar Flare Classification scale.
**Answer:**  
Solar flares are classified by their peak soft X-ray flux in the 1–8 Å (0.1–0.8 nm) band measured in Watts per square meter ($W/m^2$):
* **A-Class:** $< 10^{-7} \, W/m^2$ (Quiet background)
* **B-Class:** $10^{-7} \text{ to } 10^{-6} \, W/m^2$
* **C-Class:** $10^{-6} \text{ to } 10^{-5} \, W/m^2$ (Minor flare; minimal terrestrial impact)
* **M-Class:** $10^{-5} \text{ to } 10^{-4} \, W/m^2$ (Medium flare; causes brief radio blackouts at polar regions)
* **X-Class:** $\ge 10^{-4} \, W/m^2$ (Extreme flare; X1 = $10^{-4}$, X10 = $10^{-3}$, X20+ Carrington-scale event; triggers global HF radio blackouts and satellite safe-mode protocols)

### Q14: What was the May 2024 Solar Storm (Gannon Superstorm)?
**Answer:**  
From May 10–12, 2024, Active Region **AR13664** produced a sequence of extreme X-class flares culminating in an X5.8 flare. It triggered a **G5-class extreme geomagnetic storm** (Dst index reached $-412 \text{ nT}$, the strongest since November 2003). It caused widespread auroras across low latitudes, disrupted GPS navigation, and forced satellite operators to execute orbital corrections. We used May 2024 as our primary extreme-event positive test window.

### Q15: What is the Carrington Event of 1859?
**Answer:**  
The Carrington Event (September 1–2, 1859) was the most powerful solar storm in recorded history, estimated at **$\ge \text{X40}$**. Telegraph systems worldwide sparked and caught fire. A Carrington-scale event today would cause estimated global economic damages of **$0.6 to $2.6 Trillion** in year one due to catastrophic transformer failures, satellite destruction, and worldwide communication blackouts.

---

## 8. MLOps, Hardware Optimization & Production Engineering Questions

### Q16: How did you export PyTorch to ONNX and apply INT8 Quantization?
**Answer:**
1. Exported the trained PyTorch fusion graph using `torch.onnx.export` targeting **Opset version 17** with dynamic batch dimensions.
2. Applied ONNX Runtime dynamic quantization (`quantization.quantize_dynamic`) converting FP32 weights to **INT8 (8-bit signed integers)** for linear and convolution layers.
3. **Benefits:**
   * Reduced model file size from ~8.4 MB to **2.2 MB**.
   * Cut CPU inference latency from **180 ms down to 42.6 ms** (a $4.2\times$ speedup).
   * Enabled deployment on cheap commodity CPU instances without requiring costly GPU cloud servers.

### Q17: Where and how is the system deployed in production?
**Answer:**
* **Frontend:** React 18 / Vite with Tailwind CSS deployed on **Cloudflare Pages CDN** with global edge caching ([`aditya-scan.lsoni6870.workers.dev`](https://aditya-scan.lsoni6870.workers.dev)).
* **Backend:** FastAPI application running on **Render.com** (with in-memory TTL caching and WebSocket `/ws/live` streaming).
* **Keep-Alive Monitor:** Python daemon (`scripts/keep_alive.py`) pinging health endpoints to prevent free-tier container spin-down.

---

## 9. Professor "Trap Questions" & Bulletproof Rebuttals

### Trap 1: *"Why didn't you just use XGBoost or Random Forest on tabular features?"*
**Bulletproof Rebuttal:**  
*"Sir/Madam, traditional tabular tree models require flattening temporal sequences into static statistics (mean, max, std). Doing so destroys the **fine-grained phase dynamics** of solar reconnection—such as the 10–300s Quasi-Periodic Pulsations (QPP) and high-frequency $dI/dt$ derivative inflections captured by our Causal TCN. Furthermore, tree models cannot perform cross-modal attention between asynchronous sensors, nor can they output conformal prediction set guarantees with temporal continuity."*

### Trap 2: *"Your dataset is heavily imbalanced. Couldn't you just use SMOTE?"*
**Bulletproof Rebuttal:**  
*"SMOTE (Synthetic Minority Over-sampling Technique) operates in static Euclidean feature space by linearly interpolating between k-nearest neighbors. Applying SMOTE to high-cadence 1800-step time-series creates synthetic curves that **violate physical conservation laws** (e.g., negative flux values or discontinuous derivative spikes). Instead, we solved imbalance through **Effective Number of Samples reweighting** and **Adaptive Focal Loss**, which preserves the true physical telemetry while mathematically scaling the gradient updates."*

### Trap 3: *"Did you write all this code yourself or clone it from a repository?"*
**Bulletproof Rebuttal:**  
*"Every architectural component in `backend/pipeline/ml/` was designed and written by our team from scratch. You can verify this through our complete Git commit history spanning August 2026 to October 2026 across 18 merged Pull Requests. Every PR corresponds to institutional Form-2 sprint user stories (`US-ML-01` through `US-ML-06`), and the code directly interfaces with ISRO's specific Level-1 FITS binary structures for SoLEXS and HEL1OS—which no external public repository currently supports."*

### Trap 4: *"What if the internet is down or ISRO's PRADAN portal is inaccessible?"*
**Bulletproof Rebuttal:**  
*"AdityaScan has an automated multi-layer failover mechanism designed into our ingestion architecture (`backend/pipeline/ingestion/goes_fetcher.py`). If Aditya-L1 telemetry packets are delayed or corrupted, the system falls back to NOAA GOES primary and secondary streams. If all network feeds drop, our ONNX engine runs locally offline in historical replay mode using cached active region vectors."*

### Trap 5: *"What are the limitations of your current model?"*
*(Examiners LOVE candidates who know their limitations—it proves authenticity!)*  
**Bulletproof Rebuttal:**  
*"Currently, Aditya-L1 observes the Sun along the Sun-Earth line ($L_1$). We cannot directly observe active regions emerging on the far side of the Sun until solar rotation brings them into the east limb (~27-day solar synodic period). In future work, we plan to ingest far-side helioseismology data from SDO and integrate future ISRO missions planned for the $L_5$ vantage point."*

---

## 10. Quick-Recall Formula & Numbers Cheat Sheet

| Parameter / Metric | Exact Value | Where in Code |
| :--- | :--- | :--- |
| **True Skill Statistic (TSS)** | **0.887** | `models/training_report.json` |
| **ROC AUC** | **0.894** | `models/training_report.json` |
| **Brier Score** | **0.089** | `docs/AdityScan_Technical_Report.html` |
| **Inference Latency** | **42.6 ms** | `models/adityscan_v4_onnx_spec.json` |
| **ONNX Opset Version** | **17** | `models/adityscan_v4_onnx_spec.json` |
| **Total Model Parameters** | **~2.1 Million** | `models/model_summary.json` |
| **Forecast Horizons** | **30m, 1h, 3h, 6h, 24h** | `backend/pipeline/ml/fusion.py` |
| **TCN Receptive Field** | **127 time steps (2.1 hrs)** | `backend/pipeline/ml/fusion.py` |
| **Focal Loss Parameters** | $\alpha = 0.25, \; \gamma_{\text{base}} = 2.0$ | `backend/pipeline/ml/fusion.py` |
| **Effective Number Beta** | $\beta = 0.9999$ | `backend/pipeline/ml/balanced_sampler.py` |
| **Optimal Temperature ($T$)** | **0.863** | `models/model_summary.json` |
| **Conformal Coverage** | **90% guaranteed ($1-\alpha = 0.90$)** | `backend/pipeline/ml/uncertainty.py` |
| **RAM Ceiling during Training** | **< 5.0 GB RSS** | `backend/pipeline/ml/distributed_loader.py` |
| **SDO Cache Disk Quota** | **10.0 GB (LRU eviction)** | `backend/pipeline/ingestion/sdo_sharp_fetcher.py` |
| **Quarantine Isolation Buffer** | **24.0 Hours** | `backend/pipeline/ml/chronological_split.py` |
