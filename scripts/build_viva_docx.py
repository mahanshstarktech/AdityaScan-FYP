"""
AdityaScan Viva Voce & Defense Master Guide DOCX Generator.
Builds an institutional, publication-grade Microsoft Word (.docx) document
complete with high-resolution graphics, diagrams, styled tables, and defense callouts.
"""

from __future__ import annotations

import os
import re
import sys
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn


def set_cell_background(cell, hex_color: str):
    tcPr = cell._element.get_or_add_tcPr()
    # Remove existing shading if present
    for child in list(tcPr):
        if child.tag.endswith('shd'):
            tcPr.remove(child)
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    tcPr.append(shd)


def set_cell_margins(cell, top=120, bottom=120, left=180, right=180):
    tcPr = cell._element.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)


def add_callout(doc: Document, title: str, text: str, border_color="0F294A", bg_color="F1F5F9"):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    cell.width = Inches(6.5)
    set_cell_background(cell, bg_color)
    set_cell_margins(cell, top=140, bottom=140, left=200, right=200)

    # Left border only
    tcPr = cell._element.get_or_add_tcPr()
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:left w:val="single" w:sz="36" w:space="0" w:color="{border_color}"/>'
        f'<w:top w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'<w:bottom w:val="none"/>'
        f'</w:tcBorders>'
    )
    tcPr.append(borders)

    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(4)
    run_t = p.add_run(title + "\n")
    run_t.font.name = "Calibri"
    run_t.font.size = Pt(11)
    run_t.font.bold = True
    run_t.font.color.rgb = RGBColor(15, 41, 74)

    run_body = p.add_run(text)
    run_body.font.name = "Calibri"
    run_body.font.size = Pt(10)
    run_body.font.italic = False
    run_body.font.color.rgb = RGBColor(30, 41, 59)

    doc.add_paragraph().paragraph_format.space_after = Pt(4)


def add_figure(doc: Document, image_path: str, caption: str, width_inches=6.0):
    if not os.path.exists(image_path) or os.path.getsize(image_path) < 1000:
        return
    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.space_before = Pt(8)
    p_img.paragraph_format.space_after = Pt(2)
    run_img = p_img.add_run()
    run_img.add_picture(image_path, width=Inches(width_inches))

    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.paragraph_format.space_before = Pt(2)
    p_cap.paragraph_format.space_after = Pt(10)
    run_cap = p_cap.add_run(caption)
    run_cap.font.name = "Calibri"
    run_cap.font.size = Pt(9.5)
    run_cap.font.italic = True
    run_cap.font.color.rgb = RGBColor(71, 85, 105)


def add_code_block(doc: Document, code_text: str):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    cell.width = Inches(6.5)
    set_cell_background(cell, "0F172A")
    set_cell_margins(cell, top=100, bottom=100, left=160, right=160)

    tcPr = cell._element.get_or_add_tcPr()
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:left w:val="single" w:sz="12" w:space="0" w:color="334155"/>'
        f'<w:top w:val="single" w:sz="12" w:space="0" w:color="334155"/>'
        f'<w:right w:val="single" w:sz="12" w:space="0" w:color="334155"/>'
        f'<w:bottom w:val="single" w:sz="12" w:space="0" w:color="334155"/>'
        f'</w:tcBorders>'
    )
    tcPr.append(borders)

    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    run = p.add_run(code_text.strip())
    run.font.name = "Consolas"
    run.font.size = Pt(8.5)
    run.font.color.rgb = RGBColor(226, 232, 240)

    doc.add_paragraph().paragraph_format.space_after = Pt(4)


def add_styled_table(doc: Document, headers: list[str], rows: list[list[str]], col_widths=None):
    tbl = doc.add_table(rows=len(rows) + 1, cols=len(headers))
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False

    # Header row
    hdr_cells = tbl.rows[0].cells
    for i, title in enumerate(headers):
        hdr_cells[i].text = title
        set_cell_background(hdr_cells[i], "0F294A")
        set_cell_margins(hdr_cells[i], top=120, bottom=120, left=140, right=140)
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        for r in p.runs:
            r.font.name = "Calibri"
            r.font.size = Pt(10)
            r.font.bold = True
            r.font.color.rgb = RGBColor(255, 255, 255)

    # Data rows
    for r_idx, row_data in enumerate(rows):
        row_cells = tbl.rows[r_idx + 1].cells
        bg_col = "FFFFFF" if r_idx % 2 == 0 else "F8FAFC"
        for c_idx, val in enumerate(row_data):
            row_cells[c_idx].text = val
            set_cell_background(row_cells[c_idx], bg_col)
            set_cell_margins(row_cells[c_idx], top=100, bottom=100, left=140, right=140)
            p = row_cells[c_idx].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            for r in p.runs:
                r.font.name = "Calibri"
                r.font.size = Pt(9.5)
                r.font.color.rgb = RGBColor(30, 41, 59)

    # Set cell widths if provided
    if col_widths:
        for row in tbl.rows:
            for i, w in enumerate(col_widths):
                row.cells[i].width = Inches(w)

    doc.add_paragraph().paragraph_format.space_after = Pt(6)


def build_viva_defense_docx():
    print("[*] Generating AdityaScan Viva Defense Master Guide DOCX...")
    doc = Document()

    # Configure 1.0 inch margins
    for sec in doc.sections:
        sec.top_margin = Inches(1.0)
        sec.bottom_margin = Inches(1.0)
        sec.left_margin = Inches(1.0)
        sec.right_margin = Inches(1.0)

    # ── COVER / TITLE HEADER BLOCK ──────────────────────────────────────────
    p_inst = doc.add_paragraph()
    p_inst.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_inst.paragraph_format.space_before = Pt(12)
    p_inst.paragraph_format.space_after = Pt(2)
    run_inst = p_inst.add_run("SWAMI KESHVANAND INSTITUTE OF TECHNOLOGY, MANAGEMENT & GRAMOTHAN\n")
    run_inst.font.name = "Calibri"
    run_inst.font.size = Pt(11)
    run_inst.font.bold = True
    run_inst.font.color.rgb = RGBColor(15, 41, 74)

    run_dept = p_inst.add_run("DEPARTMENT OF COMPUTER SCIENCE & ENGINEERING (IoT)\nRamnagaria, Jagatpura, Jaipur – 302017")
    run_dept.font.name = "Calibri"
    run_dept.font.size = Pt(9.5)
    run_dept.font.color.rgb = RGBColor(71, 85, 105)

    doc.add_paragraph().paragraph_format.space_after = Pt(14)

    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_after = Pt(4)
    run_title = p_title.add_run("AdityaScan: Multi-Modal Solar Flare Forecasting using Aditya-L1 Telemetry\n")
    run_title.font.name = "Calibri"
    run_title.font.size = Pt(18)
    run_title.font.bold = True
    run_title.font.color.rgb = RGBColor(15, 41, 74)

    run_sub = p_title.add_run("Comprehensive Viva Voce & Major Project Defense Master Guide")
    run_sub.font.name = "Calibri"
    run_sub.font.size = Pt(13)
    run_sub.font.bold = True
    run_sub.font.color.rgb = RGBColor(29, 78, 216)

    # Metadata Table
    headers_meta = ["Parameter", "Official Record Details"]
    rows_meta = [
        ["Project ID", "SKIT/IOT/2023-2027/P-04"],
        ["Candidate / Examinee", "Mahansh Gaur (Team Lead — Core AI/ML & MLOps Track)"],
        ["Teammates / Collaborators", "Sanskriti Raj (Satellite Ingestion & Physics) | Somya Roy (Mission UI & DevOps)"],
        ["Academic Year", "2026–2027 (Final Year B.Tech CSE-IoT)"],
        ["Live Production System", "https://aditya-scan.lsoni6870.workers.dev (Cloudflare Pages Edge)"],
        ["Verified Git Repository", "https://github.com/mahanshstarktech/AdityaScan-FYP.git (33 PRs, 60 Tests)"],
    ]
    add_styled_table(doc, headers_meta, rows_meta, col_widths=[1.8, 4.7])

    doc.add_page_break()

    # ── SECTION 1: EXECUTIVE STRATEGY ────────────────────────────────────────
    h1 = doc.add_heading("1. Executive Defense Strategy & 3-Minute Presentation Walkthrough", level=1)
    h1.paragraph_format.space_before = Pt(8)
    h1.paragraph_format.space_after = Pt(6)

    add_callout(
        doc,
        title="THE 30-SECOND ELEVATOR PITCH (When asked: 'Briefly introduce your project')",
        text=(
            "\"Good morning/afternoon, Respected Examiners. My name is Mahansh Gaur. I am the Team Lead "
            "and Core AI/ML & MLOps architect for AdityaScan.\n\n"
            "Severe solar flares and coronal mass ejections can cause trillions of dollars in damage to orbital "
            "satellite constellations like India's NavIC and GSAT, and knock out terrestrial power grids. International "
            "agencies like NOAA SWPC issue single-sensor heuristic warnings with only 20-30 minute lead times and a TSS of ~0.50.\n\n"
            "AdityaScan is India's first physics-informed, multi-modal deep learning system ingesting live telemetry from ISRO's "
            "Aditya-L1 spacecraft (SoLEXS, HEL1OS, MAG, ASPEX-SWIS) coupled with NOAA GOES and SDO SHARP data. My neural "
            "architecture uses Dilated Causal TCNs, Bidirectional LSTMs, Active Region Coronal Arcade GNNs, and Cross-Modal Attention "
            "with calibrated conformal prediction to forecast solar flares across 5 simultaneous time horizons (from 30 minutes up to 24 hours) "
            "with a CPU inference latency of 42.6 milliseconds and an industry-leading True Skill Statistic of 0.887.\""
        ),
        border_color="0F294A",
        bg_color="F1F5F9"
    )

    p_walk = doc.add_paragraph()
    p_walk.paragraph_format.space_after = Pt(6)
    p_walk.add_run("The 3-Minute Comprehensive Defense Walkthrough:\n").bold = True
    p_walk.add_run(
        "1. The Core Hazard: Solar flares emit extreme soft and hard X-ray bursts within minutes of magnetic reconnection, "
        "followed by energetic proton radiation belts that trigger Single-Event Upsets (SEUs) and deep dielectric charging in orbital satellites.\n"
        "2. The Innovation: Stationed 1.5 million km sunward at Sun-Earth L1, Aditya-L1 observes 100% of solar phenomena without Earth eclipses. "
        "AdityaScan directly incorporates physical laws—the Neupert Effect (dI_SXR/dt proportional to I_HXR), Continuous Morlet Wavelet QPPs, "
        "and coronal arcade magnetic coupling.\n"
        "3. Mahansh Gaur's Architectural Ownership: Designed the multi-modal neural fusion engine, solved the 1:1200 flare class imbalance via "
        "Effective Number of Samples reweighting, built the memory-bounded distributed PyTorch DataLoader (<5GB RAM), eliminated temporal leakage "
        "with 24-hr quarantine buffers, and derived finite-sample 90% conformal prediction intervals."
    )

    # ── SECTION 2: CANDIDATE OWNERSHIP MATRIX ────────────────────────────────
    doc.add_heading("2. Candidate Ownership & Boundary Matrix (What Mahansh Gaur Built)", level=1)
    
    headers_own = ["Domain Track", "Mahansh Gaur (Candidate / Team Lead)", "Sanskriti Raj (Teammate)", "Somya Roy (Teammate)"]
    rows_own = [
        ["Primary Responsibility", "Core AI/ML & MLOps Pipeline Architecture", "Satellite Ingestion & Solar Physics Pipeline", "Mission Control UI & Full-Stack DevOps"],
        ["Form-2 Stories", "US-ML-01 to US-ML-06 (35 Jira Tickets: AS-101..183)", "US-ING-01 to US-PHY-06 (25 Jira Tickets)", "US-FS-01 to US-OPS-06 (25 Jira Tickets)"],
        ["Core Python / TS Files Owned", 
         "backend/pipeline/ml/fusion.py\nbackend/pipeline/ml/gnn.py\nbackend/pipeline/ml/inference_engine.py\nbackend/pipeline/ml/loss.py\nbackend/pipeline/ml/conformal_calibrator.py\nbackend/pipeline/ml/async_inference_worker.py\nbackend/pipeline/ml/balanced_sampler.py\nbackend/pipeline/ml/distributed_loader.py\nscripts/export_production_onnx.py",
         "backend/pipeline/ingestion/pradan_downloader.py\nbackend/pipeline/ingestion/solexs_loader.py\nbackend/pipeline/ingestion/helios_loader.py\nbackend/pipeline/physics/neupert_proxy.py\nbackend/pipeline/physics/wavelet_qpp.py\nbackend/pipeline/physics/mag_variational_filter.py\nbackend/pipeline/physics/plasma_beta_calculator.py\nbackend/pipeline/physics/telemetry_qc_filter.py",
         "frontend/src/routes/index.tsx\nfrontend/src/components/dashboard/ProbabilityGauges.tsx\nfrontend/src/components/dashboard/ActiveRegionStatusCards.tsx\nfrontend/src/components/dashboard/CoronalArcadeMatrix.tsx\nfrontend/src/components/dashboard/ConstellationHazardMatrix.tsx\nbackend/pipeline/api/main.py\nDockerfile & CI/CD"],
        ["Verified Deliverables", "TSS=0.887, AUC=0.942, Brier=0.0512, ONNX 42.6ms", "PRADAN batch pipeline, QPP wavelets, Plasma beta", "Cloudflare edge UI, WebSocket /ws/live push"]
    ]
    add_styled_table(doc, headers_own, rows_own, col_widths=[1.3, 2.0, 1.6, 1.6])

    add_figure(
        doc,
        "/tmp/viva_graphics/mermaid_1.png",
        "Figure 1: AdityaScan three-track engineering division and component ownership boundaries.",
        width_inches=6.0
    )

    # ── SECTION 3: ORBITAL GEOMETRY & EARLY WARNING ──────────────────────────
    doc.add_heading("3. Sun-Earth L1 Orbital Geometry & Early Warning Advantage", level=1)
    
    p_l1 = doc.add_paragraph()
    p_l1.add_run(
        "Aditya-L1 is India's pioneer solar observatory operating in a halo orbit around the Sun-Earth Lagrange Point 1 (L1), "
        "approximately 1.5 million kilometers from Earth (1% of the Sun-Earth distance). Unlike Earth-orbiting satellites in Low Earth Orbit (LEO) "
        "or Geostationary Orbit (GEO), the L1 vantage point provides three decisive physics-grounded advantages:\n\n"
        "1. 100% Uninterrupted Line-of-Sight: LEO spacecraft experience 15-16 solar eclipses per day as Earth passes between the satellite and Sun. "
        "Aditya-L1 observes the solar disk continuously with zero occultation cycles.\n"
        "2. Freedom from Radiation Belts: Low-Earth orbiters traverse the South Atlantic Anomaly (SAA), producing severe proton contamination in detectors. "
        "At L1, background radiation is purely heliospheric.\n"
        "3. 30-60 Minute Upstream Plasma Warning: Solar wind and coronal mass ejections travel at 400-2000 km/s. In-situ plasma instruments (MAG and ASPEX) "
        "detect shock compression waves and southward magnetic field rotations 30 to 60 minutes before they reach Earth's bow shock."
    )

    add_figure(
        doc,
        "/tmp/viva_graphics/mermaid_2.png",
        "Figure 2: Sun-Earth Lagrange Point 1 (L1) geometry and early warning transit timeline.",
        width_inches=6.0
    )

    add_figure(
        doc,
        "/tmp/viva_graphics/report_fig_2.png",
        "Figure 3: Aditya-L1 payload integration map for AdityaScan (SoLEXS, HEL1OS, MAG, ASPEX-SWIS, and NOAA GOES-18).",
        width_inches=6.2
    )

    # ── SECTION 4: END-TO-END ARCHITECTURE ───────────────────────────────────
    doc.add_heading("4. End-to-End System & Ingestion Dataflow Architecture", level=1)

    p_arch = doc.add_paragraph()
    p_arch.add_run(
        "The AdityaScan operational pipeline continuously transforms raw satellite FITS files, telemetry packets, and magnetograms "
        "into calibrated space weather alerts across five distinct operational layers:"
    )

    add_figure(
        doc,
        "/tmp/viva_graphics/report_fig_3.png",
        "Figure 4: AdityaScan end-to-end architecture from raw telemetry to calibrated operational multi-horizon forecasts.",
        width_inches=6.2
    )

    add_figure(
        doc,
        "/tmp/viva_graphics/report_fig_1.png",
        "Figure 5: Physical solar-to-satellite hazard chain and corresponding sensor monitoring links.",
        width_inches=6.2
    )

    # ── SECTION 5: MATHEMATICAL FORMULATIONS ─────────────────────────────────
    doc.add_heading("5. Mathematical Formulations, Physics Derivations & Rationale", level=1)

    p_math_intro = doc.add_paragraph()
    p_math_intro.add_run("All sixteen mathematical equations engineered into the AdityaScan machine learning and physics pipelines:\n").bold = True

    math_eqs = [
        ("1. Dilated Causal Convolution (TCN Backbone)",
         "y(t) = sum_{k=0}^{K-1} f(k) * x(t - d*k)\n"
         "Receptive Field: RF = 1 + sum_{l=0}^{L-1} (K_l - 1) * d_l = 1 + 2*(1+2+4+8+16+32) = 127 timesteps (2.11 hours context).",
         "Eliminates lookahead leakage with causal left-padding while achieving exponential receptive field growth with zero vanishing gradients."),
        ("2. Bidirectional LSTM Gated Magnetic Evolution",
         "f_t = sigma(W_f x_t + U_f h_{t-1} + b_f),  i_t = sigma(W_i x_t + U_i h_{t-1} + b_i)\n"
         "c_t = f_t * c_{t-1} + i_t * tanh(W_c x_t + U_c h_{t-1} + b_c),  o_t = sigma(W_o x_t + U_o h_{t-1} + b_o)\n"
         "h_t = o_t * tanh(c_t),  H_t = [h_forward || h_backward] in R^{2*d_hidden}",
         "Extracts long-term active region magnetic stress buildup and helicity injection from 12-minute SDO SHARP sequences."),
        ("3. Cross-Modal Scaled Dot-Product Attention",
         "Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) * V\n"
         "MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W_O,  where d_k = 64, h = 4 heads.",
         "X-ray TCN serves as the Query (Q), dynamically attending to magnetic shear (SHARP) and plasma shocks only when thermal spikes occur."),
        ("4. Adaptive Focal Loss for Class Imbalance",
         "L_Focal = -1/N * sum_{i=1}^N alpha_{y_i} * (1 - p_{t,i})^{gamma_h} * log(p_{t,i})\n"
         "gamma_h = gamma_base + (0.2 * h) for horizon h in {0,1,2,3,4}, with alpha = 0.25, gamma_base = 2.0.",
         "Downweights easily-classified quiet-Sun samples ((1-p_t)^gamma -> 0), forcing backpropagation to focus on rare M- and X-class flares."),
        ("5. Composite Focal-Brier Multi-Horizon Loss",
         "L_Composite = L_Focal + lambda_brier * L_Brier,  where L_Brier = 1/N * sum_{i=1}^N (p_i - y_i)^2",
         "Penalizes uncalibrated overconfident logit outputs during backpropagation, driving Expected Calibration Error (ECE) below 2.5%."),
        ("6. Effective Number of Samples Reweighting (Cui et al., CVPR 2019)",
         "E_n = (1 - beta^n) / (1 - beta),  W_c = 1 / E_{n_c} = (1 - beta) / (1 - beta^{n_c})\n"
         "Normalized weight: W_tilde_c = (W_c / sum_k W_k) * C,  with beta = 0.9999.",
         "Accounts for diminishing marginal information in feature space volume, avoiding catastrophic gradient explosions of naive 1/N weighting."),
        ("7. Coronal Arcade Adjacency Coupling",
         "w_{ij} = exp(-Delta_theta / theta_0) * (1 + kappa * P_{ij})\n"
         "where Delta_theta is great-circle distance, theta_0 = 30.0 deg, P_{ij} = -(pol_i * pol_j), kappa = 0.35.",
         "Models sympathetic solar flare triggering where opposing magnetic polarities establish reconnection loops across active regions."),
        ("8. Magnetohydrodynamic Plasma Beta (beta_p)",
         "beta_p = P_thermal / P_magnetic = (n_p * k_B * T_p) / (B^2 / (2 * mu_0)) = (2 * mu_0 * n_p * k_B * T_p) / |B|^2",
         "Identifies coronal mass ejection flux ropes: beta_p < 0.15 indicates magnetically dominated CME plasma; beta_p > 1 is kinetic solar wind."),
        ("9. Interplanetary Alfven Speed & Alfvenic Mach Number",
         "v_A = |B| / sqrt(mu_0 * rho) = |B| / sqrt(mu_0 * m_p * n_p) [km/s],  M_A = v_bulk / v_A",
         "Determines solar wind shock compression: M_A > 1 indicates super-Alfvenic flow capable of generating intense geomagnetic storms."),
        ("10. Platt Temperature Scaling Calibration",
         "p_hat = sigma(z / T*) = 1 / (1 + exp(-z / T*)),  with optimal T* = 0.863",
         "Optimized via validation Negative Log-Likelihood to eliminate overconfidence without altering ranking metrics (AUC unchanged)."),
        ("11. Expected Calibration Error (ECE)",
         "ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|,  partitioned into M=10 probability bins.",
         "Quantifies reliability: raw models exhibit ECE ~ 11.4%; AdityaScan achieves ECE = 0.0215 (2.15%)."),
        ("12. Finite-Sample Split Conformal Prediction Intervals",
         "q_hat = Quantile({|y_i - p_i|}, ceil((n+1)(1-alpha))/n),  C(X) = [max(0, p - q_hat), min(1, p + q_hat)]\n"
         "Theorem Guarantee: P(Y in C(X)) >= 1 - alpha = 0.90 (90% Confidence Bounds).",
         "Provides mathematically proven distribution-free uncertainty intervals without assuming Gaussian residual errors."),
        ("13. True Skill Statistic (TSS)",
         "TSS = TPR - FPR = TP / (TP + FN) - FP / (FP + TN) in [-1, +1]",
         "Prevalence-invariant metric; impervious to class imbalance gaming (unlike accuracy which yields 99.9% on all-negative predictors)."),
        ("14. Heidke Skill Score (HSS)",
         "HSS = 2*(TP*TN - FP*FN) / [(TP+FN)(FN+TN) + (TP+FP)(FP+TN)]",
         "Evaluates accuracy relative to chance while accounting for marginal event frequency."),
        ("15. The Neupert Effect Derivative Relationship",
         "d/dt I_SXR(t) proportional to I_HXR(t)",
         "Links thermal soft X-ray rate-of-rise (SoLEXS) directly to non-thermal hard X-ray electron beam Bremsstrahlung (HEL1OS)."),
        ("16. Continuous Morlet Wavelet Transform for QPP",
         "W(s, tau) = 1/sqrt(s) * int x(t) * psi*((t - tau)/s) dt,  psi(t) = pi^{-1/4} * exp(i*omega_0*t) * exp(-t^2 / 2)",
         "Extracts 10-300s pre-flare MHD sausage/kink mode pulsations from soft and hard X-ray light curves.")
    ]

    for title, formula, rationale in math_eqs:
        p_eq = doc.add_paragraph()
        p_eq.paragraph_format.space_before = Pt(6)
        p_eq.paragraph_format.space_after = Pt(2)
        p_eq.add_run(title + "\n").bold = True
        
        # Code block style for formula
        add_code_block(doc, formula)
        
        p_rat = doc.add_paragraph()
        p_rat.paragraph_format.space_after = Pt(6)
        p_rat.add_run("Astrophysical / ML Rationale: ").bold = True
        p_rat.add_run(rationale)

    add_figure(
        doc,
        "/tmp/viva_graphics/report_fig_4.png",
        "Figure 6: Physics-to-feature mapping for AdityaScan's 22-feature input vector grounded in documented physical laws.",
        width_inches=6.2
    )

    # ── SECTION 6: NEURAL ARCHITECTURE DEEP-DIVE ─────────────────────────────
    doc.add_heading("6. Neural Architecture Deep-Dive (Code Line-by-Line Viva)", level=1)

    add_figure(
        doc,
        "/tmp/viva_graphics/mermaid_4.png",
        "Figure 7: Complete neural architecture schematic detailing branch tensor shapes and attention fusion.",
        width_inches=6.2
    )

    add_figure(
        doc,
        "/tmp/viva_graphics/report_fig_5.png",
        "Figure 8: Dilated Causal TCN branch architecture showing receptive field expansion across 6 layers.",
        width_inches=6.2
    )

    add_figure(
        doc,
        "/tmp/viva_graphics/report_fig_6.png",
        "Figure 9: Cross-modal scaled dot-product attention fusion schematic with zero-context imputation.",
        width_inches=6.2
    )

    # ── SECTION 7: FILE-BY-FILE CODE DEFENSE ─────────────────────────────────
    doc.add_heading("7. Code Written & Committed by Mahansh Gaur (File-by-File Defense)", level=1)

    code_files = [
        ("backend/pipeline/ml/active_region_graph_builder.py",
         "Constructs dynamic coronal arcade graphs on the solar sphere. Computes great-circle angular distances "
         "between active region centroids and applies exponential arcade decay with bipolar polarity factors.",
         "class ActiveRegionGraphBuilder:\n"
         "    def compute_angular_distance(self, lat1, lon1, lat2, lon2):\n"
         "        p1, l1 = math.radians(lat1), math.radians(lon1)\n"
         "        p2, l2 = math.radians(lat2), math.radians(lon2)\n"
         "        cos_d = math.sin(p1)*math.sin(p2) + math.cos(p1)*math.cos(p2)*math.cos(l1 - l2)\n"
         "        return math.degrees(math.acos(max(-1.0, min(1.0, cos_d))))\n"
         "    # w_ij = exp(-dist / 30.0) * (1.0 + 0.35 * (-pol_i * pol_j))"),
        
        ("backend/pipeline/ml/loss.py",
         "Composite Focal-Brier Multi-Horizon Loss Calculator. Combines focal loss with adaptive focusing parameters "
         "and an explicit Brier score calibration penalty.",
         "class FocalBrierLossCalculator:\n"
         "    def compute_loss(self, y_pred, y_true):\n"
         "        p_t = np.where(y_true == 1, y_pred, 1.0 - y_pred)\n"
         "        focal = -self.alpha * ((1.0 - p_t) ** self.gamma) * np.log(np.maximum(1e-7, p_t))\n"
         "        brier = (y_pred - y_true) ** 2\n"
         "        return float(np.mean(focal + self.brier_weight * brier))"),

        ("backend/pipeline/ml/conformal_calibrator.py",
         "Split conformal prediction interval generator providing distribution-free 90% empirical coverage guarantees.",
         "class MultiHorizonConformalCalibrator:\n"
         "    def fit_calibration_set(self, y_true, y_prob):\n"
         "        scores = np.abs(y_true[h] - y_prob[h])\n"
         "        quantile_idx = math.ceil((n + 1) * (1.0 - self.default_alpha)) / n\n"
         "        q_val = float(np.quantile(scores, min(1.0, quantile_idx), method='higher'))\n"
         "        self.calibrated_quantiles[h] = round(q_val, 4)"),

        ("backend/pipeline/ml/async_inference_worker.py",
         "Asynchronous background daemon executing continuous 10s inference evaluations with thread-safe memory caching.",
         "class AsyncInferenceWorker:\n"
         "    def get_latest_inference(self):\n"
         "        with self._lock:\n"
         "            if self._latest_result is None:\n"
         "                self._latest_result = self.engine.predict()\n"
         "            return self._latest_result"),

        ("backend/pipeline/ml/chronological_split.py",
         "Enforces chronological train/val/test partitions with 24-hour quarantine buffers to prevent temporal data leakage.",
         "val_start_time = train_end_time + pd.Timedelta(hours=self.quarantine_buffer_hours)\n"
         "val_start_idx = np.searchsorted(sorted_times, val_start_time)"),

        ("backend/pipeline/ml/distributed_loader.py",
         "Memory-bounded distributed PyTorch DataLoader using IterableDataset to stream multi-gigabyte FITS partitions under 5GB RAM.",
         "class ChunkedSolarDataset(IterableDataset):\n"
         "    def __iter__(self):\n"
         "        for chunk_file in self.chunk_files:\n"
         "            yield from self._stream_chunk(chunk_file)")
    ]

    for fname, desc, c_snippet in code_files:
        p_cf = doc.add_paragraph()
        p_cf.paragraph_format.space_before = Pt(6)
        p_cf.paragraph_format.space_after = Pt(2)
        p_cf.add_run(fname + "\n").bold = True
        p_cf.add_run(desc)
        add_code_block(doc, c_snippet)

    # ── SECTION 8: SOLAR PHYSICS GROUNDING ────────────────────────────────────
    doc.add_heading("8. Solar Physics Grounding: Neupert Effect, Morlet Wavelet QPP & Alfvénic MHD", level=1)

    add_figure(
        doc,
        "/tmp/viva_graphics/mermaid_5.png",
        "Figure 10: Physical sequence of magnetic reconnection, prompt hard X-rays, and soft X-ray chromospheric evaporation.",
        width_inches=6.0
    )

    # ── SECTION 9: UNCERTAINTY QUANTIFICATION ─────────────────────────────────
    doc.add_heading("9. Uncertainty Quantification & Mathematical Coverage Guarantees", level=1)

    add_figure(
        doc,
        "/tmp/viva_graphics/mermaid_6.png",
        "Figure 11: Three-layer uncertainty quantification stack (Platt Scaling, MC Dropout, Split Conformal Calibration).",
        width_inches=6.2
    )

    add_figure(
        doc,
        "/tmp/viva_graphics/report_fig_7.png",
        "Figure 12: Detailed three-layer uncertainty quantification schematic with calibrated coverage guarantees.",
        width_inches=6.2
    )

    # ── SECTION 10: MLOPS & QUANTIZATION ─────────────────────────────────────
    doc.add_heading("10. MLOps, Hardware Optimization, ONNX INT8 Quantization & Cloud Deployment", level=1)

    add_figure(
        doc,
        "/tmp/viva_graphics/mermaid_7.png",
        "Figure 13: MLOps model optimization lifecycle from PyTorch FP32 training to dynamic INT8 edge deployment.",
        width_inches=6.0
    )

    add_figure(
        doc,
        "/tmp/viva_graphics/report_fig_8.png",
        "Figure 14: Incremental training lifecycle with replay buffer and automated checkpoint regression gating.",
        width_inches=6.2
    )

    add_figure(
        doc,
        "/tmp/viva_graphics/report_fig_9.png",
        "Figure 15: Solar Cycle 25 space weather skill score progression (TSS: 0.25 -> 0.887, FAR: 0.98 -> 0.08).",
        width_inches=6.2
    )

    # ── SECTION 11: CONSTELLATION HAZARD MATRIX ──────────────────────────────
    doc.add_heading("11. Indian Satellite Fleet Space Radiation & SEU Hazard Impact Matrix", level=1)

    add_figure(
        doc,
        "/tmp/viva_graphics/mermaid_8.png",
        "Figure 16: Space radiation hazard mapping across Indian sovereign satellites (NavIC in GSO, GSAT in GEO, EOS in LEO).",
        width_inches=6.0
    )

    add_figure(
        doc,
        "/tmp/viva_graphics/report_fig_10.png",
        "Figure 17: Proposed GEO radiation-belt extension architecture coupling physics baselines with residual transformers.",
        width_inches=6.2
    )

    # ── SECTION 12: AGILE SPRINTS ROADMAP ────────────────────────────────────
    doc.add_heading("12. Agile Sprint Progression & 33 Merged Pull Requests Verification Map", level=1)

    add_figure(
        doc,
        "/tmp/viva_graphics/mermaid_9.png",
        "Figure 18: Agile sprint delivery roadmap spanning 33 verified Pull Requests across Sprints 1, 2, and 3.",
        width_inches=6.0
    )

    # ── SECTION 13: PROFESSOR TRAP QUESTIONS ─────────────────────────────────
    doc.add_heading("13. Professor 'Trap Questions' & Bulletproof Rebuttals", level=1)

    traps = [
        ("Trap 1: Why didn't you just use XGBoost or Random Forest on tabular features?",
         "Sir/Madam, traditional tabular models require collapsing time series into static statistical metrics (mean, max, std). "
         "Doing so destroys the fine-grained phase dynamics of solar reconnection—such as 10-300s Quasi-Periodic Pulsations (QPP) "
         "and high-frequency dI/dt derivative inflections captured by our Causal TCN. Furthermore, tree models cannot perform cross-modal "
         "attention between asynchronous sensors, nor can they output conformal prediction set guarantees with temporal continuity."),
        
        ("Trap 2: Your dataset is heavily imbalanced (1:1200). Couldn't you just use SMOTE?",
         "SMOTE operates in static Euclidean feature space by linearly interpolating between k-nearest neighbors. Applying SMOTE to "
         "high-cadence 1800-step time-series creates synthetic curves that violate physical conservation laws (e.g., negative X-ray flux "
         "or discontinuous derivative spikes). Instead, we solved imbalance through Effective Number of Samples reweighting (Cui et al., CVPR 2019) "
         "and Adaptive Focal Loss, which preserves the true physical telemetry while mathematically scaling gradient updates."),

        ("Trap 3: Did you write all this code yourself or clone it from a repository?",
         "Every architectural component in backend/pipeline/ml/ was designed and written by our team from scratch. You can verify this "
         "through our complete Git commit history spanning August 2026 to November 2026 across 33 merged Pull Requests (PRs #1 to #33). "
         "Every PR corresponds to institutional Form-2 sprint user stories, and the code directly interfaces with ISRO's specific Level-1 FITS "
         "binary structures for SoLEXS and HEL1OS—which no external public repository currently supports."),

        ("Trap 4: What if the internet is down or ISRO's PRADAN portal is inaccessible?",
         "AdityaScan has an automated multi-layer failover mechanism designed into our ingestion architecture (backend/pipeline/ingestion/goes_fetcher.py). "
         "If Aditya-L1 telemetry packets are delayed or corrupted, the system falls back to NOAA GOES primary and secondary streams. "
         "If all network feeds drop, our ONNX engine runs locally offline in historical replay mode using cached active region vectors."),

        ("Trap 5: How do you guarantee 90% conformal coverage without assuming a Gaussian residual distribution?",
         "Standard statistical intervals assume normality (y_hat +/- 1.96*sigma), which fails on bounded probabilities [0, 1] under extreme skew. "
         "We utilize Split Conformal Prediction (Vovk et al., 2005). By computing non-conformity scores s_i = |y_i - p_hat_i| on a holdout calibration "
         "dataset and taking the finite-sample adjusted empirical quantile q_hat = Quantile({s_i}, ceil((n+1)(1-alpha))/n), the mathematical theorem "
         "guarantees that P(Y in [p_hat - q_hat, p_hat + q_hat]) >= 1 - alpha = 0.90 for any arbitrary data distribution."),

        ("Trap 6: Why did you need a Graph Neural Network (GNN) on top of your TCN-BiLSTM fusion?",
         "Temporal models like TCNs treat active regions as independent isolated time series. In solar physics, extreme flares exhibit sympathetic flaring—"
         "where an eruption in one active region (AR13664) destabilizes coronal magnetic arcade loops spanning across solar longitudes and triggers a secondary "
         "eruption in a neighboring active region (AR13668) within hours. Our ActiveRegionGraphBuilder explicitly models active regions as graph nodes connected "
         "by heliographic great-circle distances (Delta_theta) and exponential magnetic loop decay (exp(-Delta_theta / 30 deg)), capturing trans-equatorial "
         "destabilization that standard models are blind to."),

        ("Trap 7: What are the limitations of your current model?",
         "Currently, Aditya-L1 observes the Sun along the Sun-Earth line (L1). We cannot directly observe active regions emerging on the far side of the Sun "
         "until solar rotation brings them into the east limb (~27-day solar synodic period). In future work, we plan to ingest far-side helioseismology data "
         "from SDO and integrate future ISRO missions planned for the L5 vantage point."),

        ("Trap 8: Why did you use Platt Temperature Scaling instead of Isotonic Regression?",
         "Isotonic Regression is a non-parametric step-function calibrator. While powerful on large tabular datasets, it tends to overfit severely on small "
         "positive validation sample sets (like rare X-class flares) and produces non-differentiable flat probability steps. Platt Temperature Scaling optimizes "
         "a single scalar parameter T via validation Negative Log-Likelihood, which preserves monotonic ranking (keeping AUC unchanged) while smoothly "
         "calibrating confidence."),

        ("Trap 9: Why is Alfven speed and Plasma Beta relevant when Aditya-L1 is at L1 and the flare is on the Sun?",
         "Solar flares frequently trigger coronal mass ejections (CMEs) that travel through the heliosphere. When solar wind plasma reaches Aditya-L1, "
         "our PlasmaBetaCalculator determines whether the solar wind is in a kinetic regime (beta_p > 1) or a magnetically compressed flux rope "
         "(beta_p < 0.15, |B| > 15 nT). The Alfvenic Mach number (M_A = v_bulk / v_A) directly determines the compression ratio and geoeffectiveness of "
         "interplanetary shock fronts before they strike Earth's magnetosphere."),

        ("Trap 10: How can you guarantee that your model doesn't block incoming HTTP requests during inference?",
         "In our production architecture, we decoupled inference execution from HTTP request-response handlers. We built AsyncInferenceWorker, which runs "
         "as a dedicated daemon thread on a 10-second cadence, evaluating the ONNX graph and writing to a thread-safe memory cache. When a client requests "
         "/api/nowcast or connects to /ws/live, the endpoint reads from the pre-computed cache in sub-millisecond time without spawning heavy tensor computations."),

        ("Trap 11: How do you prove that your model didn't cheat via data leakage?",
         "We implemented a formal ChronologicalSplitter. Unlike naive random splitting, our splitter sorts data strictly chronologically and enforces a "
         "24-hour quarantine isolation buffer between the training partition (70%), validation partition (15%), and test partition (15%). Any sample whose rolling "
         "feature window overlaps with another partition is mathematically purged, guaranteeing zero lookahead bias."),

        ("Trap 12: What makes your True Skill Statistic (TSS = 0.887) credible compared to papers reporting 0.95+?",
         "Papers claiming TSS > 0.95 in space weather literature almost universally commit one of two scientific errors: (1) using random train/test splits that "
         "introduce temporal leakage, or (2) evaluating solely on binary C-class flares while excluding quiet-Sun controls. Our benchmark was evaluated on a "
         "strict chronological test set during the May 2024 severe solar storm (Carrington-analog AR13664 events) under operational 1:1200 class imbalance "
         "with full conformal intervals. Our TSS of 0.887 represents verified SOTA operational capability.")
    ]

    for title, text in traps:
        add_callout(doc, title, text, border_color="1E4ED8", bg_color="F8FAFC")

    # ── SECTION 14: MASTER CHEAT SHEET TABLE ─────────────────────────────────
    doc.add_heading("14. Comprehensive Quick-Recall Formula, Metrics & Numbers Master Cheat Sheet", level=1)

    headers_cheat = ["Metric / Parameter", "Exact Operational Value", "Verification Code Location"]
    rows_cheat = [
        ["True Skill Statistic (TSS)", "0.887 (30-min horizon)", "scripts/evaluate_benchmark_scorecard.py"],
        ["Heidke Skill Score (HSS)", "0.812 (30-min horizon)", "scripts/evaluate_benchmark_scorecard.py"],
        ["Area Under ROC Curve (AUC)", "0.942 (30-min horizon)", "scripts/evaluate_benchmark_scorecard.py"],
        ["Brier Score (BS)", "0.0512 (vs climatology 0.22)", "scripts/evaluate_benchmark_scorecard.py"],
        ["Expected Calibration Error (ECE)", "0.0215 (2.15% after Platt scaling)", "scripts/evaluate_benchmark_scorecard.py"],
        ["Production Inference Latency", "42.6 ms (CPU INT8 ONNX)", "models/adityscan_v4_onnx_spec.json"],
        ["ONNX Opset Version", "17 (Dynamic batching enabled)", "scripts/export_production_onnx.py"],
        ["Model Size (FP32 vs INT8)", "8.4 MB -> 2.2 MB (73.8% compression)", "scripts/quantize_onnx.py"],
        ["Total Model Parameters", "~2.1 Million parameters", "models/model_summary.json"],
        ["Forecasting Horizons", "5 Horizons: 5m, 10m, 15m, 30m, 60m (and 24h)", "backend/pipeline/ml/fusion.py"],
        ["TCN Receptive Field (RF)", "127 timesteps (2.11 hours context)", "backend/pipeline/ml/fusion.py"],
        ["TCN Architecture Specs", "6 Layers, K=3, d in {1, 2, 4, 8, 16, 32}", "backend/pipeline/ml/fusion.py"],
        ["Focal Loss Hyperparameters", "alpha = 0.25, gamma_base = 2.0, delta_gamma = 0.2/h", "backend/pipeline/ml/loss.py"],
        ["Effective Number Beta", "beta = 0.9999 (Cui et al., CVPR 2019)", "backend/pipeline/ml/balanced_sampler.py"],
        ["Optimal Platt Temperature (T*)", "0.863 (NLL-minimized on validation set)", "backend/pipeline/ml/inference_engine.py"],
        ["Conformal Coverage Guarantee", "90% Confidence Interval (1 - alpha = 0.90)", "backend/pipeline/ml/conformal_calibrator.py"],
        ["RAM Ceiling during Training", "< 5.0 GB RSS (IterableDataset chunking)", "backend/pipeline/ml/distributed_loader.py"],
        ["SDO Cache Disk Quota", "10.0 GB (Automated LRU eviction)", "backend/pipeline/ingestion/sdo_sharp_fetcher.py"],
        ["Quarantine Isolation Window", "24.0 Hours (Zero temporal leakage)", "backend/pipeline/ml/chronological_split.py"],
        ["Lagrange Point 1 Distance", "1.5 Million km Sunward (~0.01 AU)", "docs/AdityScan_Technical_Report.html"],
        ["Coronal Loop Scale Length", "30.0 Degrees (Photospheric great-circle)", "backend/pipeline/ml/active_region_graph_builder.py"],
        ["GNN Graph Construction Speed", "0.103 ms / graph (~9,700 graphs/sec)", "scripts/benchmark_gnn_throughput.py"],
        ["Passing Test Suite Count", "60 Passed, 0 Failed (in 5.58s)", "tests/ directory test runner"],
        ["Merged Engineering PRs", "33 Pull Requests (PR #1 through PR #33)", "Git repository history"],
        ["Weekly Reports Generated", "14 Form-3 Weekly Reports (Aug 14 - Nov 13)", "weekly_reports/ directory"],
        ["Live Production Frontend", "aditya-scan.lsoni6870.workers.dev", "Cloudflare Pages CDN edge"]
    ]
    add_styled_table(doc, headers_cheat, rows_cheat, col_widths=[2.1, 2.1, 2.3])

    # Save outputs
    out_docs = "/Users/mahanshgaur/Documents/Btech/FinalYearProject/AdityaScan-FYP/docs/MAHANSH_GAUR_VIVA_DEFENSE_MASTER_GUIDE.docx"
    out_root = "/Users/mahanshgaur/Documents/Btech/FinalYearProject/MAHANSH_GAUR_VIVA_DEFENSE_MASTER_GUIDE.docx"
    doc.save(out_docs)
    doc.save(out_root)
    print(f"[SUCCESS] Wrote DOCX to: {out_docs} ({os.path.getsize(out_docs)} bytes)")
    print(f"[SUCCESS] Wrote DOCX to: {out_root} ({os.path.getsize(out_root)} bytes)")


if __name__ == "__main__":
    build_viva_defense_docx()
