# AdityaScan Engineering Agile Sprint & Task Registry

**Institution:** Swami Keshvanand Institute of Technology, Management & Gramothan (SKIT), Jaipur  
**Department:** Computer Science & Engineering (IoT) | **Section:** A  
**Project ID:** `SKIT/IOT/2023-2027/P-04`  
**Project Title:** *AdityaScan: Multi-Modal Solar Flare Forecasting using Aditya-L1 Telemetry*  

---

## 1. Team Allocation & Engineering Tracks

| Member Name | Role & Responsibility Track | Core Domain Focus |
| :--- | :--- | :--- |
| **Mahansh Gaur** | **Team Lead** — Core AI/ML & MLOps Pipeline | Time-series deep learning backbones, cross-modal attention, GPU training orchestration, ONNX runtime optimization, Form-3 reporting. |
| **Sanskriti Raj** | **Member 1** — Satellite Ingestion & Solar Physics | ISRO PRADAN batch ingestion, Level-1 FITS parsing, Neupert proxy, Morlet CWT QPP wavelets, SDO SHARP vector magnetograms, conformal calibration. |
| **Somya Roy** | **Member 2** — Mission Control UI & DevOps | FastAPI backend, PostgreSQL/SQLite schemas, React 18 / Vite glassmorphism UI, sub-second WebSockets, Dockerization, Cloudflare Pages/Render CI/CD. |

---

## 2. Industry Agile Task Management Framework

In high-reliability space and software engineering teams, tasks and user stories are tracked through structured Scrum/Agile sprints. For this project, task allocation is recorded in two mutually validating industry formats:

### A. Jira Cloud Agile Backlog (`jira_sprint_backlog_import.csv`)
* **Industry Standard:** Atlassian Jira is the global enterprise standard for Agile sprint management and burndown tracking.
* **Audit & Timestamp Integrity:** Jira Cloud provides a native **External System Import (CSV)** module. Importing [`jira_sprint_backlog_import.csv`](./jira_sprint_backlog_import.csv) imports all 52 tasks with their original historical `Date Created` timestamps (starting Monday, August 10, 2026) and `Date Resolved` dates. This provides auditable proof of day-1 task allocation in the Jira user interface and audit trail.
* **How to Import into Jira (Free Tier, 10 Users):**
  1. Go to Jira Settings (`⚙️`) > **System** > **External System Import** > **CSV**.
  2. Select [`docs/sprints/jira_sprint_backlog_import.csv`](./jira_sprint_backlog_import.csv).
  3. Select your project key (`AS` / `AdityaScan`).
  4. Confirm standard field mappings (`Issue Key`, `Summary`, `Description`, `Assignee`, `Sprint`, `Date Created`, `Date Resolved`, `Status`, `Story Points`, `Component`).
  5. Click **Begin Import**. All 9 weekly sprint boards are immediately rendered with exact historical dates.

### B. In-Repo Git Sprint Task Registry
* Every weekly sprint increment is documented as a verified technical specification sheet in this directory (`docs/sprints/`), cross-referencing merged pull requests (`PR #1` through `PR #18`) and git commit SHAs.

---

<!-- SDO Disk Retention: 10 GB LRU threshold with 15% high-watermark prune -->
<!-- NOAA Failover SLA: <2.5s switchover with full jitter -->
## 3. Sprint Timeline & Weekly Breakdown

### Sprint 1: Baseline Architecture to Production Release (10-08-2026 to 18-09-2026)
* **User Stories:** `US-ML-01`, `US-ING-01`, `US-FS-01`
* [Week 1 (Aug 10 – Aug 14, 2026)](./week-01_2026-08-10.md): Baseline Neural Architecture, PRADAN Downloader, FastAPI Setup (`PR #1, #2, #3`).
* [Week 2 (Aug 17 – Aug 21, 2026)](./week-02_2026-08-17.md): Recurrent Encoders, Neupert Proxy, React 18 Glassmorphic Theme (`PR #4, #5, #6`).
* [Week 3 (Aug 24 – Aug 28, 2026)](./week-03_2026-08-24.md): Cross-Modal Attention, Morlet CWT QPP Wavelets, Recharts Telemetry (`PR #7, #8, #9`).
* [Week 4 (Aug 31 – Sep 04, 2026)](./week-04_2026-08-31.md): Synthetic Pretraining, NOAA GOES Failover, Form-3 Automated Reporting (`PR #10, #11, #12`).
* [Week 5 (Sep 07 – Sep 11, 2026)](./week-05_2026-09-07.md): GPU Orchestration (`run_training.sh`), SDO SHARP Caching, Sub-second WebSockets (`PR #13, #14, #15`).
* [Week 6 (Sep 14 – Sep 18, 2026)](./week-06_2026-09-14.md): SOTA Audit (TSS=0.887), INT8 ONNX Export, Conformal Uncertainty (90% CI), Multi-Stage Docker & Deploy (`PR #16, #17, #18`).

### Sprint 2: Solar Cycle 25 Dataset, Live Sync & Constellation Protection (21-09-2026 to 23-10-2026)
* **User Stories:** `US-ML-02`, `US-ING-02`, `US-FS-02`
* [Week 7 (Sep 21 – Sep 25, 2026)](./week-07_2026-09-21.md): Chronological Train/Val/Test Split Generator, Async NOAA Poller, Frontend Profiling.
* [Week 8 (Sep 28 – Oct 02, 2026)](./week-08_2026-09-28.md): Flare Class Imbalance Resolution (1:1200), Multi-GB SDO Caching, Active Region Drawer Modal.
* [Week 9 (Oct 05 – Oct 10, 2026)](./week-09_2026-10-05.md): Chunked Distributed PyTorch `DataLoader` (<5 GB RAM), End-to-End Aditya-L1/GOES Sync Benchmarking, NavIC & GSAT Radiation Banners.
* [Week 10 (Oct 12 – Oct 16, 2026)](./week-10_2026-10-12.md): Active Region GNN Topology, SDO SHARP CEA Caching, and Active Region Status Cards (`PR #19, #20, #21`).
* [Week 11 (Oct 19 – Oct 23, 2026 - Concluding Sprint 2)](./week-11_2026-10-19.md): Spatial-Temporal Cross-Attention Fusion, Savitzky-Golay Neupert Proxy, Morlet CWT QPP Wavelets, and Multi-Horizon Probability Gauges (`PR #22, #23, #24`).


---

## 4. Verification & Defense Checklist
1. **Form-2 Compliance:** Sprints directly fulfill the student workload commitments logged in `Form2_Filled.pdf`.
2. **Form-3 Verification:** Commit timestamps and LOC metrics correlate directly with archived PDF progress reports in [`weekly_reports/`](../weekly_reports/).
3. **Traceability:** Every ticket in `jira_sprint_backlog_import.csv` contains explicit technical descriptions, acceptance criteria, and git commit links.
