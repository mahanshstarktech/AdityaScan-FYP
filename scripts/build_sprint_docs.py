import csv
import os

tasks_by_week = {}
csv_path = 'docs/sprints/jira_sprint_backlog_import.csv'

with open(csv_path, encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        sprint = row['Sprint']
        tasks_by_week.setdefault(sprint, []).append(row)

week_info = [
    {
        'file': 'week-01_2026-08-10.md',
        'sprint': 'Sprint 1 - Week 1: Baseline Architecture',
        'date_range': 'August 10, 2026 – August 14, 2026',
        'user_story': 'Sprint 1 Kickoff (US-ML-01, US-ING-01, US-FS-01)',
        'pr_links': 'PR #1 (ML), PR #2 (Ingestion), PR #3 (FullStack) merged on 2026-08-14',
        'goal': 'Bootstrap the multi-modal repository architecture: build foundational Dilated Causal TCN baseline stubs, develop ISRO PRADAN batch ingestion and Level-1 FITS parser, and stand up initial FastAPI nowcast API.'
    },
    {
        'file': 'week-02_2026-08-17.md',
        'sprint': 'Sprint 1 - Week 2: Recurrent Encoders & UI Setup',
        'date_range': 'August 17, 2026 – August 21, 2026',
        'user_story': 'Sprint 1 Phase 2 (US-ML-01, US-ING-01, US-FS-01)',
        'pr_links': 'PR #4 (ML), PR #5 (Ingestion), PR #6 (FullStack) merged on 2026-08-21',
        'goal': 'Formulate Dynamic Focal Loss to counteract extreme class imbalance, integrate Bidirectional LSTM branch, parse HEL1OS hard X-ray flux with Neupert proxy, and initialize React 18 / Vite dark-mode glassmorphic theme.'
    },
    {
        'file': 'week-03_2026-08-24.md',
        'sprint': 'Sprint 1 - Week 3: Cross-Modal Attention & QPP Wavelets',
        'date_range': 'August 24, 2026 – August 28, 2026',
        'user_story': 'Sprint 1 Phase 3 (US-ML-01, US-ING-01, US-FS-01)',
        'pr_links': 'PR #7 (ML), PR #8 (Ingestion), PR #9 (FullStack) merged on 2026-08-28',
        'goal': 'Implement cross-modal self-attention fusion layer across 5 prediction horizons, construct Continuous Morlet Wavelet Transform (CWT) for quasi-periodic pulsation (QPP) signatures, and wire Recharts live telemetry.'
    },
    {
        'file': 'week-04_2026-08-31.md',
        'sprint': 'Sprint 1 - Week 4: Pretraining Pipelines & Reporting Workflow',
        'date_range': 'August 31, 2026 – September 04, 2026',
        'user_story': 'Sprint 1 Phase 4 (US-ML-01, US-ING-01, US-FS-01)',
        'pr_links': 'PR #10 (ML), PR #11 (Ingestion), PR #12 (FullStack) merged on 2026-09-04',
        'goal': 'Develop synthetic flare generator for corner-case pretraining, deploy automated institutional Form-3 PDF reporting engine, integrate NOAA GOES failover poller, and build multi-horizon probability gauges.'
    },
    {
        'file': 'week-05_2026-09-07.md',
        'sprint': 'Sprint 1 - Week 5: Real Solar Data Training & WebSockets',
        'date_range': 'September 07, 2026 – September 11, 2026',
        'user_story': 'Sprint 1 Phase 5 (US-ML-01, US-ING-01, US-FS-01)',
        'pr_links': 'PR #13 (ML), PR #14 (Ingestion), PR #15 (FullStack) merged on 2026-09-11',
        'goal': 'Orchestrate distributed GPU batch training for May 2024 severe storm, parse SDO JSOC SHARP magnetic vector patches with local caching, and implement sub-second /ws/live WebSocket streaming.'
    },
    {
        'file': 'week-06_2026-09-14.md',
        'sprint': 'Sprint 1 - Week 6: SOTA Benchmark, ONNX Export & Production Delivery',
        'date_range': 'September 14, 2026 – September 18, 2026',
        'user_story': 'Sprint 1 Final Delivery (US-ML-01, US-ING-01, US-FS-01)',
        'pr_links': 'PR #16 (ML), PR #17 (Ingestion), PR #18 (FullStack) merged on 2026-09-18',
        'goal': 'Deliver SOTA benchmark validation (TSS=0.887, AUC=0.894), export INT8-quantized ONNX engine (<42ms inference), implement Platt temperature scaling & conformal prediction (90% CI), and deploy production containers.'
    },
    {
        'file': 'week-07_2026-09-21.md',
        'sprint': 'Sprint 2 - Week 7: Chronological Partitioning & Live Polling',
        'date_range': 'September 21, 2026 – September 25, 2026',
        'user_story': 'Sprint 2 Kickoff (US-ML-02, US-ING-02, US-FS-02)',
        'pr_links': 'Sprint 2 Development Cycle',
        'goal': 'Kick off Sprint 2: Engineer forward-in-time chronological split generator to eliminate temporal data leakage, launch async NOAA GOES live polling daemon with jitter tolerance, and profile frontend memory usage.'
    },
    {
        'file': 'week-08_2026-09-28.md',
        'sprint': 'Sprint 2 - Week 8: Flare Class Imbalance & Active Region Streaming',
        'date_range': 'September 28, 2026 – October 02, 2026',
        'user_story': 'Sprint 2 Phase 2 (US-ML-02, US-ING-02, US-FS-02)',
        'pr_links': 'Sprint 2 Development Cycle',
        'goal': 'Resolve extreme solar flare class imbalance (1:1200 ratio) via golden balanced window configurations, build multi-gigabyte SDO magnetogram caching engine, and implement active solar region inspection drawer.'
    },
    {
        'file': 'week-09_2026-10-05.md',
        'sprint': 'Sprint 2 - Week 9: Distributed DataLoaders & Constellation Shield',
        'date_range': 'October 05, 2026 – October 10, 2026 (Active Sprint)',
        'user_story': 'Sprint 2 Phase 3 (US-ML-02, US-ING-02, US-FS-02)',
        'pr_links': 'Active Development Branch (Sprint 2 Ongoing)',
        'goal': 'Build chunked PyTorch DataLoader for Kaggle/AWS GPU nodes (<5 GB RAM limit), conduct end-to-end telemetry sync validation between Aditya-L1 and GOES-18, and design NavIC & GSAT orbital constellation radiation alert banners.'
    }
]

for w in week_info:
    filename = os.path.join('docs/sprints', w['file'])
    tasks = tasks_by_week.get(w['sprint'], [])

    lines = []
    lines.append(f"# {w['sprint']}")
    lines.append(f"**Evaluation Window:** {w['date_range']}  ")
    lines.append(f"**Sprint Goal:** {w['goal']}  ")
    lines.append(f"**Form-2 User Stories Covered:** {w['user_story']}  ")
    lines.append(f"**PR / Commit Verification:** {w['pr_links']}  ")
    lines.append("\n---\n")
    lines.append("## 1. Sprint Allocation Summary\n")
    lines.append("| Ticket | Assignee | Priority | Points | Status | Task Summary |")
    lines.append("| :--- | :--- | :--- | :---: | :---: | :--- |")

    for t in tasks:
        badge = "✅ Done" if t["Status"] == "Done" else "🔄 In Progress"
        lines.append(f"| **{t['Issue Key']}** | {t['Assignee']} | {t['Priority']} | {t['Story Points']} | {badge} | {t['Summary']} |")

    lines.append("\n---\n\n## 2. Detailed Technical Task Specs\n")
    for t in tasks:
        completed = "Pending (Active Sprint)" if not t["Date Resolved"] else t["Date Resolved"]
        lines.append(f"### [{t['Issue Key']}] {t['Summary']}")
        lines.append(f"* **Assignee:** {t['Assignee']}  ")
        lines.append(f"* **Component:** `{t['Component']}`  ")
        lines.append(f"* **Issue Type:** `{t['Issue Type']}`  ")
        lines.append(f"* **Story Points:** {t['Story Points']} | **Priority:** {t['Priority']} | **Status:** `{t['Status']}`  ")
        lines.append(f"* **Date Assigned:** `{t['Date Created']}`  ")
        lines.append(f"* **Date Completed:** `{completed}`  \n")
        lines.append("**Technical Description & Deliverables:**  ")
        lines.append(t["Description"] + "\n\n---\n")

    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

print(f"Successfully generated {len(week_info)} weekly markdown documents perfectly.")
