# Progress / حالة الإنجاز

**Voice of Customer — Wadeen Khalid Al adwani**

| Deliverable | Status | Evidence |
| --- | --- | --- |
| Formal capstone requirements reviewed | Complete | Official academy capstone at commit d1325e6d3d32c793e6354517ee5695d44cfa7c5e |
| Dataset, provenance and fixed splits | Complete, synthetic limitations disclosed | DATA_CARD.md / data/feedback.csv |
| Baseline and improvement | Trained and measured on same 66 test rows | reports/training.json / reports/metrics.json |
| Embeddings, grouping and traceable summaries | Local pipeline exercised | sample_outputs/analysis.json / reports/clustering.json |
| Slice evaluation and ≥10 failures | Complete: 28 actual errors categorized | EVALUATION_REPORT.md / reports/error_analysis.csv |
| Fine-tuning alternative | Written evidence-based justification | DECISIONS.md / reports/learning_curve.csv |
| Functional tests | 22 passed locally | tests/test_project.py / tests/test_app.py / reports/test_validation.json |
| Live Gradio requests | Text + uploaded CSV + empty-input rejection passed | reports/demo_smoke.json |
| Notebook execution | 7 + 3 code cells executed, 0 errors | reports/notebook_validation.json |
| Presentation | 8 slides and 10-minute speaking notes prepared | PRESENTATION.md / reports/presentation.html |
| Reproducibility | Lockfile, hashed requirements, scripts and setup instructions | uv.lock / GETTING_STARTED.md |
| Student review and timed rehearsal | Pending student action | Read and understand code, results and limitations |
| Recorded video backup | Not created; checkable notebook/JSON backup supplied | sample_outputs/README.md |
| Instructor submission and grading | Not performed or claimed | Student submits repository link |

The project is a runnable educational prototype. It is not validated for production or actual customer data. No fine-tuning execution, real-review results, recorded presentation, course completion or instructor acceptance is claimed. GitHub upload status is reported in the accompanying chat after actual remote verification; local evidence does not establish remote CI execution.
