"""Render human-readable reports from actual measured results, not hand-entered scores."""

import json
from collections import Counter
from pathlib import Path

import pandas as pd
from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parents[1]


def table(rows, columns):
    def clean(value):
        return str(value).replace('|', '/').replace('\n', ' ')
    return '\n'.join(['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |'] +
                     ['| ' + ' | '.join(clean(v) for v in row) + ' |' for row in rows])


def write(name, text):
    (ROOT / name).write_text(text.strip() + '\n', encoding='utf-8')


def main():
    metrics = json.loads((ROOT / 'reports/metrics.json').read_text())
    benchmark = json.loads((ROOT / 'reports/benchmark.json').read_text())
    training = json.loads((ROOT / 'reports/training.json').read_text())
    clustering = json.loads((ROOT / 'reports/clustering.json').read_text())
    slices = pd.read_csv(ROOT / 'reports/slices.csv')
    errors = pd.read_csv(ROOT / 'reports/error_analysis.csv')
    curve = pd.read_csv(ROOT / 'reports/learning_curve.csv')
    results = table([
        [name, m['n'], f"{m['accuracy']:.3f}", f"{m['macro_f1']:.3f}",
         f"[{m['macro_f1_95ci'][0]:.3f}, {m['macro_f1_95ci'][1]:.3f}]", f"{benchmark[name]['p95_ms']:.2f}"]
        for name, m in metrics.items() if name in ['baseline', 'improved']
    ], ['System', 'Test N', 'Accuracy', 'Macro-F1', '95% bootstrap CI', 'p95 single-comment ms'])
    slice_table = table([[r.system, r.slice, r.n, f'{r.accuracy:.3f}', f'{r.macro_f1:.3f}'] for r in slices.itertuples()],
                        ['System', 'Slice', 'N', 'Accuracy', 'Macro-F1'])
    error_counts = table(sorted(Counter(errors.error_category).items()), ['Category', 'Actual failures'])
    error_table = table([[r.id, r.text, r.label, r.improved_pred, r.error_category, r.analysis_and_next_step] for r in errors.itertuples()],
                        ['ID', 'Comment', 'Gold', 'Predicted', 'Cause', 'Analysis / next step'])
    learning_table = table([[r.train_n, f'{r.validation_macro_f1:.3f}', f'{r.fit_seconds:.4f}'] for r in curve.itertuples()],
                           ['Training N', 'Validation macro-F1', 'Fit seconds'])
    delta = metrics['paired_delta']
    write('EVALUATION_REPORT.md', f'''
# Evaluation report — تقرير التقييم

**Wadeen Khalid Al adwani** · SDAIA Academy · instructor: مضر محمد الفضلي · project 3

## Protocol and measured outcomes

198 authored synthetic comments: 102 train, 30 validation, 66 test. Labels are balanced within each split. Metrics below use the **same 66 fixed test rows** for both models, including difficult phenomena. Train-only fitting; C selected on validation from 0.5, 2, 8; selected C={training['selected_C']}. No test tuning, no test refit. The best validation macro-F1 was {max(c['validation_macro_f1'] for c in training['candidates']):.3f}.

{results}

Paired macro-F1 delta: **{delta['macro_f1']:.3f}**, 95% paired-bootstrap interval **[{delta['95ci'][0]:.3f}, {delta['95ci'][1]:.3f}]**. The interval includes zero: improvement is observed on this sample, **not demonstrated to be statistically reliable**. 500 row-resamples, seed 42, fixed three-class macro averaging. These exploratory intervals describe the authored corpus and do not establish population generalization.

The fixed confidence threshold 0.55 is an uncalibrated demonstration policy: accepted {metrics['review_policy']['accepted_n']}/66, review {metrics['review_policy']['review_n']}/66. Full-coverage model metrics are above; no selective score is invented for an empty accepted set. Human review is required for every test item in this run.

## Quantitative slices

{slice_table}

Some phenomenon slices contain fewer than all three labels: macro-F1 retains negative/neutral/positive with zero_division=0, so read it alongside accuracy and N. Gulf/standard labels refer to authored style, not verified linguistic annotation. English N=6, no English validation; language scores are unstable. No English superiority or fairness guarantee is claimed.

## Actual error analysis — {len(errors)} categorized failures

Every row below is a real incorrect improved-model prediction from `reports/predictions.csv`, not an invented example or a changed gold label. Categories use the dataset's predeclared phenomenon tag; these are hypothesis-driven analyst categories, not independently validated causes. All {len(errors)} errors are included, exceeding the ≥10 requirement.

{error_counts}

{error_table}

## Embeddings and summaries

Training TF-IDF → 32D SVD → L2 embeddings, explained variance {training['svd_explained_variance']:.3f}. Demo input: 15 comments, 5 KMeans clusters; cosine silhouette **{clustering['silhouette_cosine']:.3f}**. This is internal compactness only; no gold topic labels means topic correctness is **unmeasured**. Full batch runtime including prediction/embeddings/clustering/summary: {clustering['end_to_end_batch_ms']:.2f} ms in the measured run.

The deterministic summary recounts model predictions and quotes a masked representative comment. Counts are verified by the functional tests; summary interpretation still needs human review. No LLM-based generation quality or factuality metric is claimed.

## Learning-curve evidence for the adaptation decision

{learning_table}

Fixed selected C; stratified train subsets, validation-only scores. Scores are nonmonotonic, with tiny single samples per size, so they do not prove a scaling trend. See `DECISIONS.md` for the evidence-based no-fine-tuning justification explicitly allowed by the capstone. No encoder or LoRA fine-tuning was run.

## Reproducibility and evidence

- Source corpus SHA-256: `{training['data_sha256']}`.
- Numeric raw evidence: `reports/training.json`, `reports/metrics.json`, `reports/slices.csv`, `reports/predictions.csv`, `reports/error_analysis.csv`, `reports/learning_curve.csv`.
- Confusion matrix: `reports/confusion_matrices.png` (true labels on rows).
- Rebuild with `bash scripts/run_all.sh`; regenerate this report using `uv run --frozen python scripts/build_docs.py`.
- This is an educational prototype; real customer data, independent annotations and a new untouched test are required before deployment.
''')
    write('BENCHMARKS.md', f'''
# Benchmarks

Actual local measurements from `reports/benchmark.json`; Python {benchmark['python']}, {benchmark['platform']}, {benchmark['logical_cpus']} logical CPUs. CPU execution, single-comment batch size, 5 warmup predictions then 66 timed predictions using `time.perf_counter`. No GPU claim. Latency varies with system load; these are not service-level guarantees.

{results}

Classification p95 excludes server transport, CSV parsing, embeddings and clustering. The separate 15-comment full pipeline took {clustering['end_to_end_batch_ms']:.2f} ms; that single batch duration is **not p95**. Cosine silhouette was {clustering['silhouette_cosine']:.3f}; no gold topic accuracy measured.

Paid API cost per 1,000 comments: **USD 0** (no API calls). Compute, hosting, electricity and labor are **not included**. Training durations and validation curve are in `reports/learning_curve.csv`.

Reproduce via `bash scripts/run_all.sh`. Environment setup download time is not inference latency. Do not compare these CPU measurements to unmeasured GPU/transformer models.
''')
    write('README.md', f'''
# Voice of Customer Analyzer

**المتدربة:** Wadeen Khalid Al adwani

**المدرب:** مضر محمد الفضلي
**أكاديمية سدايا — المشروع رقم ٣ — تسليم فردي بموافقة المدرب وفق تأكيد المتدربة**

تحليل تعليقات العملاء بالعربية: **تصنيف المشاعر، تجميع المشكلات باستخدام embeddings، وتلخيص الرؤى مع تعليق ممثل لكل مجموعة**. يأخذ النظام نصوصًا أو CSV ويعرض المشاعر والموضوع والثقة وطلب المراجعة البشرية. الهدف تسهيل قراءة التعليقات المتكررة، والمقياس الرئيس Macro-F1 على مجموعة اختبار ثابتة.

**تنبيه عن النتائج:** البيانات تعليمية مؤلفة وليست تعليقات عملاء حقيقية. النظام ليس جاهزًا لقرارات تشغيلية مستقلة. تفاصيل استخدام المساعدة البرمجية موضحة في [STUDENT_PROFILE.md](STUDENT_PROFILE.md).

## تشغيل المشروع

```bash
bash scripts/run_all.sh
UV_CACHE_DIR=/workspace/.cache/uv uv run --frozen python app.py
```

Python 3.12 وuv. لا يحتاج مفتاح API أو GPU. الخطوات الكاملة وColab في [GETTING_STARTED.md](GETTING_STARTED.md). الدفتر الرئيس: [Voice_of_Customer_NLP_Wadeen_Khalid_Al_adwani.ipynb](Voice_of_Customer_NLP_Wadeen_Khalid_Al_adwani.ipynb).

## البيانات

| Split | Size | Language/style | Source |
| --- | ---: | --- | --- |
| Train | 102 | فصحى 45، خليجي 45، إنجليزي 12 | أمثلة أصلية تعليمية مؤلفة |
| Validation | 30 | فصحى 15، خليجي 15 | نفس المصدر، نصوص مختلفة |
| Test | 66 | فصحى 30، خليجي 30، إنجليزي 6 | اختبار ثابت يتضمن ظواهر صعبة |

الفئات: إيجابي / سلبي / محايد. لا تدخل التسميات أو شرائح اللغة إلى ميزات التنبؤ. فحص التكرار بعد التطبيع، والتدريب على train فقط، واختيار C على validation فقط؛ لا ندعي غياب التشابه الدلالي بين أمثلة المؤلف نفسه. راجعي [DATA_CARD.md](DATA_CARD.md).

## المنهج

```mermaid
flowchart LR
  A[نصوص أو CSV] --> B[إخفاء بريد وهاتف وتطبيع عربي]
  B --> C[Word + Char TF-IDF / Logistic Regression]
  B --> D[TF-IDF / SVD embeddings / L2]
  D --> E[KMeans issue clusters]
  C --> F[المشاعر والثقة والمراجعة]
  E --> G[ملخص أعداد وتعليق ممثل]
  F --> G
```

Baseline: unigram TF-IDF + Naive Bayes. Improvement: word bigrams + character n-grams + logistic classifier. Embeddings محلية كامنة باستخدام SVD **وليست Transformer**. أسماء المواضيع بقواعد كلمات، والملخصات قالبية مع شواهد من المدخلات؛ لا يوجد LLM. انظري [MODEL_CARD.md](MODEL_CARD.md).

## النتائج الفعلية على نفس الاختبار

{results}

التحسن {delta['macro_f1']:.3f} Macro-F1، لكن فاصل الثقة للفرق يشمل الصفر. وثقنا **{len(errors)} خطأ فعليًا مصنفًا**، وتقييمًا حسب اللغة واللهجة والظواهر. عند عتبة الثقة 0.55، جميع أمثلة الاختبار تحتاج مراجعة بشرية. تكلفة API صفر ولا تشمل الاستضافة. [التقييم الكامل](EVALUATION_REPORT.md) · [القياسات](BENCHMARKS.md).

Fine-tuning **لم يُنفذ**؛ اخترنا البديل الرسمي المسموح: تبرير مكتوب مدعوم بحجم البيانات ومنحنى التعلم والنتائج وحدودها. [DECISIONS.md](DECISIONS.md) يوضح الدليل وخطة التجربة القادمة دون اختلاق مقارنة Transformer.

## تنظيم الملفات مثل المثال

```text
data/                  feedback.csv + schema/provenance
notebooks/             exploration notebook
reports/               predictions, slices, errors, metrics, plots, smoke evidence
sample_outputs/        demo CSV + JSON outputs
scripts/               train, evaluate, run_all, smoke_test, build_docs
src/voice_of_customer/             text preprocessing + end-to-end pipeline
tests/                 split, privacy, input and measured-result checks
configs/final.json     model hyperparameters
app.py                 Gradio Arabic demo
BENCHMARKS.md          DATA_CARD.md       DECISIONS.md
EVALUATION_REPORT.md   GETTING_STARTED.md MODEL_CARD.md
PRESENTATION.md        PROGRESS.md        PROJECT_SUMMARY.json
README.md             STUDENT_PROFILE.md SUBMISSION.yml
Voice_of_Customer_NLP_Wadeen_Khalid_Al_adwani.ipynb
requirements-day1.txt  pyproject.toml     uv.lock
```

## المطابقة مع متطلبات الأكاديمية

| Requirement | Evidence |
| --- | --- |
| End-to-end pipeline + demo | `src/voice_of_customer/pipeline.py`, `app.py`, `reports/demo_smoke.json` |
| Representation + decision/generation | SVD embeddings + sentiment classifier + traceable count summaries |
| Baseline and improvement, same held-out test | `reports/predictions.csv`, `reports/metrics.json` |
| Fine-tuning OR evidence-based justification | `DECISIONS.md`, `reports/learning_curve.csv` |
| Slice evaluation + ≥10 categorized failures | `reports/slices.csv`, {len(errors)} rows in `reports/error_analysis.csv` |
| Presentation and reproducibility | 8 slides in `PRESENTATION.md`, `reports/presentation.html`, lockfile, notebook, tests |

مرجع المتطلبات: [SDAIA Academy capstone](https://github.com/SDAIAAcademy/applied-nlp/tree/main/labs/capstone) عند commit `d1325e6d3d32c793e6354517ee5695d44cfa7c5e`. الصورة مرجع لتنظيم الملفات فقط، ولا تنسب عمل أصحابها إلى هذه المتدربة.

## حدود العمل والخطوة التالية

السخرية والنفي والأسئلة ذات الألفاظ العاطفية صعبة، والإنجليزية 6 أمثلة فقط. الموضوع بقواعد، ولا يوجد تقييم ذهبي لدقة التجميع. يلزم جمع بيانات حقيقية مصرح بها، مراجعة تسمياتها، تقسيم مستقل، معايرة الثقة ومقارنة encoder قبل النشر الإنتاجي. العرض الحي والتدريب على تقديمه مسؤولية المتدربة؛ ملفات المشروع لا تثبت تقديمه أو قبول المدرب.

ترخيص MIT للكود والأمثلة الأصلية. المصادر والقيود ومساهمة المساعد معلنة.
''')
    summary = {'project_number': 3, 'project': 'Voice of Customer Analyzer', 'student': 'Wadeen Khalid Al adwani',
               'instructor': 'مضر محمد الفضلي', 'academy': 'SDAIA Academy', 'team': 'individual, student-confirmed instructor permission',
               'dataset': {'source': 'authored synthetic v1', 'total': 198, 'train': 102, 'validation': 30, 'test': 66, 'sha256': training['data_sha256']},
               'test_metrics': {'baseline_macro_f1': metrics['baseline']['macro_f1'], 'improved_macro_f1': metrics['improved']['macro_f1'], 'actual_errors': len(errors)},
               'fine_tuning': {'performed': False, 'alternative': 'evidence-based written justification', 'document': 'DECISIONS.md'},
               'representation': 'train-fitted TF-IDF/SVD latent semantic embeddings, not transformer embeddings',
               'production_ready': False, 'submitted_to_instructor': False, 'api_required': False}
    write('PROJECT_SUMMARY.json', json.dumps(summary, ensure_ascii=False, indent=2))
    presentation = f'''
# ١ — Voice of Customer: تحليل صوت العميل

**Wadeen Khalid Al adwani**

المدرب: **مضر محمد الفضلي**
أكاديمية سدايا · المشروع رقم ٣ · تسليم فردي بموافقة المدرب وفق تأكيد المتدربة

الهدف: تحويل تعليقات عربية متفرقة إلى مشاعر ومجموعات مشكلات وملخصات قابلة للمراجعة.

**ملاحظات المتحدثة — ٠:٤٥:** عرفي بنفسك وبالمشكلة، وأوضحي أن النموذج تعليمي.

---

# ٢ — المشكلة والبيانات

- مدخل: تعليق أو ملف CSV. مخرج: المشاعر والموضوع والمجموعات وملخص مع شاهد.
- 198 مثالًا مؤلفًا: train 102، validation 30، test 66.
- فصحى وخليجي وإنجليزي؛ إيجابي وسلبي ومحايد.
- البيانات ليست عملاء حقيقيين؛ أسماء وأرقام حقيقية غير مستخدمة.

**ملاحظات — ١:٠٠:** لا تصفي البيانات بأنها مجموعة عامة أو حقيقية. اشرحي بطاقة البيانات.

---

# ٣ — البنية والمقارنة

- إخفاء بريد وهاتف، تطبيع عربي مع الاحتفاظ بالنفي.
- الأساسي: TF-IDF كلمات + Naive Bayes.
- المحسن: كلمات ومقاطع حروف + Logistic Regression.
- Embeddings: SVD محلي من train فقط؛ KMeans يجمع المشكلات.
- الملخص: أعداد تنبؤات وتعليق ممثل، دون API أو LLM.

**ملاحظات — ١:٣٠:** SVD ليس Transformer. أسماء المواضيع بقواعد، وتفسير المجموعة يحتاج مراجعة.

---

# ٤ — تجربة عادلة ونتائج فعلية

{results}

نفس الاختبار، دون تدريب عليه. اختيار C على validation فقط. فرق F1 = {delta['macro_f1']:.3f}، وفاصل الثقة يشمل الصفر؛ لا ندعي تفوقًا عامًا. الإنجليزية 6 أمثلة فقط.

**ملاحظات — ١:٣٠:** Macro-F1 يوازن الفئات. وضحي لماذا الدقة وحدها لا تكفي وما يعني عدم حسم فرق النموذجين.

---

# ٥ — تحليل الأخطاء والمراجعة

{len(errors)} خطأ فعليًا، تشمل النفي والسخرية والأسئلة واللهجة والنصوص المختلطة.

- «ليست الجودة سيئة» تختلف عن «الجودة سيئة».
- السؤال عن رأي إيجابي ليس بالضرورة رأيًا إيجابيًا من الكاتب.
- الثقة غير معايرة؛ عتبة 0.55 ترسل كل اختبارنا للمراجعة.
- لا نتخذ قرارًا مستقلًا تجاه العميل من هذا النموذج.

**ملاحظات — ١:١٥:** افتحي `reports/error_analysis.csv` لإظهار الأخطاء الفعلية؛ أمثلة الشريحة توضح الظواهر وليست بديلًا عن سجل الأخطاء.

---

# ٦ — لماذا لم ننفذ fine-tuning؟

- المنهج يسمح بتبرير مدعوم بالأدلة بدل fine-tuning.
- 102 مثال تدريب مؤلفًا، دون اتفاق مقيمين مستقلين.
- منحنى validation غير رتيب؛ البيانات غير ممثلة لواقع العملاء.
- فرق الأداء غير محسوم؛ لم نقارن Transformer ولا ندعي أنه أسوأ.
- التالي: بيانات مرخصة أوسع، ثم encoder مثبت الإصدار وfine-tuning مقابل baseline على اختبار جديد.

**ملاحظات — ١:٠٠:** اشرحي قرار النطاق بوضوح، ولا تسمي تدريب Logistic Regression fine-tuning لمحول.

---

# ٧ — العرض الحي

شغلي `uv run --frozen python app.py` بعد `bash scripts/run_all.sh`.

1. أدخلي: «التوصيل سريع وكل الطلب كامل».
2. أدخلي: «التطبيق يعلق والدفع ما يتم».
3. أدخلي: «كيف أوصل لخدمة المحادثة».
4. اعرضي المشاعر والمجموعات والملخص، ثم حملي `sample_outputs/demo_input.csv`.

نسخة احتياطية قابلة للعرض: `sample_outputs/analysis.json` ومخرجات الدفتر. هذا ليس تسجيل فيديو؛ سجلي العرض بنفسك إذا طلب المدرب ذلك.

**ملاحظات — ١:٣٠:** جربي العرض قبل التقديم. لا تستخدمي بيانات عملاء شخصية.

---

# ٨ — الخلاصة والخطة

- تنفيذ كامل قابل لإعادة التشغيل: تصنيف + embeddings + تجميع + تلخيص.
- مقارنة عادلة وشرائح و{len(errors)} خطأ موثق وتبرير adaptation.
- حدود واضحة: بيانات مؤلفة صغيرة، ثقة غير معايرة، لا تقييم ذهبي للمواضيع.
- أولوية التطوير: بيانات حقيقية مصرح بها، تسميات مستقلة، معايرة، ثم مقارنة نماذج أقوى.

**ملاحظات — ٠:٣٠:** المجموع ١٠ دقائق؛ خصصي ٥ دقائق للأسئلة. العرض والمراجعة والتسليم للمدرب لم تنفذ تلقائيًا. افصحي عن المساعدة البرمجية حسب سياسة الدورة.
'''
    write('PRESENTATION.md', presentation)
    slides = presentation.strip().split('\n---\n')
    content = '\n'.join(f'<section class="slide">{MarkdownIt().render(slide)}</section>' for slide in slides)
    html = '''<!doctype html><html lang="ar" dir="rtl"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Voice of Customer — Wadeen</title><style>
body{margin:0;background:#0b1729;color:#f4f7fc;font-family:Tahoma,Arial,sans-serif} .slide{display:none;box-sizing:border-box;min-height:90vh;padding:4vw 7vw;font-size:clamp(16px,2vw,25px);line-height:1.8}.slide.active{display:block}h1{color:#67e8d0;font-size:1.6em}strong{color:#fde68a}table{border-collapse:collapse;font-size:.7em;direction:ltr}th,td{padding:.4em;border:1px solid #53617b}code{direction:ltr;display:inline-block;background:#203047;padding:.1em .4em;border-radius:5px}nav{padding:10px 7vw;display:flex;gap:20px;align-items:center}button{background:#67e8d0;border:0;padding:10px 20px;border-radius:6px;cursor:pointer;font-size:16px}@media print{.slide{display:block;page-break-after:always;background:white;color:black;min-height:auto}nav{display:none}strong,h1{color:black}}
</style><body>''' + content + '''<nav><button id="prev">السابق</button><span id="counter"></span><button id="next">التالي</button></nav><script>
const slides=[...document.querySelectorAll('.slide')];let current=0;function show(){slides.forEach((s,i)=>s.classList.toggle('active',i===current));document.querySelector('#counter').textContent=`${current+1} / ${slides.length}`;}function move(n){current=Math.max(0,Math.min(slides.length-1,current+n));show();}document.querySelector('#prev').onclick=()=>move(-1);document.querySelector('#next').onclick=()=>move(1);document.onkeydown=e=>{if(e.key==='ArrowLeft'||e.key==='PageDown')move(1);if(e.key==='ArrowRight'||e.key==='PageUp')move(-1)};show();
</script></body></html>'''
    write('reports/presentation.html', html)
    print('Generated README, evaluation, benchmarks, summary and 8-slide presentation from current reports.')


if __name__ == '__main__':
    main()
