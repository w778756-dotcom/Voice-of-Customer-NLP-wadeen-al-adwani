"""Create original project and exploration notebooks with runnable, traceable cells."""

from pathlib import Path

import nbformat as nb

ROOT = Path(__file__).resolve().parents[1]


def markdown(text):
    return nb.v4.new_markdown_cell(text.strip())


def code(text):
    return nb.v4.new_code_cell(text.strip())


BOOTSTRAP = '''
from pathlib import Path
import os, sys, json, subprocess
ROOT = Path.cwd()
if not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent
assert (ROOT / "data/feedback.csv").exists(), "Run from the project root or notebooks/"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/voice_of_customer-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/workspace/.cache")
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd
from IPython.display import display, Markdown
from voice_of_customer.pipeline import load_analyzer
from voice_of_customer.text import normalize
data = pd.read_csv(ROOT / "data/feedback.csv")
display(data.groupby(["split", "lang", "label"]).size().rename("n").to_frame())
assert not data.text.map(normalize).duplicated().any()
'''


def main():
    notebook = nb.v4.new_notebook(cells=[
        markdown('''# Voice of Customer — تحليل صوت العميل
**Wadeen Khalid Al adwani** · **المدرب: مضر محمد الفضلي** · **أكاديمية سدايا**

الهدف: تصنيف مشاعر التعليقات وتجميع المشكلات وكتابة ملخص قابل للمراجعة. البيانات **198 مثالًا مؤلفًا تعليميًا** وليست بيانات عملاء حقيقيين. الكود والبيانات والتوثيق أُعدت بمساعدة برمجية معلنة.

اختاري Python من `.venv` بعد `bash scripts/run_all.sh`. تعليمات Colab في `GETTING_STARTED.md`. Run all يعيد التدريب والتقييم الحقيقي، ولا يستخدم الاختبار في التدريب. واجهة Gradio موجودة في `app.py`؛ هذا الدفتر يعرض التجربة والنتائج والنسخة الاحتياطية للعرض.'''),
        code(BOOTSTRAP),
        markdown('''## ١ — البيانات والخصوصية
إيجابي/سلبي/محايد، فصحى وخليجي وإنجليزي. التقسيم ثابت؛ التطبيع لا يحذف النفي. لا تدخل metadata أو labels إلى الميزات. فحص exact duplicates لا يمنع التشابه الدلالي من مؤلف واحد. `DATA_CARD.md` يشرح حدود الترخيص والتعميم.'''),
        code('''from voice_of_customer.text import mask_pii
print(mask_pii("مو سيئ، البريد test@example.com والجوال 0501234567"))
display(data[data.split == "test"][["id", "text", "label", "phenomenon"]].head(8))'''),
        markdown('''## ٢ — تدريب baseline والتحسين
Baseline: word TF-IDF + Naive Bayes. Improvement: word+character TF-IDF + Logistic Regression؛ C يُختار على validation فقط. التمثيل للتجميع: TF-IDF → SVD 32D → L2، وليس Transformer.'''),
        code('''subprocess.run([sys.executable, "scripts/train.py"], cwd=ROOT, check=True)
subprocess.run([sys.executable, "scripts/evaluate.py"], cwd=ROOT, check=True)
training = json.loads((ROOT / "reports/training.json").read_text())
display(pd.DataFrame(training["candidates"]))'''),
        markdown('''## ٣ — نفس الاختبار لكل نموذج
قياس فعلي، فواصل bootstrap 95% وpaired difference. لا ندعي تفوقًا عامًا إذا شمل فاصل الفرق الصفر.'''),
        code('''metrics = json.loads((ROOT / "reports/metrics.json").read_text())
display(pd.DataFrame({name: metrics[name] for name in ["baseline", "improved"]}).T[["n", "accuracy", "macro_f1", "macro_f1_95ci"]])
print("Paired delta:", metrics["paired_delta"])
display(pd.read_csv(ROOT / "reports/slices.csv"))
display(metrics["review_policy"])'''),
        markdown('''## ٤ — أخطاء فعلية وليس أمثلة نجاح مختارة
كل خطأ في CSV له ID وتعليق وتسميات وسبب محتمل وخطوة علاج. التصنيف التشخيصي للسبب ليس مراجعة مستقلة من خبير.'''),
        code('''errors = pd.read_csv(ROOT / "reports/error_analysis.csv")
assert len(errors) >= 10 and (errors.label != errors.improved_pred).all()
display(errors.groupby("error_category").size().rename("failures").to_frame())
display(errors[["id", "text", "label", "improved_pred", "error_category"]].head(15))'''),
        markdown('''## ٥ — demo: تصنيف + embeddings + clustering + summary
يمكن تعديل قائمة التعليقات أدناه وإعادة تشغيل الخلية. المخرجات المحلية لا تتصل بـAPI. الملخص يعد التنبؤات ويعرض تعليقًا ممثلًا؛ أسماء المواضيع بقواعد وتحتاج مراجعة.'''),
        code('''comments = pd.read_csv(ROOT / "sample_outputs/demo_input.csv").text.tolist()
analyzer = load_analyzer()
result = analyzer.analyze(comments)
assert result["total"] == len(comments)
assert sum(c["size"] for c in result["clusters"]) == len(comments)
display(pd.DataFrame(result["predictions"]))
display(pd.DataFrame(result["clusters"]))
for cluster in result["clusters"]:
    print(cluster["summary"])
    print("شاهد:", cluster["representative"])
print("Embedding shape:", analyzer.embed(comments).shape)'''),
        markdown('''## ٦ — قرار عدم fine-tuning
الدورة تسمح بالتبرير المدعوم بالأدلة. البيانات المؤلفة صغيرة، ولا توجد مراجعة تسمية مستقلة، ومنحنى التعلم غير رتيب، وفرق الاختبار غير محسوم. التدريب الخطي ليس fine-tuning لمحول. لم نختبر Transformer؛ لا ندعي أن الأداء سيكون أسوأ. التالي: بيانات حقيقية مصرح بها ثم مقارنة encoder/fine-tuning على اختبار جديد. انظري `DECISIONS.md`.'''),
        code('''display(pd.read_csv(ROOT / "reports/learning_curve.csv"))
display(json.loads((ROOT / "reports/benchmark.json").read_text()))
from IPython.display import Image
display(Image(filename=str(ROOT / "reports/confusion_matrices.png")))'''),
        markdown('''## ٧ — تشغيل الواجهة والعرض
في terminal: `uv run --frozen python app.py`. اعرضي مثالًا إيجابيًا وآخر سلبيًا وسؤالًا محايدًا ثم CSV. خلايا هذا الدفتر و`sample_outputs/analysis.json` نسخة احتياطية قابلة للفحص، وليست فيديو مسجلًا. `PRESENTATION.md` ثماني شرائح وشرح لعشر دقائق. راجعي وفهمي الكود والحدود قبل التسليم.

لا توجد موافقة قبول أو تقديم عرض تلقائي. سياسة الاستعانة بالمساعد البرمجي يجب مراعاتها.'''),
    ])
    exploration = nb.v4.new_notebook(cells=[
        markdown('''# استكشاف بيانات صوت العميل
**Wadeen Khalid Al adwani** · SDAIA Academy · instructor: مضر محمد الفضلي

استكشاف تعليمي للبيانات المؤلفة وحدودها؛ لا يمثل توزيع العملاء الحقيقي. دفتر التجربة الكامل موجود في جذر المستودع.'''),
        code(BOOTSTRAP),
        code('''data["characters"] = data.text.str.len()
display(data.groupby(["split", "dialect"]).agg(n=("id", "count"), mean_chars=("characters", "mean")))
display(data.groupby(["split", "phenomenon"]).size().rename("n").to_frame())'''),
        code('''import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7, 3))
data.groupby(["split", "label"]).size().unstack().plot.bar(ax=ax)
ax.set(title="Authored synthetic examples — not customer prevalence", ylabel="Examples")
plt.tight_layout()
plt.show()'''),
        markdown('''## ملاحظات
- كل فئة متوازنة في كل split؛ الواقع قد يكون غير متوازن.
- validation عربية فقط؛ الإنجليزية exploratory.
- اختبارنا أصعب عمدًا ويشمل السخرية والنفي؛ المقاييس ليست تقديرًا عامًا لأداء العملاء.
- الخطوة التالية: بيانات مرخصة جديدة، تدقيق PII مستقل، مراجعة تسميات ومجموعات لمنع تسرب المؤلف/المصدر.'''),
    ])
    for name, document in [('Voice_of_Customer_NLP_Wadeen_Khalid_Al_adwani.ipynb', notebook), ('notebooks/01_exploration.ipynb', exploration)]:
        document.metadata = {'kernelspec': {'display_name': 'Python 3.12 (Voice of Customer)', 'language': 'python', 'name': 'python3'},
                             'language_info': {'name': 'python', 'version': '3.12.14'}}
        nb.write(document, ROOT / name)
    print('Created two original notebooks.')


if __name__ == '__main__':
    main()
