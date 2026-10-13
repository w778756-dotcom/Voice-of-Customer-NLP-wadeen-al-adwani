# Voice of Customer — Sample inputs and outputs

`demo_input.csv` contains 15 already disclosed synthetic test examples (5 per sentiment). `analysis.json` is the actual local output for that CSV, created by `scripts/evaluate.py`; counts are **predictions**, not gold labels or real-world business findings. These outputs are a checkable presentation backup, not a video recording.

Additional demo inputs:

```text
التوصيل سريع وكل الطلب كامل
التطبيق يعلق والدفع ما يتم
كيف أوصل لخدمة المحادثة
```

Run the Gradio demo or edit the main notebook's `comments` list. CSV requires UTF-8 and a `text` column. Do not upload private customer records.
