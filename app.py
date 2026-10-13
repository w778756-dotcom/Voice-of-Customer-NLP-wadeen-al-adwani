"""Arabic Gradio demo: paste feedback or upload a CSV containing a text column."""

import os
from pathlib import Path

os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/voice_of_customer-matplotlib')

import gradio as gr
import pandas as pd

from voice_of_customer.pipeline import load_analyzer


def analyze_text(comments):
    if not isinstance(comments, str):
        raise gr.Error('أدخلي تعليقات نصية، تعليقًا واحدًا في كل سطر.')
    try:
        result = load_analyzer().analyze([line.strip() for line in comments.splitlines() if line.strip()])
    except (ValueError, RuntimeError) as exc:
        raise gr.Error(str(exc)) from exc
    columns = ['text', 'sentiment_ar', 'confidence', 'needs_review', 'aspect']
    table = pd.DataFrame(result['predictions'])[columns]
    summary = '\n\n'.join(c['summary'] + '\n\n> ' + c['representative'] for c in result['clusters'])
    return table, summary, result


def analyze_csv(file):
    if not file:
        raise gr.Error('اختاري ملف CSV يحتوي على عمود text.')
    try:
        if Path(file).stat().st_size > 5_000_000:
            raise gr.Error('الحد الأقصى لحجم الملف 5 MB.')
        # Preserve empty rows instead of silently dropping customer comments.
        # Literal strings such as "NA" are feedback, not missing-value markers.
        frame = pd.read_csv(file, nrows=1001, dtype={'text': str},
                            keep_default_na=False, skip_blank_lines=False)
    except (OSError, UnicodeDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise gr.Error('تعذر قراءة الملف. استخدمي CSV صالحًا بترميز UTF-8 وعمود text.') from exc
    if ('text' not in frame.columns or not 1 <= len(frame) <= 1000
            or frame.text.isna().any() or frame.text.str.strip().eq('').any()):
        raise gr.Error('يلزم عمود text و1–1000 تعليق دون قيم فارغة.')
    return analyze_text('\n'.join(frame.text.str.replace(r'[\r\n]+', ' ', regex=True)))


def build_demo():
    with gr.Blocks(title='Voice of Customer | صوت العميل', theme=gr.themes.Soft()) as demo:
        gr.Markdown('# Voice of Customer — تحليل صوت العميل\n**Wadeen Khalid Al adwani** · أكاديمية سدايا · مشروع Voice of Customer\n\n'
                    'المدرب: **مضر محمد الفضلي**\n\n'
                    'نموذج تعليمي على بيانات مؤلفة: النتائج ليست ضمانًا للأداء على عملاء حقيقيين. '
                    'كل التحليل محلي؛ إخفاء البريد والهاتف آلي وجزئي، لذا لا ترفعي بيانات شخصية.')
        comments = gr.Textbox(label='تعليق واحد في كل سطر', lines=7,
                              value='التوصيل سريع وكل الطلب كامل\nالتطبيق يعلق والدفع ما يتم\nكيف أوصل لخدمة المحادثة')
        button = gr.Button('تحليل التعليقات', variant='primary')
        file = gr.File(label='أو ارفعي CSV بعمود text', file_types=['.csv'], type='filepath')
        csv_button = gr.Button('تحليل الملف')
        table = gr.Dataframe(label='المشاعر والموضوع والثقة والحاجة إلى المراجعة', interactive=False)
        summary = gr.Markdown()
        result = gr.JSON(label='النتيجة المنظمة')
        button.click(analyze_text, comments, [table, summary, result], api_name='analyze')
        csv_button.click(analyze_csv, file, [table, summary, result], api_name='analyze_csv')
        gr.Markdown('الثقة غير معايرة. الموضوع مستخرج بقواعد؛ المجموعات تعتمد SVD وKMeans؛ '
                    'الملخص يحصي التنبؤات ويعرض تعليقًا ممثلًا، ولا يستخدم LLM.')
    return demo


if __name__ == '__main__':
    build_demo().launch(server_name='127.0.0.1', server_port=int(os.getenv('PORT', '7860')), share=False)
