"""Start our demo, exercise actual Gradio API, then terminate only our server."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
from gradio_client import Client, handle_file
from gradio_client.exceptions import AppError

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report-path', type=Path, default=ROOT / 'reports/demo_smoke.json')
    args = parser.parse_args()
    environment = dict(os.environ, NO_PROXY='127.0.0.1,localhost', PORT='7861', GRADIO_ANALYTICS_ENABLED='False')
    with (ROOT / 'artifacts/smoke_server.log').open('w') as log:
        process = subprocess.Popen([sys.executable, 'app.py'], cwd=ROOT, env=environment, stdout=log, stderr=log)
        try:
            started = False
            with httpx.Client(trust_env=False) as http:
                for _ in range(80):
                    if process.poll() is not None:
                        raise RuntimeError((ROOT / 'artifacts/smoke_server.log').read_text())
                    try:
                        response = http.get('http://127.0.0.1:7861/config', timeout=.5)
                        if response.status_code == 200:
                            started = True
                            break
                    except httpx.HTTPError:
                        pass
                    time.sleep(.25)
                if not started:
                    raise RuntimeError('Demo failed to become responsive.')
            client = Client('http://127.0.0.1:7861', httpx_kwargs={'trust_env': False}, verbose=False)
            response = client.predict('التوصيل سريع وكل الطلب كامل\nالتطبيق يعلق والدفع ما يتم\nكيف أوصل لخدمة المحادثة', api_name='/analyze')
            output = response[2]
            assert output['total'] == 3
            assert [p['sentiment'] for p in output['predictions']] == ['positive', 'negative', 'neutral']
            csv_result = client.predict(handle_file(str(ROOT / 'sample_outputs/demo_input.csv')), api_name='/analyze_csv')[2]
            assert csv_result['total'] == 15 and sum(c['size'] for c in csv_result['clusters']) == 15
            invalid_rejected = False
            try:
                client.predict('', api_name='/analyze')
            except AppError as exc:
                invalid_rejected = '1–1000' in str(exc)
            assert invalid_rejected, 'Empty input did not produce expected validation error.'
            invalid_csv = ROOT / 'artifacts/invalid_feedback.csv'
            invalid_csv.write_text('text\nالمنتج ممتاز\n" "\n', encoding='utf-8')
            invalid_csv_rejected = False
            try:
                client.predict(handle_file(str(invalid_csv)), api_name='/analyze_csv')
            except AppError as exc:
                invalid_csv_rejected = 'دون قيم فارغة' in str(exc)
            assert invalid_csv_rejected, 'Invalid CSV rows were silently dropped.'
            record = {'status': 'passed', 'http_config': 200, 'text_api_comments': 3,
                      'csv_api_comments': 15, 'empty_input_rejected': True,
                      'invalid_csv_rejected': True,
                      'share_tunnel': False, 'server_stopped_by_test': True}
            args.report_path.parent.mkdir(parents=True, exist_ok=True)
            args.report_path.write_text(json.dumps(record, indent=2))
            print(json.dumps(record, indent=2))
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


if __name__ == '__main__':
    main()
