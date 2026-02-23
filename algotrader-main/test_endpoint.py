"""End-to-end test: start server, hit paper trade endpoint, report result."""
import subprocess, time, urllib.request, json, sys

# Start server
proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "src.api.webapp:app", "--host", "127.0.0.1", "--port", "5001"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
)

# Wait for server ready
time.sleep(6)

try:
    data = json.dumps({"strategy": "ema_crossover", "bars": 500, "capital": 100000}).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:5001/api/paper-trade/sample",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        status = resp.status
        body = json.loads(resp.read())
        print(f"Status: {status}")
        print(f"final_capital: {body.get('final_capital')}")
        print(f"total_trades: {body.get('total_trades')}")
        print(f"trades_count: {len(body.get('trades', []))}")
        print(f"has_health_report: {'health_report' in body}")
        hr = body.get("health_report", {})
        if hr:
            print(f"health_grade: {hr.get('grade')}")
            print(f"execution_ready: {hr.get('execution_ready')}")
        print("\nSUCCESS - Paper trade endpoint works!")
except urllib.error.HTTPError as e:
    print(f"HTTP Error {e.code}: {e.read().decode()}")
except Exception as e:
    print(f"Error: {e}")
finally:
    proc.terminate()
    proc.wait(timeout=5)
