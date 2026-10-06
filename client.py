import time

import requests

URL = "http://127.0.0.1:8000"

applicant = {
    "loan_id": 5001,
    "duration": 36,
    "credit_amount": 5000,
    "installment_commitment": 3,
    "age": 35,
    "existing_credits": 1,
    "purpose": "used car",
    "checking_status": "0<=X<200",
    "credit_history": "existing paid",
    "savings_status": "<100",
    "employment": "1<=X<4",
    "housing": "own",
}

# 1. Is the service open?
print("Health:", requests.get(f"{URL}/health").json())

# 2. Score a good application
r = requests.post(f"{URL}/score", json=applicant)
print(f"\nGood applicant -> status {r.status_code}")
print(r.json())

# 3. Score a bad application
bad = {**applicant, "purpose": "boat", "age": -5}
r = requests.post(f"{URL}/score", json=bad)
print(f"\nBad applicant -> status {r.status_code}")
print(r.json())

# 4. How fast is it?
n = 100
start = time.perf_counter()
for _ in range(n):
    requests.post(f"{URL}/score", json=applicant)
avg_ms = (time.perf_counter() - start) / n * 1000
print(f"\nAverage response time over {n} requests: {avg_ms:.1f} ms")