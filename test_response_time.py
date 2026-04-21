import requests
import time
import random
from pathlib import Path

URL = "http://127.0.0.1:5000/predict"
KL_ROOT = r"C:\Users\jkall\Downloads\knee_oa_classifier\KL_Sorted_Split"

times = []
for grade in range(5):
    folder = Path(KL_ROOT) / f"KL{grade}"
    images = list(folder.glob("*.jpg"))
    image_path = random.choice(images)
    
    start = time.time()
    with open(image_path, "rb") as f:
        requests.post(URL, files={"file": f})
    elapsed = time.time() - start
    times.append(elapsed)
    print(f"KL{grade}: {elapsed:.2f}s")

print(f"\nAverage response time: {sum(times)/len(times):.2f}s")
print(f"Min: {min(times):.2f}s")
print(f"Max: {max(times):.2f}s")