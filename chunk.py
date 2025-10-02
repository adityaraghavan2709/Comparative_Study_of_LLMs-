import os
import time

chunk_size = 20
total_examples = 260

while True:
    if os.path.exists("checkpoint_tracking.txt"):
        with open("checkpoint_tracking.txt", "r") as f:
            start = int(f.read())
    else:
        start = 0

    if start >= total_examples:
        print(" All chunks processed.")
        break

    print(f"Running chunk from {start} to {start + chunk_size}")
    os.system(f"python tracking_objects.py {start}")
    time.sleep(2)
