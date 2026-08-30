import time
from groq import RateLimitError


def with_retry(fn, max_attempts=5):
    for attempt in range(max_attempts):
        try:
            return fn()
        except RateLimitError:
            wait = 2 ** attempt
            print(f"  rate limited, waiting {wait}s")
            time.sleep(wait)
    raise RuntimeError("still rate limited after retries")
