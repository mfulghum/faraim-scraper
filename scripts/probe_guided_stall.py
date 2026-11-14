"""Does the guided-JSON (response_format) path cancel cleanly under
asyncio.wait_for when the server stalls on a huge generation?"""
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from openai import AsyncOpenAI
from faraim_scraper.acs_link_teacher import LINKS_FORMAT, TEACHER_MODEL, TEACHER_URL

PROMPT = ("Produce links JSON. " * 3 + "\n".join(
    f"[{i}] 14 CFR 91.{100+i} fictitious regulatory heading {i} "
    "with regulatory prose inflating tokens for a stress probe."
    for i in range(700))
    + "\n\nNow emit the links array with a long verbatim quote per row.")


async def m(timeout):
    c = AsyncOpenAI(base_url=TEACHER_URL, api_key='none', timeout=900)
    t0 = time.time()
    try:
        r = await asyncio.wait_for(
            c.chat.completions.create(
                model=TEACHER_MODEL,
                messages=[{'role': 'user', 'content': PROMPT}],
                temperature=0.2, max_tokens=16384,
                response_format=LINKS_FORMAT,
                extra_body={'reasoning_effort': 'medium'}),
            timeout=timeout)
        return f'completed {round(time.time() - t0)}s len={len(r.choices[0].message.content or "")}'
    except asyncio.TimeoutError:
        return f'TimeoutError after {round(time.time() - t0)}s'
    except Exception as e:
        return f'{type(e).__name__} after {round(time.time() - t0)}s: {str(e)[:120]}'


if __name__ == '__main__':
    print(asyncio.run(m(int(sys.argv[1]) if len(sys.argv) > 1 else 150)))
