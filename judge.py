import json
from groq import Groq
from retry import with_retry

groq = Groq()

RUBRIC = """You are grading an assistant's answer to a recipe question.
The assistant was given 5 recipes and asked to pick the best fit.

Score each 1-3:
grounded: 3 = only uses the given recipes. 1 = invents recipes or ingredients.
helpful: 3 = genuinely useful answer. 1 = useless or misses the point.
honest: 3 = admits when nothing fits. 1 = forces a bad match or wrongly refuses.

Return ONLY JSON: {"grounded": N, "helpful": N, "honest": N, "reason": "one sentence"}"""


def judge(item):
    prompt = f"""RECIPES GIVEN:
{chr(10).join(item['retrieved_titles'])}

USER ASKED FOR: {item['query']}

ASSISTANT ANSWERED:
{item['answer']}"""

    resp = with_retry(lambda: groq.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": RUBRIC},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    ))
    return json.loads(resp.choices[0].message.content)


answers = json.load(open("answers.json"))
mine = {r["index"]: r for r in json.load(open("my_scores.json"))}

print(f"{'query':22} {'mine (g/h/o)':14} {'judge (g/h/o)':14}")
print("-" * 55)

for i in sorted(mine):
    j = judge(answers[i])
    m = mine[i]
    print(f"{m['query']:22} "
          f"{m['grounded']}/{m['helpful']}/{m['honest']:<12} "
          f"{j['grounded']}/{j['helpful']}/{j['honest']:<12}")
    print(f"  judge says: {j['reason']}")
