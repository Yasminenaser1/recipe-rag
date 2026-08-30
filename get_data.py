from datasets import load_dataset
import json

ds = load_dataset("corbt/all-recipes", split="train")

recipes = []
for i in range(300):
    text = ds[i]["input"]
    title = text.split("\n")[0]
    recipes.append({"id": i, "title": title, "text": text})

with open("recipes.json", "w") as f:
    json.dump(recipes, f, indent=2)

print(f"saved {len(recipes)} recipes")
for r in recipes[:5]:
    print(" -", r["title"])
