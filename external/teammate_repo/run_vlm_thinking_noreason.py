# -*- coding: utf-8 -*-
import os
import re
import json
import time
import base64
import argparse
import mimetypes
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from openai import OpenAI


PROMPT = """You are a meteorite image classification expert.

Classify whether the main object in the image is a meteorite.

Use visual evidence such as fusion crust, regmaglypts, metallic texture, chondrules, pallasite/iron meteorite patterns, oxidation, vesicles, fossils, crystalline terrestrial minerals, artificial objects, and ordinary terrestrial rock texture.

Return JSON only:
{"prob_meteorite":0.0,"label":0,"confidence":0.0}

No explanation. No markdown.
"""


def encode_image(path):
    path = Path(path)
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def extract_json(text):
    text = str(text).strip()
    text = text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"\{.*?\}", text, re.S)
    if m:
        return json.loads(m.group(0))

    raise ValueError(f"Cannot parse JSON: {text[:300]}")


def clamp01(x):
    try:
        x = float(x)
        return max(0.0, min(1.0, x))
    except Exception:
        return None


def call_vlm(client, model, image_path, max_retries=3):
    image_url = encode_image(image_path)

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": PROMPT},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    }
                ],
                temperature=0,
                max_tokens=80,
            )

            content = resp.choices[0].message.content
            obj = extract_json(content)

            prob = clamp01(obj.get("prob_meteorite"))
            conf = clamp01(obj.get("confidence"))

            if prob is None:
                raise ValueError(f"Invalid prob_meteorite: {content}")

            return {
                "prob_vlm": prob,
                "label_vlm": int(prob >= 0.5),
                "confidence": conf if conf is not None else 0.5,
                "reason": "",
                "raw": content,
            }

        except Exception as e:
            if attempt == max_retries:
                return {
                    "prob_vlm": None,
                    "label_vlm": None,
                    "confidence": None,
                    "reason": "",
                    "raw": f"ERROR: {repr(e)}",
                }
            time.sleep(2 * attempt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--sleep", type=float, default=0.1)
    args = parser.parse_args()

    api_key = os.environ.get("SILICONFLOW_API_KEY")
    if not api_key:
        raise RuntimeError("Please set SILICONFLOW_API_KEY first.")

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.siliconflow.cn/v1",
        timeout=120,
    )

    manifest = pd.read_csv(args.manifest)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        old = pd.read_csv(out_path)
        old_valid = old[old["prob_vlm"].notna()].copy()
        done_ids = set(old_valid["id"].astype(str))
        rows = old_valid.to_dict("records")
        print(f"Resume mode: kept {len(done_ids)} valid existing rows.")
    else:
        done_ids = set()
        rows = []

    for _, row in tqdm(manifest.iterrows(), total=len(manifest)):
        img_id = str(row["id"])
        if img_id in done_ids:
            continue

        result = call_vlm(client, args.model, row["path"])

        out_row = {
            "id": img_id,
            "path": row["path"],
            **result,
        }

        if "label" in row:
            out_row["true_label"] = int(row["label"])

        rows.append(out_row)
        pd.DataFrame(rows).to_csv(out_path, index=False)
        time.sleep(args.sleep)

    print("Saved:", out_path)


if __name__ == "__main__":
    main()
