"""Build industry-framed tasks from confirmed parquet-native public datasets.

Each dataset is sampled (stratified across classes, fixed seed) into a
train/test split and framed as an 'industry agent'. Labels come straight
from the dataset's human-readable label_text, so gold is authoritative and
citable. Data is written to tasks/ (version-controlled) for full repro.

Run: python scripts/build_public_tasks.py
"""
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from teibench.tasks import Task, save_task

import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent
TASKS_DIR = ROOT / "tasks"
N_TRAIN, N_TEST = 15, 20
SEED = 7
MAXLEN = 1200


def clean(s: str) -> str:
    s = re.sub(r"\s+", " ", str(s)).strip()
    return s[:MAXLEN]


# (task_id, hf_name, hf_config, text_field, label_field, industry, instruction, role)
SPECS = [
    ("retail_review_stars", "SetFit/amazon_reviews_multi_en", None, "text", "label_text",
     "E-commerce / Retail", "Predict the product review's star rating.",
     "a retail review-rating assistant"),
    ("hospitality_review_stars", "SetFit/yelp_review_full", None, "text", "label_text",
     "Hospitality", "Predict the business review's star rating.",
     "a hospitality review-rating assistant"),
    ("media_news_desk", "SetFit/bbc-news", None, "text", "label_text",
     "Media / News", "Route the news article to the correct desk/topic.",
     "a newsroom routing assistant"),
    ("news_agency_topic", "fancyzhx/ag_news", None, "text", "label",
     "News Agency", "Categorize the news headline/snippet into a topic.",
     "a news categorization assistant"),
    ("social_emotion", "SetFit/emotion", None, "text", "label_text",
     "Social Media / CX", "Detect the dominant emotion in the message.",
     "an emotion-detection assistant"),
    ("consumer_sentiment_5", "SetFit/sst5", None, "text", "label_text",
     "Consumer Insights", "Classify the fine-grained sentiment of the statement.",
     "a fine-grained sentiment assistant"),
    ("trust_safety_hate", "SetFit/hate_speech_offensive", None, "text", "label_text",
     "Trust & Safety", "Moderate the message for hateful/offensive content.",
     "a content-moderation assistant"),
    ("platform_toxicity", "SetFit/toxic_conversations", None, "text", "label_text",
     "Online Platform", "Flag whether the comment is toxic.",
     "a toxicity-moderation assistant"),
    ("it_email_spam", "SetFit/enron_spam", None, "text", "label_text",
     "Corporate IT Security", "Triage whether the email is spam or legitimate.",
     "an email security triage assistant"),
    ("pharma_ade", "SetFit/ade_corpus_v2_classification", None, "text", "label_text",
     "Pharmacovigilance", "Detect whether the text reports an adverse drug event.",
     "a pharmacovigilance detection assistant"),
    ("retail_counterfactual", "SetFit/amazon_counterfactual_en", None, "text", "label_text",
     "Retail Analytics", "Detect whether the review contains a counterfactual statement.",
     "a review-analysis assistant"),
    ("edtech_question_router", "SetFit/student-question-categories", None, "text", "label_text",
     "EdTech", "Route the student's question to the correct category.",
     "a student-question routing assistant"),
    ("entertainment_review", "SetFit/imdb", None, "text", "label_text",
     "Entertainment", "Classify the film review's sentiment.",
     "a film-review sentiment assistant"),
    ("market_research_subjectivity", "SetFit/subj", None, "text", "label_text",
     "Market Research", "Classify whether the sentence is subjective or objective.",
     "a subjectivity-analysis assistant"),
    ("community_forum_topic", "SetFit/20_newsgroups", None, "text", "label_text",
     "Online Community", "Categorize the forum post into its newsgroup topic.",
     "a forum routing assistant"),
]


def build_classification(spec):
    task_id, name, cfg, tf, lf, industry, instruction, role = spec
    def _load(split):
        return load_dataset(name, cfg, split=split) if cfg else load_dataset(name, split=split)
    try:
        train_raw = _load("train")
        try:
            test_raw = _load("test")
        except Exception:
            test_raw = train_raw
    except Exception as e:
        print(f"  SKIP {task_id}: {str(e)[:80]}")
        return None

    # Resolve label names
    def label_str(ds, row):
        if lf == "label" and hasattr(ds.features["label"], "names"):
            return ds.features["label"].names[row["label"]]
        return str(row[lf])
    text_field = tf if tf in train_raw.features else ("text" if "text" in train_raw.features else None)
    if text_field is None:
        print(f"  SKIP {task_id}: no text field"); return None

    labels = sorted(set(label_str(train_raw, r) for r in train_raw.select(range(min(2000, len(train_raw))))))
    # clean labels: drop empty, normalize underscores in display kept as-is
    labels = [l for l in labels if l.strip()]
    if len(labels) < 2 or len(labels) > 25:
        print(f"  SKIP {task_id}: nclasses={len(labels)} out of range"); return None

    rng = random.Random(SEED)

    def sample(ds, n):
        by_lab = defaultdict(list)
        idxs = list(range(len(ds)))
        rng.shuffle(idxs)
        for i in idxs:
            r = ds[i]
            txt = clean(r[text_field])
            if not txt or len(txt) < 5:
                continue
            by_lab[label_str(ds, r)].append({"query": txt, "gold": label_str(ds, r)})
        # round-robin across labels for balance
        pools = [v for v in by_lab.values() if v]
        out, k = [], 0
        while len(out) < n and any(pools):
            p = pools[k % len(pools)]
            if p:
                out.append(p.pop())
            k += 1
            if k > 100000:
                break
        return out[:n]

    train = sample(train_raw, N_TRAIN)
    # ensure test disjoint from train when train/test are the same underlying split
    test = sample(test_raw, N_TEST)
    seen = {e["query"] for e in train}
    test = [e for e in test if e["query"] not in seen][:N_TEST]
    if len(train) < N_TRAIN or len(test) < max(10, N_TEST // 2):
        print(f"  SKIP {task_id}: insufficient (train={len(train)} test={len(test)})"); return None

    baseline = (
        f"You are {role}. {instruction} "
        f"Classify into one of these categories: {', '.join(labels)}."
    )
    t = Task(
        task_id=task_id, industry=industry, instruction=instruction,
        metric="classification", labels=labels, baseline_prompt=baseline,
        train=train, test=test,
        source=f"{name}{('/'+cfg) if cfg else ''} via HuggingFace (see dataset card for license).",
    )
    save_task(t, TASKS_DIR)
    print(f"  OK   {task_id:<28} nclasses={len(labels)} train={len(train)} test={len(test)}")
    return t


def build_gsm8k():
    try:
        train_raw = load_dataset("openai/gsm8k", "main", split="train")
        test_raw = load_dataset("openai/gsm8k", "main", split="test")
    except Exception as e:
        print(f"  SKIP edu_gsm8k: {str(e)[:80]}"); return None
    rng = random.Random(SEED)

    def gold_from(ans):
        m = re.search(r"####\s*(-?[\d,]+)", ans)
        return m.group(1).replace(",", "") if m else None

    def sample(ds, n):
        idxs = list(range(len(ds))); rng.shuffle(idxs)
        out = []
        for i in idxs:
            r = ds[i]; g = gold_from(r["answer"])
            if g is None: continue
            out.append({"query": clean(r["question"]), "gold": g})
            if len(out) >= n: break
        return out

    train, test = sample(train_raw, N_TRAIN), sample(test_raw, N_TEST)
    t = Task(
        task_id="edu_gsm8k", industry="Education", metric="numeric", labels=None,
        instruction="Solve the grade-school math word problem; give the final numeric answer.",
        baseline_prompt="You are a math tutor. Solve the problem and give the final answer.",
        train=train, test=test,
        source="openai/gsm8k (MIT) via HuggingFace.",
    )
    save_task(t, TASKS_DIR)
    print(f"  OK   edu_gsm8k                   numeric    train={len(train)} test={len(test)}")
    return t


def main():
    print("Building public-dataset tasks...")
    n = 0
    for spec in SPECS:
        if build_classification(spec):
            n += 1
    if build_gsm8k():
        n += 1
    print(f"Built {n} public tasks into {TASKS_DIR}")


if __name__ == "__main__":
    main()
