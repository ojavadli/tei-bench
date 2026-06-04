"""Author industry-specific tasks via generate-then-validate.

For industries without good public datasets, we construct labelled data with
a transparent, disclosed methodology:

  1. GENERATE: a strong model (Sonnet) produces realistic, diverse examples
     conditioned on each label's definition (the intended label is known by
     construction).
  2. VALIDATE: an independent pass (same strong model, temp 0, different
     framing — given only the example + label set) predicts the label. We KEEP
     an example only if the independent prediction matches the intended label.
     This filters ambiguous/mislabelled items, yielding clean, well-posed gold.

The agent under test (Haiku) is weaker than the labeller (Sonnet), and label
sets are deliberately fine-grained (5-7 classes), so there is genuine
headroom for TEI to improve. All data is committed for reproducibility.

Run: python scripts/build_authored_tasks.py
"""
import asyncio
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from teibench.llm import LLM, parse_json, parse_json_list
from teibench.tasks import Task, save_task
from teibench.scorers import _norm

ROOT = Path(__file__).resolve().parent.parent
TASKS_DIR = ROOT / "tasks"
GEN_MODEL = "claude-sonnet-4-5"
N_PER_LABEL = 11         # generate this many per label, keep validated ones
SEED = 11

# task_id, industry, role, instruction, {label: definition}
SPECS = [
    ("insurance_claim_type", "Insurance", "an insurance claims router",
     "Classify the insurance claim description into its claim type.",
     {"auto_collision": "vehicle accident / collision damage",
      "property_fire": "fire or smoke damage to property",
      "water_damage": "flooding, burst pipe, or water damage",
      "theft": "stolen property or burglary",
      "liability": "third-party injury or property damage the insured is liable for",
      "medical": "bodily injury or medical treatment claim"}),
    ("legal_contract_clause", "Legal", "a contract-clause classifier",
     "Classify the contract clause into its clause type.",
     {"indemnification": "one party agrees to cover losses of another",
      "termination": "conditions for ending the agreement",
      "confidentiality": "non-disclosure of information",
      "governing_law": "which jurisdiction's law applies",
      "limitation_of_liability": "caps or excludes damages",
      "payment_terms": "amounts, schedule, invoicing of payment"}),
    ("telecom_churn_reason", "Telecom", "a customer-retention analyst",
     "Classify the churn reason expressed in the customer's cancellation message.",
     {"price": "cost too high / found cheaper elsewhere",
      "coverage": "poor network coverage or signal",
      "service": "bad customer service experience",
      "speed": "slow data / internet speeds",
      "moving": "relocating / moving address",
      "competitor": "switching to a specific competitor offer"}),
    ("logistics_incident", "Logistics / Supply Chain", "a logistics incident classifier",
     "Classify the shipment incident report into an incident category.",
     {"delayed": "shipment is late / behind schedule",
      "damaged": "goods arrived damaged",
      "lost": "shipment lost or missing",
      "customs_hold": "held at customs / clearance issue",
      "wrong_address": "delivery to wrong or incomplete address",
      "partial": "partial / incomplete delivery"}),
    ("realestate_attribute", "Real Estate / PropTech", "a real-estate listing analyst",
     "Identify the primary attribute the prospective buyer is asking about.",
     {"price": "asking price / affordability",
      "location": "neighborhood / commute / area",
      "size": "square footage / number of rooms",
      "condition": "age / renovation / repair state",
      "financing": "mortgage / loan / down-payment",
      "availability": "move-in date / whether still available"}),
    ("hr_resume_fit", "Human Resources", "a recruiting screening assistant",
     "Classify the candidate note into a hiring-stage recommendation.",
     {"strong_fit": "clearly meets key requirements, advance",
      "maybe_fit": "partial match, needs further review",
      "culture_concern": "skills ok but culture/soft-skill concern",
      "overqualified": "experience far exceeds the role",
      "underqualified": "missing core required qualifications",
      "wrong_role": "better suited to a different position"}),
    ("cybersec_alert", "Cybersecurity", "a SOC alert triage analyst",
     "Classify the security alert into a threat category.",
     {"phishing": "deceptive email / credential harvesting",
      "malware": "malicious software / virus / ransomware",
      "brute_force": "repeated failed logins / password guessing",
      "data_exfiltration": "unauthorized data transfer out",
      "insider_threat": "suspicious action by an internal account",
      "ddos": "denial-of-service / traffic flood"}),
    ("energy_meter_event", "Energy / Utilities", "a smart-grid operations assistant",
     "Classify the meter/grid event description into an event type.",
     {"outage": "loss of power / blackout",
      "overconsumption": "abnormally high usage spike",
      "tamper": "suspected meter tampering",
      "voltage_anomaly": "voltage sag/surge out of range",
      "billing_dispute": "customer disputes a charge/reading",
      "maintenance": "scheduled or needed maintenance"}),
    ("agri_crop_issue", "Agriculture / AgriTech", "a crop-advisory assistant",
     "Classify the farmer's described crop problem into a cause category.",
     {"pest": "insect or animal pest damage",
      "disease": "fungal/bacterial/viral plant disease",
      "nutrient": "nutrient deficiency / soil fertility",
      "water_stress": "drought or over-watering",
      "weather_damage": "frost, hail, wind, heat damage",
      "weed": "weed competition"}),
    ("travel_intent", "Travel / Hospitality", "a travel-booking assistant",
     "Classify the traveler's request into a booking intent.",
     {"flight": "book/change a flight",
      "hotel": "book/change accommodation",
      "car_rental": "rent a car / ground transport",
      "cancellation": "cancel a booking / refund",
      "itinerary": "view or modify trip itinerary",
      "loyalty": "points / membership / status question"}),
    ("gaming_support", "Gaming", "a game player-support assistant",
     "Classify the player's support ticket into a category.",
     {"billing": "purchase / refund / subscription issue",
      "bug": "game crash / glitch / technical bug",
      "account": "login / account recovery / security",
      "gameplay": "how-to / game mechanics question",
      "toxicity_report": "reporting another player's behavior",
      "connectivity": "lag / disconnect / server issue"}),
    ("gov_service_router", "Government / Public Sector", "a citizen-services router",
     "Route the citizen's request to the correct government department.",
     {"taxation": "taxes / filing / refunds",
      "licensing": "permits / licenses / registration",
      "benefits": "social benefits / welfare / unemployment",
      "infrastructure": "roads / utilities / public works",
      "records": "birth/marriage/property records",
      "complaint": "general complaint / feedback"}),
    ("manufacturing_qa", "Manufacturing", "a quality-assurance defect classifier",
     "Classify the inspection note into a defect category.",
     {"surface": "scratches / dents / finish defects",
      "dimensional": "size / tolerance out of spec",
      "assembly": "misassembly / missing part",
      "electrical": "wiring / circuit / power defect",
      "material": "material flaw / wrong material",
      "labeling": "wrong / missing / misprinted label"}),
    ("saas_ticket_priority", "B2B SaaS", "a support-priority classifier",
     "Classify the support ticket into a priority/type bucket.",
     {"outage_critical": "full service down / blocking many users",
      "data_loss": "data missing/corrupted",
      "billing": "invoice / payment / plan issue",
      "feature_request": "asking for new capability",
      "how_to": "usage / configuration question",
      "integration": "API / third-party integration problem"}),
]

GEN_SYSTEM = """You generate realistic, diverse, single-paragraph example inputs for a
text-classification dataset. Each example must clearly and unambiguously belong to the
TARGET label (a domain expert would agree). Vary phrasing, length, and specifics. Do NOT
mention the label name verbatim in the text. Return ONLY a JSON list of strings."""

VAL_SYSTEM = """You are a careful domain expert labeling text. Given the input and the list of
allowed labels, return the single best label. Respond ONLY as JSON: {"label": "<one of the allowed labels>"}."""


async def gen_label_examples(llm, role, instruction, label, definition, labels, n):
    user = (
        f"Domain: {role}. Task: {instruction}\n"
        f"TARGET label: '{label}' — meaning: {definition}\n"
        f"All labels: {', '.join(labels)}\n"
        f"Generate {n} distinct realistic example inputs that belong to '{label}'. "
        f"JSON list of strings only."
    )
    try:
        txt = await llm.complete(model=GEN_MODEL, system=GEN_SYSTEM, user=user,
                                 temperature=0.9, max_tokens=1500, nonce=f"gen-{label}")
        items = parse_json_list(txt)
        return [str(x).strip() for x in items if str(x).strip()][:n]
    except Exception:
        return []


async def validate(llm, text, labels):
    user = f"Allowed labels: {', '.join(labels)}\n\nInput: {text}\n\nReturn the best label as JSON."
    try:
        data = await llm.complete_json(model=GEN_MODEL, system=VAL_SYSTEM, user=user,
                                       temperature=0.0, max_tokens=60, nonce="val")
        return str(data.get("label", "")).strip()
    except Exception:
        return ""


async def build_one(llm, spec):
    task_id, industry, role, instruction, label_def = spec
    labels = list(label_def.keys())
    rng = random.Random(SEED)

    # 1) generate
    gen_lists = await asyncio.gather(*[
        gen_label_examples(llm, role, instruction, lab, dfn, labels, N_PER_LABEL)
        for lab, dfn in label_def.items()
    ])
    candidates = []
    for lab, items in zip(labels, gen_lists):
        for it in items:
            candidates.append((it, lab))

    # 2) validate (keep only agreed)
    preds = await asyncio.gather(*[validate(llm, t, labels) for t, _ in candidates])
    kept = [(t, lab) for (t, lab), p in zip(candidates, preds) if _norm(p) == _norm(lab)]

    # stratified split 15/15
    by_lab = {}
    for t, lab in kept:
        by_lab.setdefault(lab, []).append(t)
    for lab in by_lab:
        rng.shuffle(by_lab[lab])
    train, test = [], []
    # round-robin to balance
    pools = {lab: list(items) for lab, items in by_lab.items()}
    target_each = 5  # up to per label
    # fill train first then test, round robin
    order = list(pools.keys())
    rr = 0
    while sum(len(p) for p in pools.values()) and (len(train) < 15 or len(test) < 24):
        lab = order[rr % len(order)]; rr += 1
        if pools[lab]:
            ex = {"query": pools[lab].pop(), "gold": lab}
            if len(train) < 15:
                train.append(ex)
            elif len(test) < 24:
                test.append(ex)
        if rr > 100000:
            break

    if len(train) < 12 or len(test) < 10:
        return None, f"insufficient validated data (train={len(train)}, test={len(test)}, kept={len(kept)}/{len(candidates)})"

    baseline = (f"You are {role}. {instruction} "
                f"Classify into one of these categories: {', '.join(labels)}.")
    t = Task(task_id=task_id, industry=industry, instruction=instruction,
             metric="classification", labels=labels, baseline_prompt=baseline,
             train=train, test=test,
             source="Authored for TEI-Bench (Sonnet-generated, Sonnet-validated; kept only label-agreed examples).")
    save_task(t, TASKS_DIR)
    return t, f"kept={len(kept)}/{len(candidates)} train={len(train)} test={len(test)} nclasses={len(labels)}"


async def main():
    llm = LLM(cache_dir=str(ROOT / ".cache"), max_concurrency=6)
    print("Authoring industry tasks (generate + validate)...")
    for spec in SPECS:
        t, msg = await build_one(llm, spec)
        tag = "OK  " if t else "SKIP"
        print(f"  {tag} {spec[0]:<26} {msg}")
    print(f"\n  usage: {llm.usage.report()}")


if __name__ == "__main__":
    asyncio.run(main())
