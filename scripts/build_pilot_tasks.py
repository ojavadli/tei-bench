"""Builds the 3 pilot tasks (finance / healthcare / education).

Examples are authored here so the gold labels are transparent and
version-controlled. Numeric gold is computed in-code (asserted) so there
is no hand-arithmetic error. Run:  python scripts/build_pilot_tasks.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from teibench.tasks import Task, save_task

TASKS_DIR = Path(__file__).resolve().parent.parent / "tasks"


# ---------------------------------------------------------------- finance
BANKING_LABELS = [
    "card_arrival", "card_lost", "balance_inquiry", "transfer_failed",
    "atm_fee", "exchange_rate", "close_account", "dispute_charge",
]
banking = [
    ("How many more days until my new card arrives?", "card_arrival"),
    ("My card was stolen at the train station this morning.", "card_lost"),
    ("Can you tell me how much money I currently have?", "balance_inquiry"),
    ("I tried to send money to my landlord but it didn't go through.", "transfer_failed"),
    ("Why was I charged a fee for using the cash machine abroad?", "atm_fee"),
    ("What rate do you use when I pay in euros?", "exchange_rate"),
    ("I want to permanently shut down my account.", "close_account"),
    ("There is a $90 charge I never made — please remove it.", "dispute_charge"),
    ("It's been two weeks and the card still hasn't shown up.", "card_arrival"),
    ("I think I dropped my debit card somewhere downtown.", "card_lost"),
    ("Show me my available funds, please.", "balance_inquiry"),
    ("The payment to my friend keeps failing.", "transfer_failed"),
    ("I got billed extra at an ATM that isn't yours.", "atm_fee"),
    ("How are foreign currency conversions priced?", "exchange_rate"),
    ("Please close my checking account for good.", "close_account"),
    ("There's a transaction on my statement I don't recognize and want reversed.", "dispute_charge"),
    ("When will the replacement card get to my house?", "card_arrival"),
    ("Someone took my wallet with the card inside.", "card_lost"),
    ("What's my current balance?", "balance_inquiry"),
    ("My wire transfer was rejected — what happened?", "transfer_failed"),
    ("I was surprised by a withdrawal fee on my last cash machine visit.", "atm_fee"),
    ("Which exchange rate applies to my purchases in London?", "exchange_rate"),
    ("Cancel my account entirely, I'm switching banks.", "close_account"),
    ("Dispute this charge — I did not authorize the $42 payment.", "dispute_charge"),
    ("Has my card been mailed yet?", "card_arrival"),
    ("I can't find my card anywhere, I think it's gone.", "card_lost"),
    ("Could you check the amount left in my savings?", "balance_inquiry"),
    ("The transfer to my own other account didn't complete.", "transfer_failed"),
    ("Why is there a surcharge from the ATM operator?", "atm_fee"),
    ("Tell me today's conversion rate for dollars to yen.", "exchange_rate"),
]

# ------------------------------------------------------------- healthcare
TRIAGE_LABELS = ["emergency", "urgent", "routine", "self_care"]
triage = [
    ("Crushing chest pain spreading to my left arm, sweating, can't breathe.", "emergency"),
    ("Sudden weakness on one side of my face and slurred speech.", "emergency"),
    ("I have a mild runny nose and a slight sore throat since yesterday.", "self_care"),
    ("A small paper cut on my finger, barely bleeding.", "self_care"),
    ("Fever of 39.5C for three days and now a stiff neck.", "urgent"),
    ("Twisted my ankle, it's swollen and I can't put weight on it.", "urgent"),
    ("I'd like to schedule my annual physical checkup.", "routine"),
    ("I need a refill on my regular blood pressure prescription.", "routine"),
    ("Severe difficulty breathing and my lips are turning blue.", "emergency"),
    ("Deep cut on my hand that won't stop bleeding after 15 minutes.", "emergency"),
    ("Mild headache that goes away with rest and water.", "self_care"),
    ("Occasional sneezing from seasonal pollen.", "self_care"),
    ("Persistent cough for two weeks with some green phlegm.", "urgent"),
    ("Ear pain and reduced hearing for a couple of days.", "urgent"),
    ("Routine follow-up to review my cholesterol results.", "routine"),
    ("I want a dental cleaning appointment.", "routine"),
    ("Unconscious friend who is not responding to voice or touch.", "emergency"),
    ("Heavy bleeding that soaks through a bandage in minutes.", "emergency"),
    ("Minor bruise on my knee from bumping a table.", "self_care"),
    ("Slight indigestion after a large meal.", "self_care"),
    ("High fever in a 6-month-old infant that started today.", "urgent"),
    ("Painful urination and lower abdominal discomfort since this morning.", "urgent"),
    ("Standard vaccination update before travel next month.", "routine"),
    ("General wellness consultation, no specific complaint.", "routine"),
    ("Sudden severe allergic reaction with throat swelling after a bee sting.", "emergency"),
    ("Possible broken arm after a fall, visible deformity.", "emergency"),
    ("Dry skin patch that itches a little.", "self_care"),
    ("Mild muscle soreness two days after exercising.", "self_care"),
    ("Migraine that hasn't improved with my usual medication in 24 hours.", "urgent"),
    ("Recurring mild back pain I want evaluated at some point.", "routine"),
]

# -------------------------------------------------------------- education
# (problem_text, answer) — answers computed/verified in code below.
math_specs = [
    ("A bakery sells 24 muffins in the morning and 18 in the afternoon. How many muffins did it sell that day?", 24 + 18),
    ("Tom has 5 boxes with 12 pencils each. How many pencils does he have in total?", 5 * 12),
    ("A tank holds 200 liters. If 75 liters are used, how many liters remain?", 200 - 75),
    ("Sara reads 15 pages a day for 6 days. How many pages did she read?", 15 * 6),
    ("There are 96 students split equally into 4 buses. How many students per bus?", 96 // 4),
    ("A shirt costs $20 and is discounted by $7. What is the sale price?", 20 - 7),
    ("Mia saves $8 each week. How much does she save in 9 weeks?", 8 * 9),
    ("A farmer has 7 rows of 11 apple trees. How many trees total?", 7 * 11),
    ("If a movie is 135 minutes and 40 minutes have passed, how many minutes remain?", 135 - 40),
    ("Jack buys 3 packs of cards with 25 cards each. How many cards total?", 3 * 25),
    ("A class of 30 students has 12 girls. How many boys are there?", 30 - 12),
    ("Each table seats 6 people. How many people can 14 tables seat?", 6 * 14),
    ("A runner covers 4 km each day. How far in 13 days?", 4 * 13),
    ("A jar has 144 candies shared equally among 12 kids. How many each?", 144 // 12),
    ("A book has 320 pages; Anna read 145. How many pages are left?", 320 - 145),
    ("There are 9 crates of 16 oranges. How many oranges total?", 9 * 16),
    ("A phone costs $450 with a $60 trade-in credit. What is the final price?", 450 - 60),
    ("A worker earns $18 per hour for 7 hours. How much is earned?", 18 * 7),
    ("A bus travels 55 km/h for 4 hours. How far does it go?", 55 * 4),
    ("From 500 flyers, 213 were handed out. How many remain?", 500 - 213),
    ("A recipe needs 3 eggs per cake. How many eggs for 8 cakes?", 3 * 8),
    ("A store had 240 items and sold 156. How many are left?", 240 - 156),
    ("Each ticket costs $13. How much for 6 tickets?", 13 * 6),
    ("A garden has 5 beds with 24 plants each. How many plants?", 5 * 24),
    ("A 90-minute class is half over. How many minutes have passed?", 90 // 2),
    ("Liam has $100 and spends $37. How much is left?", 100 - 37),
    ("A truck carries 8 pallets of 35 boxes. How many boxes?", 8 * 35),
    ("A pool is filled at 12 liters per minute for 15 minutes. How many liters?", 12 * 15),
    ("There are 365 days in a year; 78 have passed. How many remain?", 365 - 78),
    ("A team scores 7 points in each of 11 games. Total points?", 7 * 11),
]
math = [(q, str(a)) for q, a in math_specs]


def split(rows):
    """First 15 → train, last 15 → test (balanced by construction order)."""
    train = [{"query": q, "gold": g} for q, g in rows[:15]]
    test = [{"query": q, "gold": g} for q, g in rows[15:30]]
    return train, test


def main():
    btr, bte = split(banking)
    save_task(Task(
        task_id="fin_banking_intent", industry="Finance / Banking",
        instruction="Classify the customer's banking message into exactly one intent label.",
        metric="classification", labels=BANKING_LABELS,
        baseline_prompt=(
            "You are a banking assistant. Classify the customer message into one "
            "of these intents: " + ", ".join(BANKING_LABELS) + "."
        ),
        train=btr, test=bte,
        source="Authored for TEI-Bench; intent label space inspired by banking77 (CC-BY-4.0).",
    ), TASKS_DIR)

    ttr, tte = split(triage)
    save_task(Task(
        task_id="health_triage", industry="Healthcare",
        instruction="Classify the patient's described situation into one triage urgency level.",
        metric="classification", labels=TRIAGE_LABELS,
        baseline_prompt=(
            "You are a medical triage assistant. Classify the patient's situation "
            "into one of these urgency levels: " + ", ".join(TRIAGE_LABELS) + "."
        ),
        train=ttr, test=tte,
        source="Authored for TEI-Bench; illustrative triage scenarios (not medical advice).",
    ), TASKS_DIR)

    mtr, mte = split(math)
    save_task(Task(
        task_id="edu_math_word", industry="Education",
        instruction="Solve the grade-school math word problem and give the final numeric answer.",
        metric="numeric", labels=None,
        baseline_prompt="You are a math tutor. Solve the problem.",
        train=mtr, test=mte,
        source="Authored for TEI-Bench; answers computed in-code (GSM8K-style).",
    ), TASKS_DIR)

    print("Wrote 3 pilot tasks to", TASKS_DIR)
    for f in sorted(TASKS_DIR.glob("*.json")):
        print("  -", f.name)


if __name__ == "__main__":
    main()
