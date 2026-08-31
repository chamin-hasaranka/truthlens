"""
generate_sample_dataset.py
---------------------------
This utility script generates a SAMPLE Fake.csv and True.csv dataset so that
train_model.py can run end-to-end immediately after installation, without
requiring the user to manually download the full Kaggle "Fake and Real News"
dataset first.

NOTE FOR PRODUCTION USE:
For best real-world accuracy, replace these generated files with the full
Kaggle dataset:
    https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
Simply download Fake.csv and True.csv from that link and place them in this
"dataset/" folder, overwriting the sample files. The schema (columns:
title, text, subject, date) is identical, so train_model.py requires no
changes.

This script builds a few hundred synthetic-but-realistic news samples using
templated combinations of real-world topics, phrasing patterns commonly
found in legitimate journalism (True.csv) vs. sensationalist / clickbait
phrasing commonly found in fabricated news (Fake.csv).
"""

import csv
import random
import os

random.seed(42)

REAL_TOPICS = [
    "the central bank's interest rate decision",
    "the latest unemployment figures released by the labor department",
    "a new trade agreement signed between two countries",
    "the parliamentary vote on the proposed infrastructure bill",
    "the quarterly earnings report from a major technology company",
    "ongoing diplomatic talks aimed at easing regional tensions",
    "the results of the national census",
    "a newly published peer-reviewed study on climate trends",
    "the city council's approval of a new public transit expansion",
    "the health ministry's updated vaccination guidelines",
    "the supreme court's ruling on a long-running constitutional case",
    "the finance minister's budget announcement for the upcoming fiscal year",
    "a joint statement issued after the regional summit",
    "the central election commission's certification of results",
    "a report from the national weather service regarding the storm system",
]

REAL_TEMPLATES = [
    "Officials confirmed on {day} that {topic} will move forward after months of deliberation. "
    "According to a statement released by the relevant department, the decision was reached following "
    "extensive consultation with stakeholders and independent experts. Analysts say the move reflects a "
    "broader effort to address long-standing concerns raised by the public and industry groups alike. "
    "The department added that further details would be released in the coming weeks, and that an "
    "implementation timeline would be published once final approvals are completed. Representatives from "
    "opposing parties offered measured responses, with several requesting additional clarification before "
    "the next session.",

    "A report published on {day} detailed the findings related to {topic}, noting that the data had been "
    "reviewed by multiple independent analysts prior to release. The report's authors emphasized that the "
    "conclusions were based on verified figures collected over several months, and cautioned against drawing "
    "premature conclusions without considering the full context of the underlying data. Spokespersons for "
    "several agencies confirmed the accuracy of the figures in separate statements to reporters, and said "
    "they welcomed continued scrutiny of the methodology used.",

    "Government representatives announced on {day} that {topic} had been finalized after a series of "
    "negotiations that spanned several weeks. In a joint press briefing, officials outlined the next steps "
    "and answered questions from journalists regarding implementation and oversight. Independent observers "
    "noted that the process had followed standard procedural guidelines, and several policy analysts said "
    "the outcome was consistent with expectations based on prior public statements.",

    "According to documents reviewed by reporters, {topic} is expected to proceed as scheduled following "
    "approval on {day}. The relevant authority stated that all procedural requirements had been met, and "
    "that a formal announcement would follow once remaining administrative steps were completed. Several "
    "experts interviewed for this report said the development was in line with broader trends observed "
    "over the past year, while cautioning that final figures could be subject to revision.",
]

FAKE_TOPICS = [
    "a secret group of elites controlling the world's food supply",
    "a miracle cure that doctors and pharmaceutical companies are hiding",
    "a celebrity who was secretly replaced by a body double",
    "a government plot to install mind-control chips in vaccines",
    "an alien spacecraft that crashed and was covered up by the military",
    "a politician who was caught on a leaked tape admitting a massive conspiracy",
    "a bank that is about to collapse and wipe out everyone's savings overnight",
    "a new law that will secretly ban a popular everyday product",
    "a scientist who proved the earth is actually flat using shocking new evidence",
    "a billionaire who is secretly funding a plan to control the global economy",
    "a leaked document revealing a hidden world government",
    "a viral cure that big pharma doesn't want you to know about",
]

FAKE_TEMPLATES = [
    "SHOCKING: You won't believe what was just exposed about {topic}! Sources who wish to remain anonymous "
    "say this changes EVERYTHING and the mainstream media REFUSES to cover it. Share this before it gets "
    "taken down!!! Insiders claim that the truth has been hidden from the public for YEARS, and that this "
    "is just the tip of the iceberg. Wake up people, before it's too late!!!",

    "BREAKING bombshell report reveals {topic} and the mainstream media is staying SILENT. A whistleblower "
    "who claims to have worked inside the operation for over a decade says the evidence is undeniable, even "
    "though zero official sources have confirmed any part of this story. Experts are reportedly 'terrified' "
    "to speak out, according to unnamed sources close to the situation. This is the story THEY don't want "
    "you to see!",

    "You won't believe this!!! New leaked footage allegedly proves {topic}, and people online are losing "
    "their minds over it. Despite no verified evidence and zero credible reporting, thousands of shares "
    "claim this is '100% real' and 'can't be denied.' Click here before this gets banned forever. The "
    "establishment is in full panic mode trying to suppress this video!",

    "EXPOSED: The hidden truth about {topic} that NO ONE is talking about. According to totally anonymous "
    "insiders who 'can't be named for their own safety,' everything we've been told is a lie. This shocking "
    "report claims that powerful figures are working around the clock to keep this buried, but brave "
    "truth-seekers are finally bringing it to light. SHARE NOW before they delete this!!!",
]

SUBJECTS_REAL = ["politicsNews", "worldnews", "businessNews"]
SUBJECTS_FAKE = ["News", "conspiracy", "politics"]

DAYS = [
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "January 14", "February 22", "March 9", "April 30", "May 17", "June 3",
    "July 26", "August 11", "September 5", "October 18", "November 2", "December 21",
]


def build_rows(topics, templates, subjects, n_rows, title_prefix):
    rows = []
    for i in range(n_rows):
        topic = random.choice(topics)
        template = random.choice(templates)
        day = random.choice(DAYS)
        text = template.format(topic=topic, day=day)
        title = f"{title_prefix}: {topic.capitalize()}"
        subject = random.choice(subjects)
        date = f"{random.choice(DAYS)}, 2017"
        rows.append([title, text, subject, date])
    return rows


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))

    real_rows = build_rows(REAL_TOPICS, REAL_TEMPLATES, SUBJECTS_REAL, 400, "Report")
    fake_rows = build_rows(FAKE_TOPICS, FAKE_TEMPLATES, SUBJECTS_FAKE, 400, "Alert")

    real_path = os.path.join(out_dir, "True.csv")
    fake_path = os.path.join(out_dir, "Fake.csv")

    with open(real_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["title", "text", "subject", "date"])
        writer.writerows(real_rows)

    with open(fake_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["title", "text", "subject", "date"])
        writer.writerows(fake_rows)

    print(f"Generated {len(real_rows)} rows -> {real_path}")
    print(f"Generated {len(fake_rows)} rows -> {fake_path}")


if __name__ == "__main__":
    main()
