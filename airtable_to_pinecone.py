import requests

import re

def clean_srt(text):
    if not text or len(text) < 10:
        return ""
    # Remove SRT timestamps and numbering
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        if re.match(r"^\d+\s*$", line):  # line number
            continue
        if "-->" in line:  # timestamp line
            continue
        if line.strip() == "":
            continue
        cleaned.append(line.strip())
    return " ".join(cleaned)


# === UPDATE THESE VALUES ===
AIRTABLE_PAT = "pathT6pkSO8Fp0QFA.8ee10bf975e086124921f97b80f4c6f0758959d77ca4c73adcdbcb0cc4f79eb3"
BASE_ID = "app2bEfoCTnwLiBn9"
TABLE_NAME = "Table%201"  # URL-encoded space
FLASK_ENDPOINT = "https://plaud-transcript-rag.onrender.com/ingest"

headers = {
    "Authorization": f"Bearer {AIRTABLE_PAT}",
    "Content-Type": "application/json"
}

url = f"https://api.airtable.com/v0/{BASE_ID}/{TABLE_NAME}"
params = {"pageSize": 100}
all_records = []

# === PULL ALL RECORDS ===
while True:
    res = requests.get(url, headers=headers, params=params)
    res.raise_for_status()
    data = res.json()
    all_records.extend(data["records"])
    if "offset" not in data:
        break
    params["offset"] = data["offset"]

print(f"🔁 Found {len(all_records)} records")

# === SEND TO /INGEST ===
for r in all_records:
    fields = r["fields"]
    raw_transcript = fields.get("Transcript")
    transcript = clean_srt(raw_transcript)

    title = fields.get("Title") or "Untitled"

    if not transcript or len(transcript.strip()) < 10:
        print(f"⏭️ Skipping {title} (empty transcript)")
        continue

    res = requests.post(FLASK_ENDPOINT, json={
        "transcript": transcript,
        "title": title
    })

    status = "✅" if res.status_code == 200 else f"❌ {res.status_code}"
    print(f"{status} → {title}")
