import re
import json
from datetime import datetime
from collections import defaultdict

# Read the file
with open("urls/fysikk_urls.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

# Parse entries
entries = []
for line in lines[:47]:  # Only process actual lecture entries (lines 1-46)
    line = line.strip()
    if not line or line.startswith("#") or "If you need" in line:
        continue

    # Match markdown link format: - [TFY4107 - DD.MM.YYYY - Økt N](URL)
    match = re.match(r"- \[([^\]]+)\]\(([^\)]+)\)", line)
    if match:
        title = match.group(1)
        url = match.group(2)

        # Extract date and session info
        date_match = re.search(r"(\d{2}\.\d{2}\.\d{4})", title)
        if date_match:
            date_str = date_match.group(1)
            try:
                date_obj = datetime.strptime(date_str, "%d.%m.%Y")
                day_name = date_obj.strftime("%A")

                # Determine if it's Tuesday or Thursday
                if day_name == "Tuesday":
                    day_key = "tuesday"
                elif day_name == "Thursday":
                    day_key = "thursday"
                else:
                    day_key = day_name.lower()

                # Extract session number (Økt 1 or Økt 2)
                session_match = re.search(r"Økt\s*(\d+)", title)
                session_num = session_match.group(1) if session_match else None

                # Check if it's a STØT session
                is_stot = "STØT" in title

                entries.append(
                    {
                        "title": title,
                        "url": url,
                        "date": date_str,
                        "date_obj": date_obj,
                        "day_name": day_name,
                        "day_key": day_key,
                        "session": session_num,
                        "is_stot": is_stot,
                    }
                )
            except ValueError:
                # Handle date parsing errors
                pass
        else:
            # Handle entries without dates (like STØT entries)
            if "STØT" in title:
                entries.append(
                    {
                        "title": title,
                        "url": url,
                        "date": None,
                        "date_obj": None,
                        "day_name": None,
                        "day_key": "special",
                        "session": None,
                        "is_stot": True,
                    }
                )


# Calculate ISO week numbers
def get_iso_week(date_obj):
    if date_obj is None:
        return None
    return date_obj.isocalendar()[1]


# Group by week and day
grouped = defaultdict(lambda: defaultdict(lambda: {"økt1": None, "økt2": None}))
special_entries = []

for entry in entries:
    if entry["is_stot"] and entry["date_obj"] is None:
        # Special entries without dates
        special_entries.append({"title": entry["title"], "url": entry["url"]})
        continue

    week_num = get_iso_week(entry["date_obj"])
    day_key = entry["day_key"]

    if entry["session"] == "1":
        grouped[week_num][day_key]["økt1"] = {
            "title": entry["title"],
            "url": entry["url"],
        }
    elif entry["session"] == "2":
        grouped[week_num][day_key]["økt2"] = {
            "title": entry["title"],
            "url": entry["url"],
        }

# Convert to sorted JSON structure
result = {"weeks": []}

# Sort weeks
for week_num in sorted([w for w in grouped.keys() if isinstance(w, int)]):
    week_data = {"week": week_num, "days": {}}

    # Get dates for this week
    week_dates = {}
    for entry in entries:
        if entry["date_obj"] and get_iso_week(entry["date_obj"]) == week_num:
            day_key = entry["day_key"]
            if day_key not in week_dates:
                week_dates[day_key] = entry["date"]

    # Add Tuesday
    if "tuesday" in grouped[week_num]:
        tuesday_data = grouped[week_num]["tuesday"]
        week_data["days"]["tuesday"] = {
            "date": week_dates.get("tuesday"),
            "økt1": tuesday_data["økt1"],
            "økt2": tuesday_data["økt2"],
        }

    # Add Thursday
    if "thursday" in grouped[week_num]:
        thursday_data = grouped[week_num]["thursday"]
        week_data["days"]["thursday"] = {
            "date": week_dates.get("thursday"),
            "økt1": thursday_data["økt1"],
            "økt2": thursday_data["økt2"],
        }

    result["weeks"].append(week_data)

# Add special entries if any
if special_entries:
    result["special"] = special_entries

# Output JSON
output_file = "urls/fysikk_urls.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print(f"JSON saved to {output_file}")
print(f"Total weeks: {len(result['weeks'])}")
