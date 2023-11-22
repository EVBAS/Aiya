import json

def txt_to_json(txt_path, json_path):
    data = []

    with open(txt_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        conversation = {"questions": [], "answers": []}

        for line in lines:
            line = line.strip()
            if "__eou__" in line:
                parts = line.split("__eou__")
                if parts[0].strip():
                    conversation["questions"].append(parts[0].strip())
                if parts[1].strip():
                    conversation["answers"].append(parts[1].strip())
            else:
                if conversation["answers"]:
                    conversation["answers"][-1] += " " + line

        if conversation["questions"] or conversation["answers"]:
            data.append({
                "questions": conversation["questions"],
                "answers": conversation["answers"]
            })

    with open(json_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=2)

txt_to_json('awa.txt', 'awa.json')


