import json

if __name__ == '__main__':
    # Using readlines()
    file1 = open('goldNER.txt', 'r')
    Lines = file1.readlines()

    truth = {}

    # Strips the newline character
    for line in Lines:
        value = line.split("|")
        if value[0] not in truth:
            truth[value[0]] = []
        truth[value[0]].append((value[-2], value[-1]))

    # Using readlines()
    file1 = open('../output/task9.1_out_1.txt', 'r')
    Lines = file1.readlines()

    output = {}
    wrong_entities = []
    new_sent = ""
    matched_entities = []
    missing = []

    # Strips the newline character
    for line in Lines:
        value = line.split("|")
        if value[0] not in output:
            output[value[0]] = []
        output[value[0]].append((value[-2], value[-1]))

        if new_sent != value[0] and new_sent != "":
            if new_sent in truth:
                missing += [item for item in truth[new_sent] if item not in matched_entities]
            matched_entities = []
            new_sent = value[0]

        if new_sent == "":
            new_sent = value[0]

        if value[0] in truth and (value[-2], value[-1]) in truth[value[0]]:
            matched_entities.append((value[-2], value[-1]))
        else:
            wrong_entities.append((value[-2], value[-1]))

    missing_dict = {}
    for ent, type in missing:
        if type[:-1] not in missing_dict:
            missing_dict[type[:-1]] = []
        missing_dict[type[:-1]].append(ent)

    wrong_entities_dict = {}
    for ent, type in wrong_entities:
        if type[:-1] not in wrong_entities_dict:
            wrong_entities_dict[type[:-1]] = []
        wrong_entities_dict[type[:-1]].append(ent)

    print("MISSING:\n", json.dumps(missing_dict, indent=4))
    print("\n\n")
    print("WRONG:\n", json.dumps(wrong_entities_dict, indent=4))
