import os
import re
import string
import xml.etree.ElementTree as ET
from collections import Counter

import nltk
import numpy as np
from nltk import word_tokenize, TrigramCollocationFinder, QuadgramCollocationFinder
from nltk.collocations import AbstractCollocationFinder


def parse_xml(file):
    tree = ET.parse(file)
    return tree.getroot()


def get_sentence_info(child):
    return child.get('id'), child.get('text')


def tokenize(text):
    tokenized_sent = word_tokenize(text)
    tokenized_info = []
    current_index = 0

    for word in tokenized_sent:

        if not re.match("[" + string.punctuation + "]", word):
            for match in re.finditer(word, text):
                if match.start() >= current_index:
                    tokenized_info.append((word, match.start(), match.end() - 1))
                    current_index = match.end() - 1
                    break
    return tokenized_info


def evaluate(inputdir, outputfile):
    return os.system("java -jar ../eval/evaluateNER.jar " + inputdir + " ../output/" + outputfile)


def extract_entities(token_list):
    entities = []
    previous_token_offset = (0, 0)

    # TODO: Revisar treure tokens majuscules d'una lletra 'A'
    for token in token_list:
        if token[0].isupper():
            pattern = re.compile("[AEIOU]")
            # TODO Check if es la segona paraula en majuscules potser hem de canviar el type també
            if len(entities) > 0 and previous_token_offset[1] + 2 == token[1]:
                entities[len(entities) - 1]['name'] += " " + token[0]
                entities[len(entities) - 1]['offset'] = str(previous_token_offset[0]) + "-" + str(token[2])
            elif not bool(pattern.search(token[0])):
                entities.append({'name': token[0],
                                 'offset': str(token[1]) + "-" + str(token[2]),
                                 'type': "drug_n"})
            else:
                entities.append({'name': token[0],
                                 'offset': str(token[1]) + "-" + str(token[2]),
                                 'type': "brand"})
            previous_token_offset = (token[1], token[2])
            continue

        if len(entities) > 0 and previous_token_offset[1] + 2 == token[1] and any(
                substring in token[0].lower() for substring in ['agent', 'inhibitor', 'blocker', 'drug', 'type', 'medication', 'contraceptive', 'anticoagulants']):
            entities[len(entities) - 1]['name'] += " " + token[0]
            entities[len(entities) - 1]['offset'] = str(previous_token_offset[0]) + "-" + str(token[2])
            entities[len(entities) - 1]['type'] = "group"
            continue

        if token[0].lower() in ['digoxin', 'warfarin', 'phenytoin', 'theophylline', 'lithium', 'ketoconazole',
                                'cimetidine',
                                'alcohol', 'cyclosporine', 'erythromycin', 'tricyclic antidepressants', 'aspirin',
                                'carbamazepine', 'rifampin', 'amiodarone', 'quinidine', 'phenobarbital', 'indinavir',
                                'propranolol', 'methotrexate', 'diltiazem', 'cisapride',
                                'ethanol']:
            if len(entities) > 0 and previous_token_offset[1] + 2 == token[1]:
                entities[len(entities) - 1]['name'] += " " + token[0]
                entities[len(entities) - 1]['offset'] = str(previous_token_offset[0]) + "-" + str(token[2])
            else:
                entities.append({'name': token[0],
                                 'offset': str(token[1]) + "-" + str(token[2]),
                                 'type': "drug"})
            previous_token_offset = (token[1], token[2])
            continue

        if any(substring in token[0].lower() for substring in
               ['anticoagulant', 'corticosteroid', 'NSAID', 'antacid', 'contraceptive', 'diuretic', 'barbiturate']):
            if len(entities) > 0 and previous_token_offset[1] + 2 == token[1]:
                entities[len(entities) - 1]['name'] += " " + token[0]
                entities[len(entities) - 1]['offset'] = str(previous_token_offset[0]) + "-" + str(token[2])
            else:
                entities.append({'name': token[0],
                                 'offset': str(token[1]) + "-" + str(token[2]),
                                 'type': "group"})
            previous_token_offset = (token[1], token[2])
            continue

        # if re.search("(([A-Z]+)|(\d+))\-(([A-Z]+)|(\d+))", token[0]) and re.search("^(\d+[\-\.]\d+)$|^(\d+\.\d+\-\d+\.\d+)$", token[0]) is None:
        #     if len(entities) > 0 and previous_token_offset[1] + 2 == token[1]:
        #         entities[len(entities) - 1]['name'] += " " + token[0]
        #         entities[len(entities) - 1]['offset'] = str(previous_token_offset[0]) + "-" + str(token[2])
        #     else:
        #         entities.append({'name': token[0],
        #                          'offset': str(token[1]) + "-" + str(token[2]),
        #                          'type': "drug_n"})
        #     previous_token_offset = (token[1], token[2])
        #     continue

        if re.search("[a-z][\-][a-z]", token[0]) and re.search("^(\d+[\-\.]\d+)$|^(\d+\.\d+\-\d+\.\d+)$",
                                                               token[0]) is None:
            if len(entities) > 0 and previous_token_offset[1] + 2 == token[1]:
                entities[len(entities) - 1]['name'] += " " + token[0]
                entities[len(entities) - 1]['offset'] = str(previous_token_offset[0]) + "-" + str(token[2])
            else:
                entities.append({'name': token[0],
                                 'offset': str(token[1]) + "-" + str(token[2]),
                                 'type': "group"})
            previous_token_offset = (token[1], token[2])
            continue

        if re.search("\w[_%()\-]\w", token[0]) and re.search("^(\d+[\-\.]\d+)$|^(\d+\.\d+\-\d+\.\d+)$",
                                                             token[0]) is None:
            if len(entities) > 0 and previous_token_offset[1] + 2 == token[1]:
                entities[len(entities) - 1]['name'] += " " + token[0]
                entities[len(entities) - 1]['offset'] = str(previous_token_offset[0]) + "-" + str(token[2])
            else:
                entities.append({'name': token[0],
                                 'offset': str(token[1]) + "-" + str(token[2]),
                                 'type': "drug_n"})
            previous_token_offset = (token[1], token[2])
            continue

        # suffixes = ("azole", "idine", "amine", "mycin")
        suffixes = (
            "afil", "asone", "bicin", "bital", "caine", "cillin", "cycline", "dazole", "dipine",
            "dronate", "eprazole", "fenac", "floxacin", "gliptin", "glitazone", "iramine", "lamide", "mab",
            "mustine", "mycin", "nacin", "nazole", "olol", "olone", "olone", "onide", "oprazole", "parin",
            "phylline", "pramine", "pril", "profen", "ridone", "sartan", "semide", "setron", "setron", "statin",
            "tadine", "tadine", "terol", "thiazide", "tinib", "trel", "tretin", "triptan", "tyline", "vudine",
            "zepam", "zodone", "zolam", "zosin", "ine")
        if token[0].endswith(suffixes):
            if len(entities) > 0 and previous_token_offset[1] + 2 == token[1]:
                entities[len(entities) - 1]['name'] += " " + token[0]
                entities[len(entities) - 1]['offset'] = str(previous_token_offset[0]) + "-" + str(token[2])
            else:
                entities.append({'name': token[0],
                                 'offset': str(token[1]) + "-" + str(token[2]),
                                 'type': "drug"})
            previous_token_offset = (token[1], token[2])
            continue

        prefixes = ("anti")
        if token[0].startswith(prefixes) or "POC" in token[0]:
            if len(entities) > 0 and previous_token_offset[1] + 2 == token[1]:
                entities[len(entities) - 1]['name'] += " " + token[0]
                entities[len(entities) - 1]['offset'] = str(previous_token_offset[0]) + "-" + str(token[2])
            else:
                entities.append({'name': token[0],
                                 'offset': str(token[1]) + "-" + str(token[2]),
                                 'type': "group"})
            previous_token_offset = (token[1], token[2])
            continue

        if token[0][0].isupper():
            if len(entities) > 0 and previous_token_offset[1] + 2 == token[1]:
                entities[len(entities) - 1]['name'] += " " + token[0]
                entities[len(entities) - 1]['offset'] = str(previous_token_offset[0]) + "-" + str(token[2])
            else:
                entities.append({'name': token[0],
                                 'offset': str(token[1]) + "-" + str(token[2]),
                                 'type': "brand"})
            previous_token_offset = (token[1], token[2])
            continue
    return entities


def output_entities(sid, entities, output_file):
    for entity in entities:
        output_file.write(sid + "|" + entity['offset'] + "|" + entity['name'] + "|" + entity['type'] + "\n")


def get_postag_counts(token_list, entities, previous_tag_dict, entity_tag_dict):
    for entity in entities:
        entity_token = entity
        if " " in entity_token:
            entity_token = entity_token.split(" ")[0]
        if "-" in entity_token:
            entity_token = entity_token.split("-")[0]
        if "(" in entity_token:
            entity_token = entity_token.split("(")[0]
        if "." in entity_token:
            entity_token = entity_token.split(".")[0]

        for i, token in enumerate(token_list):
            if entity_token in token:
                entity_index = i
        tags = nltk.pos_tag(token_list)

        entity_tag = tags[entity_index][1]
        previous_tag = tags[entity_index - 1][1]

        if entity_tag not in entity_tag_dict:
            entity_tag_dict[entity_tag] = 0
        entity_tag_dict[entity_tag] += 1

        if previous_tag not in previous_tag_dict:
            previous_tag_dict[previous_tag] = 0
        previous_tag_dict[previous_tag] += 1

    return previous_tag_dict, entity_tag_dict


def get_truth_entities(child):
    return [ent.get('text') for ent in child.findall('entity')], [ent.get('type') for ent in child.findall('entity')]


def longestCommonPrefix(strs):
    longest_pre = ""
    if not strs: return longest_pre
    shortest_str = min(strs, key=len)
    for i in range(len(shortest_str)):
        if all([x.startswith(shortest_str[:i + 1]) for x in strs]):
            longest_pre = shortest_str[:i + 1]
        else:
            break
    return longest_pre


def print_truth_patterns(input_directory):
    all_entities = []
    all_entities_type = []
    all_token_list = []

    previous_tag_dict = {}
    entity_tag_dict = {}

    multiple_words_by_type = {}

    for filename in os.listdir(input_directory):
        root = parse_xml(input_directory + filename)
        print(" - File:", filename)

        for child in root:
            sid, text = get_sentence_info(child)
            token_list = word_tokenize(text)
            entities, entities_type = get_truth_entities(child)

            for entity, type in zip(entities, entities_type):
                if len(entity.split(' ')) > 1:
                    if type not in multiple_words_by_type:
                        multiple_words_by_type[type] = 0
                    multiple_words_by_type[type] += 1

            all_entities.append(entities)
            all_entities_type.append(entities_type)
            all_token_list.append(token_list)

            get_postag_counts(token_list, entities, previous_tag_dict, entity_tag_dict)
    sorted_previous_tag = {k: v for k, v in
                           sorted(previous_tag_dict.items(), key=lambda item: item[1], reverse=True)}
    sorted_entity_tag = {k: v for k, v in sorted(entity_tag_dict.items(), key=lambda item: item[1], reverse=True)}

    # print("LONGEST PREFIX:", longestCommonPrefix(all_entities))
    print("MULTIPLE WORDS COUNT:", multiple_words_by_type)

    all_entities = [item for sublist in all_entities for item in sublist]
    all_entities_type = [item for sublist in all_entities_type for item in sublist]

    print(Counter(all_entities))
    for entity, entity_type in zip(all_entities, all_entities_type):
        if entity.startswith("phe"):
            print(entity, "-", entity_type)

    all_entities.insert(0, "  ")
    all_entities.append("  ")
    entities_string = "  ".join(all_entities)

    finder = QuadgramCollocationFinder.from_words(entities_string)
    finder.apply_freq_filter(10)

    quadgrams = [tr for tr in finder.ngram_fd.items()]
    quadgrams = sorted(quadgrams, key=lambda tup: tup[1], reverse=True)

    for quadgram in quadgrams:
        if quadgram[0][0] == ' ':
            # Prefix
            prefix = ''.join(quadgram[0][1:])
            entities_with_prefix = [ent for ent in all_entities if ent.startswith(prefix)]
            entities_type_with_prefix = {}
            for ent, ent_type in zip(all_entities, all_entities_type):
                if ent.startswith(prefix):
                    if ent_type not in entities_type_with_prefix:
                        entities_type_with_prefix[ent_type] = 0
                    entities_type_with_prefix[ent_type] += 1
            print("Longest Prefix:", longestCommonPrefix(entities_with_prefix), len(entities_with_prefix))
            print(entities_type_with_prefix)
    print("PREVIOUS TAGS:\n", sorted_previous_tag)
    print("ENTITY TAGS:\n", sorted_entity_tag)
    print("QUADGRAMS:\n", quadgrams)


if __name__ == '__main__':
    output_file_name = "task9.1_out_1.txt"
    input_directory = '../data/Train/'

    output_file = open('../output/' + output_file_name, 'w+')
    # print_truth_patterns(input_directory)
    for filename in os.listdir(input_directory):
        root = parse_xml(input_directory + filename)
        print(" - File:", filename)
        for child in root:
            sid, text = get_sentence_info(child)
            token_list = tokenize(text)
            entities = extract_entities(token_list)
            if entities:
                print(entities)
            output_entities(sid, entities, output_file)
    # Close the file
    output_file.close()
    print(evaluate(input_directory, output_file_name))
