import os
import re
import string
import xml.etree.ElementTree as ET
from collections import Counter

import nltk
import pycrfsuite
from chemdataextractor.nlp.tokenize import ChemWordTokenizer
from nltk import word_tokenize, QuadgramCollocationFinder
from nltk.corpus import stopwords


def parse_xml(file):
    tree = ET.parse(file)
    return tree.getroot()


def get_sentence_info(child):
    return child.get('id'), child.get('text')


def chem_tokenize(text):
    cwt = ChemWordTokenizer()
    tokens = cwt.tokenize(text)
    token_indexs = cwt.span_tokenize(text)
    tokenized_info = []
    for token_index, token in zip(token_indexs, tokens):
        tokenized_info.append((token, token_index[0], token_index[1] - 1))
    return tokenized_info


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


def get_external_resources():
    file = open('../resources/DrugBank.txt', 'r', encoding="utf8")
    Lines = file.readlines()

    resources = {}

    # Strips the newline character
    for line in Lines:
        value = line.split("|")
        resources[value[0].lower()] = value[1][:-1]

    hsdb_resources = set()
    file = open('../resources/HSDB.txt', 'r', encoding="utf8")
    Lines = file.readlines()

    for line in Lines:
        value = line[:-1]
        hsdb_resources.add(value.lower())

    drug_n_resources = set()
    file = open('../resources/drug_n.csv', 'r', encoding="utf8")
    Lines = file.readlines()

    for line in Lines:
        value = line[:-1]
        drug_n_resources.add(value.lower())

    return resources, hsdb_resources, drug_n_resources


def extract_features(token_list, entities_dict, drug_n_set, hsdb_set, with_resources=False):
    entities = []
    previous_token_offset = (0, 0)
    stop_words = set(stopwords.words('english'))

    # TODO: Create features for any class (punct, nomes numeros, ...)
    for i, token in enumerate(token_list):
        features = []
        if entities_dict and with_resources:
            if token[0] in entities_dict:
                features.append("in_entities_dict=" + entities_dict[token[0]])
            elif token[0].lower() in entities_dict:
                features.append("in_entities_dict=" + entities_dict[token[0].lower()])
            else:
                features.append("in_entities_dict=False")
        else:
            features.append("in_entities_dict=False")

        if with_resources and token[0] in drug_n_set:
            features.append("is_drug_n=True")
        elif with_resources and token[0].lower() in drug_n_set:
            features.append("is_drug_n=True")
        else:
            features.append("is_drug_n=False")

        if with_resources and token[0] in hsdb_set:
            features.append("in_hsdb_set=True")
        elif with_resources and token[0].lower() in hsdb_set:
            features.append("in_hsdb_set=True")
        else:
            features.append("in_hsdb_set=False")

        lower_letters = sum(1 for c in token[0] if c.isupper())
        upper_letters = sum(1 for c in token[0] if c.isupper())
        num_digits = sum(1 for c in token[0] if c.isdigit())
        num_punctuation = sum(1 for c in token[0] if c in string.punctuation)
        num_roman = len(re.findall("[IVXDLCM]+", token[0]))

        word_shape = lower_letters + upper_letters + num_digits + num_punctuation + num_roman
        features.append("word_shape=" + str(word_shape))

        #features.append("starts_with_uppercase=" + str(token[0][0].isupper()))
        features.append("starts_with_digit=" + str(token[0][0].isdigit()))
        #features.append("more_than_two_uppercase=" + str(upper_letters > 2))

        features.append("is_stopword=" + str(token[0].lower() in stop_words and with_resources))

        stopword_set = {'of', 'the', 'and', 'in', 'with', 'to', 'be', 'or', 'is', 'not', 'by', 'for',
                        'should', 'on', 'that', 'been', 'have', 'other', 'was', 'when', 'are', 'as', 'were',
                        'no', 'has', 'these', 'an', 'this', 'such', 'at', 'from', 'it', 'if', 'there', 'after',
                        'which', 'can', 'between', 'during', 'because', 'both', 'than', 'did', 'its', 'but',
                        'some', 'who', 'any'}
        features.append("in_stopword_set=" + str(token[0].lower() in stopword_set))

        features.append("has_parenthesis=" + str('(' in token[0] and len(token[0]) > 1))
        features.append("is_punct=" + str(token[0] in {'.', ',', ';', ':', '(', ')', '-', '_', '\'', '/', '\\'}))
        features.append("has_lowercase_hyphen=" + str(
            bool(re.search("[a-z][\-][a-z]", token[0]))))

        features.append(
            "is_only_numbers=" + str(re.search("^(\d+[\-\.]\d+)$|^(\d+\.\d+\-\d+\.\d+)$", token[0]) is None))

        features.append("has_hyphen=" + str(bool(re.search("\w[_%()\-]\w", token[0]))))

        features.append("is_upper=" + str(token[0].isupper()))

        pattern = re.compile("[AEIOU]")
        features.append("has_upper_vowel=" + str(bool(pattern.search(token[0]))))

        features.append(
            "has_group_keyword=" + str(bool(len(entities) > 0 and previous_token_offset[1] + 2 == token[1] and any(
                substring in token[0].lower() for substring in
                ['agent', 'inhibitor', 'blocker', 'drug', 'type', 'medication', 'contraceptive', 'anticoagulants']))))

        has_common_drug = token[0].lower() in ['digoxin', 'warfarin', 'phenytoin', 'theophylline', 'lithium',
                                               'ketoconazole', 'cimetidine', 'alcohol', 'cyclosporine', 'erythromycin',
                                               'tricyclic antidepressants', 'aspirin', 'carbamazepine', 'rifampin',
                                               'amiodarone', 'quinidine', 'phenobarbital', 'indinavir', 'propranolol',
                                               'methotrexate', 'diltiazem', 'cisapride', 'ethanol']
        features.append("has_common_drug=" + str(has_common_drug))

        features.append("has_common_group=" + str(any(substring in token[0].lower() for substring in
                                                      ['anticoagulant', 'corticosteroid', 'NSAID', 'antacid',
                                                       'contraceptive', 'diuretic', 'barbiturate'])))

        if len(token[0]) >= 4:
            features.append("suff4=" + token[0][-4:])
            features.append("pref4=" + token[0][:4])
        else:
            features.append("suff4=" + token[0])
            features.append("pref4=" + token[0])

        if len(token[0]) >= 3:
            features.append("suff3=" + token[0][-3:])
            features.append("pref3=" + token[0][:3])
        else:
            features.append("suff3=" + token[0])
            features.append("pref3=" + token[0])

        if len(token[0]) >= 2:
            features.append("suff2=" + token[0][-2:])
            features.append("pref2=" + token[0][:2])
        else:
            features.append("suff2=" + token[0])
            features.append("pref2=" + token[0])
        #
        # if len(token[0]) >= 1:
        #     features.append("suff1=" + token[0][-1:])
        #     features.append("pref1=" + token[0][:1])
        # else:
        #     features.append("suff1=" + token[0])
        #     features.append("pref1=" + token[0])

        features.append("has_poc=" + str("POC" in token[0]))

        features.append("starts_with_uppercase=" + str(token[0][0].isupper()))

        if i != 0:
            features.append("prev_ent=" + token_list[i - 1][0])
            features.append("prev_postag=" + nltk.pos_tag([token_list[i - 1][0]])[0][1][0])
            features.append("prev_len=" + str(len(token_list[i - 1][0])))
        else:
            features.append("prev_ent=")
            features.append("prev_postag=")
            features.append("prev_len=0")

        features.append("curr_ent=" + token_list[i][0])
        features.append("curr_postag=" + nltk.pos_tag([token[0]])[0][1][0])
        features.append("curr_len=" + str(len(token_list[i][0])))

        if i != len(token_list) - 1:
            features.append("next_ent=" + token_list[i + 1][0])
            features.append("next_postag=" + nltk.pos_tag([token_list[i + 1][0]])[0][1][0])
            features.append("next_len=" + str(len(token_list[i + 1][0])))
        else:
            features.append("next_ent=")
            features.append("next_postag=")
            features.append("next_len=0")

        features.append("capitals_inside=" + str(token[0][1:].lower() != token[0][1:]))

        features.append("count_punct=" + str(len(re.findall("[\.\-+\,()]", token[0]))))

        features.append("has_roman=" + str(bool(re.search("[IVXDLCM]+", token[0]))))

        entities.append(features)

    return entities


def output_features(sid, tokens, gold_entities, features, output_file):
    for token, feature_vector, bio in zip(tokens, features, gold_entities):
        output_file.write(
            sid + "\t" + token[0] + "\t" + str(token[1]) + "\t" + str(token[2]) + "\t" + bio + "\t" + "\t".join(
                feature_vector) + "\n")
    output_file.write("\n")


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
    return list(zip([ent.get('text') for ent in child.findall('entity')],
                    [ent.get('charOffset') for ent in child.findall('entity')])), \
           [ent.get('type') for ent in child.findall('entity')]


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


def get_gold_entities(token_list, truth_entities):
    gold_entities = []
    entity_counter = 0

    for j, token in enumerate(token_list):
        if not truth_entities or len(truth_entities[0]) <= entity_counter:
            gold_entities.append("O")
            continue
        if ';' in truth_entities[0][entity_counter][1]:
            entity_offsets = truth_entities[0][entity_counter][1].split(';')
            for i, offset in enumerate(entity_offsets):
                entity_offset = offset.split('-')
                entity_offset[0] = int(entity_offset[0])
                entity_offset[1] = int(entity_offset[1])
                entity_offsets[i] = (entity_offset[0], entity_offset[1])
        else:
            entity_offset = truth_entities[0][entity_counter][1].split('-')
            entity_offset[0] = int(entity_offset[0])
            entity_offset[1] = int(entity_offset[1])
            entity_offsets = [(entity_offset[0], entity_offset[1])]

        for offset in entity_offsets:
            if offset[0] == token[1] and offset[1] == token[2]:
                # Exact match
                gold_entities.append("B-" + truth_entities[1][entity_counter])
                entity_counter += 1
                break
            elif offset[0] == token[1]:
                # Beginning match
                gold_entities.append("B-" + truth_entities[1][entity_counter])
                break
            elif offset[0] < token[1] and offset[1] > token[2]:
                # Inside match
                gold_entities.append("I-" + truth_entities[1][entity_counter])
                break
            elif offset[1] == token[2]:
                # End match
                gold_entities.append("I-" + truth_entities[1][entity_counter])
                entity_counter += 1
                break
        if len(gold_entities) == j:
            gold_entities.append('O')

    return gold_entities


def read_features(filename):
    # Using readlines()
    file1 = open('../output/' + filename, 'r')
    lines = file1.readlines()

    sentences_features = []
    sentences_bios = []

    token_features = []
    bios = []

    # Strips the newline character
    for line in lines:
        if line == "\n":
            # End of sentence
            sentences_features.append(token_features)
            sentences_bios.append(bios)

            token_features = []
            bios = []
            continue
        values = line.split("\t")

        sid = values[0]
        token_name = values[1]
        start_offset = values[2]
        end_offset = values[3]
        bio = values[4]
        feature_vector = values[5:]
        feature_vector[-1].replace('\n', '')

        bios.append(bio)
        token_features.append(feature_vector)

    return sentences_features, sentences_bios


def extract_entities_from_bio(token_list, bio_tags):
    entities = []
    previous_token_offset = (0, 0)

    for token, bio in zip(token_list, bio_tags):
        if bio.startswith('B'):
            entities.append({'name': token[0],
                             'offset': str(token[1]) + "-" + str(token[2]),
                             'type': bio.split("-")[-1]})
            previous_token_offset = (token[1], token[2])
        elif bio.startswith('I') and entities:
            entities[-1]['name'] += " " + token[0]
            entities[-1]['offset'] = str(previous_token_offset[0]) + "-" + str(token[2])
    return entities


if __name__ == '__main__':
    task = "train_crf"  # get_features, train_crf, test
    with_resources = True

    if task == "get_features":
        output_file_name = "train.txt"
        input_directory = '../data/Train/'

        entities_dict, drug_n_set, hsdb_set = get_external_resources()

        output_file = open('../output/' + output_file_name, 'w+')
        # print_truth_patterns(input_directory)
        for filename in os.listdir(input_directory):
            root = parse_xml(input_directory + filename)
            print(" - File:", filename)
            for child in root:
                sid, text = get_sentence_info(child)
                token_list = chem_tokenize(text)
                features = extract_features(token_list, entities_dict, drug_n_set, hsdb_set,
                                            with_resources=with_resources)
                truth_entities = get_truth_entities(child)
                gold_entities = get_gold_entities(token_list, truth_entities)
                output_features(sid, token_list, gold_entities, features, output_file)

        # Close the file
        output_file.close()
    elif task == "train_crf":
        sentences_features, sentences_bios = read_features("train.txt")

        trainer = pycrfsuite.Trainer(verbose=False)

        for xseq, yseq in zip(sentences_features, sentences_bios):
            trainer.append(xseq, yseq)

        trainer.set_params({
            'c1': 1.0,  # coefficient for L1 penalty
            'c2': 1e-3,  # coefficient for L2 penalty
            'max_iterations': 50,  # stop earlier

            # include transitions that are possible, but not observed
            'feature.possible_transitions': True
        })

        trainer.train('crf_model.crfsuite')
    else:
        # Devel
        output_file_name = "task9.1_out_2.txt"
        input_directory = '../data/Devel/'

        entities_dict, drug_n_set, hsdb_set = get_external_resources()

        output_file = open('../output/' + output_file_name, 'w+')
        # print_truth_patterns(input_directory)
        for filename in os.listdir(input_directory):
            root = parse_xml(input_directory + filename)
            print(" - File:", filename)
            for child in root:
                sid, text = get_sentence_info(child)
                token_list = chem_tokenize(text)
                features = extract_features(token_list, entities_dict, drug_n_set, hsdb_set,
                                            with_resources=with_resources)

                tagger = pycrfsuite.Tagger()
                tagger.open('crf_model.crfsuite')
                bio_tags = tagger.tag(features)

                entities = extract_entities_from_bio(token_list, bio_tags)
                output_entities(sid, entities, output_file)

        # Close the file
        output_file.close()
        print(evaluate(input_directory, output_file_name))
