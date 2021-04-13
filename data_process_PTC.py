import torch 
import json 
import os 

# read propoganda data and convert to json file.
articles = os.listdir("data/propoganda/train-articles")

#self.name2id = {"Red Herring":"Presenting Irrelevant Data (Red Herring)", "Misrepresentation of Someone's Position (Straw Man)": 1, "Whataboutism": 2, "Causal Oversimplification": 3, "Obfuscation, Intentional vagueness, Confusion": 4, "Appeal to authority": 5, "Black-and-white Fallacy/Dictatorship": 6, "Name calling/Labeling": 7, "Loaded Language": 8, "Exaggeration/Minimisation": 9, "Flag-waving": 10, "Doubt": 11, "Appeal to fear/prejudice": 12, "Slogans": 13, "Thought-terminating": 14, "Bandwagon": 15, "Reductio ad hitlerum": 16, "Repetition": 17, "Smears": 18, "Glittering generalities (Virtue)": 19, "None": 20}
distinct_techniques = [
 'Flag-Waving',
 'Name_Calling,Labeling',
 'Causal_Oversimplification',
 'Loaded_Language',
 'Appeal_to_Authority',
 'Slogans',
 'Appeal_to_fear-prejudice',
 'Exaggeration,Minimisation',
 'Bandwagon,Reductio_ad_hitlerum',
 'Thought-terminating_Cliches',
 'Repetition',
 'Black-and-White_Fallacy',
 'Whataboutism,Straw_Men,Red_Herring',
 'Doubt'
]     
technique_pair = {'Flag-Waving':"Flag-waving",  
                    'Name_Calling,Labeling': "Name calling/Labeling",
                    'Causal_Oversimplification': "Causal Oversimplification",
                     "Loaded_Language": "Loaded Language",
                     'Appeal_to_Authority': "Appeal to authority",
                     'Slogans': "Slogans",
                      'Appeal_to_fear-prejudice': "Appeal to fear/prejudice",
                      'Exaggeration,Minimisation':'Exaggeration/Minimisation',
                      'Bandwagon,Reductio_ad_hitlerum': ['Bandwagon', "Reductio ad hitlerum"],
                      'Thought-terminating_Cliches': "Thought-terminating",
                      "Repetition": "Repetition",
                      "Black-and-White_Fallacy": "Black-and-white Fallacy/Dictatorship",
                      'Whataboutism,Straw_Men,Red_Herring': ['Whataboutism', "Misrepresentation of Someone's Position (Straw Man)", "Presenting Irrelevant Data (Red Herring)"],
                      'Doubt':'Doubt'
                    }

all_data = []
not_found_count = 0
total = 0
for item in articles:
    # read sentence with character boundaries.
    open_file = open("data/propoganda/train-articles/%s"%item, "r", encoding='utf-8').readlines()
    open_file1 = open("data/propoganda/train-articles/%s"%item, "r", encoding='utf-8').read()
    
    # print(open_file)
    articles_id = os.path.basename( item ).split(".")[0][7:]
    sents = {}
    character_count = 0
    for line in open_file:
        start = character_count
        character_count += len(line)
        sents[line] = [start,character_count]

    # read target.
    cur_techniques = {}
    label_file = open("data/propoganda/train-labels-task2-technique-classification/article%s.task2-TC.labels"%articles_id, "r", encoding='utf-8').readlines()
    for row in label_file:
        found = False
        total += len(label_file)
        article_id, technique, start_id, end_id = row.split("\t")
        
        for sent, pair in sents.items():
            if pair[0] <= int(start_id) and pair[1] >= int(end_id):
                # find.
                covert_tech = technique_pair[technique]
                if type(covert_tech) == str:
                    final_label = [covert_tech]
                else:
                    final_label = covert_tech
                if sent not in cur_techniques:
                    cur_techniques[sent] = final_label
                else:
                     cur_techniques[sent] += (final_label)
                found = True 
                break
        if found == False:
            print("##", row)
            not_found_count += 1
    

   # assert sum([len(y) for y in cur_techniques.values()]) == len(label_file)
    for key, value in cur_techniques.items():
        cur_data = {"id":article_id+"_%s"%list(sents).index(key), "labels": value, "text":key.strip().replace('\"', '"').split(), "label_ids": []}
        all_data.append(cur_data)

    
    
out_file = open("data/propoganda/training.json", "w")
print("unable to find ", not_found_count)
print("Total ", total)
print(len(all_data))
json.dump(all_data[:int(len(all_data)*0.8)], out_file)

out_file = open("data/propoganda/dev.json", "w")
json.dump(all_data[int(len(all_data)*0.8):], out_file)





