import os
import json
from ibm_watson import LanguageTranslatorV3
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from deep_translator import GoogleTranslator
authenticator = IAMAuthenticator('FiOu5GczSyMAXXr-UldlpscfALKRMW0oS2WOfhVqulZh')
language_translator = LanguageTranslatorV3(
    version='2018-05-01',
    authenticator=authenticator
)
new_file=[]
if os.stat('final.json').st_size != 0:
    with open('final.json',encoding='utf-8') as fr:
        old_file = json.load(fr)
        for i in range(len(old_file)):
            new_file.append(old_file[i])
    print(len(new_file))
language_translator.set_service_url('https://api.us-south.language-translator.watson.cloud.ibm.com/instances/eb222f3f-367c-4a79-8836-c2375c02e85c')

with open('data/training_set_task1.txt',encoding='utf-8') as f:
    file = json.load(f)
    for i in range(600,680):
        print(i)
        new_file.append(file[i])
        translated = GoogleTranslator(source='en', target='zh').translate(file[i]['text'])
        translation = language_translator.translate(
            text=file[i]['text'].lower(),
            model_id='zh-en').get_result()
        result1=json.loads(json.dumps(translation, indent=2, ensure_ascii=False))['translations'][0]['translation']

        translated = GoogleTranslator(source='en', target='ja').translate(file[i]['text'])
        translation = language_translator.translate(
            text=file[i]['text'],
            model_id='ja-en').get_result()
        result2 = json.loads(json.dumps(translation, indent=2, ensure_ascii=False))['translations'][0]['translation']
        if(result1.lower()==result2.lower()):
            if(result1.lower()!=file[i]['text'].lower()):
                new_file.append({"id":file[i]['id'],"labels":file[i]['labels'],"text":result1})
        else:
            if (result1.lower() != file[i]['text'].lower()):
                new_file.append({"id": file[i]['id'], "labels": file[i]['labels'], "text": result1})
            if (result2.lower() != file[i]['text'].lower()):
                new_file.append({"id": file[i]['id'], "labels": file[i]['labels'], "text": result1})
print(new_file)
print(len(new_file))
with open('final.json','w') as fp:
    json.dump(new_file,fp)

