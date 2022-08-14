import requests as rq
import json


def running(sentence):
    sentence = sentence.encode('utf-8')
    return rq.post(url, data=sentence).text


def main(data_dir, parsed_data_dir):
    with open(data_dir, 'r', encoding='utf-8') as f:
        sentences = f.readlines()

    resutls = []
    for sentence in sentences:
        resutls.append(running(sentence))

    with open(parsed_data_dir, "w", encoding='utf-8') as f:
        json.dump(resutls, f, indent=4)


if __name__ == '__main__':
    url = r'http://172.16.133.173:8080/?properties={"annotators":"tokenize, ssplit, pos, lemma","outputFormat":"json", "pipelineLanguage":"en"}'
    data_dir = r''
    parsed_data_dir = r''
    main(data_dir, parsed_data_dir)
