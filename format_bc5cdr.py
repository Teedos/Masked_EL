import json
def create_dictionary():
    entities = []
    with open('/share/project/biomed/hcd/BioMedical-EL/data/bc5cdr-c/raw_data/entities.txt','r') as f:
        for line in f:
            entity = {}
            line = line.strip()
            lst = line.split('\t')
            id_ = lst[0]
            name = lst[1]
            entity["mention_id"] = id_
            #entity["title"] = name
            entity["mention"] = name
            entity["type"] = "chemical"
            entities.append(entity)
    
    with open('./data/bc5cdr-c/entity_documents.json','w') as f:
        for  entity in entities:
            json.dump(entity,f)
            f.write("\n")
        
def create_mentions():
    ##need to create : {"mention",mention_id, context_left, context_right, context_doc_id, type, label_id, label, label_title}
    
    splits = ['train', 'dev','test']
    
    for split in splits:
        # with open('/share/project/biomed/hcd/BioMedical-EL/data/bc5cdr-c/raw_data/'+split+'_corpus.txt','r') as f:
        #     for line in f:
        #         line = line.strip()
        #         if '|t|' in line:
        #             lst = line.split('|')
        #             title = lst[2]
        #             id_ = lst[0]
        
        with open('/share/project/biomed/hcd/BioMedical-EL/data/bc5cdr-c/processed_data/'+split+'/documents/documents.json','r') as f:
                documents = {}
                for line in f:
                    document = json.loads(line.strip())
                    id_ = document['document_id'].split('_')[0]
                    if id_ not in documents:
                        documents[id_] = document['text']
                    else:
                        documents[id_] = documents[id_] + ' ' + document['text']
             
        with open('/share/project/biomed/hcd/BioMedical-EL/data/bc5cdr-c/processed_data/'+split+'/mentions/mentions.json','r') as f:
            mentions = []
            for line in f:
                #print(type(line))
                line = json.loads(line.strip())
                for mention in line:
                    document_id = mention['content_document_id'].split('_')[0]
                    start = mention['start_index']
                    end = mention['end_index']
                    context_left = documents[document_id][:start]
                    context_right = documents[document_id][end+1:]
                    mentions.append({'mention': mention['text'],'context_left':context_left,'context_right':context_right,'context_doc_id':document_id,'type':'chemical','label_id':mention['label_candidate_id'],'label':mention['text']})
                #print(line)
        if split == 'dev':
            split = 'val'
        with open('./data/bc5cdr-c/processed/'+split+'.jsonl','w') as f:
            for mention in mentions:
                json.dump(mention,f)
                f.write('\n')
#create_mentions()
create_dictionary()