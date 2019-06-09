from pymongo import MongoClient

class DBIterator(object):

    """
    A class to iterate over a collection in mongodb and access it by index

     Attributes
    ----------
    collection: str
        the name of the collection to iterate over
    limit: int
        The maximum amount of element to be return by the iterator
    filters: filters to apply to db queries
    """


    def __init__(self, collection, limit = 0, filters = {}):
        client = MongoClient('localhost', 27017)
        self.db = client.TFE
        self.collection = self.db[collection]
        self.filters = filters
        if limit == 0:
            self.size = self.collection.count_documents(self.filters)
            self.indexes = [x['_id'] for x in self.collection.find(self.filters, {'_id' : True})]
        else:
            self.size = limit
            self.indexes = [x['_id'] for x in self.collection.find(self.filters, {'_id' : True}).limit(limit)]
        self.limit = limit
 
    def __iter__(self):
        if self.limit == 0:
            for obj in self.collection.find(self.filters):
                yield obj
        else:
            for obj in self.collection.find(self.filters).limit(self.limit):
                yield obj
            
    def __getitem__(self, idx):
        res = self.collection.find_one({'_id' : self.indexes[idx]})
        return res
    
    def __len__(self):
        if self.limit == 0:
            return self.size
        else:
            return self.limit

    def iterTags(self):
        if self.limit == 0:
            for t in self.collection.find(self.filters, {'type' : 1, '_id' : 0}):
                yield t['type']
        else:
            for t in self.collection.find({'_id' : {'$in' : self.indexes}}, {'type' : 1, '_id' : 0}):
                yield t['type']

class normalizedDBIterator(DBIterator):
    """
    A class to iterate over a mongodb collection, specificaly targeting fake and reliable news without nytime and beforeitsnews domains.
    """
    def __init__(self, collection, filters={}, maxLimit = 0):
        super().__init__(collection, limit=0, filters={'type' : {'$in' : ['fake', 'reliable']}})
        self.size += maxLimit
        self.indexes.append([x for x in self.collection.find({{'type' : {'$in' : ['fake', 'reliable']}, 'domain' : {'$nin' : ['nytimes.com', 'beforeitsnews.com']}}}, {'_id' : True}).limit(maxLimit)])



class TokenizedIterator(object):

    """
    A class to iterate over tokenized text in mongodb collection and access it by index

     Attributes
    ----------
    collection: str
        the name of the collection to iterate over
    limit: int
        The maximum amount of documents to be return by the iterator
    """

    def __init__(self, collection, limit = 0, filters = {}):
        self.dbiter = DBIterator(collection, limit, filters)

    def __iter__(self):
        for obj in self.dbiter:
            yield obj['tokenized_text']

    def __getitem__(self, idx):
        return self.dbiter[idx]['tokenized_text']

    def __len__(self):
        return len(self.dbiter)

    def iterTags(self):
        return self.dbiter.iterTags()
    
    def getTags(self, idx):
        return self.dbiter[idx]['type']

def getFakeNewsNet():
    import json
    import os
    import pandas as pd

    fake_directories = ['../../../Data/FakeNewsNet-master/Data/BuzzFeed/FakeNewsContent', '../../../Data/FakeNewsNet-master/Data/PolitiFact/FakeNewsContent']
    real_directories = ['../../../Data/FakeNewsNet-master/Data/BuzzFeed/RealNewsContent', '../../../Data/FakeNewsNet-master/Data/PolitiFact/RealNewsContent']

    fake_files_list = []
    for fake_dir in fake_directories:
        for root, directory, files in os.walk(fake_dir):
            for name in files:
                fake_files_list.append(os.path.join(root, name))
    real_files_list = []
    for real_dir in real_directories:
        for root, directory, files in os.walk(real_dir):
            for name in files:
                real_files_list.append(os.path.join(root, name))
    # Open the first file in order to retreive dictionary keys
    with open(fake_files_list[0]) as f:
        j = json.loads(f.read())
    keys = j.keys()
    data = pd.DataFrame(columns=keys)
    for file_name in fake_files_list:
        with open(file_name) as f:
            j = json.loads(f.read())
            j['type'] = 'fake'
            data = data.append(j, ignore_index=True)
    for file_name in real_files_list:
        with open(file_name) as f:
            j = json.loads(f.read())
            j['type'] = 'reliable'
            data = data.append(j, ignore_index=True) 

    return data['text'].values, data['type'].values    

