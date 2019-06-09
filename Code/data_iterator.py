from pymongo import MongoClient


class data_iterator:
    def __init__(self, *args, **kwargs):
        self.batch_size = kwargs['batch_size']
        self.num_batch = kwargs['num_batch']
        self.ids = kwargs['ids']
        self.current_batch = 0
        self.data = [i for i in range(0, 1000)]
        self.labels = [0 for i in range(0, 1000)]

    def infos(self):
        print(self.batch_size, self.num_batch)

    def next(self):
        self.current_batch += 1

    def reset(self):
        self.current_batch = 0

    @property
    def provide_data(self):
        return self.data[self.current_batch * self.batch_size : (self.current_batch + 1) * self.batch_size]
    @property
    def provide_label(self):
        return self.labels[self.current_batch * self.batch_size : (self.current_batch + 1) * self.batch_size]