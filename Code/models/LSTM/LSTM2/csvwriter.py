import codecs
import csv

class csvwriter():
    def __init__(self, fileName, ENCODING = 'utf-8'):
        self.fileName = fileName
        self.ENCODING = ENCODING
        self.fp = codecs.open(self.fileName, "w", self.ENCODING)
        self.writer = csv.writer(self.fp)

    def write(self, array):
        self.writer.writerow(array)
        self.fp.flush()

    def close(self):
        self.fp.close()
