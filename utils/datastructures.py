

class KM_MismatchDS:
    def __init__(self, path):
        self.path = path
        self.data = self.build()
    def build(self):
        data = {}
        with open(self.path, "r") as data_file:
            for line in data_file:
                kmers = list(map(int, line.split(",")))
                root = kmers[0]
                children = kmers[1:]
                data[root] = set(children)
        return data
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return len(self.data)
                


