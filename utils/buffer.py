import random


class Buffer(object):
    def __init__(self, maxlen):
        self.index = 0
        self.queue = []
        self.maxlen = maxlen

    def __str__(self):
        return ' '.join([str(i) for i in self.queue])

    def size(self):
        return len(self.queue)

    def isEmpty(self):
        return len(self.queue) == []

    def push(self, data):
        if len(self.queue) == self.maxlen:
            self.queue[self.index] = data
        else:
            self.queue.append(data)
        self.index = (self.index + 1) % self.maxlen

    def replace_priority(self, item, new_priority):
        found = False
        for i in range(len(self.queue)):
            if self.queue[i][1] == item:
                self.queue[i][0] = new_priority
                found = True
                break
        if not found:
            self.push((new_priority, item))

    def sample(self, n, replace=True):
        if replace:
            return random.sample(self.queue, n)
        random.shuffle(self.queue)
        elements = self.queue[:n]
        self.queue = self.queue[n:]
        return elements

    def peek_n_priority(self, n):
        try:
            sorted_queue = sorted(self.queue, key=lambda x: x[0])
            return sorted_queue[:n]
        except IndexError:
            print()
            exit()

    def peek_priority(self):
        try:
            sorted_queue = sorted(self.queue, key=lambda x: x[0])
            return sorted_queue[0]
        except IndexError:
            print()
            exit()

    def pop_priority(self):
        try:
            max = 0
            for i in range(len(self.queue)):
                if self.queue[i][0] > self.queue[max][0]:
                    max = i
            item = self.queue[max]
            del self.queue[max]
            return item
        except IndexError:
            print()
            exit()

    def pop_recency(self):
        if self.index > 0:
            try:
                del self.queue[0]
            except IndexError:
                print()
                exit()


