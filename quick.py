"""
d: dictionary when key is a number and value is tuple of (value, timestamp of last update)
common: None if set_all() never called or a if set_all() called
timestamp: current timestamp of the current run. incremented every call of set() or set_all()
set_all_ts: timestamp of last set_all() call

Logic:
- set(key, value): Everytime a key-value is set, we update its value, increment class timestamp and update the key's timestamp
- set_all(value): We update common, increment class timestamp and update set_all timestamp
- get(key): If timestamp of key is bigger than set_all timestamp, means the key was updated after the last set_all call
so we return its value from dictionary.
Else: means set_all was called after the last update of the key, so we return common value.
"""


class Quick(object):
    def __init__(self):
        self.d = {}
        self.common = None
        self.timestamp = 0
        self.set_all_ts = 0

    def __str__(self):
        l = []
        for key in self.d.keys():
            item = f'{key}: {self.get(key)}'
            l.append(item)
        return '{' + f"{', '.join([str(item) for item in l])}" + '}'

    def set(self, key, value):
        self.timestamp += 1
        self.d[key] = [value, self.timestamp]

    def get(self, key):
        if key not in self.d.keys():
            raise Exception(f'Key not found {key}')
        value, ts = self.d[key]
        if ts > self.set_all_ts:
            return self.d[key][0]
        return self.common

    def set_all(self, value):
        self.common = value
        self.timestamp += 1
        self.set_all_ts = self.timestamp


if __name__ == '__main__':
    Q = Quick()

    Q.set(1, 2)
    Q.set(2, 3)
    Q.set(10, 11)
    Q.set(11, 12)
    # Q.get(9)  # exception: key not found
    print(Q.get(2))  # 3
    print(Q.get(1))  # 2
    print(Q)  # {1: 2, 2: 3, 10: 11}

    Q.set_all(999)
    print(Q)  # {1: 999, 2: 999, 10: 999}
    Q.set(2, 4)
    Q.set(10, 12)
    print(Q)  # {1: 888, 2: 8, 10: 7, 11: 888}

    Q.set_all(888)
    print(Q)  # {1: 888, 2: 888, 10: 888, 11: 888}
