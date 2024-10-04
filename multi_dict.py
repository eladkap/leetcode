class MultiDict:
    def __init__(self):
        self._d = {}
        self._iterator = -1
        self._keys_array = []

    def __str__(self):
        return str(self._d)

    def __repr__(self):
        return repr(self._d)

    def __call__(self):
        pass

    def __len__(self):
        return len(self._d.keys())

    def __getitem__(self, key):
        if key not in self._d.keys():
            raise KeyError('Key not found')
        return self._d[key]

    def __setitem__(self, key, value):
        if key not in self._d.keys():
            self._d[key] = [value]
            self._keys_array.append(key)
        else:
            self._d[key].append(value)

    def __iter__(self):
        return self

    def __next__(self):
        self._iterator += 1
        if self._iterator < len(self._keys_array):
            return self._keys_array[self._iterator]
        else:
            raise StopIteration

    def get_keys(self):
        return self._keys_array


if __name__ == '__main__':
    md = MultiDict()
    md['A'] = 'Alabama'
    md['M'] = 'Mishigen'
    md['M'] = 'Minnesota'
    md['M'] = 'Maine'
    md['O'] = 'Ohio'
    print(len(md))

    print(md)
    print(md.get_keys())

    for key in md:
        print(f"{key}: {md[key]}")
