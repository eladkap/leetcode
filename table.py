class Table:
    def __init__(self):
        self.records = []
        self.columns = []

    def add_column(self, column: str):
        self.columns.append(column)

    def add_record(self, rec: dict):
        self.records.append(rec)

    def __str__(self):
        lines = []
        lines.append('\t\t'.join(self.columns))
        for record in self.records:
            line = '\t\t'.join(str(record[column]) for column in self.columns)
            lines.append(line)
        return '\n'.join(lines)


if __name__ == '__main__':
    T = Table()
    T.add_column('name')
    T.add_column('age')
    T.add_column('title')

    rec = {
        'name': 'Alon',
        'age': 20,
        'title': 'Senior'
    }

    T.add_record(rec)

    print(T)
