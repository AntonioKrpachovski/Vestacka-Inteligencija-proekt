import csv

def read_file(file_name):
    with open(file_name) as doc:
        csv_reader = csv.reader(doc, delimiter=';')
        dataset = list(csv_reader)[1:]

    return dataset