import csv;
import time;
import numpy as np;


def write_into_folder(a, file_name):

    file_path = '../output_files/' + file_name + '.csv'
    print(file_path)
    csvFile = open(file_path, 'w')
    writer = csv.writer(csvFile)

    for i in list(range(0, len(a))):
        writer.writerow(a[i, :])
    csvFile.close()
    print(file_name + ' file write successfully.....')


