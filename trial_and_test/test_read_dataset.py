import os
import csv

with open("evaluation_setup//fold1_test.csv", 'r') as f:
    reader = csv.reader(f)
    for i, row in enumerate(reader, 0):
        # Skip the title
        if i == 0:
            continue
        
        print(row)
          
        break