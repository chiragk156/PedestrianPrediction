import os
import csv

path = './extracted'
directions = [0,45,90,135,180,225,270,315]

for dir in directions:
    csv_file = open(os.path.join(path,'labels_'+str(dir)+'.csv'))
    line = csv_file.readline()
    f = open('./extracted/labels1_'+str(dir)+'.csv', 'w')
    while line:
        data = line.split(',')
        print(data)
        writer = csv.writer(f)
        if len(data)>1:
            writer.writerow([data[0],int(data[1].split("\\")[0]),str(dir)])
        line = csv_file.readline()
    f.close()
    csv_file.close()