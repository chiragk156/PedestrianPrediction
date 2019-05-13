import os
import csv

path = './labels'
directions = [0,45,90,135,180,225,270,315]

for dir in directions:
    csv_file = open(os.path.join(path,'labels1_'+str(dir)+'.csv'),'r')
    line = csv_file.readline()
    f = open('./labels/labels2_'+str(dir)+'.csv', 'w')
    while line:
        data = line.split(',')
        writer = csv.writer(f)
        if len(data)>2:
            label = int(data[1].split("\\")[0])
            if label == 3:
                writer.writerow([data[0],2,dir])
            else:
                writer.writerow([data[0],label,dir])
        line = csv_file.readline()
    csv_file.close()
    f.close()