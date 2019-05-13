import matplotlib.pyplot as plt
import os
import csv
mydir = './extracted/225/'
category=[]
plt.ion()

f = open('./extracted/labels1_225.csv', 'w')
writer = csv.writer(f)

count=0
for imagename in os.listdir(mydir):
	imagepath = os.path.join(mydir, imagename)
	image = plt.imread(imagepath)
	plt.imshow(image)
	plt.pause(0.05)
	print(count)
	count+=1
	print(imagepath)
	label = input('enter: ')
	row = [imagepath, label]
	writer.writerow(row)