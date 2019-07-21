import re
import os
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import numpy as np
path = 'D:\Machine Learning\Avon_logs\logs\prod'
files = os.listdir(path)
dateOfOccurence=[]
timeStamp=[]
fileName=[]
errorDescription=[]
matches = []
reg = re.compile(".*Exception.*|Error in.*|.*TypeMismatchException.*|.*NumberFormatException.*|.*GOMAC.*")
lineNo=0
for name in files:
	print(name)
	f=open(path+'\\'+name, encoding='utf-8')
	for line in f:
		lineNo+=1
		#print(line)
		#print(lineNo)
		#print(line)
		matches = reg.search(line)
		if(matches):
			#print(line)
			reason=""
			mylist=re.sub(r"[\t\s]", " ", line).split(' ')
			r = re.compile("\d{8}|\d{2}:\d{2}:\d{2}|.+ctrl|\\.|.+validator|.+dao.+|.*util.*|impl.+|.+timer.+|.*GOMAC.*|.*TypeMismatchException.*")
			newlist = list(filter(r.match, mylist))
			#print(newlist)
			if(len(newlist)==3):
				#print(line)
				#print(mylist)
				index=0
				for i in mylist:
					index+=1
					ind=re.search(r"\bImp_Acct_Nr=.*", i)
					if(ind):
						#print(mylist)
						#print(index)
					#index=mylist.index("Imp_Acct_Nr=")
						for x in range (index,len(mylist)):
							reason+=" "+mylist[x]
						newlist.append(reason)
						dateOfOccurence.append(newlist[0])
						timeStamp.append(newlist[1])
						fileName.append(newlist[2])
						errorDescription.append(newlist[3])
			#if(len(newlist)!=2):	
				#print(newlist)
#print(dateOfOccurence,timeStamp,fileName,errorDescription)
df = pd.DataFrame({'dateOfOccurence': dateOfOccurence,'date_timeStamp': timeStamp,'errorfile': fileName,'errorDescription': errorDescription})
writer = pd.ExcelWriter('Prod_log_DataSheet.xlsx', engine='xlsxwriter')
df.to_excel(writer, sheet_name='Sheet1',index=False)
writer.save()
print('created')