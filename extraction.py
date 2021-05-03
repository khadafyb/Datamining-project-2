
#import statements
import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split

#this should take in raceid and then return the year and circuit id
def raceid_to_circuit(x):
    with open('races.csv','r') as df3:
        read3=csv.reader(df3)
        next(read3)
        for row in read3:
            if x == int(row[0]):
                return int(row[3]), int(row[1]) #returns the circuit id, year
##
##
#circuit_info=['NA']*73 #73 different circuits below is what should circuit_info should be changed to
circuit_info = [ 0 for x in range(74)]
##for i in range(len(circuit_info)):
##    circuit_info[i]=[0]*68 #there should be 68 years
##        
###we are generating an array that has each raceid and their fastest time
##with open('lapTimes.csv','r') as df1:
##    read=csv.reader(df1)
##    next(read)
##    faster=[0]*1010
##    for i in range(0,1010):
##        faster[i]=[200000,0]
##    for row in read:
##        rid=row[0]
##        #1009 raceid's
##        for i in range(0,1010):
##            if i == int(rid):
##                if int(faster[i][0]) > int(row[5]):
##                    faster[i]=[int(row[5]), int(rid)] #this will get us the fastest time for laps and the rid
##    for i in range(0,len(faster)):
##        if faster[i][0] == 200000:
##            faster[i][0]=0
##
###times are recorded in milliseconds
##with open('pitStops.csv', 'r') as df2:
##    read2=csv.reader(df2)
##    next(read2)
##    needed_information=[0]*1010 #blank information
##    pittime=[1000000]*1010 
##    for row in read2:
##        for i in range(0,1010):
##            if i == int(row[0]):
##                if pittime[i]>int(row[6]):
##                    pittime[i]=int(row[6])
##    for i in range(0, len(pittime)):
##        if pittime[i] == 1000000:
##            pittime[i]=0
##    for i in range(0,1010): 
##        #needed_information[i]=[ raceid_to_circuit(int(faster[i][1])), int(faster[i][0]), pittime[i]] #this should return an array with circuit id, year, fastest lap and fastest pit
##        if raceid_to_circuit(int(faster[i][1])) == None:
##            pass #find a way to handle this if None meaning we don't get back a circuit and year
##        else:
##            info=raceid_to_circuit((faster[i][1])) #this will yield a string array that have the circuitid , year
##            circuit_info[(info[0])-1][(info[1])-1950]= int(faster[i][0]), pittime[i]
##
###year-1950 for index number
###circuit id - 1
##
###print(circuit_info) #the array is how daff wants it, it goes [circuit#[year#[fastest laptime, fastest pittime]]] add year number to remove the empty spaces
##for i in range(len(circuit_info)):
##    print(circuit_info[i])
##
###needed functions to gather information:
##
##
##
###[driverID[average lap, average pit, fastest lap, fastest pit, amount of laps, amount of pits]] over whole career
###[driverID[raceID[fastest lap, fastest pit]]
###
##racer_info = ['NULL']* 843 #i forget what gets inserted here
from scipy.stats import linregress
LAPMAX=200000
PITMAX=200000
LAPCOUNT=0
PITCOUNT=0
#avg_speed=[[0,0,PITMAX,LAPMAX,LAPCOUNT,PITCOUNT]]*843
avg_speed=[ [0,0,PITMAX,LAPMAX,LAPCOUNT,PITCOUNT] for x in range(844)]

with open('lapTimes.csv', 'r') as df4:
    read4=csv.reader(df4)
    next(read4)
    #generated needed arrays
    for row in read4:
        for i in range(1,844):
            if int(row[1])==i:
                avg_speed[i-1][0]=avg_speed[i-1][0]+int(row[5])
                avg_speed[i-1][4]=avg_speed[i-1][4]+1
                if int(row[5]) < avg_speed[i-1][3]:
                    avg_speed[i-1][3]=int(row[5])
    df4.close()
with open('pitStops.csv','r') as df5:
    read5=csv.reader(df5)
    next(read5)
    for row in read5:
        for i in range(1,844):
            if int(row[1])==i:
                avg_speed[i-1][1]=avg_speed[i-1][1]+int(row[6])
                avg_speed[i-1][5]=avg_speed[i-1][5]+1
                if avg_speed[i-1][2]>int(row[6]):
                    avg_speed[i-1][2]=int(row[6])
##for i in range(1,844):
##    avg_speed[i-1][0]=float(avg_speed[i-1][0]/avg_speed[i-1][4])
##    avg_speed[i-1][1]=float(avg_speed[i-1][1]/avg_speed[i-1][5])
print(avg_speed)
avgs=[ [0,0] for x in range(844)] #avg lap and pit times of driver
for i in range (1,844):
    if avg_speed[i-1][4] > 0:
        avgs[i-1][0]=float(avg_speed[i-1][0]/avg_speed[i-1][4])
    if avg_speed[i-1][5] > 0:
        avgs[i-1][1]=float(avg_speed[i-1][1]/avg_speed[i-1][5])
print(avgs)
##for i in avgs:
##fig=plt.figure()
##ax=fig.add_subplot(111)
x=[]
y=[]
for i in range(1,844):
    x=avgs[i-1][0]
    y=avgs[i-1][1]
    ##plt.subplot(111)
    plt.plot(i,x,'o',ms=1.0,color='red')
    plt.plot(i,y,'s',ms=1.0,color='blue')
##plt.legend(loc='upper center')
plt.title('average lap and average pit driver comparison')
plt.xlabel('driver id')
plt.ylabel('average times (ms)')
plt.show()

j=[0.0 for h in range(1,844)]
k=[0.0 for g in range(1,844)]
for i in range(1,844):
    j[i-1]=avgs[i-1][0]
    k[i-1]=avgs[i-1][1]
plt.plot(j,k,'o',ms=1.0,color='red')
plt.suptitle('average lap time vs average pit time')
plt.xlabel('average lap time (ms)')
plt.ylabel('average pit time (ms)')
z=np.polyfit(j,k,1)
e=np.poly1d(z)(j)
slope,intercept,r_value,p_value,std_err=linregress(j,k)
print("slope: %f, intercept: %f" %(slope,intercept))
print("R-squared: %f" % r_value**2)
plt.plot(j,e,'r',label='trend')
plt.title("y=%.6fx+%.6f"%(z[0],z[1]))
plt.show()

##KNN calculation test
##X_train, X_test, y_train, y_test=train_test_split(j,k,test_size=0.25,random_state=0)
##from sklearn.preprocessing import StandardScaler
##sc = StandardScaler()
##X_train=sc.fit_transform(X_train)
##X_test=sc.transform(X_test)
##from sklearn.neighbors import KNrighborsClassifier
##classifier=KNeighborsClassifier(n_neighbors=2)
##classifiier.fit(X_train, y_train)
##y_pred=classifier.predict(X_test)
##from sklearn.metrics import confusion_matrix
##cm=confusion_matrix(y_test,y_pred)
##print(cm)


##fig=plt.figure()
##ax=fig.add_subplot(111)   
####ax.scatter(range(0,844),x,s=10,c='b',marker="s",label='avg lap')
####ax.scatter(range(0,844),y,s=10,c='r',marker="o",label='avg pit')
####plt.legend(loc='upper left');
##ax.scatter(x,y,s=10,c='r',marker="o",label='avg pit')
##plt.title('average lap vs average pit')
##plt.xlabel('average lap')
##plt.ylabel('average pit')
##plt.show()
    
fast_time= [ [0,0] for x in range(844) ] #fastest lap and pit times for each driver
for i in range(1,844):
    fast_time[i-1][0]=avg_speed[i-1][3]
    fast_time[i-1][1]=avg_speed[i-1][2]
print(fast_time)

p=[0.0 for h in range(1,844)]
q=[0.0 for g in range(1,844)]
for i in range(1,844):
    p[i-1]=fast_time[i-1][0]
    q[i-1]=fast_time[i-1][1]
plt.plot(q,p,'o',ms=1.0,color='red')
plt.suptitle('fastest lap vs fastest pit')
plt.ylabel('max lap time (ms)')
plt.xlabel('max pit time (ms)')
z=np.polyfit(q,p,1)
e=np.poly1d(z)(q)
slope,intercept,r_value,p_value,std_err=linregress(q,p)
print("slope: %f, intercept: %f" %(slope,intercept))
print("R-squared: %f" % r_value**2)
plt.plot(q,e,'r',label='trend')
plt.title("y=%.6fx+%.6f"%(z[0],z[1]))
plt.show()


x=[0.0 for h in range(1,844)]
y=[0.0 for g in range(1,844)]
for i in range(1,844):
    x[i-1]=fast_time[i-1][0]
    y[i-1]=fast_time[i-1][1]
plt.plot(range(1,844),x,'o',ms=1.0,color='red',label='lap max')
plt.plot(range(1,844),y,'s',ms=1.0,color='blue',label='pit max')
plt.title('lap max and pit max driver comparison')
plt.legend(loc='upper left')
plt.xlabel('driverid')
plt.ylabel('max times (ms)')
plt.show()
# [driverID[year number[
#raceid=[[LAPMAX,PITMAX]]*1009
raceid=[ [LAPMAX,PITMAX] for x in range(1010) ]

with open('lapTimes.csv','r') as df6:
    read6=csv.reader(df6)
    next(read6)
    for row in read6:
        for i in range(1,1010):
            if int(row[0])== i:
                if int(row[5]) < raceid[i-1][0]:
                    raceid[i-1][0]=int(row[5])
                    
with open('pitStops.csv', 'r') as df7:
    read7=csv.reader(df7)
    next(read7)
    for row in read7:
        for i in range(1,1010):
            if int(row[0])==i:
                if int(row[6])< raceid[i-1][1]:
                    raceid[i-1][1]=int(row[6])
                    #print(int(row[6]))
#it won't set it to 0 and I am not sure why
for i in range(1,1010):
    if raceid[i-1][0] == LAPMAX:
        raceid[i-1][0]=0
    if raceid[i-1][1]==PITMAX:
        raceid[i-1][1]=0

amount=[ [0,0] for x in range(844)]
for i in range(1,844):
    amount[i-1][0]=avg_speed[i-1][4]
    amount[i-1][1]=avg_speed[i-1][5]
print(amount)

print(raceid)
p=[0.0 for h in range(1,844)]
q=[0.0 for g in range(1,844)]
for i in range(1,844):
    p[i-1]=raceid[i-1][0]
    q[i-1]=raceid[i-1][1]
plt.plot(q,p,'o',ms=1.0,color='red')
plt.suptitle('average lap vs average pit')
plt.ylabel('max lap time (ms)')
plt.xlabel('max pit time (ms)')
z=np.polyfit(j,k,1)
plt.title("y=%.6fx+%.6f"%(z[0],z[1]))
plt.show()


x=[0.0 for h in range(1,844)]
y=[0.0 for g in range(1,844)]
for i in range(1,844):
    x[i-1]=raceid[i-1][0]
    y[i-1]=raceid[i-1][1]
plt.plot(range(1,844),x,'o',ms=1.0,color='red',label='lap max')
plt.plot(range(1,844),y,'s',ms=1.0,color='blue',label='pit max')
plt.title('lap max and pit max driver comparison')
plt.legend(loc='upper left')
plt.xlabel('driverid')
plt.ylabel('max times (ms)')
plt.show()

