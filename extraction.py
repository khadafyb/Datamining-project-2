#import statements
import csv

#this should take in raceid and then return the year and circuit id
def raceid_to_circuit(x):
    with open('races.csv','r') as df3:
        read3=csv.reader(df3)
        next(read3)
        for row in read3:
            if x == int(row[0]):
                return int(row[3]), int(row[1]) #returns the circuit id, year


circuit_info=['NA']*73 #73 different circuits
for i in range(len(circuit_info)):
    circuit_info[i]=['NA']*68 #there should be 68 years
        
#we are generating an array that has each raceid and their fastest time
with open('lapTimes.csv','r') as df1:
    read=csv.reader(df1)
    next(read)
    faster=[0]*1010
    for i in range(0,1010):
        faster[i]=[200000,0]
    for row in read:
        rid=row[0]
        #1009 raceid's
        for i in range(0,1010):
            if i == int(rid):
                if int(faster[i][0]) > int(row[5]):
                    faster[i]=[int(row[5]), int(rid)] #this will get us the fastest time for laps and the rid
    for i in range(0,len(faster)):
        if faster[i][0] == 200000:
            faster[i][0]=0

#times are recorded in milliseconds
with open('pitStops.csv', 'r') as df2:
    read2=csv.reader(df2)
    next(read2)
    needed_information=[0]*1010 #blank information
    pittime=[1000000]*1010 
    for row in read2:
        for i in range(0,1010):
            if i == int(row[0]):
                if pittime[i]>int(row[6]):
                    pittime[i]=int(row[6])
    for i in range(0, len(pittime)):
        if pittime[i] == 1000000:
            pittime[i]=0
    for i in range(0,1010): 
        #needed_information[i]=[ raceid_to_circuit(int(faster[i][1])), int(faster[i][0]), pittime[i]] #this should return an array with circuit id, year, fastest lap and fastest pit
        if raceid_to_circuit(int(faster[i][1])) == None:
            pass #find a way to handle this if None meaning we don't get back a circuit and year
        else:
            info=raceid_to_circuit((faster[i][1])) #this will yield a string array that have the circuitid , year
            circuit_info[(info[0])-1][(info[1])-1950]= int(faster[i][0]), pittime[i]

#year-1950 for index number
#circuit id - 1

#print(circuit_info) #the array is how daff wants it, it goes [circuit#[year#[fastest laptime, fastest pittime]]] add year number to remove the empty spaces
for i in range(len(circuit_info)):
    print(circuit_info[i])

#needed functions to gather information:



#[driverID[average lap, average pit, fastest lap, fastest pit, amount of laps, amount of pits]] over whole career
#[driverID[raceID[fastest lap, fastest pit]]
#
racer_info = ['NULL']* 843 #i forget what gets inserted here
LAPMAX=200000
PITMAX=100000
LAPCOUNT=0
PITCOUNT=0
avg_speed=[[0,0,PITMAX,LAPMAX,LAPCOUNT,PITCOUNT]]*843

with open('lapTimes.csv', 'r') as df4:
    read4=csv.reader(df4)
    next(read4)
    #generated needed arrays
    for row in read4:
        for i in range(1,844):
            if int(row[1])==i:
                avg_speed[i-1][0]=avg_speed[i-1][0]+int(row[5])
                avg_speed[i-1][4]=avg_speed[i-1][4]+1
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
for i in range(1,844):
    avg_speed[i-1][0]=float(avg_speed[i-1][0]/avg_speed[i-1][4])
    avg_speed[i-1][1]=float(avg_speed[i-1][1]/avg_speed[i-1][5])
print(avg_speed)


# [driverID[year number[
raceid=[[LAPMAX,PITMAX]]*1009

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

for i in range(1,1010):
    if raceid[i-1][0] == LAPMAX:
        raceid[i-1][0]=0
    if raceid[i-1][1]==PITMAX:
        raceid[i-1][1]=0

print(raceid)

