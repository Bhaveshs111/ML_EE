
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt
                                                          
from prettytable import PrettyTable as PT

def csvfiles():
    list_csv=[]
    current_folder=os.getcwd()
    files=os.listdir(current_folder)
    for file in files:
        if file.split(".")[-1]=="csv":
            list_csv.append(file)
    return(list_csv)

def Table(x_test,y_test,y_predict):
        table=PT()
        table.field_names=["I1,I2,I3:","predicted [L1,L2,L3,G]:","Actual [L1,L2,L3,G]:"]

        ##convert array to list
        x_test_list=list(x_test)
        y_test_list=list(y_test)
        y_predict_list=list(y_predict)

        for i in range(len(x_test)):
            table.add_row([x_test_list[i],y_predict_list[i],y_test_list[i]])
        print(table)
        



def main(user_name):
    print(f"Hello {user_name}\n\nwelcome to Fault classification system:-")
    print("Press ENTER key to proceed")
    input()
    csv_files=csvfiles()                                    ##this is the list of csv files.
    print("choose a csv file for preparing model:-\n")
    for position,name in enumerate(csv_files):
        print(f"{position}---->{name}")

    model_csv=csv_files[int(input("Enter the index of the file:"))]


    df=pd.read_csv(model_csv)
  
    #Data preprocessing to improve the model accuracy.
    
    arrays=df.iloc[:,:].values   #numpy.ndarray
    newl=[]
    for row in arrays:
        if(all(row[-4:]==0) or any(row[1:4]>85) ):
            newl.append(row)
    arr=np.array(newl)
    df= pd.DataFrame(arr)

    #change the index of row and column for new csv file.
    
    model_input=df.iloc[:,[1,2,3]].values     #<class 'numpy.ndarray'> #change the index of row and column for new csv file.
    fault_output=df.iloc[:,[4,5,6,7]].values



    p=float(input("Enter percentage value for training data:\n") )       #percentage of total data
    test_data=1-(p/100)
    x_train,x_test,y_train,y_test=train_test_split(model_input,fault_output,test_size=test_data)    #in which args must be nD array

    ### Here x_train,x_test,y_train,y_test are 2D array.###


    print("\nModel creation in progression\n")

    from sklearn.neighbors import KNeighborsClassifier
    classifier_obj = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2) #where p=2 is equivalent to the Euclidean distance
    classifier_obj.fit(x_train,y_train)     #args are arrays in sklearn..

    print("KNN Model has been created.")                 # It means clusters have been fixed. (c1,c2,c3)


    
    print("would you like to test this model using testing data?")
    ans=int(input("if yes-->Enter 1:"))
    if ans==1:
        y_predict=classifier_obj.predict(x_test)
    
        print("press 1 to see predicted values of testing data ,else 0:-")

        if int(input())==1:
            Table(x_test,y_test,y_predict)

 
        print("\n")
        
        print("-------------------------------------------------------")
        true=0
        for i in range(len(y_predict)):
            x=(list(y_predict)[i])==(list(y_test)[i])
            if (all(x)):
                true+=1
        accuracy=(true/len(y_predict)*100)
        print(f"Accuracy = {round(accuracy,2)}%")
        
        print("-------------------------------------------------------")

    print("Now you can predict the fault using this model")

    while(True):
        try:
            print("\nEnter the 3-phase current values with [Ia,Ib,Ic] format:")
            x_newdata=eval(input())
            print(f" Fault type: {classifier_obj.predict(np.array([x_newdata]))}")
            x=int(input("\nEnter 0 for exit and 1 for continue:"))
            if x==0:
                break
        except:
            print("Error.")
            print("Enter the data in given format.")


if __name__=="__main__":
    name=input("Enter your name : ")
    main(name)
    input()
