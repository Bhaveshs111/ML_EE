#KNN LC
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
                                                          #DSS-->(sum of yi - Y_bar)
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
        table.field_names=["voltage(p.u),LSF:","predicted value of cluster(D.G.):","Actual value of cluster(D.G.):"]

        ##convert array to list
        x_test_list=list(x_test)
        y_test_list=list(y_test)
        y_predict_list=list(y_predict)

        for i in range(len(x_test)):
            table.add_row([x_test_list[i],y_predict_list[i],y_test_list[i]])
        print(table)
        



def main(user_name):
    print(f"Hello {user_name}\n\nwelcome to Load classification system:-")
    print("Press ENTER key to proceed")
    input()
    csv_files=csvfiles()                                    ##this is the list of csv files.
    print("choose a csv file for preparing model:-\n")
    for position,name in enumerate(csv_files):
        print(f"{position}---->{name}")

    model_csv=csv_files[int(input("Enter the index of the file:"))]

    df=pd.read_csv(model_csv)

#change the index of row and column for new csv file.
    
    model_input=df.iloc[:,[1,2]].values     #<class 'numpy.ndarray'> 
    clusters=df.iloc[:,-1].values



    p=float(input("Enter percentage value for training data:\n") )                              #percentage of total data
    test_data=1-(p/100)
    x_train,x_test,y_train,y_test=train_test_split(model_input,clusters,test_size=test_data)    #in which args must be nD array

    print("\nModel creation in progression\n")



    from sklearn.neighbors import KNeighborsClassifier
    classifier_obj = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
    #where p=2 is equivalent to the Euclidean distance
    #metric='minkowski': This is the default parameter and it decides the distance between the points.
    
    classifier_obj.fit(x_train,y_train)    

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
            if((y_predict)[i]==(y_test)[i]):
                true+=1
        accuracy=(true/len(y_predict)*100)
        print(f"Accuracy = {round(accuracy,2)}%")
        
        print("-------------------------------------------------------")
        

        """
        print("-------------------------------------------------------")
        
        # Making the Confusion Matrix
    
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test,y_predict)
        print(f"Confusion Matrix:")
        print(cm)
        print("-------------------------------------------------------")
        """

    print("Now you can predict the cluster(D.G.) using this model:")

    while(True):
        try:
            print("\nEnter the data with [voltage(p.u),LSF] format:")
            x_newdata=eval(input())
            print(f"cluster(D.G.) number: {classifier_obj.predict(np.array([x_newdata]))}")
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
