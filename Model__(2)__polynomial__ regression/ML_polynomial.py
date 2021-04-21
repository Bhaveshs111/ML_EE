
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import  LinearRegression
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score as ModelAccuracy                       #to get performance factor(1-(ESS/DSS)) #sum of square of errors                                              #DSS-->(sum of yi - Y_bar)
from prettytable import PrettyTable as PT

from sklearn.preprocessing import PolynomialFeatures

def csvfiles():
    list_csv=[]
    current_folder=os.getcwd()
    files=os.listdir(current_folder)
    for f in files:
        if f.split(".")[-1]=="csv":
            list_csv.append(f)
    return(list_csv)

def Table(x_test,y_test,y_predict):
        table=PT()
        table.field_names=["Years:","predicted value of power:","Actual value of power:"]

        ##convert array to list
        x_test_list=list(x_test)
        y_test_list=list(y_test)
        y_predict_list=list(y_predict)

        for i in range(len(x_test)):
            table.add_row([x_test_list[i],y_predict_list[i],y_test_list[i]])
        print(table)
        
def trained_graph(x_train,x_train_p,y_train,myobj_regression):
    plt.scatter(x_train,y_train,color='green',label='training data(x_train,y_train)')
    plt.plot(x_train,myobj_regression.predict(x_train_p),color='blue',label='Best Fit')
    plt.title("Gross power vs Years")
    plt.xlabel('Years')
    plt.ylabel('Gross power(GWH)')
    plt.legend()
    plt.show()
    
def finalgraph(x_train,x_train_p,y_train,x_test,y_test,y_predict,myobj_regression):
    
    plt.scatter(x_train,y_train,color='green',label='training data(x_train,y_train)')
    plt.plot(x_train,myobj_regression.predict(x_train_p),color='blue',label='Best Fit')
    
    plt.scatter(x_test,y_test,color='red',label='known test data(x_test,y_test)')
    plt.scatter(x_test,y_predict,color='black',label='Predicted test data(x_test,y_pred)')

    plt.title("Gross power vs Years")
    plt.xlabel('Years')
    plt.ylabel('Gross power(GWH)')
    plt.legend()
    plt.show()


def main(user_name):
    print(f"Hello {user_name}\n\nwelcome to Power prediction system:-")
    print("Press ENTER key to proceed")
    input()
    csv_files=csvfiles()                                    ##this is the list of csv files.
    print("choose a csv file for preparing model:-\n")
    for position,name in enumerate(csv_files):
        print(f"{position}---->{name}")

    model_csv=csv_files[int(input("Enter the index of the file:"))]

    df=pd.read_csv(model_csv)

    x_1d=df.iloc[:,0].values     #<class 'numpy.ndarray'> x=year and y=Total power generated
    y_1d=df.iloc[:,1].values    

    x=[[i] for i in x_1d]        #[[1],[2],...] 
    y=[[i] for i in y_1d]

    Exp=np.array(x)              #[[1] [2] [3] ...] in 2D array
    salary=np.array(y)

    p=float(input("Enter percentage value for training data:\n") )       #percentage of total data
    test_data=1-(p/100)
    x_train, x_test,y_train,y_test = train_test_split(Exp,salary,test_size=test_data)    #in which args must be nD array

    ### Here x_train,x_test,y_train,y_test are 2D array.###

    poly=PolynomialFeatures(degree=3)        ####formula :y=w1x+w2x²+..+b   
    x_train_p=poly.fit_transform(x_train)   #####here x_train will be converted to the x_train_p
    x_test_p=poly.fit_transform(x_test)
    

    print("\nModel creation in progression\n")

    myobj_regression=LinearRegression()
    myobj_regression.fit(x_train_p,y_train)     #args are arrays in sklearn..
    print("Model has been created.")                 # It means (m=slop) and (c=intercept) has been fixed. (y=mx+c)
    ans=int(input("For graph--> 1,else 0:-"))
    if ans==1:
        trained_graph(x_train,x_train_p,y_train,myobj_regression)

    
    print("would you like to test this model using testing data?")
    ans=int(input("if yes-->Enter 1:"))
    if ans==1:
        y_predict=myobj_regression.predict(x_test_p)
    
        print("press 1 to see predicted values of testing data ,else 0:-")

        if int(input())==1:
            Table(x_test,y_test,y_predict)

        
        graph=int(input("if you want to see graph of the model then Enter:1 ,else Enter 0 :-\n"))
        if graph==1:
            finalgraph(x_train,x_train_p,y_train,x_test,y_test,y_predict,myobj_regression)

        pf= ModelAccuracy(y_test,y_predict)                                 ##pf=performance_fact
        if pf<0.10:
            print("Error ,Run again.")
        print("\n")
        print("-------------------------------------------------------")
        print(f"This model is {pf*100 } % accurate.\n")
        print("-------------------------------------------------------")

    print("Now you can predict Gross power of any year using this model")

    while(True):
        
        print("\nEnter the Year value yyyy format:")
        x_newdata=eval(input())
        xn=np.array([[x_newdata]])
        x_n=poly.fit_transform(xn)
        
        print(f"TotalPowewr(gwh): {myobj_regression.predict(x_n)}")
        x=int(input("\nEnter 0 for exit and 1 for continue:"))
        if x==0:
            break


if __name__=="__main__":
    name=input("Enter your name : ")
    main(name)
    input()
