import os
import numpy as np
from matplotlib import pyplot as plt

def analyse_data():
    path1 = "E:\\download\\Challenge_dataset\\Challenge_dataset\\train"       
    # path1 = "./Challenge_dataset/train"       
    files1 = os.listdir(path1)           
    num1 = len(files1)                   
    num2 = []                            
    for i in range(num1):                
        path2 = path1 +'//' +files1[i]   
        files2 = os.listdir(path2)       
        num2.append(len(files2))         
    
    class_names = np.array([i for i in range(21)])
    num2 = np.array(num2)
    avg = np.mean(num2)
    cov = np.cov(num2)
    print(avg, cov)

    fig, ax = plt.subplots()
    ax.scatter(class_names, num2)
    ax.hlines(y=avg, xmin=0, xmax=20, linewidth=2, linestyles='-', color='r', label="Average")
    ax.plot(class_names, num2, color='y', label="Number of data from each class")
    ax.set_title("Data distribution with mean " + str(round(avg,2)))
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of images")
    plt.show() 




if __name__ == "__main__":
    analyse_data()
        