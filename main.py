import sys,random
from tabulate import tabulate
classes={}
dataSize=None
trainingData=[]
testData=[]
analysisGrid=[]
class Classification:
    def __init__(self,name):
        self.name=name
        self.val=len(classes)
        if self.val>1:
            raise Exception("More than 2 classes!")
        classes[self.name]=self

    def __str__(self):
        return self.name
class Item:
    def __init__(self,data):
        global dataSize,classes
        self.vector=data.split(",")[:-1]
        self.vector=[float(i) for i in self.vector]
        if dataSize is None:
            dataSize=len(self.vector)
        elif dataSize != len(self.vector):
            raise Exception("Inconsistent Vector Size!")
        if data.split(",")[-1] not in classes:
            Classification(data.split(",")[-1])
        self.classification=classes[data.split(",")[-1]]
    def __str__(self):
        return ",".join([str(i) for i in self.vector]+[self.classification.name])
    def classIs(self,cls):
        if type(cls)==int:
            return cls == self.classification.val
        if type(cls)==str:
            return cls == self.classification.name
        return cls==self.classification

class Perceptron:
    def __init__(self,alpha):
        self.alpha=alpha
        self.theta=random.uniform(0, 7)
        self.weight=[random.uniform(0, 7) for i in range(0,dataSize)]
    def train(self,x):
        wx=0
        for i in range(dataSize):
            wx+=x.vector[i]*self.weight[i]
        y=int(wx>=self.theta)
        if y!=x.classification.val:
            wxs=[]
            for i in range(dataSize):
                wxs.append(self.weight[i]+(x.classification.val-y)*self.alpha*x.vector[i])
            self.weight=wxs
            self.theta=self.theta + (x.classification.val-y) * self.alpha * -1
    def test(self,x):
        wx=0
        for i in range(dataSize):
            wx+=x.vector[i]*self.weight[i]
        y = int(wx >= self.theta)
        return y

def readTo(file,l):
    with open(file) as file:
        for line in file:
            l.append(Item(line.rstrip()))
def test(percy,x):
    y=percy.test(x)
    analysisGrid[x.classification.val]+=1
    if y==x.classification.val:
        analysisGrid[x.classification.val+2] += 1
    return [x,classes[list(classes.keys())[y]]]
def printRep(percy):
    tab=[]
    a=b=0
    for i in range(len(classes)):
        a+=analysisGrid[i+2]
        b+=analysisGrid[i]
        tab.append([classes[list(classes.keys())[i]],"{}/{}".format(analysisGrid[i+2],analysisGrid[i]),"{}%".format(analysisGrid[i+2]/analysisGrid[i]*100)])
    tab.append(["total","{}/{}".format(a,b),"{}%".format(a/b*100)])
    print(tabulate(tab))
    tab=[["Theta",percy.theta],["Weight",percy.weight]]
    print(tabulate(tab))

if __name__ == "__main__":
    trainingFile = sys.argv[1]
    testFile = sys.argv[2]
    readTo(trainingFile, trainingData)
    readTo(testFile, testData)
    print(classes)
    percy=Perceptron(float(sys.argv[3]))
    for i in range(0,int(sys.argv[4])):
        print("-"*150)
        print("k= "+str(i))
        tab=[]
        analysisGrid=[0]*4
        random.shuffle(trainingData)
        print("size: "+str(len(trainingData)))
        [percy.train(x) for x in trainingData]
        for j in testData:
            r=test(percy,j)
            tab.append(r)
        print(tabulate(tab,["Sample","Prediction"]))
        printRep(percy)



