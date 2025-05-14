
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

attributes={
    "Outlook":["Sunny","Overcast","Rain"],
    "Temperature":["Hot","Mild","Cool"],
    "Humidity":["High","Normal"],
    "Wind":["Weak","Strong"]
}

examples=[
    {"Outlook":"Sunny","Temperature":"Hot","Humidity":"High","Wind":"Weak","PlayTennis":False},
    {"Outlook":"Sunny","Temperature":"Hot","Humidity":"High","Wind":"Strong","PlayTennis":False},
    {"Outlook":"Overcast","Temperature":"Hot","Humidity":"High","Wind":"Weak","PlayTennis":True},
    {"Outlook":"Rain","Temperature":"Mild","Humidity":"High","Wind":"Weak","PlayTennis":True},
    {"Outlook":"Rain","Temperature":"Cool","Humidity":"Normal","Wind":"Weak","PlayTennis":True},
    {"Outlook":"Rain","Temperature":"Cool","Humidity":"Normal","Wind":"Strong","PlayTennis":False},
    {"Outlook":"Overcast","Temperature":"Cool","Humidity":"Normal","Wind":"Strong","PlayTennis":True},
    {"Outlook":"Sunny","Temperature":"Mild","Humidity":"High","Wind":"Weak","PlayTennis":False},
    {"Outlook":"Sunny","Temperature":"Cool","Humidity":"Normal","Wind":"Weak","PlayTennis":True},
    {"Outlook":"Rain","Temperature":"Mild","Humidity":"Normal","Wind":"Weak","PlayTennis":True},
    {"Outlook":"Sunny","Temperature":"Mild","Humidity":"Normal","Wind":"Strong","PlayTennis":True},
    {"Outlook":"Overcast","Temperature":"Mild","Humidity":"High","Wind":"Strong","PlayTennis":True},
    {"Outlook":"Overcast","Temperature":"Hot","Humidity":"Normal","Wind":"Weak","PlayTennis":True},
    {"Outlook":"Rain","Temperature":"Mild","Humidity":"High","Wind":"Strong","PlayTennis":False}
]

def split(data:list,attr:str):
    result={}
    for a in attributes[attr]:
        a_data=[e for e in data if e[attr]==a]
        if len(a_data)>0:
            result[a]=a_data
    return result


def print_tree(T,print_data=False):
    for i in range(len(T)):
        print(i,{f:T[i][f] for f in T[i] if f!="data"})
    print('-'*60,"print tree over")

def popular_label(data_:list,d_attr:str):
    ntrues=len([e for e in data_ if e[d_attr]==1])
    nfalses=len(data_)-ntrues
    return 1 if ntrues>nfalses else 0
def build_dt(train_data,d_attr,nleaves_threshold=3,percent_threshold=0.95):
    def most_label_percent(data_:list):
        ntrues=len([i for i in data_ if i[d_attr]==1])
        nfalses=len(data_)-ntrues
        return ntrues/len(data_)if ntrues>nfalses else nfalses/len(data_)
    T=[]
    trace=[]
    queue=[{"data":train_data,"available_attrs":list(attributes.keys()),"par":-1}]
    while len(queue)>0:
        node=queue.pop(0)
        if (len(node["data"])<nleaves_threshold)or(most_label_percent(node["data"])>percent_threshold)or(len(node["available_attrs"])<=0):
            node["label"]=popular_label(node["data"],d_attr)
            if node["par"]>=0:
                T[node["par"]]["children"][node["value"]]=len(T)
            T.append(node)
            #print_tree(T)
            eval_train=eval(T,d_attr,train_data)
            trace.append(eval_train)
            print("Train:",eval_train)
            continue
        attrs_ig={a:information_gain_ratio(node["data"],a,d_attr)for a in node["available_attrs"]}
        selected_attr=max(attrs_ig,key=attrs_ig.get)
        node["attr"]=selected_attr
        node["children"]={}
        if node["par"]>=0:
            T[node["par"]]["children"][node["value"]]=len(T)
        T.append(node)
        eval_train=eval(T,d_attr,train_data)
        trace.append(eval_train)
        print("Train:",eval_train)
        value_data=split(node["data"],selected_attr)
        for i in value_data:
            available_attrs=node["available_attrs"].copy()
            available_attrs.remove(selected_attr)
            child_node={"data":value_data[i],"value":i,"available_attrs":available_attrs,"par":len(T)-1}
            queue.append(child_node)
    return T,trace

def eval(T,d_attr,data):
    n=len(data)
    ncorrects=0
    for i in data:
        if predict(T,d_attr,i)==i[d_attr]:
            ncorrects+=1
    return ncorrects/n
def predict(T,d_attr,example):
    assert len(T)>0,"empty tree."
    index=0
    while True:
        if "label"in T[index].keys():
            return T[index]["label"]
        if"children"in T[index].keys() and example[T[index]["attr"]] in T[index]["children"].keys():
            index=T[index]["children"][example[T[index]["attr"]]]
            continue
        else:

            return popular_label(T[index]["data"],d_attr)
    return 0

def information_gain_ratio(data:list,attr:str,d_attr:str):

    def entropy(dis_:list):

        return -np.sum([d*np.log2(d) if np.abs(d)>1e-10 else 0 for d in dis_])
    def distribution(data_):

        ntrues=len([e for e in data_ if e[d_attr]==1])
        return [ntrues/len(data_),(len(data_)-ntrues)/len(data_)]
    before=entropy(distribution(data))
    after=0.0
    n=len(data)
    for i in attributes[attr]:
        a_data=[j for j in data if j[attr]==i]
        if len(a_data)>0:
            after+=entropy(distribution(a_data))*(len(a_data)/n)
    return before-after

def main():
    #load data:
    train_data=examples
    print("Training size:",len(train_data))
    T,trace=build_dt(train_data,d_attr="PlayTennis")
    print_tree(T)
    plt.plot(trace)
    plt.title("Training process(Accuracy on training data)")
    plt.legend(["training data"],loc="lower right")
    plt.xlabel("Tree size")
    plt.ylabel("Accuracy")
    plt.show()

if __name__ =="__main__":
    main()