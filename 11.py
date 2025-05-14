import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv(r"C:\Users\Yujing\Desktop\titanic_train.csv")
test = pd.read_csv(r"C:\Users\Yujing\Desktop\test.csv")

train['Age'] = train['Age'].fillna(train['Age'].median())
test['Age'] = test['Age'].fillna(test['Age'].median())

train['Sex'] = train['Sex'].map({'male': 0, 'female': 1})
test['Sex'] = test['Sex'].map({'male': 0, 'female': 1})

train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])
test['Embarked'] = test['Embarked'].fillna(test['Embarked'].mode()[0])

X_train = train[['Pclass', 'Sex', 'Age', 'Embarked', 'SibSp', 'Parch']]
X_test = test[['Pclass', 'Sex', 'Age', 'Embarked', 'SibSp', 'Parch']]

# 预测对象
Y_train = train['Survived']

attribute = {
    "Pclass": [1, 2, 3],
    "Sex": [0, 1],
    "Age": ["Childrens", "Teenagers", "Adults", "Elders"],
    "SibSp": ["no", "few", "many"],
    "Parch": ["no", "few", "many"],
    "Embarked": ['S', 'C', 'Q']
}

def build_dt(train_data, d_attr, attributes, nleaves_threshold=3, percent_threshold=0.95):
    def most_label_percent(data):
        ntrues = len(data[data[d_attr] == 1])
        nfalses = len(data[data[d_attr] == 0])
        return ntrues / len(data) if ntrues > nfalses else nfalses / len(data)

    T = []
    trace = []
    queue = [{"data": train_data, "available_attrs": list(attributes.keys()), "par": -1}]

    while len(queue) > 0:
        node = queue.pop(0)
        if (len(node["data"]) < nleaves_threshold) or (most_label_percent(node["data"]) > percent_threshold) or (
                len(node["available_attrs"]) <= 0):
            node["label"] = popular_label(node["data"], d_attr)
            if node["par"] >= 0:
                T[node["par"]]["children"][node["value"]] = len(T)
            T.append(node)
            eval_train = eval_tree(T, d_attr, train_data)
            trace.append(eval_train)
            print("Train:", eval_train)
            continue

        attrs_ig = {a: information_gain_ratio(node["data"], a, d_attr) for a in node["available_attrs"]}
        selected_attr = max(attrs_ig, key=attrs_ig.get)
        node["attr"] = selected_attr
        node["children"] = {}
        if node["par"] >= 0:
            T[node["par"]]["children"][node["value"]] = len(T)
        T.append(node)
        eval_train = eval_tree(T, d_attr, train_data)
        trace.append(eval_train)
        print("Train:", eval_train)

        value_data = split(node["data"], selected_attr)
        for v in value_data:
            available_attrs = node["available_attrs"].copy()
            available_attrs.remove(selected_attr)
            child_node = {"data": value_data[v], "value": v, "available_attrs": available_attrs, "par": len(T) - 1}
            queue.append(child_node)
    return T, trace

def print_tree(T):
    for i in range(len(T)):
        print(i, {f: T[i][f] for f in T[i] if f != "data"})
    print("#" * 30, "print_tree")

def popular_label(data, d_attr):
    ntrues = len(data[data[d_attr] == 1])
    nfalses = len(data[data[d_attr] == 0])
    return True if ntrues > nfalses else False

def eval_tree(T, d_attr, data):
    n = len(data)
    ncorrects = 0
    for _, e in data.iterrows():
        if predict(T, d_attr, e) == e[d_attr]:
            ncorrects += 1
    return ncorrects / n

def predict(T, d_attr, example):
    index = 0
    while True:
        if "label" in T[index]:
            return T[index]["label"]
        if "children" in T[index] and example[T[index]["attr"]] in T[index]["children"]:
            index = T[index]["children"][example[T[index]["attr"]]]
            continue
        else:
            return popular_label(T[index]["data"], d_attr)
    return False

def information_gain_ratio(data, attr, d_attr):
    def entropy(dis):
        return -np.sum([d * np.log2(d) if np.abs(d) > 1e-10 else 0 for d in dis])

    def distribution(data):
        ntrues = len(data[data[d_attr] == 1])
        return [ntrues / len(data), (len(data) - ntrues) / len(data)]

    before = entropy(distribution(data))
    after = 0.0
    n = len(data)
    for a in attribute[attr]:
        a_data = data[data[attr] == a]
        if len(a_data) > 0:
            after += entropy(distribution(a_data)) * (len(a_data) / n)
    split_info = entropy([len(data[data[attr] == a]) / n for a in attribute[attr]])
    return (before - after) / split_info if split_info != 0 else 0

def split(data, attr):
    result = {}
    for a in attribute[attr]:
        a_data = data[data[attr] == a]
        if len(a_data) > 0:  # 样本非空
            result[a] = a_data
    return result

def process_data(train_data, test_data):
    print('train_data:', '*' * 30)
    print(train_data.columns)
    train_data = train_data.drop(['PassengerId', 'Name', 'Ticket', 'Fare', 'Cabin'], axis=1)

    # Age
    mean_age = train_data['Age'].mean()
    train_data['Age'] = train_data['Age'].fillna(mean_age)
    train_data['Age'] = pd.cut(train_data['Age'], bins=[0, 12, 18, 65, np.inf],
                               labels=['Childrens', 'Teenagers', 'Adults', 'Elders'])

    # Sibsp
    train_data['SibSp'] = pd.cut(train_data['SibSp'], bins=[-1, 0, 3, np.inf], labels=['no', 'few', 'many'])

    # Parch
    train_data['Parch'] = pd.cut(train_data['Parch'], bins=[-1, 0, 3, np.inf], labels=['no', 'few', 'many'])

    # finally
    print(train_data.columns)
    for feature in train_data.columns:
        print(feature, ":", train_data[feature].unique())

    print('test_data:', '*' * 40)
    test_data = test_data.drop(['PassengerId', 'Name', 'Ticket', 'Fare', 'Cabin'], axis=1)

    # Age
    test_data['Age'] = test_data['Age'].fillna(mean_age)
    test_data['Age'] = pd.cut(test_data['Age'], bins=[0, 12, 18, 65, np.inf],
                              labels=['Childrens', 'Teenagers', 'Adults', 'Elders'])

    # Sibsp
    test_data['SibSp'] = pd.cut(test_data['SibSp'], bins=[-1, 0, 3, np.inf], labels=['no', 'few', 'many'])

    # Parch
    test_data['Parch'] = pd.cut(test_data['Parch'], bins=[-1, 0, 3, np.inf], labels=['no', 'few', 'many'])

    # finally
    print(test_data.columns)
    for feature in test_data.columns:
        print(feature, ":", train_data[feature].unique())

    train_data = train_data.to_dict(orient='records')
    test_data = test_data.to_dict(orient='records')
    return train_data, test_data

def main():
    train_data, test_data = process_data(train, test)
    train_df = pd.DataFrame(train_data)  # Convert to DataFrame
    print("Training size:", len(train_data))
    T, trace = build_dt(train_df, d_attr="Survived", attributes=attribute)
    print_tree(T)
    plt.plot(trace)
    plt.title("Training process (Accuracy on training data)")
    plt.xlabel("Tree size (number of nodes)")
    plt.ylabel("Accuracy")
    plt.show()

if __name__ == "__main__":
    main()
