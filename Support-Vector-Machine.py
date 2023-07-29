import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler,MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report

data = pd.read_csv("diabetes.csv", delimiter=";")
print(data.info())

result = data.describe()

# Visualization
plt.figure(figsize=(5,5))
sns.histplot(data["Outcome"])
plt.title("Label Distribution")
plt.show()

sns.heatmap(data.corr(), annot=True)
plt.show()

# split data into train and test
target = data["Outcome"]
x = data.drop(["Outcome"], axis=1)
y = target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# random_state: để cố định bộ train - test, dễ đánh giá so sánh 2 models
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
cls = SVC()
cls.fit(x_train, y_train)
y_predict = cls.predict(x_test)
for i, j in zip(y_test, y_predict):
    print("Actual: {} - Predict: {}".format(i, j))
cls.score(x_test, y_test)
print(classification_report(y_test, y_predict))

cm = np.array(confusion_matrix(y_test, y_predict, labels=[0,1]))
confusion = pd.DataFrame(cm, index=["Khoe","Benh"], columns=["Khoe","Benh"])
sns.heatmap(confusion, annot=True, fmt="g")
plt.savefig("confusion_matrix.png")
