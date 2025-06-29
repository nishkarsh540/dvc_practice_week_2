import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

df = pd.read_csv('data/iris.csv')
X = df.drop('species', axis=1)
y = df['species']

_, X_test, _, y_test = train_test_split(X, y, test_size=0.2)

model = joblib.load('model.joblib')
y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.savefig("metrics.png")

with open("report.md", "a") as f:
    f.write("## Confusion Matrix\n\n")
    f.write("![Confusion Matrix](metrics.png)\n")
