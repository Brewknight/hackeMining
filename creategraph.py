import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("./datasets/LSIaccuracy.csv", sep="\t")

plt.plot(data['Components'], data['Accuracy'])
plt.show()