from paretoset import paretoset
import pandas as pd

data = pd.read_csv("./input.csv")
mask = paretoset(data, sense=["max", "min"])
quality = data["quality"].tolist()
size = data["size"].tolist()
print("quality size pareto")
for i in range(len(mask)):
    print(str(quality[i])+" "+str(size[i])+" "+str(int(mask[i])))
