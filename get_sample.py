import pandas as pd
import json

test = pd.read_csv("model_data/fashion-mnist_test.csv")

# read the first row
sample = test.drop("label", axis=1).iloc[55].tolist()

print(json.dumps({"pixels": sample}))

