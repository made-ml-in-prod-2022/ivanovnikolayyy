import numpy as np
import pandas as pd
import requests

TARGET_COL = "condition"


if __name__ == "__main__":
    data = pd.read_csv("data/train.csv", dtype=float)
    data = data.drop(columns=[TARGET_COL])
    request_features = list(data.columns)

    for i in range(3):
        request_data = [
            x.item() if isinstance(x, np.generic) else x for x in data.iloc[i].tolist()
        ]
        print(request_data)
        response = requests.get(
            "http://localhost:8000/predict/",
            json={"data": request_data, "features": request_features},
        )
        print(response.status_code)
        print(response.json())
