import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import sys
import numpy as np
import logging
import pandas as pd
log = logging.getLogger(__name__)
logging.basicConfig( format='%(asctime)s %(levelname)-8s %(message)s', level= logging.DEBUG)
print("INIZIATO")

N_ROWS = 10000

test_data = pd.read_csv("data/mushroom_overload.csv",nrows=N_ROWS)

pipeline= None

try:
    with open('pipeline.pkl', 'rb') as f:
        pipeline = pickle.load(f)
        log.info("Model OK")

except Exception as e:
    print(e)
    log.error("CANNOT OPEN FILE")
    sys.exit(0)

for i in range(0,100):
    randix = np.random.randint(0,N_ROWS-1)

    test_row = pd.DataFrame([test_data.iloc[randix,:]])
    print(f"------- {pipeline.predict(test_row)}------{test_row["class"]}")
    



