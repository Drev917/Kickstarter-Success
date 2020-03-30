# Imports

import pandas as pd
import numpy as np


# create ks dataframe
ks = pd.DataFrame(pd.read_csv("ks.csv"))

# correlation of columns
ks_Cor = ks.corr(method='pearson')
