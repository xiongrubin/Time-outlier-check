import numpy as np
from scipy import stats
from demo.model import BaseModel


class Hotelling(BaseModel):
    def __init__(self):
        pass

    def detect(self, data, threshold):

        data = self.convert_to_nparray(data)

        # Set the threshold of abnormality
        abn_th = stats.chi2.interval(1-threshold, 1)[1]

        # Covert raw data into the degree of abnormality
        avg = np.average(data)
        var = np.var(data)
        data_abn = [(x - avg)**2 / var for x in data]

        # Abnormality determination
        result = []
        for (index, x) in enumerate(data_abn):
            if x > abn_th:
                result.append((index, data[index]))
        return result
