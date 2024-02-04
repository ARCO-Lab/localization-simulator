import pandas as pd
import os
from datetime import datetime


class Result():

    def __init__(self, *args) -> None:
        self.algs = args
        self.df = pd.DataFrame(index=self.algs ,columns=['Information', 'MSE', 'Run Time'])
    
    def add(self,algName,result):
        self.df.loc[algName] = result

    def toCSV(self, name=None):
        path = f"{os.getcwd()}/results"
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Folder 'results' created successfully.")
        name = f"{name}.csv" if name else f"{datetime.now().strftime('%B_%d_%Y_%I_%M_%S')}.csv"
        self.df.to_csv(f"{path}/{name}",index=False)


if __name__ == "__main__":
    r = Result("Greedy","Evolutionary","Random")
    r.add("Greedy",[12,13,14])
    print(r.df)
    r.toCSV("test")