
import pandas as pd
import pathlib as path

class ReadDatabase:
    
    def __init__(self) -> None:
        return None
    
    def getDatabase(self, file_name = ''):
       
        if(file_name != ''):
            return path.Path(file_name)
        else:
            return path.Path("winequality-white.csv")
    
pass