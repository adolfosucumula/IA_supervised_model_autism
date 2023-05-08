
class NormalizerIAModel:

    def __init__(self) -> None:
        pass
    
        # step - features normalization
    def normalize_data(self, dataset, t=None):

        if(t == 1):
            d = dataset.copy()  # Min Max normalization
            for each_collum in range(0, dataset.shape[1]-1):
                max_collum = dataset.iloc[:, each_collum].max()
                min_collum = dataset.iloc[:, each_collum].min()
                d.iloc[:, each_collum] = (
                    d.iloc[:, each_collum] - min_collum)/(max_collum - min_collum)

        elif(t == 2):
            d = dataset.copy()  # Mean normalization
            for each_collum in range(0, dataset.shape[1]-1):
                max_collum = dataset.iloc[:, each_collum].max()
                min_collum = dataset.iloc[:, each_collum].min()
                mean_df = dataset.iloc[:, each_collum].mean()
                d.iloc[:, each_collum] = (
                    d.iloc[:, each_collum] - mean_df)/(max_collum - min_collum)

        else:
            d = dataset.copy()  # Standardization normalization
            for each_collum in range(0, dataset.shape[1]-1):
                mean_df = dataset.iloc[:, each_collum].mean()
                std = dataset.iloc[:, each_collum].std()
                d.iloc[:, each_collum] = (d.iloc[:, each_collum] - mean_df)/(std)

        return d
