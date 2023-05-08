

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process.kernels import RBF

from sklearn.metrics import classification_report # Get reports
from sklearn.feature_selection  import SelectKBest # The way to perfome the feature selection
from sklearn.feature_selection import chi2, mutual_info_classif, f_regression, f_classif, mutual_info_regression, SelectPercentile, SelectFdr, SelectFpr, SelectFwe, GenericUnivariateSelect

class IAClassifiersModel:
    
    def __init__(self) -> None:
        pass
    
    def classifiers():
        return [
            SVC(gamma = 'auto'),
            GaussianProcessClassifier(1.0 * RBF(1.0)),
            LinearSVC(),
            SGDClassifier(),
            KNeighborsClassifier(),
            LogisticRegression(solver='lbfgs'),
            LogisticRegressionCV(cv = 3),
            BaggingClassifier(),
            ExtraTreesClassifier(n_estimators=300),
            RandomForestClassifier(max_depth=5, n_estimators=300, max_features=1),
            GaussianNB(),
            DecisionTreeClassifier(max_depth=5),
            MLPClassifier(alpha=1, max_iter=1000),
            AdaBoostClassifier(),
            LinearDiscriminantAnalysis(),
            QuadraticDiscriminantAnalysis(),
            OneVsRestClassifier(LinearSVC(random_state=0)),
            GradientBoostingClassifier(),
            SGDClassifier(),
        ]