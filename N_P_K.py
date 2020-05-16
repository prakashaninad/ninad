import sys
print('Python: {}'.format(sys.version))
# scipy
import scipy
print('scipy: {}'.format(scipy.__version__))
# numpy
import numpy
print('numpy: {}'.format(numpy.__version__))
# matplotlib
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))
# pandas
import pandas
print('pandas: {}'.format(pandas.__version__))
# scikit-learn
import sklearn
print('sklearn: {}'.format(sklearn.__version__))# Load libraries
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import numpy as np
import statsmodels.formula.api as smf
from sklearn import metrics
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import matplotlib.pyplot as plt

df = pd.read_csv('npk.csv')
#df = df.drop("SOP", axis=1)

print(df.shape)
print(df.head(10))
print(df.describe())
#print(df.groupby('class').size())


features = ['N', 'P', 'K']
features1 = ['Urea']


x = df.loc[:, features].values
y = df.loc[:, features1].values


#x = StandardScaler().fit_transform(x)
#y = StandardScaler().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=1)
print(y_test)
print(X_test)


plt.plot(X_train,y_train)
plt.title('Graph')
plt.xlabel('NPK2D', color='#1C2833')
plt.ylabel('UREA', color='#1C2833')
plt.legend(loc='upper left')
plt.show()



pls2 = PLSRegression(n_components=2)
pls2.fit(X_train, y_train)
PLSRegression()
Y_pred = pls2.predict(X_test)

print(Y_pred)
print(np.sqrt(metrics.mean_squared_error(y_test, Y_pred)))
print(np.sqrt(metrics.r2_score(y_test, Y_pred)))



N = [70]
P = [30]
K = [10]
list_of_tuples = list(zip(N, P, K))
list_of_tuples
df = pd.DataFrame(list_of_tuples)


print(df)
Y1_pred = pls2.predict(df)
print(Y1_pred)

pickle.dump(pls2, open('model8.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model8.pkl','rb'))