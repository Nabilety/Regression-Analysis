import pandas as pd
import numpy as np

columns = ['Overall Qual', 'Overall Cond', 'Gr Liv Area',
           'Central Air', 'Total Bsmt SF', 'SalePrice']

try:
    df = pd.read_csv('AmesHousing.txt', sep='\t', usecols=columns)
except FileNotFoundError:
    print("File not found! Please make sure 'AmesHousing.txt' exists")

print(df.head())
print(df.shape) # as expected we have a DataFrame consisting of 2930 data rows, and 6 feature columns

# convert 'Central Air' variable from string to integer encoding using .map
df['Central Air'] = df['Central Air'].map({'N': 0, 'Y': 1})

# check if there are any columns with missing values
print(df.isnull().sum()) # 'Total Bsmt SF' contains one missing value

# Since this is a relatively large dataset, easiest way to deal with this missing feature value
# will be to remove the corresponding example from the dataset (check chapter 4 for alternativ methods)
df = df.dropna(axis=0)
print(df.isnull().sum())



# ## Visualizing the important characteristics of a dataset
# - scatterplots & histograms
import matplotlib.pyplot as plt
from mlxtend.plotting import scatterplotmatrix

scatterplotmatrix(df.values, figsize=(12, 10),
                  names=df.columns, alpha=0.5)
plt.tight_layout()
plt.show()


# Create correlation matrix to quantify and summarize linear relationship between variables
# Correlation matrix are closely related to covariance matrix. As iti s a rescaled version.
# In fact correlation matrix is identical to a covariance matrix computed from standardized features

from mlxtend.plotting import heatmap

cm = np.corrcoef(df.values.T)
hm = heatmap(cm, row_names=df.columns, column_names=df.columns)
plt.tight_layout()
plt.show() # note we regard SalePrice as our target variable. So we check for high correlations with this

# we can see that Gr Liv Area has 0.71 which we will use to introduce the concepts of a simple linear regression model

# # Implementing an ordinary least squares linear regression model

# ...

# ## Solving regression for regression parameters with gradient descent

# In Adaline we used (MSE) as loss function, here we will use OLS which is identical.
# However, in OLS we have no threshold function, so we obtain continuous target values instead of class label 0 and 1.


class LinearRegressionGD:
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.array([0.])
        self.losses_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
            self.b_ += self.eta * 2.0 * errors.mean()
            loss = (errors**2).mean()
            self.losses_.append(loss)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        return self.net_input(X)


# Use Gr Living Area as explanatory variable to train a model that predicts SalePrice.
# Furthermore standardize variables for better convergence of the GD algorithm
from sklearn.preprocessing import StandardScaler
X = df[['Gr Liv Area']].values
y = df['SalePrice'].values
sc_x = StandardScaler()
sc_y = StandardScaler()
X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()
lr = LinearRegressionGD(eta=0.1)
lr.fit(X_std, y_std)

# Note workaround regarding y_std using np.newaxis and flatten.
# Most preprocessing classes in scikit-learn expect data to be stored in two-dimensional arrays
# In previous examples using np.newaxis in y[:, np.newaxis] added a new dimension to the array.
# Then after StandardScaler, returned the scaled variable, we converted it back to the original one-dimensional
# array representation using flatten() for convenience.

# Plot loss as function of the number of epochs (complete iterations) over the training dataset when using optimization
# algorithms such as GD, to check that the algorithm converged to a loss minimum (here global loss minimum)
plt.plot(range(1, lr.n_iter+1), lr.losses_)
plt.ylabel('MSE')
plt.xlabel('Epoch')
plt.show()


# visualize how well the linear regression line fits the training data.
# to do that we define a simple helper function which will plot a scatterplot of the training examples and add the line
def lin_regplot(X, y, model):
    plt.scatter(X, y, c='steelblue', edgecolor='white', s=70)
    plt.plot(X, model.predict(X), color='black', lw=2)

lin_regplot(X_std, y_std, lr)
plt.xlabel('Living area above ground (standardized)')
plt.ylabel('Sale price (standardized)')
plt.show()

# Although the observations makes sense, the data also shows that living area size does not explain house prices very well
# in many cases. Later we will find out how to quantify the performance of a regression model.

#Interestingly we observe several outliers, for example three data points corresponding to a standardized living area greater than 6
# We will also deal with this later

# In certain application we want to report predicted outcome variables on their original scale.
# To scale the predicted price back onto the price in US dollars scale, we apply the inverse_transform from StandardScaler
feature_std = sc_x.transform(np.array([[2500]]))
target_std = lr.predict(feature_std)
target_reverted = sc_y.inverse_transform(target_std.reshape(-1, 1))
print(f'Sales price: ${target_reverted.flatten()[0]:.2f}')
# Here we used previously trained linear regression model to predict the price of a house with an aboveground living area of 2500 square feet.
# According to our model, such house will be worth $292,507.07

#As a side note, it is also worth mentioning that we technically don’t have to update the intercept parameter
# (for instance, the bias unit, b) if we are working with standardized variables, since the y axis
# intercept is always 0 in those cases. We can quickly confirm this by printing the model parameters:
print(f'Slope: {lr.w_[0]:.3f}')
print(f'Intercept: {lr.b_[0]:.3f}')



# ## Estimating the coefficient of a regression model via scikit-learn
from sklearn.linear_model import LinearRegression
slr = LinearRegression()
slr.fit(X, y)
y_pred = slr.predict(X)
print(f'Slope: {slr.coef_[0]:.3f}')
print(f'Intercept: {slr.intercept_:.3f}')
# as we can see, scikit-learns LinearRgression fitted with unstandardized Gr Liv Area and SalePrice variables
# yielded different model coefficients isnce the features have not been standardized.
# However if we compare it to our GD implementation by plotting SalePrice against Gr Liv Area, we can qualitatively see it fits the data similarly well

lin_regplot(X, y, slr)
plt.xlabel('Living area above ground in square feet')
plt.ylabel('Sale price in U.S. dollars')
plt.tight_layout()
plt.show()


# **Normal Equations** alternative:



# adding a column vector of "ones"
Xb = np.hstack((np.ones((X.shape[0], 1)), X))
w = np.zeros(X.shape[1])
z = np.linalg.inv(np.dot(Xb.T, Xb))
w = np.dot(z, np.dot(Xb.T, y))

print(f'Slope: {w[1]:.3f}')
print(f'Intercept: {w[0]:.3f}')



# # Fitting a robust regression model using RANSAC
from sklearn.linear_model import RANSACRegressor
ransac = RANSACRegressor(LinearRegression(),
                         max_trials=100, # default value
                         min_samples=0.95,
                         residual_threshold=None, # default value median absolute deviation (MAD) to select the inlier threshold
                         random_state=123)
ransac.fit(X, y)

# Obtain inliers and outliers from the fitted RANSAC linear regression model and plot together with the linear fit
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
line_X = np.arange(3, 10, 1)
line_y_ransac = ransac.predict(line_X[:, np.newaxis])
plt.scatter(X[inlier_mask], y[inlier_mask],
            c='steelblue', edgecolors='white',
            marker='o', label='Inliers')
plt.scatter(X[outlier_mask], y[outlier_mask],
            c='limegreen', edgecolors='white',
            marker='s', label='Outliers')
plt.plot(line_X, line_y_ransac, color='black', lw=2)
plt.xlabel('Living area above ground in square feet')
plt.ylabel('Sale  price in U.S. dollars')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# Print slope and intercept of the model:
print(f'Slope: {ransac.estimator_.coef_[0]:.3f}')
print(f'Intercept: {ransac.estimator_.intercept_:.3f}')
# the linear regression line is slightly different from the fit that we obtained previous without RANSAC

# Compute MAD for the dataset:
def mean_absolute_deviation(data):
    return np.mean(np.abs(data - np.mean(data)))
print(mean_absolute_deviation(y))
# thus if we want to identify fewer data points as outliers, we can choose a residual_threshold value greater than MAD
# using RANSAC we reduced the potential effect of the outliers in this dataset, but we still don't know if the approach
# will have a positive effect on the predictive performance on unseen data or not
# For this we need to approaches to evaluate a regression model.



# # Evaluating the performance of linear regression models


# We will now use all five features in the dataset and train a multiple regression model instead of a simple one
from sklearn.model_selection import train_test_split
target = 'SalePrice'
features = df.columns[df.columns != target]
X = df[features].values
y = df[target].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
slr = LinearRegression()
slr.fit(X_train, y_train)
y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)

# Since our model uses multiple explanatory variables (features) now, we can't visualize a linear regression line
# (hyperplace to be exact) in a 2-dimensional plot, however we can plot the residuals (difference or vertical distance between actual and predicted values)
# versus the predicted values to diagnose our regression model. Residual plots are commonly used graphical tool
# for diagnosing regression models. They help detect nonlinearity and outlier, and check whether errors are randomly distributed.

# plot residual plot where we subtract true target variables from predicted response
x_max = np.max([np.max(y_train_pred), np.max(y_test_pred)])
x_min = np.min([np.min(y_train_pred), np.min(y_test_pred)])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3), sharey=True)

ax1.scatter(y_test_pred, y_test_pred - y_test,
            c='limegreen', marker='s',
            edgecolor='white',
            label='Test data')
ax2.scatter(y_train_pred, y_train_pred - y_train,
            c='steelblue', marker='o', edgecolor='white',
            label='Training data')
ax1.set_ylabel('Residuals')

for ax in (ax1, ax2):
    ax.set_xlabel('Predicted values')
    ax.legend(loc='upper left')
    ax.hlines(y=0, xmin=x_min-100, xmax=x_max+100,
              color='black', lw=2)
plt.tight_layout()
plt.show()


# Compute MSE of our training and test predictions
from sklearn.metrics import mean_squared_error
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)

print(f'MSE train: {mse_train:.2f}')
print(f'MSE test: {mse_test:.2f}')
# as we ccan see the MSE on the training dataset is larger than the test set.
# which indicates that our model slightly overfits the training data

# intuitively we want to show the error on the original unit scale (dollar, and not dollar-square).
# so we choose to compute the square root of MSE called root mean squared error, or mean absolute error (MAE)
from sklearn.metrics import mean_absolute_error
mae_train = mean_absolute_error(y_train, y_train_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)
print(f'MAE train: {mae_train:.2f}')
print(f'MAE test: {mae_test:.2f}')
# based on this we can say the model makes an error of aprrox $25,000 on average

# Using MAE and MSE for comparing models, we need to be aware that these are unbounded
# in contrast to the classification accuracy. Meaning, the interpretations of the MAE and MSE depend on the dataset
# and feature scaling. I.e. if the sale prices were presented as multiples of 1000 (with K suffix), the same model would
# yield a lower MAE compared to a model that worked with unscaled features:
# $500K − 550K| < |$500,000 − 550,000|

# therefore it may be useful to report the coefficient of determination (R^2), which is can be understood
# as the standardized version of MSE, for better interpretability of the model's performance.
# In other words, R^2 is the fraction of response variance that is captured by the model

# R^2 = 1 - SSE/SST
from sklearn.metrics import r2_score
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
print(f'R^2 train: {train_r2:.2f}, {test_r2:.2f}')

# For the training dataset R^2 is bounded between 0 and 1, but can become negative for test dataset.
# Negative R^2 can be mean the regression models fits the data worse than a horizontal line representing the sample mean
# this often happens due to extreme overfitting or forgetting to scale the test set same way as we scale the training set
# R^2 = 1 means the model fits data perfectly, with corresponding MSE = 0.

# We obtained a R^2 of our model to 0.77, not great but also not bad given a small feature space.
# However R^2 on the test dataset is only slightly smaller 0.75, indicating our model is only overfitting slightly



# # Using regularized methods for regression

# Ridge regression model:
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=1.0) # note regularization strength regulated with alpha parameter, which is similar to λ

# LASSO regressor:
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=1.0) # similar parameter as previous

#ElasticNet, compromise between ridge and LASSO, allowing us to vary the L1 and L2 ratio:
from sklearn.linear_model import ElasticNet
elanet = ElasticNet(alpha=1.0, l1_ratio=0.5)
# for example, if we set l1_ration to 1.0, the ElasticNet regressor would be equal to LASSO regression.


# # Turning a linear regression model into a curve - polynomial regression
# In cases where linear relationship between explanatory and response variables are violated

# Adding polynomial terms using scikit-learn

# add a quadratic term (d = 2) to a simple regression problem with one explanatory variable.
# then compare the polynomial to the linear fit
from sklearn.preprocessing import PolynomialFeatures
X = np.array([258.0, 270.0, 294.0, 320.0, 342.0,
              368.0, 396.0, 446.0, 480.0, 586.0])[:, np.newaxis]
y = np.array([236.4, 234.4, 252.8, 298.6, 314.2,
              342.2, 360.8, 368.0, 391.2, 390.8])
lr = LinearRegression()
pr = LinearRegression()
quadratic = PolynomialFeatures(degree=2)
X_quad = quadratic.fit_transform(X)

# Fit a simple linear regression model for comparison
lr.fit(X, y)
X_fit = np.arange(250, 600, 10)[:, np.newaxis]
y_lin_fit = lr.predict(X_fit)

# fit multiple regression model on the transformed features for polynomial regression
pr.fit(X_quad, y)
y_quad_fit = pr.predict(quadratic.fit_transform(X_fit))

# Plot the results
plt.scatter(X, y, label='Training points')
plt.plot(X_fit, y_lin_fit,
         label='Linear fit', linestyle='--')
plt.plot(X_fit, y_quad_fit,
         label='Quadratic fit')
plt.xlabel('Explanatory variable')
plt.ylabel('Predicted or known target values')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# hence, the polynomial fit captures the relationshp between the response and explanatory variables much better
# than the linear fit.

# Next compute MSE and R^2 evaluation metrics
y_lin_pred = lr.predict(X)
y_quad_fit = pr.predict(X_quad)
mse_lin = mean_squared_error(y, y_lin_pred)
mse_quad = mean_squared_error(y, y_quad_fit)
print(f'Training MSE linear: {mse_lin:.3f}, quadratic: {mse_quad:.3f}')

lin_r2 = r2_score(y, y_lin_pred)
quad_r2 = r2_score(y, y_quad_fit)
print(f'Training R^2 linear: {lin_r2:.3f}, quadratic: {quad_r2:.3f}')
# hence, MSE decreased from 570 (linear fit) to 61 (quadratic fit); also the coefficient of determination reflects a closer fit
# of the quadratic model (R^2 = 0.982) as opposed to the linear fit (R^2 = 0.832)




# ## Modeling nonlinear relationships in the Ames Housing dataset

# Model relationship between sale prices and the living area above ground using quadratic and third-degree (cubic)
# polynomials and compare that to a linear fit:

# Start by removing the three outliers with a living area greater than 4000 square feet, as observed in previous figures
# so they don't skew our regression fits:
X = df[['Gr Liv Area']].values
y = df['SalePrice'].values
X = X[(df['Gr Liv Area'] < 4000)]
y = y[(df['Gr Liv Area'] < 4000)]

# Fit the regression models:
regr = LinearRegression()

# create quadratic and cubic features
quadratic = PolynomialFeatures(degree=2)
cubic = PolynomialFeatures(degree=3)
X_quad = quadratic.fit_transform(X)
X_cubic = cubic.fit_transform(X)

# fit the features
X_fit = np.arange(X.min()-1, X.max()+2, 1)[:, np.newaxis]
regr = regr.fit(X, y)
y_lin_fit = regr.predict(X_fit)
linear_r2 = r2_score(y, regr.predict(X))
regr = regr.fit(X_quad, y)
y_quad_fit = regr.predict(quadratic.fit_transform(X_fit))
quadratic_r2 = r2_score(y, regr.predict(X_quad))
regr = regr.fit(X_cubic, y)
y_cubic_fit = regr.predict(cubic.fit_transform(X_fit))
cubic_r2 = r2_score(y, regr.predict(X_cubic))

# plot results
plt.scatter(X, y, label='Training points', color='lightgray')
plt.plot(X_fit, y_lin_fit, label=f'Linear (d=1), $R^2$={linear_r2:.2f}',
          color='blue',
          lw=2,
          linestyle=':')
plt.plot(X_fit, y_quad_fit, label=f'Quadratic (d=2), $R^2$={quadratic_r2:.2f}',
          color='red',
          lw=2,
          linestyle='-')
plt.plot(X_fit, y_cubic_fit, label=f'Cubic (d=3), $R^2$={cubic_r2:.2f}',
          color='green',
          lw=2,
          linestyle='--')
plt.xlabel('Living area above ground in square feet')
plt.ylabel('Sale price in U.S. dollars')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# As shown, using quadratic or cubic feature does not really have an effect. That's because the relationship between
# the two variables appear to be linear. So, let's take a look at another feautre. Namely 'Overall Qual'

X = df[['Overall Qual']].values
y = df['SalePrice'].values

regr = LinearRegression()

# create quadratic features
quadratic = PolynomialFeatures(degree=2)
cubic = PolynomialFeatures(degree=3)
X_quad = quadratic.fit_transform(X)
X_cubic = cubic.fit_transform(X)

# fit features
X_fit = np.arange(X.min()-1, X.max()+2, 1)[:, np.newaxis]

regr = regr.fit(X, y)
y_lin_fit = regr.predict(X_fit)
linear_r2 = r2_score(y, regr.predict(X))

regr = regr.fit(X_quad, y)
y_quad_fit = regr.predict(quadratic.fit_transform(X_fit))
quadratic_r2 = r2_score(y, regr.predict(X_quad))

regr = regr.fit(X_cubic, y)
y_cubic_fit = regr.predict(cubic.fit_transform(X_fit))
cubic_r2 = r2_score(y, regr.predict(X_cubic))


# plot results
plt.scatter(X, y, label='Training points', color='lightgray')

plt.plot(X_fit, y_lin_fit,
         label=f'Linear (d=1), $R^2$={linear_r2:.2f}',
         color='blue',
         lw=2,
         linestyle=':')

plt.plot(X_fit, y_quad_fit,
         label=f'Quadratic (d=2), $R^2$={quadratic_r2:.2f}',
         color='red',
         lw=2,
         linestyle='-')

plt.plot(X_fit, y_cubic_fit,
         label=f'Cubic (d=3), $R^2$={cubic_r2:.2f}',
         color='green',
         lw=2,
         linestyle='--')


plt.xlabel('Overall quality of the house')
plt.ylabel('Sale price in U.S. dollars')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# As shown, quadric and cubic fits capture the relationship between sale prices and overal quality of the house better than
# the linear fit. However, be aware that adding more and more polynomial features increases complexity of a model
# and thus increase the risk of overfitting. So in practice it is always recommended to evaluate the performance of the
# models on separate test dataset to estimate the generalization performance


# # Dealing with nonlinear relationships using random forests

# ...

# ## Decision tree regression
# Gini/Entropy impurity were the measures we used for classificaiton. But for decision tree regression we will need
# an impurity metric suitable for continuous variables, so we define impurity measure of node t, as MSE
# Often referred to as within-node variance in context of decision tree regression, hence why splitting criterion also better known as variance reduction

from sklearn.tree import DecisionTreeRegressor
X = df[['Gr Liv Area']].values
#X = df[['Overall Qual']].values
y = df['SalePrice'].values

tree = DecisionTreeRegressor(max_depth=3)
tree.fit(X, y)
sort_idx =  X.flatten().argsort()
lin_regplot(X[sort_idx], y[sort_idx], tree)
plt.xlabel('Living area above ground in square feet')
plt.ylabel('Sale price in U.S. dollars')
plt.tight_layout()
plt.show()

# A limitation of this model is that it does not capture continuity and differentiability of desired prediction
# Additionally we need to be careful about choosing an appropriate value for the depth of the tree to not overfit/underfit



# ## Random forest regression
# once again the only difference between RF classification and regression is we use MSE criterion to grow individual trees
# and the prdicted target variable is calculated as the average prediction across all decision trees

# fit a random forest regression model on 70% of examples and evaluate its performance on the remaining 30%
target = 'SalePrice'
features = df.columns[df.columns != target]

X = df[features].values
y = df[target].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=123)





from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor(n_estimators=1000,
                               criterion='squared_error',
                               random_state=1,
                               n_jobs=-1)
forest.fit(X_train, y_train)
y_train_pred = forest.predict(X_train)
y_test_pred = forest.predict(X_test)


mae_train = mean_absolute_error(y_train, y_train_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)
print(f'MAE train: {mae_train:.2f}')
print(f'MAE test: {mae_test:.2f}')


r2_train = r2_score(y_train, y_train_pred)
r2_test =r2_score(y_test, y_test_pred)
print(f'R^2 train: {r2_train:.2f}')
print(f'R^2 test: {r2_test:.2f}')


# unfortunately, as show the RF tends to overfit the training data. However it still is able to explain the relationship
# between the target and explanatory variable relatively well (R^2 = 0.85 on test dataset).
# In comparison with the linear model previous used that was overfitting less, but performed worse on the test set (R^2 = 0.75)


# Residual prediction
x_max = np.max([np.max(y_train_pred), np.max(y_test_pred)])
x_min = np.min([np.min(y_train_pred), np.min(y_test_pred)])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3), sharey=True)

ax1.scatter(y_test_pred, y_test_pred - y_test,
            c='limegreen', marker='s', edgecolor='white',
            label='Test data')
ax2.scatter(y_train_pred, y_train_pred - y_train,
            c='steelblue', marker='o', edgecolor='white',
            label='Training data')
ax1.set_ylabel('Residuals')

for ax in (ax1, ax2):
    ax.set_xlabel('Predicted values')
    ax.legend(loc='upper left')
    ax.hlines(y=0, xmin=x_min-100, xmax=x_max+100, color='black', lw=2)

plt.tight_layout()
plt.show()

# As we concluded earlier from the R^2 coefficient earlier, we see the model fits the training data
# better than the test data, as indicated by the outliers in the y-axis direction. Also the distribution of
# the residuals does not seem to be completely random around the zero center point, indicating the model is not
# able t o capture all the exploratory information. However the residual plot indicates a large improvement
# over the residual plot of the linear model that we plotted earlier. Ideally our model error should be random
# or unpredictable. Meaning, the erro of the prediction should not be related to any of the information contained
# in the explanatory variables; rather it should reflect the randomness of the real-world distributions or patterns
# If we find patterns in the prediction errors, i.e., by inspecting the residual plot, it means that the residual plots
# contain predictive information. A common reason for this could be that explanatory information is leaking into those residuals.
# Unfortunately there is no universal approach for dealing with non-randomness in residual plots and requires experimentation
# Depending on the data available to use, we may be able to improve the model by transforming variables, tuning hyperparameters
# of the learning algorithm, choosing simpler/complex models, removing outliers, or including additional variables.
