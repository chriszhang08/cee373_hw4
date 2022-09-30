from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
from sklearn.preprocessing import PolynomialFeatures

def df_to_xy(df):
    # convert a pandas dataframe to x and y numpy arrays
    # return x and y
    x = df['X'].values.reshape(-1,1)
    y = df['y'].values.reshape(-1,1)
    return x, y

def fit_linear_regression(x, y, model):
    # fit a linear regression model to the data
    # return the model
    # make the y intercept 0
    model.fit(x,y)
    return model

def fit_polynomial_regression(x, y, degree):
    # fit a polynomial regression model to the data
    # return the model
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    x_poly = poly.fit_transform(x)
    model = LinearRegression()
    model.fit(x_poly, y)
    return x_poly, model

def plot_linear_regression(x, y, y1):
    # plot the data and the linear regression model
    plt.scatter(x, y, s=8, color='blue')
    plt.plot(x, y1, color='red')
    plt.xlabel('Movie Budget in (millions of USD)')
    plt.ylabel('IMDb Rating')
    plt.title('Movie Budget vs. IMDb Rating')
    plt.show()

def clean_fantasy_data(df):
    df = df.loc[df['Pos'] == 'RB', ['Player', 'Tgt', 'RushingAtt',
                                    'FantasyPoints']]  # filter rows based off position, and filter columns
    df['Usage'] = df['Tgt'] + df['RushingAtt']  # create a new column for usage

    df['UsageRank'] = df['Usage'].rank(ascending=False)

    df['FantasyPointsRank'] = df['FantasyPoints'].rank(ascending=False)

    df.sort_values(by='UsageRank').head(15)
    # remove all elements where fantasy points is empty
    df = df.loc[df['FantasyPoints'].notna(), :]
    x = df['Usage'].values.reshape(-1, 1)
    y = df['FantasyPoints'].values.reshape(-1, 1)
    return x, y

#%%

df = pd.read_csv('2021.csv')
# split the dataframe in 2 halves, x < 0 and x > 0
model = LinearRegression()

X, y = clean_fantasy_data(df)
model = fit_linear_regression(X, y, model)
plot_linear_regression(X, y, model.predict(X))

scores = model.score(X, y)
print(scores)
# print final model equation
print('y = {:.2f} + {:.2f}x'.format(model.intercept_[0], model.coef_[0][0]))

# X, y = df_to_xy(df2)
# model = fit_linear_regression(X, y, model)
# plot_linear_regression(X, y, model.predict(X))

# X, y = df_to_xy(df)
# X_plot, model = fit_polynomial_regression(X, y, 7)
# plot_linear_regression(X, y, model.predict(X_plot), model)
# # array([[ 1.19781792e-01, -7.16171081e-04,  8.67103337e-06]])

#%%
# print how well the model fits the data
scores = model.score(X, y)
# scores = model.score(X_plot, y)
print(scores)
print(model.coef_)

#%%
df = pd.read_csv('movieData.csv')
# isolate just the budget and IMDB rating columns
df = df.loc[:, ['Title', 'budget', 'IMDb_rating']]

#%%
# clean the budget in terms of thousands
# in terms of millions
df['budget'] = df['budget'].floordiv(1000000)

#%%
def clean_movie_data(df):
    x = df['budget'].values.reshape(-1, 1)
    y = df['IMDb_rating'].values.reshape(-1, 1)
    return x, y
budget, rating = clean_movie_data(df)

#%%
model = LinearRegression()
model = fit_linear_regression(budget, rating, model)

#%%
plot_linear_regression(budget, rating, model.predict(budget))
scores = model.score(budget, rating)
print(scores)
# print final model equation
print('y = {:.2f} + {:.2f}x'.format(model.intercept_[0], model.coef_[0][0]))
