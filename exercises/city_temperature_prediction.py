import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go

pio.templates.default = "simple_white"

pd.options.mode.chained_assignment = None

def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=[2])
    df = df.dropna()
    df = df[df['Day'].between(1, 31)]
    df = df[df['Month'].between(1, 12)]
    df["DayOfYear"] = pd.Series(map(lambda d: d.day_of_year, df["Date"]))
    df = df[df['Temp'] > -50]

    return df



if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data('../datasets/City_Temperature.csv')

    # Question 2 - Exploring data for specific country
    israel_data = df[df['Country'] == 'Israel']
    israel_data['Year'] = israel_data.Year.astype(str)
    fig = px.scatter(israel_data, x='DayOfYear', y='Temp', color='Year',
                     title='Average Daily Temperature as a Function of The '
                           'Day')

    fig.show()
    std_df = israel_data.groupby(israel_data["Month"]).std()
    fig = px.bar(std_df, y="Temp", labels={"Temp":"Standart deviation of "
                                                  "temperature"}, title =
    "Standard Deviation of Temperature at Every Month")
    fig.show()

    # Question 3 - Exploring differences between countries
    groupby_country = df.groupby([df['Country'], df['Month']])
    std = groupby_country.std().reset_index()
    mean = groupby_country.mean().reset_index()
    mean['std'] = std['Temp']
    fig = px.line(mean, x="Month", y="Temp",labels={"Temp":"Average "
                                                           "Temperature"},
                                                           color="Country",
                  error_y='std', title="Average Temperature for Every "
                                       "Country and STD")
    fig.show()

    # Question 4 - Fitting model for different values of `k`
    y = israel_data.pop('Temp')
    train_X, train_y, test_X, test_y = split_train_test(israel_data, y, 0.75)
    polynom_degree = []
    loss_values = []
    for i in range(1, 11):
        Model = PolynomialFitting(i)
        Model.fit(train_X['DayOfYear'].values, train_y.values)
        loss = round(Model.loss(test_X['DayOfYear'].values, test_y.values), 2)
        print(loss)
        loss_values.append(loss)
        polynom_degree.append(i)
    df_polynom_loss = pd.DataFrame(polynom_degree, columns=['Polynom Degree'])
    df_polynom_loss['Loss'] = loss_values
    fig = px.bar(df_polynom_loss, x='Polynom Degree',
                 y='Loss',title="Test Error for Each Value "
                                                 "of k")
    fig.show()

    # Question 5 - Evaluating fitted model on different countries
    degree = loss_values.index(min(loss_values)) + 1
    model_5 = PolynomialFitting(degree)
    model_5.fit(israel_data['DayOfYear'], y)
    countries_data = {country: df[df['Country'] == country] for country in
                      df['Country'].unique()}
    del countries_data['Israel']
    loss_dict = dict.fromkeys(countries_data.keys())
    for country, data in countries_data.items():
        y_real = data.pop('Temp')
        loss_dict[country] = model_5.loss(data['DayOfYear'],y_real)
    loss_df = pd.DataFrame(loss_dict.values(), index=loss_dict.keys(),
                           columns=["Loss"])
    fig = px.bar(loss_df, y="Loss", labels={"index": "Country"}, title="Loss for Every Country on Fitted Model")
    fig.show()


