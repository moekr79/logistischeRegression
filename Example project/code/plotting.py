import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from regression import OwnLinearRegressor

def plot_reg(regressor, data):
    """
    Function plotting data points and plotting the regression line.
    """
    plt.figure(figsize=(10, 6))
    ax = sns.scatterplot(data=data, x="total_bill", y="tip")

    # calculate beginning and end of regression line
    X = np.array(data["total_bill"])
    min_max = np.array([np.min(X), np.max(X)])
    pred_min_max = regressor.predict(min_max.reshape(-1, 1))
    # draw line
    plt.plot(min_max, pred_min_max, c="r")

    # set labels and title
    plt.xlabel("Gesamtrechnung (in Dollar)")
    plt.ylabel("Trinkgeld (in Dollar)")
    plt.title(f"Einfache lineare Regression zwischen den Variablen total_bill and the tip")
    # Show the plot
    plt.show()


def plot_conf_band(regressor, alpha=0.05):
    """
    Plots confidenc band around regression line.
    """
    plt.figure(figsize=(10, 6))
    ax = plt.scatter(x=regressor.x, y=regressor.y)

    # calculate beginning and end of regression line
    min_max = np.array([np.min(regressor.x), np.max(regressor.x)])
    pred_min_max = regressor.predict(min_max.reshape(-1, 1))
    # draw line
    plt.plot(min_max, pred_min_max, c="r")
    # calculate confidence intervall at multiple x values
    x_i = np.linspace(min_max[0], min_max[1], 100)
    y_band = regressor.conf_intervall(x=x_i, alpha=alpha)
    # plot lower
    plt.plot(x_i, y_band[:,0], c="y")
    # plot upper
    plt.plot(x_i, y_band[:,1], c="y")
    # plot band
    plt.fill_between(x_i, y_band[:,0], y_band[:,1], alpha=.5, color="y")

    # set labels and title
    plt.xlabel("Gesamt Rechnung (in Dollar)")
    plt.ylabel("Trinkgeld (in Dollar)")
    plt.title(f"Einfache lineare Regression zwischen Trinkgeld und Rechung mit Konfidenzband")
    # Show the plot
    plt.show()


def scatter_tips(data, days_to_plot=["Thur", "Fri", "Sat", "Sun"], ci_bool=True, ci_alpha=.95):
    """
    Plots regression total_bill vs tip for every day seperately.
    Allows selection of days, bool to plot confidence band, and alpha.
    """
    sns.set_palette("bright")  # Set a color palette
    # filter data set on a subset of days
    day_data = data[[day in days_to_plot for day in data['day']]]
    
    ax = sns.lmplot(data=day_data, x="total_bill", y="tip", hue='day', ci=(ci_bool and ci_alpha), facet_kws={"xlim":[0, 53], "ylim":[0, 11]})
    ax.figure.set_size_inches(10, 6)
    # Set labels and title
    plt.xlabel("Gesamtrechnung")
    plt.ylabel("Trinkgeld")
    plt.title("Tageweise Regression zwischen den Variablen total_bill and the tip")

    # Show the plot
    plt.show()