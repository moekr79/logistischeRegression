import seaborn as sns
import numpy as np
import scipy as sp

class OwnLinearRegressor:
    """
    Create class to apply a linear regression on the data.

    Attributes:
        x: data used to predict
        y: data to predict

    """
    def __init__(self, data, x, y):
        self.data = data
        self.extract_data(x, y)
        self.predict = None
        self.n = np.shape(self.x_tilde)[1]
        self.stats = None

    def extract_data(self, x, y):
        """
        Sets data:
            x: data used to predict
            y: data to predict
        """
        self.x = np.array(self.data[x]).reshape(-1, 1)
        self.x_tilde = self.stack_ones(self.x)
        self.y = np.array(self.data[y])
        self.mean_x = np.mean(self.x)
        self.mean_y = np.mean(self.y)
        self.calc_var_x()

    def umvu_beta(self):
        """
        Calculating beta hat based on given dataset.
        Definition is in the main notebook.

        Sets:
            umvu-estimator for the linear model
            beta @ [1, X] = y
        """
        # computes umvu-estimator
        self.beta = np.linalg.inv(self.x_tilde@self.x_tilde.T)@self.x_tilde@self.y

    def umvu_reg(self):
        """
        returns predict function for linear regression.
        Estimates beta using function umvu_beta_hat.

        Sets:
            prediction function z -> beta @ [1, z]
        """
        self.umvu_beta()
        # computes lambda function which computes predictions
        self.predict = lambda z:self.beta@self.stack_ones(z)
    
    @staticmethod
    def stack_ones(x):
        """
        Stacks a row of one on top of the regressors.
        """
        return np.vstack([np.ones(np.shape(x)[0]), x.T])

    def conf_intervall(self, x, alpha):
        """
        Returns the confidence intervall at a point x.
        Uses the prediction function and the data (X, y).


        Args:
            x: datapoint(s) at which to calculate the interval at.
            alpha: confidence level (between 0 and 1)
        Returns:
            list of intervalls [y_min, y_max] for each datapoint in x.
        """
        y_hat = self.predict(x)
        self.calc_conf_band_size(alpha)
        conf_band = self.conf_band(x)
        y_min = y_hat - conf_band
        y_max = y_hat + conf_band
        return np.stack([y_min, y_max], axis=-1)

    def calc_conf_band_size(self, alpha):
        """
        Calculates deviation of the prediction to get the confidence intervall.

        Args:
            alpha: confidence level (between 0 and 1)
        Returns:
            function to calculate deviation of (1 - alpha) confidence intervall
            symmetrical around prediction at a specific point
        """
        t_quantile = sp.stats.t.ppf(1-alpha/2., self.n-2)
        self.calc_pred_error()
        self.conf_band = lambda x:t_quantile*np.sqrt(self.pred_error)*np.sqrt(1/self.n+(x-self.mean_x)**2/self.var_x**2)
    
    def calc_pred_error(self):
        """
        Calculate squared error of regression.
        """
        self.pred_error = 1/(self.n-2)*np.sum((self.y-self.predict(self.x))**2)
        return self.pred_error
    
    def calc_var_x(self):
        """
        Calculates estimator for variance within the regressors.
        """
        self.var_x = np.sqrt(np.sum((self.x-self.mean_x)**2))

    def calc_t_stat(self):
        """
        Function to calculate the t statistics to a prediction.
        Only in the one dimensional case!
        """
        # slope of regression
        slope = self.beta[1]
        # update pred error
        self.calc_pred_error()
        # calculate t-stat
        self.t_stat = slope*self.var_x/np.sqrt(self.pred_error)
        return self.t_stat
    
    def calc_p_value(self):
        """
        Calculates the p-value of the OLS regressor.
        """
        # calc p-value given the t-statistics
        self.p_value = (1-sp.stats.t.cdf(self.t_stat, self.n-2))*2
        return self.p_value
    