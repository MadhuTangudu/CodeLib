"""
Timeseries module.

**Available routines:**

- class ``FB_Prophet``: Builds Prophet model using cross validation.

Credits
-------
::

    Authors:
        - Diptesh
        - Madhu

    Date: Sep 25, 2021
"""

# pylint: disable=invalid-name
# pylint: disable=R0902,R0903,R0913,C0413

from typing import List, Dict, Any

import re
import sys
from inspect import getsourcefile
from os.path import abspath

import datetime
import pandas as pd
import numpy as np

from fbprophet import Prophet

path = abspath(getsourcefile(lambda: 0))
path = re.sub(r"(.+\/)(.+.py)", "\\1", path)
sys.path.insert(0, path)

import metrics  # noqa: F841


def regressor_index(m, name):
    """Given the name of a regressor, return its (column) index in the `beta` matrix.
    Parameters
    ----------
    m: Prophet model object, after fitting.
    name: Name of the regressor, as passed into the `add_regressor` function.
    Returns
    -------
    The column index of the regressor in the `beta` matrix.
    """
    return np.extract(
        m.train_component_cols[name] == 1, m.train_component_cols.index
    )[0]


def regressor_coefficients(m):
    """Summarise the coefficients of the extra regressors used in the model.
    For additive regressors, the coefficient represents the incremental impact
    on `y` of a unit increase in the regressor. For multiplicative regressors,
    the incremental impact is equal to `trend(t)` multiplied by the coefficient.
    Coefficients are measured on the original scale of the training data.
    Parameters
    ----------
    m: Prophet model object, after fitting.
    Returns
    -------
    pd.DataFrame containing:
    - `regressor`: Name of the regressor
    - `regressor_mode`: Whether the regressor has an additive or multiplicative
        effect on `y`.
    - `center`: The mean of the regressor if it was standardized. Otherwise 0.
    - `coef_lower`: Lower bound for the coefficient, estimated from the MCMC samples.
        Only different to `coef` if `mcmc_samples > 0`.
    - `coef`: Expected value of the coefficient.
    - `coef_upper`: Upper bound for the coefficient, estimated from MCMC samples.
        Only to different to `coef` if `mcmc_samples > 0`.
    """
    assert len(m.extra_regressors) > 0, 'No extra regressors found.'
    coefs = []
    for regressor, params in m.extra_regressors.items():
        beta = m.params['beta'][:, regressor_index(m, regressor)]
        if params['mode'] == 'additive':
            coef = beta * m.y_scale / params['std']
        else:
            coef = beta / params['std']
        percentiles = [
            (1 - m.interval_width) / 2,
            1 - (1 - m.interval_width) / 2,
        ]
        coef_bounds = np.quantile(coef, q=percentiles)
        record = {
            'regressor': regressor,
            'regressor_mode': params['mode'],
            'center': params['mu'],
            'coef_lower': coef_bounds[0],
            'coef': np.mean(coef),
            'coef_upper': coef_bounds[1],
        }
        coefs.append(record)

    return pd.DataFrame(coefs)

class FBProphet():
    """FB Prophet module.

    Objective:
        - Build
          `FB Prophet <https://peerj.com/preprints/3190/>`_

    Parameters
    ----------
    df : pandas.DataFrame

        Pandas dataframe containing the `y_var`, `epoch` and `x_var`

    y_var : str

        Dependant variable (the default is "y")

    epoch : str

        Time epoch (the default is "ds")

    x_var : List[str]

        Independant variables (the default is None)


    k_fold : int, optional

        Number of cross validations folds (the default is None)

    param : dict, optional

        FB Prophet parameters (the default is None).
        In case of None, the parameters will default to::

            changepoint_prior_scale: [0.001, 0.01, 0.5]
            seasonality_prior_scale: [0.01, 0.1, 10.0]

    Returns
    -------
    model : object

        Final optimal model.

    best_params_ : Dict

        Best parameters amongst the given parameters.

    model_summary : Dict

        Model summary containing key metrics like RMSE, MSE, MAE, MAPE

    Methods
    -------
    predict

    Example
    -------
    >>> mod = FBProphet(df=df_ip, y_var="y", x_var=["x1", "x2", "x3"])
    >>> df_op = mod.predict(x_predict)

    """

    def __init__(self,
                 df: pd.DataFrame,
                 y_var: str = "y",
                 x_var: List[str] = None,
                 epoch: str = "ds",
                 k_fold: int = None,
                 test_size: float = 0.2,
                 param: Dict = None):
        """Initialize variables for module ``FB Prophet``."""
        self.y_var = y_var
        self.x_var = x_var
        self.epoch = epoch
        if self.x_var is None:
            self.df = df[[self.epoch] + [self.y_var]]
        else:
            self.df = df[[self.epoch] + [self.y_var] + self.x_var]
        self.df.rename(columns={self.y_var:"y"}, inplace=True)
        self.df.rename(columns={self.epoch:"ds"}, inplace=True)
        self.df = self.df.reset_index(drop=True)
        self.model = None
        self.k_fold = k_fold
        self.test_size = test_size
        self.max_epoch = max(self.df["ds"])
        if param is not None or k_fold is not None:
            raise Exception("cross validated not implemented yet!.")
        self.param = param
        self.best_params_ = None
        self.model_summary = None
        self._fit()
        self._compute_metrics()


    def _fit(self) -> Dict[str, Any]:
        """Fit FB Prophet model."""
        # split into train test data sets
        df_train = self.df.iloc[:int(len(self.df)*self.test_size), :]
        # df_test = self.df.iloc[int(len(self.df)*self.test_size):, :]
        self.model = Prophet(interval_width=0.95)
        if self.x_var is not None:
            for var in self.x_var:
                self.model.add_regressor(var)
        self.model.fit(df_train)
        if self.x_var is not None:
            self.best_params_ = regressor_coefficients(self.model)

    def _compute_metrics(self):
        """Compute commonly used metrics to evaluate the model."""
        y = self.df.loc[:, "y"].values.tolist()
        if self.x_var is None:
            y_hat = list(self.model.predict(self.df[["ds"]])["yhat"])
        else:
            y_hat = list(self.model.predict(self.df[["ds"]+self.x_var])["yhat"])
        model_summary = {"mae": np.round(metrics.mae(y, y_hat), 3),
                         "mape": np.round(metrics.mape(y, y_hat), 3),
                         "rmse": np.round(metrics.rmse(y, y_hat), 3)}
        model_summary["mse"] = np.round(model_summary["rmse"] ** 2, 3)
        self.model_summary = model_summary

    def predict(self,
                x_predict: pd.DataFrame = None,
                period: int = None) -> pd.DataFrame:
        """Predict y_var/target variable.

        Parameters
        ----------
        df_predict : pd.DataFrame

            Pandas dataframe containing `x_var`, "epoch" (defaults to None).

        period : int

            Number of future epochs to be predicted. (defaults to None)

        Returns
        -------
        pd.DataFrame

            Pandas dataframe containing predicted `y_var`, 'epoch' and `x_var`.

        """
        # Exception handling
        if x_predict is None and self.x_var is not None:
            raise Exception("Please check input arguments.")
        if (period is not None
                and (x_predict is not None or self.x_var is not None)):
            raise Exception("Please check input arguments.")
        if (x_predict is None and self.x_var is None) and period is not None:
            if period <= 0:
                raise Exception("Please check period argument.")
        # Prediction
        if period is not None:
            dateList = []
            for x in range(1, period+1):
                dateList.append(self.max_epoch+datetime.timedelta(days=x))
            x_predict = pd.DataFrame(dateList)
            x_predict.columns = [self.epoch]
        if x_predict is not None and self.x_var is None:
            df_op = x_predict.copy(deep=True)
            x_predict.rename(columns={self.epoch:"ds"}, inplace=True)
        else:
            df_op = x_predict.copy(deep=True)
            x_predict.rename(columns={self.epoch:"ds"}, inplace=True)
            x_predict = x_predict[["ds"]+self.x_var]
        y_hat = self.model.predict(x_predict)["yhat"]
        df_op.insert(loc=0, column=self.y_var, value=y_hat)
        return df_op
