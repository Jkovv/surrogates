from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from sklearn.metrics import mean_squared_error

def create_gpr_model(params):
    kernel = C(1.0) * Matern(length_scale=params['length_scale'], nu=params['nu']) \
             + WhiteKernel(noise_level=params['alpha'])
    
    model = GaussianProcessRegressor(
        kernel=kernel, 
        n_restarts_optimizer=3 
    )
    return model

def train_and_eval_gpr(params, train_set, val_set):
    model = create_gpr_model(params)
    model.fit(train_set[0], train_set[1])
    
    y_pred = model.predict(val_set[0])
    mse = mean_squared_error(val_set[1], y_pred)
    return mse, model