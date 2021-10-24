from cvxpylayers.tensorflow import CvxpyLayer
import cvxpy as cp
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
tf.compat.v1.enable_eager_execution()
tf.config.run_functions_eagerly(True)

class Denoise(layers.Layer):
  def __init__(self, input_shape, lam_val,**kwargs):
    super(Denoise, self).__init__(**kwargs)
    tf.executing_eagerly()
    self.input_dim,_ = input_shape
    self.y_parm = cp.Parameter(self.input_dim)
    self.lambda_parm = cp.Constant(value=lam_val)
    self.x_var = cp.Variable((self.input_dim))
    self.objective = cp.sum_squares(self.y_parm-self.x_var) + self.lambda_parm*cp.sum_squares(cp.diff(self.x_var,2))
    self.constrains = [self.x_var<=self.y_parm,self.x_var>=0]
    self.problem = cp.Problem(cp.Minimize(self.objective),constraints=self.constrains)
    self.cvxpy_layer = CvxpyLayer(problem=self.problem, parameters = [self.y_parm], variables=[self.x_var])

  def call(self, data):
    tf.executing_eagerly()
    denoised = np.zeros(shape=(28,28))
    denoised_train = np.ndarray(shape=data.shape)
    img_ix =0
    for img in data: 
        for i in range(img.shape[0]):
            sig = img[i,:]
            sig = tf.convert_to_tensor(sig)
            ans, = self.cvxpy_layer(sig)
            denoised[i,:] = ans
        denoised = denoised.T
        for i in range(img.shape[1]):
            sig = denoised[:,i]
            sig = tf.convert_to_tensor(sig)
            ans, = self.cvxpy_layer(sig)
            denoised[:,i] = ans
        denoised_train[img_ix,:,:] = denoised.T
        img_ix+=1
    a,b,c = denoised_train.shape
    return denoised_train.reshape(a,b,c,1)