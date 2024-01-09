from scipy.odr import ODR, Model,Data,RealData
import numpy as np
import matplotlib.pyplot as plt 
# from inspect import signature

class PremadeFunctions:
  @staticmethod
  def func_linear(beta,x):
    '''
    function form: y=m*x+b
    m = beta[0]
    b = beta[1]
    '''
    m,b = beta
    return m*x+b
  @staticmethod
  def func_gaussian(beta,x):
    """
    a	=	height of the curve's peak
    b	=	the position of the center of the peak
    c	=	the standard deviation
    """
    a,b,c = beta
    return a*np.exp(-((x-b)**2)/(2*c**2))

  def func_damp_oscillator(beta,x):
    """
    a,c,g = beta
    returns a/np.sqrt((c**2-x**2)**2 + (g*x)**2)
    """
    a,c,g = beta
    return a/np.sqrt((c**2-x**2)**2 + (g*x)**2)
    
class ODRPlotter():
  @staticmethod
  def make_fit_label(beta_names,beta,cov_mat,decimals=3,latex=None):
    errs = [F"{i:0.3e}" for i in np.sqrt(np.diag(cov_mat))]
    label = "ODR Fit Curve\n"
    beta = np.round(beta,decimals)
    for name,value,err in zip(beta_names,beta,errs):
      label += F"{name}: {value} $\pm$ {err}" + "\n"
    if(latex):
      label += f"Fit: {latex}"
    return label
  
  @staticmethod
  def fit_and_plot(x_data,
                      y_data,
                      beta0,
                      fit_func=PremadeFunctions.func_linear,
                      x_err=None,
                      y_err=None,
                      beta_names=None,
                      fig = None,
                      ax = None,
                      ms=3,
                      alpha=0.65,
                      latex=None):

    '''
    fit function of form def fit_func(beta,x)

    beta_names: name your parameters in the same order as beta

    beta0: initial start params, defaulted at 1's

    use fig,ax if you want an automated plot,
    beta_names must also be given to create a label
    '''
    data = RealData(x_data, y_data, x_err, y_err)
    model = Model(fit_func)
    if(beta0!=None):
      odr = ODR(data, model, beta0=beta0)
    else:
      odr = ODR(data, model)
    odr.set_job(fit_type=0)
    output = odr.run()
    X = np.linspace(np.min(x_data),np.max(x_data),1000)
    if(fig!=None and ax!=None):


      ax.errorbar(x_data,y_data,y_err,x_err,"o",label="Experimental Data",ms=ms,alpha=alpha)
      ax.plot(X,fit_func(output.beta,X),label=ODRPlotter.make_fit_label(beta_names,output.beta,output.cov_beta,latex=latex))
      ax.legend(bbox_to_anchor = (1,1),shadow=True)
      plt.subplots_adjust(right=0.8)
    return output,fit_func

class LSPlotter():
  @staticmethod
  def make_fit_label(beta_names,beta,sd_beta,decimals=3,latex = None):
    errs = [F"{i:0.3e}" for i in sd_beta]
    label = "Least Squares Fit\n"
    beta = np.round(beta,decimals)
    for name,value,err in zip(beta_names,beta,errs):
      label += F"{name}: {value} $\pm$ {err}" + "\n"
    if(latex):
      label += f"Fit: {latex}"
    return label
  
  @staticmethod
  def fit_and_plot(x_data,
                      y_data,
                      beta0,
                      fit_func=PremadeFunctions.func_linear,
                      x_err = None,
                      y_err = None,
                      beta_names=None,
                      fig = None,
                      ax = None,
                      ms=3,
                      alpha=0.65,
                      latex = None):
    '''
    fit function of form def fit_func(beta,x)

    beta_names: name your parameters in the same order as beta

    beta0: initial start params, defaulted at 1's

    use fig,ax if you want an automated plot,
    beta_names must also be given to create a label
    '''
    data = RealData(x_data, y_data, x_err, y_err)
    model = Model(fit_func)
    if(beta0!=None):
      odr = ODR(data, model, beta0=beta0)
    else:
      odr = ODR(data, model)
    odr.set_job(fit_type=2)
    output = odr.run()
    if(fig!=None and ax!=None):
      X = np.linspace(np.min(x_data),np.max(x_data),1000)


      ax.errorbar(x_data,y_data,y_err,x_err,"o",label="Experimental Data",ms=ms,alpha=alpha)
      ax.plot(X,fit_func(output.beta,X),label=LSPlotter.make_fit_label(beta_names,output.beta,output.sd_beta,latex=latex))
      ax.legend(bbox_to_anchor = (1,1),shadow=True)
      plt.subplots_adjust(right=0.8)
    return output,fit_func