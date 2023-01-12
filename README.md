# Model-Performance-Estimation

# The Problem

The biggest problem facing any potentially modeling approach for foot traffic going forward is a lack of ground truth. The problem manifests in two forms: 

* Firstly a lack of ground truth means we do not have the data to fine tune our model weights when we receive a new source of supply. 
* Secondly, we have the issue of data drift

The first issue is obvious. If we do not have the ground truth available for a time period overlapping with the new supply we receive, then we have to make the (more than likely untrue) assumption that the weights we have trained for other suppliers will be useful for the new supply. This is far from ideal.

The second idea is a little more difficult to explain. Essentially, even if you have a model that was fully trained on a robust base of ground truth, that model can become less accurate over time as the distribution of the model inputs changes. This causes the decision boundaries that your model learned to become no longer optimal.

![Data Drift](https://cdn-images-1.medium.com/max/700/0*RaCL8Lw28SXyevWx.png)

In a scenario where you have additional ground truth to train on, this can be mitigated by continually updating the model over time, training it to fit the new data. If you are lacking ground truth however, this drift can occur unseen.

# Solutions

There are three possible solutions to this problem.

1. Expend effort and resources to source additional ground truth:
  1. Scraping websites for event attendance
  2. Using employee effort to source publicly available data
  3. Purchase data from or partner with outside companies to acquire ground truth data sets
2. Simulate ground truth:
  1. Utilizing past data, methods such as Monte Carlo or Generative Adversarial Networks could be used to create “ground truth”
3. Finding methods to detect and mitigate data drift that do not require ground truth. 

While options 1 & 2 are both viable, and the ideal scenario may be a combination of the three, this page will focus on the third option.

# Direct Loss Estimation

Direct loss estimation (DLE) is a fairly simple concept. During the training of the initial model, when ground truth is available, a second model is trained. This is called the nanny model. The purpose of nanny model is to learn to predict the loss of the primary or child model.

There exists, currently a package to do this their explanation of the logic is below (from [NannyML](https://nannyml.readthedocs.io/en/main/how_it_works/performance_estimation.html)):

Let’s denote with $f$ the monitored model and $h$ the nanny model. Let’s assume we are interested in estimating mean absolute error (MAE) of $f$ for some analysis data for which targets are not available. $f$ was trained on train data and used on reference data providing $f(X_{reference})$ predictions. Targets for reference set $y_{reference}$ are available. The algorithm runs as follows:

1. For each observation of reference data calculate loss which in case of MAE is absolute error of $f$, i.e. $AE_{reference}= |y_{reference}-f(X_{reference})$.

2. Train a nanny model on reference data. As features use the monitored model features $X_{reference}$ and monitored model predictions $f(X_{reference})$ . The target is absolute error $AE_{reference}$ calculated in previous step. So $\hat{AE_{reference}} = h(X,f(X))$ .

3. Estimate performance of child model on analysis data: estimate absolute error for each observation $\hat{AE_{reference}}$
 with $h$ and calculate mean of $\hat{AE_{reference}}$ to get MAE.


## Simple Example

A good example of how this works can be demonstrated with a linear regression fit to a heteroscedastic process.

![Linear Regression on a Heteroskedastic Process](https://nannyml.readthedocs.io/en/main/_images/how-it-works-dle-regression.svg)

The regression (red line) gives us a point estimate of y, but if we were unaware of the ground truth, we would not know that the variance of the model is non-constant. As such we can train a second linear regression model which uses the absolute error between the regression and actual values as the target. This takes advantage of the fact that while linear regression generates errors that are mean 0, the  Absolute Error loss function is not mean 0.

![Errors Mean 0](https://nannyml.readthedocs.io/en/main/_images/how-it-works-dle-regression-errors-hist.svg)

![Absolute Error Not Mean 0](https://nannyml.readthedocs.io/en/main/_images/how-it-works-dle-regression-abs-errors-hist.svg)

This means we can fit a linear regression that takes the input x1 and the model output y-hat as inputs and uses the model absolute error (AE) as a target. In other words it takes the form.

$$\hat{AE} =\hat{|y-\hat{y}|} = \beta_{0} + \beta_{1}x_{1} + \beta_{2}\hat{y} + \epsilon$$

We can now add prediction intervals to our original regression line using

$$PI = \hat{y} \pm \hat{AE}$$

Once the nanny model is trained, we can also now get the estimation of the mean absolute error for our model using only the model estimates and the input x values.

![Model with Prediction Intervals](https://nannyml.readthedocs.io/en/main/_images/how-it-works-dle-regression-PI.svg)

## Assumptions

There is theoretically no restriction on what kind of models can be used for the nanny model, as long as it takes in the original model’s inputs/predictions and returns the estimated AE as it’s output. The nanny model should be treated as any other ML model, and can be made better or worse by alteration of hyper parameters. It must be tuned as any other model is tuned.

If the nanny model is well trained, then it should perform well even given data drift, however like many statistical concepts, it has some base assumptions that should be met (from [NannyML](https://nannyml.readthedocs.io/en/main/how_it_works/performance_estimation.html))

### **There is no concept drift.**
* While dealing well with covariate shift, DLE will not work under concept drift. This shouldn’t happen when the child model is has access to all the variables affecting the outcome and the problem is stationary. An example of a stationary model would be forecasting energy demand for heating purposes. Since the phyiscal laws underpinning the problem are the same, energy demand based on outside temperature should stay the same. However if energy prices became too high and people decide to heat their houses less because they couldn’t pay, then our model would experience concept drift.

### **There is no covariate shift to previously unseen regions in the input space.**
* The monitored model will most likely not work if the drift happens to subregions in the inputs space that were not seen before. In such case the monitored model has not been able to learn how to predict the target. The same applies to the nanny model - it cannot predict how big of an error the monitored model will make. There might be no error at all, if the monitored model happens to extrapolate well. Using the same example - heat demand forecasting model will most likely fail during extremely warm days during winter that did not happen before (i.e. were not included in the model training data).

### **The noise is heteroscedastic around the monitored model target and it is dependent on the monitored model input features.**
* This is equivalent to there are regions where the monitored model performs better or worse. DLE also works when the noise is homoscedastic (noise distribution around the target is constant) but then the true performance of the monitored model is constant (depending on the metric used, it will be constant for MAE and MSE, it will change when measured e.g. with MAPE). Variation of true performance on the samples of data will be then an effect of sampling error only. Heat demand forecasting model is again a good example here. It is known that such models perform worse in some periods, for example in intermediate periods (that would be spring and autumn in Europe). The demand in such periods is governed by many factors that are hard to account for in the demand predicting model, therefore for the similar conditions (date, time, weather etc.) the target variable takes different values as it is affected by these unobserved variables. On the other hand during winter these models are precise as the demand is mostly driven by the outdoor temperature.

### **The sample of data used for estimation is large enough.**

# Implementation

Implementation can be seen in the attached Colab Notebook


# Sources

[Estimating Model Performance without Ground Truth](https://towardsai.net/p/l/estimating-model-performance-without-ground-truth)

[Nanny ML Read the Docs](https://nannyml.readthedocs.io/en/main/how_it_works/performance_estimation.html)
