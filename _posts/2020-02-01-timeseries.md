---
layout: post
title:  "Time Series Forecasting"
info: "Using econometric and deep learning methods for time-series forecasting/predictions"
tech: "Python, pymongo, API"
type: B Company
img: "/assets/img/forecasting/demo.gif" 
img_dimensions: ["400","270"]
concepts: "Econometric, Time-Series Modeling, Neural Networks"
type: "blog"
link: "https://github.com/udipbohara/forecasting_dash"
tags: ["Time-Series", "Deep Learning"]
usemathjax: true
---


  <div style="text-align: center">
  <i class="fa fa-code"></i> <a href="https://github.com/udipbohara/forecasting_dash">Code to this Repo</a>
  </div>
<br>

Time-Series Forecasting is a very relevant business problem that can be highly effective in proper allocation of resources to meet the need/demand. In this particular project, I take on the problem of forecasting Electricity demand for all the regions of the USA. 


# Part 1: EDA 

This app takes EIA data from all the regions from the US for electricity demand and uses econometric modeling for forecasting for the following regions

<div>
    <center>
    <img src="/assets/img/forecasting/main.png"/>
        <br>
     <text><b> Fig: Different Regions and their Corresponding States </b> <br> 
         <i> (Source: https://www.eia.gov)</i>
     </text>
    </center>
</div>


Here is a json file I have created for each region and its corresponding states. 
```json
[
{"region":"CAL","states":["CA"]},
{"region":"CENT","states":["ND","SD","MN","IA","MO","NE","KS","OK"]},
{"region":"FLA","states":["FL"]},
{"region":"MIDW","states":["MN","MI","IL","IN","KY","MS","AR","LA","WI"]},
{"region":"MIDA","states":["PA","OH","WV","MD","DE","NJ","VA"]},
{"region":"NE","states":["ME","VT","NH","MA","CT","RI"]},
{"region":"CAR","states":["NC","SC"]},
{"region":"NW","states":["WA","OR","ID","MT","WY","NV","UT","CO"]},
{"region":"SE","states":["MS","AL","GA"]},
{"region":"SW","states":["AZ","NM"]},
{"region":"TEN","states":["TN"]},
{"region":"TEX","states":["TX"]},
{"region":"NY","states":["NY"]}
]

```

### Let us take the example of NY:

The Electricity Data can be found in the [U.S Energy Information Administration(EIA) website](https://www.eia.gov/realtime_grid/?src=data#/status?end=20201015T07) which can be pulled with their API.

The energy demand data is available in thousand mega-watthours and is avwailable hourly from 2015. 

__Note__: The consumption is given in Thousand $MWh$. However, I am going to refer it as just MWh in this project for the sake of simplicity.

<img src="/assets/img/forecasting/1_a.png">


Here is the Frequency plot for the data. As we can see the mean is around 18041. If we plot the KDE for each year, we can see that there is a small shift towards lower consumption in 2019.
This could mean that there are investments in alternative source of energy or could just be an anomaly.

<p style="text-align: center;"><b>Two plots that show</b></p>

<div class="row"> 
  <div class="column">
    <img src="/assets/img/forecasting/1_b.png" width='100%'>
  </div>
  <div class="column">
    <img src="/assets/img/forecasting/1_c.png" width='92%' height='84%'>
  </div>
</div>

It is interesting to see how the electricity demand is stratified by different temporal changes. 


We can see that the highest consumptions are during July and August which can be attributed to Summer months and the usage of Air-conditioining increases the demand.
We can clearly see that the highlest consumption is around 6:00PM EST for the weekdays. Whereas the weekends has comparatively lower demand. This can be attributed to holidays.

<p style="text-align: center;"><b>Frequency and Yearly KDE plots</b></p>
<div class="row"> 
  <div class="column">
    <img src="/assets/img/forecasting/1_d.png" width='100%'>
  </div>
  <div class="column">
    <img src="/assets/img/forecasting/1_e.png" width='100%' height='95%'>
  </div>
</div>

<br>
This can futher be validated with a box/whisker plot:

Here, Saturday and Sunday are automatically non-working days whereas Holidays are accounted for weekdays to see the distribution of working vs non working days.

<img src="/assets/img/forecasting/1_f.png">



### Consider Weather data.
Since we can clearly see that temperature affects the demand.

Weather data can be obtained from [National Oceanic and Atmospheric Administration(NOAA)](https://www.noaa.gov)


Plotting the consumption values sampled by day with temperatures gives us this:
Here, I have plotted summer as months from June to Oct and Nov to May as Winter

<img src="/assets/img/forecasting/2_a.png" width="70%" height="70%">



# Time-Series Modeling: Econometric Approach

__Time-Series vs Traditional Machine learning?__

Typical ML methods are characterised by nonparametric modelling and have very relaxed assumptions while Time-series models are "tightly-parametric". Also, Typical machine learning methods assume that your data is independent and identically distributed, which isn't true for time series data. Therefore they are at a disadvantage compared to time series techniques, in terms of accuracy.

Stationary: It is a key assumption for these models. A stationary time series is one whose properties do not depend on the time at which the series is observed.14 Thus, time series with trends, or with seasonality, are not stationary — the trend and seasonality will affect the value of the time series at different times



Tests for stationarity:
Unit root test: (Augmented) Dickey Fuller
Stationarity test: KPSS test

Concept of Unit-root tests:
- Null hypothesis: Unit-root (is a stochastic trend in a time series, sometimes called a “random walk with drift”)
- Alternative hypothesis: Process has root outside the unit circle, which is usually equivalent to stationarity or trend stationarity

Concept of Stationarity tests
- Null hypothesis: (Trend) Stationarity
- Alternative hypothesis: There is a unit root.


ADF test statistic: -4.71 (p-val: 0.00)
KPSS test statistic: 0.18 (p-val: 0.10)

Here KPSS test states that there is no stationarity.

Seasonality: Seasonality stands for repeated pattern in the data.

Autocorrelation:
Autocorrelation can be seen as the measure of internal correlation in a time series. It is a way of measuring and explaining the internal association between observations. Autocorrelation measures historic time data called lags. Autocorrelation functions are used to plot such correlations with respect to lag. It shows if the previous state has an effect on the current state. Autocorrelation is also used to determine if the series is stationary or not. If the series is stationary, the autocorrelation will fall to zero almost instantaneously, however, if it is non-stationary, it will gradually decrease.

Partial Autocorrelation:
PACF is a partial auto-correlation function. Basically instead of finding correlations of present with lags like ACF, it finds correlation of the residuals (which remains after removing the effects which are already explained by the earlier lag(s)) with the next lag value hence ‘partial’ and not ‘complete’ as we remove already found variations before we find the next correlation.

SARIMA Model: SARIMA (Seasonal Auto Regressive Integrated Moving Average Model) 


__Auto regressive (AR)__ process , a time series is said to be AR when present value of the time series can be obtained using previous values of the same time series i.e the present value is weighted average of its past values. This is determined by the PACF plot.
__Moving average (MA)__ process, a process where the present value of series is defined as a linear combination of past errors.
We find optimum features or order of the AR process using the PACF plot, as it removes variations explained by earlier lags so we get only the relevant features.


ARIMA(p,d,q) is generalized form where p is the order of the autoregressive polynomial(PACF), q is the order of the moving average polynomial(ACF) and d is the order of difference needed to achieve stationarity. 

Example: ARMA(1,1), Here is what the equation for Demand $(D)$ be

$$D_{t} =  \beta_{0} + \beta_{1}D_{t-1} + \phi_{1}\epsilon_{t-1} + \epsilon_{t}$$ <br>

where, <br>
$$\beta_{0} + \beta_{1}D_{t-1}$$ is the AR(1) <br>which would answer how much demand today based on demand yesterday?

$$\phi_{1}\epsilon_{t-1} + \epsilon_{t}$$ is the MA(1) <br>which would answer how much demand today based on the residual error (error by how much we were wrong by) from yesterday. $\epsilon_{t}$ is the error from today.

Hence, for our forecast of basic ARMA(1,1) would be:

$$\hat{D}_{t} =   \beta_{0} + \beta_{1}D_{t-1} + \phi_{1}\epsilon_{t-1}$$



Seasonal Decompostion of the demand:

<img src="/assets/img/forecasting/3_a.png" width="90%" height="90%">

- residual is what's left over when the trend and seasonality have been removed. 

<img src="/assets/img/forecasting/3_b.png" width="50%" height="50%">

There is clear seasonality trend (7 days) which we can take away by doing a differencing.


After doing the differencing, the data looks like this:

<img src="/assets/img/forecasting/3_c.png" width="50%" height="50%">

and the ACF/PACF plots look like this:

<img src="/assets/img/forecasting/3_d.png" width="60%" height="60%">


## Multiple Seasonality:

__Exogenous Variables  with $$SARIMAX(p,d,q)(P,D,Q)m$$:__
$$p,d,q$$ were discussed earlier. $$P,D,Q$$ and m are as follows:
- $$P$$ : The order of the seasonal component for the auto-regressive (AR) model.
- $$D$$ : THe integration order of the seasonal process.
- $$Q$$ : The order of the seasonal component of the moving average(MA) model.
- $$m$$ : The number of observations per season cycle

There are two types of exogenous variables that I am adding to the forecasting:

- Temperature : minimum daily temperatures as they have effect on the consumption
- Seasonality : multiple seasonality
- Weekends: 'bool of is weekend or not' as weekends clearly have an effect on the consumption

SARIMAX is SARIMA with exogenous variables added. Since we have multiple seasonality. Weekly and then yearly, we can attribute to the yearly seasonality by adding exogenous variables in form of Fourier terms. Since there are complex seasonality involved with these time series, it is difficult to accurately parametrize them. 

<img src="/assets/img/forecasting/3_e.png" width="60%" height="60%">

Looking at 2 years AutoCorrelation plots, we can see that there are multiple seasonalities to be accounted for.
I decided to add seasonality for every 7, 150 and 365 days.

[Fourier terms can be used to account for multiple seasonality]((https://otexts.com/fpp2/dhr.html))

7 day seasonality can be explicitly given to the model but the other seasonalities need to be sent as exogenous values in [pmdarima](https://alkaline-ml.com/pmdarima/0.9.0/modules/generated/pyramid.arima.auto_arima.html)

The convention is to add anywhere from 1 to 7 fourier terms. Doing a gridsearch on it, I found the optimal parameters to be 4 


I decided to add two more seasonal components 636(Summer component) and 516(Winter component). These are highly parametric. I also decided to add weather data (temperature) as exogenous variables as we can clearly see the impact of temperature on eletricity consumption. 

Here is the snippet of how it was done: __Note__: I have used daily data from 2015 to 2020 for training and testing according to the parameters. So there is a large amount of data to be trained on. 

```python
def optimal_exog_search(y=df1.Consumption, weather=nyc_weather['TMIN'].values):
    exog_trial = pd.DataFrame({'date': y.index})
    exog_trial = exog_trial.set_index(pd.PeriodIndex(exog_trial['date'], freq='D'))
    exog_trial['is_weekend'] = np.where(exog_trial.index.dayofweek < 5,0,1)
    exog_trial['TMIN'] =weather
    exog_trial= exog_trial.drop(columns=['date'])
    models = {}
    for i in range(1,8):
        if i ==1:
            exog_trial[f'sin1'] = np.sin(np.pi * exog_trial.index.dayofyear / 365.25)
            exog_trial[f'cos1'] = np.cos(np.pi * exog_trial.index.dayofyear / 365.25)
            exog_trial[f'sin1'] = np.sin(np.pi * exog_trial.index.dayofyear / 150)
            exog_trial[f'cos1'] = np.cos(np.pi * exog_trial.index.dayofyear / 150)
        else:
            for j in range(1,i+1):
                exog_trial[f'sin{j}'] = np.sin(j * np.pi * exog_trial.index.dayofyear / 365.25)
                exog_trial[f'cos{j}'] = np.cos(j * np.pi * exog_trial.index.dayofyear / 365.25)
                exog_trial[f'sin{j}'] = np.sin(j * np.pi * exog_trial.index.dayofyear / 150)
                exog_trial[f'cos{j}'] = np.cos(j * np.pi * exog_trial.index.dayofyear / 150)

        test = len(y) - 100 
        y_to_train = y.iloc[:test]    
        exog_trial_to_train = exog_trial.iloc[:test]
        arima_exog_trial_model = auto_arima(y=y_to_train, exogenous=exog_trial_to_train, seasonal=True, m=7, error_action='ignore',suppress_warnings=True) 
        model = f'{str(arima_exog_trial_model.order)}{str(arima_exog_trial_model.seasonal_order)} + Number of Exog: {i}'
        models[model] = arima_exog_trial_model.aic()

    min_aic_key = min(models.keys(), key=(lambda k: models[k]))
    print('Best model: {}, AIC: {}'.format(min_aic_key , models[min_aic_key]))
```

```
Output: Best model: (1, 1, 3)(2, 0, 1, 7) + Number of Exog: 5, AIC: 35689.827416808985
```

I used different windows, to test the results. I then calculate the absolute errors for each day of the prediction window.

<img src="/assets/img/forecasting/4_a.png">
<img src="/assets/img/forecasting/4_b.png" width="70%" height="70%">

----

<img src="/assets/img/forecasting/4_c.png" width="90%" height="90%">
<img src="/assets/img/forecasting/4_d.png" width="70%" height="70%">

----
<img src="/assets/img/forecasting/6_a.png" width='100%'>
<img src="/assets/img/forecasting/6_b.png" width="70%" height="70%">

----

<img src="/assets/img/forecasting/5_e.png" width='100%'>
<img src="/assets/img/forecasting/5_f.png" width="70%" height="70%">

We can see that the model does fairly well considering that I was unable to capture the full nature of the series. I have tried my best to add exogenous variables and account for seasonality in these but since these series have such high yield and variables that are unpredictable, it is difficult to get very precise results. Smaller time windows are easier to forecast compared to larger ones. This is very apparent when the model tries to forecast for 365 days. Firstly, 

Hence, This model would be appropriate for shorter forecasts.

Similarly, If we take example of California with the same process mentioned above, these are the parameters and results obtained:
<img src="/assets/img/forecasting/5_b.png">
<img src="/assets/img/forecasting/5_a.png" width="70%" height="70%">

----

<img src="/assets/img/forecasting/5_c.png">
<img src="/assets/img/forecasting/5_d.png" width="70%" height="70%">

---

## Neural Networks:
Even though this project work is not focued on neural networks, I wanted  to see how it would perform. Here are the results of it.

__LSTM:__ 
A common LSTM (Long-Short-Term-Memory) unit is composed of a cell, an input gate, an output gate and a forget gate. The cell remembers values over arbitrary time intervals and the three gates regulate the flow of information into and out of the cell.LSTMs can be used to model univariate time series forecasting problems.These are problems comprised of a single series of observations and a model is required to learn from the series of past observations to predict the next value in the sequence.
I have used previous day as the predictor for current day i.e, convert series to supervised learning, i.e. lagged time-series is input, next value is target. I have cross validated the dataset and then used the best fit model to make predictions. I did a parameterized search to get the best LSTM model.


<img src="/assets/img/forecasting/7_a.png">

---

<img src="/assets/img/forecasting/7_b.png">

__Dilated Causal Convolutional Network__: [Source](https://github.com/kristpapadopoulos/seriesnet/blob/master/seriesnet.py)
I have modified the original program to make step-wise daily predictions based on the data from previous date. 
The network contains stacks of dilated convolutions that allow it to access a broad range of history when forecasting, a ReLU activation function and conditioning is performed by applying multiple convolutional filters in parallel to separate time series which allows for the fast processing of data and the exploitation of the correlation structure between the multivariate time series. It is based on deep convolutional [WaveNet architecture](https://deepmind.com/blog/article/wavenet-generative-model-raw-audio). 

<img src="/assets/img/forecasting/8_a.png">


<img src="/assets/img/forecasting/8_b.png">

### LSTM vs SARIMAX vs Dilated CNN.
- Generally LSTM works better if we are dealing with huge amount of data and enough training data is available, while ARIMA is better for smaller datasets.
- ARIMA requires a series of parameters (p,q,d) which must be calculated based on data, while LSTM does not require setting such parameters. However, there are some hyperparameters we need to tune for LSTM.

The mean absolute errors in the plots were very similar for both New York and California. Even though the models are very different in their approach for prediction, 

<img src="/assets/img/forecasting/8_c.png">

---

<img src="/assets/img/forecasting/8_d.png">

For both regions, Dilated CNN performed the best. The season for this might be the availability of large data. I have ~1800 datapoints to be trained on. Due to that, the LSTM and CNN networks performed well. I have only compared these for daily predictions which do not have a higher margin of error. Predictions for larger time frames require further attention.

# Dash App
After finishing the modeling, I decided to include the functionality to an app integrated in dash. 

This Interactive app utilizes data from https://www.eia.gov/opendata/ API to pull live electricity consumption data.<br>
live temperature data from https://noaa.gov to pull electricity data. 


### Features/Tasks Completed:
- Live clock
- Update of news every 30 minutes
- Historical data for consumption of electricity for each region ( from 2016 ) upto hourly granularity with date sliders.
- Live temperature and electricity consumption data (updated each hour) 
- Forecasting of electricity consumption using SARIMA model (working on others) (daily, weekly, monthly consumptions)
- Cumulative consumptions using chloropeth maps (get consumption within a timeframe or individual days), polar chart(get consumption for each year for each region based on the seasons)
<br>

__Current progress (GIF):__

<img src="/assets/img/forecasting/demo.gif">


### Currently working on :
- integrating a MongoDB database for backend support to make API calls for Heroku Deployment





