---
layout: post
title:  "Time Series Forecasting"
info: "Restaurant Reservation Bot"
tech: "Python, Slack API"
type: B Company
img: "/assets/img/forecasting/demo.gif" 
img_dimensions: ["400","270"]
concepts: "Econometric, Time-Series Modeling, Neural Networks"
type: "blog"
link: "https://github.com/udipbohara/forecasting_dash"
usemathjax: true
---


  <div style="text-align: center">
  <i class="fa fa-code"></i> <a href="https://github.com/udipbohara/forecasting_dash">Code to this Repo</a>
  </div>
<br>

# Part 1: EDA 

This app takes EIA data from all the regions from the US for electricity demand and uses econometric modeling for forecasting for the following regions

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


Here is the Frequency plot for the data. As we can see the mean is around 18041 

<img src="/assets/img/forecasting/1_b.png" width="60%" height="60%"/>


If we plot the KDE for each year, we can see that there is a small shift towards lower consumption in 2019.
This could mean that there are investments in alternative source of energy or could just be an anomaly.
<img src="/assets/img/forecasting/1_c.png" width="60%" height="60%"/>

It is interesting to see how the electricity demand is stratified by different temporal changes. 

<img src="/assets/img/forecasting/1_d.png" width="70%" height="70%">
We can see that the highest consumptions are during July and August which can be attributed to Summer months and the usage of Air-conditioining increases the demand.


<img src="/assets/img/forecasting/1_e.png">
We can clearly see that the highlest consumption is around 6:00PM EST for the weekdays. Whereas the weekends has comparatively lower demand. This can be attributed to holidays.

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

Typical ML methods are characterised by nonparametric modelling and have very relaxed assumptions while Time-series models are "tightly-parametric".

Stationary: It is a key assumption for these models. A stationary time series is one whose properties do not depend on the time at which the series is observed.14 Thus, time series with trends, or with seasonality, are not stationary — the trend and seasonality will affect the value of the time series at different times



Tests for stationarity:
Unit root test: (Augmented) Dickey Fuller
Stationarity test: KPSS test

Concept of Unit-root tests:
- Null hypothesis: Unit-root (is a stochastic trend in a time series, sometimes called a “random walk with drift”;)
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

Hence, for our prediction of basic ARMA(1,1) would be:

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


Refering to the plot the first inuition would be a SARIMA(4,1,1)

Interactive Dash app visualizing and forecasting electricity consumption in the US through econometric modeling and FB prophet (LSTM etc).

This Interactive app utilizes data from https://www.eia.gov/opendata/ API to pull electricity consumption data.<br>
temperature data from https://noaa.gov to pull electricity data 

To run the app:
- run app.py after installing requirements.txt to your venv
















### Features/Tasks Completed:
- Live clock
- Update of news every 30 minutes
- Historical data for consumption of electricity for each region ( from 2016 ) upto hourly granularity with date sliders.
- Live temperature and electricity consumption data (updated each hour) 
- Forecasting of electricity consumption using SARIMA model (working on others) (daily, weekly, monthly consumptions)
    - The ARIMA model uses exogenous variables (temperature for better prediction)
- Cumulative consumptions using chloropeth maps (get consumption within a timeframe or individual days), polar chart(get consumption for each year for each region based on the seasons)
<br>

__current progress (GIF):__

<img src="/assets/img/forecasting/demo.gif">


### Currently working on :
- integrating a MongoDB database for backend support to make API calls for Heroku Deployment
- integrating regions for effectively predictions.




