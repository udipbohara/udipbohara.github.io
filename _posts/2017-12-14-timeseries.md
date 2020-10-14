---
layout: post
title:  "Time Series Forecasting"
info: "Restaurant Reservation Bot"
tech: "Python, Slack API"
type: B Company
img: "/assets/img/forecasting/demo.gif" 
img_dimensions: ["400","270"]
concepts: "Natural Language Processing, SOcial Media mining"
type: "blog"
link: "https://github.com/udipbohara/forecasting_dash"
---


  <div style="text-align: center">
  <i class="fa fa-code"></i> <a href="https://github.com/udipbohara/forecasting_dash">Code to this Repo</a>
  </div>
<br>
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




