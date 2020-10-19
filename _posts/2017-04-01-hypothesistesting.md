---
layout: post
title:  "Hypothesis testing"
info: "AB testing fundamentals mathematics"
tech: "python"
type: Toy 
img: "/assets/img/testingjupytermd/output_4_1.png" 
type: "blog"
usemathjax: true
---
<p> Exploring the Concepts of Hypothesis (A/B) Testing with Python</p>
<h1> Table of Contents: </h1>
<a href="#C1">Jump to 1. Introduction</a><br>
<a href="#C2">Jump to 2. T-Test from Scratch</a><br>
<a href="#C3">Jump to 3. Normality</a><br>
<a href="#C4">Jump to 4. Power Analysis</a><br>

<h2 id="C1">1. Introduction</h2>
Formally, __'A statistical hypothesis is a hypothesis that is testable on the basis of observed data modeled as the realised values taken by a collection of random variables.'__[ (Stuart, Ord, Arnold 2010) ](https://www.wiley.com/en-us/Kendall%27s+Advanced+Theory+of+Statistics%2C+Volume+2A%2C+Classical+Inference+and+the+Linear+Model%2C+6th+Edition-p-9780470689240). <br> Informally, A __statistical hypothesis__ is an assumption about population parameter(measurable characteristic of a population such as mean or standard deviation) and __Hypothesis testing__ is the formal procedure used by statisticians (traditionally) to accept or reject the hypothesis. This is a method of _statistical inference_.

To make this more intuitive, let us take a very simple example: <br>

Let us assume that you work for a e-Commerce website and want to increase the user engagement. We decide to initially run an experiment (A/B test) where we change the color of the checkout button for some users from red to green and see if it increases user engagement. For the sake of this example, we define a metric of interest as revenue-per-user. We perform necessary randomization and segment users to A(Control) and B(Treatment). For A users, we do not make any change whereas for B users we introduce the change and after the end of the experiment we observe that the revenue-per-user for B has gone up by 37% whereas for control it is 23% .



There are two types of hypothesis.
- Null Hypothesis : Denoted by $$H_{0}$$ is the hypothesis that there is no difference between the two samples means.
- Alternative Hypothesis : Denoted by  $$H_{A}$$ is the counterfactual of $$H_{0}$$. There is a difference between the two samples means.

<div>
    <center>
    <img src="/assets/img/testingjupytermd/fig-1.png" width="500"/>
        <br>
     <text><b> Fig 1: A simple A/B Framework for Online Controlled Experiments</b> <br> 
         <i> (Source: https://www.optimizely.com/optimization-glossary/ab-testing/)</i>
     </text>
    </center>
</div>


### Experiment setup:
We decide to take 1000 users for Control and 1000 for Treatment(Variation in the above figure). We run the experiment for a week and report back the results. For the sake of this example, we assume that the revenue per user in the experiment spans from 50 dollars to 100 dollars. We use our __key-metric as 'revenue-per-user'__ and report back the results.

So, our experiment includes the following.
- __Metric__ = Revenue-per-user
- __Parameter__ = Also known as the experimental variable that is thought to influence - Change of color of the checkout button.
- __Variant__ = Controls and Treatments (A and B) 
- __Randomization Unit__ = We use a pseudo randomization process to map the users to the variants.

Here, <br>
$$H_{0} : Mean_{Control} = Mean_{Treatment}$$ <br>
$$H_{A} : Mean_{Control} \neq Mean_{Treatment}$$


__For the purpose of this experiment I am simulating the results__


```python
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random 
from scipy.stats import norm
random.seed(30)

import matplotlib.pylab as pylab
params = {'legend.fontsize': 'medium',
          'figure.figsize': (12.5, 9),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)


def generate_data(sample_number):
    revenue_per_user_range = np.arange(50, 100, 0.012)

    #Note THIS IS NOT the Randomization unit. I am just assigning 1000 'users' with random revenue-per-user quantities from the above list
    control = random.choices(revenue_per_user_range, k=sample_number)
    treatment = random.choices(revenue_per_user_range, k=sample_number)
    
    print("{:25s}{:s}{:8.3f}".format("Mean of Control",":", np.mean(control)))
    print("{:25s}{:s}{:8.3f}".format("Mean of Treatment",":", np.mean(treatment)))
    print("{:25s}{:s}{:8.3f}".format("Variance of Control",":", np.var(control)))
    print("{:25s}{:s}{:8.3f}".format("Variance of Treatment",":", np.var(treatment)))
    
    return control,treatment

control,treatment = generate_data(1000)
```

    Mean of Control          :  74.331
    Mean of Treatment        :  75.439
    Variance of Control      : 210.644
    Variance of Treatment    : 208.542


We are going to compute Welch's T-test. We then plot a histogram for both Treatment and Control and it looks like this:


```python
plt.subplot(2,1,1)
plt.hist(control, edgecolor = "black")
plt.axvline(x= np.mean(control), color = "red", linewidth = 2, label='Mean ' + "%.2f" % np.mean(control))
plt.legend()
plt.title("REVENUE PER USER",pad=30)
plt.xlabel("Control"), plt.ylabel("Frequency")
plt.subplot(2,1,2)
plt.hist(treatment, edgecolor = "black")
plt.axvline(x= np.mean(treatment), color = "red", linewidth = 2, label='Mean ' + "%.2f" % np.mean(treatment))
plt.xlabel("Treatment"), plt.ylabel("Frequency")
plt.ylim(0, 120) 
plt.legend()
plt.tight_layout(pad=3.0)
```


    
![png](/assets/img/testingjupytermd/output_10_0.png)
    


We shall use relative distance as our reporting metric. 



```python
relative_difference = (np.mean(treatment) - np.mean(control)) / np.mean(control)
relative_difference * 100
```




    1.4908478181577318



Now, we have a relative increase of 1.49 percent when comparing the Control and Treatment. Our treatment population performed 'clearly' better and we maybe inclined to push it Treatment into production. But the question is, __is this difference really convincing or is it due to chance? Would the same results occur every time we took the measurements?__. This is where the t-test (one of the many tests) comes in. What it does is not only look at the mean but also the __variance__ (Separate Chapter on the importance of variance) of the two samples and helps compute a p-value. 
<br>
__p-value__ : P-value is the probability of obtaining the result at least as extreme as the sample result based on the assumption that the null hypothesis is true. Contextually, the probability of obtaining the result (difference of 1.49% between Control and Treatment) if the $H_{0}$ was true (there is no difference in the means of the Control and Treatment.)


```python
print("Welch's T test t statistic :", stats.ttest_ind(treatment,control,equal_var = False).statistic)
Welch's T test t statistic : 1.7107370172887062
#computing the p-value
print("Welch's T test p-value :", stats.ttest_ind(treatment,control,equal_var = False).pvalue)
Welch's T test p-value : 0.08728490410371684
```

Since the common threshold of statistitical significance is 0.05, Here we have failed to reject the null hypothesis. That means that the result we obtained was due to chance. Therefore, it is best for us not to push forward with production.

***

__One Common Misconception__: __Your data has to follow a normal distribution__. <br>

However, The t-test assumes that the means of the different samples are normally distributed; it does not assume that the sample population is normally distributed. By the central limit theorem, means of samples from a population with finite variance approach a normal distribution regardless of the distribution of the population.

## Illustration of Central Limit Theorem:

In probability theory, the central limit theorem establishes that in some situations, when independent random variables are added, their properly normalized sum tends toward a normal distribution, even if the original variables themselves are not normally distributed. To illustrate this, let us take 20 samples from each of the variants 


```python
# a list of sample mean
control_means,treatment_means =[],[]
samplesize = 20

#running 500 simulations of : taking 20 samples and calculating their means
for j in range(0,500):
    sample_control = random.choices(control, k=samplesize)
    control_means.append(sum(sample_control)/len(sample_control))
    sample_treatment = random.choices(treatment, k=samplesize)
    treatment_means.append(sum(sample_treatment)/len(sample_treatment))
```


```python

fig = plt.figure(figsize=(30, 10))

ax1 = plt.subplot(1, 2, 1)

plt.hist(control_means, color='grey', edgecolor = "black")
plt.axvline(x= np.mean(control_means), color = "blue", ls='--', linewidth = 4, alpha=0.5, label= f'Samples Mean: ' + "%.2f" % np.mean(control_means))
plt.axvline(x= np.mean(control), color = "red", linewidth = 2, label=f'True Mean: ' + "%.2f" % np.mean(control))
plt.title('Distribution of the sample means is approx. normally distributed')
plt.xlabel("Control")
plt.ylabel("Frequency")
plt.legend()

ax1 = plt.subplot(1, 2, 2)

plt.hist(treatment_means, color='grey', edgecolor = "black")
plt.axvline(x= np.mean(treatment_means), color = "blue", ls='--', linewidth = 4, alpha=0.5, label= f'Samples Mean: ' + "%.2f" % np.mean(treatment_means))
plt.axvline(x= np.mean(treatment), color = "red", linewidth = 2,label= f'True Mean: ' + "%.2f" % np.mean(treatment_means))
plt.xlabel("Control")
plt.ylabel("Frequency")
plt.legend()
```

![png](/assets/img/testingjupytermd/output_21_1.png)
    
```python
fig = plt.figure(figsize=(30, 10))

ax1 = plt.subplot(1, 2, 1)
# Generate some data for this demonstration.
data = control_means
mu, std = norm.fit(data)
plt.hist(data, bins=25, density=True, alpha=0.6, color='g')
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
plt.axvline(x= mu, color = "black", alpha=0.5,linestyle='--')
title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
ax1.title.set_text('Control Distribution')

#observed distribution
ax2 =plt.subplot(1, 2, 2)
ax2.title.set_text('Observed distribution')
data = treatment_means
mu,std = norm.fit(data)
plt.axvline(x= mu, color = "black", alpha=0.5,linestyle='--')
plt.hist(data, bins=25, density=True, alpha=0.6, color='b')
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
ax1.title.set_text('Treatment Distribution')
plt.show()
```

![png](/assets/img/testingjupytermd/output_22_0.png)
    

----
----
<br>
<br>
<h2 id="C2">2. T-Test from Scratch</h2>

___One of the more important aspects of AB testing is explainability to stake-holders. How do you interpret your results? In order to do so, it is necessary to have a fundamental grasp of how the algorithm works.
To clarify it we will be lookinag at two different ways to how we can reach our conclusions from the T Test___:
- P-value 
- Confidence Interval

__Generate Data:__

```python
def generate_data(sample_number):
    revenue_per_user_range = np.arange(50, 100, 0.012) 
    #Note THIS IS NOT the Randomization unit. I am just assigning 1000 'users' with random revenue-per-user quantities from the above list
    control = random.choices(revenue_per_user_range, k=sample_number)
    treatment = random.choices(revenue_per_user_range, k=sample_number)
    return control,treatment

control,treatment  = generate_data(1000)
```

## Stating the Hypothesis: 
It is important to state the hypothesis before the experiment. THat would lead to decrease in any bias that may come along.
Since we tested a new method, we will assume that it will lead to a difference. __Since, We do not know that it will lead to a positive change, we are going to assume that there will be a change in mean but not in which direction (better or worse).__ _This would depend on whatever the goal is, and the choice of one-tailed_ 

Then we formally, formulate our hypothesis.

Here, <br>
Our Null hypothesis is that, there is no difference in the means of the revenue-per-user between the Control and Treatment.<br>
Our Alternate hypotheis is that, there is a difference in the means of the revenue-per-user between the Control and Treatment.

$$H_{0} : Mean_{Control} = Mean_{Treatment}$$ <br>
$$H_{A} : Mean_{Control} \neq Mean_{Treatment}$$

<br>

Note: A one-tailed test is where you are only interested in one direction. <br>
_One-tailed test_, <br>  

$$H_{A} : Mean_{Control} > Mean_{Treatment}$$ <br> OR $$Mean_{Control} < Mean_{Treatment}$$

### Simple EDA

Doing some plots is always an easy way to visually get an idea of how the experiment went (specially when you have a larger number of variants. First we start with a KDE plot which is Kernel Density Plot(probability Density of a continuous variable).

__Note__: _While kernel density estimation produces a probability distribution, the height of the curve at each point gives a density, not a probability. A probability can be obtained only by integrating the density across a range._

We are also going to do a box plot as well as swarm plot to see the observations of the data. 



```python
f, (ax1, ax2) = plt.subplots(2, 1)
plt.subplots_adjust(hspace=0.6)

sns.boxplot(data=[control,treatment], orient="h", palette="Set2", ax=ax1)
sns.swarmplot(data=[control,treatment],orient="h", color=".05", size=3, alpha=0.7, ax=ax1)

#sns.catplot(data=[control,treatment],orient="h",  ax=ax1)
ax1.set_yticklabels(['Control','Treatment'])
ax1.set_ylabel('Variant')
ax1.set_xlabel('Count')
ax1.set_title('Box plot')

sns.kdeplot(data=control, color='r', shade=True, label = 'Control', ax=ax2)
sns.kdeplot(data=treatment, color='b', shade=True, label = 'Treatment', ax=ax2)
ax2.set_ylabel('Probability Density')
ax2.set_xlabel('Count')
plt.title('Visualizing the Probability Density')
plt.legend()
```
    
![png](/assets/img/testingjupytermd/output_8_1.png)
    
In Hypothesis testing, plots definitely help us give a bigger picture of the experiment. However, statistical testing is essential in assisting with decision-making.


### T-test

One of the most common methods of calculating significance in Controlled Online Experiments is __T-test__.

To reiterate from the previous chapter, T-test __compares the mean of two difference samples__ with consideration to the variance of them, ultimately giving the significance of how different they are from each other. Welch's T-test differs from Student's T-test in the assumption of variances. Student's t-test assumes that the variance is the same for both the sample whereas Welch's T-test assumes unequal variances. Here we focus on independent samples T test meaning that the data we observe is from difference samples (no cross over or leakage) _more on this later_.

<u>__Student's T test:__</u>

$$ t = \frac{\bar{X}_{1} - \bar{X}_{2} }{\sqrt{s^{2}({ \frac{1}{N_{1}} + \frac{1}{N_{2}})}}} $$

<u>__Welch's T test:__</u>

$$t = \frac{\bar{X}_{1} - \bar{X}_{2} }{\sqrt{\frac{s_{1}^{2}}{N_{1}} + \frac{s_{2}^{2}}{N_{2}}}} $$

where, <br>
- $$\bar{X} =  sample mean$$
- $$s = standard deviation$$
- $$N = sample size$$


__Practically, Welch's T-test is just equal to Student's T-test if the variances are equal.__ Personally, I would use Welch's t-test just because it catches the error if the variances are not equal.

Intuitively, The larger the t score, the more difference there is between groups. The smaller the t score, the more similarity there is between groups.

```python
control_mean = np.mean(control)
treatment_mean = np.mean(treatment)
control_stddev = np.std(control)
treatment_stddev = np.std(treatment)
N = 1000

relative_difference = (np.mean(treatment) - np.mean(control)) / np.mean(control)* 10
observed_t = (treatment_mean - control_mean)/ np.sqrt( (control_stddev)**2/N + (treatment_stddev)**2/N )


print('The relative improvement from control to treatment was: ', relative_difference)
print('The observed t-statistic from the test was:', observed_t)
```

    The relative improvement from control to treatment was:  0.14908478181577317
    The observed t-statistic from the test was: 1.7115930278588054


***

We can see that the t value we got from our T test is 1.321. To interpret this, let us see the T distribution. 

The t-distribution is symmetric and bell-shaped, like the normal distribution, but has heavier tails, meaning that it is more prone to producing values that fall far from its mean. 

Normally, where x is the sample mean, μ is the population mean, s is the standard deviation of the sample, n is the sample size, the degrees of freedom are equal to n - 1. 

__The t-distribution centers on zero because it assumes that the null hypothesis is true and is the approximation of a normal distribution. When the null is true, your study is most likely to obtain a t-value near zero and less liable to produce t-values further from zero in either direction__


The degrees of freedom in a statistical calculation represent how many values involved in a calculation have the freedom to vary. There are multiple ways to infer the value for this. A general rule of thumb is 'N-1' which accounts for the number of samples minus 1. But we are going to use the formula formula for it. 

$$df = \frac{\left [ \frac{s_{1}^{2} }{n_{1}} + \frac{s_{2}^{2} }{n_{2}} \right ]^{2}}{\frac{\left (\frac{s_{1}^{2}}{n_{1}}\right )^{2}}{n_{1}-1} + \frac{\left (\frac{s_{2}^{2}}{n_{2}} \right )^{2}}{n_{2}-1}}$$


This formula uses all the values we have already seen.


```python
top =  (control_stddev)**2/N + (treatment_stddev)**2/N 
bottom1 = 1/(N-1)*(control_stddev**2/N)
bottom2 = 1/(N-1)*(treatment_stddev**2/N)
df = top/(bottom1+bottom2)
df
```
    999.0000000000001



## p-value and critical value:

__p-value__:
As mentioned above, p-value is the probability of obtaining the statistic equal or extreme to the observation given the null hypothesis is true. 

__Critical value__:
Critical value can be calculated by using ppf : Percent point function/ quantile function. The quantile function, associated with a probability distribution of a random variable, specifies the value of the random variable such that the probability of the variable being less than or equal to that value equals the given probability. Contextually, ___The critical value is the t value associated with obtaining the p-value of the threshold we desire. Generally alpha = 0.05___


```python
# put it all in a nifty function
def t_statistic(tails, control,treatment,alpha):
    control_mean = np.mean(control)
    treatment_mean = np.mean(treatment)
    control_stddev = np.std(control)
    treatment_stddev = np.std(treatment)
    N1 = len(control)
    N2 = len(treatment)
    relative_difference = (np.mean(treatment) - np.mean(control)) / np.mean(control)* 10
    observed_t = (treatment_mean - control_mean)/ np.sqrt( (control_stddev)**2/N1 + (treatment_stddev)**2/N2 )
    top =  (control_stddev)**2/N1 + (treatment_stddev)**2/N2 
    bottom = 1/(N1-1)*(control_stddev**2/N1) + 1/(N2-1)*(treatment_stddev**2/N2)
    df = top/bottom
    #calculate the p-value 
    #1-the area under the curve (using, cumulative distribution function)
    #one-tailed
    if tails == 1:
        p=(1-t.cdf(observed_t, df))
    else:
    #two tailed
        p=(1-t.cdf(observed_t, df))*2
    #critical value can be calculated by using ppf : Percent point function (inverse of cdf — percentiles).
    critical_value = t.ppf(1.0 - alpha/2, df)
    return (round(observed_t,3),np.floor(df), round(p,3), round(critical_value,3))
```


```python
observed_t, df, p_value,critical_value = t_statistic(tails=2,control=control,treatment=treatment,alpha=0.05)

print('t={}, df={}, p_val={}, cv={}'.format(observed_t,df,p_value, critical_value))
```

    t=1.712, df=999.0, cv=0.087, p=1.962


### Visualizing the T -distribution: 
T-distribution is the probability density function of t statistic. Here we use the degrees of freedom as 999 which we calculated from the above formula.
As the t-distribution approaches larger degrees of freedom, it has the same values as the [z-distribution](https://www.dummies.com/education/math/statistics/using-the-z-distribution-to-find-the-standard-deviation-in-a-statistical-sample/)

The probability density function* for t-distribution is given by this __indimidating equation__ : 

$$f(t) = \frac{\Gamma(\frac{\nu+1}{2})} {\sqrt{\nu\pi}\,\Gamma(\frac{\nu}{2})} \left(1+\frac{t^2}{\nu} \right)^{\!-\frac{\nu+1}{2}}\!$$


where, <br>
$$\nu$$  is the number of degrees of freedom and <br>
$$\Gamma$$  is the gamma function. $$\Gamma = {\displaystyle \Gamma (n)=(n-1)!\ .}$$

_In probability theory, a probability density function*, or density of a continuous random variable, is a function whose value at any given sample in the sample space can be interpreted as providing a relative likelihood that the value of the random variable would equal that sample._

So the basic steps we have followed till now:
- Find the t-statistic
- Find the degrees of freedom

Now,
- We refer to the t-distribution table and find our critical thresholds
- Then compare it with our observed t-statistic 
- Find the corresponding p-value and accept or reject the null hypothesis 

__The Critical Thresholds that is conventionally chosen is 95% confidence interval or (p-value = 0.05)__

<div>
    <center>
    <img src="/assets/img/testingjupytermd/fig-2.png" width="500"/>
        <br>
     <text><b> Fig 2: T-distribution reference table</b> <br> 
         <i> (Source: https://www.sjsu.edu/faculty/gerstman/StatPrimer/t-table.pdf)</i>
     </text>
    </center>
</div>



## P-value method:

If the p-value is less than the significance value (alpha value) we can reject the null hypothesis.

The P value is the corresponding cumulative distribution function of the observed t-value.


```python
p=(1-t.cdf(observed_t, df))*2
print('The observed p-value from the test was:', p )
```

    The observed p-value from the test was: 0.08720702304438843



```python
def p_plot(df,observed_t,critical_value,alpha):
    df = df
    f, ax = plt.subplots(1)
    ax.set_ylim(bottom=0,top=0.41)
    plt.plot((observed_t, observed_t), (0, stats.norm.pdf(observed_t)), color='b', linestyle='-.')
    plt.plot((-observed_t, -observed_t), (0, stats.norm.pdf(observed_t)), color='b', linestyle='-.')
    plt.plot((-critical_value, -critical_value), (0, 0.2), color='r', linestyle='-.')
    plt.plot((critical_value, critical_value), (0, 0.2), color='r', linestyle='-.')

    x = np.linspace(t.ppf(0.001, df),
                    t.ppf(0.999, df), 100)

    #plot probability density function 
    plt.fill_between(x=x, 
                    y1= t.pdf(x, df) ,
                    facecolor='grey',
                    alpha=0.35)
    #upper threshold
    plt.fill_between(x=np.arange(-4,-critical_value,0.01),
                    y1= t.pdf(np.arange(-4,-critical_value,0.01), df),
                    facecolor='red',
                    alpha=0.5)              
    #lower threshold
    plt.fill_between(x=np.arange(critical_value,4,0.01), 
                    y1=t.pdf(np.arange(critical_value,4,0.01),df),
                    facecolor='red',
                    alpha=0.5)
    #texts
    plt.text(x=critical_value, y=0.2, size='large', s= r'$t_{crit}$ = '+ f'{critical_value}')
    plt.text(x=-critical_value-0.2, y=0.2, size='large', s= r'$t_{crit}$ = '+ f'-{critical_value}')

    plt.text(x=observed_t-0.7, y=t.pdf(observed_t, df), size='large', s= f"observed $t$ = {round(observed_t/2,3)}")
    plt.text(x=-observed_t, y=t.pdf(observed_t, df), size='large', s= f"observed $t$ = {round(-observed_t/2,3)}")

    plt.text(x=observed_t-0.7, y=0.02, size='large', s= f"$p$ = {round(p/2,3)}")
    plt.text(x=-observed_t, y=0.02, size='large', s= f"$p$ = {round(p/2,3)}")

    plt.text(x=critical_value, y=t.pdf(critical_value, df), size='large', s=r'$\alpha$ =' + f'{alpha/2}')
    plt.text(x=-critical_value-0.7, y=t.pdf(critical_value, df), size='large', s=r'$\alpha$ ='+ f'{alpha/2}')

    plt.text(x=-3.3,y=0.3,size='x-large', s =r'Reject Null Hypothesis if $p < \alpha$')
    plt.title('T- Distribution')

p_plot(df=999,observed_t=observed_t,critical_value=critical_value, alpha=0.05)
```

![png](/assets/img/testingjupytermd/output_29_0.png)
    


### Confidence Interval Method:
__If zero lies outside of the observed confidence interval of the observed distribution, we can reject the null hypothesis.__

### Using Normal Distribution

{% raw %} $${\displaystyle f(x)={\frac {1}{\sigma {\sqrt {2\pi }}}}e^{-{\frac {1}{2}}\left({\frac {x-\mu }{\sigma }}\right)^{2}}}$$ {% endraw %}
    
$$\mu$$  is the mean or expectation of the distribution (and also its median and mode), while the parameter 

$$\sigma$$  is its standard deviation
    
__Steps Involved__:
- Find the T statistic 
- Find the interval estimate for the difference for the Two Independent Random Samples <br>

95% Confidence Interval Estimate = $$ \Delta  \pm$$ margin of error
        
Where,
    __margin of error = T critical * Standard Error__

{% raw %} $$t_{\tfrac{\alpha }{2}} {\sqrt {{\frac {s_{1}^{2}}{n_{1}}}+{\frac {s_{2}^{2}}{n_{2}}}}}$$ {% endraw %}
    
If Zero (0) Lies in the Observed CI, Fail to reject the null hypothesis


```python
def normal_dist(x,mean,std):
    return 1/(std * np.sqrt(2 * np.pi)) * np.exp(- np.square(x - mean)/ (2 * np.square(std)))
np.mean(treatment) - np.mean(control)
```
    1.1081640000000448




```python
control_mean = np.mean(control)
treatment_mean = np.mean(treatment)
control_stddev = np.std(control)
treatment_stddev = np.std(treatment)

t_critical = 1.962

standard_error = np.sqrt( ( control_stddev**2/1000) + (treatment_stddev**2/1000) ) 

std = standard_error

d = np.mean(treatment) - np.mean(control)

upper_margin_error = round(t_critical*standard_error,3)
lower_margin_error = -upper_margin_error
null_ci = lower_margin_error,upper_margin_error 
high= d+(t_critical*standard_error)
low = d-(t_critical*standard_error)
observed_ci = round(low,3),  round(high,3)

print('Observed CI is ', observed_ci)
Observed CI is  (-0.162, 2.378)

control_mean = np.mean(control)
treatment_mean = np.mean(treatment)
control_stddev = np.std(control)
treatment_stddev = np.std(treatment)

t_critical = 1.962
standard_error = np.sqrt( ( control_stddev**2/1000) + (treatment_stddev**2/1000) ) 
std = standard_error
d = np.mean(treatment) - np.mean(control)
upper_margin_error = round(t_critical*standard_error,3)
lower_margin_error = -upper_margin_error
null_ci = lower_margin_error,upper_margin_error 
high= d+(t_critical*standard_error)
low = d-(t_critical*standard_error)
observed_ci = round(low,3),  round(high,3)
#from z table, the area is 0.9066 > 0.05 so null hypothesis 
df = 999
upper_margin_error = round(t_critical*standard_error,4)
lower_margin_error = - upper_margin_error
#stats.norm.pdf(np.arange(-4,-2,0.01)

fig = plt.figure(figsize=(30, 10))

ax1 = plt.subplot(1, 2, 1)
ax1.set_ylim(bottom=0,top=0.64)
ax1.title.set_text('Null distribution')
plt.axvline(x= 0, color = "black", alpha=0.5,linestyle='--')

plt.fill_between(x=np.arange(null_ci[0],null_ci[1],0.01), 
                 y1= normal_dist(np.arange(null_ci[0],null_ci[1],0.01),0,std) ,
                 facecolor='red',
                 alpha=0.35)

plt.text(x=-0.2, y=0.15,size='xx-large',s= "95% C.I")

plt.fill_between(x=np.arange(-2.5,2.5,0.01), 
                 y1= normal_dist(np.arange(-2.5,2.5,0.01),0,std) ,
                 facecolor='grey',
                 alpha=0.35)



plt.text(x=null_ci[0], y=normal_dist(null_ci[0],0,std), size='large', s= f'$-1.96\sigma = {null_ci[0]}$')
plt.text(x=null_ci[1], y=normal_dist(null_ci[1],0,std), size='large', s= f'$1.96\sigma = {null_ci[1]}$')

plt.plot((null_ci[0], null_ci[0]), (0, 0.15), color='r', linestyle='-.')
plt.plot((null_ci[1], null_ci[1]), (0, 0.15), color='r', linestyle='-.')

#observed distribution
ax2 =plt.subplot(1, 2, 2)
ax2.set_ylim(bottom=0,top=0.64)
ax2.title.set_text('Observed distribution')

plt.axvline(x= d, color = "black", alpha=0.5,linestyle='--')
plt.axvline(x= 0, color = "red", alpha=0.5,linestyle='--')
plt.fill_between(x=np.arange(observed_ci[0], observed_ci[1],0.01), 
                 y1= normal_dist(np.arange(observed_ci[0], observed_ci[1],0.01),d,std) ,
                 facecolor='blue',
                 alpha=0.35)
plt.fill_between(x=np.arange(-2,4,0.01), 
                 y1= normal_dist(np.arange(-2,4,0.01),d,std) ,
                 facecolor='grey',
                 alpha=0.35)


plt.text(x=observed_ci[0], y=normal_dist(observed_ci[0],d,std), size='large', s= f'$\Delta - 1.96\sigma = {observed_ci[0]}$')
plt.text(x=observed_ci[1], y=normal_dist(observed_ci[1],d,std), size='large', s= f'$\Delta + 1.96\sigma = {observed_ci[1]}$')

plt.plot((observed_ci[0], observed_ci[0]), (0, 0.15), color='b', linestyle='-.')
plt.plot((observed_ci[1], observed_ci[1]), (0, 0.15), color='b', linestyle='-.')

plt.xlabel(f'$\Delta$ distribution')
plt.show()


```
    
![png](/assets/img/testingjupytermd/output_36_0.png)
    


### Confidence Interval should not contain zero for the observed distribution


```python
f, ax = plt.subplots(1)
ax.set_ylim(bottom=0,top=0.7)

plt.axvline(x= 0, color = "black", alpha=0.5,linestyle='--')

plt.fill_between(x=np.arange(null_ci[0],null_ci[1],0.01), 
                 y1= normal_dist(np.arange(null_ci[0],null_ci[1],0.01),0,std) ,
                 facecolor='red',
                 alpha=0.35)


plt.fill_between(x=np.arange(-2.5,2.5,0.01), 
                 y1= normal_dist(np.arange(-2.5,2.5,0.01),0,std) ,
                 facecolor='grey',
                 alpha=0.35)

plt.fill_between(x=np.arange(-2,4,0.01), 
                 y1= normal_dist(np.arange(-2,4,0.01),d,std) ,
                 facecolor='grey',
               )

plt.fill_between(x=np.arange(observed_ci[0], observed_ci[1],0.01), 
                 y1= normal_dist(np.arange(observed_ci[0], observed_ci[1],0.01),d,std) ,
                 facecolor='blue',
                 )


plt.title('Zero lies in the 95% CI, so fail to reject Null Hypothesis')
```
    
![png](/assets/img/testingjupytermd/output_38_1.png)
    
### Confidence Interval Plot: 
This gives a better visual understanding of the plot

```python
def confidence_intervalPlot(list_ci, **kwargs):
        ci = list_ci[i]
        axes = plt.gca()
        axes.set_ylim([-0.5,1])
        plt.axvline(x= 0, color = 'grey', alpha=0.3, linewidth=5)    
        off = i * 0.2
        plt.yticks([])
    
        plt.plot((ci[0], ci[1]), (off, off), linewidth=3, linestyle='--', color=colors[i], label=legends[i])
        plt.plot((ci[0]+ci[1])/2, off, 'o', color=colors[i], markersize=15)
        
        plt.plot((ci[0], ci[0]), (off-0.05, off+.05), color=colors[i], linestyle='-')
        plt.plot((ci[1], ci[1]), (off-0.05, off+.05), color=colors[i], linestyle='-')
        
        plt.text(ci[0], off-0.05, s=f'${ci[0]}$', size = 13)
        plt.text(ci[1]-0.21, off-0.05, s=f'${ci[1]}$', size = 13)
        plt.legend()
        plt.title('Confidence Intervals')
```


```python
def confidence_intervalPlot(list_ci, legends):
    colors = ['r','b','g','c','m']
    for i in range(len(list_ci)):
        ci = list_ci[i]
        axes = plt.gca()
        axes.set_ylim([-0.5,1])
        plt.axvline(x= 0, color = 'grey', alpha=0.3, linewidth=5)    
        off = i * 0.2
        plt.yticks([])
        
        plt.plot((ci[0], ci[1]), (off, off), linewidth=3, linestyle='--', color=colors[i], label=legends[i])
        plt.plot((ci[0]+ci[1])/2, off, 'o', color=colors[i], markersize=15)
        
        plt.plot((ci[0], ci[0]), (off-0.05, off+.05), color=colors[i], linestyle='-')
        plt.plot((ci[1], ci[1]), (off-0.05, off+.05), color=colors[i], linestyle='-')
        
        plt.text(ci[0], off-0.05, s=f'${ci[0]}$', size = 13)
        plt.text(ci[1]-0.21, off-0.05, s=f'${ci[1]}$', size = 13)
        plt.legend()
        plt.title('Confidence Intervals')
```


```python
confidence_intervalPlot([null_ci, observed_ci],['Null', 'Observed'])

```
![png](/assets/img/testingjupytermd/output_42_0.png)
    
```python
confidence_intervalPlot([null_ci, observed_ci, (0.112,2.123)],['Null', 'Observed','sample1'])
```
![png](/assets/img/testingjupytermd/output_43_0.png)
    


----
----
<br>
<br>
<h2 id="C3">3. Normality</h2>

- We have the assumption that the distribution of the means $\Delta$ follows a normal distribution. 

### Why is it important?
- We have the assumption that the distribution of the means follows a normal distribution. This is a strong assumption to have and needs to be tested/ensured before further methods. Why is that so? Well, a 1.96 * Standard error estimates the population to be within 95% confidence interval in a normal distribution.

For example, let us consider an event where we are interested in knowing what is in the 5% of a probability distribution and we assume our distribution to be normal when it is not. You can clearly observe from the plot below that the areas under the curve are vastly different. __Any statistic we would calculate under the assumption would be wrong__


```python
def compare_plot(df,alpha):
    critical_value = stats.t.ppf(1.0 - alpha, df)
    f, ax = plt.subplots(1)
    ax.set_ylim(bottom=0,top=0.41)
    x = np.linspace(stats.t.ppf(0.01, df),
                    stats.t.ppf(0.99, df), 100)
    plt.plot(x, stats.t.pdf(x, df), color = 'r', label =f'$t-distribution$ with df 3 (not normal)')
    plt.fill_between(x=np.arange(-4.54,-critical_value,0.01),
                     y1= t.pdf(np.arange(-4.54,-critical_value,0.01), df),
                        facecolor='r',
                        alpha=0.5)     
    plt.text(x=(-critical_value-4.54)/1.7, y=0.04, color='red',size='large', s= f'$Area = {alpha}$')
    # plotting standard normal distribution
    x_axis = np.arange(-4, 4, 0.01)
    plt.plot(x_axis, stats.norm.pdf(x_axis,0,1), color ='b', label = f'$z- distribution$ (normal)')
    
    plt.fill_between(x=np.arange(-4,-critical_value,0.01), 
                y1= stats.norm.pdf(np.arange(-4,-critical_value,0.01)),
                facecolor='b',
                alpha=0.5)
    plt.axvline(x=-critical_value, color = "black", alpha=0.5,linestyle='--')
    plt.text(x=-critical_value, y=0.04, size='large',color='b', s= f'$Area = {round(stats.norm.cdf(-critical_value),3)}$')
    plt.legend()
    plt.title('Error due to incorrect Assumption of Normal Distribution')
```

__If we would have assumed our 't-distribution with 3 df' to be normal and had applied normal distribution formula to it to obtain the area of the curve (here, p-value), it would have been 0.061 or 6.1%. However, The true area would have been 0.11 which is 11%. Our calculation in the area would have been off by 4.9% which is a significant error and any inference that would follow would have serious flaw in it__ 


```python
compare_plot(3,0.11)
```
![png](/assets/img/testingjupytermd/output_6_0.png)
    


## Tests for Normality:

There are three common method sof checking for normality:
- __Graphical Methods :__ 
Easiest way to visually assess if there is normalitity. Histogram, box-plot and stem-and-leaf plot are the common ones.

- __Numerical Methods :__
Numerical methods include skewness and kurtosis coefficients.

- __Normality Tests :__
Inlcude more formal methods done with statistical inferences.

I am going to use these three distributions as the basis for this notebook.
These are sampled accordingly
- __data1:__ normal distribution
- __data2:__ gamma distribution with shape 1.1
- __data3:__ uniform distribution 


```python
#generate data

#from normal dist
data1 = np.array([np.random.normal() for _ in range(1000)])
#from gamma dist with 1.1 scale 
data2 = np.array([np.random.gamma(1.1) for _ in range(1000)])
#from uniform dist 
data3 = np.array([np.random.uniform() for _ in range(1000)])
```

## Graphical methods:

### 1) Histogram plot


```python
bins = 20
fig = plt.figure(figsize=(30, 10))
ax1 = plt.subplot(1, 3, 1)
plt.hist(data1, bins = bins, color='cornflowerblue')
plt.title('Data 1 - normal')
ax2 = plt.subplot(1, 3, 2)
plt.hist(data2, bins = bins, color='red')
plt.title('Data 2 -gamma with shape 1.1')
ax2 = plt.subplot(1, 3, 3)
plt.hist(data3, bins = bins, color ='green')
plt.title('Data 3 - uniform distribution')
plt.show()
```
![png](/assets/img/testingjupytermd/output_12_0.png)
    
### 2) Box plots: 
A boxplot has the following elements:

- Minimum : the lowest data point excluding any outliers.
- Maximum : the largest data point excluding any outliers.
- Median (Q2 / 50th percentile) : the middle value of the dataset.
- First quartile (Q1 / 25th percentile) : also known as the lower quartile qn(0.25), is the median of the lower half of the dataset.
- Third quartile (Q3 / 75th percentile) : also known as the upper quartile qn(0.75), is the median of the upper half of the dataset
- Outliers: (shown as green circles)

<div>
    <center>
    <img src="/assets/img/testingjupytermd/fig-3.png" width="500"/>
        <br>
     <text><b> Fig 1: Ideal boxplot and probability density function for a normal distribution</b> <br> 
         <i> (Source: https://towardsdatascience.com/understanding-boxplots-5e2df7bcbd51)</i>
     </text>
    </center>
</div>



```python
fig = plt.figure(figsize=(30, 10))

ax1 = plt.subplot(1, 2, 1)
ax1 = sns.violinplot(data=[data1,data2,data3], orient="h", palette="Set2")
ax1.set_yticklabels(['Data 1','Data 2','Data 3'])
ax1.set_title('Box Plots for the data distribution')


ax2 = plt.subplot(1, 2, 2)
ax2 = sns.boxplot(data=[data1,data2,data3], orient="h", palette="Set2")
ax2.set_yticklabels(['Data 1','Data 2','Data 3'])
ax2.set_title('Violin Plots for the data distribution (combination of box plots and kde plots)')

```
    
![png](/assets/img/testingjupytermd/output_15_1.png)
    


### 3) Q-Q Plots:

Quantile plot is a grpahical tool to assess if the set of data came from a desired distribution. It is a scatterplot created by two sets of quantiles against one other. If both sets of quantiles came from the same distribution, we would see the points forming a line that is roughly straight. For Normal distribution, with a mean of 0. The 0.5 quantile, or 50th percentile, is 0. Half the data lie below 0. The 0.95 quantile, or 95th percentile, is about 1.64. Similarly, 95 percent of the data lie below 1.64. 
__We are comparing theoretical 'ideal normal distribution quantiles' with the observed one. If they line up, we have a normal distribution.  If our data adheres to the theorized distribution, it will follow the standardized line__

___Note: In the KDE plot below, The y-axis in a density plot is the probability density function for the kernel density estimation. However, we need to be careful to specify this is a probability density and not a probability. The difference is the probability density is the probability per unit on the x-axis. To convert to an actual probability, we need to find the area under the curve for a specific interval on the x-axis.___ -https://towardsdatascience.com/histograms-and-density-plots-in-python-f6bda88f5ac0


```python
sm.qqplot(np.random.normal(1,2,1000),color='green',line='s')
plt.title('Ideal qq-plot for a normal distribution')
plt.tight_layout()
```  
![png](/assets/img/testingjupytermd/output_18_0.png)
    
```python
def qqPlot():
    pp = sm.ProbPlot(np.array(x4), fit=True)
    qq = pp.qqplot(marker='.', markerfacecolor='red', markeredgecolor='k')
    sm.qqline(qq.axes[0], line='45', fmt='k--')
```

```python
fig = plt.figure(figsize=(30, 10))
ax1 = plt.subplot(1, 2, 1)
ax1.set_title('KDE plot', fontsize=30)
ax1.set_ylabel('Density', fontsize=30)
ax1.set_xlabel('x', fontsize=30)
ax1.set_xlim(left=-4,right=4)
sns.distplot(data1,hist=False ,kde_kws={"color": "blue", "lw": 3}, label='Data 1')
sns.distplot(data2,hist=False, kde_kws={"color": "brown", "lw": 3}, label='Data 2')
sns.distplot(data3 ,hist=False, kde_kws={"color": "green", "lw": 3},label='Data 3')
plt.setp(ax1.get_legend().get_texts(), fontsize='22')
ax2 = plt.subplot(1, 2, 2)
sm.qqplot(data1,ax=ax2,color='blue',line='s')
sm.qqplot(data2,ax=ax2,color='brown',line='s')
sm.qqplot(data3,ax=ax2,color='green',line='s')
ax2.get_lines()[0].set_markerfacecolor('C0')
ax2.set_title('QQ Plot', fontsize=30)
ax2.set_ylabel('Observed Quantiles', fontsize=30)
ax2.set_xlabel('Theoretical Quantiles', fontsize=30)

plt.subplots_adjust(hspace = 200)
plt.tight_layout()
```

![png](/assets/img/testingjupytermd/output_20_0.png)
    
## Numerical methods:

### 1) Skewness:
Skewness is a measure of the asymmetry of the probability distrbution of a random variable about its mean. It tells us the amount and direction of departure from horizontal symmetry. If skewness is 0, the data are perfetcly symmetrical. If the skewness is negative, then the distribution is skewed to the left, while if the skew is positive then the distribution is skewed to the right 
Generally,
- If skewness is less than -1 or greater than 1, the distribution is highly skewed.
- If skewness is between -1 and -0.5 or between 0.5 and 1, the distribution is moderately skewed.
- If skewness is between -0.5 and 0.5, the distribution is approximately symmetric.


Intuitively, The greater the skew, the greater the distance betweren mode, median and the mean 
- For postiive skew: mode < median < mean
- For negative skew: mean < median < mode

__The skewness for a normal distribution is zero, and any symmetric data should have a skewness near zero.__

### 2) Kurtosis: 
measures the peakedness or flatness of a distribution.
Positive kurtosis indicates a thin pointed distribution.
Negative kurtosis indicates a broad flat distribution.

__The Kurtosis for a for a normal distribution would be close to 0.__

__Moment based calculation:__
Moments of a function are quantitative measures related to the shape of the function's graph. In statistics, 
Moments are a set of statistical parameters to measure a distribution
Without much detail, First moment of any distribution is the mean, second is the variance, third is the skewness and fourth is the kurtosis .


```python
def skew_kurtosis(data):
    n = len(data)
    mu = np.mean(data)
    std = np.std(data)
    skew= n/((n-1)*(n-2)) * np.sum((data-mu)**3)/std**3
    kurtosis = n*(n+1)/((n-1)*(n-2)*(n-3)) * np.sum((data-mu)**4)/std**4 -  3*(n-1)**2/((n-2)*(n-3))
    return skew,kurtosis
```


```python
skew1,kurtosis1 = skew_kurtosis(data1)
skew2,kurtosis2 = skew_kurtosis(data2)
skew3,kurtosis3 = skew_kurtosis(data3)

fig = plt.figure(figsize=(15, 8))

sns.distplot(data1,hist=False ,kde_kws={"color": "blue", "lw": 3}, 
             label=f'Data 1, $\Sigma = {round(skew1,3)}, \ K = {round(kurtosis1,3)}$')
sns.distplot(data2,hist=False, kde_kws={"color": "brown", "lw": 3}, 
            label=f'Data 2, $\Sigma = {round(skew2,3)}, \ K = {round(kurtosis2,3)}$')
sns.distplot(data3 ,hist=False, kde_kws={"color": "green", "lw": 3},
            label=f'Data 3, $\Sigma = {round(skew3,3)}, \ K = {round(kurtosis3,3)}$')

plt.title(f'Skewness( $\Sigma$ ) and Kurtosis ($\ K $) for our distributions')
plt.legend(fontsize=18)
plt.show()
```

![png](/assets/img/testingjupytermd/output_24_0.png)
    
## Normality Tests

### 1) Empericial Distribution Function (EDF) tets:

_Empirical Distribution Function : The EDF is calculated by ordering all of the unique observations in the data sample and calculating the cumulative probability for each as the number of observations less than or equal to a given observation divided by the total number of observations. t's an estimate of the population cdf based on the sample; specifically if you treat the proportions of the sample at each distinct data value and treat it like it was a probability in the population, you get the ECDF.
Simply put, Empirical cumulative distribution function is a cumulative sum of frequencies of observed $x_{i}$ divided by total sample size_

__The distinction is which probability measure is used. For the empirical CDF, you use the probability measure defined by the frequency counts in an empirical sample.__

__The empirical CDF is built from an actual data set. (rank of the ordered x values divided by the total number of values in the distribution). The CDF is a theoretical construct - it is what you would see if you could take infinitely many samples.__

These tests check for the normality of the data by comparing the Empirical Distribution Function (EDF) of the observed data with the cumulative distribution function (CDF) of a normal distrubtion to see if there is a good agreement between them. 

The most famous ones are:
- Kolmogorov-Smirnov Test:
- Lilliefors Test
- Anderson-Darling Test

In the graph below, we can see that the sample we drew out the normal distribution most closely resembles the ideal CDF.

#### a) Kolmogorov-Smirnov Test:
Measures the largest distance between the EDF and the theoritical function. If the difference is larger than the critical-value for the KS statistic, you reject the null hypothesis. Where,

$$T = sup_{x}\left | F^{*}(x) - F_{n}(x) \right |$$ <br>
where, $$sup_{x}$$ stands for suprenum (greatest), $$F^{*}(x)$$ is the hypothesized distribution function (contextually a normal distribution) and $$F_{n}(x)$$ is the EDF estimated based on the random sample (our observed data).
The KS test of normality,  $$F^{*}(x)$$ is taken to be a normal distribution with known mean, $\mu$ and standard deviation $\sigma$ . 

The KS test statistic is meant for testing:

$$H_{0} = F^{*}(x) = F_{n}(x)$$ for all x from $$ \infty$$ to $$\infty$$ (Contextually, The data follows a uniform distribution) <br>
$$H_{A} = F^{*}(x) \neq F_{n}(x)$$ for all x from $$- \infty to \infty$$ for atleast one value of x  (Contextually, The data does not follow a uniform distribution)


for KS test statistic: _http://www.real-statistics.com/statistics-tables/kolmogorov-smirnov-table/_


```python
#KS test for data with len >= 5                                    
def KS(data, plot=False):
    """
    args : data as an array
    returns : returns the result of the KS test 
    """
    assert len(data) >= 50, 'for this example, provide data with length > 50'
    data = np.sort(data)
    #from ks_crit table for alpha 0.05, and len(data) >=50, 
    ks_crit = 1.35810/np.sqrt(len(data))
    #edf
    edf_data =  np.arange(len(data)) / len(data)
    #null
    cdf_null = [stats.norm.cdf(x) for x in data]

    #two-tailed-test
    diff = max(np.abs(cdf_null - edf_data))
    
    if diff > ks_crit:

        text = 'Reject Null : t-crit: {:.3f} >  max difference: {:.3f}'.format(ks_crit, diff )
    else:
        text = 'Fail to Reject Null  t-crit: {:.3f} < max difference: {:.3f}'.format(ks_crit, diff )
        
    if plot:
        plt.plot(data, edf_data, c = 'red', label='Observed EDF')
        plt.plot(data, cdf_null, c='green', label='Expected CDF')
        plt.fill_between(data, edf_data,cdf_null, alpha = .5, label='Difference')
        plt.legend()
        plt.title(text)
            
    else:
        print(text)
    
   # return KS_result(ks_crit,)
```


```python
fig = plt.figure(figsize=(30, 10))
ax1 = plt.subplot(1, 3, 1)
KS(data1, plot=True)
ax2 = plt.subplot(1, 3, 2)
KS(data2, plot=True)
ax2 = plt.subplot(1, 3, 3)
KS(data3, plot=True)
``` 
![png](/assets/img/testingjupytermd/output_30_0.png)
    


#### b) Lilliefors Test:
It is the modificiation of the KS test. KS test is ideal for situations where the parameters of the hypothesized distribution are completely known. In contrast, the parameters for LF test are estimated based on the sample. It uses the same calculations as the KS test but the critical values are smaller, hence is less likely to show that data is normally distributed. In this case, Kolmogorov-Smirnov test based on the critical values in the Kolmogorov-Smirnov Table yields results that are too conservative. 

The Lilliefors test statistic are smaller so are less prone to type 1 errors.

$$F^{*}(x)$$ is taken to be a normal distribution with known mean, $$\mu = \bar{X}$$, the sample mean and standard deviation $$s^{2}$$ the sample variance. 

for Lilliefors test statistic: http://www.real-statistics.com/statistics-tables/lilliefors-test-table/


#### c) Anderson-Darling Test:

Anderson-Darling (AD) test is the most powerful EDF tests belongs to the quadratic class of the EDF statistic which is based on the squared difference. It is a modification of the Kolmogorov-Smirnov (K-S) test and gives more weight to the tails than does the K-S test hence increasing the sensitivity (true positive rate). This lowers the type 2 error (failing to reject the null hypothesis when the null hypothesis is false) . 

__Unlike KS test, the AD test is more concerned with the tails, (which would would be more of a concern of more extreme tails for example, risk mitigation)__



$$(F^{*}(x) - F_{n}(x))^{2}$$

A modified form of the equation is commonly adapted by D'Agnosti

{% raw %} $$A^{{*2}}=A^{2}\left(1+{\frac  {0.75}{n}}+{\frac  {2.25}{n^{2}}}\right).$$ {% endraw %}
   
Normality is rejected if {% raw %}$$A^{{*2}}$${% endraw %} exceeds 0.631, 0.752, 0.873, 1.034 or 1.159 at 10%, 5%, 2.5%, 1% and 0.5% significance levels, and the procedure is valid for sample sizes with atleast 8 data points. 

http://www.real-statistics.com/non-parametric-tests/goodness-of-fit-tests/anderson-darling-test/

{% raw %}$$A^{2}=-n-{\frac  {1}{n}}\sum _{{i=1}}^{n}(2i-1)(\ln \Phi (Y_{i})+\ln(1-\Phi (Y_{{n+1-i}}))).$${% endraw %}


```python
def anderson_test(x):
    
    '''
    args: distribution with len (x) > 8
    returns: returns the ad_statistic and the result at alpha levels (0.1,0.05,0.25,0.01,0.005)
    '''
    
    assert len(x) >= 8, "please have data size larger than 8'"
    x = np.sort(x)
    N = len(x)     
    xbar,s = np.mean(x),np.std(x, ddof=1) #sample metrics
    w = (x - xbar) / s
    
    cdf = np.log(norm.cdf(w))
    sf = np.log(1-norm.cdf(w)) #survival  #or norm.ppf(w)
    #ranks
    i = np.arange(1, N + 1)
 
    # A2 = -N - S where S = weighted difference between theoretical distribution and reverse theoretical distribution
    A2 = -N - np.sum((2*i - 1.0) / N * (cdf + sf[::-1]))
    
   # plt.plot(cdf, label='CDF of the distribution')
   # plt.plot(sf[::-1])
    ad_statistic = A2 * (1 + 0.75/N + 2.25/N**2)   
    sig_levels = [0.1,0.05,0.25,0.01,0.005]
    thresholds =[0.631, 0.752, 0.873, 1.034,1.159]
    reference = dict(zip(sig_levels,thresholds))
 
    for sig_level, threshold in reference.items():
        if ad_statistic > threshold:
            print(f'Reject Null hypothesis at alpha = {sig_level}, AD-Statistic = {round(ad_statistic,3)} > Threshold {threshold}')
        else:
            print(f'Fail to Reject Null hypothesis at alpha = {sig_level}, AD-Statistic = {round(ad_statistic,3)} < Threshold {threshold}')
        
    return ad_statistic

```


```python
print('data1 results: ')
ad_statistic1 = anderson_test(data1)

print('data2 results: ')
ad_statistic2 = anderson_test(data2)

print('data3 results: ')
ad_statistic3 = anderson_test(data3)
```

    data1 results: 
    Fail to Reject Null hypothesis at alpha = 0.1, AD-Statistic = 0.16 < Threshold 0.631
    Fail to Reject Null hypothesis at alpha = 0.05, AD-Statistic = 0.16 < Threshold 0.752
    Fail to Reject Null hypothesis at alpha = 0.25, AD-Statistic = 0.16 < Threshold 0.873
    Fail to Reject Null hypothesis at alpha = 0.01, AD-Statistic = 0.16 < Threshold 1.034
    Fail to Reject Null hypothesis at alpha = 0.005, AD-Statistic = 0.16 < Threshold 1.159
    data2 results: 
    Reject Null hypothesis at alpha = 0.1, AD-Statistic = 38.595 > Threshold 0.631
    Reject Null hypothesis at alpha = 0.05, AD-Statistic = 38.595 > Threshold 0.752
    Reject Null hypothesis at alpha = 0.25, AD-Statistic = 38.595 > Threshold 0.873
    Reject Null hypothesis at alpha = 0.01, AD-Statistic = 38.595 > Threshold 1.034
    Reject Null hypothesis at alpha = 0.005, AD-Statistic = 38.595 > Threshold 1.159
    data3 results: 
    Reject Null hypothesis at alpha = 0.1, AD-Statistic = 11.662 > Threshold 0.631
    Reject Null hypothesis at alpha = 0.05, AD-Statistic = 11.662 > Threshold 0.752
    Reject Null hypothesis at alpha = 0.25, AD-Statistic = 11.662 > Threshold 0.873
    Reject Null hypothesis at alpha = 0.01, AD-Statistic = 11.662 > Threshold 1.034
    Reject Null hypothesis at alpha = 0.005, AD-Statistic = 11.662 > Threshold 1.159



### 2) Regression and Correlation tests:
These are based on the ratio of two weighted least-quares estimate of scale obtained from other order statistics. The two estimates are the normally distributed weighted least quares estimates and the sample variance from other population.

The most famous one is :
- Shapiro-Wilk Test

#### a) Shapiro-Wilk Test:
This test was the first test that was able to detect departures from normality due to either skewness or kurtosis, or both. It has become the preferred test because of its good power properties. 

{% raw %}$${\displaystyle W={\left(\sum _{i=1}^{n}a_{i}y_{(i)}\right)^{2} \over \sum _{i=1}^{n}(y_{i}-{\overline {y}})^{2}},}$$ {% endraw %}
    
 #the denominator is the sum of squared errors $SS$, <br>
 
where, 
{% raw %}$$y_{{(i)}}$$ is the $$i^{th}$$ order statistic (the $$i^{th}$$-smallest number in the sample) <br>{% endraw %}
$$ \overline {y}$$ = sample mean, <br>
{% raw %} $${\displaystyle (a_{1},\dots ,a_{n})={m^{\mathsf {T}}V^{-1} \over C},}$${% endraw %}
{% raw %}$${\displaystyle C=(m^{\mathsf {T}}V^{-1}V^{-1}m)^{1/2}}$$,{% endraw %} <br>
{% raw %}$$m=(m_{1},\dots ,m_{n})^{{{\mathsf  {T}}}}$${% endraw %},   are expected values of the order statistics of independent and identically distributed random variables sampled from the standard normal dsitrubtion and $$V$$ is the covariance matrix of those statistics.

The Shapiro-Wilk test was originally limited to sample size less than 50. But, many iterations of reforms later it is suitable for usage for any $n$ in range $3 \leq n \leq 5000$ [approximation algorithm AS R94 (Royston, 1995) ](https://link.springer.com/article/10.1007%2FBF01891203)

__Note: 0 < $$W$$ < 1. Small values of W are evidence of departure from normality__

Alternatively, 


$$W = \frac{b^{2}}{SS}$$

where $$b$$ is the numerator from the above equation and $$SS$$ is simply the sum of squared errors.

### AS r94 algorithm: 

__This was implemented to accomodate sizes of 12 to 5000.__

Pseudocode: <br>

The following version of the Shapiro-Wilk Test handles samples between 12 and 5,000 elements.

- Sort the data in ascending order $$x_{1}, \leq ......   \leq  x_{n}$$
- Define the values $$M_{1},.... M_{n}$$, by , quantile_function of  $$\frac{i-0.375}{n+0.25}$$
- find $$m = \sum_{i=1}^{n}*M_{i}^{2}$$
- Set $$u = \frac{1}{\sqrt{n}}$$ and define the coefficients $$a_{1},.... a_{n}$$, <br>
where, <br>
$$a_{n} = -2.706056u^{5} + 4.434685u^{4} - 2.071190u^{3} - 0.147981u^{2} + 0.221157u + m_{n}m^{-0.5} $$
$$a_{n-1} = -3.582633u^{5} + 5.682633u^{4} - 1.752461u^{3} - 0.293762u^{2} + 0.042981u + m_{n-1}m^{-0.5} $$
and, <br>
$$a_{i} = \frac{m_{i}}{\sqrt{\varepsilon }}$ for 2 < i < n - 1,
$$a_{2} = -a_{n-1}$$, and $$a_{1} = -a_{n}$$
where, 
$$\varepsilon = \frac{  m - 2m_{n}^{2}   - 2m_{n-1}^{2} }{ 1 - 2a_{n}^{2}   - 2a_{n-1}^{2} }$$

  Interestingly, $$a_{i} = -a_{n-i+1}$$

- And, we have all the variables needed for. 

$${\displaystyle W={\left(\sum _{i=1}^{n}a_{i}y_{(i)}\right)^{2} \over \sum _{i=1}^{n}(y_{i}-{\overline       {y}})^{2}},}$$

- Amazingly, the statistic $$ln(1-W)$$ , is approximately normally distribution with, <br>
  mean , $$\mu = 0.0038915(lnn)^{3} - 0.083751(lnn)^{2} - 0.31082lnn - 0.5861$$ and <br>
  std, $$\sigma = e^{0.0030302(lnn)^{3}-0.082676lnn - 0.4803}$$

- Finally, we can get the test statistic using the standard normal distribution:
  $$z = \frac{ln(1-W)-\mu}{\sigma}$$
  
- If the p-value $$\leq \alpha$$, we reject the null hypothesis that the data is normally distributed.




```python
#Royston's Algorithm
shapiro_result = namedtuple('shapiro_result', ('statistic',
                                               'p_value',
                                               'significance_level',
                                              'result'))
def extended_shapiro(data):
    assert len(data) >= 12 and len(data) < 5000, "please have data size larger than 12 and less than 5000'"
    data = np.sort(np.array(data))
    
    #inverse of the cummulative standarized normal distribution.
    N = len(data)
    i = np.arange(1, N+1)
    
    M = norm.ppf((i-.375)/(N +.25))
    m = np.sum(np.square(M))
    
    a = np.empty(shape=(N,))
    u = 1/np.sqrt(N)
    
    a[N-1] = -2.706056*u**5 + 4.434685*u**4 - 2.071190*u**3 - 0.147981*u**2 + 0.221157*u + M[N-1] * m** -0.5
    a[N-2] = -3.582633*u**5 + 5.682633*u**4 - 1.752461*u**3 - 0.293762*u**2 + 0.042981*u + M[N-2]* m**-0.5
    a[0] = -a[N-1]
    a[1] = -a[N-2]
    
    idx = np.arange(2,N-2)
    eps = (m - 2*M[N-1]**2 - 2*M[N-2]**2)/ (1 - 2 * a[N-1]**2 - 2*a[N-2]**2 )

    a[idx] = M[idx] / np.sqrt(eps)
    
    W = (np.sum(a*data))**2/ np.sum((data-np.mean(data))**2)

    # for p value,
    mu = 0.0038915*(np.log(N))**3 - 0.083751*(np.log(N))**2 - 0.31082*np.log(N) - 1.5861
    sd = np.exp(0.0030302*(np.log(N))**2 - 0.082676* np.log(N) -0.4803)
    z = (np.log(1-W)-mu)/sd
    p_value = 1- norm.cdf(z)
    
    if p_value < 0.05:
        result = 'Reject Null Hypothesis'
    else:
        result = 'Fail to reject Null Hypothesis'
    
    return shapiro_result(W, p_value, 0.05, result)
```

Let us compare this formula with the library provided by scipy stats.


```python
from scipy.stats import shapiro

shapiro_test1 = stats.shapiro(data1)
print('Result from the library')
print(shapiro_test1)
print()

shapiro_test_scratch = extended_shapiro(data1)
print('Shapiro results for code written above')
print(shapiro_test_scratch)

```

    Result from the library
    ShapiroResult(statistic=0.9988949298858643, pvalue=0.8143659234046936)
    
    Shapiro results for code written above
    shapiro_result(statistic=0.9988953351122871, p_value=0.8146001462403867, significance_level=0.05, result='Fail to reject Null Hypothesis')


__Pretty darn close!__ :) 


```python
shapiro_test_scratch2 = extended_shapiro(data2)
print(shapiro_test_scratch2)
shapiro_test_scratch3 = extended_shapiro(data3)
print(shapiro_test_scratch3)
```

    shapiro_result(statistic=0.8433054008246615, p_value=0.0, significance_level=0.05, result='Reject Null Hypothesis')
    shapiro_result(statistic=0.954069401094459, p_value=0.0, significance_level=0.05, result='Reject Null Hypothesis')


# Which one is the best test?

Monte Carlo simulation has found that Shapiro–Wilk has the best power for a given significance, followed closely by Anderson–Darling when comparing the Shapiro–Wilk, Kolmogorov–Smirnov, Lilliefors and Anderson–Darling tests - "Razali, Nornadiah; Wah, Yap Bee (2011). "Power comparisons of Shapiro–Wilk, Kolmogorov–Smirnov, Lilliefors and Anderson–Darling tests"



# How to ensure normality for our statistical tests? Have enough sample sizes.

One of the strategies to observe the $$\Delta$$ to follow a normal distribution is to have enough sample sizes. Let us illustrate an example of observing data from a $$\gamma$$ distribution. 
Let us take 1, 3, 10, 100, 200 samples each from the population and see how the $\Delta$ looks like 


```python
population = np.random.beta(100, 0.1, size=10000000)*10

skew,_=skew_kurtosis(population)
print(skew)

    -6.227838471698967

plt.hist(population, bins = 20)
plt.xlabel('X')
plt.ylabel('Count')
plt.title('Histogram plot of the distribution')
```
 
![png](/assets/img/testingjupytermd/output_49_1.png)
    
```python
def plot_density(data, title, c):
    sns.distplot(data, color="cornflowerblue",
                            kde_kws = {'color':c, 
                           'linewidth':2, 'linestyle':'--'})
    plt.grid(False)
    plt.xlabel('Values')
    plt.ylabel('Density')
    plt.title(title, fontsize=22)
        
plot_density(population,r'Density Distribution of the $\beta$  Distribution '+ f'Skewness, $\Sigma = {round(skew,3)}$', 'black')
```


    
![png](/assets/img/testingjupytermd/output_50_0.png)
    



```python
def sample_generator():
    my_lists = [[],[],[],[],[],[]]
    sample_sizes = [1, 3, 10, 100,1000,100000]
    for i,sample_size in enumerate(sample_sizes):
        for j in range(1,sample_size+1):
            sample_variant = random.choices(population, k=500)
            mean = sum(sample_variant)/len(sample_variant)
            my_lists[i].append(mean)
    return my_lists
        
my_lists = sample_generator()
x1,x2,x3,x4,x5,x6 = my_lists[0], my_lists[1], my_lists[2],my_lists[3],my_lists[4],my_lists[5]
```


```python
# Import data
x1,x2,x3,x4,x5,x6= my_lists[0], my_lists[1], my_lists[2],my_lists[3],my_lists[4],my_lists[5]
# plot
#fig, axes = plt.subplots(1, 3, figsize=(20, 10), sharey=True)
f, ax = plt.subplots(1)
#ax.set_xlim(left=0.8,right=1.1)
sns.distplot(x1 , color="dodgerblue", hist=False, label=f'$n = 1$')
sns.distplot(x2 , color="deeppink", hist=False, label=f'$n = 3$')
sns.distplot(x3 , color="gold", hist=False, label=f'$n = 10$')
sns.distplot(x4 , color="green", hist=False, label=f'$n = 100$')
sns.distplot(x5 , color="black", hist=False, label=f'$n = 1000$')
sns.distplot(x6 , color="black", hist=False, label=f'$n = 10000$',kde_kws = {'color':'#8e00ce', 
                           'linewidth':3, 'linestyle':'--'})

plt.title(f'Distribution of $\mu$ approaches normal distribution with increased sample sizes')
plt.yticks([])
plt.xticks([])
plt.legend()
```
    
![png](/assets/img/testingjupytermd/output_52_2.png)
    
```python
x1,x2,x3,x4,x5,x6 = my_lists[0], my_lists[1], my_lists[2],my_lists[3],my_lists[4],my_lists[5]
fig = plt.figure(figsize=(30, 10))
ax1 = plt.subplot(1, 2, 1)

ax1.set_title(f'Distribution of $\Delta$ approaches normal distribution with increased sample sizes',fontsize=30)
ax1.set_ylabel('Density', fontsize=30)
ax1.set_xlabel('x', fontsize=30)
#ax1.set_xlim(left=-4,right=4)
sns.distplot(x1 , color="dodgerblue", hist=False, label=f'$n = 1$')
sns.distplot(x2 , color="deeppink", hist=False, label=f'$n = 3$')
sns.distplot(x3 , color="gold", hist=False, label=f'$n = 10$')
sns.distplot(x4 , color="green", hist=False, label=f'$n = 100$')
sns.distplot(x5 , color="black", hist=False, label=f'$n = 1000$')
sns.distplot(x6 , color="black", hist=False, label=f'$n = 10000$',kde_kws = {'color':'#8e00ce', 
                           'linewidth':3, 'linestyle':'--'})

plt.setp(ax1.get_legend().get_texts(), fontsize='22')
ax2 = plt.subplot(1, 2, 2)
sm.qqplot(np.array(x1),ax=ax2,color='blue',line='s')
sm.qqplot(np.array(x2),ax=ax2,color='brown',line='s')
sm.qqplot(np.array(x3),ax=ax2,color='green',line='s')
sm.qqplot(np.array(x4),ax=ax2,color='black',line='s')
sm.qqplot(np.array(x5),ax=ax2,color='green',line='s')
sm.qqplot(np.array(x6),ax=ax2,color='pink',line='s')

ax2.get_lines()[0].set_markerfacecolor('C0')
ax2.set_title('QQ Plot', fontsize=30)
ax2.set_ylabel('Observed Quantiles', fontsize=30)
ax2.set_xlabel('Theoretical Quantiles', fontsize=30)

plt.subplots_adjust(hspace = 200)
plt.tight_layout()
```


    
![png](/assets/img/testingjupytermd/output_53_0.png)
    



```python
fig = plt.figure(figsize=(30, 15))

ax1 = plt.subplot(2, 2, 1)
plot_density(population,r'Density Distribution of the $\beta$ Distribution','#004F2D' )
ax2 = plt.subplot(2, 2, 2)
sm.qqplot(np.array(population), ax=ax2,color='blue',line='s')
plt.title('QQ-plot for the Population')
ax3 = plt.subplot(2, 2, 3)
plot_density(x6, f'Density Distribution of the Sample Distribution ($n=1000$)', "black")
ax4 = plt.subplot(2, 2, 4)
sm.qqplot(np.array(x6), ax =ax4, color='black',line='s')
plt.title('QQ-plot for the sample-size = 10000')

plt.tight_layout()


```

### Practical Application:

Metrics such as __revenue metrics__, generally tend to have a high skewness associated with it. A simple way to effectively reduce the skewness is to introduce capped methods. 

"General recommendation for minimum number of samples needed for the average $$\Delta$$ to approach normal distribution is $$355s^{2}$$ where $$S$$ is the skewness coefficient, for |skewness| > 1." - [Seven Rules of Thumb for Web Site Experimenters](https://www.exp-platform.com/Documents/2014%20experimentersRulesOfThumb.pdf)

# Effective way to reduce skewness:

It is not always easy to exercise the above method to minimum sample sizes. Limitations in available sample population may occur due to plenty of reasons such as:
- Conflict with other on-going experiments (leading to sample mismatch ratio (SRM) )
- Unavailabilibity of the recommended size for sample population due to limit in the number of consumers availability
- Unavailability to fully account for covariances for variant triggering, hence having to reduce sample size. 
- Other issues to account for such as residual effects, cannibalism and so on.
- Multiple variants A/B/C/D which could exhaust the number of available users. 

One of the effective methods of reducing skewness would be to introduce capping. It would effectively reduce the required number of sample sizes. 


### Example:
Example: Say you work at a company and then you want to do A/B/C/D test on revenue-per-user, where A is the Control and the others are different treatments. You look at your population with metric of interest and see there is the following distribution with the recommended sample size per variant. 


```python
example_pop = np.random.beta(8, 0.01, size=10000000)*10
skew_example,_ = skew_kurtosis(example_pop)
min_samples = round(355 * (np.abs(skew_example))**2 )
```


```python

#plt.title(f'Recommended  Minimum sample required for each variant is {round(min_samples)}')
plot_density(example_pop,r'Density Distribution of Revenue-per-user  Distribution: '+ f'Skewness, $\Sigma = {round(skew_example,3)}$'
            , '#89023E')
plt.text(5,10, s = r'Recommended variant size ($n =355\Sigma^{2}) > $ ' + f'{min_samples}', fontsize=15)
```
    
![png](/assets/img/testingjupytermd/output_58_1.png)
    


As we can see the above dataset is highly skewed and the recommendation is minimum of 101271 for each variant. This is of concern as it would mean that we would need total of about 400,000 users. Let us cap the revenue per user at 10


```python
capped_pop = example_pop[example_pop<10]
skew_capped,_ = skew_kurtosis(capped_pop)
capped_samples = round(355 * (np.abs(skew_capped))**2 )
plot_density(example_pop,r'Density Distribution of Capped Distribution: '+ f'Skewness, $\Sigma = {round(skew_capped,3)}$'
            , '#B80C09')
plt.text(5,10, s = r'Recommended variant size ($n =355\Sigma^{2}) > $ ' + f'{capped_samples}', fontsize=15)
```
    
![png](/assets/img/testingjupytermd/output_60_1.png)
    


__We were effectively able to reduce the recommended number of each variant size by 73986 by reducing the skweness by capping the revenue-per-user to $10__

# References:

- [Power comparisons of Shapiro-Wilk, Kolmogorov-Smirnov, Liliefors and Anderson-Darling tets](https://www.researchgate.net/publication/267205556_Power_Comparisons_of_Shapiro-Wilk_Kolmogorov-Smirnov_Lilliefors_and_Anderson-Darling_Tests)


- [Goodness of fit Techniques -Agostino](https://www.google.com/books/edition/Goodness_of_Fit_Techniques/WLs6DwAAQBAJ?hl=en&gbpv=1&printsec=frontcover)

- [EDF Statistics for Goodness of Fit and Some Comparisons](https://www.jstor.org/stable/pdf/2286009.pdf?refreqid=excelsior%3Acb1eddf02a66e467e82f48c11291db36)

- [Approximating the Shapiro-Wilk W-test for non-normality](https://link.springer.com/article/10.1007%2FBF01891203)

- [Critical values tables for different tests](https://www.epa.gov/sites/production/files/2015-10/documents/monitoring_appendd_1997.pdf)

- [Shapiro-Wilk Expanded Test in Excel](https://www.real-statistics.com/tests-normality-and-symmetry/statistical-tests-normality-symmetry/shapiro-wilk-expanded-test/)

- [Seven Rules of Thumb for Web Site Experimenters](https://www.exp-platform.com/Documents/2014%20experimentersRulesOfThumb.pdf)


----
----
<br>
<br>

<h2 id="C4">4. Power Analysis</h2>


### Type I and Type II errors:

__Type I error__ is concluding that there is a significant difference between Treatment and Control when there is no real difference.<br>
OR<
is the rejection of a true null hypothesis (false positive).

__Type II error ($$\beta$$)__ is concluding that there is no significant difference between Treatment and Control when there actually is a difference.<br> 
OR
is the non-rejection of false null hypothesis (false negative).'


__Power__ is the probability of decting a difference between the variants(Treatment and Control) when there is a significant difference between them,
OR <br>

$$Power = P(reject\ H_{0} | H_{1} is\ true) = 1 - P(fail\ to\ reject H_{0} | H_{0}is\ false) = 1 - Type II\ Error (\beta)$$

<br> <br>
Alternatively, with the assumption of desired confidence level 95% (or $$\alpha = 0.05$$), <br>

$$Power_{\delta} = P(|T| \geq 1.96 | true\ diff\ is\ \delta)$$  <br>
Where, T = T statistic from T-test and delta is the difference in the means


__Minimum detectable effect__: It is the quantified amount/magnitude of the observation we are interested in. For example, when comparing between two variants (Control and Treatment), what is the minimum detectable effect? Answers, what is the quantified magnitude of the effect (Treatment) we introduced? These are generally based on heuristics or goals of your experiment. For example, you believe that there would be a conversion rate increment of 3% in your treatment group. 


```python
x = np.arange(-5,7.5, .01)
y1 = stats.norm.pdf(x)
y2 = stats.norm.pdf(x,loc=3.5, scale=1.2 )  

plt.plot(x, y1)
plt.plot(x,y2)


plt.axvline(x= 2, color = "black", linewidth = 2,linestyle='--')

#type1
type1x = np.arange(2,7.5, .01)
plt.fill_between(type1x, stats.norm.pdf(type1x),label='Type 1 Error'  )

#type2
type2x = np.arange(-2,2,0.01)
plt.fill_between(x=type2x, 
                 y1= stats.norm.pdf(type2x,loc=3.5, scale=1.2) ,
                 facecolor='red', label='Type 2 Error',
                 alpha=0.8)

plt.fill_between(x=np.arange(-5,2,0.01), y1= stats.norm.pdf(np.arange(-5,2,0.01)) ,
                 facecolor='grey', label='Correct Inference',
                 alpha=0.1)

plt.fill_between(x=x, y1= stats.norm.pdf(x,loc=3.5, scale=1.2) ,
                 facecolor='grey',alpha=0.1)


plt.legend()
plt.grid(False)
plt.yticks([])
plt.xticks([])
plt.ylim([0,0.41])

plt.text(x=-0.8, y=0.15, s= "Null Hypothesis",fontsize=12)
plt.text(x=2.5, y=0.13, s= "Alternative Hypothesis", fontsize=12)
plt.text(x= 2.1, y = 0.38, s = r't-critical / $$\alpha$$',fontsize=12)

plt.title('Type I and Type II error in Hypothesis Testing', fontsize=20)
```

    
![png](/assets/img/testingjupytermd/output_4_1.png)
    


### Effect size: 

It is the quantified amount/magnitude of the observation we are interested in. For example, when comparing between two variants (Control and Treatment), what is the effect size? Answers, what is the quantified magnitude of the effect (Treatment) we introduced? 

#### Cohen's d:
Cohen's d is one of the popular methods of calculating the effect size.
Cohen's $$d$$ is the sandardized mean difference,

$$d = \frac{\mu_{1}-\mu_{2}}{\sigma}$$

__Assumption of equal variances__ 

The general guideline for the magnitude of the effect size is:
- Small effect size: $$0 <  d < 0.2$$
- Medium effect size: $$0.2 <  d < 0.8$$
- Large effect size: $$d > 0.8$$


## Power Analysis

We essentially want the Power of our statistical test to be as high as possible. 
Power will tend to be greater when:
- The effect size is large
- The sample size is large
- The variance of the variants are small
- The significance level ($$\alpha$$) is high. 


### 1) Effect size is large:
If the effect of our treatment is large, we are bound to have higher power. (The probability to reject the null hypothesis increases with the increase in effect size).

It is a very important limitation in hypotehsis testing as it makes a relative comparison: The size of the treatment effect relative to the difference expected by chance.
If the standard error is very small, then the treatment effect can also be very small and still be bigger than chance.
Therefore, a significant effect does not necessarily mean a big effect.
Also, if the sample size is large enough, any treatment effect, no matter how small, can be enough for us to reject hypothesis. 


```python
def plot_normal(mean,std,fill_between=False, **kwargs):
    variance = np.square(std)
    x = np.arange(-5,15,.01)
    plt.axvline(x= mean, color = "black", alpha=0.5,linestyle='--') 
    f = np.exp(-np.square(x-mean)/2*variance)/(np.sqrt(2*np.pi*variance))    
    if fill_between:
        plt.fill_between(x, f, color='red',alpha=0.5)
    plt.plot(x,f, **kwargs)
    
def compare(mean1,mean2,std,txty):
    d = (mean1 + mean2) / std
    plot_normal(mean1,std,  color='cornflowerblue', label='Control')
    plot_normal(mean2,std, fill_between = True, linestyle='--', color='red',label='Treatment')
    plt.text(x=(mean1+mean2)/2-0.5, y= txty,s = f'$$d={d}$$', fontsize=15)
    plt.grid(False)
    plt.legend()
    #plt.yticks([])
    
fig = plt.figure(figsize=(30, 10))
ax1 = plt.subplot(1, 2, 1)


compare(3,5,0.5, txty=0.80)
plt.xlim([-4,12])
plt.ylim([0,0.9])
plt.title(f'$$\Delta = 2$$ , $$\sigma = 0.5$$',fontsize=25)

ax2 = plt.subplot(1, 2, 2)
compare(3,5,1, txty=0.41)
plt.xlim([-4,12])
plt.ylim([0,0.45])
plt.title(f'$$\Delta = 2$$ , $$\sigma = 1$$', fontsize=25)

plt.suptitle('Cohen’s $$d$$: smaller $$\sigma$$ = larger effect size.', fontsize=30)

```
    
![png](/assets/img/testingjupytermd/output_9_1.png)
    

### Power Analysis using Monte-Carlo simulation:

We normally want the desired power to be 0.8 ie. the probability of rejecting the null hypothesis when the alternative hypotheis is true to be 0.8


```python
def plot_simulation(x,y,title,xaxis):
    plt.ylim([0,1.1])
    plt.xlim([0, y.max()+y.min()])
    plt.plot(y, x, c='black', linewidth='2')
    plt.scatter(y, x, c='r', s=40,alpha=0.5)
    plt.ylabel('Power')
    plt.xlabel(xaxis)
    plt.title(title)
    plt.axhline(y =0.8, color = "black", alpha=0.5,linestyle='--')
    plt.axhline(y =0.8, color = "black", alpha=0.5,linestyle='--')
    plt.grid(False)
```


```python
#this is going to be our sampling paramter. 
#this is the function we wrote in 2_-testfromScratch.
from utils import t_statistic

from scipy.stats import norm, binom

sample_mean = 25
sample_sd = 10 #we are using larger standard deviation so that the power is not high enough early.
sample_data = norm.rvs(loc=sample_mean, scale=sample_sd, size=20000)
```

## Using varied sample sizes


```python
sample_sizes = range(50, 10000, 100) # Sample sizes we will test over
alpha = 0.05 
simulations = 1000 #number of simulations per iteration
relative_effect = 1.03 #fixed relative effect 

power_dist = np.empty((len(sample_sizes), 2))
for i in range(0, len(sample_sizes)): 
    N = sample_sizes[i]
    
    control_data = sample_data[0:N]
    variant_data = control_data * relative_effect 
    
    sig_results = []
    for j in range(0, simulations):
        
        # Randomly allocate the sample data to the control and variant   
        control_sample = np.random.choice(control_data, N//2, replace=False)
        variant_sample =  np.random.choice(variant_data, N//2, replace=False)
        
        # Use Welch's t-test, make no assumptions on tests for equal variances
        _,_,p_val,_ = t_statistic(2, control_sample,variant_sample,alpha)
        
        #We are assuming the null hypothesis to be true, hence power would be the times we would get alpha value
        #we are rejecting the null hypothesis.
        sig_results.append(p_val <= alpha) 
    
    #getting the probability of rejecting the null hypothesis.
    power_dist[i,] = [N, np.mean(sig_results)] 
    
power, sample_sizes = power_dist[:,1], power_dist[:,0]
```


```python
plot_simulation(power,sample_sizes,'Power increases with increase in sample sizes','sample sizes')
```


    
![png](/assets/img/testingjupytermd/output_16_0.png)
    


Size-Lehr's Equation:

__Rule of Thumb for number of sample sizes__


$$n = \frac{16}{\Delta ^{2}}$$

where
$$\Delta = \frac{\mu_{0} - \mu_{1}}{\sigma}$$


# References/Resouces:

[Visualize Cohen's Effect size:](https://rpsychologist.com/d3/nhst/) <br>
[Statistical Rules of Thumb](http://www.vanbelle.org/chapters/webchapter2.pdf)
