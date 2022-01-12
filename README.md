Florida Institute of Technology
















Final Project: An Attempt at Predicting Cryptocurrency Prices using Neural Networks







David E. Nieves-Acaron

Dr. Ryan White

MTH 4320 
12/14/2021

**Abstract**

The basic idea behind this work is that young crypto-currency projects are based off social media exposure to spread the news and draw more people in. This project seeks to find a way to predict the trends of a cryptocurrency project. Note that due to complications with obtaining Social Media information about more obscure cryptocurrency projects, only Bitcoin is being considered due to the ample amount of Tweet information there is about it. Regardless, it is apparent from watching the state of the cryptocurrency markets and its reactions to influential figure’s tweets, that one could feasibly attempt predicting the trends of cryptocurrency prices from social media. This work attempts a naive version of this by trying out different Neural Network architectures such as Recurrent Neural Networks (RNN’s) and Long Short-Term Memory (LSTM’s) and attempting to predict future price activity. Unfortunately, due to difficulties in obtaining detailed social media information about obscure cryptocurrencies, the work was limited to using social media and price activity data about Bitcoin rather than any other cryptocurrencies. Moreover, it also details the effort needed to process the Bitcoin price and social media data into feature vectors that can be processed by Neural Networks, the ingestion of this data into a visualization engine known as Kibana, and finally, the logging of the performance of the models into a service known as Weights and Biases.

**Introduction**

As anyone who has participated in the launch of a new cryptocurrency project may know, social media exposure is crucial for getting new members to join the project and invest in the asset, be it an initial coin offering or some other form of blockchain-utilizing technology. In this case, this work will focus on initial coin offerings (ICO’s) due to there being plentiful data and records with regards to most of them, and due to how direct the investment is. What is meant by the latter is the fact that users invest directly into the project via a currency conversion (usually Ethereum or Binance into the newly launched coin) and thus directly affect the price rather than some other form of investing such as social media. Coincidentally, however, the reason why social media matters in the first place is that because a lot of smaller cryptocurrencies do not demonstrate any meaningful technical contributions or any real future for the project. In other words, almost anyone who is investing in these types of cryptocurrencies knows that the project may carry momentum and yield Xprofits almost exclusively on the basis of social media exposure and on “how much the next guy will put in”. For that reason, the main line of reasoning for the inputs to any type of model trying to predict future movements in the price of the asset is not just the actual statistics regarding the asset (prices, lows, highs, openings, closings, specialized indicators, etc...) but also social media metrics such as how many combined views the social media account of the project obtained. It is for this very reason that whenever joining a new project’s telegram page (one of the most common ways of keeping in the know-how with the status of the coin), one will likely be asked to share, like, and follow al the posts and channels related to the project. Some other projects even reward would-be investors by directly handing out a quantity of the coins depending on how much the user has participated in sharing the project’s posts in an activity known as an “airdrop”.

In addition to social media, one other feature of interest could be how many exchanges the cryptocurrency gets listed on, since this too drives the popularity of the coin up and therefore the price. Moreover, some cryptos from some blockchains are more likely to be seen as more “suspect” due in part to the nature of how easy it is to create a project for that blockchain. Specifically, one good example which demonstrates a general observation by the author rather than an actual law is that cryptocurrencies hosted in the Ethereum blockchain are more likely to be trusted than cryptocurrencies hosted in Binance Smart Chain (BSC) due to the high gas fees associated with the Ethereum blockchain which serve to dissuade some would-be scammers. Note that this observation is to be treated as a trend observed by the author, and that scams in Ethereum still occur and are likely relatively common, though probably not as common as in Binance Smart Chain. The reason why it matters whether a cryptocurrency is more likely to be a scam or not is that it can affect how popular it might become and could thus serve as an input to whatever model we use to predict future trends for the cryptocurrency.  This could be as simple as having a one hot for whichever blockchain is being used. Thankfully, there are not that many prominent blockchains and it is likely that only three inputs are really required for this data:

Bitcoin, Ethereum,BinanceSmartChain,Other

Note that although the concept of social media exposure is extremely important for young Cryptocurrency prices, it still affects even the biggest of cryptocurrencies such as Ethereum and Bitcoin. That can be considered (among other things) a significant flaw of this work’s approach. 

**Literature Review**

Trying to predict the movements of markets such something that many great minds have dedicated their efforts to.  Some examples include predicting Foreign Exchanges, local exchanges, the stock market, and more prominently, the market of cryptocurrencies. Due to Bitcoin’s special place in the cryptocurrency world, (especially due to its place as the highest valued cryptocurrency) there have been researchers attempting to predict how the price of Bitcoin will change. One such group of researchers, namely Sara Abdali and Ben Hoskins from Stanford University, attempted to predict the price of Bitcoin by using sentiment analysis of tweets solely related to Bitcoin. Their approach consisted of “simple “ models (Naive Bayes and Support Vector Machines) as well as a more complicated BERT transformer model. 

These researchers also showcase some related work such as “Sentiment analysis on stock social media for stock price movement prediction” (CITE), which used a similar approach of a simple SVM and a neural network for an average accuracy of 56% (with the former researchers considering this to be a benchmark). With regards to data collection, the authors also used the Twitter API to collect tweets with hashtags mentioning Bitcoin (very similar to my approach) and noted the API’s limitations of only being able to collect a few thousand tweets per day. The authors also noted the novelty of using a language processing model for use in sentiment analysis for financial assets and securities. The inputs for the model were the actual 

Another study sought to measure how media sentiment, comprised largely blog posts and their associated sentiment, interacts with Bitcoin prices and found that there is a tendency for investors to overreact on news for short time periods (Karalevicius et al). One of its more unique features is the use of psycho-semantic dictionaries.

On the less computationally expensive side, another paper found that the Twitter sentiment ratio correlated very positively with Bitcoin prices. The sentiment analysis was less computationally expensive due to its use of support vector machines (SVMs). Surprisingly, this study also found a relationship between Bitcoin prices and other features such as the number of Wikipedia search queries, the exchange rate between the United States Dollar and the Euro (in this case a negative relation), the number of Bitcoins in circulation (more of a long-term effect) and finally, the Standard and Poor's 500 stock market index (also a negative relation)( Georgoula et al). 

Another paper attempted to use what is known as a stochastic neural network, which is based on the random walk theory used in finance. In summary, what they attempt to do is add some randomness to the neural network by a quantity they refer to as “reaction” to better match the random movements seen in real life. Note that this stochastic element regardless makes sure to take into account previous time steps’ activations, thus making it akin to how RNN’s and LSTM’s work, although in a more stochastic fashion. Considering the results obtained by testing against Ethereum, Bitcoin, and Litecoin data, it seems to be a promising concept (P. Jay et al).

Relating more towards the work at hand, another study performed a comparison of different Machine Learning methods such as WaveNets, Recurrent Neural Networks, Support Vector Machines, Random Forests, ARIMA, and Long Short-Term Memory. I found this study to be the most informative for the purposes of this report since it compared methods from many different fields (Finance, Machine Learning, etc...) to provide a summary of their results. The data that was used for this work was very similar to the one used in this project (close, open, high, low prices, etc...). Curiously enough, they chose to exclude the years 2017 and 2018 due to them being considered outliers (lots of variance perhaps due to the crash throughout the winter). Their hyperparameter tuning in part consisted of random searches. For their LSTM models, they used the Adam optimizer as well as three layers (they tested more, but initial results did not seem too promising) and the hyperbolic tangent function.   Surprisingly, the LSTM model proved to be the second worst performing model, with SVR (support vector regression) and ARIMA as the best performing models (L. Felizardo et al). This helps to feed some scepticism on my part as to neural networks being a good approach for these chaotic systems. 

**Outputs**

The outputs of the model were simply the logarithm of the opening, closing, high, and low prices. These were chosen like they were so as to avoid exploding gradients. Some more work could be done into choosing a way to standardize them. 

**Model**

The three main models used are Recurrent Neural Networks (RNNs), Long Short-Term Memory (LSTMs), and finally, Transformers. The first model will serve as a basis for the other results. It is essentially a neural network that allows previous outputs to be used as inputs while having hidden states. As opposed to a regular feed forward neural network, this type has the advantage of taking into account historical information during the calculation of the output. Some of their disadvantages include the long computation times required for training them, and the loss of information from many time steps back due to the gradient usually vanishing (exploding gradients can happen with this type of model) (Amidi et al). On the other hand, Long Short-Term Memory units seek to manage the vanishing gradient problem by ... Finally, Transformers are types of models that make heavy use of attention mechanisms, which generally take into account inputs that are farther back than what would be possible with LSTMs and RNNs.

Training

The data obtained was from Kaggle user Kash who works as a Data Scientist at Kaleidofin. It was comprised of several columns, primarily regarding information about Tweets that mention Bitcoin in the hashtag or in the actual content of the tweet itself. This data was then aggregated (see ProcessDateData.py) into all the activity for a given day. The following shows some examples of some samples obtained from the Twitter activity dataset:

![Graphical user interface, text, application, email Description automatically generated](./images/Aspose.Words.75b88a02-aea9-458e-ad3c-2daf8349b990.001.png)

As can be seen, the data available corresponds to the column fields of "user\_name", "user\_location", "user\_description", "user\_created", "user\_followers","user\_friends" , "user\_favourites", "user\_verified", "date", "text", "hashtags", "source", and "is\_retweet" respectively. This data was then aggregated into time samples by taking the sum of the user\_friends, “user\_followers”, “user\_favourites”, quantity of however many hashtags there were and the quantity of “user\_verified” to produce a feature vector for a given time step. The unit of time chosen for aggregation was an hour since a day would have proved to not be fine enough and not provide enough data. Note that the feature vector is somewhat scaled so that its inputs do not cause the gradient to explode.

` `Here is an example of what a feature vector would look like for a given time step:

![](./images/Aspose.Words.75b88a02-aea9-458e-ad3c-2daf8349b990.002.png)

Note that since the date ranges for this input data are between   11:59:04 PM, February 10th, 2021, and 10:56:51 AM August 18th, 2021, some historical pricing data was needed on an hourly basis. In response to this, the historical hourly Bitcoin pricing data was obtained from the following link (“GEMINI EXCHANGE ”), specifically targeting the Gemini exchange’s pricing data. From there, the inputs could be matched to the outputs by using the timestamp (with some help from the datetime library in Python). 

From there on, the data for the input was exported to ElasticSearch and visualized for ease of visualization.

![Graphical user interface, text, application Description automatically generated](./images/Aspose.Words.75b88a02-aea9-458e-ad3c-2daf8349b990.003.png)

![Chart, pie chart Description automatically generated](./images/Aspose.Words.75b88a02-aea9-458e-ad3c-2daf8349b990.004.png)

![A picture containing text, blackboard Description automatically generated](./images/Aspose.Words.75b88a02-aea9-458e-ad3c-2daf8349b990.005.png)

For data augmentation, some basic statistics were taken of the input and output data to make sure that the addition of random noise to the data was not completely without reason. To be more specific, the random noise added to the data was usually one half of the corresponding standard deviation of the figure as shown below:

||Average|Stddev|Variance|Median|Max|Min|
| :- | :- | :- | :- | :- | :- | :- |
|Total Friends Reached|1231562|9155523|83823600000000|1115330|7362947|5056|
|Total User Followers|8500028|9157030|83851200000000|6211977|66956223|4392|
|Total User Favourites|6587493|6594598|43488700000000|5559928|69946949|20067|
|Total Verified Users|7.204132|10.50792|110.416479|4|162|0|
|Total Number of Hashtags|4914.655|4256.424|18117141.14|4413|29972|6|


||log Average|log Stddev|log Variance|log Median|log Max|log Min|
| :- | :- | :- | :- | :- | :- | :- |
|Total Friends  Reached|6.090456|6.961683|13.92336635|6.047403|6.867052|3.703807|
|Total User Followers|6.92942|6.961755|13.92350925|6.79323|7.825791|3.642662|
|Total User Favourites|6.81872|6.819188|13.63837666|6.745069|7.844769|4.302482|
|Total Verified Users|0.857582|1.021517|2.043033894|0.60206|2.209515|NaN|
|Total Number of Hashtags|3.691493|3.629045|7.258089668|3.644734|4.476716|0.778151|

The following are the equivalent tables for the outputs:



||Average|Stddev|Variance|Median|Max|Min|
| :- | :- | :- | :- | :- | :- | :- |
|Open|12170.38|16031.86|2.57E+08|7085.52|68636.96|0|
|High|12240.84|17711.6|3.14E+08|7120.98|69000|243.6|
|Low|12094.15|15928.17|2.54E+08|7048.99|68477.94|0|
|Close|12171.25|16032.48|2.57E+08|7085.64|68636.96|243.6|
|Volume|139.8234|267.1487|71368.45|59.41659|8526.751|0|


||log Average|log Stddev|log Variance|log Median|log Max|log Min|
| :- | :- | :- | :- | :- | :- | :- |
|Open|4.085304|4.204984|8.409968|3.850372|4.836558|NaN|
|High|4.087811|4.248258|8.496516|3.85254|4.838849|2.386677|
|Low|4.082575|4.202166|8.404332|3.848127|4.835551|NaN|
|Close|4.085335|4.205001|8.410001|3.850379|4.836558|2.386677|
|Volume|2.14558|2.426753|4.853506|1.773908|3.930784|NaN|

From there, the data was augmented by adding random noise to it (within the previously mentioned guidelines). The code for adding random inputs and outputs is shown below:

![A screenshot of a computer Description automatically generated with medium confidence](./images/Aspose.Words.75b88a02-aea9-458e-ad3c-2daf8349b990.006.png)

![A screenshot of a computer Description automatically generated with medium confidence](./images/Aspose.Words.75b88a02-aea9-458e-ad3c-2daf8349b990.007.png)

**Results**

The models I tried were variations of a simple RNN architecture I tried out as a benchmark. That 

benchmark was able to obtain a Mean Squared Error of about 0.167 for the validation score. 

Note that as mentioned in the code for this section (“Results.ipynb”), this validation score is 

highly suspect and might be due to a bad implementation and a lack of knowledge on the part of 

the user of how RNN’s work with Keras. Regardless, from that point on, the effort was 

concentrated on lowering the loss and making the Neural Network learn the best. Different 

variations of architectures were tried out, as well as different activation functions (surprisingly, 

linear activations worked the best for this), different types of Dropout (no dropout seemed to 

work the best, but more testing is needed), different optimizers (finally settled on Stochastic 

Gradient Descent), and different loss functions (MeanAbsoluteError seemingly being 

the most useful of them). The following is a showcase of the losses obtained while running the 

experiments:

![Graphical user interface, chart Description automatically generated](./images/Aspose.Words.75b88a02-aea9-458e-ad3c-2daf8349b990.008.png)

For the full details of the results of each run, the Weights and Biases page can be viewed, as can 

the Jupyter Notebook (“Results.ipynb”). The best loss obtained was about 0.1154 in RunXII.

**Conclusion**

As with any model, this is an imperfect representation of the real world, especially taking into account the stochastic nature of the problem. Some features which should be taken into account are: the effects that real-world events have on the price of cryptocurrencies, most notably demonstrated with the effects of events such as the onset of the COVID-19 pandemic, the announcement of the IRS increasing its workforce to tackle crypto-related tax fraud, Elon Musk announcing that Tesla would accept cryptocurrencies and subsequently announcing that it could not accept cryptocurrencies, as well as China’s government cracking down on cryptocurrency activity, to name a few. With regards to this, one other significant price influencer not taken into account is the change that larger cryptocurrencies have on smaller cryptocurrencies. As a general principle, one will find that the activity and prices of the two largest cryptocurrencies by circulation (Bitcoin and Ethereum) have a significant effect on the price of smaller cryptocurrencies. Of course, part of this effect comes down to the previously mentioned influence that real-world events have on cryptocurrencies in general, so there is a bit of overlap between the last two mentioned events.  In addition, this approach assumes that simply sharing a tweet or posting on social media with a related hashtag will raise the popularity of the coin along with the price. There may be outliers where for some reason a crypto gets widely deemed to be a scam and thus receives many negative tweets and thus dissuades many people from investing in it. However, based upon a simple observation of the cryptocurrency social media space, this is likely uncommon.


Works Cited

Amidi, Afshine, and Shervine Amidi. “Recurrent Neural Networks Cheatsheet Star.” *CS 230 Recurrent Neural Networks Cheatsheet*, Stanford University, https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks. 

Derakhshan, Ali, en Hamid Beigy. “Sentiment analysis on stock social media 

for stock price movement prediction”. *Engineering Applications of Artificial Intelligence* 85 (2019): 569–578. Web.

Georgoula, Ifigeneia and Pournarakis, Demitrios and Bilanakos, Christos and Sotiropoulos, 

Dionisios and Sotiropoulos, Dionisios and Giaglis, George M., Using Time-Series and Sentiment Analysis to Detect the Determinants of Bitcoin Prices (May 17, 2015). Available at SSRN: <https://ssrn.com/abstract=2607167> or [http://dx.doi.org/10.2139/ssrn.2607167](https://dx.doi.org/10.2139/ssrn.2607167)

[Karalevicius, V.](https://www.emerald.com/insight/search?q=Vytautas%20Karalevicius "Vytautas Karalevicius"), [Degrande, N.](https://www.emerald.com/insight/search?q=Niels%20Degrande "Niels Degrande") and [De Weerdt, J.](https://www.emerald.com/insight/search?q=Jochen%20De%20Weerdt "Jochen De Weerdt") (2018), "Using sentiment 

analysis to predict interday Bitcoin price movements", [*Journal of Risk Finance*](https://www.emerald.com/insight/publication/issn/1526-5943), Vol. 19 No. 1, pp. 56-75. <https://doi.org/10.1108/JRF-06-2017-0092>

L. Felizardo, R. Oliveira, E. Del-Moral-Hernandez and F. Cozman, "Comparative study of Bitcoin price 

prediction using WaveNets, Recurrent Neural Networks and other Machine Learning Methods," *2019 6th International Conference on Behavioral, Economic and Socio-Cultural Computing (BESC)*, 2019, pp. 1-6, doi: 10.1109/BESC48373.2019.8963009

P. Jay, V. Kalariya, P. Parmar, S. Tanwar, N. Kumar and M. Alazab, "Stochastic Neural Networks for 

Cryptocurrency Price Prediction," in *IEEE Access*, vol. 8, pp. 82804-82818, 2020, doi: 10.1109/ACCESS.2020.2990659. 
