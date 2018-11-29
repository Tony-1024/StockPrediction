# StockPrediction
Machine Learning Algorithm To Predict Stock Price With Two Ways - LSTM and WNN+LSTM

Here is the code for predicting stock price. The dataset is from the New York Stock Exchange (NYSE). I randomly select a stock (NYSE code: AAPL), data samples are shown below:
![alt text](https://github.com/Tony-1024/StockPrediction/blob/master/charts/Dataset.jpg)

Then, implement the Neural Network with two ways: Pure LSTM, and Wavelet+LSTM. 

Haar Close Denoising:
![alt text](https://github.com/Tony-1024/StockPrediction/blob/master/charts/Close%20Denoising.jpg)
![alt text](https://github.com/Tony-1024/StockPrediction/blob/master/charts/Close%20Denoising-in%20one.jpg)

Train the two models and make prediction. The results are shown as below:
![alt text](https://github.com/Tony-1024/StockPrediction/blob/master/charts/Comparison.jpg)

RMSE of the two models:
![alt text](https://github.com/Tony-1024/StockPrediction/blob/master/charts/RMSE.jpg)
