# logistic-regression-from-scratch
This repository has the implementation of Logistic Regression algorithm from scratch, using SGD (Stochastic Gradient Descent). Scikit Learn library is not used.
<h1>Logistic Regression</h1>

![image](https://user-images.githubusercontent.com/86348193/218715508-1938f92e-afa4-4c30-82a4-77692aa9f939.png) <br>
Logistic Regression is a classification algorithm which is an example of supervised machine learning. It is used to predict the probability of a binary (yes/no) event occurring. An example of logistic regression could be applying machine learning to determine if a person is likely to be infected with COVID-19 or not.<br>
Logistic Regression can be used using Scikit Learn's SGDClassifier module with loss as 'log_loss' but here in this project, we are implementing Logistic Regression from scratch, without using Scikit Learn library. To implement from scratch, we need to know what is Stochastic Gradient Descent(SGD) algorithm.<br>
<h1>Stochastic Gradient Descent</h1>

![image](https://user-images.githubusercontent.com/86348193/218715763-963e3997-2bbc-4c35-92ef-b05e3a850cd0.png) <br>
Stochastic Gradient Descent is an optimization algorithm often used in machine learning applications to find the model parameters that correspond to the best fit between predicted and actual outputs. With the help of this algorithm, we can find minima for a loss or maxima for a score (f1, accuracy, etc).<br>
To understand SGD or GD in detail, you can refer my <a href="https://medium.com/@rohan-dawkhar/the-power-of-gradient-descent-in-machine-learning-169f59ca391e" target="_blank">Medium Blog</a> where I've explained Gradient Descent in a very simple and understable manner.<br>
<h1>Logistic Regression Custom Implementation</h1>
* Initialize the weight_vector and intercept term to zeros.

* Create a loss function

 $log loss = -1*\frac{1}{n}\Sigma_{for each Yt,Y_{pred}}(Ytlog10(Y_{pred})+(1-Yt)log10(1-Y_{pred}))$
- for each epoch:

    - for each data point in train:

        - calculate the gradient of loss function w.r.t each weight in weight vector

            $dw^{(t)} = x_n(y_n − σ((w^{(t)})^{T} x_n+b^{t}))- \frac{λ}{N}w^{(t)})$ <br>

        - Calculate the gradient of the intercept

           $ db^{(t)} = y_n- σ((w^{(t)})^{T} x_n+b^{t}))$

        - Update weights and intercept<br>
            $w^{(t+1)}← w^{(t)}+α(dw^{(t)}) $<br>

            $b^{(t+1)}←b^{(t)}+α(db^{(t)}) $
    - calculate the log loss for train and test with the updated weights
<br>
<h1>Conclusion</h1>
After the SKLearn and Custom Implementations of Logistic Regression, Both the weights results are compared and we found that <b>the difference between the optimal weights(w) of scikitlearn's Logistic Regression and custom implemented Logistic Regression is in terms of 10⁻³ (for almost weights).<br>
It concludes that our custom implementation for Logistic Regression with L2 Regularization is correct.</b>
