# Skew Normal Distribution Parameter Estimator

This project uses **Feedforward Neural Network** to predict the parameters of the **Skew Normal Distribution**.

## Neural Network Layout

![image](/images/NNLayout.jpg)

## Usage

```bash
from SNParameterEstimator import ParameterEstimator

# Computer the parameters using the function
mu, sd, alpha = ParameterEstimator(x_values, y_values, 501)
```

**mu**: Location (Mean)  
**sd**: Scale (Standard Deviation)  
**alpha**: Skewness

## Limitations

1. This model tends to be more precise when
    1. **mu** is close to **0**.
    2. **sd** is larger than **1.0**.
    3. **alpha** is larger than **1.0**.

## Guess of Reasons and Future Work

1. The reason why this model does not work well in all range of values might because 2 hidden layers are not enought for prediction. Thus, increasing the number of layers of using RNN instead might improve the performance.
2. The hyperparameters, like learning rate and decay, need to be tuned for the better performance.
3. Cases like when the data is not completed have not been tested for this model.

## References

1. Ideas of splitting the neurons for parameters: https://medium.com/hal24k-techblog/a-guide-to-generating-probability-distributions-with-neural-networks-ffc4efacd6a4
2. Derivative of error function: https://proofwiki.org/wiki/Derivative_of_Error_Function
3. Derivative calculator: https://www.derivative-calculator.net/