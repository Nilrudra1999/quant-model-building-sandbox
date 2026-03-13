## Model Technical Specifications Sheet

<b>Details the model's build and internal mechanisms</b>

### Fundamentals

The model will output the Intrinsic value of a company based on a formula developed by Benjamin Graham. All variables required by the model must be available within the stock's balance sheet, and must work universally for any stock within either the TSX or the NYSE.

The following videos will be used to implemented the model:<br>
https://www.youtube.com/watch?v=8jmjxXc5H8c <br>
https://www.youtube.com/watch?v=d0EBO-vs0GM

### Data required

- EPS (earnings per share): should be calculated so user provides variables for the calculation
- P/E ratio: should be calculated so user provides variables for the calculation
- Growth Rate: should be calculated so user provides variables for the calculation
- Average Yield of AAA bonds: determined by model based on sector
- Current yield of AAA bonds: provided by user directly

### Model Structure

Neural network of models which calculate things, use assessment algorithms, and predictions algorithms.

Variables such as:

- EPS
- P/E ratio
- Growth Rate

Will be calculated by my model. Meanwhile, Average Yield of AAA bonds will be predicted/determined based on sector and other factors by a neuron. This can be saved in the following manner when using Scikit-learn to develope the model:

```python
# 1. Define your custom formula
def my_custom_formula(X):
    # Example: Calculate a ratio or interaction between columns
    # Let's say we multiply the first two columns
    combined_feature = X[:, 0] * X[:, 1]
    return np.column_stack((X, combined_feature))

# 2. Build the Pipeline
pipeline = Pipeline([
    ('custom_math', FunctionTransformer(my_custom_formula)),
    ('neural_net', Perceptron()) # Or any MLPClassifier
])

# 3. Save the whole thing
import joblib
joblib.dump(pipeline, 'full_model.joblib')
```
