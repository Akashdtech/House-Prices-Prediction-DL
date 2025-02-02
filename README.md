# House-Prices-Prediction-DL
This code implements a machine learning pipeline for predicting housing prices in California using a neural network model. The pipeline involves the following steps.

Data Loading and Preparation:

    The fetch_california_housing dataset is loaded, containing features like the median income, house age, and average rooms, and the target variable 'Price' (housing prices).
    A DataFrame is created from the dataset, and the target variable 'Price' is added as a new column.

Data Visualization:

    sweetviz is used to generate a comprehensive visual report of the dataset, providing insights into the distribution and relationships of the features.

Data Preprocessing:

    The dataset is split into features (X) and target (y).
    The data is then split into training and testing sets (80% training, 20% testing) using train_test_split.
    Features are standardized using StandardScaler to improve model performance.

Model Building:

    A neural network model is defined using TensorFlow/Keras.
    The model consists of three fully connected layers with relu activation functions and a final output layer with a single neuron for regression.
    A Dropout layer is included to prevent overfitting.

Model Training:

    The model is compiled using the Adam optimizer and mean squared error (MSE) as the loss function. It is then trained for 10 epochs with a batch size of 32, using 10% of the training data for validation.

Model Evaluation:

    The model's performance is evaluated on the test set, and both Mean Squared Error (MSE) and Mean Absolute Error (MAE) are calculated.
    The model's predictions are compared to the actual values using the mean_squared_error and mean_absolute_error functions from sklearn.

New Data Prediction:

    A new sample (a hypothetical housing unit) is created and standardized using the same scaler.
    The trained model is used to predict the price for this new sample.

Outputs:

    The model’s performance is reported with MSE and MAE.
    The predicted price for the new housing sample is printed.

This script demonstrates a complete machine learning pipeline, from data loading and preprocessing to model evaluation and prediction.
