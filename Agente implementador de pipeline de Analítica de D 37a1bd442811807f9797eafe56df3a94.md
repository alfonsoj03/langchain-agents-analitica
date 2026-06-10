# Agente implementador de pipeline de Analítica de Datos

Fecha: 8 de junio de 2026 21:46
Revisado: No

1. Data integration → Take from csv to dataset
2. Remove redundant or irrelevant variables → Can this be completely automated with corrleation matrix and such?
3. Statistical description of data → Generate an HTML report.
4. Atypical data / Error cleaning.
5. Null data cleaning → Imputation → Fully automate-able

FOR PROBABILITY-BASED ML MODELS

- Numeric variables discretize

FOR MATH FORMULAS BASED ML MODELS

- Numeric variable normalization
- Dummy creation for categoric variables
- Numeric codification for objective variable (IF IT IS CATEGORY)

1. 70-30 division of data → Predictors are X and Target is Y
2. Train model, for ex:

```go
from sklearn import neighbors
model_Knn = neighbors.KNeighborsClassifier(n_neighbors=1, metric='euclidean') #minkowski
model_Knn.fit(X_train, Y_train) #70% datos
```

According to the target variable (numeric or cat) → Evaluation of results (MSE, RMSE, MAE, MAPE).

1. Save models (in a model.pkl), along with labelencoder if used and variables (X.columns._values)

## Deployment

1. Load the model saved in the last step of the past guide. (REMEMBER THAT HERE CAN COME MULTIPLE MODELS, LABEL ENCODERS, AND VARIABLES TOO).
2. Read future data. Can be taken from endpoint body parameters.
3. Prepare future data. Mirror steps of data preparation so dummies if any must be the same as the ones used to create the model.
4. If normalized numerical values → Used the minmaxscaler that must have been brought in the model.pkl so the scale is the same.
5. After these steps, dataset must be re-indexed with missing columns brought from “variables” saved in model.pkl. This application of variables reconciles future data with ammount of columns added by dummies in preparation. N of columns of future data must be the same as the final dataset used to train the model.
6. Make the prediction using the model from model.pkl.
7. Reverse transform results with label encoder. Not applicable if target variable is numeric like in this case.
8. Paste results into last column.
9. Return the complete dataset with results.

Remember that this deployment phase is the transformations done to data on endpoint calls, this is data manipulation so the data arrives reconciled with expected format for the model prediction.