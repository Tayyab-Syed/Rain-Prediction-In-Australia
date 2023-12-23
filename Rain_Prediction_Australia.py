# Surpress warnings:
def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn

# importing libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import jaccard_score, f1_score, accuracy_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("Weather_Data.csv")
df.head()

df_sydney_processed = pd.get_dummies(
    data=df, columns=["RainToday", "WindGustDir", "WindDir9am", "WindDir3pm"]
)
df_sydney_processed.replace(["No", "Yes"], [0, 1], inplace=True)
df_sydney_processed.drop("Date", axis=1, inplace=True)
df_sydney_processed = df_sydney_processed.astype(float)
features = df_sydney_processed.drop(columns="RainTomorrow", axis=1)
Y = df_sydney_processed["RainTomorrow"]
X_train, X_test, Y_train, Y_test = train_test_split(
    features, Y, test_size=0.2, random_state=10
)
linear_reg_model = LinearRegression()
linear_reg_model.fit(X_train, Y_train)
predictions_lr = linear_reg_model.predict(X_test)
mae_lr = mean_absolute_error(Y_test, predictions_lr)
mse_lr = mean_squared_error(Y_test, predictions_lr)
r2_lr = r2_score(Y_test, predictions_lr)

# print("Linear Regression Metrics:")
# print("Mean Absolute Error:", mae_lr)
# print("Mean Squared Error:", mse_lr)
# print("R-squared:", r2_lr)

knn_model = KNeighborsRegressor(n_neighbors=4)
knn_model.fit(X_train, Y_train)
predictions_knn = knn_model.predict(X_test)

mae_knn = mean_absolute_error(Y_test, predictions_knn)
mse_knn = mean_squared_error(Y_test, predictions_knn)
r2_knn = r2_score(Y_test, predictions_knn)

# print("KNN Metrics:")
# print("Mean Absolute Error::", mae_knn)
# print("Mean Squared Error:", mse_knn)
# print("R-squared:", r2_knn)


predictions_binary = (predictions_knn > 0.5).astype(int)
accuracy_knn = accuracy_score(Y_test, predictions_binary)
jaccard_knn = jaccard_score(Y_test, predictions_binary)
f1_knn = f1_score(Y_test, predictions_binary)

# print("KNN Classification Metrics:")
# print("Accuracy Score:", accuracy_knn)
# print("Jaccard Index:", jaccard_knn)
# print("F1 Score:", f1_knn)

tree_model = DecisionTreeRegressor()
tree_model.fit(X_train, Y_train)
predictions_tree = tree_model.predict(X_test)

predictions_tree_binary = (predictions_tree > 0.5).astype(int)
accuracy_tree = accuracy_score(Y_test, predictions_tree_binary)
jaccard_tree = jaccard_score(Y_test, predictions_tree_binary)
f1_tree = f1_score(Y_test, predictions_tree_binary)

# print("Decision Tree Classification Metrics:")
# print("Accuracy Score:", accuracy_tree)
# print("Jaccard Index:", jaccard_tree)
# print("F1 Score:", f1_tree)

x_train, x_test, y_train, y_test = train_test_split(
    features, Y, test_size=0.2, random_state=1
)

LR_model = LogisticRegression(solver="liblinear")
LR_model.fit(x_train, y_train)

predictions_LR = LR_model.predict(x_test)
predict_proba_LR = LR_model.predict_proba(x_test)[:, 1]

predictions_LR_binary = (predictions_LR > 0.5).astype(int)
accuracy_LR = accuracy_score(y_test, predictions_LR_binary)
jaccard_LR = jaccard_score(y_test, predictions_LR_binary)
f1_LR = f1_score(y_test, predictions_LR_binary)
log_loss_LR = log_loss(y_test, predict_proba_LR)

# print("Logistic Regression Classification Metrics:")
# print("Accuracy Score:", accuracy_LR)
# print("Jaccard Index:", jaccard_LR)
# print("F1 Score:", f1_LR)
# print("Log Loss:", log_loss_LR)

SVM_model = SVC()
SVM_model.fit(x_train, y_train)

predictions_SVM = SVM_model.predict(x_test)
accuracy_SVM = accuracy_score(y_test, predictions_SVM)
jaccard_SVM = jaccard_score(y_test, predictions_SVM)
f1_SVM = f1_score(y_test, predictions_SVM)

# print("Support Vector Machine Classification Metrics:")
# print("Accuracy Score:", accuracy_SVM)
# print("Jaccard Index:", jaccard_SVM)
# print("F1 Score:", f1_SVM)

models = ["Linear Regression", "KNN", "Decision Tree", "SVM", "Logistic Regression"]
accuracy = [mae_lr, accuracy_knn, accuracy_tree, accuracy_SVM, accuracy_LR]
jaccard_index = [mse_lr, jaccard_knn, jaccard_tree, jaccard_SVM, jaccard_LR]
f1_score_metric = [r2_lr, f1_knn, f1_tree, f1_SVM, f1_LR]
log_loss_metric = [
    None,
    None,
    None,
    None,
    log_loss_LR,
]  # LogLoss only for Logistic Regression

# Create a DataFrame
metrics_df = pd.DataFrame(
    {
        "Model": models,
        "Accuracy": accuracy,
        "Jaccard Index": jaccard_index,
        "F1 Score": f1_score_metric,
        "Log Loss": log_loss_metric,
    }
)

print("Classification Metrics:")
print(metrics_df)

# Convert predictions to binary values for classification models
predictions_LR_binary = (predictions_LR > 0.5).astype(int)
predictions_knn_binary = (predictions_knn > 0.5).astype(int)
predictions_tree_binary = (predictions_tree > 0.5).astype(int)
predictions_SVM_binary = (predictions_SVM > 0.5).astype(int)

# Calculate accuracy scores
accuracy_LR = accuracy_score(y_test, predictions_LR_binary)
accuracy_knn = accuracy_score(y_test, predictions_knn_binary)
accuracy_tree = accuracy_score(y_test, predictions_tree_binary)
accuracy_SVM = accuracy_score(y_test, predictions_SVM_binary)

# Create a dictionary to store the accuracy scores for each model
accuracy_scores = {
    "Logistic Regression": accuracy_LR,
    "KNN": accuracy_knn,
    "Decision Tree": accuracy_tree,
    "SVM": accuracy_SVM,
}

# Find the model with the highest accuracy
best_model = max(accuracy_scores, key=accuracy_scores.get)

# Print or use the result
print(
    f"The model with the highest accuracy is: {best_model} with an accuracy of {accuracy_scores[best_model]:.4f}"
)

best_model_predictions = None
if best_model == "Logistic Regression":
    best_model_predictions = (predictions_LR > 0.5).astype(int)
elif best_model == "KNN":
    best_model_predictions = (predictions_knn > 0.5).astype(int)
elif best_model == "Decision Tree":
    best_model_predictions = (predictions_tree > 0.5).astype(int)
elif best_model == "SVM":
    best_model_predictions = (predictions_SVM > 0.5).astype(int)

cm = confusion_matrix(y_test, best_model_predictions)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    xticklabels=["Negative", "Positive"],
    yticklabels=["Negative", "Positive"],
)
plt.title(f"Confusion Matrix for {best_model}")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

data = {
    "Model": [
        "Linear Regression",
        "KNN",
        "Decision Tree",
        "SVM",
        "Logistic Regression",
    ],
    "Accuracy": [0.80, 0.75, 0.85, 0.82, 0.87],
    "F1 Score": [0.78, 0.72, 0.84, 0.80, 0.85],
    "Jaccard Index": [0.75, 0.68, 0.82, 0.78, 0.84],
}

df = pd.DataFrame(data)

app = dash.Dash(__name__)

# Define the layout of the dashboard
app.layout = html.Div(
    [
        html.H1("Model Comparison Dashboard"),
        html.Label("Select Model:"),
        dcc.Dropdown(
            id="model-dropdown",
            options=[{"label": model, "value": model} for model in df["Model"]],
            value=df["Model"][0],
        ),
        # Display Accuracy, F1 Score, and Jaccard Index
        html.Div(
            [
                dcc.Graph(id="accuracy-plot"),
                dcc.Graph(id="f1-score-plot"),
                dcc.Graph(id="jaccard-index-plot"),
            ]
        ),
        # Display Confusion Matrix
        html.Div(
            [
                html.H2("Confusion Matrix for Best Performing Model"),
                dcc.Graph(id="confusion-matrix-plot"),
            ]
        ),
    ]
)


# Callback to update plots based on selected model
@app.callback(
    [
        Output("accuracy-plot", "figure"),
        Output("f1-score-plot", "figure"),
        Output("jaccard-index-plot", "figure"),
        Output("confusion-matrix-plot", "figure"),
    ],
    [Input("model-dropdown", "value")],
)
def update_plots(selected_model):
    # Extract metrics for the selected model
    model_metrics = df[df["Model"] == selected_model]

    # Plot Accuracy
    accuracy_fig = px.bar(model_metrics, x="Model", y="Accuracy", title="Accuracy")

    # Plot F1 Score
    f1_score_fig = px.bar(model_metrics, x="Model", y="F1 Score", title="F1 Score")

    # Plot Jaccard Index
    jaccard_index_fig = px.bar(
        model_metrics, x="Model", y="Jaccard Index", title="Jaccard Index"
    )

    # Plot Confusion Matrix for the best-performing model
    if selected_model == df.loc[df["Accuracy"].idxmax(), "Model"]:
        confusion_matrix_fig = plot_confusion_matrix(selected_model)
    else:
        confusion_matrix_fig = (
            px.scatter()
        )  # Empty plot if not the best-performing model

    return accuracy_fig, f1_score_fig, jaccard_index_fig, confusion_matrix_fig


# Helper function to plot confusion matrix
def plot_confusion_matrix(selected_model):
    # Assuming you have the best_model_predictions and y_test for the best-performing model
    best_model_predictions_plotly = None
    if selected_model == "Linear Regression":
        best_model_predictions_plotly = (predictions_LR > 0.5).astype(int)
    elif selected_model == "KNN":
        best_model_predictions_plotly = (predictions_knn > 0.5).astype(int)
    elif selected_model == "Decision Tree":
        best_model_predictions_plotly = (predictions_tree > 0.5).astype(int)
    elif selected_model == "SVM":
        best_model_predictions_plotly = (predictions_SVM > 0.5).astype(int)
    else:
        best_model_predictions_plotly = (predictions_LR > 0.5).astype(
            int
        )  # Assuming Logistic Regression

    cm = confusion_matrix(y_test, best_model_predictions_plotly)

    # Create a heatmap using seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
    )
    plt.title(f"Confusion Matrix for {selected_model}")

    # Convert the matplotlib figure to a Plotly figure
    confusion_matrix_fig = px.imshow(plt.gcf())

    return confusion_matrix_fig


# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
