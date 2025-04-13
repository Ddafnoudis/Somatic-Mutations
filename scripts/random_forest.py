"""
Random Forest Classification and Roc-Curve
"""
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, roc_curve, auc


def random_forest_train_test_validation(X_train, y_train, X_test, y_test, X_val, y_val, target_classes, seed, rf_best_params):
    """
    Random Forest Classification
    """
    # Pass the parameters to the model
    rfc = RandomForestClassifier(n_estimators=rf_best_params['n_estimators'], 
                                 max_features=rf_best_params['max_features'],
                                 bootstrap=rf_best_params['bootstrap'],
                                 random_state=seed)
    # Fit the trainin sets
    rfc.fit(X_train, y_train)
    # Predictions on test set
    y_pred = rfc.predict(X_test)
    # Predictions on validation set
    y_pred_val = rfc.predict(X_val)
    # Accuracy
    b_accuracy = balanced_accuracy_score(y_test, y_pred)
    print(f"Accuracy in test: {b_accuracy}\n")

    # Accuracy on validation set
    b_accuracy_val = balanced_accuracy_score(y_val, y_pred_val)
    print(f"Accuracy in validation: {b_accuracy_val}\n")

    # Classification report
    report = classification_report(y_test, y_pred, target_names=target_classes)
    # Classification report for validation set
    report_val = classification_report(y_val, y_pred_val, target_names=target_classes)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:\n{cm}\n")

    # Confusion Matrix for validation set
    cm_val = confusion_matrix(y_val, y_pred_val)
    print(f"Confusion Matrix in validation:\n{cm_val}\n")

    # Plot confusion matrix (train)
    fig = px.imshow(cm, text_auto=True, title="Confusion Matrix: Train-Test set",
                labels=dict(x="Predicted Labels", y="True Labels", color="Productivity"),
                x=target_classes,
                y=target_classes)
    fig.update_xaxes(side="top")
    fig.show()

    # Plot confusion matrix (validation)
    fig = px.imshow(cm_val, text_auto=True, title="Confusion Matrix: Validation set",
                labels=dict(x="Predicted Labels", y="True Labels", color="Productivity"),
                x=target_classes,
                y=target_classes)
    fig.update_xaxes(side="top")
    fig.show()

    # Save the results
    # np.savetxt('result_files/cv_score.txt', cv_score)
    
    # Save the results
    with open('result_files/rf_folder/balanced_accuracy.txt', "w") as f:
        f.write(str(b_accuracy))

    with open('result_files/rf_folder/balanced_accuracy_val.txt', "w") as f:
        f.write(str(b_accuracy_val))
    
    with open('result_files/rf_folder/class_report.txt', "w") as f:
        f.write(report)

    with open('result_files/rf_folder/class_report_val.txt', "w") as f:
        f.write(report_val)

    # Roc Curve
    n_classes = len(np.unique(y_val))
    y_score = rfc.predict_proba(X_val)
    lb = LabelBinarizer().fit(y_val)
    y_onehot_test = lb.transform(y_val)

    # Initialize variables for ROC curve
    fpr = {}
    tpr = {}
    thresh = {}
    roc_auc = {}

    # Calculate ROC curve and AUC for each class
    for i in range(n_classes):
        fpr[i], tpr[i], thresh[i] = roc_curve(y_onehot_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot using Plotly
    fig = go.Figure()

    # Plot ROC curve for each class
    for i in range(n_classes):
        fig.add_trace(go.Scatter(x=fpr[i], y=tpr[i],
                                mode='lines',
                                name=f'{target_classes[i]} vs Rest (AUC={roc_auc[i]:.2f})'))

    # Add diagonal line
    fig.update_layout(
    title='Multiclass ROC curve',
    xaxis_title='False Positive Rate',
    yaxis_title='True Positive Rate',
    xaxis=dict(range=[0, 1], title_font=dict(size=20)), 
    yaxis=dict(range=[0, 1.05], title_font=dict(size=20)),  
    legend=dict(
        x=1,
        y=0,
        traceorder='normal',
        font=dict(
            family='sans-serif',
            size=10,
            color='black'
        ),
        bgcolor='LightSteelBlue',
        bordercolor='Black',
        borderwidth=1
    ),
    width=1200,
    height=600
    )
    fig.show()
    # Save the plot
    fig.write_image("result_files/rf_folder/roc_curve.png")


    return b_accuracy, report, cm, b_accuracy_val, report_val, cm_val


if __name__=="__main__":
    random_forest_train_test_validation()
