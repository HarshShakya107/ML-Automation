import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
try:
    from xgboost import XGBClassifier
    xgb_available = True
except ImportError:
    xgb_available = False

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import xgboost as xgb

st.set_page_config(page_title="AutoML Studio", layout="wide")
st.markdown("<h1 style='text-align:center; color:#00bfff;'> AutoML Studio — Train ML Models Without Code</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

st.sidebar.header("Dataset Upload")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])
use_example = st.sidebar.checkbox("Use Example Dataset (Iris)", value=False)

if use_example and not uploaded_file:
    from sklearn.datasets import load_iris
    iris = load_iris(as_frame=True)
    df = iris.frame
    st.success("Using Iris example dataset")
    st.write(df.head())
elif uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset loaded successfully")
    st.write("Data Preview")
    st.dataframe(df.head())
else:
    st.warning("Please upload a CSV file or select example dataset to continue.")
    st.stop()

st.sidebar.header("Target Selection")
target_col = st.sidebar.selectbox("Select Target Column", df.columns)
X = df.drop(columns=[target_col])
y = df[target_col]

def encode_data(X, y):
    x_encoded = X.copy()
    for c in X.columns:
        if X[c].dtype == 'object' or str(X[c].dtype).startswith('category'):
            x_encoded[c] = LabelEncoder().fit_transform(X[c].astype(str))

    if y.dtype == 'object' or str(y.dtype).startswith('category'):
        target_le = LabelEncoder()
        y_encoded = target_le.fit_transform(y.astype(str))
        label_map = dict(enumerate(target_le.classes_))
    else:
        y_encoded = y.values
        label_map = None
    return x_encoded, y_encoded, label_map

x_encoded, y_encoded, label_map = encode_data(X, y)

st.sidebar.header("Data Split & Scaling")
test_size = st.sidebar.slider("Test Size", 0.1, 0.3, 0.2)
random_state = 42
X_train, X_test, y_train, y_test = train_test_split(
    x_encoded, y_encoded, test_size=test_size, random_state=random_state
)

do_scale = st.sidebar.checkbox("Apply StandardScaler", value=True)
scaler = None
if do_scale:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    st.sidebar.success("StandardScaler applied")

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align:center; color:#ff9933;'>Classification Models — Predict Categories</h2>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8,tab9, tab10= st.tabs(["Random Forest", "Decision Tree", "K-Nearest Neighbors","Logistic Regression","SVM Classifier","Gradient Boosting (GBM)","XGBoost Classifier"," Navie Bayes","AdaBoost Classifier", "Extra Trees Classifier"])

from sklearn.utils.multiclass import type_of_target

with tab1:
    st.header("Random Forest Classifier")
    n_estimators = st.number_input("Number of Trees", min_value=10, max_value=1000, value=100, step=10, key="rf_n_estimators")
    max_depth = st.number_input("Max Depth (0 = None)", min_value=0, max_value=100, value=0, key="rf_max_depth")

    if st.button("Train Random Forest", key="train_rf"):
        
        target_type = type_of_target(y_train)
        if target_type == "continuous":
            st.error("Invalid Target Selected: You selected a continuous (numeric) column as the target for a classification model.\n\n👉 Please choose a categorical/discrete column instead.")
            st.stop()

        model = RandomForestClassifier(
            n_estimators=int(n_estimators),
            max_depth=None if int(max_depth) == 0 else int(max_depth),
            random_state=random_state,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        st.success(f"✅ Accuracy: {acc:.4f}")

        cm = confusion_matrix(y_test, preds)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap="cool", ax=ax)
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)

        st.text("Classification Report:")
        st.text(classification_report(y_test, preds))

        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        features_sorted = [X.columns[i] for i in indices]
        fig_imp, ax_imp = plt.subplots(figsize=(8, 5))
        sns.barplot(x=importances[indices], y=features_sorted, palette="viridis", ax=ax_imp)
        ax_imp.set_title("Feature Importance (Random Forest)")
        ax_imp.set_xlabel("Importance Score")
        st.pyplot(fig_imp)

        os.makedirs("models", exist_ok=True)
        model_artifact = {"model": model, "scaler": scaler, "label_map": label_map}
        model_path = os.path.join("models", "random_forest.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model_artifact, f)
        with open(model_path, 'rb') as f:
            st.download_button("Download Model", f, file_name="random_forest.pkl", key="dl_rf")


with tab2:
    st.header("Decision Tree Classifier")
    criterion = st.selectbox("Criterion", ["gini", "entropy", "log_loss"], key="dt_criterion")
    max_depth_dt = st.number_input("Max Depth (0 = None)", min_value=0, max_value=100, value=0, key="dt_max_depth")

    if st.button("Train Decision Tree", key="train_dt"):
        target_type = type_of_target(y_train)
        if target_type == "continuous":
            st.error("Invalid Target Selected: You selected a continuous (numeric) column as the target for a classification model.\n\n👉 Please choose a categorical/discrete column instead.")
            st.stop()

        model = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=None if int(max_depth_dt) == 0 else int(max_depth_dt),
            random_state=random_state
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        st.success(f"✅ Accuracy: {acc:.4f}")

        cm = confusion_matrix(y_test, preds)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap="Greens", ax=ax)
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)

        st.text("Classification Report:")
        st.text(classification_report(y_test, preds))

        from sklearn import tree
        fig_dt, ax_dt = plt.subplots(figsize=(12, 8))
        tree.plot_tree(model, filled=True, feature_names=X.columns, class_names=[str(c) for c in np.unique(y)], fontsize=8)
        st.pyplot(fig_dt)

        os.makedirs("models", exist_ok=True)
        model_artifact = {"model": model, "scaler": scaler, "label_map": label_map}
        model_path = os.path.join("models", "decision_tree.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model_artifact, f)
        with open(model_path, 'rb') as f:
            st.download_button("⬇Download Model", f, file_name="decision_tree.pkl", key="dl_dt")

with tab3:
    st.header("K-Nearest Neighbors (KNN)")
    n_neighbors = st.slider("Number of Neighbors (K)", 1, 20, 5, key="knn_neighbors")
    metric = st.selectbox("Distance Metric", ["minkowski", "euclidean", "manhattan"], key="knn_metric")

    st.markdown("Decision Boundary Visualization Options")
    selected_features = st.multiselect(
        "Select 2 features to visualize decision boundary",
        options=list(X.columns),
        default=list(X.columns[:2]) if len(X.columns) >= 2 else list(X.columns),
        key="feature_selector_1"
    )

    if st.button("Train KNN", key="train_knn"):
        target_type = type_of_target(y_train)
        if target_type == "continuous":
            st.error("Invalid Target Selected: You selected a continuous (numeric) column as the target for a classification model.\n\n👉 Please choose a categorical/discrete column instead.")
            st.stop()

        model = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
        try:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            st.success(f"Accuracy: {acc:.4f}")

            cm = confusion_matrix(y_test, preds)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap="OrRd", ax=ax)
            ax.set_title("Confusion Matrix")
            st.pyplot(fig)

            st.text("Classification Report:")
            st.text(classification_report(y_test, preds))

            if len(selected_features) == 2:
                X_vis = x_encoded[selected_features].values 

                X_vis_train, X_vis_test, y_vis_train, y_vis_test = train_test_split(
                    X_vis, y_encoded, test_size=test_size, random_state=random_state
                )

                if do_scale:
                    X_vis_train = scaler.fit_transform(X_vis_train)
                    X_vis_test = scaler.transform(X_vis_test)

                model_vis = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
                model_vis.fit(X_vis_train, y_vis_train)

                x_min, x_max = X_vis_train[:, 0].min() - 1, X_vis_train[:, 0].max() + 1
                y_min, y_max = X_vis_train[:, 1].min() - 1, X_vis_train[:, 1].max() + 1
                xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                                     np.arange(y_min, y_max, 0.02))
                Z = model_vis.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)

                fig_db, ax_db = plt.subplots(figsize=(8, 6))
                ax_db.contourf(xx, yy, Z, alpha=0.4, cmap='coolwarm')
                ax_db.scatter(X_vis_train[:, 0], X_vis_train[:, 1], c=y_vis_train, edgecolor='k', cmap='coolwarm')
                ax_db.set_title(f"KNN Decision Boundary ({selected_features[0]} vs {selected_features[1]})")
                ax_db.set_xlabel(selected_features[0])
                ax_db.set_ylabel(selected_features[1])
                st.pyplot(fig_db)
            else:
                st.info("Please select exactly 2 features to visualize the decision boundary.")

            os.makedirs("models", exist_ok=True)
            model_artifact = {"model": model, "scaler": scaler, "label_map": label_map}
            model_path = os.path.join("models", "knn.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(model_artifact, f)
            with open(model_path, 'rb') as f:
                st.download_button("⬇Download Model", f, file_name="knn.pkl", key="dl_knn")

        except ValueError as e:
            st.error(f"Model Training Failed: {str(e)}")
            st.info("Tip: KNN doesn’t support NaN values — they’ve been automatically handled, but check dataset if issue persists.")

with tab4:
    st.header("Logistic Regression")
    c_val = st.number_input("Inverse Regularization Strength (C)", 0.01, 10.0, 1.0, 0.1)
    penalty = st.selectbox("Penalty", ["l2", "none"])
    solver = st.selectbox("Solver", ["lbfgs", "liblinear", "newton-cg", "saga"])

    st.markdown("Decision Boundary Visualization Options")
    selected_features = st.multiselect(
        "Select 2 features to visualize decision boundary",
        options=list(X.columns),
        default=list(X.columns[:2]) if len(X.columns) >= 2 else list(X.columns),
        key="feature_selector_2"
    )

    if st.button("Train Logistic Regression", key="train_lr"):
        target_type = type_of_target(y_train)
        if target_type == "continuous":
            st.error("Invalid Target Selected: You selected a continuous (numeric) column as the target for a classification model.\n\n👉 Please choose a categorical/discrete column instead.")
            st.stop()
        model = LogisticRegression(C=c_val, penalty=penalty if penalty != "none" else None,
                                   solver=solver, max_iter=1000)
        try:
           model.fit(X_train, y_train)
           preds = model.predict(X_test)
           acc = accuracy_score(y_test, preds)
           st.success(f"Accuracy: {acc:.4f}")

           cm = confusion_matrix(y_test, preds)
           fig, ax = plt.subplots()
           sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax)
           ax.set_title("Confusion Matrix (Logistic Regression)")
           st.pyplot(fig)

           st.text("Classification Report:")
           st.text(classification_report(y_test, preds))

           if len(selected_features) == 2:
                X_vis = x_encoded[selected_features].values  
                X_vis_train, X_vis_test, y_vis_train, y_vis_test = train_test_split(
                    X_vis, y_encoded, test_size=test_size, random_state=random_state
                )

                if do_scale:
                    X_vis_train = scaler.fit_transform(X_vis_train)
                    X_vis_test = scaler.transform(X_vis_test)

                model_vis = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
                model_vis.fit(X_vis_train, y_vis_train)

                x_min, x_max = X_vis_train[:, 0].min() - 1, X_vis_train[:, 0].max() + 1
                y_min, y_max = X_vis_train[:, 1].min() - 1, X_vis_train[:, 1].max() + 1
                xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                                     np.arange(y_min, y_max, 0.02))
                Z = model_vis.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)

                fig_db, ax_db = plt.subplots(figsize=(8, 6))
                ax_db.contourf(xx, yy, Z, alpha=0.4, cmap='coolwarm')
                ax_db.scatter(X_vis_train[:, 0], X_vis_train[:, 1], c=y_vis_train, edgecolor='k', cmap='coolwarm')
                ax_db.set_title(f"KNN Decision Boundary ({selected_features[0]} vs {selected_features[1]})")
                ax_db.set_xlabel(selected_features[0])
                ax_db.set_ylabel(selected_features[1])
                st.pyplot(fig_db)
           else:
                st.info("Please select exactly 2 features to visualize the decision boundary.")

           os.makedirs("models", exist_ok=True)
           with open("models/logistic_regression.pkl", "wb") as f:
               pickle.dump({"model": model, "scaler": scaler, "label_map": label_map}, f)
           with open("models/logistic_regression.pkl", 'rb') as f:
               st.download_button(" Download Model", f, file_name="logistic_regression.pkl")
        except ValueError as e:
              st.error(f"Model Training Failed: {str(e)}")
              st.info("Tip: Logistic Regression doesn’t support NaN values — they’ve been automatically handled, but check dataset if issue persists.")
       

with tab5:
    st.header("Support Vector Machine (SVM)")
    kernel = st.selectbox("Kernel", ["linear", "rbf", "poly", "sigmoid"])
    c_val = st.number_input("Regularization (C)", 0.01, 10.0, 1.0, 0.1)
    gamma = st.selectbox("Gamma", ["scale", "auto"])

    st.markdown("Decision Boundary Visualization Options")
    selected_features = st.multiselect(
        "Select 2 features to visualize decision boundary",
        options=list(X.columns),
        default=list(X.columns[:2]) if len(X.columns) >= 2 else list(X.columns),
        key="feature_selector_3"
    )

    if st.button("Train SVM", key="train_svm"):
        target_type = type_of_target(y_train)
        if target_type == "continuous":
            st.error("Invalid Target Selected: You selected a continuous (numeric) column as the target for a classification model.\n\n👉 Please choose a categorical/discrete column instead.")
            st.stop()
        model = SVC(kernel=kernel, C=c_val, gamma=gamma, probability=True)
        try:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            st.success(f"Accuracy: {acc:.4f}")

            cm = confusion_matrix(y_test, preds)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap="Purples", ax=ax)
            ax.set_title("Confusion Matrix (SVM)")
            st.pyplot(fig)
            st.text("Classification Report:")
            st.text(classification_report(y_test, preds))

            if len(selected_features) == 2:
                X_vis = x_encoded[selected_features].values  
                X_vis_train, X_vis_test, y_vis_train, y_vis_test = train_test_split(
                    X_vis, y_encoded, test_size=test_size, random_state=random_state
                )

                if do_scale:
                    X_vis_train = scaler.fit_transform(X_vis_train)
                    X_vis_test = scaler.transform(X_vis_test)

                model_vis = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
                model_vis.fit(X_vis_train, y_vis_train)

                x_min, x_max = X_vis_train[:, 0].min() - 1, X_vis_train[:, 0].max() + 1
                y_min, y_max = X_vis_train[:, 1].min() - 1, X_vis_train[:, 1].max() + 1
                xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                                     np.arange(y_min, y_max, 0.02))
                Z = model_vis.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)

                fig_db, ax_db = plt.subplots(figsize=(8, 6))
                ax_db.contourf(xx, yy, Z, alpha=0.4, cmap='coolwarm')
                ax_db.scatter(X_vis_train[:, 0], X_vis_train[:, 1], c=y_vis_train, edgecolor='k', cmap='coolwarm')
                ax_db.set_title(f"KNN Decision Boundary ({selected_features[0]} vs {selected_features[1]})")
                ax_db.set_xlabel(selected_features[0])
                ax_db.set_ylabel(selected_features[1])
                st.pyplot(fig_db)
            else:
                st.info("Please select exactly 2 features to visualize the decision boundary.")

            os.makedirs("models", exist_ok=True)
            with open("models/svm.pkl", "wb") as f:
                pickle.dump({"model": model, "scaler": scaler, "label_map": label_map}, f)
            with open("models/svm.pkl", 'rb') as f:
                st.download_button("⬇Download Model", f, file_name="svm.pkl")
        except ValueError as e:
              st.error(f"Model Training Failed: {str(e)}")
              st.info("Tip: SVC doesn’t support NaN values — they’ve been automatically handled, but check dataset if issue persists.")
               

with tab6:
    st.header("Gradient Boosting Classifier (GBM)")
    n_estimators = st.number_input("Number of Estimators", 10, 500, 100, 10)
    learning_rate = st.slider("Learning Rate", 0.01, 1.0, 0.1)
    max_depth = st.slider("Max Depth", 1, 10, 3)

    if st.button("Train Gradient Boosting", key="train_gbm"):
        target_type = type_of_target(y_train)
        if target_type == "continuous":
            st.error("Invalid Target Selected: You selected a continuous (numeric) column as the target for a classification model.\n\n👉 Please choose a categorical/discrete column instead.")
            st.stop()
        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state
        )
        try:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            st.success(f"Accuracy: {acc:.4f}")

            cm = confusion_matrix(y_test, preds)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap="YlGnBu", ax=ax)
            ax.set_title("Confusion Matrix (GBM)")
            st.pyplot(fig)
            st.text("Classification Report:")
            st.text(classification_report(y_test, preds))

            importances = model.feature_importances_
            fig_imp, ax_imp = plt.subplots(figsize=(8, 5))
            sns.barplot(x=importances, y=X.columns, ax=ax_imp, palette="mako")
            ax_imp.set_title("Feature Importances (GBM)")
            st.pyplot(fig_imp)

            os.makedirs("models", exist_ok=True)
            with open("models/gbm.pkl", "wb") as f:
                pickle.dump({"model": model, "scaler": scaler, "label_map": label_map}, f)
            with open("models/gbm.pkl", 'rb') as f:
                st.download_button("⬇Download Model", f, file_name="gbm.pkl")
        except ValueError as e:
              st.error(f"Model Training Failed: {str(e)}")
              st.info("Tip: Gradiant Boosting doesn’t support NaN values — they’ve been automatically handled, but check dataset if issue persists.")

with tab7:
    st.header("XGBoost Classifier")
    if not xgb_available:
        st.warning("XGBoost not installed. Run `pip install xgboost` to enable this feature.")
    else:
        n_estimators = st.number_input("Number of Estimators", 10, 1000, 100, 10, key="xgb_n")
        learning_rate = st.slider("Learning Rate", 0.01, 1.0, 0.1, key="xgb_lr")
        max_depth = st.slider("Max Depth", 1, 10, 3, key="xgb_md")

        if st.button("Train XGBoost", key="train_xgb"):
            target_type = type_of_target(y_train)
            if target_type == "continuous":
               st.error("Invalid Target Selected: You selected a continuous (numeric) column as the target for a classification model.\n\n👉 Please choose a categorical/discrete column instead.")
               st.stop()
            model = XGBClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=random_state
            )
            try:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                acc = accuracy_score(y_test, preds)
                st.success(f"Accuracy: {acc:.4f}")

                cm = confusion_matrix(y_test, preds)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap="Reds", ax=ax)
                ax.set_title("Confusion Matrix (XGBoost)")
                st.pyplot(fig)
                st.text("Classification Report:")
                st.text(classification_report(y_test, preds))

                importances = model.feature_importances_
                fig_imp, ax_imp = plt.subplots(figsize=(8, 5))
                sns.barplot(x=importances, y=X.columns, ax=ax_imp, palette="rocket")
                ax_imp.set_title("Feature Importances (XGBoost)")
                st.pyplot(fig_imp)

                os.makedirs("models", exist_ok=True)
                with open("models/xgboost.pkl", "wb") as f:
                    pickle.dump({"model": model, "scaler": scaler, "label_map": label_map}, f)
                with open("models/xgboost.pkl", 'rb') as f:
                    st.download_button("⬇Download Model", f, file_name="xgboost.pkl")
            except ValueError as e:
                  st.error(f"Model Training Failed: {str(e)}")
                  st.info("Tip: XGBoost doesn’t support NaN values — they’ve been automatically handled, but check dataset if issue persists.")

with tab8:
    st.header("Naive Bayes Classifier")

    nb_type = st.selectbox("Select Naive Bayes Type", 
                           ["GaussianNB", "MultinomialNB", "BernoulliNB"], key="nb_type")
    
    st.markdown("Decision Boundary Visualization Options")
    selected_features = st.multiselect(
        "Select 2 features to visualize decision boundary",
        options=list(X.columns),
        default=list(X.columns[:2]) if len(X.columns) >= 2 else list(X.columns),
        key="feature_selector_4"
    )

    if st.button("Train Naive Bayes", key="train_nb"):
        target_type = type_of_target(y_train)
        if target_type == "continuous":
            st.error("Invalid Target: You selected a continuous (numeric) column as the target for classification.")
            st.stop()

        try:
            if nb_type == "GaussianNB":
                model = GaussianNB()
            elif nb_type == "MultinomialNB":
                model = MultinomialNB()
            else:
                model = BernoulliNB()

            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            acc = accuracy_score(y_test, preds)
            st.success(f"✅ Model Trained Successfully — Accuracy: {acc:.4f}")

            cm = confusion_matrix(y_test, preds)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax)
            ax.set_title(f"{nb_type} - Confusion Matrix")
            st.pyplot(fig)

            st.text("📋 Classification Report:")
            st.text(classification_report(y_test, preds))

            if len(selected_features) == 2:
                X_vis = x_encoded[selected_features].values  
                X_vis_train, X_vis_test, y_vis_train, y_vis_test = train_test_split(
                    X_vis, y_encoded, test_size=test_size, random_state=random_state
                )

                if do_scale:
                    X_vis_train = scaler.fit_transform(X_vis_train)
                    X_vis_test = scaler.transform(X_vis_test)

                model_vis = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
                model_vis.fit(X_vis_train, y_vis_train)

                x_min, x_max = X_vis_train[:, 0].min() - 1, X_vis_train[:, 0].max() + 1
                y_min, y_max = X_vis_train[:, 1].min() - 1, X_vis_train[:, 1].max() + 1
                xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                                     np.arange(y_min, y_max, 0.02))
                Z = model_vis.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)

                fig_db, ax_db = plt.subplots(figsize=(8, 6))
                ax_db.contourf(xx, yy, Z, alpha=0.4, cmap='coolwarm')
                ax_db.scatter(X_vis_train[:, 0], X_vis_train[:, 1], c=y_vis_train, edgecolor='k', cmap='coolwarm')
                ax_db.set_title(f"KNN Decision Boundary ({selected_features[0]} vs {selected_features[1]})")
                ax_db.set_xlabel(selected_features[0])
                ax_db.set_ylabel(selected_features[1])
                st.pyplot(fig_db)
            else:
                st.info("📊Please select exactly 2 features to visualize the decision boundary.")

            # Save Model
            os.makedirs("models", exist_ok=True)
            model_artifact = {"model": model, "scaler": scaler, "label_map": label_map}
            model_path = os.path.join("models", f"{nb_type.lower()}_model.pkl")

            with open(model_path, "wb") as f:
                pickle.dump(model_artifact, f)

            with open(model_path, "rb") as f:
                st.download_button("⬇Download Model", f, file_name=f"{nb_type.lower()}_model.pkl")

        except Exception as e:
            st.error(f"Model Training Failed: {str(e)}")
            st.info("Tip: Ensure all categorical columns are label encoded and no NaN values exist.")


with tab9:
    st.header("AdaBoost Classifier")

    n_estimators = st.slider("Number of Estimators", 10, 500, 50, step=10, key="ada_estimators")
    learning_rate = st.slider("Learning Rate", 0.01, 2.0, 1.0, 0.05, key="ada_lr")

    st.markdown("Decision Boundary Visualization Options")
    selected_features = st.multiselect(
        "Select 2 features to visualize decision boundary",
        options=list(X.columns),
        default=list(X.columns[:2]) if len(X.columns) >= 2 else list(X.columns),
        key="feature_selector_5"
    )

    if st.button("Train AdaBoost", key="train_ada"):
        target_type = type_of_target(y_train)
        if target_type == "continuous":
            st.error("Continuous target selected. Please choose a categorical target.")
            st.stop()

        model = AdaBoostClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=42
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        st.success(f"Accuracy: {acc:.4f}")

        cm = confusion_matrix(y_test, preds)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap="YlOrBr", ax=ax)
        ax.set_title("Confusion Matrix (AdaBoost)")
        st.pyplot(fig)

        st.text("Classification Report:")
        st.text(classification_report(y_test, preds))

        if len(selected_features) == 2:
                X_vis = x_encoded[selected_features].values  
                X_vis_train, X_vis_test, y_vis_train, y_vis_test = train_test_split(
                    X_vis, y_encoded, test_size=test_size, random_state=random_state
                )

                if do_scale:
                    X_vis_train = scaler.fit_transform(X_vis_train)
                    X_vis_test = scaler.transform(X_vis_test)

                model_vis = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
                model_vis.fit(X_vis_train, y_vis_train)

                x_min, x_max = X_vis_train[:, 0].min() - 1, X_vis_train[:, 0].max() + 1
                y_min, y_max = X_vis_train[:, 1].min() - 1, X_vis_train[:, 1].max() + 1
                xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                                     np.arange(y_min, y_max, 0.02))
                Z = model_vis.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)

                fig_db, ax_db = plt.subplots(figsize=(8, 6))
                ax_db.contourf(xx, yy, Z, alpha=0.4, cmap='coolwarm')
                ax_db.scatter(X_vis_train[:, 0], X_vis_train[:, 1], c=y_vis_train, edgecolor='k', cmap='coolwarm')
                ax_db.set_title(f"KNN Decision Boundary ({selected_features[0]} vs {selected_features[1]})")
                ax_db.set_xlabel(selected_features[0])
                ax_db.set_ylabel(selected_features[1])
                st.pyplot(fig_db)
        else:
                st.info("Please select exactly 2 features to visualize the decision boundary.")

        os.makedirs("models", exist_ok=True)
        model_artifact = {"model": model, "scaler": scaler, "label_map": label_map}
        model_path = os.path.join("models", "adaboost.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model_artifact, f)
        with open(model_path, "rb") as f:
            st.download_button("⬇Download AdaBoost Model", f, file_name="adaboost.pkl", key="dl_ada")



with tab10:
    st.header("🌲 Extra Trees Classifier")

    n_estimators = st.slider("Number of Trees", 10, 1000, 100, step=10, key="et_n_estimators")
    max_depth = st.number_input("Max Depth (0 = None)", 0, 100, 0, key="et_max_depth")

    st.markdown("Decision Boundary Visualization Options")
    selected_features = st.multiselect(
        "Select 2 features to visualize decision boundary",
        options=list(X.columns),
        default=list(X.columns[:2]) if len(X.columns) >= 2 else list(X.columns),
        key="feature_selector_6"
    )

    if st.button("🚀 Train Extra Trees", key="train_et"):
        target_type = type_of_target(y_train)
        if target_type == "continuous":
            st.error("Continuous target selected. Please choose a categorical target.")
            st.stop()

        model = ExtraTreesClassifier(
            n_estimators=int(n_estimators),
            max_depth=None if int(max_depth) == 0 else int(max_depth),
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        st.success(f"Accuracy: {acc:.4f}")

        cm = confusion_matrix(y_test, preds)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap="BuGn", ax=ax)
        ax.set_title("Confusion Matrix (Extra Trees)")
        st.pyplot(fig)

        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        features_sorted = [X.columns[i] for i in indices]
        fig_imp, ax_imp = plt.subplots(figsize=(8, 5))
        sns.barplot(x=importances[indices], y=features_sorted, palette="cubehelix", ax=ax_imp)
        ax_imp.set_title("Feature Importance (Extra Trees)")
        ax_imp.set_xlabel("Importance Score")
        st.pyplot(fig_imp)

        if len(selected_features) == 2:
                X_vis = x_encoded[selected_features].values
                X_vis_train, X_vis_test, y_vis_train, y_vis_test = train_test_split(
                    X_vis, y_encoded, test_size=test_size, random_state=random_state
                )

                if do_scale:
                    X_vis_train = scaler.fit_transform(X_vis_train)
                    X_vis_test = scaler.transform(X_vis_test)

                model_vis = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
                model_vis.fit(X_vis_train, y_vis_train)

                x_min, x_max = X_vis_train[:, 0].min() - 1, X_vis_train[:, 0].max() + 1
                y_min, y_max = X_vis_train[:, 1].min() - 1, X_vis_train[:, 1].max() + 1
                xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                                     np.arange(y_min, y_max, 0.02))
                Z = model_vis.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)

                fig_db, ax_db = plt.subplots(figsize=(8, 6))
                ax_db.contourf(xx, yy, Z, alpha=0.4, cmap='coolwarm')
                ax_db.scatter(X_vis_train[:, 0], X_vis_train[:, 1], c=y_vis_train, edgecolor='k', cmap='coolwarm')
                ax_db.set_title(f"KNN Decision Boundary ({selected_features[0]} vs {selected_features[1]})")
                ax_db.set_xlabel(selected_features[0])
                ax_db.set_ylabel(selected_features[1])
                st.pyplot(fig_db)
        else:
                st.info("Please select exactly 2 features to visualize the decision boundary.")

        # Save Model
        os.makedirs("models", exist_ok=True)
        model_artifact = {"model": model, "scaler": scaler, "label_map": label_map}
        model_path = os.path.join("models", "extra_trees.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model_artifact, f)
        with open(model_path, "rb") as f:
            st.download_button("⬇Download Extra Trees Model", f, file_name="extra_trees.pkl", key="dl_et")




st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align:center; color:#ff9933;'>Regression Models — Predict Continuous Values</h2>", unsafe_allow_html=True)

tabr1, tabr2, tabr3, tabr4, tabr5, tabr6, tabr7, tabr8, tabr9 = st.tabs([
    " Linear Regression", 
    " Decision Tree Regressor", 
    " Random Forest Regressor",
    " Ridge Regression",
    " Lasso Regression",
    " Gradient Boosting Regressor",
    " XGBoost Regressor",
    " SVR (Support Vector Regressor)",
    " KNN Regressor"
])

def train_and_display(model, model_name):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    st.success(f"{model_name} R² Score: {r2:.4f}")
    st.write(f"MSE: {mse:.4f}")
    st.write(f"MAE: {mae:.4f}")
    

    fig, ax = plt.subplots()
    ax.scatter(y_test, preds, alpha=0.7, color="#007acc")
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"Actual vs Predicted ({model_name})")
    st.pyplot(fig)
    
    os.makedirs("models", exist_ok=True)
    model_artifact = {"model": model, "scaler": scaler}
    model_path = os.path.join("models", f"{model_name.lower().replace(' ', '_')}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model_artifact, f)
    with open(model_path, "rb") as f:
        st.download_button(f"Download {model_name}", f, file_name=f"{model_name.lower().replace(' ', '_')}.pkl")

def has_missing_values(X):
    if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
        return X.isnull().sum().sum() > 0
    elif isinstance(X, np.ndarray):
        return np.isnan(X).sum() > 0
    else:
        return False

with tabr1:
    st.header(" Linear Regression")
    if st.button("Train Linear Regression", key="train_lre"):
        if has_missing_values(X_train):
            st.error("Linear Regression cannot handle missing values (NaN). Please fill or remove NaNs before training.")
        else:
            model = LinearRegression()
            train_and_display(model, "Linear Regression")

with tabr2:
    st.header("Decision Tree Regressor")
    max_depth = st.number_input("Max Depth (0 = None)", 0, 100, 0, key="dtr_max_depth")
    if st.button("Train Decision Tree Regressor", key="train_dtr"):
        model = DecisionTreeRegressor(max_depth=None if int(max_depth)==0 else int(max_depth), random_state=42)
        train_and_display(model, "Decision Tree Regressor")

with tabr3:
    st.header("Random Forest Regressor")
    n_estimators = st.slider("Number of Trees", 10, 1000, 100, step=10, key="rfr_estimators")
    max_depth = st.number_input("Max Depth (0 = None)", 0, 100, 0, key="rfr_max_depth")
    if st.button("Train Random Forest Regressor", key="train_rfr"):
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=None if int(max_depth)==0 else int(max_depth),
                                      random_state=42, n_jobs=-1)
        train_and_display(model, "Random Forest Regressor")

with tabr4:
    st.header("Ridge Regression")
    alpha = st.slider("Regularization Strength (alpha)", 0.0, 10.0, 1.0, 0.1, key="ridge_alpha")
    if st.button("Train Ridge Regression", key="train_ridge"):
        if has_missing_values(X_train):
            st.error("Ridge Regression cannot handle missing values (NaN). Please fill or remove NaNs before training.")
        else:
            model = Ridge(alpha=alpha)
            train_and_display(model, "Ridge Regression")

with tabr5:
    st.header("Lasso Regression")
    alpha = st.slider("Regularization Strength (alpha)", 0.0, 10.0, 1.0, 0.1, key="lasso_alpha")
    if st.button("Train Lasso Regression", key="train_lasso"):
        if has_missing_values(X_train):
            st.error("Lasso Regression cannot handle missing values (NaN). Please fill or remove NaNs before training.")
        else:
            model = Lasso(alpha=alpha)
            train_and_display(model, "Lasso Regression")

with tabr6:
    st.header("Gradient Boosting Regressor")
    n_estimators = st.slider("Number of Estimators", 50, 500, 100, key="gbr_estimators")
    learning_rate = st.slider("Learning Rate", 0.01, 1.0, 0.1, key="gbr_lr")
    if st.button("Train Gradient Boosting", key="train_gbr"):
        if has_missing_values(X_train):
            st.error("Gradient Boosting cannot handle missing values (NaN). Please fill or remove NaNs before training.")
        else:
            model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
            train_and_display(model, "Gradient Boosting Regressor")

with tabr7:
    st.header("XGBoost Regressor")
    n_estimators = st.slider("Number of Estimators", 50, 1000, 100, step=50, key="xgb_estimators")
    learning_rate = st.slider("Learning Rate", 0.01, 1.0, 0.1, key="xgb_lre")
    if st.button("Train XGBoost Regressor", key="train_xgbe"):
        if has_missing_values(X_train):
            st.error("XGBoost cannot handle missing values (NaN). Please fill or remove NaNs before training.")
        else:
            model = xgb.XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42, n_jobs=-1)
            train_and_display(model, "XGBoost Regressor")

with tabr8:
    st.header("📘 Support Vector Regressor (SVR)")
    kernel = st.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"], key="svr_kernel")
    C = st.slider("Regularization (C)", 0.1, 10.0, 1.0, key="svr_C")
    if st.button("Train SVR", key="train_svr"):
        if has_missing_values(X_train):
            st.error("SVR cannot handle missing values (NaN). Please fill or remove NaNs before training.")
        else:
            model = SVR(kernel=kernel, C=C)
            train_and_display(model, "SVR Regressor")

with tabr9:
    st.header("KNN Regressor")
    n_neighbors = st.slider("Number of Neighbors (K)", 1, 20, 5, key="knn_neighborsr")
    if st.button("Train KNN Regressor", key="train_knnr"):
        if has_missing_values(X_train):
            st.error("KNN cannot handle missing values (NaN). Please fill or remove NaNs before training.")
        else:
            model = KNeighborsRegressor(n_neighbors=n_neighbors)
            train_and_display(model, "KNN Regressor")


st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;'>Built by Harsh Shakya</h4>", unsafe_allow_html=True)
