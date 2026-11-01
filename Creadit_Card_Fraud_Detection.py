import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import NearMiss
    IMB_AVAILABLE = True
except:
    IMB_AVAILABLE = False

st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="wide")
st.title("💳 14. Credit Card Fraud Detection")
st.markdown("**Project Idea:** Fraud detection as classification | 284,807 transactions (0.172% fraud) | SMOTE + NearMiss | LR, RF, SVM, DT, KNN | ROC & Precision-Recall")

st.sidebar.header("Project Info")
st.sidebar.markdown("""
**Dataset:** European Cardholders (Kaggle - ULB)
**Tools:** Python, Scikit-learn, Pandas, Seaborn, Matplotlib, imbalanced-learn
**Techniques:** SMOTE, NearMiss, Cross-validation, ROC, Precision-Recall
""")

@st.cache_data
def generate_synthetic_data(n_samples=5000):
    np.random.seed(42)
    n_fraud = int(n_samples * 0.0172)
    n_normal = n_samples - n_fraud
    def create_data(n, fraud=False):
        data = {}
        data['Time'] = np.random.uniform(0, 172792, n)
        for i in range(1, 29):
            data[f'V{i}'] = np.random.normal(loc=(-2 if fraud else 0), scale=1.5, size=n)
        data['Amount'] = np.random.exponential(120 if fraud else 50, n)
        data['Class'] = np.ones(n) if fraud else np.zeros(n)
        return pd.DataFrame(data)
    df_normal = create_data(n_normal, False)
    df_fraud = create_data(n_fraud, True)
    df = pd.concat([df_normal, df_fraud]).sample(frac=1, random_state=42).reset_index(drop=True)
    return df

uploaded = st.file_uploader("Dataset upload karo (creditcard.csv) ya demo data use karo", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    st.success(f"Dataset Loaded: {df.shape[0]} rows")
else:
    st.info("Demo data generate ho raha hai (Real dataset 284,807 rows ka hota hai, structure same hai)")
    df = generate_synthetic_data(5000)

col1, col2, col3 = st.columns(3)
col1.metric("Total Transactions", df.shape[0])
col2.metric("Fraud Cases", int(df['Class'].sum()))
col3.metric("Fraud %", f"{df['Class'].sum()/len(df)*100:.3f}%")

fig, ax = plt.subplots()
sns.countplot(x='Class', data=df, ax=ax)
ax.set_xticklabels(['Normal (0)', 'Fraud (1)'])
ax.set_title("Extreme Class Imbalance (0.172% fraud)")
st.pyplot(fig)

X = df.drop('Class', axis=1)
y = df['Class']
scaler = StandardScaler()
X[['Time', 'Amount']] = scaler.fit_transform(X[['Time', 'Amount']])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

st.subheader("🔧 Handling Class Imbalance via SMOTE & NearMiss")
if IMB_AVAILABLE:
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    st.write(f"After SMOTE Oversampling: Balanced {np.bincount(y_train_smote.astype(int))}")
    nearmiss = NearMiss(version=1)
    X_train_nm, y_train_nm = nearmiss.fit_resample(X_train, y_train)
    st.write(f"After NearMiss Undersampling: {np.bincount(y_train_nm.astype(int))}")
else:
    st.error("imbalanced-learn install karo: py -3.11 -m pip install imbalanced-learn")
    X_train_smote, y_train_smote = X_train, y_train

X_tr, y_tr = X_train_smote, y_train_smote

st.subheader("🤖 Compare Classifiers - Logistic Regression, Random Forest, SVM, Decision Tree, KNN")
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(probability=True, kernel='rbf', random_state=42)
}
selected = st.multiselect("Classifiers select karo", list(classifiers.keys()), default=["Logistic Regression", "Random Forest"])

results = []
roc_fig, ax_roc = plt.subplots()
pr_fig, ax_pr = plt.subplots()

for name in selected:
    clf = classifiers[name]
    cv = StratifiedKFold(n_splits=3)
    cv_scores = cross_val_score(clf, X_tr, y_tr, cv=cv, scoring='f1')
    clf.fit(X_tr, y_tr)
    y_proba = clf.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    results.append({"Model": name, "CV F1": round(cv_scores.mean(),3), "ROC AUC": round(roc_auc,3)})
    ax_roc.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")
    ax_pr.plot(recall, precision, label=f"{name}")

ax_roc.plot([0,1],[0,1],'k--')
ax_roc.set_title("ROC Curves"); ax_roc.set_xlabel("FPR"); ax_roc.set_ylabel("TPR"); ax_roc.legend()
st.pyplot(roc_fig)
ax_pr.set_title("Precision-Recall Curves"); ax_pr.set_xlabel("Recall"); ax_pr.set_ylabel("Precision"); ax_pr.legend()
st.pyplot(pr_fig)

if results:
    st.dataframe(pd.DataFrame(results).sort_values("ROC AUC", ascending=False))

if selected:
    best_name = max(results, key=lambda x: x["ROC AUC"])["Model"] if results else selected[0]
    st.subheader(f"📊 Detailed Report - Best Model: {best_name}")
    best_clf = classifiers[best_name]
    best_clf.fit(X_tr, y_tr)
    y_pred_best = best_clf.predict(X_test)
    st.text(classification_report(y_test, y_pred_best))
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred_best), annot=True, fmt='d', ax=ax_cm, cmap='Blues')
    ax_cm.set_title("Confusion Matrix")
    st.pyplot(fig_cm)