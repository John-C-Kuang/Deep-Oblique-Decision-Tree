import preprocess
import data_partition as dp
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main():
    data = preprocess.preprocess_wine_quality()
    X_train, X_test, y_train, y_test = dp.partition_data(data, "quality")
    # dt_classifier = RandomForestClassifier(random_state=42)
    dt_classifier = KNeighborsClassifier(n_neighbors=3, metric="cosine")
    dt_classifier.fit(X_train, y_train)

    y_pred = dt_classifier.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))





if __name__ == "__main__":
    main()

