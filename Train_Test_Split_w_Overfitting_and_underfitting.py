# After creating arrays for the features and target variable, 
# i will split them into training and test sets, 
# fit a k-NN classifier to the training data, 
# and then compute its accuracy using the .score() method.

# Import necessary modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Create feature and target arrays
# Create an array for the features using digits.data and an array for the target using digits.target.
X = digits.data
y = digits.target

# Split into training and test set
# Create stratified training and test sets using 0.2 for the size of the test set. 
# Use a random state of 42. 
# Stratify the split according to the labels 
# so that they are distributed in the training and test sets as they are in the original dataset.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify=y)


# The training and testing sets are available to you in the workspace as:
# X_train, X_test, y_train, y_test. 
# In addition, KNeighborsClassifier has been imported from sklearn.neighbors.


# Setup arrays to store train and test accuracies
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Loop over different values of k, 
# this will be the factor each each accuray list (training and testing) will will use in their measurement.
for i, k in enumerate(neighbors):
    # Fit the classifier with k neighbors to the training data.
    # First, Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors=k)

    # Secondly, Fit the classifier to the training data
    knn.fit(X_train, y_train)
    
    # Compute accuracy scores the training set and test set
    # this is to be done separately using the .score() method 
    # and assign the results to the train_accuracy and test_accuracy arrays respectively.
    
    #Compute accuracy on the training set,
    train_accuracy[i] = knn.score(X_train, y_train)

    #Compute accuracy on the testing set
    test_accuracy[i] = knn.score(X_test, y_test)

# Generate plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()
