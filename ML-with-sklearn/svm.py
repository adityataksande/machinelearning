import matplotlib.pyplot as plt
from sklearn import datasets,svm

digits  = datasets.load_digits() # loading the data

clf = svm.SVC(gamma=0.0001,C=100)  # tuning the model

print(len(digits.data))

X,y = digits.data[:-1], digits.target[:-1] # getting ready with or input and target

clf.fit(X,y) # fitting model

print("Prediction :" , clf.predict([digits.data[-8]])) # making prediction

plt.imshow(digits.images[-8],cmap=plt.cm.gray_r, interpolation="nearest") # getting an image of the predicted value
plt.show() # showing the image