#SVM Python implementation

#read in svm data
svm_dat = numpy.genfromtxt('svm.csv',delimiter = ',')
class1 = svm_dat[0:5]
class2 = svm_dat[5:10]
plt.figure()
plt.scatter(class1[:,0],class1[:,1], c = 'blue', marker = 'o')
plt.scatter(class2[:,0],class2[:,1], c = 'red', marker = 'x')

#plot the data, support vectors and the seperating hyperplane

from sklearn import svm
X = np.column_stack((svm_dat[:,0],svm_dat[:,1]))
y = svm_dat[:,2]
clf = svm.SVC(kernel='linear')
clf.fit(X, y)
clf.support_vectors_
plt.scatter(class1[:,0],class1[:,1], c = 'blue', marker = 'o',s=20)
plt.scatter(class2[:,0],class2[:,1], c = 'red', marker = 'x',s=20)
plt.scatter(clf.support_vectors_[0,0],clf.support_vectors_[0,1],c='blue',marker = '*',s=200)
plt.scatter(clf.support_vectors_[1,0],clf.support_vectors_[1,1],c='red',marker = '<', s = 100)
#Get the separating hyperplane
w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-15, 10)
yy = a * xx - (clf.intercept_[0]) / w[1]
t = numpy.linspace(-15,10,100) # 100 linearly spaced numbers
plt.plot(xx,yy)
