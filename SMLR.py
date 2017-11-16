import sklearn.datasets
import sklearn.linear_model
import numpy as np
import matplotlib.pyplot
from sklearn.cluster import KMeans

def rmse(A):
     sum = 0
     for i in range(len(A)):
         sum = sum + A[i]**2
     return (sum/len(A))**0.5


if __name__ == "__main__":
# Load boston dataset
    boston = sklearn.datasets.load_boston()
    # Make the training data

# Shuffle the whole data
    shuffleIdx = range(500)
    np.random.shuffle(shuffleIdx)
    test_features = boston.data[shuffleIdx[:100]]
    print "test_feature"
    print test_features[0]
    test_targets  = boston.target[shuffleIdx[:100]] 
# Train with Cross Validation
    ridgeRegression = sklearn.linear_model.RidgeCV(alphas=[0.01, 0.05, 0.1,0.5, 1.0, 10.0])
# creat mat_coef and mat_result
    mat =  np.zeros((13,10))
    mat_result = np.zeros((100,10))
    index = 50
    for i in range(10):
	shuffleIdx = range(500)
        np.random.shuffle(shuffleIdx)
        train_features = boston.data[:100]
        train_targets = boston.target[:100]
        ridgeRegression.fit(train_features , train_targets )
        mat[:,i] = ridgeRegression.coef_
        mat_result[:,i] = ridgeRegression.predict(test_features)
#test
    mean_mat = np.mean(mat[0])
    covariancemat = np.cov(mat.T)
    print "xie fang cha juzhen is "
    print covariancemat
#compute tezheng xiangliang    
    a,u=np.linalg.eig(covariancemat)
    print "tezhengzhi = "
    print a
    u0 = np.abs(u[0,:])
# 1 wei change into n wei
    u1 = np.array(u0).reshape(len(u0),1)
    print "u="
    print u1
# label
    kmean = KMeans(n_clusters=3).fit(u1)
    label = kmean.labels_
    centor = kmean.cluster_centers_
    print "label ="
    print label
    print "centor = "
    print centor
    throw = np.where(centor== np.max(centor))
    print "throw[0,0]"
    print throw[0][0]
# sum_u
    sum_u  = 0
    for i  in range (10):
        if( label[i] == throw[0][0]):
            sum_u = sum_u
        else:
            sum_u = sum_u + u1[i]
    print "sum_u is "
    print  sum_u
#sum_result
    sum_result = np.zeros((100,1))
    SMLR_result = np.zeros((100,1))
    for j in range(100):
        for i in range(10):
            if( label[i] == throw[0][0]):
                sum_result[j] = sum_result[j]
            else:
                sum_result[j] = sum_result[j] + mat_result[j,i]*u1[i]
        SMLR_result[j] =sum_result[j] / sum_u
    test_result = np.array(test_targets).reshape(len(test_targets),1)
    cha_SMLR = test_result - SMLR_result
    std_SMLR = rmse(cha_SMLR)
# average
    average_result = np.zeros((100,1))
    for i in range (100):
        average_result[i] = np.mean(mat_result[i])
    cha_average = test_result - average_result
    std_average = rmse(cha_average)
# mid
    mid_result = np.zeros((100,1))
    for i in range (100):
        mid_result[i] = np.median(mat_result[i])
    cha_mid = test_result - mid_result
    std_mid = rmse(cha_mid)

    print std_SMLR
    print std_average
    print std_mid

# plot
    X = np.linspace(0,100,100)
    matplotlib.pyplot.plot(X, cha_SMLR,'r',label = 'SMLR')
    matplotlib.pyplot.plot(X, cha_average,'*',label = 'average')
    matplotlib.pyplot.plot(X, cha_mid,'b',label = 'mid')
    matplotlib.pyplot.legend(loc='upper center', shadow=True)
    #matplotlib.pyplot.show()



                
