import numpy as np
from sklearn import metrics,neighbors
import sklearn
from collections import Counter
#-------------------------------------------------------------------------
'''
    Problem 1: k nearest neighbor 
    In this problem, you will implement a classification method using k nearest neighbors. 
    The main goal of this problem is to get familiar with the basic settings of classification problems. 
    KNN is a simple method for classification problems.

    You need to install following python package:
        nose
        numpy 
        scikit-learn
    You could type the following line in the terminal to install the package:
        pip3 install numpy 
        pip3 install nose 
        pip3 install sklearn

'''

#--------------------------
def Terms_and_Conditions():
    ''' 
        By submitting this homework or changing this function, you agree with the following terms:
       (1) Not sharing your code/solution with any student before and after the homework due. For example, sending your code segment to another student, putting your solution online or lending your laptop (if your laptop contains your solution or your Dropbox automatically synchronize your solution between your home computer and your laptop) to another student to work on this homework will violate this term.
       (2) Not using anyone's code in this homework, build your own solution. For example, using some code segments from another student or online resources due to any reason (like too busy recently) will violate this term. Changing other people's code as your solution (such as changing the variable names) will also violate this term.
       (3) When discussing with any other student about this homework, only discuss high-level ideas or using pseudo-code. Don't discuss about the solution at the code level. For example, discussing with another student about the solution of a function (which needs 5 lines of code to solve), and then working on the solution "independently", however the code of the two solutions are exactly the same, or only with minor differences  (like changing variable names) will violate this term.
      All violations of (1),(2) or (3) will be handled in accordance with the WPI Academic Honesty Policy.  For more details, please visit: https://www.wpi.edu/about/policies/academic-integrity/dishonesty
      Historical Data: in one year, we ended up finding 25% of the students in the class violating this term in their homework submissions and we handled ALL of these violations according to the WPI Academic Honesty Policy. 
    '''
    #****************************************
    ## CHANGE CODE HERE
    Read_and_Agree = True  #if you have read and agree with the term above, change "False" to "True".
    #****************************************
    return Read_and_Agree
 
#--------------------------
def compute_distance(Xtrain, Xtest):
    '''
        compute the Euclidean distance between instances in a test set and a training set 
        Input:
            Xtrain: the feature matrix of the training dataset, a float python matrix of shape (n_train by p). Here n_train is the number of data instance in the training set, p is the number of features/dimensions.
            Xtest: the feature matrix of the test dataset, a float python matrix of shape (n_test by p). Here n_test is the number of data instance in the test set, p is the number of features/dimensions.
        Output:
            D: the distance between instances in Xtest and Xtrain, a float python matrix of shape (ntest, ntrain), the (i,j)-th element of D represents the Euclidean distance between the i-th instance in Xtest and j-th instance in Xtrain.

        For example, if we have a training set of 3 instances with 2 dimensional features:
            Xtrain = 1, 2
                     2, 1
                     2, 2
        We also have a test dataset of 2 isntances:
            Xtest =  3, 4
                     5, 6
        The Euclidean distance between the first test instance (1,2) and the first training instance (3,4) is computed as:
            D[0,0] = square_root( (1-3)^2 + (2-4)^2 )  = 2.828 
        Similarly we can compute all the pairs between Xtest and Xtrain, then D is a matrix of shape 2 X 3:
            D[i,j] is the Euclidean distance between the i-th instance in Xtest and the j-th instance in Xtrain.

    '''
    #########################################
    ## INSERT YOUR CODE HERE

    D = sklearn.metrics.pairwise.euclidean_distances(Xtest,Xtrain)
    
    #########################################
    return D 


    ''' TEST: Now you can test the correctness of your code above by typing `nosetests -v test1.py:test_compute_distance' in the terminal.  '''




#--------------------------
def k_nearest_neighbor(Xtrain, Ytrain, Xtest, K = 3):
    '''
        compute the labels of test data using the K nearest neighbor classifier.
        Input:
            Xtrain: the feature matrix of the training dataset, a float numpy matrix of shape (n_train by p). Here n_train is the number of data instance in the training set, p is the number of features/dimensions.
            Ytrain: the label vector of the training dataset, an integer python list of length n_train. Each element in the list represents the label of the training instance. The values can be 0, ..., or num_class-1. num_class is the number of classes in the dataset.
            Xtest: the feature matrix of the test dataset, a float python matrix of shape (n_test by p). Here n_test is the number of data instance in the test set, p is the number of features/dimensions.
            K: the number of neighbors to consider for classification.
        Output:
            Ytest: the predicted labels of test data, an integer numpy vector of length ntest.
        Note: you cannot use any existing package for KNN classifier.
    '''
    #########################################
    ## INSERT YOUR CODE HERE

    # number of testing instances
    numtest = len(Xtest)


    # compute distances between test set and training set
    
    D = compute_distance(Xtrain,Xtest)

    smallest = np.array([np.argpartition(row,K-1)[:K] for row in D])

    Ytest = []
    print(smallest)
    for s in smallest:
        idx = s[K-2]
        print("i",idx)
        Ytest.append(Ytrain[idx])

    Ytest = np.array(Ytest)
    #########################################
    return Ytest 

    ''' TEST: Now you can test the correctness of your code above by typing `nosetests -v test1.py:test_k_nearest_neighbor' in the terminal.  '''




#--------------------------------------------

''' TEST Problem 2: 
        Now you can test the correctness of all the above functions by typing `nosetests -v test1.py' in the terminal.  

        If your code passed all the tests, you will see the following message in the terminal:
            ----------- Problem 1 (10 points in total)-------------- ... ok
            (3 points) compute_distance ... ok
            (5 points) k_nearest_neighbor ... ok
            (2 points) test on a dataset ... ok
            ----------------------------------------------------------------------
            Ran 5 tests in 0.336s
            
            OK




    ERROR Message: 
        If your code has an error, you will see an error message with the line number of the error:
        For example:

            ======================================================================
            FAIL: (3points) test_compute_distance()
            ----------------------------------------------------------------------
            Traceback (most recent call last):
              File "/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/nose/case.py", line 198, in runTest
                self.test(*self.arg)
              File "test1.py", line 47, in test_compute_distance
                assert np.allclose(D, D_true, atol = 1e-4)            
              AssertionError

        This error message means:
            (1) You are using python 3.6, 
                    See: "... Versions/3.6/lib/python3.6/ ... "
            (2) Your code failed in Line 47, the test_compute_distance function in test1.py file   
                    See: " ...  File "test1.py", line 47, in test_compute_distance "
            (3) The specific test that failed is that the D should equals to the true answer D_true, but in your code, a different result is returned.
                    See: "  assert np.allclose(D, D_true, atol = 1e-4)              
                            AssertionError "

    Debug: 

        To debug your code, you could insert a print statement before Line 47 of test1.py file:
            print(D)
        Then run the test again.

        Now after the error message, the value of D will be printed like this:

            -------------------- >> begin captured stdout << ---------------------
           [[1.         1.4        1.        ]
            [2.         2.         2.        ]] 
            --------------------- >> end captured stdout << ----------------------

        Then we know that the value of D output by your current code.



'''




#--------------------------------------------


