import numpy as np
import matplotlib.pyplot as plt
import copy
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import RepeatedKFold
random.seed(0)
def load_train_dataset():

    #each element in attribute_txt is a attribute list representing a sample
    attribute_text = np.loadtxt("//Users//mengtianao//Documents//ece602//project/UCI HAR Dataset//train//X_train.txt")
    label_text = np.loadtxt("//Users//mengtianao//Documents//ece602//project//UCI HAR Dataset//train//y_train.txt")
    
    return attribute_text, label_text

def load_test_dataset():
    attribute_text = np.loadtxt("//Users//mengtianao//Documents//ece602//project/UCI HAR Dataset//test//X_test.txt")
    label_text = np.loadtxt("//Users//mengtianao//Documents//ece602//project//UCI HAR Dataset//test//y_test.txt")

    return attribute_text, label_text

def KNN(attribute_train, label_train, num_neighbors):
    KNN_classfier = KNeighborsClassifier(n_neighbors=num_neighbors)
    KNN_classfier.fit(attribute_train, label_train)
    return KNN_classfier

def KNN_predict(KNN_classfier, attribute_ele):
    predict = KNN_classfier.predict([attribute_ele])
    return predict

def cal_error_rate(attribute_test, label_test, KNN_classfier):
    count = 0
    error_KNN = 0
    for i in attribute_test:
        predict = KNN_predict(KNN_classfier, i)
        if (predict != label_test[count]):
            error_KNN += 1
            count += 1
            continue
        count += 1

    error_rate_KNN = error_KNN / len(attribute_test)
    return error_rate_KNN

# Vi (t + 1) = w * Vi(t) + r1 * c1 * (Pi - Xi) + r2 * c2 * (Gi - Xi)
# r1 and r2 are uniformed distributed between 0 and 1
# w * Vi(t) is called inertia term, w is inertia coefficient
# c1 and c2 are acceleration coefficient
# r1 * c1 * (Pi - Xi) is called cognitive component
# r2 * c2 * (Gi - Xi) is called social component
# Xi (t + 1) = Xi(t) + V(t + 1)


def cost_func(attribute_train, label_train, attribute_test, label_test, num_neighbors):
    KNN_classfier = KNN(attribute_train, label_train, num_neighbors)
    error_rate = cal_error_rate(attribute_test, label_test, KNN_classfier)
    return error_rate






def normal(feature):
    feature_copy = copy.deepcopy(feature)


    sample_max = np.max(feature_copy, axis=0)
    print(len(sample_max))
    for i in range(len(sample_max)):
        sum = 0
        count = 0
        for sample in feature_copy:
            sum += sample[i]
            count += 1

        mean = sum / count
        print("mean: ", mean)

        for sample in feature_copy:
            sample[i] = (sample[i] - mean) / sample_max[i]



    return feature_copy



class particle_class:

    # position is k,
    def __init__(self, Position = -1, Velocity = -1, test_tool = -1 , Best = [-1, -1]):

        self.Position = Position
        self.Velocity = Velocity
        self.test_tool = test_tool
        self.Best = Best


class parameters:

    # position is k,
    def __init__(self, MaxIt = -1, nPOP = -1, max_num_neighbors = -1, min_num_neighbors = -1, w = -1.0, w_damp = -1.0, c1 = -1, c2 = -1):

        self.MaxIt = MaxIt
        self.nPOP = nPOP
        self.max_num_neighbors = max_num_neighbors
        self.min_num_neighbors = min_num_neighbors
        self.w = w
        self.w_damp = w_damp
        self.c1 = c1
        self.c2 = c2
        self.max_vel = (max_num_neighbors - min_num_neighbors) * 0.2
        self.min_vel = -self.max_vel

class Train_Test_data:

    # position is k,
    def __init__(self, attribute_train = None, label_train = None, attribute_test = None , label_test = None):

        self.attribute_train = attribute_train
        self.label_train = label_train
        self.attribute_test = attribute_test
        self.label_test = label_test

# Vi (t + 1) = w * Vi(t) + r1 * c1 * (Pi - Xi) + r2 * c2 * (Gi - Xi)
# r1 and r2 are uniformed distributed between 0 and 1
# w * Vi(t) is called inertia term, w is inertia coefficient
# c1 and c2 are acceleration coefficient
# r1 * c1 * (Pi - Xi) is called cognitive component
# r2 * c2 * (Gi - Xi) is called social component
# Xi (t + 1) = Xi(t) + V(t + 1)

def PSO(Data, parameters):
    # Initialization
    GlobalBest = [-1, float("inf")] #[position, cost]
    Population = []
    for i in range(parameters.nPOP):
        #print(i)
        num_neighbors = np.random.randint(1, parameters.max_num_neighbors)
        velocity = 0
        test_res = cost_func(Data.attribute_train, Data.label_train, Data.attribute_test, Data.label_test, num_neighbors)
        Best_pos = num_neighbors
        Best_test_res = test_res
        particle  = particle_class(Position = num_neighbors, Velocity = velocity, test_tool = test_res, Best = [Best_pos, Best_test_res])
        #print("particle.best: ", particle.Best[1])
        if (particle.Best[1] < GlobalBest[1]):
            GlobalBest[0] = particle.Best[0]
            GlobalBest[1] = particle.Best[1]
        Population.append(particle)


    GlobalBests = [GlobalBest] #hold the best globalbest at each iteration
    #main loop of PSO
    for i in range(parameters.MaxIt):
        #print(i)
        for j in range(parameters.nPOP):
            #update velocity of ith particle
            Population[j].Velocity = parameters.w * Population[j].Velocity + parameters.c1 * np.random.uniform(0,1) * (Population[j].Best[0] - Population[j].Position) + parameters.c2 * np.random.uniform(0,1) * (GlobalBest[0] - Population[j].Position)

            #apply velocity limit
            Population[j].Velocity = max(Population[j].Velocity, parameters.min_vel)
            Population[j].Velocity = min(Population[j].Velocity, parameters.max_vel)
            #print("velocity: ", Population[j].Velocity)
            # update position of ith particle
            Population[j].Position = int(Population[j].Position + Population[j].Velocity)
            #apply upper and lower bound
            Population[j].Position = max(Population[j].Position, parameters.min_num_neighbors)
            Population[j].Position = min(Population[j].Position, parameters.max_num_neighbors)
            #print("position: ", Population[j].Position)


            ##update cost of ith particle
            test_res = cost_func(Data.attribute_train, Data.label_train, Data.attribute_test, Data.label_test, Population[j].Position)
            Population[j].test_tool = test_res

            # update personal best of ith particle
            if (test_res < Population[j].Best[1]):
                Population[j].Best[0] = Population[j].Position
                Population[j].Best[1] = test_res

                # update global best of ith particle
                if (Population[j].Best[1] < GlobalBest[1]):

                    GlobalBest[0] = Population[j].Best[0]
                    GlobalBest[1] = Population[j].Best[1]

        # store the best cost
        GlobalBests.append(GlobalBest)
        #print("iteration: ", i)
        #print("global best: ", GlobalBest)

        #damp the w
        parameters.w = parameters.w * parameters.w_damp

    return Population, GlobalBests, GlobalBest

def single_cross(X,y, nPOP, MaxIt):
    error_rate_list = []
    rkf = RepeatedKFold(n_splits=10, n_repeats=1, random_state=0)
    count = 0
    for train_index, test_index in rkf.split(X):
        print("count: ", count)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # Parameters of PSO
        min_num_neighbors = 1
        max_num_neighbors = 40  # 4- 40
        # standard PSO
        w = 1  # inertia coefficient
        w_damp = 0.99  # damping ratio of inertia coefficient
        c1 = 2  # personal acceleration coefficient
        c2 = 2  # social acceleration coefficient
        parameter = parameters(MaxIt=MaxIt, nPOP=nPOP, max_num_neighbors=max_num_neighbors,
                                min_num_neighbors=min_num_neighbors, w=w, w_damp=w_damp, c1=c1, c2=c2)
        Data = Train_Test_data(attribute_train=X_train, label_train=y_train, attribute_test=X_test,
                               label_test=y_test)
        Population, GlobalBests, GlobalBest = PSO(Data, parameter)
        error_rate_list.append(GlobalBest[1])
        count += 1
    #print("error list: ", error_rate_list)
    sum = 0
    count = 0
    for i in error_rate_list:
        sum += i
        count += 1

    mean_error_rate = sum / count
    return mean_error_rate

def ten_times_ten_fold(X,y, nPOP, MaxIt):
    error_rate_list = []
    count = 0
    for i in range(10):
        print("count: ", count)
        error_rate = single_cross(X,y, nPOP, MaxIt)
        error_rate_list.append(error_rate)
        count += 1

    sum = 0
    count = 0
    for i in error_rate_list:
        sum += i
        count += 1

    mean_error_rate = sum / count
    return mean_error_rate



if __name__ == "__main__":

    attribute_train, label_train = load_train_dataset()
    attribute_test, label_test = load_test_dataset()

    #data process

    pca1 = PCA(n_components= 0.95)
    #attribute_train_nor = normal(attribute_train)
    #attribute_test_nor = normal(attribute_test)

    newData_train = pca1.fit_transform(attribute_train)
    #pca2 = PCA(n_components = len(newData_train[0]))
    newData_test = pca1.transform(attribute_test)

    print("PCA len: ", len(newData_train[0]))
    print("len: ", len(newData_train))
    print("PCA len: ", len(newData_test[0]))
    
    #q2 use nPOP = 1, maxiter = 1 to train my model, and report the 10-times 10-folds accuracy
    q2_error_rate = ten_times_ten_fold(newData_train, label_train, 1, 1)
    print("when parameter nPOP = 1 and maxiter = 1, 10-time-10-fold accuarcy: ", 1 - q2_error_rate)

    #plot the error rate vary with the change of max iter number
    maxiter_list = [1, 3, 5, 10, 20, 30, 40]
    error_rate_maxiter_list = []
    for i in maxiter_list:

        error_rate_maxiter = single_cross(newData_train,label_train, 4, i)
        error_rate_maxiter_list.append(error_rate_maxiter)

    print("error_rate_maxiter_list: ", error_rate_maxiter_list)

    # plot the error rate vary with the change of nPOP
    error_rate_nPOP_list = []
    nPOP_list = [1, 4, 8, 12, 20, 30, 40]
    for i in nPOP_list:
        error_rate_nPOP = single_cross(newData_train,label_train, i, 10)
        error_rate_nPOP_list.append(error_rate_nPOP)

    print("error_rate_nPOP: ", error_rate_nPOP_list)

    #plot
    plt.figure(1)
    x1 = maxiter_list
    y1 = error_rate_maxiter_list
    plt.plot(x1, y1,'r--')
    plt.title('error rate varies with the change of max number of iter number')
    plt.xlabel('maxiter')
    plt.ylabel('error rate')

    plt.figure(2)
    x2 = nPOP_list
    y2 = error_rate_nPOP_list
    plt.plot(x2, y2,'r--')
    plt.title('error rate varies with the change of POP number')
    plt.xlabel('nPOP')
    plt.ylabel('error rate')
    plt.show()


    """
    
    KNN_classfier = KNN(attribute_train, label_train, 5)
    error_rate_KNN = cal_error_rate(attribute_test, label_test, KNN_classfier)
    print("error rate KNN: ", error_rate_KNN)
    """

    """
    #normal_attribute_train = normal(np.array(attribute_train))
    #print(normal_attribute_train)

    # Vi (t + 1) = w * Vi(t) + r1 * c1 * (Pi - Xi) + r2 * c2 * (Gi - Xi)
    # r1 and r2 are uniformed distributed between 0 and 1
    # w * Vi(t) is called inertia term, w is inertia coefficient
    # c1 and c2 are acceleration coefficient
    # r1 * c1 * (Pi - Xi) is called cognitive component
    # r2 * c2 * (Gi - Xi) is called social component
    # Xi (t + 1) = Xi(t) + V(t + 1)
    """
    """
    # Parameters of PSO
    MaxIt = 10 # max number of iterations 10 -100

    nPOP = 2 # number of particles

    min_num_neighbors = 1
    max_num_neighbors = 40 # 4- 40

    #standard PSO
    w = 1 # inertia coefficient
    w_damp = 0.99 #damping ratio of inertia coefficient

    c1 = 2 #personal acceleration coefficient
    c2 = 2 #social acceleration coefficient

    parameters = parameters(MaxIt = MaxIt, nPOP = nPOP, max_num_neighbors = max_num_neighbors, min_num_neighbors = min_num_neighbors, w = w, w_damp = w_damp, c1 = c1, c2 = c1)
    Data = Train_Test_data(attribute_train = newData_train, label_train = label_train,  attribute_test = newData_test , label_test = label_test)
    Population, GlobalBests, GlobalBest = PSO(Data, parameters)
    """






























