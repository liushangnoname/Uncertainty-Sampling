import numpy as np
import numpy.ma as ma
from scipy.special import xlogy
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

def uncert(x, x_cal, err_cal, kernel = lambda z : np.exp(-np.power(z, 2)), width = 1e0):
    if x.ndim == 1:
        x = np.reshape(x, (1, -1))
    if x_cal.ndim == 1:
        x_cal = np.reshape(x_cal, (1, -1))
    num_cal = x_cal.shape[0]
    dim = x.shape[1] + 1
    dist = distance_matrix(x, x_cal)
    ker = kernel(dist / width / np.power(num_cal, 1.0/(dim+2.0)))
    normalizer = ker.sum(axis = 1, keepdims = True)
    ker = np.divide(ker, normalizer)
    mse = np.matmul(ker, err_cal.flatten())
    return mse


def logistic(x, coeff_, intercept_):
    pred_ = np.matmul(x, coeff_) + intercept_
    max_ = np.max(pred_, axis = 1, keepdims = True)
    pred_ -= max_
    pred_ = np.exp(pred_)
    sum_ = np.sum(pred_, axis = 1, keepdims = True)
    return np.divide(pred_, sum_)

def grad_logistic_regression(x, y, coeff_, intercept_):
    q_ = logistic(x, coeff_, intercept_)
    diff_ = q_ - y
    concat_ = np.concatenate((x, np.ones(shape = (x.shape[0], 1))), axis = 1)
    return concat_.T @ diff_ / float(x.shape[0])

def loss_crossentropy(p_, q_):
    return np.sum(-xlogy(p_, q_), axis = 1)


def grad_hinge(x, y, coeff_, intercept_):
    pred_ = np.matmul(x, coeff_) + intercept_
    remain_ = ma.masked_array(pred_, mask = y)
    second_choice = np.zeros_like(y)
    sc = np.argmax(remain_, axis = 1, keepdims = True)
    np.put_along_axis(second_choice, sc, 1, axis = 1)
    hinge = np.maximum(0.0, 1.0 - np.sum(np.multiply(y - second_choice, pred_), axis = 1))
    inds = hinge <= 0.0
    y_hinge = np.copy(y)
    y_hinge[inds, :] = 0
    second_choice_hinge = np.copy(second_choice)
    second_choice_hinge[inds, :] = 0
    diff_ = second_choice_hinge - y_hinge
    concat_ = np.concatenate((x, np.ones(shape = (x.shape[0], 1))), axis = 1)
    return concat_.T @ diff_ / float(x.shape[0])

def loss_hinge(y, pred_):
    remain_ = ma.masked_array(pred_, mask = y)
    second_choice = np.zeros_like(y)
    sc = np.argmax(remain_, axis = 1, keepdims = True)
    np.put_along_axis(second_choice, sc, 1, axis = 1)
    hinge = np.maximum(0.0, 1.0 - np.sum(np.multiply(y - second_choice, pred_), axis = 1))
    return hinge


def grad_linear_regression(x, y, coeff_, intercept_):
    pred_ = np.matmul(x, coeff_) + intercept_
    diff_ = pred_ - y
    concat_ = np.concatenate((x, np.ones(shape = (x.shape[0], 1))), axis = 1)
    return 2 * concat_.T @ diff_ / float(x.shape[0])

def loss_squared(y, pred_):
    return np.sum(np.power(pred_ - y, 2), axis = 1)


def vec_to_prob(vec):
    inds = vec < 0.0
    vec[inds] = 0.0
    Z = vec.sum()
    if Z < 1e-14:
        return np.ones(shape = vec.shape) / np.ones(shape = vec.shape).sum()
    else:
        return vec / Z

def uncertain_sampling(x, y, 
                       init_method = 'zero', 
                       init = None,
                       query = 'active',
                       width = 5e-1,
                       max_iter = 1e4, 
                       step_size = 3e-2, 
                       warm_start = 0,
                       sep_valid = False,
                       valid_ratio = 0.2,
                       update_freq = 5,
                       batch_size = 32,
                       method = 'logistic_regression',
                       ):
    num = x.shape[0]
    dim = x.shape[1]
    K = y.shape[1]
    if init_method == 'random':
        # Random initialization
        coeff = np.random.normal(size = (dim, K))
        intercept = np.random.normal(size = (1, K))
    elif init_method == 'zero':
        # Zero initialization
        coeff = np.zeros(shape = (dim, K))
        intercept = np.zeros(shape = (1, K))
    elif init_method == 'fixed':
        # Fixed initialization
        coeff = init[:dim, :]
        intercept = init[dim:, :]
    else:
        print('init_method should be \'random\', \'zero\', or \'fixed\' ')
    queries = 0
    bar_coeff = np.copy(coeff)
    bar_intercept = np.copy(intercept)
    coeff_list = []
    intercept_list = []
    coeff_list.append(np.copy(coeff))
    intercept_list.append(np.copy(intercept))
    has_valid = False
    batches = math.ceil(max_iter / batch_size)
    has_U = False
    for count in tqdm(range(batches)):
        # Pool-based
        if query == 'passive' or count < warm_start:
            U = np.ones(shape = (num,))
        elif query == 'active' or query == 'softmax':
            if has_U == False:
                U = np.ones(shape = (num,))
                has_U = True
            elif count % update_freq == 0:
                if method == 'logistic_regression':
                    U = uncert(x, x[valid], loss_crossentropy(y[valid], logistic(x[valid], coeff, intercept)),
                               width = width)
                elif method == 'svm':
                    U = uncert(x, x[valid], loss_hinge(y[valid], np.matmul(x[valid], coeff) + intercept),
                               width = width)
                elif method == 'linear_regression':
                    U = uncert(x, x[valid], loss_squared(y[valid], np.matmul(x[valid], coeff) + intercept),
                               width = width)
                else:
                    print('method should be \'logistic_regression\', \'svm\', or \'linear_regression\' ')
                    return
                if query == 'softmax':
                    U -= np.max(U)
                    U = np.exp(U)
        else:
            print('query method should be \'active\', \'passive\', or \'softmax\' ')
            return
        prob = vec_to_prob(U)
        if count < batches - 1:
            query_size = batch_size
        else:
            query_size = int(max_iter) - count * batch_size
        choice = np.random.choice(np.arange(num), size = query_size, p = prob, replace = True)
        if method == 'logistic_regression':
            grad = grad_logistic_regression(np.reshape(x[choice], (query_size, -1)), np.reshape(y[choice], (query_size, -1)),
                                            coeff, intercept)
        elif method == 'svm':
            grad = grad_hinge(np.reshape(x[choice], (query_size, -1)), np.reshape(y[choice], (query_size, -1)), 
                              coeff, intercept)
        elif method == 'linear_regression':
            grad = grad_linear_regression(np.reshape(x[choice], (query_size, -1)), np.reshape(y[choice], (query_size, -1)), 
                                          coeff, intercept)
        else:
            print('method should be \'logistic_regression\', \'svm\', or \'linear_regression\' ')
            return
            
        queries += query_size
        if sep_valid:
            if count % int(1.0 / valid_ratio) == 0:
                if has_valid:
                    valid = np.concatenate((valid, np.copy(choice)))
                else:
                    valid = np.copy(choice)
        else:
            if has_valid:
                valid = np.concatenate((valid, np.copy(choice)))
            else:
                valid = np.copy(choice)
        
        coeff -= grad[:dim, :] * step_size
        intercept -= grad[dim:, :] * step_size
        bar_coeff = (1.0 - 1.0 / float(count + 2)) * bar_coeff + 1.0 / float(count + 2) * coeff
        bar_intercept = (1.0 - 1.0 / float(count + 2)) * bar_intercept + 1.0 / float(count + 2) * intercept
        coeff_list.append(np.copy(bar_coeff))
        intercept_list.append(np.copy(bar_intercept))
    return coeff_list, intercept_list

def accuracy(x, y, coeff_, intercept_, method = 'logistic_regression'):
    if method == 'logistic_regression':
        pred_ = logistic(x, coeff_, intercept_)
        hit = np.argmax(y, axis = 1) == np.argmax(pred_, axis = 1)
    elif method == 'svm':
        pred_ = np.matmul(x, coeff_) + intercept_
        hit = np.argmax(y, axis = 1) == np.argmax(pred_, axis = 1)
    return float(np.sum(hit)) / float(x.shape[0])

def plot_accuracy(x, y, coeff_list, intercept_list, is_print = False, method = 'logistic_regression'):
    acc_list = []
    for i in range(len(coeff_list)):
        acc_list.append(accuracy(x, y, coeff_list[i], intercept_list[i], method))
    if is_print:
        plt.plot(acc_list)
        plt.show()
    return acc_list

def plot_loss(x, y, coeff_list, intercept_list, is_print = False, log_transform = False):
    loss_list = []
    for i in range(len(coeff_list)):
        if log_transform:
            loss_list.append(np.average(loss_squared(np.exp(np.matmul(x, coeff_list[i]) + intercept_list[i]) - 1.0, y)))
        else:
            loss_list.append(np.average(loss_squared(np.matmul(x, coeff_list[i]) + intercept_list[i], y)))
    if is_print:
        plt.plot(loss_list)
        plt.show()
    return loss_list