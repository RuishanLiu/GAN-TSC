import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score
from functions.tabular.generate_fake_tabular import generate_gan, generate_munge

def train_model(X_train, X_test, y_train, y_test, model):
    '''Train Sklearn Models'''
    model.fit(X_train, y_train)
    return model

def teacher_prediction(batch_x):
    return teacher.predict_proba(batch_x)[:, 1]

def get_auc(model, X_test, y_test):
    y_test_pred = model.predict(X_test)
    auc = roc_auc_score(y_test, y_test_pred)
    return auc

def train(X_train, y_prob, X_test, y_test, n_estimator):  
    student = RandomForestRegressor(n_estimators=n_estimator, n_jobs=-1)
    student.fit(X_train, y_prob)
    y_test_pred = student.predict(X_test)
    auc = get_auc(student, X_test, y_test)
    return auc

def experiment(mode, n_estimator, X_train=None, y_train=None, X_test=None, y_test=None, X_fake=None):
    '''Return test auc of tasks.
        - mode = 'training', 'fake_data', 'mixture', 'origin', 'classification' '''
    if mode == 'w/o_teacher_classifier':
        model = RandomForestClassifier(n_estimators = n_estimator, n_jobs = -1)
        model = train_model(X_train, X_test, y_train, y_test, model)
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    elif mode == 'w/o_teacher_regressor':
        auc = train(X_train, y_train, X_test, y_test, n_estimator) 
    elif mode == 'training_only':
        y_prob = teacher_prediction(X_train)
        auc = train(X_train, y_prob, X_test, y_test, n_estimator) 
    elif 'fake_only' in mode:
        y_prob_fake = teacher_prediction(X_fake)
        auc = train(X_fake, y_prob_fake, X_test, y_test, n_estimator)
    elif mode == 'mixture':
        y_prob = teacher_prediction(X_train)
        y_prob_fake = teacher_prediction(X_fake)
        X_mix = np.concatenate([X_train, X_fake], axis=0)
        y_prob_mix = np.concatenate([y_prob, y_prob_fake], axis=0)
        auc = train(X_mix, y_prob_mix, X_test, y_test, n_estimator) 
    return auc


if __name__ == '__main__':
    
    ######################################### Load Data ##############################################
    
    data_path = 'data/magic.npz'
    data = np.load(data_path)
    X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test'] # load_benchmark()

    ######################################## Train Teacher ###########################################

    print('\nTraining Teacher - Random Forest with 500 Trees')
    teacher = RandomForestClassifier(n_estimators = 500, n_jobs = -1)
    teacher = train_model(X_train, X_test, y_train, y_test, teacher)
    auc_teacher = roc_auc_score(y_test, teacher.predict_proba(X_test)[:, 1])
    print('Teacher Test AUC: %.3f \n' % (auc_teacher))

    ###################################### Generate GAN Data #########################################

    print('Generating GAN Data')
    model_path = 'models/tabular/netG.h5'
    X_gan = generate_gan(int(9*X_train.shape[0]), model_path)
    
    ############################### Generate MUNGE Augmentation Data ###################################

    print('Generating MUNGE Data')
    X_munge, label_fake = generate_munge(X_train, y_train, 9, s=1.5, p=0.1)

    ######################################### Compression #############################################

    modes = ['w/o_teacher_regressor', 'training_only', 'fake_only', 'mixture', 'munge_fake_only', 'w/o_teacher_classifier']
    names = ['w/o teacher', 'Training Data Only', 'GAN Data Only', 'Training & GAN', 'MUNGE Augmentation', 'w/o teacher']
    n_trees = np.array(np.linspace(1, 20, 5), dtype=np.int)

    print('\nTeacher Test AUC: %.3f' % (auc_teacher))
    print('\nStudent Test AUC:')
    print(''.join(['{:<24}'.format('Number of Trees')]+['{:>4}  '.format(n_tree) for n_tree in n_trees])) 
    for mode, name in zip(modes, names):
        auces = []
        X_fake = X_munge if 'munge' in mode else X_gan
        for n_tree in n_trees:  
            auc = experiment(mode, n_tree, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, X_fake=X_fake)
            auces.append(auc)
        if mode == 'w/o_teacher_classifier':
            print('Extra: If the student is a classifier')
        print(''.join(['{:<24}'.format(name)]+['{0:.3f} '.format(auc) for auc in auces]))
    print('\n')