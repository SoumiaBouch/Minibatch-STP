
"""
 For the experiment of generation of black box adversarial attacks on a deep neural network classifier
 we used the codes from https://github.com/IBM/ZOSVRG-BlackBox-Adv , just add  this .py file to the 
 optimization_methods folder. 

"""

import numpy as np
import pickle

np.random.seed(2023)

def MiSTP(delImgAT_Init, MGR, objfunc):

    
    T = MGR.parSet['nStage']*MGR.parSet['M']

    best_Loss = 1e10
    best_delImgAT = delImgAT_Init
    curret_delImgAT = delImgAT_Init

    attack_loss_values = []
    overall_loss_values = []
    distortion_loss_values = []
    query_count_values = []

    objfunc.evaluate(curret_delImgAT, np.array([]), False)

    attack_loss_values.append(objfunc.Loss_Attack)
    overall_loss_values.append(objfunc.Loss_Overall)
    distortion_loss_values.append(objfunc.Loss_L2)
    query_count_values.append(objfunc.query_count)

    for T_idx in range(T):

        randBatchIdx = np.random.choice(np.arange(0, MGR.parSet['nFunc']), MGR.parSet['batch_size'], replace=False)

        search_direction = objfunc.Draw_UnitSphere()
        
        x_plus = curret_delImgAT +  delImgAT_Init.size * MGR.parSet['eta'] * search_direction
        F_plus = objfunc.evaluate(x_plus, randBatchIdx, addQueryCount = True)

        x_minus = curret_delImgAT - delImgAT_Init.size * MGR.parSet['eta'] * search_direction
        F_minus = objfunc.evaluate(x_minus, randBatchIdx, addQueryCount = True)

        x = curret_delImgAT
        F = objfunc.evaluate(x, randBatchIdx, addQueryCount = True)

        min_idx = np.argmin([F, F_plus, F_minus])

        if min_idx==1:
            curret_delImgAT = x_plus
        if min_idx==2:
            curret_delImgAT = x_minus


        objfunc.evaluate(curret_delImgAT, np.array([]), False)

        attack_loss_values.append(objfunc.Loss_Attack)
        overall_loss_values.append(objfunc.Loss_Overall)
        distortion_loss_values.append(objfunc.Loss_L2)
        query_count_values.append(objfunc.query_count)


        if(T_idx%100 == 0):
            print('Iteration Index: ', T_idx)
            objfunc.print_current_loss()
        if(objfunc.Loss_Attack <= 1e-20 and objfunc.Loss_Overall < best_Loss):
            best_Loss = objfunc.Loss_Overall
            best_delImgAT = curret_delImgAT
            #print('Updating best delta image record')

        MGR.logHandler.write('Iteration Index: ' + str(T_idx))
        MGR.logHandler.write(' Query_Count: ' + str(objfunc.query_count))
        MGR.logHandler.write(' Loss_Overall: ' + str(objfunc.Loss_Overall))
        MGR.logHandler.write(' Loss_Distortion: ' + str(objfunc.Loss_L2))
        MGR.logHandler.write(' Loss_Attack: ' + str(objfunc.Loss_Attack))
        MGR.logHandler.write(' Current_Best_Distortion: ' + str(best_Loss))
        MGR.logHandler.write('\n')


    with open('MiSTP_attack_loss_values', "wb") as fp: pickle.dump(attack_loss_values,fp)
    with open('MiSTP_overall_loss_values', "wb") as fp: pickle.dump(overall_loss_values,fp)
    with open('MiSTP_distortion_loss_values', "wb") as fp: pickle.dump(distortion_loss_values,fp)
    with open('MiSTP_query_count_values', "wb") as fp: pickle.dump(query_count_values,fp)
    
    return best_delImgAT
