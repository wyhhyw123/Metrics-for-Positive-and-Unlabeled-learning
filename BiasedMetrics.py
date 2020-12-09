# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 19:58:58 2020

@author: 27264
"""

import pandas as pd 
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix as cm
import random 
from scipy import stats


class BiasedMetrics():
    def __init__(self,beta,sampling_times,sub_sample,random_state):
        self.beta = beta
        self.sampling_times = sampling_times
        self.random_state = random_state
        self.sub_sample = sub_sample
        self.lower_mats = []
        self.upper_mats = []
        self.probs_record = []

    def labeled_confusion_matrix(self,mat,thres,sub_sample):
        ### 给定一个概率阈值时，计算带标签样本的混淆矩阵和TPR
        tmp_mat = mat[mat[:,1]==1]
        n_sample = tmp_mat.shape[0]
        if sub_sample < 1:
            sample_idx = sorted(random.sample(range(n_sample),
                                              int(sub_sample*n_sample)))
            tmp_mat = tmp_mat[sample_idx,:]
        n_sample = tmp_mat.shape[0]
        TP = len(np.where((tmp_mat[:,0]>=thres )&(tmp_mat[:,1]==1))[0])
        FN = n_sample - TP
        confu_mat = np.array([[TP,FN],
                              [0,0]])
    
        return confu_mat,TP/n_sample
    
    
    def cal_tpr_CI(self, mat,thres,sub_sample):
        ### 计算带标签样本TPR的95% 置信区间
        labeled_tpr_record = []
        for i in range(self.sampling_times):
            tpr = self.labeled_confusion_matrix(mat,thres,self.sub_sample)[1]
    
            labeled_tpr_record.append(tpr)
        tpr_std = np.std(labeled_tpr_record)
        tpr_mean = np.mean(labeled_tpr_record)
    
        CI_L, CI_U = stats.norm.interval(0.95,tpr_mean,tpr_std)
    
        if np.isnan(CI_L):
    
            CI_L = 0
        if np.isnan(CI_U):
            CI_U = 0
    
        ### 95% 置信区间
        return CI_L, CI_U
    
    
    def confusion_mat_for_ublabeled(self,mat,thres,n_sample,theta):
        head = mat[mat[:,0]>=thres].shape[0]
        TP = min(head,theta)
        FN = self.beta*n_sample - TP
        FP = head - TP
        TN = n_sample - self.beta*n_sample - FP
     
        res  = np.array([[TP,FN],
                              [FP,TN]])
        return res
    
    def unlabeled_confusion_matrix(self,mat,beta, CI,thres):
        ### 计算未标注样本的上下界混淆矩阵
        tmp_mat = mat[mat[:,1]==0]
        n_sample = tmp_mat.shape[0]
    
        lb_theta = int(CI[0]*self.beta*n_sample)+1
        ub_theta = int(CI[1]*self.beta*n_sample)
    
        lb_confu_mat = self.confusion_mat_for_ublabeled(tmp_mat,thres,n_sample,lb_theta)
        ub_confu_mat = self.confusion_mat_for_ublabeled(tmp_mat,thres,n_sample,ub_theta)
        return lb_confu_mat,ub_confu_mat
    
    
    def cal_tpr(self,mat):
        tpr = mat[0,0]/(mat[0,0]+mat[0,1])
        return tpr
    def cal_fpr(self,mat):
        fpr = mat[1,0]/(mat[1,0]+mat[1,1])
        return fpr

    
    
    def cal_confusion_matrics(self,real,prob):
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        if not isinstance(real,np.ndarray):
            real = np.array(real)
        if not isinstance(prob,np.ndarray):
            prob = np.array(prob)
        real = real.reshape(-1,1)
        prob = prob.reshape(-1,1)
        res_mat = np.hstack([prob,real])
        res_mat = res_mat[res_mat[:,0].argsort()[::-1]]
        
        ### 计算带标签正样本的不同阈值下的混淆矩阵
        probs = res_mat[:,0].tolist()
        probs = [probs[0]+1]+probs+[probs[-1]-1]
        
        pre_TP = -1
        for prob in probs:
            
            ### 计算给定概率阈值下带标签正样本的混淆矩阵
            labeled_confu_mat,TP = self.labeled_confusion_matrix(res_mat,prob,sub_sample = 1)
       
            ### 计算给定概率阈值下带标签正样本的混淆矩阵
       
            if TP!= pre_TP:
                pre_TP = TP
                
                tpr_CI = self.cal_tpr_CI( res_mat,prob,sampling_size)
    
            unlabel_cm_lower, unlabel_cm_upper = self.unlabeled_confusion_matrix(res_mat,beta, tpr_CI,prob)
    
            '''
            [[TP,FN],
            [FP,TN]]]
            '''
            
            mat_lower = labeled_confu_mat+unlabel_cm_lower
            mat_upper = labeled_confu_mat+unlabel_cm_upper
            
            self.lower_mats.append(mat_lower)
            self.upper_mats.append(mat_upper)
            self.probs_record.append(prob)
    
    def cal_tpr_fpr_lb_ub(self):
        ### 计算不同概率阈值对应下的TPR和FPR，可用于画ROC曲线
        lower_fpr = []
        lower_tpr = []
        upper_fpr = []
        upper_tpr = []
        for i in range(len(self.lower_mats)):
            mat_lower,mat_upper = self.lower_mats[i],self.upper_mats[i]
            
            
            l_fpr = self.cal_fpr(mat_lower)
            l_tpr = self.cal_tpr(mat_lower)
            u_fpr = self.cal_fpr(mat_upper)
            u_tpr = self.cal_tpr(mat_upper)
    
            if l_fpr>1 or u_fpr>1:
                break
            lower_fpr.append(l_fpr)
            lower_tpr.append(l_tpr)
            upper_fpr.append(u_fpr)
            upper_tpr.append(u_tpr)


        return lower_fpr,lower_tpr,upper_fpr,upper_tpr
            
    def cal_f1(self,mat,alpha):
        P = mat[0,0]/(mat[0,0]+mat[1,0])
        R = mat[0,0]/(mat[0,0]+mat[0,1])
        f1 = (1+alpha**2)*P*R/((alpha**2)*(P+R))
        return f1
        
    
    def F1_score(self,real,prob,prob_thres = 0.5,alpha = 1):
        tmp_probs  = np.abs(np.array(self.probs_record)-prob_thres)
        idx = np.argmin(tmp_probs)
        mat_lower = self.lower_mats[idx]
        mat_upper = self.upper_mats[idx]
        lb_f1_score = self.cal_f1(mat_lower,alpha)
        ub_f1_score = self.cal_f1(mat_upper,alpha)
        return lb_f1_score,ub_f1_score
    
    def cal_accu(self,mat):
        return (mat[0,0]+mat[1,1])/(mat[0,0]+mat[1,1]+mat[0,1]+mat[1,0])
    
    def accuracy_score(self,real,prob,prob_thres=0.5):
        tmp_probs  = np.abs(np.array(self.probs_record)-prob_thres)
        idx = np.argmin(tmp_probs)
        mat_lower = self.lower_mats[idx]
        mat_upper = self.upper_mats[idx]
        lb_accu = self.cal_accu(mat_lower)
        ub_accu = self.cal_accu(mat_upper)
        return lb_accu,ub_accu
        
        
        
        
    
prob = [random.random() for i in range(300)]
prob = np.linspace(0,1,300)
real = [0,1,0,1,1]*60
beta = 0.5
sampling_times = 200
sampling_size = 0.8
random_state = 2020

biasm = BiasedMetrics(beta,sampling_times,sampling_size,random_state)
biasm.cal_confusion_matrics(real,prob)
print(biasm.F1_score(real,prob))

print(biasm.accuracy_score(real,prob))

pred = [1 if i>=0.5  else 0 for i in prob]
from sklearn.metrics import accuracy_score,f1_score
accuracy_score(real,pred)
f1_score(real,pred)

lower_fpr,lower_tpr,upper_fpr,upper_tpr = biasm.cal_tpr_fpr_lb_ub()




import matplotlib.pyplot as plt
plt.plot(lower_fpr,lower_tpr,'red')
plt.plot(upper_fpr,upper_tpr,'blue')

fpr,tpr,_ = roc_curve(real,prob)
plt.plot(fpr,tpr,'green')
