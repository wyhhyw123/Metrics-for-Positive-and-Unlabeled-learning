# Metrics-for-Positive-and-Unlabeled-learning
Metrics for Positive and Unlabeled learning
对于PU learning，计算AUC，准确率，F1值三个指标。
先运行cal_confusion_matrics()函数，计算不同阈值下的混淆矩阵，然后：
（1）cal_tpr_fpr_lb_ub，计算上下界TPR和FPR，可用于画ROC曲线；
（2）F1_score，计算F1值；
（3）accuracy_score，计算准确率；
注意，不能计算AUC，只能给出ROC曲线。

++++++++++++++++++++++++++
参考文献：Assessing binary classifiers using only positive and unlabeled data

