# GBDT-11.03-
[功能描述]：本项目为GBDT算法实现程序，，GBDT算法是一种迭代的决策树算法，该算法由多棵决策树组成，所有树的结论累加起来做最终答案。它在被提出之初就和SVM一起被认为是泛化能力（generalization)较强的算法  
[开发环境]：项目开发软件为python3.6版本，要同时安装pychram和Anaconda。具体安装和环境配置方法见文档：https://github.com/saberliliy/python-Environment-configuration-document  
[项目结构简介]：项目由gbdt_base.py,regression_tree,utils以及dataset数据集组成。其中gbdt_base.py为GBDT算法程序，regression_tree为决策树程序，utils为自定义的工具函数，包括数据集切分，数据拟合函数等，dataset为波士顿房价数据集，作为本次算法的测试数据。  
[依赖包]：项目运行需要安装以下依赖包：pandas，matplotlib，seaborn，scikit-learn,numpy,keras,tensorflow，itertools，os，copy，utils，statistics，安装具体方式请参考python环境配置文档中的添加依赖包方法。  
