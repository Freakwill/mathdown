#!/usr/bin/env python


from mathdown import *

print(comment.parse_string(f"<!--hh\nsdfs-->"))
raise
# 
print(chapter.parse_string(r"""# 分类

[TOC]
分类问题是一种在生活实践中经常遇到的问题，其目标是将关注的一批对象划分到不同的类别中。在机器学习中，分类被规定为一种监督学习，其根本任务是建立从输入变量到类别标签（输出变量）的映射：$\mathcal{X}\to \mathcal{Y}=\{1,\cdots,K\}$。

**定义 分类模型一般形式**
简单了解一下分类问题的统计学习模型，
$$
Y|x,\theta\sim Cat(f(x,\theta))
$$
其中向量值映射$f:\mathcal{X}\times\Theta\to \Delta^{K-1}$, 其第$k$个分量$f_k$是$Y=k$在$x$下的条件概率，即
$$
P(Y=k|x,\theta)=f_k(x,\theta)
$$
习惯称$P(Y=k|x,w)$为输入$x$对$k$类的**响应概率**。

该模型完全由函数$f(x,\theta)$决定；我们的任务不是直接建立分类映射$X\to\{1,\cdots,K\}$，而是学习将一个Catogorical分布作为输出的函数$f$。分类模型或学习器称为**分类器**；有时分类器特指分类映射$\gamma:\mathcal{X}\to \mathcal{Y}$。


*注* 从模型（1）的形式来看，分类问题一般比回归问题困难。我们还发现，因为分类器学习的关键是构造合理的连续值映射$f$，所以分类问题可自然地转化成回归问题。

当$K=2$时，分类问题/模型称为二分类问题或者0-1分类问题/模型，且$Y$的值域一般规定为$\{0,1\}$；当$K>2$时，分类问题/模型称为多分类问题/模型。对于0-1分类问题，可用 Bernoulli 分布表示
$$
Y\sim B(f(x,\theta)),
$$
其中函数$f:X\times \Theta\to [0,1]$，因为只要知道$Y=1$的概率就可以了。

根据原则x.x，当用分类模型进行预测时，我们总是用如下公式，
$$
\hat{Y} = Mode(Y|x,\theta) =\argmax_k \{f_k(x,\theta)\},
$$
也就是用输出变量的众数作为预测值，而不是其期望。众数可能不只有一个。此时，我们可以选择其中标签编号最小的那个，或者从中随机选择一个。

**约定** 对$K$类分类问题，我们约定类别标签$Y$的值域是$\{1,\cdots, K\}$。当然，你也可以直接用实际类名作为类别标签，如$\{c_1,\cdots,c_K\}$。$x:k$表示样本$x$属于第$k$类或被打上类别标签$k$。

下面我们开始讨论分类问题的判别模型（判别型分类器）。分类问题还特别适合用生成模型（生成型分类器），将在X.X节被具体讨论。

## 判别函数和条件似然函数
这一节讲两个概念：判别函数和一般分类模型的条件似然函数，主要为一般的判别型分类器的讨论做准备。

### 判别函数

**判别函数**是分类问题一个非常有用的概念。估计参数后，分类模型可用判别函数做出最终的预测。

**定义 判别函数（或决策函数）**
**判别函数**指一组函数$\delta_k:\mathcal{X}\to \R$，可导出分类器$x\mapsto\argmax_k \delta_k(x)$。
若判别函数满足
$$
\argmax_k \delta_k(x)=\argmax_k P(Y=k|x),x\in \mathcal{X}
$$
则称之为分布/分类模型$P(Y|X)$的判别函数。

*注* 机器学习中，判别函数是需要被估计的。估计结果$\hat{\delta}_k(x)$同样被称为判别函数。

一个模型的判别函数不是唯一的；我们总是选择其表达式最简洁的函数作为分类问题的判别函数。对于0-1分类问题，若把标签$Y$的值域规定为$\{-1,1\}$，则判别函数$\delta$的定义简化可为：
$$
\argmax_k P(Y=k|x)= \mathrm{sign}(\delta(x)), k=\pm 1
$$
判别函数为线性的分类模型称为**线性分类模型**。本章主要关注线性分类模型。

为了直观了解分类效果，引入“决策边界”的概念。

**定义(决策边界)**
设$\delta_k(x)$是判别函数，定义$k$类和$l$类的决策边界为集合
$$
\{x\in \mathcal{X}|\delta_k(x)=\delta_l(x)\}
$$
而决策边界划分出的区域称为*决策区域*。更严格的决策边界应该限制为，$\{x\in \mathcal{X}|\delta_k(x)=\delta_l(x)\leq\delta_c(x),\forall c\neq k,l\}$。

决策边界是两个不同类的数据点集发生接触或混合的边缘地带。线性分类模型的决策边界显然是一个超平面，而其决策区域是一定是超凸多面体。因此直接用超平面分隔不同类数据的模型也称为**线性分类模型**。

### 条件似然函数

根据事实X.X，若观测到独立样本$\{(x_i,y_i)\}$，则分类模型的条件对数似然函数为
$$
l(\theta)
=-\sum_i H(e_{y_i},f(x_i,\theta))
$$
其中$e_{y_i}$是仅第$y_i$个分量为$1$的指示向量（即类别标签$y_i$的**onehot编码**）。

*注* （x.x）的$H$是两个概率向量之间的熵。

(X.X)仿佛暗示退化分布$e_{y_i}$就是$y_i$的真实分布。然而真实情况可能不是这样：样本中相同的样本点可以属于不同的类（见注X.X）。因此退化分布$e_{y_i}$可被称为$Y$的“理想分布”。MLE的目的就是减少理想分布/经验分布和模型分布的不同。

现在令$\hat{y}_i=f(x_i,\theta)$，并对类别标签进行onehot编码得$y_i$是，即将原模型的输出变量的分布看成回归模型$f(x,\theta)$的输出。我们有下述事实。

**事实**
分类模型(1)的条件似然为
$$
l(\theta)= -\sum_iH(y_i,\hat{y}_i),
$$
即分类模型(1) 的MLE等价于交叉熵作为损失函数的回归模型$y\sim f(x,\theta):\mathcal{X}\to \Delta^{K-1}$的学习问题。

特别地，对0-1分类模型有，
$$
l(\theta)=\sum_{y_i=1} \ln f(x_i,\theta)+\sum_{y_i=0} \ln (1-f(x_i,\theta))\\
=-\sum_i H_b(y_i,f(x_i,\theta)),
$$
其中$H_b(p,q):=-(p\log q + (1-p)\log(1-q))$。


常识告诉我们，一个对象最终只能划分到一个类。但这是理想情况。因为现实情况中人们通常只能用对象的部分属性进行类别判断，所以数据中同一个数据点被标上多于两个的类名是完全有可能的。一个图片其实也只是对象的一个侧面而已。此外，采样过程也可能存在各种误差，包括不同的观察者对同一个对象标记不同类名。


## Logistics 回归
Logistics 回归是最基本也是最知名的判别型分类器。虽然叫做Logistics 回归，但是它确实是一个分类器，只是利用了回归算法。正如前文所述，把分类问题转化为回归问题一直是一种常规的做法。

### 0-1 情形

**定义（0-1 Logistic回归模型）**
一个0-1 Logistic回归模型由下述条件分布决定：
$$
Y|x,w\sim B(\frac{1}{1+e^{-x\cdot w}}), P(Y=1|x,w)\sim \frac{1}{1+e^{-x\cdot w}}
$$
即
$$\mathrm{logit}P(Y=1|x,w) = x\cdot w\\
\text{或}P(Y=1|x,w) = \mathrm{expit}(x\cdot w)
$$
其中 $\mathrm{logit} p:=\log \frac{p}{1-p}$, $\mathrm{expit} a=\mathrm{logit}^{-1} a:=\frac{1}{1+e^{-a}}$。

`expit`最初出现在神经网络中，也称为`sigmoid`激活函数。根据定义X.X，不难得出下面这个事实。

**事实**
0-1 Logistic回归等价于用`sigmod`作为其激活函数的单层神经网络（**感知机**）。

Logistics 回归的几何解释是非常直观的。给定参数$w$，$P(Y=1|x)$ 是数据点$x$和分割超平面$x\cdot w=0$之间的距离的递增函数。这个距离称为**间隔**，即$x\cdot w$。它是有符号的，点在分割超平面某一侧为正，分到类1，在另一侧为负，分到类0。简单地说，两个类的数据可以被超平面$x\cdot w=0$分隔。这个事实是线性分类器的原始思路。显然，分割超平面$x\cdot w=0$正是Logistic回归的决策边界。


这个模型似乎暗示存在超平面$x\cdot w$严格分离两类数据点，即**线性可分**。然而数据一般是线性不可分的。既然假定分类器是线性的，就得保留容错性，即所谓的“软划分”。Logistics回归就是这样一种软划分。

*注* “数据点”这个称谓强调我们是在几何学或向量空间的语境下讨论数据的。

**定义 带link函数的线性回归**
输出$Y$在函数$g$作用下与输入$X$存在线性关系：$g(Y)\sim X\beta$，其中$g$称为**link函数**。这个关系到处的判别模型$P(Y|X)$称为带link函数的线性回归，简称$g$-线性回归。

(X.X)推出下述事实。

**事实**
0-1 Logistic回归等价于$\mathrm{logit}$-线性回归。因此，它也被称为**logit回归**。（注意logit-线性回归的输出是概率值，而不是0-1 Logistic回归的输出。）

*注* 事实X.X的等价性不是绝对意义上的，而是说，两者能给出相似的参数估计/学习准确度，并在理想条件下还能得到相同的估计。本书中的等价性都是这个意义上的。

观测到样本$\{(x_i,y_i)\}$后，可令$p_i=y_i\in\{0,1\}$，作为logit-广义回归的样本，因为正常情况下，一个样本只归属于一个类。我们想用 `logit`对$p_i$预处理，从而把 Logistics 回归转化成普通线性回归。然而，真实数据是$0,1$构成的，并不在logit的定义域中，不能直接执行这个预处理。为此，可定义一个“clip-logit函数”代替它：
$$
\mathrm{logit}_{\epsilon} p := \mathrm{logit} u(p,\epsilon)
$$
其中$u(p,\epsilon):=\begin{cases}
\epsilon, &0\leq p\leq \epsilon,\\
1-\epsilon, &1-\epsilon\leq p \leq 1,\\
p, & \epsilon < p < 1-\epsilon,
\end{cases}$
这其实就是在作用logit之前，把1调整为$1-\epsilon$，0调整为$\epsilon$，其中$\epsilon$是一个很小的数。这种调整不是没有道理。从统计模型的角度讲，任何一个数据点都以一定的非0概率归到0类或1类，即使实际上每个数据点只能被归到一类，何况数据中包含“相同输入不同标签”的样本完全是可能的。


#### 参数估计

事实 X.X 用线性回归方法给出了参数$w$的一种估计。现在讨论$w$的MLE。直接根据（6），写出对数似然函数，
$$
l(w)
=-\sum_{y_i=1}\ln(1+e^{-x_i\cdot w})-\sum_{y_i=0}\ln (1+e^{x_i\cdot w})\\
=\sum_{y_i=1} x_i\cdot w-\sum_i\ln (1+e^{x_i\cdot w}).
$$
其中$\sum_{y_i=1}$表示对满足$y_i=1$的样本求和（下同）。

显然，MLE对应于下述优化问题，
$$
\min_{w}\{J(w) =\sum_i\ln (1+e^{x_i\cdot w})-\sum_{y_i=1} x_i\cdot w\}
$$
用SGD解（14），可得局部最优解。为此，计算经验风险$J$的梯度
$$
\nabla J(w)=\sum_i(\frac{1}{1+e^{-x_i\cdot w}}-y_i)x_i=X^T(p_w-y),
$$
其中$X$是设计矩阵，$y$是所有输出值$y_i$构成的列向量，$p_w$是响应概率$\frac{1}{1+e^{-x_i\cdot w}}$构成的列向量。

显然(X.X)是一个无约束的凸优化问题。因此SGD的迭代过程将收敛到全局最优解。下面让我们来认真计算一下损失函数$J$的Hessian矩阵：
$$
H_J(w)=\{\sum_i\frac{e^{-x_i\cdot w}}{(1+e^{-x_i\cdot w})^2}x_{ij}x_{ik}\}_{jk}\\
=X^TD_wX,
$$
其中$X$是$N\times p$的设计矩阵，且$N$阶对角矩阵，
$$D_w:=\mathrm{diag} \{\frac{e^{-x_i\cdot w}}{(1+e^{-x_i\cdot w})^2}\}=\mathrm{diag} \{p_w\circ (1-p_w)\}
$$
这个Hessian矩阵显然是一个（半）正定矩阵。因此可用比SGD更高效的**Newton-Raphson法**：
$$
w \leftarrow w + H_J(w)^{-1}\nabla J(w)\\
=w+(X^TD_wX)^{-1}X^T(y-p_{w}),
$$
一般$H_J(w)$总是正定的，否则可用$H_J(w)+\alpha,\alpha>0$代替。

我们把迭代（X.X）分成两步:
   $$
   z= Xw+D_w^{-1}(y-p_{w})\\
   w \leftarrow (X^TD_wX)^{-1}X^TD_wz
   $$
然后写出下述算法。

**算法 Logistic回归的Newton-Raphson法/迭代重加权最小二乘法**
1. 初始化参数$w_0$
2. 定义第$t$次迭代的**矫正响应**
   $$
   z_t= Xw_t+(y-p_{w_t})\oslash (p_{w_t}\circ (1-p_{w_t}))\in\R^N
   $$
3. 如下更新参数值，
   $$
   w_{t+1} = (X^TD_{w_t}X)^{-1}X^TD_{w_t}z_t
   $$
4. 迭代2-3步直到收敛

可以把算法X.X形容为一个“矫正-参数估计/更新”交替迭代算法。参数更新(X.X)正好是优化问题$\min_w\|z_t-Xw\|_{D_{w_t}}$的解，即线性回归$z_t\sim Xw$的加权最小二乘估计。算法X.X实际上将优化问题（X.X）分解成一系列的加权最小二乘法，因此也被称为**迭代重加权最小二乘法（IRLS）**。


#### 替代损失函数与间隔模型

和回归模型一样，分类模型可以有多种损失函数，而且损失函数直接影响算法的设计。不同的损失函数，意味着不同的优化策略，也产生不同的近似解。

为了使分类模型的损失函数具有可微性，我们用实数值预测离散值。下面介绍三个常用的关于Bernoulli随机变量（取值为$\pm 1$）在实数值预测下的“损失函数”：
- **指数损失函数**，$l_{\mathrm{exp}}(y,\hat{y}):=e^{-y\hat{y}}$
- **合叶损失函数**，$l_{\mathrm{hinge}}(y,\hat{y}):=(1-y\hat{y})_+$
- **对数损失函数（或Logistic损失函数）**，$l_{\log}(y,\hat{y}):=\ln(1+e^{-y\hat{y}})$
其中真实值$y\in\{-1,1\}$，而预测值$\hat{y}\in\R$。

*注* 这些损失函数都是$y\hat{y}$的函数，所以写成$l(y\hat{y})$的形式即可。

构造这类损失函数的方法是多种多样的，但合理的损失函数$l(y,\hat{y})$应该使期望风险$El(Y,\hat{y})$只在$\hat{y}$与$Y$的众数（-1或1）同号时达到最小值，其中$Y\sim B(p)$（取值为$\pm1$），即
$$
\mathrm{sign}(\hat{y})=\begin{cases}
-1, & P(Y=1)\leq P(Y=-1),\\
1,  & P(Y=1)> P(Y=-1).
\end{cases}
$$
除0-1损失函数以外，它们被统称为**替代损失函数**。

我们知道，0-1分类的Logistics回归用二元交叉熵作为损失函数（见（13））。但也可以利用上述损失函数。

先改用$\pm 1$对两个类编号。再把(X.X)改写成，
$$
J(\omega)\propto\sum_i\ln(1+e^{-y_ix_i\cdot w}),
$$
即将$l_{\mathrm{log}}(y,\hat{y})$作为线性回归$y\sim x\cdot w$的损失函数。

根据几何解释，如果间隔$x_i\cdot w>(<)0$，那么应该有$y_i=1(-1)$，否则一定存在误差，且该误差大小可由点线距离$-y_i(x_i\cdot w)$衡量。我们的目的是最小化该误差。于是构造Logistics回归的一个等价模型：
$$
\min_w \sum_ie^{-y_i(x_i\cdot w)}
$$
即将$l_{\mathrm{exp}}(y,\hat{y})$作为线性回归$y\sim x\cdot w$的损失函数。不难看出，指数损失函数是对数损失函数的近似。且当$y_ix_i\cdot w$很大时，指数损失函数逼近指数损失函数。这就是说，当样本有较好的分割超平面时，两者的表现是相当的。

**定义 间隔模型**
上述线性回归模型$y\sim x\cdot w$的输出指的是间隔(取值于$\R$)，而损失可看作间隔和类别标签(取值于$\{-1,1\}$)的比较：
$$
\min_w \sum_il(y_i, x_i\cdot w)
$$
其中$l$合理的替代损失函数，如指数损失函数。此类模型被统称为**间隔模型**。

**事实** 0-1 Logistics回归等价于间隔模型(X.X)。

*注* 强调一下：Logistics回归的输出是类别标签；间隔模型的输出是间隔；事实X.X中以`logit`为link的线性模型的输出是概率值。

<center>Logistic回归等价模型</center>

|模型|关系式|损失函数|输出估计值的含义|备注|
|---|---|---|---|---|
|Logistic回归|$y\sim B(\mathrm{expit}(w\cdot x))$|$H_b(y,\hat{y})$|类别标签|标准形式，$y$取值为$0,1$|
|采用交叉熵损失的单层神经网络|$y\sim \mathrm{expit} (w\cdot x)$|$(\mathrm{logit} p- \hat{y})^2$|类别概率|与Logistic回归等同|
|logit-线性回归|$\mathrm{logit} p\sim w\cdot x$|$(\mathrm{logit} p- \hat{y})^2$|类别概率|参数估计结果和MLE相差较大|
|间隔模型|$y\sim w\cdot x$|$\ln(1+e^{-y\hat{y}})$|间隔|与Logistic回归完全等价，$y$取值为$\pm 1$|
|其他间隔模型|同上|其他基于间隔的替代损失函数|-|参数估计结果和MLE可能有差异|

【替代损失函数示意图】

### 多分类Logistic回归
0-1分类Logistic回归扩展到多分类情形并不困难。

**定义 Logistic 多分类器**
考虑$K$类分类模型：
$$
P(Y=k|x,W)\sim e^{x\cdot w_k},k=1,\cdots, K
$$
其中$Y$是类别标签取$1,\cdots, K$，$x\in\R^p$是输入变量，$w_k$是$W$的第$k$行。(归一化系数再一次被省略了)

为了使分布满足可识别性，可规定$w_K=0$或$\sum_kw_k=0$。具体用如下函数计算响应概率，
$$
\{P(Y=k|x)\}_k= \mathrm{softmax}(\{x\cdot w_k\}_k)= \mathrm{softmax}(Wx),
$$
其中
$$
\mathrm{softmax}(x):=\frac{e^{x_k}}{\sum_k e^{x_k}}:\R^n\to\Delta^{K-1}
$$
若规定 $x_K=0$ (或者$\sum_kx_k=0$), 则`softmax`是1-1的。它的逆$\mathrm{softmax}^{-1}:p\mapsto a$，其中$a_K=0,a_k=\ln\frac{p_k}{p_0},k=1,\cdots,K-1$.

现在，对输出做onehot编码。类似事实X.X，我们立刻得到下述结论。
**事实**
Logistic 多分类器等价于用 `softmax` 作为其激活函数的单层神经网络 (损失函数为交叉熵)。

编码结果显然不在`softmax^{-1}`的定义域中。可按照（8）那样处理，并将Logistic 多分类器转化成多输出的线性回归:
$$
\mathrm{softmax}^{-1}(p) \approx Wx.
$$

**事实**
Logistic 多分类器等价于 $\mathrm{softmax}$ -线性回归。（故又称之为 $\mathrm{softmax}$ -回归）

$N$个数据输出的onehot编码构成所谓的**指示响应矩阵(Indicator response matrix)** (或**0-1响应矩阵**)，其大小为$N\times K$。我们把所有响应概率$p(y=k|x_i), i=1,\cdots,N$也拼成一个大小为$N\times K$的矩阵，称为**概率响应矩阵**。因此，（多）分类模型本质上就是让响应概率矩阵逼近输出的指示响应矩阵。

<!-- ### 多分类转化成单分类 -->


## 分类的生成模型

分类器的生成模型的一般形式由两部分组成：类别$Y$的先验分布$P(Y)$和生成分布$P(X|Y)$。显然这就给出了联合分布$P(X,Y)$。

**定义 Bayes分类器**
若先验分布和生成分布分别为$P(Y=k),P(X=x|k), k=1\cdots, K$，则可用Bayes公式导出判别函数，
$$
\delta_k(x) = \ln P(X=x|k) + \ln P(Y=k),
$$
分类结果正好是$Y$的MAPE。我们把这类模型（连同判别函数$\gamma(x)=\arg\max_k \delta_k(x)$）统称为**Bayes分类器**。

### 参数解绑形式/Bayes分类器标准形式
X.X节指出，一般生成模型采用所谓的参数解绑形式，其中的先验分布和生成分布都是独立设计的，即这两个分布有独立的参数形式：
$$
P(Y,X|\theta)=P(X|Y,\theta_1)P(Y,\theta_2),
$$
其中$\theta=(\theta_1,\theta_2)\in \Theta_1\times \Theta_2$。

对分类模型，我们只考虑下面一种更实用的参数解绑形式。这也是对Bayes分类器的常规理解。

**定义（参数解绑形式I/Bayes分类器标准形式）**
合理假设$Y \sim Cat(\pi)$，且$P(X|Y=k)\sim p_k(x)$。模型完整的参数形式是：
$$
P(X=x,Y=k|\theta)= p_k(x)\pi_k,k=1,\cdots,K,
$$
其中参数$\theta$包含$\pi=\{\pi_k\}\in \Delta^{K-1}$和分布集$p=\{p_k(\cdot)\}$ (在具体的模型中，$p_k(\cdot)$的取值范围应该有合理的规定，当然与$\pi_k$的取值无关)。

了解一下模型(X.X)的对数似然：
$$
l(p,\pi)\propto \sum_k \sum_{y_i=k}\ln p_k(x_i)+\sum_kN_k\ln \pi_k,
$$

显然先验分布$\pi$可直接用频率估计，即$\hat{\pi}_k=\frac{N_k}{N},k=1,\cdots,K$，其中$N_k$为$k$类样本出现的个数，亦可记为$N(y_i=k)$。$p$的 MLE 则要最大化目标函数
$$ \sum_k \sum_{y_i=k}\ln p_k(x_i)
$$

一般$p_k(x)$都来自一个特定的分布族。也就是分类模型中，每个子类有相似的分布。笔者将这个idea称为**同质/同分布族假设**。

**定义（基于特定分布族的Bayes分类器）**
定义X.X中$p_k(x)$都来自一个特定的分布族$\mathcal{F}$，形如$p(x|\beta_0,\beta_k)$，其中$\beta_0$代表公共参数而$\beta_k$代表第$k$类的私有参数。称对应的分类器为**基于分布族$\mathcal{F}$的Bayes分类器**，简称 **$\mathcal{F}$-Bayes分类器**。

**定义（生成参数解绑形式II）**
$$
P(X=x,Y=k|\theta)= p(x|\theta_k)\pi_k,k=1,\cdots,K,
$$
其中$\pi_k$是类别的先验分布, $\theta_k$是类$k$的生成分布的参数。

参数解绑是相对而言的。如果定义X.X是参数解绑形式的“基础型”，那么定义X.X就是“完全(full)型”。后者的显著特点是，可以分割模型参数，使每个类的生成分布都由独立的参数调控。一般地，一种模型会有标准/规范/典型/公认的形式。比其解绑程度高的，就是解绑形式；比其解绑程度低的，就是**捆绑形式(tied form)**。

定义X.X的参数解绑形式的样本似然变得非常有特点：
$$
l(\{\pi_k\},\{\theta_k\})\propto 
\sum_k \sum_{y_i=k}\ln p(x_i|\theta_k)+\sum_kN_k\ln \pi_k,
$$
其中$\{(x_i,y_i)\}$是样本，$N_k$为$k$类样本大小。先验分布$\pi$依然用频率估计。$\theta_k$的MLE归结为下述优化问题：
$$
\max_{\theta_k} \sum_{y_i=k} \ln p(x_i|\theta_k),
$$
即根据子样本$S(y_i=k)$对每个类的生成分布$p(x_i|\theta_k)$执行MLE。

**定义（full型-$\mathcal{F}$-Bayes分类器）**
若$p(x|\theta_k)$可来自同一个分布族$\mathcal{F}$，则(X.X)相当于$\mathcal{F}$-Bayes分类器没有公共参数的情形，可称为**full型-$\mathcal{F}$-Bayes分类器**。

考察一下模型(X.X)中$X$的边缘分布：
$$
p(x)=\sum_kp_k(x)\pi_k
$$
**定义（混合分布/混合模型）**
形如(X.X)的分布被称为**混合分布/混合模型**，其中$p_k(x)$是混合分布/混合模型的基分布。当$p_k(x)$像定义X.X那样来自同一个分布族$\mathcal{F}$，那么称该边缘分布为 **$\mathcal{F}$-混合分布/混合模型**。


### 线性判别分析

现在在模型(X.X)的基础上增加**Gaussian 型假设**：
$$X|Y=k\sim N(\mu_k,\Sigma_k),k=1,\cdots, K
$$
即每个类的样本点都服从Gaussian分布。

**定义 Gaussian-Bayes分类器**
在Gaussian假设下的分类器是基于Gaussian分布的Bayes分类器，即**Gaussian-Bayes分类器**。

*注* Gaussian-Bayes分类器的边缘分布是**Gaussian混合分布**
$$
X\sim \sum_k \pi_k N(X;\mu_k,\Sigma_k).
$$
作为统计学习中被频繁使用的分布，混合Gaussian分布最为著名的应用可能还是聚类。（见第X章）

这一节介绍的分类模型在Gaussian假设的基础上，做了进一步限制：令$\Sigma_k=\Sigma,k=1,\cdots,K$，即每个类的方差相同。

**定义（线性判别分析）**
线性判别分析（LDA）的模型由下述分布刻画，
$$
X|Y=k\sim  N(X;\mu_k,\Sigma), Y\sim Cat(\{\pi_k\}).
$$


LDA模型的分类映射是
$$
\gamma(x)=\argmax_k\delta_k(x),
$$
其中$\delta_k$是**Fisher 判别函数（或决策函数）**，定义如下，
$$
\delta_k(x):=x^T\Sigma^{-1}\mu_k-\frac{1}{2}\|\mu_k\|_{\Sigma^{-1}}^2+\ln\pi_k.
$$


下面这个结论不会令人意外。读者请自行验证。
**事实**
设$\pi_k=\pi_l$，则$k$类和$l$类的决策边界是$\mu_k$和$\mu_l$的中垂超平面（垂直在内积$\langle x,y\rangle_{\Sigma^{-1}}$的意义上）。(若$\pi_k> \pi_l$，则决策边界向$\mu_l$平移。)

简单改进一下判别函数的计算。若对样本执行**球形化**操作，即令$x^*= \Sigma^{-\frac{1}{2}}x$，则判别函数可等价地写成，
$$
\delta_k(x)\equiv\delta_k^*(x^*):=\langle x^*,\mu^*_k\rangle-\frac{1}{2}\|\mu^*_k\|^2_2+\ln \pi_k,k=1,\cdots,K
$$

其中$\mu^*_k=\mu_k\Sigma^{-\frac{1}{2}}$是球形化后数据$\{x_i^*\}$的中心。球形化操作就是数据的标准化（未中心化），也就是使协方差变矩阵为单位矩阵。程序实现时，考虑只存储$\mu_k^*,\Sigma^{-\frac{1}{2}}$，并用$\delta_k^*$做预测。

#### 参数估计
根据统计学关于Gaussian分布的基本常识，LDA模型参数估计如下：

$$
\hat{\pi}_k= \frac{N_k}{N},\\
\hat{\mu}_k= \frac{1}{N_k}\sum_{y_i=k}x_i,\\
\hat{\Sigma} = \frac{1}{N-K}\sum_k\sum_{y_i=k}(x_i-\mu_k)(x_i-\mu_k)^T,
$$

其中$N_k$是第$k$类样本的个数，$N$是所有样本个数, $K$是类的个数。$\hat{\Sigma}$的表达式下述事实立刻得出：对每个类中心化后，即对$y_i=k$的样本执行变换$x_i\leftarrow x_i-\mu_k$，数据点是独立同分布。此外$\hat{\Sigma}$还可写成，
$$
\hat{\Sigma}=\sum_k\frac{N_k-1}{N-K}S_k\approx\sum_k\pi_kS_k\\
S_k:=\frac{1}{N_k-1}\sum_{y_i=k}(x_i-\mu_k)(x_i-\mu_k)^T
$$
即每个类的方差的加权和（凸组合），权重近似为先验概率(的估计)。因为$S_k$（即第$k$类样本方差）都是$\Sigma$的无偏估计，所以$\hat{\Sigma}$自然是$\Sigma$的无偏估计。因此，LDA学习过程本质上是对每个服从Gaussian分布的类，独立估计均值向量和协方差矩阵。

根据多元Gaussian分布的知识，可以推算参数的置信区间，并做相应的假设检验。


#### 生成任务
生成任务的实质是，基于模型的生成分布$P(X|y)$生成指定$y$类的样本$X$。

LDA属于生成模型，能够完成生成任务。不过对实际数据来说，它的生成功能极其贫乏。我们可以用下述简单算法，随机生成一个样本。它本质上是混合Gaussian分布的采样过程。

**算法 LDA生成算法**
输入已知的或估计得到参数$\pi,\{\mu_k\},\Sigma$，
1. $k\sim Cat(\pi)$（或手动设置）
2. $x\sim N(\mu_k, \Sigma)$

*注* 第2步，在具体实现时，应利用基于Gaussian分布性质的采样技巧：$x=\Sigma^{\frac{1}{2}} \xi+\mu_k,\xi\sim N(0, 1)$。不过现代计算机语言提供的软件都自动实现了这种技巧。

*注* AI生成已经成为时尚。其复杂性和功能也远超LDA，如生成人眼难以区别的图像和生动的故事。算法x.x主要目的是让读者了解生成算法的基本原理。

### 二次判别分析

现在放弃$\Sigma_k=\Sigma$的假设，即每个类的方差不一定相同。

**定义 二次判别分析（QDA）模型**
QDA就是Guassian-Bayes模型，表示为$(\pi,\{\mu_k,\Sigma_k\})$。它的默认形式中没有加入任何限制。

容易证明QDA的判别函数为，
$$
\delta^{Q}_k(x):=-\frac{1}{2}\|x-\mu_k\|_{\Sigma_k^{-1}}^2-\frac{1}{2}\ln\det \Sigma_k+\ln\pi_k
$$

它的决策边界将是一个二次曲线，故称为“二次判别分析”。按照线性分类器的标准解释，它不是线性的。不过它可以看成关于$\frac{p(p+1)}{2}$维输入$\{x^{(j_1)}x^{(j_2)},j_1\leq j_2\}$的线性模型，其中$x^{(j)}$是$x$的第$j$个分量。

根据(X.X), QDA参数估计更简单：$\mu_k$的估计同LDA；$\Sigma_k$的估计如下，
$$
\hat{\Sigma}_k =\frac{1}{N_k-1}\sum_{y_i=k}(x_i-\hat{\mu}_k)(x_i-\hat{\mu}_k)^T.
$$

根据对协方差矩阵$\Sigma_k$的进一步约束，可将QDA分成四种类型。见表X.X。当然，还可以构造其他类型的约束。读者应该根据实际需求选择最合适的约束。

<center>QDA四种常见类型</center>

|类型|约束|参数估计|参数个数（不计均值）|
|---|---|---|---|
|full|无约束|(X.X)|$\frac{p(p-1)}{2}K$|
|diag|$\Sigma_k$均为对角矩阵|$\mathrm{diag}\{\frac{1}{N_k-1}\sum_{y_i=k}(x_{ij}-\hat{\mu}_{ij})^2\}$|$pK$|
|sphereical|$\Sigma_k$均为数量矩阵，记为$\sigma_k$|$\frac{1}{pN_k-1}\sum_{y_i=k,j}(x_{ij}-\hat{\mu}_{ij})^2$|$K$|
|tied|$\Sigma_k=\Sigma$|同LDA|$\frac{p(p-1)}{2}$|

### LDA/QDA的正则化
QDA增加了多倍于LDA的参数，其方差估计的样本也变小了，容易导致过拟合。现对QDA做改进：第$k$类的方差修正为
$$
\hat{\Sigma}_k(\alpha)\leftarrow\alpha\hat{\Sigma}_k+(1-\alpha)\hat{\Sigma}
$$
其中超参数$0\leq \alpha\leq 1$可以用来调节判别模型拟合性能。$\hat{\Sigma},\hat{\Sigma}_k$分别是(25)，(27)给出的估计。这就是**正则判别分析（RDA）模型**，其设计思路类似于正则化线性回归。不难推测，这个修正方法相当于，假设参数$\Sigma_k$服从一个公共的以$\Sigma$为未知参数的先验分布：
$$
X|y=k \sim N(\mu_k,\Sigma_k),\Sigma_k\sim P(\Sigma)
$$

如何设计先验分布$P(\Sigma)$，以及如何估计超参数$\Sigma$留给读者思考。

### Bayes-Bayes分类器
正则化方法让我们联系到Bayes方法。应用Bayes方法的Bayes分类器可被称为“**Bayes-Bayes分类器**”。

本节只讨论该模型的一种极简形式：
$$
P(X,Y=k)\sim p(x|\theta_k)\pi_k,k=1,\cdots,K,\\
\pi=\{\pi_k\}\in\Delta^{K-1},\pi\sim Dir(\alpha),\alpha=\{\alpha_k\}
$$
即在混合模型$(\pi,\{\theta_k\})$的基础上，增加参数先验分布$\pi\sim Dir(\alpha)$。

如果参数$\alpha_k>0$已知，那么问题非常简单。$\theta_k$的估计同混合模型，不受先验分布的影响。$\pi$的估计基于定理X.X：
$$
\hat\pi_k\sim N(y_i=k)+\alpha_k
$$
这相当于对无先验的估计$\hat\pi_k\sim N(y_i=k)$做了光滑化处理。

**定理**
假设$X$是离散随机变量，取$K$个值，如$1,\cdots,K$，且
$$
P(X|\pi)\sim Cat(\pi), \pi\sim Dir(\alpha)
$$
其中Dirichlet分布参数$\alpha=\{\alpha_k,k=1,\cdots K\}$是已知的。则后验分布
$$
P(\pi|\{x_i\})\sim Dir(\alpha')
$$
因此$\pi$的MAPE是$\hat{\pi}\sim\alpha'-1$(或用后验均值估计$\hat{\pi}\sim\alpha'$)，其中$\alpha'=\{\alpha_k+N(x_i=k)\}$，$\{x_i\}$是样本。

*注* 事实X.X指出Dirichlet分布是Categorical分布的**共轭先验**。

超参数$\alpha$也称为**先验伪计数**或**虚拟计数**。仿佛计算经验计数/频数$N(y_i=k)$之前，就存在大小为$\alpha$的样本。

当参数$\alpha$未知时，(X.X)是层次/经验 Bayes 模型。问题立刻变得intractable。笔者把它留给读者去深入探讨。

### LDA降秩

LDA的一个有趣应用，是对数据进行降维处理---这里称之为“降秩”。因为利用了数据的类别标签，比起单纯使用基于无监督学习的降维方法，如**主成分分析（PCA）**，LDA更高效。

假设参数$\mu_k,\Sigma$已知（或已被估计），LDA降秩由下述算法实现。

**算法 LDA降秩**

输入：样本$X$，秩$r$
返回：关键矩阵$\Sigma^{-\frac{1}{2}},V^*$，变换结果$Z$

1. 均值矩阵$M:K\times p$是每个类的均值构成的矩阵, 即$M_{k\cdot}=\mu_k^T$, 令$B= Cov M:p\times p$, 称为**类间协方差矩阵**($\Sigma$相对地被称为**类内协方差矩阵**)；
2. 球形化: $M^*=M\Sigma^{-\frac{1}{2}},X^*=X\Sigma^{-\frac{1}{2}}$（这个作用会使样本服从球形正态分布）
3. 计算$B^*=Cov(M^*)=\Sigma^{-\frac{1}{2}}B \Sigma^{-\frac{1}{2}}:p\times p$ 
4. 对$B^*$执行($r$阶)特征值分解$B^*\approx V^*DV^{*T}, V^*\in O(p\times r)$, $D$为$r$阶对角矩阵（注意不是对$B$进行特征值分解)。
5. 令$Z=(X^*-\frac{1}{K}1_N\sum_k\mu_k^{*T})V^{*}=(X-\frac{1}{K}\sum_k\mu_k^T)V:N\times r$，即将PCA逆变换作用在每个样本上，其中$V=\Sigma^{-\frac{1}{2}}V^*$。(减去$\frac{1}{K}\sum_k\mu_k^*$目的是进行中心化，是PCA的一个必要步骤。)

*注* 关于PCA详细内容，请参考X.X节。算法X.X其实只涉及矩阵的特征值分解。

LDA降秩时，PCA学习的是每个类的均值$\{\mu_k\}$而不是所有样本，但学习后得到的变换作用在每个样本上；和PCA相比，LDA降秩充分利用了类别信息。我们用一个“文字公式”来概括算法x.x：LDA降秩 = 球形化 + PCA。

**定义 Rayleigh 商** $p$阶对称矩阵$B$相对正定矩阵$\Sigma$的 **（广义）Rayleigh 商**为，
$$
R_{B,\Sigma}(x):=\frac{x^TBx}{x^T\Sigma x},
$$
其中$x\in\R^p$是任一非零向量（定义域可限制在$\|x\|=1$上）。

*注* 当$\Sigma$时单位矩阵时，就是经典的Rayleigh 商。显然通过变换$x\mapsto\Sigma^{-\frac{1}{2}}x$，即可将广义 Rayleigh 商转化成经典Rayleigh 商。

**定义1 Fisher 问题/Fisher 判别分析**
$$
\max_{v\in\R^p} R_{B,\Sigma}(v),
$$
其中$B,\Sigma$是算法X.X中的类间协方差矩阵和类内方差矩阵。

优化问题(x.x)的目标函数也称为“Fisher 准则”；(x.x)的解$v$被称为**Fisher方向/Fisher轴/判别坐标(cannoical variates)**。（x.x）的几何意义是在$v$方向上对样本$X$做投影，使投影结果的类内方差$v^T\Sigma v$最小化，同时类间$v^TBv$方差最大化。LDA降秩正好解决了Fisher问题。

**事实**
(x.x)的解正是$V=\Sigma^{-\frac{1}{2}}V^*$的第一个列向量。

更一般地，可以证明算法X.X中的$V=\Sigma^{-\frac{1}{2}}V^*$是下述矩阵形式的Fisher问题的解。

**定义2 一般 Fisher 问题/Fisher 判别分析**
$$
\max_W \frac{\mathrm{tr}(W^TBW)}{\mathrm{tr}(W^T\Sigma W)}\\
\mathrm{s.t.} W\in O(n\times r)
$$

**事实** LDA 降秩等价于 Fisher 问题(x.x)。

降维是机器学习非常实用的预处理方法。LDA降秩起到类似PCA的降维作用，可以作为任何机器学习模型的预处理。

*例* 我们可以构造“降秩 LDA”模型: 这个被LDA降秩预处理的机器学习模型正好又是LDA。该模型的**判别函数**为
$$
\delta_k(x):=\langle x', \mu_k'\rangle-\frac{1}{2}\|\mu_k'\|^2+\ln\pi_k\\
=\langle x, \mu_k\rangle_{\Sigma^{-\frac{1}{2}}V^* V^{*T}\Sigma^{-\frac{1}{2}}}-\frac{1}{2}\|\mu_k\|^2_{\Sigma^{-\frac{1}{2}}V^* V^{*T}\Sigma^{-\frac{1}{2}}}+\ln\pi_k,
$$
其中$x'=V^{*}\Sigma^{-\frac{1}{2}}x,\mu_k'=V^{*}\Sigma^{-\frac{1}{2}}\mu_k$，$V^*$有LDA降秩算法导出。不难发现，降秩 LDA是LDA的近似。它等价于，将LDA的协方差矩阵逆$\Sigma^{-1}$替换为$\Sigma^{-\frac{1}{2}}V^* V^{*T}\Sigma^{-\frac{1}{2}}$。降秩 LDA 分类效果将透露出原模型真正需要的输入变量的“主成分”。

"""
))
