#!/usr/bin/env python


from mathdown import *

# print(text.parse_string(r"""ad
# **算法 半监督 K-means 算法**
# 输入：有标签数据$D^1=\{(x_i,z_i)\}$, 无标签数据$D^0=\{x_i'\}$
# 返回：分类器（或聚类中心）
# 0. 令$\mathcal{Z}^1$是$D^1$中出现过的标签集, $\mathcal{Z}^0=\mathcal{Z}\setminus \mathcal{Z}^1$是$D^1$中未出现的标签集；
# 1. 根据$D^1$，初始化聚类中心$\{\mu_z,z \in \mathcal{Z}^1\}$（直接取$D^1$中每一类的均值），再根据$D^0$，初始化聚类中心$\{\mu_z,z\in \mathcal{Z}^0\}$；
# 2. 重置分类器/分类结果，$\gamma(x)=\argmin_{z\in C} \|x-\mu_z\|,x\in D^0$；
# 3. 更新聚类中心$\mu_z=\frac{1}{N_z}\sum_{\gamma(x_i)=z}x_i,x_i\in D^0\cup D^1$；
# 4. 重复 2-3 直到收敛；
# """))

print(comment.parse_string("""<!--对数似然 = 
   F函数+ K-L散度=Q函数+交叉熵。-->"""))
raise

print(chapter.transform_string(r"""# 降维

本章的题目是“降维”。降维是机器学习的重要方法。降维通常用作某个学习器的预处理。降维处理可以有效减少样本属性数量，减小模型复杂度，从而克服过拟合。

降维的目的是建立一个合理的变换$T:\R^p\to \R^q$，其中$p>q$。这看似不现实，因为信息被压缩了。然而，降维是真实存在的。在分析图像时，生物视觉系统绝地不会将它理解成一个维数为计算机显示图像所需分辨率的数据。在这样庞大的数据空间里，任何视觉系统，无论是数字的还是生物的，将无所作为。这也就是“维数灾难”。因此，可以猜测视觉系统能够抽出部分的“图像特征”重点分析，而不是对每个像素点逐一分析。事实证明，本文介绍的模型确实能起到如同生物视觉那样的“降维”功能，比如在处理图像这类高维数据时有较好的表现。

降维一个有趣应用是平面嵌入：将高维数据降到2维，再用这写2维数据作为平面点集绘制到平面上。这样，人们可以直观观察这些平面上的数据点，来推测数据的内在规律。

本章介绍PCA等经典的降维模型。不过它们的功能不限于降维，如果降维仅指一种预处理方法。

## 矩阵论基础
本章我们主要关注线性降维，即降维变换$T$是线性的，因此可以用一个矩阵表示。为了学习这些降维的方法，我们先掌握一些矩阵论的基础知识。

**约定** 向量**外积**运算$u\circ v:=uv^T:\R^m\times \R^n\to \R^{m\times n}$，并令
$$
A=[[s;U,V]]:=\sum_{k} s_k(u_k\circ v_k)
$$
其中$s=\{s_k\}$。

<!-- *注* 这个抽象值向量称为$U$和$V$的*Khatri-Rao乘积*，记为$U\bullet V$，且总被展平成一个矩阵，即把$u_k\circ v_k$拉直成列向量$vec(u_k\circ v_k)$再横向拼成矩阵。此时有$vec(A)=s^T (U\bullet V)$。 -->

### 矩阵分解
本章介绍的模型几乎都可以归结为矩阵分解问题：对任一$A\in\R^{m\times n}$有下述分解，
$$
A\approx BC,\\
B\in \R^{m\times l},C\in \R^{l\times n},l\leq m,n
$$
其中$B,C$满足一定条件。

(x.x)的近似程度用一个定义在矩阵空间上距离函数$d$衡量（不必是严格意义上的距离，见X.X节），其常见形式如下（可称为独立形式），
$$
d(A,B)=\sum_{ij}d(A_{ij},(B)_{ij})
$$
其中等式右侧的$d$事定义在$\R$上的距离。有时还要考虑(x.x)的“加权形式”：
$$
d(A,B)=\sum_{ij}w_{ij}d(A_{ij},(B)_{ij})
$$

引入矩阵距离$d$后，(x.x)被转化为下述优化问题/统计决策模型，
$$
\min_{B,C}d(A, BC)
$$
其中$B,C$满足一定条件。

### 矩阵的特征值分解（谱分解)和奇异值分解
介绍两个线性代数中的极为重要的矩阵分解。它们本身有广泛的应用，又是一些重要算法的基石。

**定理 特征值分解(EVD)/谱分解**
若$A\in \R^{n\times n}$是对称矩阵，则有矩阵分解，
$$
A= V\Lambda V^T, V\in O(n),
$$

其中$\Lambda$是对角矩阵，其对角线元素为$A$的特征值，$V$的列向量称为特征向量。$A$是对称正定矩阵，记作$A\geq 0$，当且仅当$\Lambda$非负。若$\mathrm{rank} A=r$, 则 $\Lambda$只有$r$个非零数。

*注* 不计顺序，$\Lambda$是唯一的，故总是降序排列对角线上的元素。若$\Lambda$没有相同元素，则不计顺序和符号，$V$也是唯一的。

*注* 对于本章内容，我们不能忽视矩阵分解的唯一性，因为它直接决定了模型的可识别性。当然不具有唯一性也不一定意味着模型是失败的。

**定理 奇异值分解（SVD）**
若$A\in\R^{m\times n}$，则有矩阵分解，
$$
A= USV^T, U\in O_m, V\in O_n
$$

其中$S:\R^{m\times n}$是非负对角矩阵，其对角线元素为$A$的奇异值。(若规定$S$的对角线元素降序排列，则$S$是唯一的。)

*注* SVD中，若$S$没有相同元素，则不计顺序和符号，$U,V$是唯一的。$U,V$中的列向量也被称为奇异向量。

**定义 SVD的紧形式**
人们更倾向于使用**SVD的紧形式**：$A= US V^T, l=\min\{m,n\}$，其中$U\in O_{m\times l},V\in O_{n\times l}$（非方阵的正交矩阵），分别是非紧形式SVD中$U,V$的前$l$列向量形成的子矩阵，$S$是$l$阶方阵，也就是非紧形式中的$S$的$l$阶（顺序）主子矩阵。若$\mathrm{rank} A=r$, 则$S$只有$r$个非零数。此时也可把SVD紧形式写成
\[
  A=U_rS_rV_r^T, U_r\in O_{m\times r},V_r\in O_{n\times r}
\]
其中$U_r,V_r$分别对应于非紧形式SVD中$U,V$的前$r$个列向量，$S_r$是$r$阶方阵，也就是非紧形式中的$S$的$r$阶主子矩阵。

根据约定，SVD还可以写成
$$
A=\sum_{k=1}^r s_ku_kv_k^T=[[s;U,V]]
$$
其中$u_k,v_k$分别是$U_r,V_r$第$k$个列向量，$s=\{s_k\}$。

### 截断SVD
当$r<\mathrm{rank}A$时，分解(x,x)或(x.x)只是近似成立。

**定义 截断SVD**：
（$r$阶）截断SVD为
$$
A\approx \sum_{k=1}^r s_k u_k v_k^T(=U_rS_rV_r^T).
$$
其中$U_r,S_r,V_r$意义同上。

截断SVD至少是三种等价的矩阵优化问题的解。首先，下面这个定理说明了截断SVD等价于所谓的“低秩逼近”。

**Eckart–Young–Mirsky 定理(低秩逼近问题)**

截断 SVD: $A\approx \sum_{k=1}^r s_ku_kv_k^T=U_rS_rV_r^T$, 正好是下述低秩逼近问题的解
$$
\min_{A_r} \|A-A_r\|_F\\
\mathrm{rank} A_r\leq r
$$

其次，易证下述事实。

**事实**
优化问题（X.X）等价于下面两种更易理解的优化问题：
1. 矩阵分解形式：
$$
\min_{V_r, B} \|A-BV_r^T\|_F\\
V_r\in O_{n\times r}, B\in \R^{m\times r}
$$
1. 自编码器形式：
$$
\min_{V_r} \|A-AV_rV_r^T\|_F=\max_{V_r} \|AV_r\|_F\\
V_r\in O_{n\times r}
$$
这里的等价性指（X.X）的解$A_r$一定可以表示成$BV_r^T$和$AV_rV_r^T$。($V_r$的正交性不是必须的，仅仅为了迎合SVD的形式。)

优化问题(X.X)符合矩阵分解形式(x.x)，其中距离是Euclidean距离。它还提供了一种近似求解SVD的方法：用坐标下降法最小化二元函数，
$$
L(B,V_r)=\|A-BV_r^T\|_F
$$
显然相对于每个坐标$B$或$V_r$的优化，都是一个最小二乘问题。这就是著名的**交替最小二乘法(ALS)**。不过单纯的最小二乘法不能保证$V_r$的正交性。

必须注意，解一个固定$r$的矩阵优化问题是得不到SVD的具体形式的。要得到SVD的具体形式，必须解一系列$r=1,\cdots, n$的优化问题，并保证(x)和(x)中的$V_r$是某个特定矩阵$V$的前$r$列。此外，$r\geq r_0$的优化问题依赖$r<r_0$的优化问题，因为要固定它们的解$V_r,r<r_0$。我们称这个求解过程为“相继优化”。这也等价于把（x.x）看做向量值目标函数$\{\|A-BV_r^T\|_F,r=1,\cdots,n\}$的优化问题，其中该目标函数值的序是向量的“字典序”。因此，我们给出下述定义。

**定义 相继优化问题**
一个相继优化问题可以粗略表示为：
$$
\min_x F(x)=\{f_r(x),r=1,\cdots, n\}\\
x\in\Omega
$$
其中目标函数值$F(x)$的序是向量的字典序，$\Omega$表示可行域。

(x.x)的求解过程通常是：先解$r=1$对应的优化问题，得到最优解集$\Omega_1\subset\Omega$。然后解$r=2$且$x\in\Omega_2\subset\Omega_1$对应的优化问题。如此继续，直到得到整个问题的最优解集$\Omega_r$。

### SVD概率解释
最后尝试给出SVD的一个简单概率解释。原因是Frobienus矩阵范数使我们联想到了Gaussian分布。SVD等价于一系列关于矩阵型随机变量的统计模型：

**定义（Gaussian分解模型）**
$$
A\sim N(Z_rV_r^T,\sigma^2_r),r=1,\cdots,n
$$
即$A$中元素独立并服从Gaussian分布(相对于参数$Z, V$)，其中$Z_r，V_r$是矩阵型未知参数，意义同上，而$\sigma^2_r\in\R$不影响参数估计。

注意，这是关于模型(x.x)的参数$Z,V$的MLE构成的一个相继优化问题。

<!-- 我们不是非得把样本矩阵$A$看成样本$X_{i\cdot}$的设计矩阵，并把矩阵分解理解成线性空间的向量分解不可，而是可以把$X$看成一个矩阵值的样本点，或者被排列成矩阵形式的数量值样本。 -->

### 加权SVD

**定义 加权SVD**
加权SVD对SVD矩阵分解形式(x.x)做出下述改造：
$$
\min_{V_r, B} \|W\circ (A-BV_r^T)\|_F\\
V_r\in O_{n\times r}, B\in \R^{m\times r}
$$
其中$W$是事先给定的权重矩阵。

显然，(x.x)符合(x.x)的形式，且求解该问题等价于对$W\circ A$做SVD：先利用SVD $W\circ A=U_rS_rV_r^T$，解得$V_r$，再令$B$满足$W\circ B=U_rS_r$。

### 应用

当我们遗忘一种代数学理论的起源和背景时，回顾该理论的一些应用是非常有益的。这样做有助于我们加深对其内涵和意义的理解。

#### 自编码器

回忆自解码模型的大致形式：
$$
\min_{\phi,\psi} l(x,\psi(\phi(x))),
$$
其中$\phi$和$\psi$分别为编码器和解码器，$l(\cdot)$是某个损失函数。当$\psi,\phi$是线性算子时，称之为**线性自编码器**。显然截断SVD的自编码器形式给出了在线性条件下的一个解。截断SVD也因此成了最原始的编码-解码模型——线性自编码器。

把待分解的矩阵$A\in\R^{m\times n}$理解成$m$个$n$维数据点。我们给出如下事实。

**事实**
截断SVD给出一个线性自编码器：$(V_r,V_r^T)$，其中编码矩阵$V_r$将$\R^n$上的数据点投影到低维空间$\R^r$，解码矩阵$V_r^T$将低维数据重构为原始数据。

*注* 在线性代数里，这个编码-解码的过程也称为“分解-重构”。

#### 图像处理

在计算机中，图像是用像素矩阵存储的，其中每个像素都可用数字表示。我们相信对像素矩阵进行截断SVD可以起到压缩图像的作用。对$N\times N$大小的图像，$r$阶截断SVD可以有$2r/N$的压缩比率。

像素矩阵元素取值于$[0,256)$，而不是$\R$。SVD并不反映这个事实。一个改良技巧是，可以先用1-1映射$p\mapsto \mathrm{logit}((p+0.5)/256)$对图像预处理，再执行截断SVD。当然可以选择其他将有限区间映射到$\R$的1-1映射。如果要重构图像，那么不要忘记对SVD的重构结果做逆变换$x\mapsto 256\mathrm{expit}(x)-0.5$。

【实验】

#### 文本处理

**潜在语义分析（LSA）** 是截断SVD的一个文本处理方面的应用。LSA处理对象是所谓的文档；文档的概念已经在前文介绍过了。LSA有两个步骤：构造词频矩阵，对词频矩阵做截断SVD。

虽然LSA确实可以起到信息压缩的作用，但是人们很难解释对词频矩阵应用SVD的结果的实际意义。由于LSA效果也差强人意，笔者就不演示了。相信读者有能力理解并自行实现之。之后会讨论另一个效果更好且有概率解释的类似LSA的文本分析方法。

## 主成分分析

**主成分分析（PCA）** 是一种非常重要的，可能也是使用最广泛的，知名度最高的降维方法。我们将说明，它本质上就是（截断）SVD。

出于方便，设$X$是$p$维0-期望随机向量，即$EX=0$（否则考虑$X-EX$）。令$\Sigma=Cov(X,X)$为$X$的协方差矩阵。对$\Sigma$做如下特征值分解：
$$
\Sigma = V\Lambda V^T=\sum_k\lambda_kv_kv_k^T,V\in O(p)
$$
若令$Cov Z=\Lambda$，其中$\Lambda$是$p$阶对角矩阵，则可以把$X$表示成$X=ZV^T$。

<!-- *注* 规定$X$是模型输入，而$Z$是输出。 -->

### 基本概念
在PCA的语境中，人们提出很多有现实意义的概念。设$X$是$p$维随机向量，主成分数（对应截断SVD的阶）为$r\leq q$。

**定义**
- $Z_k=Xv_k$称为 **$k$-主成分/因子**，其中$v_k$是(x.x)中的特征向量，称为第$k$个**载荷（loading）**，也称为**主轴（principal axes）/主成分方向**，$k=1,\cdots, p$。
- $r$个主成分张成的$r$维空间称为**主子空间（principal subspace）**。
- $k$-主成分的**贡献率** 定义为$\frac{\lambda_k}{\sum_j\lambda_j}$。相应的**累积贡献率** 定义为$\frac{\sum_{j=1}^k\lambda_j}{\sum_j\lambda_j}, k=1,\cdots, p$，其中特征值$\lambda_j=Var(Z_j)=v_j^T\Sigma v_j, j=1,\cdots, p$。贡献率就是特征值的归一化。
  
*注* 第$k$个**载荷（loading）**有时也定义为$\lambda_kv_k$。

*注* 贡献率更专业的称谓是“被解释方差比率”。我们不是非得计算出所有特征值才能计算某个主成分的贡献率，因为有$\mathrm{tr}\Sigma=\sum_i\lambda_i$。

累积贡献率概念非常有用，可以用来衡量降维的信息损失。当它达到0.9时，我们就很满意了。此外，降序排列的$\lambda_i$的递减速度应该足够快，或者说少数几个特征值的和接近于$1$，否则达不到很好的降维效果。

也可以对相关矩阵，$\rho=corr(X,X)$，进行特征值分解$\rho =V^T\Lambda V$。此时，贡献率和累积贡献率分别为: $\frac{\lambda_k}{p},\frac{\sum_{i=1}^k\lambda_i}{p}$，其中$\lambda_k$是$\Lambda$对角线上第$k$个元素。

### 模型定义
根据上文讨论，给出PCA简明的“代数式”定义。

**定义1 PCA**
设$p$维随机（行）向量$X$，$p$维随机（行）向量$Z$满足且$EX=EZ=0$。PCA由下述等式表示：
$$
X=ZV^T, V\in O_p, Cov(Z)=\Lambda,
$$
其中$\Lambda$是非负对角矩阵（规定对角线上元素降序排列），$V$是模型的未知参数。(若$EX\neq 0$，则可令$X=ZV^T+\mu$，其中$\mu=EX$。)

*注* 定义中，“模型输入”$X$和“模型输出”$Z$之间存在确定的代数关系，因此PCA是一种退化的/确定的模型。

**事实**
PCA等价于SVD，即参数$V$的估计正好是$X=ZV^T$中的$V$，其中$X$是设计矩阵。（参数$V$的估计应该写成$\hat{V}$，但习惯上还是写成$V$。）

*证明* 由于$X$的协方差矩阵$\Sigma$未知，考虑$\hat{\Sigma}\propto X^TX$是样本协方差矩阵，其中$X:N\times p$是设计矩阵。同样对$\hat{\Sigma}$做如下特征值分解：
$$
\hat\Sigma = V\Lambda V^T
$$
这等价于$X$的SVD：$X=USV^T$，其中$\Lambda\propto S^2$。

PCA显然有降维功能：
$$
X\approx Z_rV_r^T
$$
其中$V_r$分别是$V$的前$r$列，$Z_r$是前$r$个元素，其中$r<p$。$X\mapsto X V_r$就是降维映射。

定义X.X的隐变量式的解释是，分布复杂的$p$维随机变量$X$可以通过$p$个分布相对简单的独立的一维随机变量（即主成分）$Z_1,\cdots,Z_p$，线性组合得到，而这些主成分是需要从样本中推断出来的，而且肯定是样本的线性组合，即线性统计量。$Z_k$是构想的，并不能真正观测到它的样本，也就是隐变量。人们自然希望用方差最大的几个$Z_k,k=1,\cdots, r$就能恢复$X$，比如$Z_1$就包含尽可能多的信息。这在降维处理中非常关键。PCA 用方差近似地衡量随机变量的信息，因此我们有，
$$
\max_{v_1} Var Z_1\\
Z_1=X\cdot v_1,\|v_1\|=1
$$
其中$v_1$是$V$的第一列，并做2范数下的归一化。对于$Z_2$，要求也是一样的，但是$Z_2$不应该包含$Z_1$的信息。这一点由协方差$Cov(Z_1,Z_2)=0$来保证。同时$v_2$归一化。以此类推，求出所有$Z_j,j=1,\cdots, p$。听上去很复杂，但是如前所述，利用SVD就可解出$v_j$。

上述讨论给出PCA另一个定义。

**定义2（PCA）**
PCA是下述优化问题：
$$
\max_{v_j} \lambda_j=Var Z_j, Z_j=X\cdot v_j, \\
\|v_j\|=1, Cov(Z_j, Z_k)=0,k<j, j=1, \cdots, p,
$$
其中$v_j$是$Z_j$的载荷。 

(X.X)是一种相继最大化问题。这个定义其实更能反映PCA原始思想。只是定义X.X更简洁，也符合隐变量模型的形式。

*注* PCA之所以对服从Gaussian分布的数据效果好，就是因为Gaussian分布的方差相当于熵，而正交性等价于独立性。不过，PCA对一般数据依然起到良好的作用。

另一种思路是直接定义PCA的风险函数。

**定义3 PCA统计决策模型**
PCA降维等价于下述一系列统计决策（只写期望风险）：
$$
\min_{V_r\in O(p\times r),Z_r} E\|X- Z_rV_r^T\|_2^2 = \min_{V_r\in O(p\times r)}  E\|X- XV_rV_r^T\|_2^2\\
= \max_{V_r\in O(p\times r)}  \{E\|XV_r\|_2^2\},r=1,\cdots,p
$$
其中$X$是$p$维随机（行）变量，$Z_r$是$r$维随机（行）变量，且各分量独立，$V_r$始终是某个正交矩阵$V\in O(p)$的前$r$列。

设$\Sigma=Cov X$。则有
$$
E\|XV_r\|_2^2 = V_r^T\Sigma V_r
$$
我们立刻发现PCA和LDA降秩（Fisher问题）之间的区别：前者只考虑总体方差，而后者考虑了类内方差和类间方差。

### 算法

我们已经介绍完PCA原理。现在，我们写出完整的PCA算法，即模型(X.X)的参数估计。

**算法 PCA算法**
输入：样本矩阵$X:N\times p$
返回：变换$V$和变换结果$Z$
1. 先对$X$中心化，依然记为$X:N\times p$，中心记为$\mu$。计算样本协方差$\hat{\Sigma}=Cov(X):p\times p$；
2. 对$\hat\Sigma$执行特征值分解，得到$Cov{Z}=\Lambda =V^T \hat{\Sigma} V$；(如果指定了$r$个主成分，那么对$\hat\Sigma$执行（$r$阶）特征值分解。此时$Z:N\times r, V: p\times r$。)
3. 执行变换$Z=XV: N\times p$；

*注* 算法X.X用的是特征值分解，但结果和直接对$X$做SVD是一样的。

*注* 重构较为简单，只需令$\hat{x}_i=z_iV+\mu$。如果已知$EX=0$，那么不应该对样本做中心化处理，即使样本均值非0。

因为有中心化处理，所以，如果说截断SVD是线性自编码器，那么PCA就是仿射自编码器，即编码和解码都是仿射变换。

<center>PCA示意图，相互垂直的两个绿色线条表示两个主成分</center>

![](https://programmathically.com/wp-content/uploads/2021/08/pca-2-dimensions.png)

### Karhunen-Loeve(KL) 基/变换

回顾$\R^p$上正交分解开始：
$$
X=\sum_{k=1}^r\alpha_kv_k,
$$
其中$\{v_k\}$是标准正交基, $\alpha_k=\langle X, v_k\rangle$。系数向量$\{\alpha_k\}$是变量$X$在基$\{v_k\}$下的坐标，根据X.X节符号约定记成$\{\alpha_k\}=crd_{\{v_k\}}(X)$。(6)的矩阵形式为 $X=ZV^T$，其中$Z$的第$k$列$Z_k=\alpha_k$, $V$是列向量 $v_k$组成的矩阵。

*注* (6)不需要限制在$\R^p$上，在一般内积空间上也有意义。

现在设$X$是$p$维随机向量。系数$\alpha_k$就成了是随机变量，即随机变量$X$在基$\{v_k\}$下的随机坐标。

出于方便，这里也假定$EX=E\alpha=0$，否则考虑分解，$X-EX=\sum_k\alpha_kv_k$。

我们来简单解释一下(6)的统计学意义。对固定的向量进行正交分解是我们熟悉的；每种近似分解都有固定的误差$\|X-\sum_{k=1}^r\alpha_kv_k\|^2$，其中$r$是基的个数。但现在$X$是随机的。正交分解肯定也是不确定的；误差也是随机的。我们的目标是，用$\{v_k\}$的一个子集有效地表示$X\approx \sum_{k=1}^r\alpha_kv_k$，即使误差的期望值尽可能小。

**定义 Karhunen-Loeve（KL）模型**
暂时用$O_{p,r}$表示所有含$r$个$p$维向量的标准正交系构成的集合 (对应于正交矩阵集$O(p\times r)$)。KL模型就是下述优化问题：
$$
\min_{\{v_k\}\in O_{p,r}} E\|X-\sum_{k=1}^r\alpha_kv_k\|^2
$$
作为（7）的最优解，$v_k$称为**KL-基**。相应的优化算法提供了一个**KL-变换**: $X\mapsto \alpha$。严格地说，(x.x)应该被理解成$r=1,\cdots p$的相继优化问题。

将优化问题(X.X)写成矩阵形式，立刻发现它等价于(x.x)。因此我们有下述事实。

**事实**
KL模型和PCA是等价的。**KL-基**$v_k$就是PCA中的载荷，而相应的坐标$\alpha_k$就是PCA中的因子。

KL模型和PCA是完全等价的，只是两者看问题的角度稍有不同：PCA把$X$看成分量为随机变量的向量，而KL模型把$X$看成取向量值的随机变量。

<center>KL变换示意图，每个数据点对应于KL基上的坐标（主成分的值）</center>

![](https://programmathically.com/wp-content/uploads/2021/08/pca_figure1-2048x2048.jpeg)

和Fourier变换、小波变换的基不同，KL基(SVD/PCA的基)需要通过数据学习，而不是事先指定。我们可以称这样的基为“**经验基**”或“**自适应基**”。经验基提高了模型的自适应能力，但也增加了计算量。

### PCA的几何解释

KL变换导致的误差$\|X-\sum_{k\leq r}\alpha_kv_k\|$，正好是$X$到$\{v_k,k=1,\cdots,r\}$张成的子空间的距离。因此PCA的几何解释就是，找到一个子空间，使得所有数据点到这个子空间的平均距离最小。由此也不难想象，主子空间总是贯穿整个数据点集。

根据这个事实/解释，我们可以提出一个更一般的降维方法：“主子流行分析”，即
$$
\min_{M} E d(X,M)
$$
其中$M$遍历$p$维样本空间$\mathcal{X}$中某一类$r$维子流行，$d(x,M)$表示点$x$到$M$的距离。特别当$r=1(/2)$时，称之为“主曲线(/曲面)分析”。此外，K-means聚类其实是一种$0$维子流形(有限点集)分析，可称为“主点分析”(见习题x.x)。关于主子流形分析的定义该不够严格。由于涉及新领域的知识，我们也不打算深入讨论这种降维方法。

显然，PCA是主子流形分析最简单的一种，其中子流形是样本空间$R^p$的子空间，可称为“主子空间分析”。至此，我们至少给出了五种PCA的等价刻画。

### 主成分数选择

PCA主成分数和Kmean聚类数的概念相当，但它的选择有所不同。Kmean聚类数的选择主要参考聚类数对损失函数变化的影响程度，而PCA主成分数的选择主要参考累计贡献率的大小。前文提到我们希望累积贡献率能达到0.9。我们可选择此时对应的主成分数。

<!-- ### 代码 -->

<!-- #### numpy

numpy提供了实现SVD的函数`svd`。我们可以在此基础出上实现PCA。

```python
import numpy as np
import numpy.linalg as LA
# X is the matrix of samples
U, s, Vh = LA.svd(np.cov(X.T), full_matrices=True)
Z = X @ Vh.H  # =X @ U
# Vh.H == U == loadings/ eigen vectors
# eigenvals for lambda_i
```

#### scikit-learn 实现
numpy/scipy提供了实现SVD的函数`svd`。我们可以在此基础出上实现PCA。我们推荐直接用scikit-learn提供的`PCA`。下面解释scikit-learn的`PCA`对象的主要属性和方法的意义。

```python
import numpy as np
from sklearn.decomposition import PCA

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
pca = PCA(n_components=2)
pca.fit(X)

# pca.explained_variance # eig(Sigma)/N
print(pca.explained_variance_ratio_)   # lambda_i / sum_i lambda_i
[0.9924... 0.0075...]
print(pca.singular_values_)            # sigma_i (sqrt lambda_i)
[6.30061... 0.54980...]
pca.components_     # V^T (row vectors of `components_` are) loadings
Z = pca.transform(X)    # Z=XV, factors
pca.inverse_transform(Z) # Xi
pca.inverse_transform(pca.transform(X)) # reconstruct/projection
```

<center>`PCA/scikit-learn`属性与方法的含义（$X$已中心化）</center>

| `PCA/scikit-learn`属性与方法         | PCA概念              | 符号                                                    |
| ------------------------- | ------------------ | ----------------------------------------------------- |
| components_               | 载荷（loadings）/KL-基（主轴、主成分方向） | $V^T$ ($V$的列向量)                                          |
| X_transformed             | 因子（factors）/主成分/KL-坐标        | $Z(=XV)$                                            |
| singular_values_          | 奇异值                | $\sigma_j=\sqrt{\lambda_j}$                           |
| explained_variance_       | 特征值                | $\lambda_j$                                           |
| explained_variance_ratio_ | 贡献率                | $\lambda_j / \sum_j \lambda_j$ ($\lambda_j$: 特征值) |
| transform                 | KL-变换              | $XV$                                                |
| inverse_transform         | KL-变换的逆            | $ZV^T$                                              |
| mean_                   | 中心/均值              | $\mu$                                                 |

#### statsmodels实现

```python
# data: n X p - array
# not standardized, not normalized
# data: Variables in columns, observations in rows
from statsmodels.multivariate.pca import PCA

pc = PCA(data, ncomp=p, standardize=False, normalize=False) # Z=Xe
# pc.loadings: e^
# pc.factors: Z^
# pc.factors == data (demean) * pc.loadings
# pc.coeff = pc.loadings ^ -1
``` 
-->

<!-- #### 源码解读

下面是`scikit-learn`提供的类`PCA`关键方法 `_fit_full`的代码片段。

```python
# Center data
self.mean_ = np.mean(X, axis=0)
X -= self.mean_

U, S, Vt = linalg.svd(X, full_matrices=False)
# flip eigenvectors' sign to enforce deterministic output
U, Vt = svd_flip(U, Vt)

components_ = Vt

# Get variance explained by singular values
explained_variance_ = (S ** 2) / (n_samples - 1)
total_var = explained_variance_.sum()
explained_variance_ratio_ = explained_variance_ / total_var
singular_values_ = S.copy()  # Store the singular values.
``` 
-->

<!-- #### 代码

图像是高维数据，可用PCA进行降维。一般来说，图像总是用像素向量表示，整个图像数据集组成一个“图像-设计矩阵”。执行PCA后，可以得到特征向量，称为*特征图像*。若是人脸数据集，就称之为*特征人脸*。根据上文讨论，我们可以对图像进行编码、平面嵌入或者生成新的图像。

这是笔者的作品：[图像编码与生成](https://gitee.com/williamzjc/image-encoder)。
-->

### 生成任务和概率PCA

严格地说，PCA不是一个生成模型，因为没有明确给出$P(X,Z)$的分布。不过，由于$Z$的维数较低，生成$Z$要比直接生成$X$容易。生成任务的粗略做法就是独立地生成$Z$每个分量。

引入 Gaussian 型假设：$p$维随机向量$X$服从Gaussian分布，而$Z$是独立地服从Gaussian分布的。具体地，
$$
X=ZV^T=U D V^T,U\sim N(0,1)
$$
其中$U$是$p$维随机向量，$D$是$p$阶对角矩阵。因此我们直接从$N(0,1)$采样，然后作用$DV^T$即可生成$X$。设置主成分数为$r$，并对相关矩阵进行截断。采样$L$次：
$$
U_r:L\times r \sim N(0,1)\\
X\approx Z_rV_r^T=U_rD_rV_r^T:L\times p
$$

样本生成或者采样是机器学习中的重要任务也是手段。从复杂的分布进行采样是非常有挑战性的。人们自然希望，从极其简单的分布（如标准正态分布）中采样就可以模拟复杂分布。现在，PCA使这个愿望成为可能，尽管它的功能还比较简单。

上述讨论引出一个真正的统计模型，其中$X,Z$之间没有给出决定型关系。

**定义 (概率PCA)**
和PCA不同，概率PCA(pPCA)给出了明确的分布：
$$
X|Z\sim N(ZW^T,\sigma^2I),Z\sim N(0,1),
$$
其中$X$是$p$维随机向量，$Z$是不可观测的$r$维随机向量，$W\in\R^{p\times r}$。

PPCA当然是一种生成模型，且有$X\sim N(0,WW^T+\sigma^2)$。参数估计一样可以用SVD得到。

**算法 PPCA参数估计算法**
1. 计算样本协方差矩阵$\hat\Sigma$；
2. 执行特征值分解（或截断SVD）：$\hat\Sigma=V \Lambda V^T, \Lambda=\mathrm{diag}\{\lambda_k,k=1,\cdots,p\}$;
3. $\sigma^2$的MLE，$\hat\sigma^2=\frac{1}{p-r}\sum_{k=r+1}^p\lambda_k$；
4. $W$的MLE，$\hat W=V_r(\Lambda_r -\hat\sigma^2)^{1/2}$，其中$V_r$是$V$的前$r$列，$\Lambda_r$是$\Lambda$的$r$阶顺序主子式；

不难证明下述事实。

**事实**
当$\sigma^2\to 0$时，pPCA退化成PCA。也就是说，PCA是pPCA的极限形式。(两者之间的关系类似于GMM和Kmeans之间的关系。)

*注* $\hat W$一般不是正交矩阵。此外，$\hat W=V_r(\Lambda_r -\hat\sigma^2)^{1/2}Q$都是允许的，其中$Q\in O(r)$。为了避免这种歧义，可以实现规定$W$只能分解成一个正交矩阵和对角矩阵的乘积。

### 因子分析模型

可以定义一个比PCA/pPCA更一般的模型。

**定义（因子分析模型）**
因子分析模型（FA）可表达为:
$$
X=ZW+\epsilon,
$$
其中$X$为$p$维随机向量，$Z$是不可观测的$r$维随机向量, 而随机向量$\epsilon$表示误差且$E\epsilon=0$(一般要求$\sigma$服从正态分布)。$W$是一个大小为$r\times p$的矩阵，不一定正交。

*注* (x.x)完全就是一个线性回归的形式。
*注* 一般当$r\leq p$时，FA模型才有意义。一般我们会合理地规定$Z\sim N(\mu,\Lambda)$, 其中$\Lambda$是对角矩阵，(出于方便可令$\mu=0$)。

显然，根据定义的形式FA的求解归结为矩阵近似分解。这和PCA是相似的；两者的解也应该是近似的；而且如果对$Z$增加独立性限制，那么两者可被认为是等价的。在模型的现实意义上，FA和PCA有细微的差异：前者强调作为随机向量的样本$X$的每个分量近似是一些固定隐变量$Z_1,\cdots,Z_r$的线性组合，后者强调主成分$Z$的各个分量是样本$X$的线性组合；前者的目标是隐变量$Z$的线性组合尽可能解释样本$X$；后者希望样本$X$的线性组合$Z$尽可能保留信息。“因子分析”和“成分分析”看问题的视角似乎是相反的。不过，“因子”和“成分”都是模型里的隐变量，往往得到相似的模型。


## 独立成分分析 (ICA)

ICA和PCA目的相似的，都是要从混合信号中分离出不相关的主要信号。它的发明源自一类和“鸡尾酒派对”相关的问题：在嘈杂的环境中，我们如何听清任何一个被自己关注的人说的话。ICA不再强调分离出的信号的正交性，而是直接关注其独立性。这使得ICA可以处理任何混合信息，而不像PCA那样只对服从Gaussian分布的数据表现出色。然而，作为一种通用的成分分析模型，ICA无法用代数方法求解，因此算法对计算效率要求更高。

### 模型

根据上文描述，ICA似乎跟符合我们心目中对信号分解的要求。下面给出ICA的准确定义。

**定义（ICA 模型I）**
ICA也是一种FA模型: $X\approx ZW$，其中隐变量$Z$各分量是独立的，称为**独立成分**，即
$$
Z_k\perp\!\!\!\perp Z_j,k\neq j = 1, \cdots, r.
$$
其中参数$W$被称为**混合矩阵**，表示独立信号$Z$混合的方式，一般被假定是满秩的。

在特情况下，即$Z$服从Gaussian分布，且$W\in O(r\times p)$，ICA和PCA等价。一般情况下，我们要建立新的算法推断未知参数$W$和对应的$Z$。

根据定义X.X，ICA很难被直接求解/参数估计。需要先给出独立性的量化方法。这时我们想到用互信息$I(Z)$充当ICA的风险函数。

**定义（ICA 模型II）**
ICA 模型表示为下述优化问题：
$$
\min_{Z,W} I(Z)\\
X=ZW, W:r\times p,
$$
其中$X$是$p$维随机向量，$Z$是不可观测的$r$维随机向量。（出于方便，约定$EX=EZ=0$）

然而(13)没有PCA那样的代数解，计算很耗时。为此人们开发了许多近似模型/算法。其中之一是FastICA 模型[]。这个模型来自这样的idea：随机向量$Z$的非Gaussian性达到最大时，$Z$就接近独立。这个idea 显然受中心极限定理的启发。于是(13)可近似表达为：
$$
\max_{Z,W} NG(Z)\\
X=ZW,
$$
其中 NG 是一个衡量随机向量的非Gaussian性的函数。

<!-- ### 源码解读

下面是`scikit-learn`提供的`FastICA` 算法源码核心片段。

```python
p_ = float(X.shape[1])
for ii in range(max_iter):
    gwtx, g_wtx = g(np.dot(W, X), fun_args)
    W1 = _sym_decorrelation(np.dot(gwtx, X.T) / p_ - g_wtx[:, np.newaxis] * W)
    ...
    lim = max(abs(abs(np.diag(np.dot(W1, W.T))) - 1))
    W = W1
    if lim < tol:
        break
```  -->

### 应用

#### 去噪

去噪的基本假设是噪声和原信号独立。ICA理论上可以分离出这两个信号，达到去噪的目的。

#### 生成

一般情况下它比PCA更有效, 因为我们总认为因子是独立的，但不必然服从正态分布。

由于ICA没有给出明确的分布形式，它也不能算是生成模型。ICA图片生成算法的原理和PCA是一样的：对降维后的数据进行核密度估计。

**算法 ICA生成算法**
输入：数据集$X$
输出：新的数据集$X^*$
1. 用ICA计算$X$对应的独立成分$Z$；
2. 由于独立性，只需对$Y$的每一列进行1维核密度估计，进而用1维分布独立生成$Z^*$的每一列；
3. $X^* = Z^*W$；


![](../src/hanzi1.jpg)

<!-- 金融问题
![](../src/ICA-demo.png)
-->

## 非负矩阵分解

**非负矩阵分解（NMF）**也是一种极其常用的矩阵分解模型/算法，目前被用于图像处理、文本分析等众多邻域。和PCA或ICA相比，对一些特殊矩阵的分解，NMF有更好的效果，比如图片（集）的像素矩阵、文本分类中的词袋和加权图的邻接矩阵。为使NMF更为实用，人们还开发了相应的快速算法。本节要介绍的就是NMF的模型及其算法。

### 模型介绍

**非负矩阵**指所有元素非负的矩阵，可记为$\R_{+}^{m\times n}$。大小为$N\times p$的非负矩阵$X\in\R_{+}^{N\times n}$总有如下近似低秩分解：
$$
X\approx WH,
$$
其中$W\in \R_{+}^{N\times q}, H\in \R_{+}^{q\times p}$, $q\leq p$。在机器学习的语境下，$X$就是设计矩阵，$W,H$是未知参数。

从线性空间分解的角度看，$X$的每一行都有近似分解
$$
X_{i\cdot}\approx\sum_kw_{ik} h_k,
$$
其中$H$的行向量族$\{h_k\}$作为分解的基，$W$的每一行是对应$X$行向量的系数，视为隐变量。在PCA中，相应分解式的基被称为特征向量，而在NMF中，我们可用“主题向量”称呼它。


*注* 其实，NMF中的$H$不一定是基，只是一个用来重构的向量族，但是限定它是基也是合理的：$H$的行向量线性独立。一组被用来重构信息的向量，在本书中都统称为基，也许称为“号码簿 (codebook)” 或“字典（dictionary）”更合适。NMF中，笔者直接称$H$为基。

*注* (15)是NMF最常见的记法。有的作者写成$X\approx WH^T$；还有的把$W$看成基。这显然不是本质的。

按照x.x节为NMF引入距离：设$d(x,y)$是$\R^+$上一个合理的距离函数/损失函数，并将它自然地提升到非负矩阵上，
$$
d(A,B):=\sum_{ij}d(A_{ij},B_{ij}),
$$

**定义 NMF** 
关于设计矩阵$X\in\R_{+}^{N\times n}$的NMF可由下述优化问题表示：
$$
\min_{W,H}d(X,WH),W\in \R_{+}^{N\times q}, H\in \R_{+}^{q\times p}
$$
其中$d(X,WH)$就是矩阵分解(X.X)的损失函数。一般要求$q< p$，否者(x.x)是平凡的。

NMF中两个常用的距离是 **广义Kullback-Leibler散度**（简称散度）和**Euclidean距离**，分别定义为，
$$
d(x\|y):=x\ln \frac{x}{y} -x+y,x,y\geq 0\\
d(x,y):=|x-y|^2
$$

*注* 散度被写成$d(x\|y)$，是用来强调该函数不是对称的。令$d(0\|y)=y,d(x\|0)=\infty, x\neq0$。

我们可以将NMF的$W$作为$X$的编码，但是NMF本身并没有直接提供编码器。这一点和PCA不同。我们不得不通过求解下述优化问题，为新的样本$x$计算其编码。
$$
\min_{w}\|x-wH\|
$$
其中$x:N'\times p$是大小为$N'$的新样本，$w:N'\times r$非负，$H:r\times p$已知（已被估计）。(x.x)没有显式解，因此可被称为“隐式自编码”。

### 乘法更新规则
特殊的约束使NMF和多数机器学习模型一样没有代数解法。人们转而构造基于迭代的近似优化方法算法，其中最著名的优化方法由Lee 和Seung(2001)提出的**乘法更新规则（MU）**。

对应于两个距离(x.x)和(x.x)的NMF分别被称为**散度-NMF**和**Euclidean(Gaussian)-NMF**，并可分别表示为下述两个优化问题：
$$
\min_{W,H} D(X\|WH)=\sum_{ij}d(x_{ij}\|(WH)_{ij}),
$$
和
$$
\min_{W,H} \|X-WH\|_F^2=\sum_{ij}|x_{ij}-(WH)_{ij}|^2,
$$
其中$X,W,H$都是非负矩阵，$\|\cdot\|_F$表示Frobenius矩阵范数。

(19)显然可以用ALS求解，但是迭代过程中要保持非负性，如将负数值修正为0，即非负最小二乘法。（18）可用GD求解，其关键是计算梯度：
$$
\begin{cases}
\nabla_{w_{ik}}L=\sum_j h_{kj}x_{ij}/(WH)_{ij}-\sum_j h_{kj},\\
\nabla_{h_{kj}}L=\sum_i w_{ik}x_{ij}/(WH)_{ij}-\sum_i w_{ik}.
\end{cases}
$$

这也会导致负数解，如果不做任何修正的话。一种由GD导出的近似快速算法，即乘法更新规则，可以自然地克服这个问题。

求解（18）的MU为，
$$
\begin{cases}
w_{ik} \leftarrow w_{ik}\frac{\sum_j h_{kj}x_{ij}/(WH)_{ij}}{\sum_j h_{kj}},\\
h_{kj} \leftarrow h_{kj}\frac{\sum_i w_{ik}x_{ij}/(WH)_{ij}}{\sum_i w_{ik}},
\end{cases}
\\
\iff
\begin{cases}
W\leftarrow W\circ (X\oslash WH)H_1^T,\\
H\leftarrow H\circ (W^T)_1(X\oslash WH).
\end{cases}
$$
其中 $X_1$是$X$的行归一化（若$X$非负，则$X_1$是随机矩阵）。

相似地，求解（19）的MU为：

$$
\begin{cases}
w_{ik} \leftarrow w_{ik}\frac{(XH^T)_{ik}}{(WHH^T)_{ik}},\\
h_{kj} \leftarrow h_{kj}\frac{(W^TX)_{kj}}{(W^TWH)_{kj}},
\end{cases}
\\ \iff
\begin{cases}
W \leftarrow W\circ(XH^T)\oslash (WHH^T),\\
H \leftarrow H\circ(W^TX)\oslash (W^TWH).
\end{cases}
$$

*注* 对于Euclidean距离，也可用ALS算法。但要注意非负性限制。

乘法更新规则的推导其实很简单。设GD的迭代格式为: $x\leftarrow x+\lambda(y-z)，y,z\geq 0$，则令 $\lambda=\frac{x}{z}$，就可以改造成MU：$x\leftarrow x\frac{y}{z}$。显然，MU迭代过程中，$x$不变号。上文所有MU，都是这样导出的。做一下简单的调整：令$\lambda=\frac{x}{z+\alpha}$，则有迭代，$x\leftarrow x\frac{y+\alpha}{z+\alpha}$，其中$\alpha$是可调的超参数 ($\alpha\geq 0$或者在迭代过程中保证$y+\alpha,z+\alpha\geq 0$)。例如，（13）可改造为，
$$
\begin{cases}
w_{ik} \leftarrow w_{ik}\frac{\sum_j h_{kj}x_{ij}/(WH)_{ij}+\alpha}{\sum_j h_{kj}+\alpha},\\
h_{kj} \leftarrow h_{kj}\frac{\sum_i w_{ik}x_{ij}/(WH)_{ij}+\alpha}{\sum_i w_{ik}+\alpha}.
\end{cases}
$$
显然对每个分量，$\alpha$可以不同。

#### 随机矩阵的NMF

**定义（随机矩阵分解）**
如果NMF中非负矩阵$X$是更为特殊的随机矩阵，那么可以要求$W,H$皆为随机矩阵。这种分解成为**随机矩阵分解(SMF)**。

SMF的求解没有特别之处，只需在每次用（21）、（22）迭代时，对$W,H$做归一化处理。

现考虑把散度作为损失函数。根据随机矩阵的性质有
$$
D(X\| WH) = H(X,WH)+C\\
=-\sum_{ij}x_{ij}\log (WH)_{ij}+C
$$
其中$C$是一个只与$X$有关的常数。此时，归结为下述优化问题——这是一个用交叉熵作为损失函数的优化问题：
$$
\min \{L(W,H)=H(X,WH)\}.
$$
遗憾的是优化问题(24)没有对应的乘法跟新规则，只能改用其他约束优化方法。

#### NMF/SMF几何解释

仔细观察SMF：
$$
X_{ij}\approx\sum_{k=1}^q W_{ik}H_{kj}.
$$
我们把概率向量$h_k=H_{k\cdot},k=1,\cdots q$看作基。这些基分布在$p-1$维的单纯形$\Delta^{p-1}$上，而$X_{i\cdot}$是基的凸组合，其中系数（重心坐标）为$\{W_{i\cdot}, k=1,\cdots,q\}$。显然这个凸组合包含在$\Delta^{p-1}$的$q-1$维子单纯形中。由于存在误差，现实中$X_{ij}$分布在该子单纯的小领域上。

NMF有类似的几何解释：基向量$h_k=H_{k\cdot},k=1,\cdots q$分布在$p$维锥$\R_{+}^{p}$上，而$X_{i\cdot}$是基的非负组合，其中系数为$\{W_{i\cdot}, k=1,\cdots,q\}$，分布在$q$维子锥（的小领域）上。

<!-- 【示意图】 -->

<center>三类矩阵分解模型和几何体的关系</center>

|模型|数据分布的几何体|将维后的几何体|
|---|---|---|
|PCA|全空间|低维子空间|
|NMF|锥|低维子锥|
|SMF|单纯形|低维子单纯形|

#### 矩阵三因子分解

最后注意，NMF的解通常是不唯一的。若$(W,H)$是NMF的解，则 $(WA, BH)$ 也是, 其中 $AB=1$, 并保持$WA,BH$非负性。把$X$分解成三个矩阵的乘积可以一定程度克服解的不唯一性：
$$
X\approx WDH,
$$
其中$D$是非负对角矩阵且其对角线上元素降序排列，而$W$列归一（column stochastic），即列向量范数为1，$H$行归一（row stochastic），即行向量范数为1。这里的归一化可以基于任何范数，但默认基于1范数，此时$W^T,H$为随机矩阵。我们将（26）和SVD进行比较。两者在形式上是一样的。正如SVD的奇异值表达了特征向量的重要性，$D$的对角线元素表达了主题向量的重要性——不妨称它们为重要性系数。和SVD一样，我们希望看到的是重要性系数快速衰减。(X.X)只是对NMF做了改写，也被称为“NMF标准形式”。

(X.X)暗含了一种新的矩阵分解形式。请看如下定义。

**定义 非负矩阵三因子分解（NMTF）**：
非负矩阵$X$有如下分解：
$$
X\approx WDH,
$$
其中$W,H$和$D$都是一般的非负矩阵（$D$不一定是对角矩阵；$W,H$也不一定是归一化的）。

#### NMF的概率解释

本小节给出NMF的概率解释。SVD/PCA和正态分布相联系，而NMF和Poisson分布相联系。

之前，我们提到过设计矩阵$X$可被视为排列成矩阵形式的数量值样本，所以定义下述模型。

**定义 Poisson 分解(PF)模型**
Poisson 分解(PF)模型假设，
$$                                                
X\sim P(WH),
$$
即设计矩阵$X$中的元素$x_{ij}\sim P((WH)_{ij})$，且独立（准确地说，相对于未知参数$W,H$条件独立）。

这个PF模型就是NMF的概率解释。读者应该认识到，这个解释只适用于$X$元素是自然数的情形。

根据Poisson分布的似然函数$\ln p(x|\lambda)\sim x\ln (\lambda/x)-\lambda$，写出PF的似然函数，
$$
l(W,H)=\sum_{ij}\big(x_{ij}\log ((WH)_{ij}/x_{ij})-(WH)_{ij}\big)\\
\sim -d(X\|WH).
$$
其中$d$是散度。

**事实**
若$X$的元素均为自然数，则NMF（选用散度作为距离）等价于PF。

*注* 和统计模型标准定义略有不同，PF是描述整个样本$X$的统计模型。在PF中，$W,H$都是参数，而不必把其中一个看成隐变量。见X.X节就提到过，参数可以被视作隐变量。

下面做一个简单的改进就可以令PF适用于连续值情形。我们引入一个伸缩因子$\nu>0$，使得
$$
\lfloor{\nu x_{ij}}\rfloor\sim P((WH)_{ij}).
$$
参数估计是建立在矩阵$\lfloor{\nu X}\rfloor$上。因子$\nu$可以控制取整运算带来的信息损失。重构时，可根据$x\approx \frac{WH}{\nu}$近似估计。

<!-- #### 原型分析（archetypal analysis）。

**定义 原型分析**
原型分析定义为下述非负线性自编码器：
$$
\min_{W,B}\|X-WBX\|
$$
其中$W,B$通常被要求是随机矩阵，分别充当编码器和解码器。 -->

### 概率潜在语义分析(PLSA/PLSI)

PLSA是LSA的概率版本，比LSA更接近实际情况，而且有确切的概率解释。[]它也可认为是MNF在文本处理中的应用。

先回忆词袋$N(w_j,d)$：$w_j$是词典$\mathcal{W}$中第$j$个词；$d$是文档随机变量，取值于文档集$\mathcal{D}$。再引入主题随机变量$z$，其样本空间记为$\mathcal{Z}$。

*注* 在pLSA中，文档并不完全等同于词序列，相反，可认为是一个特殊的随机变量，其样本空间记作$\mathcal{D}$。$\mathcal{D}$是一个不必做任何解释的抽象集合。主题是新引入的概念。主题空间$\mathcal{Z}$是一个有限集。一般$\mathcal{Z}$的元素个数总是远小于$\mathcal{W}$的。

*注* 和所有离散随机变量一样，可以用数字直接为主题编号，也可以用主题名表示，如$\{z_1,\cdots, z_k\}$。

PLSA的基本假设是存在条件独立关系：
$$
d\perp\!\!\!\perp w|z
$$

#### PLSA的定义
PLSA有两种形式：生成模型和共现模型。我们先来考察第一种形式。

**定义（PLSA生成模型）**
根据基本假设(X.X)，模型可由下述条件-联合分布刻画：
$$
p(w,z|d)=p(w|z)p(z|d), (w,d,z)\in \mathcal{W}\times  \mathcal{D}\times \mathcal{Z}
$$
其中$z$是隐变量，而模型未知参数是主题-词分布$p(w|z)$和映射$d\mapsto p(z|d)$。人们习惯于它的样本形式（条件-边缘似然）：
$$
L(p;D)=\prod_{ij}p(w_j|d_i)^{N(w_j,d_i)},\\
p(w_j|d_i)=\sum_z p(w_j|z)p(z|d_i).
$$
其中$D$表示词-文档样本，随机矩阵$p(w_j|z),p(z|d_i)$是未知参数(用$p$概括)。($N(w_j,d_i)$是词袋)

概率图为：$d \to z \to w$。

*注* PLSA生成模型是非参数模型。但是，定义中那样把它理解成参数模型并无不妥。此外，映射$d\mapsto p(z|d)$的估计只能限制在样本$d_i$上。因此把随机矩阵$p(z_k|d_i)$当做参数也是允许的。(注意，参数本身不能随样本变化而变化)如果这是一个参数映射$d\mapsto p(z|d,\beta)$，那么$\beta$就是PLSA常规的参数。

*注* 不难发现，固定$d$，词分布$p(w|d)$其实也是一个混合分布——Categorical 混合分布，其中基分布为$p(w|z),z\in\mathcal{Z}$。

**事实** 模型(32)等价于NMF/SMF：$X\approx WH$。（因此SMF的几何解释就是pLSA的几何解释。）

*证明* PLSA参数估计的基本原理（由MLE导出）是，估计$\sum_k \hat{p}(w_j|z_k)\hat{p}(z_k|d_i)$应该和词袋$\hat{p}(w_j|d_i)\sim N(w_j,d_i)$接近，即
$$
\sum_z \hat{p}(w_j|z)\hat{p}(z|d_i)\approx \hat{p}(w_j|d_i).
$$

把(X.X)写成矩阵分解的形式：$X\approx WH$，其中
$$
X_{ij}=\hat{p}(w_j|d_i),W_{ik}=\hat{p}(z=k|d_i),H_{kj}=\hat{p}(w_j|z=k)
$$
显然这是一个NMF/SMF（所有矩阵都是随机矩阵），其中主题-词分布$p(w|d)$就是主题向量。

PLSA生成模型模拟一个文档的书写过程：先选定一个文档（选择的概率并不重要），再以一定概率选择一个主题，然后根据主题选择一个具体的词并将其写入文档，如此继续写入任意多个词。

下面讨论pLSA的另一种形式：共现模型。概率图为 `z-> w, d`

**定义（PLSA共现模型）**
PLSA共现模型分布为，
$$
p(w,d,z) = p(w|z)p(z)p(d|z), (w,d,z)\in \mathcal{W}\times  \mathcal{D}\times \mathcal{Z}
$$
其中$z$是隐变量，$p(w|z), p(z)$和映射$d\mapsto p(d|z)$是未知参数。它的样本形式（边缘似然）为：
$$
L(p;D)=\prod_{ij}p(w_j,d_i)^{N(w_j,d_i)},\\
p(w_j,d_i)=\sum_z p(w_j|z)p(z)p(d_i|z),
$$
其中$D$表示词-文档样本，随机矩阵$p(w_j|z_k), p(d_i|z_k)$和概率向量$p(z)$是未知参数(用$p$概括)。主题边缘分布$p(z_k)$代表主题$z_k$的重要性，且总是降序排列。

模型X.X的参数估计归结为：
$$
\sum_z \hat{p}(w_j|z)\hat{p}(z)\hat{p}(d_i|z)\approx \hat{p}(w_j|d_i)\sim N(w_j,d_i).
$$

**事实** PLSA(33)等价于NMF标准形式（对角NMTF）：$X=WDH$，其中非负矩阵$X$所有元素求和为1，$D$是对角矩阵且对角线上元素求和为1。
*证明* 和事实x.x的证明是类似的。把(X.X)写成矩阵分解的形式：$X\approx WDH$，其中
$$
X_{ij}=\hat{p}(w_j|d_i),W_{ik}=\hat{p}(d_i|z_k),H_{zj}=\hat{p}(w_j|z_k),D_{kk}=\hat{p}(z_k)
$$

*注* 两个模型中的$W$互为转置矩阵。当然，你可以把矩阵形式(X.X)写成$X=W^TDH$，其中$W$按(X.X)中定义。

"**共现**"是一种朴素但具有普遍意义的机器学习思想，被广泛应用于文本处理中。PLSA共现模型只是它的一种特殊形式，也是最简单的一种形式：**文档级共现**。可以设想存在比文档级共现更精确的形式，如**上下文级共现**，处理词和上下文的联合概率$p(w,c)$，其中$c$是$w$的上下文，一般是一个出现在$w$附近的词序列。

生成模型被想象成一个文档的书写过程。共现模型看上去没有生成模型那样的直观解释，它更像是在同时写多个文档：先选择一个主题，然后根据一定概率从词典中选择一个词，再以一定概率写入某个文档，如此反复。生成模型不依赖文档边缘分布$p(d)$，而在共现模型中，$p(d)$代表任意词写入文档$d$的概率。

严谨的读者一定发现，pLSA模型（以生成形式为例）似乎不满足无监督学习的严格定义。它打破了我们习以为常的“输入-输出范式”。即使把$d$看作输入，把$z$看作输出，模型也没有包含判别分布$p(z|d)$明确的表达式。此外，作为可观测变量，词$w$的身份也不明确。这一点会在X.X节得到澄清。

无论共现模型还是生成模型的训练最终只依赖共现矩阵。因此可以效仿朴素Bayes分类器，定义下述模型。

**定义 pLSA多项式模型**
pLSA多项式模型的输入$x$被规定为$x^{(j)}=N(w_j,d),w_j\in \mathcal{W}$，且服从多项式分布:
$$
x\sim M(p(w_j|d), L)\\
p(w_j|d)=\sum_z p(w_j|z)p(z|d)
$$
其中$L$是固定的文档长度，其余符号同定义X.X。

上述定义对应于生成模型。如果$L$是未知的参数，其估计就是文档长度，那么这个定义将对应于共现模型。

最后强调一下，pLSA的样本空间是$\mathcal{W}\times  \mathcal{D}\times \mathcal{Z}$，而观测到的词-文档样本应该是一个二元组集合$\{(d,w)\}$。具体地，如果文档$d_i$中观察到词序列$w_{il},l=1,\cdots,L_i$，那么最终观测到的样本应该是(如表x.x)
$$\{(d_i,w_{il}),i=1,\cdots,N,l=1,\cdots,L_i\}
$$

根据模型的样本形式，pLSA的样本空间也可以是$\mathcal{W}^*\times  \mathcal{D}\times \mathcal{Z}^*$。最终观测到的样本应该是(如表x.x)
$$
\{(d_i,\{w_{il},l=1,\cdots,L_i\}),i=1,\cdots,N\}
$$

<center>$d$列可以是文档名或也可以是某种标识符；$w$列是文档中的词，可以遍历所有文档中词也可以从文档中随机抽词。</center>

|$d$|$w$|$z$|
|---|---|---|
|The Elements of Statistical Learning|Statistical|？|
|The Elements of Statistical Learning|learning|？|
|...|...|？|
|Deep Learning|Inventors|？|
|Deep Learning|have|？|
|...|...|？|

<center>$d$同上；$w^*$列是文档中的词序列（因为独立性假设，顺序并不重要）</center>

|$d$|$w^*$|$z^*$|
|---|---|---|
|The Elements of Statistical Learning|Statistical, learning, ...|？|
|Deep Learning|Inventors, have, ...|？|


#### PLSA的EM算法

PLSA确实可以用NMF的算法来求解，不过它默认采用 EM 算法求解。如果直接用NMF来解，并在最后对$W,H$进行/列归一化处理，那么结果可能会和EM 算法不一样。

写出pLSA共现模型的对数似然：
$$
l(p)=\sum_{ij}N(w_j,d_i)\log p(d_i,w_j)\\
=\sum_{ij}N(w_j,d_i) \log \sum_z(p(w_j|z)p(z)p(d_i|z))
$$
其中$p$概括地表示所有参数。直接最大化该似然函数是困难的。下面写出近似求解共现模型的EM算法（算法的具体构造见X.X节）。

**PLSA共现模型EM算法**

输入：共现数据$X=\{N(w_j,d_i)\}$（或用共现概率$p(w_j,d_i)$代替）

返回：参数$p(w_j|z),p(d_i|z),p(z)$

1. 初始化$p(w_j|z),p(d_i|z),p(z)$；
2. 迭代执行，直到收敛
  - E步：$p(z|w_j,d_i)\sim p(w_j|z)p(z)p(d_i|z)$
  - M步：
    $$
    p(w_j|z) \sim \sum_i N(w_j,d_i)p(z|w_j,d_i),\\
    p(d_i|z) \sim \sum_j N(w_j,d_i)p(z|w_j,d_i), \\
    p(z) \sim  \sum_{ij} N(w_j,d_i)p(z|w_j,d_i).
    $$

写出更加简洁的矩阵形式（矩阵意义见上文）：
$$
H\leftarrow [H\circ DW^T(X\oslash WDH)]_r,\\
W\leftarrow [W\circ (X\oslash WDH) DH^T]_c,\\
D\leftarrow [D\circ (W^T (X\oslash WDH) H^T)]_d,
$$
其中$[X]_r,[X]_c$分别表示矩阵$X$行、列归一化，$[X]_d$表示对角矩阵$X$对角线元素归一化。从表达式来看，(34)和(33)是及其相似的，但是计算结果确实会不一致。可见迭代对归一化操作是敏感的。

<!-- 
*注* (34) 中的三个矩阵是并行计算的，而(33)是串行计算的；两者还是有微弱的不同。
-->

生成模型迭代公式更简洁一点。根据概率表达式，只需令（X.X）中的$D$为单位矩阵，便得pLSA生成模型的矩阵迭代形式：
$$
H\leftarrow [H\circ W^T(N\oslash WH)]_r,\\
W\leftarrow [W\circ (N\oslash WH) H^T]_r=[W\circ (X\oslash WH) H^T]_r,
$$
其中$X_{ij}=p(w_j|d_i), N_{ij}=N(w_j, d_i)$。请留意生成模型和共现模型中$W, H$的异同。此外，pLSA生成模型中$H$的迭代居然依赖共现矩阵$N$，而对应的NMF/SMF算法不依赖。这也暗示EM算法和NMF/SMF算法的结果可能是不一样的。

*注* 理论上$W$的第二个迭代公式不改变$W$是随机矩阵的事实，但是出于存在计算误差的可能，在每次迭代时，这两个矩阵都有必要进行行归一化。

#### PLSA聚类

PLSA的两种模型不仅可以用主题分布编码一个文档，达到降维的目的，甚至可以对文档进行粗略聚类。文档中出现最多的主题被视为该文档的类型，即$\argmax_{z}\{p(z|d_i)\}$。实际上，我们把主题当做文档（模型输入）的类别标签 (模型输出)。有$q$个主题，就有$q$个类别，而$p(z|d_i)$是文档属于类型$z$的概率。

正如分类模型可以转化为回归模型，聚类模型也可以转化为降维模型。NMF/pLSA就是一个很好的例证。 

然而，这个聚类模型的形式和一般的基于输入-输出分布的聚类模型很不同。在pLSA生成模型中，作为判别函数的条件概率$p(z|d)$没有显式表达式，而只有隐式表达(X.X)。据此给出了PLSA另一种等价定义，但其表述和定义X.X有微妙差别。

**定义 PLSA聚类模型**
PLSA的输入-输出分布$p(z|d)$是满足下述方程的解：
$$
p(w_j|d)=\sum_z p(w_j|z)p(z|d)
$$
其中随机矩阵$p(w_j|z_k)$和映射$d\mapsto p(w_j|d)$是未知参数。

设已得到估计$\hat{p}(w_j|z_k), \hat{p}(w_j|d)$，让我们考察模型的预测过程。对训练样本$d_i$，分布估计$\hat{p}(z|d_i)$已做出判别。对不在训练样本中的文档$d$，我们不能显式地计算出来，而是按照算法X.X的步骤计算。

**PLSA聚类/分类预测算法**
1. 估计词分布$\hat{p}(w_j|d)\sim N(w_j,d)$，
2. 求解关于主题分布$\hat{p}(z|d)$的超定方程组
$$
\hat{p}(w_j|d)\approx\sum_z\hat{p}(w_j|z)\hat{p}(z|d)
$$
再由解$\hat{p}(z|d)$对其进行判别。

我们大致说清楚了，pLSA生成模型和标准定义中的统计学习模型的主要差异。
这种隐式设计的坏处是给预测或泛化带来点麻烦，但好处是使得无监督学习不依赖关于输入-输出的参数分布。当然聚类算法通常不必处理训练样本以外的数据。

<!-- *注* []说pLSA不能用来预测是不对的。一个无监督学习模型是否用于对新输入做预测，完全取决于用户的意愿。 -->

*注* 人们称pLSA这样的模型为**方面模型(aspect model)**，其中隐变量$z$也被称为**方面**，而分布$p(z|d)$是$d$的**方面表示**。

最后，对pLSA共现模型的解释是相同的，不再赘述。

#### PLSA分类

上文提到pLSA是一个聚类模型，其中主题$z$充当类别标签。因此，如果我们能观测到作为类别标签/主题$z$，那么自然得到pLSA的有监督版本：

**PLSA-分类器（基于生成模型）** 分类器$p(z|d)$满足方程组：
$$
p(w|d)=\sum_z p(w|z)p(z|d),
$$
其中$p(w|z)$和映射$d\mapsto p(w|d)$是未知参数。注意，$z$是可观测的。

如果要求一个文档只属于某一个类，即$p(z|d)$是指示向量，那么下述事实，
$$
p(w|d)= p(w|z)\iff d:z
$$
这个等式过于苛刻。这也说明pLSA的独立性条件确实很强，一般只是近似成立。

*注* 遵循约定X.X，$d:z$表示样本中文档$d$归属于特定的文档类$z$。

现在利用事实(X.X)，给出参数估计
$$
\hat{p}(w_j|z)\sim \sum_{d:z}N(w_j,d)+\alpha
$$
其中$\alpha\geq 0$为可选的光滑化系数。算法X.X指出，PLSA分类器的响应/判别概率$p(z|d)$的估计，是下述超定线性方程组的解，
$$
\hat{p}(w_j|d)=\sum_z \hat{p}(w_j|z)\hat{p}(z|d)
$$
然后分类器对新文档$d$做出预测：$c(d)=\argmax_{k}\hat{p}(z|d)$。显然，$\hat{p}(w_j|d)$与$\hat{p}(w_j|z)$越接近，$d$越可能属于$z$类。（为什么？）这正是多项式朴素Bayes分类器的原理。当$\hat{p}(w_j|d)$落入某个范围时，pLSA将给出与多项式朴素Bayes分类器一致的结果。现在，我们可以断言：

**事实**
PLSA-分类器（基于生成模型）等价于多项式朴素Bayes分类器。（这也暗示多项式朴素Bayes分类器的无监督版本等价于pLSA。）

### 加权NMF与缺失数据填补

本节特别介绍NMF的一个重要应用，**缺失数据填补**：样本矩阵$X$中存在未观测到的元素，需要用观测到的元素进行推断。

我们先考虑“加权NMF”，即NMF的加权形式。

**定义 加权NMF**
**加权NMF**由下述损失函数刻画：
$$
d_R(X,WH):= \sum_{ij}R_{ij}d(X_{ij},(WH)_{ij}),R_{ij}\geq 0,
$$
其中矩阵$R$不应该出现整行或整列是0的情况。

根据定义，加权NMF依然是一种NMF，只是它的损失函数不是按照(x.x)那样构造的。可以证明，当$d$是散度或 Euclidean 距离时，加权NMF的MU规则分别为（只给出矩阵形式）
$$
H\leftarrow H\circ W^T(X\circ R) \oslash W^T(WH\circ R),\\
W\leftarrow W\circ (X\circ R)H^T \oslash (WH\circ R)H^T,
$$
或
$$
H\leftarrow H\circ W^T(X \oslash WH\circ R)\oslash W^TR,\\
W\leftarrow W\circ (X\oslash WH\circ R)H^T \oslash RH^T.
$$

引入$R$是**缺失矩阵**（另见(X.X)）来描述数据缺失情况：
$$
R_{ij}:=\begin{cases}
1, & X_{ij}\text{未缺失},\\
0, & X_{ij}\text{缺失}.
\end{cases}
$$

$X$的每个**填补**$X'$都是满足下述条件的非负矩阵，
$$
X'_{ij}\begin{cases}
=X_{ij}, & R_{ij}=1,\\
\geq 0, & R_{ij}=0.
\end{cases}
$$

**定义 缺失数据填补模型/算法/任务**
一个用于填补缺失数据的模型/算法$f$，接收$X,R$，返回$X'$，且只能利用$R_{ij}=1$的元素，可记作
$$
X' = f(X\circ R, R)
$$

在对$X$的填补$X'$应用NMF后，有一个对应的损失$L(X')=\min_{W,H}d(X',WH)$。最好的填补应该使该损失最小，即$X'$是如下优化问题的解：
$$
\min_{X',W,H}\sum_{ij}d(X'_{ij},(WH)_{ij})=\min_{W,H} \sum_{R_{ij}=1}d(X_{ij},(WH)_{ij})\\
=\min_{W,H} \sum_{ij}R_{ij}d(X_{ij},(WH)_{ij}),
$$
若解得$W,H$，则有
$$
X'_{ij}=\begin{cases}
X_{ij}, & R_{ij}=1,\\
(WH)_{ij}, & R_{ij}=0.
\end{cases}
$$
因此数据缺失NMF是一种加权NMF，其中权重为缺失矩阵，而加权NMF的迭代算法正是缺失数据填补算法。

从PF的角度讲，更能说明问题。当$R_{ij}=1$时，有$X_{ij}\sim P((WH)_{ij})$，当$R_{ij}$时，未观测到$X_{ij}$的样本。自然得到如下似然函数，
$$
\sum_{R_{ij}=1}\ln P(X_{ij}|(WH)_{ij}).
$$
易证参数$W,H$的MLE就是优化问题(41)的解，其中距离函数$d$选为散度。

显然，任何矩阵分解都可以构造出相应的数据填补方法。请读者考虑如何用x.x节的加权SVD做数据填补。


## 其他话题

本节简单介绍几个关于PCA/ICA/MNF的高级话题。

### 预训练与微调
预训练与微调是当下流行的机器学习方法。预训练与微调是用来配合深度学习或大模型的使用的。这不妨碍我们在统计学习的语境下解释这个方法。

**定义（预处理与预训练）**
利用学习器$f':Z=T(X)\to Y$和变换$T$（如KL变换），可构造新学习器$f=f'T:X\to Y$。这是**预处理**的一般形式，其中$T$称为**预处理器**。预处理器有时也要基于样本进行训练，这个过程或所涉及的方法称为**预训练**。

预训练一般采用无监督学习，即不依赖输出。本书提到的所有降维方法都是这类预训练。但它也不排斥监督学习，如X.X节解提到的LDA降秩法是有监督的。

**定义 微调**
一个带有预处理器$T(x,v)$的学习器可以表示为$f(T(x,v),w)$，其中$v$是预处理器$T$的参数，$w$是学习器$f$的参数。设预训练已给出$v$的估计$\hat{v}$。一个迭代算法叫做**微调**，如果它用以$\hat{v}$作为$v$的初始值（$w$的初始值另外给出）训练整个学习器，即迭代$v$和$w$，且以迭代$w$为主。

*例* 基于PCA预处理的线性模型称为**主成分回归（PCR）**。设$EX=0$，且PCA预处理$Z=XV$，其中$V\in\R^{p\times r}$，则模型为
$$
y|X\sim XV\beta,\beta\in\R^r
$$
$V$和$Z$由PCA算得；模型参数$\beta$的估计归结为线性模型$y\sim Z\beta$的参数估计。之后可通过坐标下降法交替迭代$\beta$和$V$进一步降低损失。这就是微调。


### 框架 PCA
PCA中的基必须满足正交性。这个要求太强，使PCA的应用受到很大的限制。下面引入框架的概念来代替正交基。

**定义 $\R^n$-框架**
向量族$\{v_k\in\R^n,k=1,\cdots,M\}$是框架，当且仅当它是一个完全系，即线性包$\mathrm{span} \{v_k\}=\R^n$（显然$M\geq n$）。若把$v_k$作为列向量水平叠成矩阵$V:\R^{n\times M}$，则该定义等价于，$V$行满秩。

回到KL变换。现在，我们放宽对分解的要求：用框架$\{v_k\}$对随机变量$X$进行分解，即
$$
X\approx\sum_k\alpha_kv_k=ZV^T,
$$
其中框架矩阵$V$仅满足行满秩，$Z$是框架系数$\{\alpha_k\}$构成的随机向量。这个分解可以简单归结为下述优化问题：
$$
\min_{Z,V} \|X-ZV\|
$$
其中矩阵范数也可由用户定制。这个优化问题可称为**框架PCA**或**框架FA**。类似可以构造**框架ICA**。

注意一般情况下，对任何确定的$X$，框架系数$\{\alpha_k\}$都是不唯一的，并且通常是稀疏的（大部分分量小于一个阈值）。稀疏矩阵$Z$的存储要特别处理，不应该按照普通矩阵那样做，否则不能发挥框架分解的优势。

有一种灵活的处理办法：先对样本进行合理划分，再对每一组样本独立执行PCA。具体地说，先对样本$X:N\times p$进行划分：令$\gamma:\{x_1,\cdots,x_N\}\to \{1,\cdots,G\}$是分组映射（其实是一个分类映射），得到$G$组样本，
$$
X_g:=\{\gamma(x_i)=g\}:N_g\times p, g=1,\cdots, G
$$
其中$N_g$是第$g$组样本大小。再对$X_g$的执行成分数为$r_g$的PCA：$X_g\approx Z_gV_g^T$，其中$Z_g:N_g\times r_g,V_g:p\times r_g$。每个划分$\gamma$都会产生一个PCA重构误差。我们希望这个误差越小越好。

**定义 分组PCA**
根据上述讨论，给出优化问题，
$$
\min_{\gamma,\{Z_g\},\{V_g\}}\sum_g\|X_g - Z_gV_g^T\|_F^2\\
=\min_{\gamma,\{V_g\}}\sum_g\|X_g - X_gV_gV_g^T\|_F^2,
$$
其中$V_g\in O(p\times r_g)$。

*注* 分组PCA有$\sum_gr_g$个基向量，但是每个数据点被降到不超过$r_g$的维数。

**事实**
分组PCA是一种硬聚类决策模型。

直接搜索最优分组映射$\gamma$非常intractable。推荐用坐标下降法近似求解优化问题(X.X)。

**算法 分组PCA算法**
- 当$\gamma$固定的情况下，归结为$G$个独立的PCA。
- 当$V_g$定的情况下，计算分离器（分类结果）
$$
  \gamma(x) = \min_{g}\|x - V_gV_g^Tx\|_F,
$$

显然，优化问题(X.X)包含的分解$X_g\approx Z_gV_g^T$不必局限于PCA。此外，因为$r_g$足够小，且$V_g$独立存储，所以用户不用顾虑系数的稀疏性。

上述分析表明，分组PCA是一个把子空间当做聚类中心的聚类模型。不妨称之为“子空间聚类”。结合X.X节，我们可以提出更一般的“子流形聚类”：
$$
\min_{\gamma,\{M_g\}} \sum_g\sum_{\gamma(x_i)=g} d(x_i,M_g)
$$
关于这类模型的深入讨论留给读者。(见习题x.x)

### 鲁棒性

（44）的条件极其微弱。可以追加很多条件，比如限制$Z,V$元素的取值范围。
我们考虑增加正则化条件。
$$
\min \|X-ZV^T\| + \alpha \|Z\|+\beta\|V\|,
$$
其中所有矩阵范数可自定制。如果增加非负性条件，那么就可以包含NMF。

### 非负张量分解（NTF）

基于PF，我们可以构造下“复合模型（composite model）”。这是因为服从Possion分布的随机变量加和依然服从Possion分布。
$$
x_{ij}=\sum_k c^{k}_{ij}, c^{k}_{ij}\sim P(W_{ik}H_{kj}),
$$
其矩阵形式为
$$
X=\sum_k C_k,C_k\sim P(H_{k:}\circ W_{:k}).
$$

也就是说，NMF可以重写为$X\sim\sum_kH_{k:}\circ  W_{:k}$。这进一步暗示我们可以超越矩阵分解的限制。

**定义 张量分解**
对$r$阶随机张量$X\in \R^{p_1\times\cdots\times p_r}$，有分解
$$
X=\sum_k C_k,C_k\sim P(H^{(1)}_{:k}\circ H^{(2)}_{:k}\circ \cdots \circ H^{(r)}_{:k}),
$$
其中$r$个向量的外积的定义是
$$
h^{(1)}\circ\cdots\circ h^{(r)} := \{h^{(1)}_{i_1}\cdots h^{(r)}_{i_r},1\leq i_1\leq p_1,\cdots,1\leq i_r\leq p_r\}:\R^{p_1}\times\cdots\times \R^{p_r}\to\R^{p_1\times \cdots \times p_r}.
$$
记
$$
[[H_1, H_2,\cdots, H_r]]:=\sum_k H^{(1)}_{:k}\circ H^{(2)}_{:k}\circ \cdots \circ H^{(r)}_{:k}.
$$
(45)意味着，对$r$阶张量，有近似分解
$$
X\approx [[H_1, H_2,\cdots, H_r]].
$$

在非负限制下，这就是著名的*非负CP分解 (NCPD)*。它是一种特殊的也是最早出现的非负张量分解。我们把$H_i,i=1,\cdots,r$称为因子矩阵。当$r=2$时，退化为NMF。

从线性空间分解的角度看，NCPD将分解的基限制在矩阵张量积形式：
$$
x\approx [[w;H_{1},\cdots, H_{r-1}]]:= \sum_k w_k H^{(1)}_{:k}\circ \cdots \circ H^{(r-1)}_{:k}.
$$
对基的限制意味着模型包含了某种先验知识。正是这种知识使NCPD不会退化成NMF。要知道，仅仅把$r$阶张量展平为矩阵然后执行NMF，并不能实现NCPD。

## 总结

本章我们重点介绍了最经典的三种降维方法：PCA、ICA、NMF。它们有相似的表达形式、概念和功能，在多个领域有广泛应用。三个模型及其推广形式都把参数估计问题转化成矩阵近似分解。关于矩阵近似分解的几种经典形式见[]。

PCA是最受欢迎的降维方法。我们给出了多个等价定义，加深读者对该模型的认识。对于某些类型的数据，我们应该使用NMF。PLSA就是NMF的一成功的应用。

<center> 目前所学模型的亲缘关系</center>

| 类型（功能）    | 监督学习         | 无监督学习  | 关键分布|
| ----- | ---------- | ---- | --- |
| 回归/降维 | 线性回归       | PCA/ICA  | Gaussian分布|
| 分类/降维 | 非负最小二乘法 | MNF  |Poisson 分布|
| 分类/聚类/降维 | 朴素Bayes分类器 | PLSA | Categorical分布 |
| 分类/聚类 | QDA/LDA        | GMM  |Gaussian分布|

---

*练习*

1. 验证$\|A-AV_rV_r^T\|_F^2=\|A\|_F^2-\|AV_r\|_F^2$，其中$A\in\R^{n\times n},V_r\in O(n\times r)$。
2. 请读者完整写出(x.x)的ALS算法，且能保证$V_r\in O(n\times r)$，并用计算机语言实现。如果要完全给出SVD的具体分解形式，那么有该如何设计算法？
3. 分别在Euclidean距离和散度下，给出求解NMTF，$X\approx WDH$的迭代公式。
4. 请说明K-means聚类是Euclidean距离下NMF的一种约束形式，其中$W$是0-1响应矩阵。（这个事实意味着K-means聚类不仅是GMM的极限形式，也是NMF的特殊形式。但请注意，K-means聚类的聚类数$K=q$不一定小于样本维数$p$。）
5. 模仿PCA/SVD的线性编码器，构造如下“非负线性自编码器”:
$$
\min_{V,H}\|X-XVH\|
$$
其中$V:p\times r,H:r\times p$都是非负矩阵，分别充当编码器和解码器。这样的自编码器构造是否合理？K-means聚类是一种非负线性自编码器吗？
1. 考虑**对偶pLSA（共现形式）**[]：  
   $$
   p(w_j,d_i)=\sum_{kl} p(w_j|y_l)p(y_l, z_k)p(d_i|z_k),
   $$
   其中隐变量$y,z$分别为词概念和文档主题，其余符号同pLSA。和pSLA相比，对偶pSLA区分了词的聚类（概念）和文档的聚类（主题）。试说明，对偶pLSA和NMTF的等价性。
2. 设独立随机变量$X_{ij}\sim NB(\alpha, \frac{(WH)_{ij}}{(WH)_{ij}+\alpha})$，其中$NB(\alpha,p),\alpha\geq 0,0\leq p\leq 1$是负二项式分布，其密度函数为
   $$
   f(x|\alpha,p)=\frac{\Gamma(x+\alpha)}{x!\Gamma(\alpha)}p^y(1-p)^\alpha, x\geq 0.
   $$
  请说明这个参数估计等价于下述损失函数下的NMF
  $$
  d_\alpha(a\|b)=a\log\frac{a}{b}-(a+\alpha)\log\frac{a+\alpha}{b+\alpha}，
  $$
  并构造估计非负矩阵$W,H$的算法——负二次项矩阵分解（NBMF）。
1. 证明NBMF和下述Poisson-gamma混合模型等价。
   $$
   a_{ij}\sim \Gamma(\alpha,\alpha)\\
   X_{ij}|a_{ij}\sim P(a_{ij}(WH)_{ij})
   $$
2. 设$X,Y$是两个有限集，说$R$是其上的关系如果它是$X\times Y$的子集，即$R\subset X\times Y$。称$A\times B$为特征关系，其中$A\subset X,B\subset Y$。$r$个特征关系的并$\bigcup_{k=1}^rA_k\times B_k$称为$r$-简单关系。给定任何关系$R\subset X\times Y$，如何找到与之最近似的$r$-简单关系，其中$r<|X|,|Y|$？
3. X.X节和X.X节介绍的是基于生成模型的PLSA聚类器和分类器。请读者思考如何定义基于共现模型的PLSA聚类器和分类器。
4.  设若干（不同维度）流形的并依然是流形，即形如$\bigcup_gM_g$。请说明子流形聚类和子流形分析是等价的。
5.  请比较一下相继优化和决策树的递归优化。


---

*参考文献*

"""
))
