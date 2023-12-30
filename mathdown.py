#!/usr/bin/env python

import pyparsing as pp
ppu = pp.unicode
pp.ParserElement.setDefaultWhitespaceChars(" ")

NL = pp.LineEnd()
single_newline = NL + ~pp.FollowedBy(NL)
strict_single_newline = NL + ~pp.FollowedBy(NL | pp.one_of(("-", "*", "+")) | (pp.common.integer+pp.Literal('.').leave_whitespace()))
two_or_more_newlines = NL[2, ...].set_parse_action(lambda t: '\n\n')
one_or_more_newlines = NL[1, ...].set_parse_action(lambda t: '\n\n' if len(t)>=2 else '\n')
zero_or_more_newlines = NL[0, ...]
two_newlines = NL + NL

KEYWORDS = ("定义", "事实", "算法", "注", "例", "定理")


def indent(x):
    if isinstance(x, str):
        x = x.strip().split('\n')
    return '\n'.join(['  '+i.strip() for i in x])


def _join(x):
    if len(x) <= 1:
        return x
    else:
        res = x[0]
        for xi, xj in zip(x[1:], x[:-1]):
            if xj != '\n':
                xi = ' ' + xi
            res += xi
        return res

class LatexAction:

    def __str__(self):
        return self.to_latex()


class LatexEnv(LatexAction):

    def __init__(self, env='align', name=None):
        self.env = env
        self.name = name
        
    def __call__(self, t):
        name = self.name or t.get('name', None)
        content = t['body']
        if name is None:
            return f"""\\begin{{{self.env}}}
{indent(content)}
\\end{{{self.env}}}"""
        else:
            return f"""\\begin{{{self.env}}}[{name}]
{indent(content)}
\\end{{{self.env}}}"""


class LatexCommand(LatexAction):

    def __init__(self, name):
        self.name = name

    def __call__(self, t):
        return f"\\{self.name}{{{t['body']}}}"


class Textbf(LatexCommand):

    def __init__(self):
        super().__init__('textbf')

    def __call__(self, t):
        if any(map(t['body'].strip().startswith, KEYWORDS)):
            return t
        return f"\\textbf{{{t['body']}}}"


class Textit(LatexCommand):

    def __init__(self):
        super().__init__('textit')

    def __call__(self, t):
        if any(map(t['body'].strip().startswith, KEYWORDS)):
            return t
        return f"\\textit{{{t['body']}}}"


class Section(LatexCommand):

    def __init__(self):
        super().__init__('section')

    def __call__(self, t):
        n_sub = len(t['sharp']) - 2
        return f"\\{'sub' * n_sub +self.name}{{{t['body']}}}\n"

class Chapter(LatexCommand):

    def __init__(self):
        super().__init__('chapter')

    def __call__(self, t):
        return f"\\{self.name}{{{t['body']}}}\n"

class LatexList(LatexAction):

    def __init__(self, env='itemize', form=None):
        self.env = env
        self.form = form
        
    def __call__(self, t):
        form = self.form or t.get('form', None)
        content = t
        if form is None:
            return f"""\\begin{{{self.env}}}
{indent(content)}
\\end{{{self.env}}}"""
        else:
            return f"""\\begin{{{self.env}}}[{form}]
{indent(content)}
\\end{{{self.env}}}"""


language = True
if language is None:
    marks = r'|/()[]{}\,.;:!?"='
    chars = pp.alphanums + marks
else:
    marks = r'|/()[]{}\,.;:!?"=' + '“”，、。；：！？（）'
    chars = pp.alphanums + marks + ppu.Chinese.alphas

words = pp.Word(chars, chars + '- ').set_parse_action(lambda t: t[0].strip())
digit = pp.Word(pp.nums)

div = pp.Forward()
equation = pp.QuotedString('$$', multiline=True)('body').set_parse_action(LatexEnv('equation'))
equation_where = equation + '其中' + div
inline = pp.QuotedString('$', unquote_results=False)
bold = pp.QuotedString('**', multiline=False)('body').set_parse_action(Textbf())
italy = pp.QuotedString('*', multiline=False)('body').set_parse_action(Textit())
quote = pp.QuotedString('`', multiline=False).set_parse_action(lambda t: f"\\verb|{t[0]}|")
center = pp.QuotedString('<center>', end_quote_char='</center>').set_parse_action(lambda t: f"Tab/Fig{t[0]}")
code = pp.QuotedString('```', multiline=True)('body').set_parse_action(LatexEnv('coding'))

image = '!' + pp.QuotedString('[', end_quote_char=']') + pp.QuotedString('(', end_quote_char=')')

span = pp.Combine(inline+pp.Optional('-'+words)) | bold | italy | quote | words
title = pp.Combine(pp.OneOrMore(pp.Combine(inline+pp.Optional('-'+words)) | words))
div <<= pp.OneOrMore(equation | span | strict_single_newline).set_parse_action(_join)
ulist = pp.DelimitedList((pp.Suppress('- ') + div).set_parse_action(lambda t: '\\item '+t[0]), one_or_more_newlines).set_parse_action(LatexList())
olist = pp.DelimitedList((pp.Suppress(pp.common.integer+pp.Literal('. ').leave_whitespace()) + div).set_parse_action(lambda t: '\\item '+t[0]), one_or_more_newlines).set_parse_action(LatexList(env='enumerate'))
paragraph = code | center | pp.DelimitedList(ulist | olist | div, single_newline)

remark_key = pp.Suppress('*注*')
example_key = pp.Suppress('*例*')
proof_key = pp.Suppress('*证明*')
def_key = pp.Suppress('**定义') + pp.Optional(pp.Suppress(pp.nums)) + pp.Optional((pp.QuotedString('[', end_quote_char=']') | pp.QuotedString('（', end_quote_char='）') | words)('name')) + pp.Suppress('**')
fact_key = pp.Suppress('**事实') + pp.Optional((pp.QuotedString('（', end_quote_char='）') | words)('name')) + pp.Suppress('**')
algo_key = pp.Suppress('**算法') + pp.Optional(words('name')) + pp.Suppress('**')
thm_key = pp.Suppress('**定理') + pp.Optional((pp.QuotedString('[', end_quote_char=']') | pp.QuotedString('（', end_quote_char='）') | words)('name')) + pp.Suppress('**')

remark = remark_key + paragraph('body')
remark.set_parse_action(LatexEnv('remark'))
remarks = pp.DelimitedList(remark, one_or_more_newlines)

example = example_key + paragraph('body')
example.set_parse_action(LatexEnv('example'))

fact = fact_key  + zero_or_more_newlines + paragraph('body')
fact.set_parse_action(LatexEnv('fact'))

definition = def_key + zero_or_more_newlines + paragraph('body')
definition.set_parse_action(LatexEnv('definition'))

algorithm = algo_key + zero_or_more_newlines + paragraph('body')
algorithm.set_parse_action(LatexEnv('algorithm'))

theorem = thm_key + zero_or_more_newlines + paragraph('body')
theorem.set_parse_action(LatexEnv('theorem'))

proof = proof_key + zero_or_more_newlines + paragraph('body')
proof.set_parse_action(LatexEnv('proof'))

section_title = pp.Literal('##')('sharp') + title('body') + pp.FollowedBy(NL)
section_title.set_parse_action(LatexCommand('section'))

chapter_title = pp.Literal('#')('sharp') + title('body') + pp.FollowedBy(NL)
chapter_title.set_parse_action(LatexCommand('chapter'))

subsection_title = '###' + title('body') + pp.FollowedBy(NL)
subsection_title.set_parse_action(LatexCommand('subsection'))
subsubsection_title = '####' + title('body') + pp.FollowedBy(NL)
subsubsection_title.set_parse_action(LatexCommand('subsubsection'))

comment = pp.QuotedString('<!--', end_quote_char='-->', multiline=True)('body').set_parse_action(LatexCommand('comment'))

block = (definition | fact | pp.Combine(theorem + pp.Optional(one_or_more_newlines + proof)) | algorithm | remarks | example) + pp.FollowedBy(two_or_more_newlines)

text = pp.DelimitedList(comment | block | paragraph, one_or_more_newlines).set_parse_action('\n\n'.join)

subsubsection = subsubsection_title + one_or_more_newlines + text
subsection = subsection_title + one_or_more_newlines + pp.Optional(text) + pp.Optional(one_or_more_newlines  + pp.DelimitedList(subsubsection, one_or_more_newlines))
section = section_title + one_or_more_newlines + pp.Optional(text) + pp.Optional(one_or_more_newlines + pp.DelimitedList(subsection, one_or_more_newlines))
chapter = zero_or_more_newlines + chapter_title + one_or_more_newlines + text + one_or_more_newlines + pp.DelimitedList(section, two_or_more_newlines)

if __name__ == '__main__':

    print(chapter.parse_string(r"""# 线性回归

[TOC]

线性回归是统计学习最基础的模型。在机器学习理论诞生以前，人们就把它作为统计学基本问题做了全面深入的研究。及时在AI时代，线性回归依然普遍应用于自然科学和社会科学的实证研究中。

简单回顾一下作为监督学习的回归模型：
$$
P(Y|X,\beta,\sigma^2)\sim N(f(X,\beta),\sigma^2)
$$
或
$$
Y=f(X,\beta)+\epsilon, \epsilon\sim N(0,\sigma^2)
$$
不难证明，模型参数$\beta$的MLE就是下述优化问题的解：
$$
\min_{\beta} \sum_i|y_i-f(x_i,\beta)|^2
$$
其中$\{(x_i,y_i)\}$是样本。$\sigma^2$的MLE正好是均方误差，即(1)目标函数值除以样本数。此外，（1）和最小化下述风险函数是等价的：
$$
\min_\beta J(\beta)=E|Y- f(X,\beta)|^2\propto \sum_i|y_i-f(x_i,\beta)|^2
$$

(1)或(2)称为最小二乘法，其解为$\beta$的最小二乘估计。这是最常见的机器学习模型/方法，或者说机器学习就诞生于这个公式。线性回归研究最简单的情形：映射$f(x,\beta)$是线性的。

## 线性回归模型

从现在开始，我们将正式讨论具体的统计学习模型。本节从一个向导级模型开始，逐步引导读者进入统计学习的殿堂。

### 1维线性回归

回忆一下高中时代的1维线性回归：
$$
y=ax+b,x\in\R
$$
其中$y$和$x$都是1维随机变量（比如身高和体重，收入和智商），而$a, b\in\R$是两个未知参数，分别称为**斜率**和**截距**。这个关系式反映了人们对两个变量存在线性关系的基本认识。

高中生会直接用 **普通最小二乘法（OLS）** 求解（3）:
$$
\min_{a,b} \sum_i(y_i-ax_i-b)^2,
$$
其中$\{(x_i,y_i)\}$是一组观测到的数据。(X.X)的解记为$\hat{a},\hat{b}$，并定义预测和预测误差：
$$
\hat{y}=\hat{a}x+\hat{b}, \hat{\epsilon}=\frac{1}{N}\sum_i|y_i-\hat{y}_i|^2
$$
这是一种“直接回归方法”，因为没有明确的统计学解释。对中学生来说，即使同时学习了概率论和线性回归的知识，把两者联系起来也是困难的。然而，在统计学习的视角下，线性回归被严格定义为一种特殊的统计模型，而OLS的解是该模型的参数估计。

#### 实例与代码

从网络上获取了一段时期的金价和银价。现在用1维线性回归发现两者的关系。

```python
from sklearn.linear_model import LinearRegression

data = pd.read_csv('data/gold-silver.csv')
X , y= data[['gold']], data['silver']

lr = LinearRegression()
lr.fit(X,y)

lr.coef_[0]   # 1.612
lr.intercept_ # -0.0003276
lr.score(X,y) # 0.4162
```

![](../src/gold-silver.png)

结果符合常识：通常黄金和白银在股市上有相似的表现，且白银比黄金波动较剧烈。

### 线性回归模型

通常一个输出变量不会只受一个自变量影响。如果存在多个自变量，那么就要用多维线性回归。下面给出线性回归最规范的定义。

**定义1[标准线性模型（样本形式）]**
标准线性模型也称**Gaussian线性模型**的表达式如下，
$$
Y=X\beta+\epsilon,\epsilon\sim N(0,\sigma^2)
$$
等价地，$Y\sim N(X\beta,\sigma^2)$，或更准确地，$Y|\beta,\sigma^2\sim N(X\beta,\sigma^2)$，其中**设计矩阵** $X:N\times p$(被认为是已知的、非随机的)，误差$\epsilon$是$N$维随机变量，$\beta\in\R^p,\sigma^2>0$是未知参数。


*注* 按照符号约定，$\epsilon\sim N(0,\sigma^2)$表示随机向量$\epsilon$各分量独立服从$N(0,\sigma^2)$。

定义X.X是从“样本”的角度建立线性回归模型。我们也可以从“总体”的角度建立线性回归模型。

**定义2[标准线性模型（总体形式）]**
标准线性模型本质上是一个条件正态分布：
$$
Y=x\cdot \beta+\epsilon=\sum_{j=1}^p\beta_jx_j +\epsilon,\\
\epsilon\sim N(0,\sigma^2),
$$
等价地，
$$
Y|x \sim N(x\cdot \beta,\sigma^2)
$$
其中$x$是$p$维随机向量，而误差$\epsilon$是随机变量，$\beta,\sigma^2$是未知参数。

上述两个定义是等价的，如果样本是独立的。当样本不独立时，需要把(X.X)改造成下述模型。

**定义 广义线性模型（GLM）**
广义线性模型表示为，
$$
Y=X \beta+\epsilon,\epsilon\sim N(0,\Sigma)
$$
其中$\Sigma$是一般的半正定矩阵。

模型（7）不能简单归结为模型（6）。对于其他较为复杂的回归模型，我们一般只考虑样本独立的情形，因而总采用“总体形式”。

线性模型的另一种推广是，不再要求$\epsilon$服从正态分布，而只描述其数字特征。

**定义 Gauss-Markov 线性回归**
线性模型的误差$\epsilon$满足
$$
E(\epsilon)=0,Var(\epsilon)=\sigma^2
$$
类似定义**广义 Gauss-Markov 线性回归**，其中误差$\epsilon$满足$Cov(\epsilon)=\sigma^2\Sigma$。

关于线性回归，人们还提出过多种附加的假设和限制，给出了许多有意思的模型和相关结论。本书总假定$X$是满秩的。关于满秩性，注意下述两个事实。

**事实**
- 若随机变量族$\{X_j,j=1\cdots,p\}$线性相关，则设计矩阵$X=\{X_{ij}\}$几乎必然不满秩（满秩的概率为0）；
- 若$\{X_j\}$线性独立，则当$N$充分大时，$X$几乎必然线性独立。

事实X.X表明，在设计模型时，我们尽可能保证随机变量族$\{X_j,j=1\cdots,p\}$线性独立。

### 线性模型的参数估计

和其他机器学习相比，线性回归有非常完整的参数估计理论。不仅能得到参数的估计值，还能得到参数的分布。这是非常有价值的成果。

#### Gauss-Markov 线性模型的参数估计

Gauss-Markov 线性模型没有给出具体分布。我们不能用MLE，而是根据ERM原则，求解下述优化问题：
$$
\min_\beta \{\sum_i(y_i-x_i\cdot \beta)^2=\|y-X\beta\|_2^2\}
$$
其中$\{(x_i,y_i)\}$是独立样本。我们发现这就是上文的OLS。OLS简单之处在于存在代数解法，见下述事实。

**事实**
OLS (x.x) 的解就是**法方程**
$$
X^TX\beta=X^Ty
$$
的解，称为**最小二乘估计（LSE）**，其中$X$是设计矩阵，$y$是对应的输出样本列向量。

因为本书只考虑$X$满秩的情形，所以有估计$\hat{\beta}=(X^TX)^{-1}X^TY$。记广义逆
$$
X^+:=(X^TX)^{-1}X^T
$$
于是，有$\hat{\beta}=X^+Y$。

不基于具体分布的线性模型的一般形式是: 
$$
\min_\beta E l(Y,x\cdot \beta),
$$
其中$l$是损失函数。对应的经验形式为$\min_\beta \sum_il(y_i-x_i\cdot \beta)$。如果$l$不是平方损失函数$l(y,\hat{y})=|y-\hat{y}|^2$，那么一般就不存在（10）的代数解法。

<!-- 此外，OLS也等价于方差最小化原则：
$$
\min_\beta \{E(Y-x\cdot \beta)^2=Var(Y|\beta)\}
$$ -->

#### Gaussian线性模型及MLE

现在考察Gaussian线性模型（定义X.X）。若观测到独立样本$\{(x_i,y_i)\}$，则有似然（省略次要常数）
$$
l(\beta,\sigma^2)=\sum_i\ln p(y_i|x_i,\beta,\sigma^2)\\
=  -\frac{1}{2\sigma^2}\sum_i|y_i-x_i\cdot\beta|^2 - \frac{N}{2}\ln 2\pi \sigma^2\\
\sim  -\frac{1}{\sigma^2}\sum_i|y_i-x_i\cdot\beta|^2 - N\ln \sigma^2
$$

如果你试图最大化该似然函数，那么会毫不意外地发现$\beta$的MLE就是它的LSE $\hat{\beta}$，并且$\sigma^2$的MLE是误差平方和的平均值
$$
\hat{\sigma}^2=\frac{1}{N}\sum_i(y_i-x_i\cdot \hat{\beta})^2
$$
且对应最大似然值为$-\frac{N}{2}(\ln\hat{\sigma}^2+\ln{2\pi}+1)$。

### 参数的无偏估计（UE）

这一节根据统计学理论，进一步了解线性回归解（参数统计量）的性质。先引入基本统计量:

- $\beta$的MLE/LSE $\hat{\beta}=X^+Y$；
- $Y$的预测$\hat{Y}=X\hat{\beta}$；
- **hat 矩阵**$H:=XX^+$, 则 $\hat{Y}=HY$；
- 误差$\hat{\epsilon}:=\hat{Y}-Y=(H-1)Y$；
- $\sigma^2$估计$\hat{\sigma}^2:=\frac{1}{N-p}|\hat{\epsilon}|^2$，$p<N$；

*注* 所有统计量都是关于$Y$的函数，如$\beta$估计的严格写法是$\hat{\beta}(Y)$，而$X$依然视为非随机变量。

<!-- 事实
$\hat{\beta}=AY$是$\beta$的LUE 当且仅当 $AX=1$ ($A$是 $X$的左逆)。
 -->

下面给出线性回归中非常基本的一些结论。请读者回顾定义X.X的几类统计量。

**定理[Gauss-Markov 定理]**
对Gauss-Markov 线性模型，有
1. $\hat{\beta}$是$\beta$的 LSE(/LUE)，也$\beta$唯一的 BLUE。
2. $\hat{\sigma}^2$是$\sigma^2$的UE。

当$X$不满秩时, $\beta$的LSE是不唯一的，可表示为$\hat{\beta}=(X^TX)^{-}XY$，也就是说，参数分布$N(X\beta,\sigma^2)$不满足可识别性。然而作为$X\beta$的BLUE，$X\hat{\beta}$却还是唯一的。

*注* $A^-$是一种条件很弱的广义逆，任何不可逆矩阵都有无限个这种广义逆。[x]

Gaussian 线性模型比Gauss-Markov 线性模型更为特殊，具有更多的性质。

**定理** 对Gaussian 线性模型有下述事实:
1. $\hat{\beta}$是$\beta$的MLE，也是唯一的MVUE。
2. $\frac{N-p}{N}\hat{\sigma}^2=\frac{|\hat{\epsilon}|^2}{N}$是$\sigma$的MLE，而$\hat{\sigma}^2$是$\sigma^2$唯一的MVUE。
3. $\hat{\beta}\sim N(\beta,\sigma^2(X^TX)^{-1}), \frac{\hat{\sigma}^2}{{\sigma^2}}\sim \chi^2(N-p)$ 且两者独立。

*证明* 依赖 **Lehmann-Scheffe 定理** 和**指数族**的概念。具体证明步骤可参考[]。

定理X.X给出了参数的具体分布。这是非常有价值的，在假设检验中发挥主要作用。

### 线性代数基本事实

**约定** 在**内积空间**$H$中，$G(\{v_i\})$ 为向量集$\{v_i\}$的Gram矩$\{\langle v_i, v_j\rangle\}$；crd 表示一组固定基下的坐标，即
$$
\mathrm{crd}_{\{v_i\}}(x)=a_i \iff x=\sum_ia_iv_i
$$
其中$v_i$是一组基。

存在纯代数的解法可能是线性模型被最早发现和深入讨论的原因。其他模型一般不会有代数解法。下述有关线性代数的定理和知识可以帮助你更好的理解线性模型参数估计的原理。

定义点$x\in\R^n$到线性子空间$M\subset \R^n$的距离为
$$
d(y,M):=\inf_{x\in M}\|x-y\|
$$

**定理（$\R^n$投影定理）**
若$x\in\R^n$，$M\subset \R^n$（子空间），则存在唯一的
$$
y^*=P_My\in M
$$
使得$d(y,M)=\|y-y^*\|=\sqrt{\|y\|^2-\|y^*\|^2}$。其中$P_M$为$M$上的**正交投影算子**。

设矩阵$X\in \R^{p\times n}$的列向量是$\R^n$中的向量族，其张成的子空间$M=\mathrm{span} X\subset \R^n$。若$X$满秩，即$X$（的列向量）是$M$的一组基，则$y^*=X\beta$，即$crd(y^*,X)=\beta =X^+y$。若$X$不满秩，则$\beta = (X^TX)^{-}X^Ty$不唯一。


当（实）内积空间$H$是有限维时，$H$同构于$\R^n$，且投影定理自动成立。当$H$是无限维时，投影定理的推广就不平凡了。

**定理（Hilbert 空间上投影定理）**
设$H$是Hilbert 空间，且$x, M$分别是$H$中的点和闭子空间。则优化问题$d(x,M):=\inf_{z\in M}\|x-z\|$（即计算$x$到$M$的距离）存在唯一解 $$
x^*=P_Mx\in M，d(x,M)=\|x-x^*\|=\sqrt{\|x\|^2-\|x^*\|^2}
$$
其中$P_M$为$M$上的正交投影算子。此外，若$v_i$是$M$的 (Riesz) 基, 则有
$$
\mathrm{crd}_{\{v_i\}}(x^*) = G(\{v_i\})^{-1}\{\langle x,v_i\rangle\}
$$

*注* 线性空间总和一个数域关联在一起。这里仅限于实数域$\R$，但投影定理在复数域$\mathbb{C}$上也成立。

*注* 无限维空间中的基会表现出和有限维空间中的基不一样的性质。本书对之不做深入讨论。

所有实数值随机变量当然构成无限维空间，但是线性回归可以限制在有限维空间$\mathrm{span}\{X_1,\cdots,X_p,Y\}$上讨论。$\beta$估计就是$Y$在子空间$\mathrm{span}\{X_1,\cdots,X_p\}$上的投影的坐标，而$Y$的预测$\hat{Y}$就是该投影。

## 代码

sklearn提供的线性回归类`LinearRegression`实现了多元线性回归。

### scikit-learn/Python
`scikit-learn` 实现了一般线性回归。默认情况下，截距和其他参数分离。设置`fit_intercept=False`，则截距恒为0，此时在数组`X`上增加一列1，一样可以实现截距的估计。（留给读者尝试。）

```python
# X, Y are the input/output data
from sklearn.linear_model import *

lr = LinearRegression()
lr.fit(X, Y)

```

这里的`score`指R方分数 (见X.X节)，越接近1越好。

### statsmodels/Python

`statsmodels`是Python的统计学库。它的野心是代替SPSS等统计学软件。和`scikit-learn`的`LinearRegression`不同，`statsmodels`提供的`OLS`不仅能用来预测，而且可以输出相信的统计报告，包括系数的置信区间和相关的假设检验。下一节将帮助大家解读`statsmodels`的统计报告。

```python
# X, y are the input/output data, where X and y should be a 2d and 1d array respectively
# X <- [X,1], add 1-column to fit intercept
import statsmodels.api as sm
ols = sm.OLS(y, X)
res = ols.fit(X, y)
print(res.summary())
```

### numpy.linalg/Python

```python
import numpy.linalg as LA
# X <- [X,1], add 1-column to fit intercept
weights, residuals, _, _ = LA.lstsq(X, y, rcond=None)
coef_, intercept_ = weights
```

`scipy`提供了和`numpy`一样的函数。`scikit-learn`的线性回归模型完全基于`scipy` 提供的函数。

### R/lm
R 是为统计计算而生的语言。它的Formula特别符合书写习惯。下面是R实现的线性回归程序。

```R
# x, y are the names of variables
data <- read.csv("data.csv")
lm = lm(y~x, data = data)
summary(lm)
```

### Keras/Python

对高维数据，可用`scikit-learn`提供的 `SGDRegressor`代替 `LinearRegression`近似求解，后者实现了SGD算法。另一个选择是用 `Keras`搭建神经网络。代码如下：

```python
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(units=1, activation='linear', input_dim=3))
model.compile(loss='mse', optimizer='adam')
model.fit(X, y, batch_size=16, verbose=0, epochs=20000)

print(model.weights)
```

### 实验

`code/linear-regression.py`

## 线性模型的假设检验

我们已经完成了基于线性回归的统计学习的主要任务。但线性模型还有很多内容值得我们更深入地研究。不满足于参数估计的研究，在本节我们介绍线性回归引出的假设检验问题，如某个自变量，即输入的某个分量，是否对应变量存在显著影响。与此同时，帮助读者看懂线性回归统计报告单。

再次强调一下有用的基本统计量：
  $$
  \hat{\beta}=X^{+}Y,\\
  \hat{Y}=HY=X\hat{\beta},\\
   \hat{\epsilon}:=\hat{Y}-Y=(H-1)Y,\\
  \hat{\sigma}^2:=\frac{1}{N-p}|\hat{\epsilon}|^2,
  $$

进一步引入重要统计量：

- 总体平方和$SST:=\sum_i(Y_i-\bar{Y})^2,\bar{Y}=\frac{\sum_iY_i}{N}$,
- 误差（预测残差）平方和$SSE(/PRESS) :=\sum_i(Y_i-\hat{Y}_i)^2=\|\hat{\epsilon}\|^2$, 
- 回归平方和$SSR:=\sum_i(\hat{Y}_i-\bar{Y})^2$
- 回归均方和$MSR:=\frac{SSR}{p-1}$，其中$p-1$称为**回归自由度**
- 误差均方$MSE:=\frac{SSE}{N-p}$（即$\hat{\sigma}^2$），其中$N-p$称为**残差自由度**
- 总体均方（无偏样本方差）$MST:=\frac{SST}{N-1}$，其中$N-1$称为**总自由度**

显然，$SST=SSE+SSR$；三个自由度满足$(N-1)=(N-p)+(p-1)$。

*注* 自由度(df)和秩的概念相当，总是和矩阵/向量组的秩, 即是向量组张成空间维度或方程解空间维度有关，比如$\mathrm{rank}(H-1)=N-p$。它被用来将有偏估计矫正为无偏估计。

**约定** 始终用$p$表示属性个数(输入维度)，$N$表示样本大小。

### 假设检验

回忆假设检验的原理。拒绝域$W$通常由一个统计量$T(X)$来构造，比如
$$
X\in W\iff T(X)>T_{1-\alpha}
$$
其中$T_{1-\alpha}$是$T(X)$的$1-\alpha$分位数，$\alpha$是显著性水平。显然我们只需计算p值$1-F(T(X))$，其中$F$是$T(X)$的分布。若p值小于$\alpha$，则$X\in W$，并以显著水平$\alpha$拒绝零假设，否则接受零假设。

在线性回归$y\sim \sum_j\beta_jX_j$中，输入变量的分量$X_j$也称为预测因子或预测器。我们自然希望至少一个预测因子$X_j$是有用的，否则模型将失去预测的作用。什么时候预测因子$X_j$是有用的呢？当然是当$\beta_j$不为0的时候。因此，我们应该找到足够的证据证明至少有一个$\beta_j\neq 0$。于是建立下述假设检验。

**检验 1** $H_0: \beta_1=\cdots=\beta_{p}=0$

构造统计量
$$F:=\frac{MSR}{MSE}\sim F(p-1,N-p)
$$
$MSR$代表的是回归模型可解释的信息量；$MSE$代表真实数据中不能被回归模型解释的信息量。于是，统计量$F$反映了回归模型可解释信息量的比率。根据上文所述，如果$F$的p值很小，那么拒绝零假设。这表明不是所有属性都是多余的。否则我们的希望可能就落空了。

只能判断至少一个预测因子是有用的，显然不能满足人们的实际需求。我们下一步就要做更精准的检验：第$j$个特征是多余的，即$\beta_j=0$。

**检验 2** $H_0: \beta_j=0$

构造统计量
$$
t:=\frac{\hat{\beta_j}-\beta_j}{\sqrt{S_{jj}}}\sim t(N-p)
$$
其中$S:=MSE(X^TX)^{-1}$。

若$\sigma^2$是已知的, 则直接用统计量$\hat{\beta}_j\sim N(\beta_j,S_{jj})$，其中$S_{jj}$是矩阵$S=\sigma^2(X^TX)^{-1}$的主对角线元素。

对每个$j$，执行检验2是不能代替检验1的。如果检验1接受了$\beta_1=0$和$\beta_2=0$，那么可以淘汰预测因子$X_1,X_2$其中一个，但没有理由都淘汰。淘汰其中一个，比如$X_1$，之后，还需要再一次在不含$X_1$的线性模型下对另一个$X_2$进行检验。

检验1只能一次性检验所有属性整体的相关性，而检验2只能一次检验一个属性的重要性。两者局限性很明显。如何检验任何一部分属性的相关性呢？我们可以设计如下假设检验。

**检验 3** $H_0:  \beta_i=0,i\in I^c$, 其中$I\subset \{1,\cdots, p\},|I|=r$

并构造统计量
$$
F:=\frac{SSE(I)-SSE}{p-r}/\frac{SSE}{N-p}\sim F(p-r, N-p)
$$

$F$的意义和使用和检验 1是相似的：$F$的分子代表属性集$I$对应的子模型未能解释的信息量，且其值越大，表示子模型解释能力越差，越可能拒绝$H_0$。

检验 3显然包含了检验1和2。它相当于说，属性集$I$对应的子模型$y=X_I\beta_I$是有效的，无需添加更多自变量。还有更一般的形式：$H_0:  A\beta=0$，其中$A$是大小为$r\times p$的矩阵。

### 预测与置信区间

对输入$x_0$，给出预测值$\hat{y}_0=x_0\cdot \hat{\beta}$。现做出假设$H_0$: 输入$x_0$的预测值为$y_0$。令
$$S:=MSE(1+x_0^T(X^TX)^{-1}x_0)$$
并构造统计量（也称**pivotal量**）
$$
\frac{\hat{y}_0-y_0}{\sqrt{S}}\sim t(N-p)
$$
根据相应的p值便可推断假设是否成立。

不过实际上，人们不习惯提出这种假设。统计量(X.X)的构造是用来解答这样的问题：人们不满足于计算出预测值$\hat{y}_0$，而是渴望了解预测值可能的范围，而预测值超出这个范围的可能性很小。对于数值随机变量而言，这个范围通常是一个区间，称为**置信区间**；参数落入置信区间的概率为**置信水平**（“置信区间套住参数的概率”这样表达可能更合适，强调置信区间是一个随机变量）。置信区间由区间上下界决定，它们也是统计量。

根据(X.X)，可得
$$
P(\hat{y}_0- t_{\frac{\alpha}{2}}(N-p)\sqrt{S}\leq y_0\leq \hat{y}_0+ t_{\frac{\alpha}{2}}(N-p)\sqrt{S})\geq \alpha,
$$
其中随机闭区间$[\hat{y}_0- t_{\frac{\alpha}{2}}(N-p)\sqrt{S},\hat{y}_0+ t_{\frac{\alpha}{2}}(N-p)\sqrt{S}]$就是预测值$y_0$的置信水平为$\alpha$的置信区间。习惯性地记为，
$$
y_0\sim \hat{y}_0\pm t_{\frac{\alpha}{2}}(N-p)\sqrt{S}
$$

当样本大小$N$充分大时，有$\frac{y_0-\hat{y}_0}{\sqrt{MSE}}\to N(0,1),N\to\infty$ (依分布收敛)。我们不仅得到了误差$y_0-\hat{y}_0$的渐进分布，而且得到了$y_0$易于计算的置信区间$\hat{y}_0\pm \Phi_{\frac{\alpha}{2}}\sqrt{MSE}$。我们发现，置信区间大小基本上由$MSE$决定。

#### 正态性检验

线性回归自身依赖于很多假设：除了变量之间存在线性关系，还有残差必须服从正态分布。本节简单介绍几个实用的和正态分布相关的假设检验。

$H_0$: $\epsilon \sim N(0,\sigma^2)$，其中$\sigma^2$已知，比如$\epsilon$是线性回归的残差。

有一种比较直观且通用的检验方法是**Q-Q图**。假设样本$x_{i}\sim F,i=1,\cdots,N$，所谓Q-Q图就是点集$\{(q_i,x_{(i)})\}$，其中$x_{(i)}$是次序统计量，$q_i$是样本分布$F$的$\frac{i}{N}$-分位数，即$F(q_i)=\hat{F}(x_{(i)})=\frac{i}{N+1}$，其中$\hat{F}$是样本经验分布。它本质上在比较经验累计分布$\hat{F}(x_i)$和理论分布$F$之间的差异。理想情况下，$q_i=x_{(i)}$，即Q-Q图位于直线$y=x$上；Q-Q图越偏离直线$y=x$，样本越不可能服从$F$。$F(q_i)$的值不一定是$\frac{i}{N}$的形式，而且$F^{-1}(1)$可能是无穷大。

因此，若用Q-Q图检验上述假设，则令$x_i$为线性回归的样本残差$\epsilon_i$，$F$是正态分布（若方差$\sigma^2$未知，则需事先估计）。

Q-Q图不是量化方法，只能帮助人们在直观上初步判断随机变量是否服从某个特定的分布。常用的量化的方法是**Jarque-Bera 检验**和**Liliefore 检验**。Jarque-Bera检验用Jarque-Bera统计量
$$
JB:=\frac{N−p}{6}(S^2+\frac{1}{4}(K−3)^2),
$$
其中$S,K$分别是残差的*偏度*和*峰度*。Liliefore 检验用Kolmogorov-Smirnov统计量，
$$
D:= \max_i |F(x_i)-\hat{F}(x_i)|,
$$
其中$\hat{F}$是样本经验分布，$F$是理论分布。对本检验来说，$F$是残差的理论分布，即正态分布$N(0,\sigma^2)$，其中若方差未知，则需要用估计值$\hat{\sigma}^2$代替。总体上，Jarque-Bera 检验计算简单，但效果不佳；而Liliefore 检验计算复杂，但其结论可靠。

### 属性选择与拟合优度

属性选择具有重要的理论和实践意义。原始的属性集$A=\{X_j\}$中可能混入了次要的属性。这些无关的属性不仅增加了模型的规模而且浪费计算资源，甚至导致过拟合。因此人们希望从属性集中选择一个子集$I=\{X_{j_k}\}$，在不显著增加模型误差的前提下缩小模型规模。这么做可以通常可以克服过拟合，也是Occam原则的体现。

**约定** 每个属性子集$I\subset A$都导出一个独特的线性模型。在这个“子模型”上可以定义一系列统计量，记为$T(I)$，其中$T$是原模型统计量的符号，如$SSR(I)$表示属性子集$I$导出的模型的平方残差和。

#### R方和修正R方
MSE是绝对误差，适合用来比较两个模型拟合的情况和在训练时充当损失函数，但很难反应拟合的真实情况，比如当$y_i$的数值都比较小时，即使令所有$\hat{y}_i=0$，MSE也不会很大，而改变数据的单位就会影响MSE的值。因此，在判断拟合的情况时，采用相对误差$\frac{SSE}{SST}$更合适。

R方统计量$R^2$（也称$R^2$分数或决定性系数）是拟合优度的常用指标，定义如下
$$
R^2:=\frac{SSR}{SST}=1-\frac{SSE}{SST}
$$
根据定义，$R^2$越接近1，模型误差越小，拟合越好。$SSR$代表已解释部分的方差（$SSE$自然代表未解释部分的方差），而方差衡量信息量，因此$R^2$表示已解释的“信息量”占整个样本信息量的比重，以此反应拟合的程度。

然而$R^2$不能评估过拟合程度。为此人们构造其修正版本$R_{\mathrm{adj}}^2$来应对过拟合问题：
$$R^2_{\mathrm{adj}}:=1-\frac{MSE}{MST}=1-\frac{N-1}{N-p}(1-R^2)
$$
$R_{\mathrm{adj}}^2$是属性个数$p$的反比例函数。当属性过多时，自然容易出现过拟合，此时$R_{\mathrm{adj}}^2$也较小。因此，读取表X.X中`Adj. R-squared`的值，可以大致判断模型的优劣。

$R_{\mathrm{adj}}^2$非常适合进行属性选择。我们在前文讨论了，如何用假设检验判断一个属性（即预测因子）的重要性，然后考虑删除那些不重要的属性。利用$R^2_{\mathrm{adj}}$，我们可以直接求解下述优化问题：
$$
\max_{I\subset A} R_{\mathrm{adj}}^2(I)
$$
其中$R_{\mathrm{adj}}^2(I)$是属性子集$I$对应的模型的修正$R^2$分数。当然，遍历$A$的子集并不是一件轻松的事情。实际计算中，人们采用贪婪策略搜索满意的属性子集。

<!-- 介绍$R^2$的另一个动机是，它是`scikit-learn`提供的回归模型默认的误差函数。（见第X.X节） -->

#### Akaike 信息准则（AIC）

著名的Akaike 信息准则（AIC）一定程度上也能控制属性个数（模型复杂度），其定义如下。
$$
AIC := 2p-2 l(\hat{\theta})
$$
其中$p$是属性个数，$l(\hat{\theta})$是对数似然值。

另一个与之紧密相关的是Bayesian信息准则（BIC）：
$$
AIC := p\ln N -2 l(\hat{\theta})
$$
其中$N$是样本大小。

<!--
设有属性子集$I$，$|I|=p$。定义Mallows的 $C_p$统计量：
$$
C_p:=\frac{SSE(I)}{MSE}-(N-2p)
$$
其中$N$是样本大小，$k$是属性个数（$I$元素个数），$SSE(I)$是只用属性子集$I$时模型的残差平方和，$MSE$是考虑全部属性的均方误差。最小化$C_p$，得到最优模型。

在线性回归中，Mallows的 $C_p$与AIC非常相似：$AIC \approx N\ln SSE(I)+2p$。
-->

#### $PRESS$统计量

除此之外，起类似作用的还要Allen的 $PRESS$统计量 (留一法误差)
$$
PRESS:=\sum(y_i-\hat{y}_{i,-i})^2\\
P^2:=\frac{PRESS}{SST}
$$
其中$\hat{y}_{i,-i}$是删除第$i$个样本$(x_i,y_i)$后对$x_i$的预测值（$y_i$的估计值）。$PRESS(/P^2)$是$SSE(/R^2)$的防过拟合版本。

### 读懂statsmodels统计报告

`statsmodels`是Python的有关统计学计算的第三方库。下面是`statsmodels`打印的统计报告。除了参数估计（`coef`列），报告包含了一些重要的统计量，让用户对模型性能有全面的了解。

```yaml
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                 silver   R-squared:                       0.416
Model:                            OLS   Adj. R-squared:                  0.384
Method:                 Least Squares   F-statistic:                     12.83
Date:                Mon, 27 Sep 2021   Prob (F-statistic):            0.00213
Time:                        11:43:45   Log-Likelihood:                 64.845
No. Observations:                  20   AIC:                            -125.7
Df Residuals:                      18   BIC:                            -123.7
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const         -0.0003      0.002     -0.145      0.887      -0.005       0.004
gold           1.6125      0.450      3.582      0.002       0.667       2.558
==============================================================================
Omnibus:                        1.272   Durbin-Watson:                   1.919
Prob(Omnibus):                  0.529   Jarque-Bera (JB):                0.787
Skew:                          -0.479   Prob(JB):                        0.675
Kurtosis:                       2.838   Cond. No.                         202.
==============================================================================
```

下表罗列了部分重要的统计量，并给出解释。

<center>`statsmodels`统计报告中的统计量（Prob指统计量p值，表中未列出）</center>

| 统计量代码                | 含义                 | 备注                     |
| -------------------- | ------------------ | ---------------------- |
| `R-squared`          | $R^2$分数            | 越接近1误差越小               |
| `Adj. R-squared`          | 修正$R^2$分数          | 越接近1误差越小               |
| `F-statistic`        | $F$统计量             | 检验模型相关性                |
|`Log-Likelihood`|最大对数似然值|越大越好|
|`AIC`|Akaike信息准则|越小越好|
|`BIC`|Bayesian信息准则|越小越好|
| `t`                  | $t$统计量             | 检验单个因子相关性              |
| `Durbin-Watson`      | Durbin-Watson相关性检验 | 接近2，认为残差是独立的           |
| `Jarque-Bera`        | Jarque-Bera统计量     | 越接近0，残差越可能服从正态分布       |
| `Kurtosis`           | 样本残差峰度             | 越偏离3，残差越偏离正态分布         |
| `Skew`               | 样本残差偏度             | 越接近0，残差越偏离正态分布         |
|`Omnibus`|Omnibus统计量|检验$\beta_j$都相等|

根据统计报告的解释，预测因子`const`和`gold`系数的估计值分别为-0.0003和1.6125，置信水平为0.95的置信区间分别为[-0.005, 0.004]和[0.667, 2.558]。这也是报告最重要的内容。由此给出如下回归模型，
$$
y\sim \beta_0+\beta_1 x, \beta_0=-0.0003,\beta_1=1.6125.
$$

根据t-检验（p值在`p>|t|`列），可接受零假设$H_0:\beta_0=0$，但拒绝零假设$H_0:\beta_1=0$，当然也拒绝$H_0:\beta_0=\beta_1=0$。

最后，可以接受残差服从独立正态分布的假设。

更多细节见[ordinary least squares](http://www.statsmodels.org/stable/regression.html)。


关于线性回归的统计分析还有很多内容。这里提到的都是比较常见的内容，主要是为了让读者看懂线性回归统计报告。不难发现，从假设检验到属性选择，统计量的分布是最关键的。本书不是统计学专业书籍，不会给出这些分布的推导过程。欲深入学习，请参考。

## Bayes 线性模型

Bayes方法是一种通用统计学方法。Bayes线性模型是Bayes方法在线性模型中的应用，和其他Bayes模型相比有非常独特的形式。

### 回忆 Bayes公式

考虑监督学习模型/条件分布$p(y|x,\theta)$。把它看成参数$\theta$下关于$y$的分布。给出相应的“条件Bayes公式”：
$$
p(\theta|x,y)=\frac{p(y|x,\theta)p(\theta|x)}{\int_\theta p(y|x,\theta)p(\theta|x)d\theta}
$$
合理假定$\theta\perp\!\!\!\perp x$，得到
$$
p(\theta|x,y)=\frac{p(y|x,\theta)p(\theta)}{\int_\theta p(y|x,\theta)p(\theta)d\theta}
$$
监督学习的MAPE是下述优化问题的解，
$$
\max_\theta \sum_i\log p(\theta|x_i,y_i)
$$
其中$\{(x_i,y_i)\}$是独立样本。

*注* 监督学习模型中，$x$被认为是非随机的，因此简写成$p(y|\theta)$也不会影响理论分析，不过还是建议写成$p(y|x,\theta)$。


### Bayesian 线性回归

回顾线性回归的分布, 
$$
Y|x,\theta\sim N(x\beta,\sigma^2),\theta=(\beta,\sigma^2),\beta\perp\sigma^2
$$

根据$\theta$的分布，Bayesian线性回归可有如下几种类型。

#### 类型1

假设线性模型的系数$\beta$有先验分布$\beta\sim N(\mu_0,\Sigma_0)$，其中$\mu_0,\Sigma_0$已知。(一般设$\mu_0=0$)

概率图：`beta -> y <- (sigma)`

出于方便，记$b=X^Ty,G=X^TX$。我们有后验分布
$$
p(\beta|y,\sigma^2)=\cdots
$$
其中`...`留给读者填写。

现在假设 $\sigma^2$ 是已知的。通过简单的运算，得到后验分布 $\beta|y\sim N(\mu,\Sigma)$，其中
$$
\mu=\Sigma(\sigma^{-2}b+\Sigma^{-1}_0\mu_0)=(\sigma^2+\Sigma_0 G)^{-1}(\Sigma _0b+\sigma^2\mu_0)\\
\Sigma=(\Sigma^{-1}_0+\sigma^{-2} G)^{-1}=\sigma^2(\sigma^2+\Sigma_0 G)^{-1}\Sigma_0\\
$$

设$\mu_0=0$, 则$\mu=(\sigma^{2}\Sigma^{-1}_0+G)^{-1}b$。若其中 $\Sigma_0$ 充分大（范数或谱半径充分大），则$\mu\approx G^{-1}b$，即得$\beta$的LSE。若$\Sigma_0^{-1}=\lambda$，则$\mu=(\sigma^{2}\lambda+G)^{-1}b$。这验证了，Bayes线性回归等价于Ridge回归，其中正则项系数相当于线性回归系数$\beta$的精确度。（见习题）


现在处理$\sigma^2$未知的情形。在该情形中，后验分布没有闭合形式，只能通过迭代近似求解。

**算法**
输入数据，设置$\mu_0,\Sigma_0,\sigma^2_0$
输出$\mu,\Sigma,\sigma^2$
1. 初始化 $\sigma^2\leftarrow\sigma^2_0$；
2. 由（14）得 $\mu,\Sigma$；
3. 计算$\sigma^2\leftarrow\frac{1}{N}(\mu^TG\mu-2\mu^Tb+\|y\|^2)=\frac{1}{N}\|X\mu-y\|^2$；
4. 重复 2-3 直到收敛；

输出结果中的$\mu,\sigma^2$分别是模型参数$\beta,\sigma^2$ 的MAPE(近似解)。该算法是一种（分块）坐标下降法，且由于问题的特殊性，单次下降迭代都存在精确解。

*事实*
类型1 Bayes线性回归相当于能自动调整正则化系数的Ridge回归。

#### 类型2 ---相关向量机（RVM）

假设系数$\beta$有先验分布：$\beta_j\sim N(0,\alpha^{-1}_j),j=1,\cdots,p$且独立，其中未知超参数$\alpha_j^{-1}\sim 1$($\R^+$上的平坦分布)。简记为$$
\beta\sim N(0,\alpha^{-1})\\
\alpha^{-1}\sim 1
$$
依然令$\sigma^2\sim 1$。超参数$\alpha$的分量$\alpha_j$代表的是$\beta_j=0$的置信程度。$\alpha_j$越大（$\alpha_j^{-1}$越小），越能接受$\beta_j=0$。

在此假设下的线性回归称为相关向量机。模型的PMG:
$(\alpha)\to \beta\rightarrow y\leftarrow (\sigma^2)$

考虑计算MMLE: 
$$
\max_{\alpha,\sigma^2} \ln p(y|\alpha, \sigma^2)
$$

然后转化为求解关于 ($\alpha,\sigma^2$)的似然方程。

**证据逼近过程 (Evidence approximation process(EAP))**
输入/输出：略

- 初始化: $\alpha_0,\sigma^2_0$

- 迭代:

  1. 用（14）更新 $\mu,\Sigma$ （注意$\Sigma_0=\mathrm{diag}\{\alpha_0^{-1}\}$）;
  2. 令$\gamma_j= 1-\alpha_j\Sigma_{jj}$;
  3. $\alpha_j\leftarrow\gamma_j/\mu_j^2$;
  4. $\sigma^{2}\leftarrow\frac{\|y-X \mu\|^2}{N-\sum_j\gamma_j}$;
  5. 重复1-4直到满足收敛条件;

*注* EAP是一种EM算法, 0相对于E步, 1-3相当于M步。

**事实(Auto Relevance Determination)**

若$\alpha_j\to \infty$，则 $\Sigma(j,:),\Sigma(:,j)\to 0,  \mu_j\to 0$且$\alpha_j\Sigma_{jj}\to 1$。

此时，$\beta_j \overset{p}{\to} 0$，意味着$x_j$作用不大，否则$x_j$是所谓的**相关向量（relevance vectors）**。

#### 类型 3(层次Bayes模型)

假设$\beta\sim N(0,\alpha^{-1})$，其中未知超参数$\alpha\sim \Gamma(a,b)$, $a, b$是已知数值或未知超参数。这是一个真正的层次 Bayes 模型；细节留给读者补充。这里我们首次考虑设计$\sigma^2$的先验分布，$\sigma^2\sim\Gamma(c,d)$。

`scikit-learn`提供的学习器`BayesianRidge`实现了这个模型，其中超参数$a,b,c,d$是事先给定的数值。

<!-- $$
\alpha\to \beta\rightarrow y\leftarrow \sigma^2
$$ -->

<!-- #### `pymc4/python`实现

见 [Bayesian-Linear-Regression-using-PyMC3](https://ostwalprasad.github.io/machine-learning/Bayesian-Linear-Regression-using-PyMC3.html) -->

### 基于Bayes方法的持续学习

Bayes线性模型的完全有参数的分布决定。学习是参数分布到参数分布的映射，而映射本身由数据决定。这样的映射可以复合在一起，相当于学习的持续进行。我们就称之为“持续学习”。在线学习、迁移学习和增量学习都可以称为持续学习，因为它们都实现了下述映射
$$
\mathcal{M}\times D\to \mathcal{M}
$$
其中$\mathcal{M}$都是模型空间，$D$是数据集。这种学习算法可使学习过的模型根据新数据（增量数据）持续地学习。基于Bayes方法的持续学习的原理是用参数后验分布$p(\theta|D)$表征一个模型，并根据Bayes公式对参数后验分布持续更新：
$$
p(\theta|D, D') = \frac{p(D'|\theta)p(\theta|D)}{\int p(D'|\theta)p(\theta|D)d\theta },
$$
其中$D$是旧数据（由其训练得到模型$p(\theta|D)$），$D'$是增量数据。该公式实现模型根性：

$$
p(\theta|D),D'\to p(\theta|D, D')
$$

我们将Bayes方法的持续学习应用于线性回归。一种实现见[incremental-linear](https://github.com/Freakwill/incremental-linear)。下面是算法的大致框架。

*增量线性回归算法 (大致框架)*
输入数据$X,Y$
输出参数估计
1. 设$\beta\sim N(0,\alpha^{-1}), \alpha,\sigma^2\sim 1$;
2. 用EAP估计$\beta$的分布$\beta\sim N(\mu,\Sigma)$;
3. 用类型1的Bayes模型，对增量样本$X',Y'$估计$\beta$;（必要的话，也估计$\sigma^2$）


### 正则化线性回归

Bayes方法可以起到正则化的作用，但我们有更直接的方法：在目标函数上增加正则项。两个最典型的正则化线性回归是ridge回归和lasso回归，它们的表达式分别是：

$$
\min_\beta\sum_i|y_i-\beta \cdot x_i|^2 + \lambda\|\beta\|_2^2 
$$
和
$$
\min_\beta\sum_i|y_i-\beta \cdot x_i|^2 + \lambda\|\beta\|_1 
$$
其中$\|\beta\|_2,\|\beta\|_1$是正则项，$\lambda>0$是正则项系数。

*事实*
Ridge 回归和lasso 回归分别对应于参数$\beta$服从Gaussian先验和Laplacian先验的线性回归。一般而言，正则项会对应于一个参数先验分布。

不难发现，ridge回归有闭合形式的解：
$$
\hat{\beta}^{\mathrm{ridge}}=(X^TX+\lambda)^{-1}X^Ty\\
=(X^TX+\lambda)^{-1}X^TX\hat{\beta}\\
=V(D^2+\lambda)^{-1}D^{2}V^T\hat{\beta},
$$
其中$\hat{\beta}$是参数$\beta$的LSE，$VDV^T$是$X^TX$的特征值分解。但loss回归的解$\hat{\beta}^{\mathrm{loss}}$就没有闭合形式。

当$X$的列向量集是标准正交时候，两种$\beta$的估计和LSE有下述关系
$$
\hat{\beta}^{\mathrm{ridge}}=\frac{\hat{\beta}}{1+\lambda},\\
\hat{\beta}^{\mathrm{loss}}=\mathrm{soft}_\lambda(\hat{\beta}),
$$
其中$\mathrm{soft}_\lambda(x):=\begin{cases}x-\lambda, &x>\lambda, \\0,& |x|\leq \lambda, \\x+\lambda, & x<-\lambda.\end{cases}$。

两种估计的绝对值都严格小于LSE的绝对值，因此正则化方法也称为**收缩方法(Shrinkage Methods)**。$\hat{\beta}^{\mathrm{ridge}}$和$\hat{\beta}^{\mathrm{loss}}$都是有偏估计，但由于估计量的绝对值的收缩，其方差缩小了。

还可从关系式(X.X)看出Lasso起到属性选择的作用，因为当$\hat{\beta}_j$绝对值太小时，$\hat{\beta}^{\mathrm{loss}}_j=0$，对应属性$X_j$被删除。这的确和根据假设检验删除次要属性的作用类似。但和属性选择不同，Lasso 并不单单是删除系数绝对值较小的属性，还会缩小其他属性的系数绝对值。属性选择相当于对$\hat\beta$作用了函数$\mathrm{hard}_\lambda(x):=\begin{cases}x, &|x|>\lambda, \\0,& |x|\leq \lambda.\end{cases}$。

人们将两种正则项结合在一起得到所谓的**弹性网络（elastic-net）**：$\alpha\|\beta\|_2^2+(1-\alpha)\|\beta\|_1,0\leq \alpha\leq 1$。这种构造模型的策略简单有效，在其他地方也能见到。

<!-- *注* 有文献把Ridge正则项写成$\frac{1}{2}\|\beta\|_2^2$；弹性网络正则项写成$\frac{\alpha}{2}\|\beta\|_2^2+(1-\alpha)\|\beta\|_1$。这当然不是必须的。 -->

## 其他话题

最后简单讨论关于线性回归更为高级的话题，包括广义线性回归、线性混合模型和线性模型的迭代算法等。

### 广义线性回归

为广义线性回归(定义X.X)引入假设：$D\epsilon=\sigma^2\Sigma>0$，其中$\Sigma\in\R^{N\times N}$已知。

**事实** 根据上述假设，广义线性回归$y=X\beta +\epsilon$可以转化成普通线性回归：
$$
\tilde{y}=\tilde{X}\beta+\tilde{\epsilon}, D\tilde{\epsilon}=\sigma^2\\
\tilde{X}=\Sigma^{-\frac{1}{2}}X,\tilde{y}=\Sigma^{-\frac{1}{2}}y
$$

利用事实X.X，立刻得到参数的**广义线性最小二乘估计（GLSE）**：
$$
\hat{\beta}_{GLSE}:=(X^T\Sigma^{-1}X)^{-1}X^T\Sigma^{-1}y\\
\hat{\sigma}^2:=\frac{\hat{\epsilon}^T\Sigma^{-1}\hat{\epsilon}}{N-p}
$$

**事实** 
进一步，若$\epsilon \sim N(0,\sigma^2\Sigma)$，则
1. $\hat{\beta}_{GLSE}\sim N(\beta, (X^T\Sigma^{-1}X)^{-1})$是 $\beta$-UE（唯一 MVUE）;
2. $(N-p)\hat{\sigma}^2/\sigma^2\sim \chi^2_{N-p}$是$\sigma^2$-UE（唯一 MVUE）;
3. $\hat{\beta}_{GLSE}\perp \hat{\sigma}^2$;

实际情况中，$\Sigma$通常是未知的。此时估计$\beta$和$\Sigma$都会变得非常困难。人们希望$\beta$的估计具有鲁棒性，即未知的$\Sigma$不会影响$\beta$的估计。下述事实说明在某些特殊条件下，GLSE=LSE。

**事实（GLSE/LSE的鲁棒性）**
$\hat{\beta}_{GLSE}=X^+y$当且仅当 span $X$ (和/或span $X^{\perp}$)是$\Sigma$-不变的，当且仅当 $\Sigma = (X,R)\begin{pmatrix}\Lambda_1,0\\0,\Lambda_2\end{pmatrix}(X,R)^T$, 其中 $\Lambda_1,\Lambda_2$分别是$p$阶和$N-p$阶对称矩阵，$R$是$X$的正交补，即$R^TX=0, R\in\R^{N\times (N-p)}$。


### 线性混合模型

**定义 线性混合模型**
线性混合模型一般形式为，
$$
y=X\beta+U\xi+\epsilon, \xi\sim N(0,\Sigma),\epsilon\sim N(0,\sigma^2)
$$
其中$X:\R^{N\times p},U:\R^{N\times q}$是给定的设计矩阵, $q$维随机变量$\xi$被称为**随机效应**，其协方差矩阵为$\Sigma:\R^{q\times q}$，且$\xi\perp \epsilon$，参数$\Sigma,\sigma^2$均未知。

模型（15）相当于一个部分Bayes线性回归：
$$
y=(X,U)\begin{pmatrix}\beta\\\xi\end{pmatrix}+\epsilon, \xi\sim N(0,\Sigma),\epsilon\sim N(0,\sigma^2)
$$
其中扩展后的系数向量$(\beta;\xi)$中只有部分分量有先验分布。因此，它的参数估计也比较复杂。请参考[]。

线性混合模型的一个特例是**随机截距线性模型**。它由$K$个有公共参数$\beta$的线性回归
$$
y_k=x\cdot \beta+\xi_k+\epsilon_k, k=1,\cdots,K
$$
其中$\xi_k$是1维随机效应。这相当于对$U$进行one-hot 编码。



### 线性回归迭代算法

虽然线性回归有精确的代数解，但是当样本很大时，最关键的矩阵逆运算依然很耗时。我们知道线性回归的参数估计归结为

本节介绍两种基于梯度的迭代算法。

考虑（广义）线性回归$y\sim N(X\beta,\Sigma^{-1})$，其中$\Sigma^{-1}$已知，$\beta$未知。模型参数$\beta$的估计归结为下述优化问题：
$$
\min_\beta \{\|y-X\beta\|^2_{\Sigma^{-1}}\propto \frac{1}{2}\beta^T A\beta-b^T\beta\}
$$
其中$A:=X^T\Sigma^{-1}X,b^T:=y^T\Sigma^{-1}X$。

计算出目标函数$f(\beta)=\frac{1}{2}\beta^T A\beta-b^T\beta$的梯度，$\nabla f(\beta) = A \beta - b$。GD的每一步迭代都是朝着梯度反方向前进：
$$
\beta_{t+1}=\beta_t-\alpha \nabla f(\beta)
$$

GD的好处是，不依赖平方损失函数。比如解基于$p$范数的线性回归：
$$
\min_\beta \sum_i |y_i-x_i\cdot\beta|^p
$$

设GD的每一步迭代都是朝着方向$d$前进，则前进的步长$\alpha$可选择下述优化问题的解：
$$
\min_{\alpha>0} f(\beta+\alpha d)
$$
其中$\beta$是当前可行解，$\beta+\alpha d$代表更新后的解。

**算法 共轭梯度下降法**
1. 初始化参数$\beta_0$. 
2. $k$从0迭代，直到收敛（不会超过$p$步）
   1. 当$k>0$时，计算梯度变化率$\eta_{k} = \|g_{k}\|^2  / \|g_{k-1}\|^2$，否则设$\eta_{0}=0$；（记$g_k:=\nabla f(\beta_k)$）
   2. 计算搜索方向(共轭方向)$d_k = -g_k + \eta_k d_{k-1}$；
   3. 在方向$d_k$上，选择最优步长$\alpha_k=\frac{g_k^Tg_k}{d_k^TAd_k}$；
   4. 更新$\beta_{k+1} = \beta_k + \alpha_k d_k$；



<!-- ## 非线性回归 
当本章开头的$f(x,\beta)$不再是线性函数时，我们得到的就是非线性回归。

这里我们只考察一个问题：如果数据是由非线性模型$y\sim f(x),x\in\R^p$产生的，那么用线性回归拟合数据时会带来多大的误差。

我们明确一下，线性回归必须保留截距，即设计矩阵$X$的最后一列为1。易证
$$
x_0(X^TX)^{-1}X^T\cdot 1 =1
$$

若存在Taylor展开$f(x)=f(x_0)+\nabla f(x_0)(x-x_0)+H+O(\|x-x_0\|^3)$
则
$$
\hat{f}(x_0)=x_0(X^TX)^{-1}X^T y = f(x_0) + x_0 X^+ (X-x_0)(X-x_0) + O()
$$
-->

---

*练习*

1. 如何将机器学习模型$y\sim \lambda e^{\mu x}$转化成线性回归，其中$x,y$是变量，$\lambda, \mu$是未知参数?
2. 基于污染Gaussian分布的线性回归可以处理异常值问题，其中误差项服从污染Gaussian分布$\epsilon \sim p\phi(x;0,\sigma^2)+(1-p)\phi(x;0,\sigma_0^2)$。请给出参数估计公式，并将之应用于有异常点的线性回归问题。($\phi$是Gaussian分布的密度函数)
3. $H$是设计矩阵$X$对应的hat矩阵，证明$rank(H)=p,rank(H-1)=N-p$。假设$X$满秩。
4. 如何检验$H_0:A\beta=0$，其中$A$是一个$q\times p$的矩阵，$q\leq p$？显然这涵盖了正文中三种关于参数的假设检验。
5. 给线性模型$y\sim X\beta$增加正则项$\lambda\|\beta\|_W$得到更一般的ridge 回归模型，其中$W$是正定矩阵。给出这个模型的参数估计，并指出与该模型等价的Bayes回归。
6. 设Bayes线性回归中$\sigma^2\sim HC(0,1)$（见下图），请读者讨论参数估计。
7. 线性混合模型还有另一种形式为，
   $$
   y=X\beta+\sum_kU_k\xi_k+\epsilon, \xi_k\sim N(0,\Sigma_k),\epsilon\sim N(0,\sigma^2)
   $$
   其中$U_k\in\R^{N\times q_k}$，$\xi_k$是$q_k$维随机效应，其协方差矩阵为$\Sigma_k$。请把它转化成 (15)的形式。

![](https://ostwalprasad.github.io/images/p4/bayesian.PNG)
<!-- 8. 用GD求解(x.x) -->

*参考文献*
"""
))

