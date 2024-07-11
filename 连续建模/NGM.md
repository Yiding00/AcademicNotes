# Neural Graphical Model in Continuous-Time: Consistency Guarantees and Algorithms

## 绪论

### 问题引入

本文假设底层结构为ODE。

## 连续时间图建模

`定义 1`（神经动态结构模型（NDSM））。我们称 $\mathbf x = (x_1,\dots,x_d):[0,T]\rightarrow \mathcal X^d$ 遵循神经动态结构模型，如果存在函数 $f_1,\dots,f_d\in\mathcal F$，其中 $f_j: \mathcal X^d \rightarrow \mathbb R$，且满足以下条件：
$$
dx_j(t) = f_j(\mathbf x(t))dt + dw_j(t), \qquad \mathbf x(t_0) = \mathbf x_0, \qquad t\in[0,T]
$$
其中 $\mathcal F$ 定义为具有参数集 $\theta\in\Theta$（定义在有界的实值区间内）的解析前馈神经网络空间，$w_j(t)$ 表示独立生成于过程 $j$ 的标准布朗运动。



`假设 1`（观测过程）。实际中的数据是 $\mathbf x$ 在 $n$ 个时间点 $(t_1,\dots, t_n)$ 的部分观测序列，这些时间点是从具有正强度的时间点过程中采样的，
$$
(\mathbf x_1, \dots,\mathbf x_n) \sim \mathcal N(\mathbf\mu,\Sigma_n),
$$
其依赖结构编码在 $\Sigma_n\in\mathbb R^{n\times n}$ 中。时间上越接近的两个观测值，我们期望它们的相关性越高。我们假设数据已经标准化，即 $\Sigma_n$ 的对角元素等于 1。$\mathbf\mu$ 是均值过程的实例，可以由一个常微分方程系统描述 $d\mathbf x(t) = \mathbf f(\mathbf x(t))dt, \mathbf x(0)=\mathbf x_0, t\in[0,T]$。



`定义 2`（局部独立性）。如果在每个时间点 $t$，给定 $z$ 的过去值直到时间 $t$，对 $\mathbb E(x_t| \mathcal H_t(y,z))$ 的可预测信息与 $x$ 和 $y$ 的过去值直到时间 $t$ 给出的信息相同，那么过程 $x$ 在给定 $z$ 的情况下局部独立于 $y$，其中 $\mathcal H_t(y,z)$ 是由 $y$ 和 $z$ 在时间 $t$ 之前生成的过滤。



`引理 1`（局部独立图的唯一性）。在神经动态结构模型的背景下，如果且仅如果 $x_k$ 出现在 $x_j$ 的微分方程中，即 $||\partial_k f_j||*{L_2} \neq 0$，两个过程在给定其他过程的任意子集的情况下是局部相关的。此外，对于任何使得 $||\partial_k f_j'||*{L_2} = 0$ 的 $\mathbf f'$，存在一个等效的向量场 $\mathbf f$，使得其列向量的欧几里得范数 $||[A_1^j]_{\cdot k}||_2 = 0$。



`定义 3`（局部一致性）。估计量 $\mathbf f_{\theta}=(f_1,\dots,f_d)$ 是局部一致的，如果对于任何 $\delta > 0$，存在 $N_{\delta}$ 和 $T_{\delta}$，使得当 $n > N_\delta$ 和 $T > T_{\delta}$ 时，对于所有 $k, j \in {1,\dots,d}$，若 $x_k$ 对 $x_j$ 局部重要，则 $||\partial_k f_j||*{L_2} \neq 0$\footnote{$\partial_k f_j$ 表示 $f_j$ 关于第 $k$ 个自变量的偏导数，$||\cdot||*{L_2}$ 是函数的 $L_2$ 范数。}；否则，$||\partial_k f_j||_{L_2} = 0$，且概率至少为 $1-\delta$。



`引理 2`（泛化界）。假设 $\Sigma_n$ 可逆，并令 $\alpha = (\alpha_1,\dots,\alpha_n)$，其中 $\alpha_1 > \dots > \alpha_n > 0$ 是其特征值。对于任意 $\delta > 0$，存在 $C_\delta > 0$，使得
$$
|\mathcal R_n(f_{\theta}) - \mathcal R(f_{\theta})| \leq C_\delta \left( \frac{||\alpha||_2}{n} \right) \sqrt{\log\left(\frac{n}{||\alpha||_2}\right)},
$$
且概率至少为 $1-\delta$。



`引理 3`（自适应组Lasso的收敛性）。令 $\tilde\theta_n\in\Theta$ 为带有自适应组Lasso约束的 (\ref{optim}) 的参数解。对于任意 $\delta > 0$，假设 $\lambda_{\text{AGL}}\rightarrow 0$，则存在 $v > 0, C_{\delta} > 0, N_{\delta} > 0$ 和 $T_{\delta}>0$ 使得
$$
\underset{\theta\in\Theta^*}{\text{min}}\hspace{0.3cm}||\tilde\theta_n - \theta|| \leq C_{\delta}\left( \lambda_{\text{AGL}} + \left( \frac{||\alpha||_2}{n} \right) \sqrt{\log\left(\frac{n}{||\alpha||_2}\right)}\right)^{\frac{1}{\nu}},
$$
且概率至少为 $1-\delta$。



`引理 4`（自适应组Lasso的局部一致性）。令 $\gamma > 0$，$\epsilon > 0$，$\nu >0$，$\lambda_{\textrm{AGL}} = \Omega((\dfrac{n}{||\alpha||*2})^{-\gamma / \nu + \epsilon})$，并且 $\lambda*{\textrm{AGL}} = \Omega(\lambda_{\textrm{GL}}^{\gamma + \epsilon})$，则自适应组 Lasso 是局部一致的。



`引理 5`（自适应组Lasso的有限样本局部一致性）。在引理 4 的条件下，假设问题设计中满足最小限制强度假设，对于特定的 $n$ 和 $\alpha$ 值，自适应组Lasso以高概率精确恢复结构 $G$。