---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: '0.8'
    jupytext_version: '1.4.1'
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Combined neural network and adaptive DSP training for long-haul optical communications
```{only} html
[![DOI](https://badgen.net/badge/DOI/10.1109%2FJLT.2021.3111437/blue)](https://ieeexplore.ieee.org/document/9534655)
```
## I. Introduction
Digital back propagation (DBP) is the standard digital signal processing(DSP) algorithm for fiber nonlinear compensation {cite}`ip2008compensation` in long-haul systems for single channel and wavelength division multiplexing (WDM) systems {cite}`leibrich2003efficient, mateo2010efficient`. Over the past decade, different variants of DBP such as filtered DBP (FDBP) {cite}`du2010improved, rafique2011compensation` are proposed to improve their performance/complexity trade-offs. FDBP assumes a parameterized low-pass filter (LPF) for the signal power waveform to obtain an improved phase rotation at the nonlinear step of the DBP and it has been well studied theoretically and experimentally. An extension to FDBP is enhanced DBP (EDBP) where the LPF taps are free optimization variables {cite}`secondini2014enhanced, secondini2016single`. Hager et al. studied through simulations the advantages of separately optimizing each LPF {cite}`hager2020physics` and a single-channel 1-step EDBP was also experimentally demonstrated {cite}`secondini2016single`. For WDM systems, it is recently demonstrated that coupled-channel EDBP has optimal performance by accounting for XPM through simulations {cite}`Civelli:21`. Intra-channel four-wave mixing (IFWM) based-compensation techniques, sometimes referred to as correlated DBP, are based on perturbation theory and have also been studied through simulations {cite}`liang2015correlated, liang2017perturbation`. In recent years, machine learning (ML) techniques are applied to DBP by treating the iterative linear and nonlinear operation of DBP as the linear and nonlinear operations of a deep neural network (DNN). Hager et al. {cite}`hager2018nonlinear` showed how ML-based approaches can reduce complexity for the same performance through simulations. In addition, our previous work experimentally demonstrated such concept in WDM systems and further provided semi-analytical explanations of the optimal linear filter and nonlinear phase rotation configurations {cite}`fan2020advancing` learned by ML. 
```{figure} ./images/fig1.svg
---
height: 250px
name: time-varying-minibatch
---
Time varying effects such as laser phase noise or state of polarization (SOP) rotations (denoted collectively as $\theta(t)$ above) induce different distortions to different batches and complicates the ML training process.
```
When dividing the input signal sequence into batches for ML training as show in {numref}`time-varying-minibatch`, time-varying impairments such as state of polarization (SOP) rotations or laser phase noise will affect each batch differently. This complicates the NN training as it intertwines with adaptive DSP that are used to track and compensate the time-varying impairments. Fundamentally, the complications arise from the fact that 1) virtually all ML training methodologies are based on data in batches as they are often more hardware-efficient, flexible for different optimization strategies while adaptive DSP training in optical communications are symbol-by-symbol based, and 2) when an NN precedes adaptive DSP, the NN output cannot be used to calculate the mean squared error (MSE) for backpropagation of gradients since they are corrupted by time-varying impairments. Consequently, in all aforementioned works, NN parameters learning are separated from time-varying parameters learning/tracking by adaptive DSP techniques by 1) conducting simulation studies where time-varying impairments are absent {cite}`fan2020advancing, hager2018nonlinear, sidelnikov2021advanced`; 2) appending NN blocks after all the standard DSP blocks so that the NN is essentially immune to time-varying impairments; 3) defining the cost function to be the MSE between the NN output and another benchmark algorithm (e.g. DBP) so that time-varying impairments affect both algorithms equally and are largely cancelled out {cite}`secondini2014enhanced, secondini2016single`, and 4) first estimating the time-varying parameters for the received signal sequence by conventional adaptive DSPs, then re-applies the original signal sequence as NN inputs, append the time-varying parameters estimates to the NN outputs to eliminate the time-varying effects so that symbol-by-symbol MSE and standard ML training algorithms can be properly implemented {cite}`ghazisaeidi2020deep, oliari2020revisiting`. Unfortunately, all these schemes either ignore realistic time-varying effects, work only on impractical system setups, prohibit optimal placement of ML blocks in the whole DSP chain, or require impractical training methodologies with excessive memory or time delay in order to decouple the training of NN and adaptive DSP parameters. Consequently, ML has not been properly integrated into other DSP blocks and their potentials are not yet fully realized in long-haul digital coherent communications.
In this paper, we proposed to combine the learning of NN and time-varying parameters of the optical link by transforming the adaptive DSP as an additional *stateful NN* layer appended to the main NN output so that the whole parameter training process will amend itself to standard batch-based backpropagation-like training algorithm in ML. In this case, conventional adaptive filter state updates are expressed through either symbol-by-symbol or block-by-block state evolutions which are calculated during the forward pass of the backpropagation algorithm. The stateful NN layer outputs are used to calculate the MSE and initiate the gradients calculation for the backward pass. Note that stateful NN is not a brand-new concept in ML: in fact, recurrent neural network (RNN) is one example of such stateful models {cite}`lipton2015critical`. At each time step $t$, the input is transformed to an internal state that evolves with time and help determine the present and future outputs. Such transformation is typically a nonlinear mapping with trainable parameters. In contrast, the internal states in our case is exactly the adaptive filter tap values (or *filter states*) at each iteration. While the input is used to update the filter states at each iteration, the optimized filter states are ultimately functions of $t$ and not the input data since the filter states are meant to track the time-varying impairments of the optical link.
With the proposed combined learning methodology, we derived the complete gradients and update procedures of the generalized DBP(GDBP) algorithm where all the filter taps of the linear and nonlinear steps of DBP are optimized with concurrent mitigation of polarization effects, frequency offset, carrier phase tracking and other residual link impairments using additional adaptive DSP. The GDBP demonstrated an average gain of 0.13 and 0.36 over FDBP and DBP in a 7 x 288 Gb/s polarization multiplexed (PM)-16QAM transmission experiments over 1125 km. For complexity-constraint scenarios with short filter taps, we show that GDBP can largely retain its performance and demonstrated a gain of 1 dB over FDBP/EDBP which highlight the importance of optimized linear filter taps. As the most general form with largest number of optimization variables for a given total number of steps, the GDBP is the first experimental demonstration of optimal single-channel DBP based-fiber nonlinearity compensation algorithm.
In addition, we chose JAX{cite}`jax2018github`, a Python library designed for high-performance numerical computing and machine learning research due to its} flexibility and scalability. We developed a JAX-based framework called ‘COMMPLAX’{cite}`commplax2021github` that can easily and practically implement the proposed stateful NN learning methodology and fully exploit the benefits of ML tools and adaptive DSP. COMMPLAX is a set of new versatile and powerful tools that go beyond the algorithms used in this paper and can provide a single common run-time efficient coding platform for DSP/ML research in optical communications. The codes and data for the results in this paper are available at [https://github.com/remifan/gdbp_study](https://github.com/remifan/gdbp_study).

## II. Algorithm structure and coding framework
The proposed NN + adaptive DSP structure to compensate fiber nonlinearity is shown in {numref}`gdbp-dsp`. The NN for fiber nonlinearity compensation consists of multiple interleaving layers of linear filters (called D-filters) and nonlinear filters (called N-filters) where the nonlinear phase rotations are filtered by the signal power waveform of the D-filter output. In this setup, conventional DBP corresponds to the standard chromatic dispersion (CD) equalizer for D-filter and a 1-tap N-filter. FDBP uses a $L_{N}$-tap N-filter with Gaussian, super-Gaussian triangular or brick-wall shape whose parameters are typically brute-force optimized {cite}`de2015optimization`.  EDBP relaxes the filter shape constraint so that each tap of the N-filter is optimized individually. Finally, the most generalized GDBP optimizes the individual D-filter taps in addition to EDBP. Note that all these DBP variants in comparison are not intended to be trained in a real-time manner. Therefore, GDBP has the same computational complexity as that of EDBP and FDBP.
```{figure} ./images/fig2.svg
---
height: 240px
name: gdbp-dsp
---
The NN GDBP and adaptive DSP structure proposed in our study. Conv1D: 1-D convolution layer. BPN: Batch power normalization; MIMO: multiple- input-multiple-output filters. FOE: frequency offset estimator; Note that MIMO and FOE are adaptive DSP.
```
In our study, the GDBP is followed by a frequency offset estimator(FOE) consisting of a radius directed equalizer (RDE) {cite}`fatadin2009blind` and Kalman filtering (KF) {cite}`inoue2014carrier`. Then, we use a residual filter (R-filter) that aims to implicitly learn and collectively mitigate all the residual unaccounted linear and/or nonlinear static impairments along the link. This is followed by adaptive DSP algorithms to compensate time-varying polarization effects and laser phase noise. As a good trade-off between transmission performance and complexity, we chose a T/2 fractionally spaced multiple-input-multiple-output (MIMO) decision directed least mean squares (DDLMS) filter {cite}`mori2012novel` to jointly perform polarization demultiplexing, symbol timing and phase equalization in our work. As will be shown in later sections, the R-filter is necessary since it improves overall transmission performance, and its functionalities cannot be realized by the MIMO filters which are often constrained to a few taps for optimal steady-state performance {cite}`mori2012novel`. The R-filter and adaptive DSP are additional NN layers appended to the GDBP, thus forming an overall NN structure. 

Let $p_{x\left(y\right)}(t)$ be the coherently detected waveform at the receiver for the $x$- or $y$-polarization followed by 2 times oversampling with sampling period $T/2$. Denote the $k^{th}$ batch of samples as GDBP input as

\begin{equation*}
\mathbf{P}_{1,x(y)}\left[k\right]=[p_{x\left(y\right)}\left(c_{k}T/2\right), p_{x\left(y\right)}\left(\left(c_{k}+1\right)T/2\right),\cdots , p_{x\left(y\right)}\left((c_{k}+G)T/2\right)]
\end{equation*}

where $c_{k}$ is the starting sample index for the $k^{th}$ data batch and $G$ is the length of the input batch. The output label $\mathbf{D}_{x(y)}[k]$of the overall neural network are either the corresponding original transmitted symbol sequence $\mathbf{E}_{x(y)}\left[k\right]$ or the decided symbol sequence with length $B$. We will now describe the forward and backward pass of the NN backpropagation algorithm in presence of time-varying impairments.

### A. Forward pass
In the forward pass of the backpropagation algorithm, let $\mathbf{P}_{m,x(y)}[k]$, $\mathbf{Q}_{m,x(y)}[k]$ be the signals within the $m$-th GDBP step so that

\begin{align*}
\tag{1}
\mathbf{Q}_{m,x\left(y\right)}\left[k\right] & =\mathbf{P}_{m,x\left(y\right)}\left[k\right]*\mathbf{d}_{m,x\left(y\right)}\left[k\right], \\
\tag{2}
\mathbf{P}_{m+1,x(y)}[k] & ={\mathbf{Q}_{m,x(y)}}[k]\odot e^{j({\left| \mathbf{Q}_{m,x}\left[k\right]\right| ^{2}}*{\mathbf{n}_{m,xx\left(yx\right)}}[k]+{\left| \mathbf{Q}_{m,y}[k]\right| ^{2}}*{\mathbf{n}_{m,xy\left(yy\right)}}[k])}
\end{align*}
Where $\odot $ denotes element-wise multiplication, $\mathbf{d}_{m,x}[k]$, $\mathbf{n}_{m,xx}[k]$ are the D-filter and N-filter with length $L_{D}$ and $L_{N}$ respectively and $\mathbf{*}$ denotes the convolution operation. For a $M$-step GDBP, the output $\mathbf{P}_{M+1,x(y)}[k]$\footnote{A batch power normalization maybe applied to $\mathbf{P}_{M+1,x(y)}[k]$ to obtain $\mathbf{P'}_{M+1,x(y)}[k]$ for improving the stability of subsequent adaptive DSP training.}is fed into frequency offset estimator based on RDE-KF with output
\begin{equation*}
\tag{3}
\mathbf{U}_{x(y)}\left[k\right]=\mathbf{P}_{M+1,x(y)}[k]\odot e^{j\left(\varphi _{k-1}+\pi \mathbf{f}_{k}\mathbf{T}\right)}
\end{equation*}

followed by the R-filter with output
\begin{equation*}
\tag{4}
\mathbf{S}_{x(y)}[k]=\mathbf{U}_{x(y)}[k]\mathbf{*r}_{x(y)}[k]
\end{equation*}

where $\mathbf{r}_{x(y)}[k]$ are the R-filter taps with length $L_{R}$, $\mathbf{f}_{k}$ is the FO estimate for each sample in the $k^{th}$ batch, $\mathbf{T}$ is an upper triangular matrix with $\mathbf{T}_{ab}=T$ for $b\geq a$ and 0 otherwise. In this case, $\varphi _{k}=\varphi _{k-1}+\pi \mathbf{f}_{k}\mathbf{T}$ denotes the FO-induced accumulated phase at the end of the $k^{th}$ batch. Note that in our implementation, $\mathbf{f}_{k}$is the linear interpolation of FOE’s block estimate {cite}`inoue2014carrier` over a batch of 5 blocks.  $\mathbf{S}_{x(y)}[k]$ is then passed into the set of MIMO filters $\mathbf{h}_{xx}$, $\mathbf{h}_{xy}$, $\mathbf{h}_{yx}$, $\mathbf{h}_{yy}$, with length $L$ that track and equalize time-varying polarization impairments and down sample to 1 sample per symbol. Within this batch of $\mathbf{S}_{x(y)}[k]$, define the $i^{th}$ MIMO output as
\begin{equation*}
\tag{5}
v_{x(y)}[k,i]=\mathbf{h}_{xx(yx)}^{\left(k,i\right)}\mathbf{S}_{x}^{'}[k,2i]+\mathbf{h}_{xy(yy)}^{\left(k,i\right)}\mathbf{S}_{y}^{'}[k,2i]
\end{equation*}

where
\begin{equation*}
\tag{6}
\mathbf{S}_{x(y)}^{'}\left[k,2i\right]=\left[s_{x\left(y\right)}\left[c_{k}+2i\right], s_{x\left(y\right)}\left[c_{k}+2i+1\right],\cdots ,s_{x\left(y\right)}[c_{k}+2i+L-1]\right]
\end{equation*}

is a truncated version of the $\mathbf{S}_{x(y)}[k]$. The MIMO filter taps are updated in a symbol-by-symbol fashion using the normalized LMS algorithm [23] 

\begin{align*}
\tag{7}
\mathbf{h}_{xx(yx)}^{\left(k,i+1\right)}&=\mathbf{h}_{xx(yx)}^{\left(k,i\right)}+\frac{\mu _{h}}{\left| \mathbf{S}_{x}^{'}\left[2i\right]\right| ^{2}+\varepsilon }e_{x(y)}\left[k,i\right]\odot \mathbf{S}_{x}^{\boldsymbol{'*}}\left[2i\right] \\
\mathbf{h}_{yy(xy)}^{\left(k,i+1\right)}&=\mathbf{h}_{yy(xy)}^{\left(k,i\right)}+\frac{\mu _{h}}{\left| \mathbf{S}_{y}^{'}\left[2i\right]\right| ^{2}+\varepsilon }e_{y(x)}\left[k,i\right]\odot \mathbf{S}_{y}^{\boldsymbol{'*}}\left[2i\right] 
\end{align*}

where $\mu _{h}$ is the step-size parameter, $\varepsilon $ is a small value used for numerical stabilization. Note the MIMO taps are updating symbol-by-symbol within this $k^{th}$ batch in the *forward pass* and hence from the perspective of NN training, the NN has a state that evolves within a batch and across data batches. The MSE is given by
\begin{equation*}
\tag{8}
e_{x(y)}[k,i]=d_{x(y)}[k,i]-v_{x(y)}[k,i]e^{j{\arg (f_{x(y)}}[k,i])}
\end{equation*}

where $d_{x(y)}[k,i]$ is the reference symbol, which is either the original transmitted symbols as pilots in training mode or detected symbol $D(f_{x(y)}[k,i]v_{x(y)}[k,i])$ in decision-directed tracking mode. The 1-tap carrier phase tracker $f_{x(y)}[k,i]$ is {cite}`mori2012novel` updated according to
\begin{equation*}
\tag{9}
f_{x(y)}\left[k,i+1\right]=f_{x(y)}\left[k,i\right]+\frac{\mu _{f}}{\left| v_{x(y)}[k,i]\right| ^{2}+\varepsilon }\left(d_{x(y)}[k,i]-f_{x(y)}[k,i]v_{x(y)}[k,i]\right)v_{x(y)}^{*}\left[k,i\right]
\end{equation*}

where $\mu _{f}$ is the step size. It should be noted that the calculation of MSE $e_{x(y)}[k,i]$ also evolves with symbol-by-symbol update of $\mathbf{h}_{xx}^{\left(k,i\right)},\mathbf{h}_{xy}^{\left(k,i\right)},\mathbf{h}_{yx}^{\left(k,i\right)}$and $\mathbf{h}_{yy}^{\left(k,i\right)}$within a batch of data, which will be reflected in the backward pass.

### B. Backward pass
Starting from the loss at the $i^{th}$ symbol of the $k^{th}$ batch
\begin{equation*}
\tag{10}
\begin{array}{ll}
\mathcal{L}[k,i] & =e_{x}[k,i]e_{x}^{*}[k,i]+e_{y}[k,i]e_{y}^{*}[k,i]\\
e_{x(y)}\left[k,i\right] & =\left(\mathbf{S}_{x\left(y\right)}^{'}\left[k,2i\right]\mathbf{h}_{xx\left(yy\right)}^{\left(k,i\right)}+\mathbf{S}_{y\left(x\right)}^{'}\left[k,2i\right]\mathbf{h}_{xy\left(yx\right)}^{\left(k,i\right)}\right)f_{x\left(y\right)}\left[k,i\right]-d_{x\left(y\right)}\left[k,i\right],
\end{array}
\end{equation*}

we apply the chain rule step-by-step and first derive the gradient of $\mathcal{L}(i)$ with respect to $\mathbf{S}_{x}^{\boldsymbol{'}}[k,2i]$ as 
\begin{equation*}
\tag{11}
\frac{\partial \mathcal{L}[k,i]}{\partial \mathbf{S}_{x}^{\boldsymbol{'}}[k,2i]}=\frac{\partial (e_{x}[k,i]e_{x}^{*}[k,i]+e_{y}[k,i]e_{y}^{*}[k,i])}{\partial \mathbf{S}_{x}^{\boldsymbol{'}}[k,2i]}.
\end{equation*}

Note that the above notation $\partial (\cdot )/\partial (\cdot )$ is called co-gradient in some literature {cite}`kreutz2009complex` to distinguish from the gradient operator $\nabla =\left(\partial (\cdot )/\partial (\cdot )\right)^{*}$ that represents the steepest descent direction. Using Wirtinger calculus for complex variables, we obtain
\begin{equation*}
\tag{12}
\begin{array}{ll}
\frac{\partial \mathcal{L}\left[k,i\right]}{\partial \mathbf{S}_{x}^{\boldsymbol{'}}\left[k,2i\right]} & =\frac{\partial \left(e_{x}\left[k,i\right]e_{x}^{*}\left[k,i\right]+e_{y}\left[k,i\right]e_{y}^{*}\left[k,i\right]\right)}{\partial \mathbf{S}_{x}^{\boldsymbol{'}}\left[k,2i\right]}\\ & =e_{x}^{*}\left[k,i\right]\frac{\partial e_{x}\left[k,i\right]}{\partial \mathbf{S}_{x}^{\boldsymbol{'}}\left[k,2i\right]}+e_{y}^{*}\left[k,i\right]\frac{\partial e_{y}\left[k,i\right]}{\partial \mathbf{S}_{x}^{\boldsymbol{'}}\left[k,2i\right]}\\ & =e_{x}^{*}\left[k,i\right]f_{x}\left[k,i\right]\mathbf{h}_{xx}^{\left(k,i\right)T}+e_{y}^{*}\left[k,i\right]f_{y}\left[k,i\right]\mathbf{h}_{xy}^{\left(k,i\right)T}.
\end{array}
\end{equation*}

Similarly,
\begin{equation*}
\tag{13}
\frac{\partial \mathcal{L}[k,i]}{\partial \mathbf{S}_{y}^{\boldsymbol{'}}[k,2i]}=e_{x}^{*}\left[k,i\right]f_{x}\left[k,i\right]\mathbf{h}_{yx}^{\left(k,i\right)T}+e_{y}^{*}\left[k,i\right]f_{y}\left[k,i\right]\mathbf{h}_{yy}^{\left(k,i\right)T}.
\end{equation*}

To calculate the co-gradient of loss with respect to the whole batch $\mathbf{S}_{x(y)}[k]$, we first express
\begin{equation*}
\tag{14}
\frac{\partial \mathcal{L}[k,i]}{\partial \mathbf{S}_{x(y)}[k]}=\left[0,0,\cdots ,0,\frac{\partial \mathcal{L}\left[k,i\right]}{\partial \mathbf{S}_{x\left(y\right)}^{\boldsymbol{'}}\left[k,2i\right]},0,\cdots ,0,0\right]
\end{equation*}

as a row vector with $2i$ zeros preceding $\frac{\partial \mathcal{L}\left[k,i\right]}{\partial \mathbf{S}_{x(y)}^{\boldsymbol{'}}\left[k,2i\right]}$ followed by $G-L-2i$ zeros. Note that the location of non-zero entries of $\frac{\partial \mathcal{L}[k,i]}{\partial \mathbf{S}_{x(y)}[k]}$ shift with $i$. Now, let $l\left[k\right]=\sum _{i=0}^{B-1}\mathcal{L}[k,i]$ be the total batch loss. The overall backpropagating co-gradient is then given by
\begin{equation*}
\tag{15}
\frac{\partial l[k]}{\partial \mathbf{S}_{x(y)}[k]}=\sum _{i=0}^{B-1}\frac{\partial \mathcal{L}[k,i]}{\partial \mathbf{S}_{x(y)}[k]}.
\end{equation*}

For the R-filter, the co-gradients with respect to the inputs are computed by convolving the reversed weight kernel with the gradients with respect to the output {cite}`mathieu2013fast` i.e.
\begin{equation*}
\tag{16}
\frac{\partial l[k]}{\partial \mathbf{U}_{x(y)}[k]}=\frac{\partial l[k]}{\partial \mathbf{S}_{x(y)}[k]}*\overleftarrow {\mathbf{r}_{x(y)}[\mathrm{k}]}
\end{equation*}

```{margin}
**Simplifications**

In practice, we deal with sequences with finite length, in which case the excat backpropagation formula of convolution depends on its mode/padding style. We have a short notebook to explain this {doc}`./notebooks/bp_conv1d`.
```

where $\overleftarrow {\left(\cdot \right)}$ denotes the flipped/reversed operation. This co-gradient will be backpropagated to update the GDBP parameters. On the other hand, the gradient for R-filter update is
\begin{equation*}
\tag{17}
\nabla _{{\mathbf{r}_{x(y)}}[k]}=\left(\frac{\partial l[k]}{\partial \mathbf{r}_{x(y)}[k]}\right)^{*}=\left(\frac{\partial l[k]}{\partial \mathbf{S}_{x(y)}[k]}*\overleftarrow {\mathbf{U}_{x(y)}[k]}\right)^{*}
\end{equation*}

and the co-gradients for the FOE layer is simply
\begin{equation*}
\tag{18}
\frac{\partial l[k]}{\partial \mathbf{P}_{M+1,x(y)}\left[k\right]}=\frac{\partial l[k]}{\partial \mathbf{U}_{x(y)}[k]}\odot e^{j\left(\varphi _{k-1}+\pi \mathbf{f}_{k}\mathbf{T}\right)}.
\end{equation*}

Finally, to update the GDBP parameters, one begins with $\frac{\partial l[k]}{\partial \mathbf{P}_{M+1,x(y)}[k]}$ and apply the chain rule to obtain
\begin{align*}
\tag{19}
\frac{\partial l\left[k\right]}{\partial \mathbf{Q}_{m,x\left(y\right)}\left[k\right]} & = \frac{\partial l\left[k\right]}{\partial \mathbf{P}_{m+1,x\left(y\right)}\left[k\right]}\odot \biggl[\mathbf{g}_{x\left(y\right)}\left[k\right]\\ &\quad +\left(j\mathbf{Q}_{m,x\left(y\right)}\left[k\right]\odot \mathbf{g}_{x\left(y\right)}\left[k\right]\right)*\overleftarrow{\mathbf{n}_{m,xx\left(yy\right)}\left[k\right]}\odot \mathbf{Q}_{m,x\left(y\right)}^{\boldsymbol{*}}\left[k\right]\\ &\quad +\mathbf{g}_{y\left(x\right)}\left[k\right]+\left(j\mathbf{Q}_{m,y\left(x\right)}\left[k\right]\odot \mathbf{g}_{y\left(x\right)}\left[k\right]\right)*\overleftarrow{\mathbf{n}_{m,yx\left(xy\right)}\left[k\right]}\odot \mathbf{Q}_{m,x\left(y\right)}^{\boldsymbol{*}}\left[k\right]\biggr] \\
\tag{20}
\frac{\partial l[k]}{\partial \mathbf{P}_{m,x(y)}\left[k\right]} & = \frac{\partial l[k]}{\partial \mathbf{Q}_{m,x(y)}\left[k\right]}*\mathbf{d}_{m,x\left(y\right)}\left[k\right]
\end{align*}

where $\mathbf{g}_{x\left(y\right)}\left[k\right]=e^{j({\left| \mathbf{Q}_{m,x}\left[k\right]\right| ^{2}}*{\mathbf{n}_{m,xx\left(yx\right)}}[k]+{\left| \mathbf{Q}_{m,y}\left[k\right]\right| ^{2}}*{\mathbf{n}_{m,xy\left(yy\right)}}[k])}$. The corresponding gradients of the $m^{th}$ GDBP layer are
\begin{align*}
\tag{21}
\nabla _{{\mathbf{n}_{m,xx\left(yy\right)}}\left[k\right]} & =\left(\frac{\partial l\left[k\right]}{\partial \mathbf{n}_{m,xx\left(yy\right)}\left[k\right]}\right)^{*}\\ & =\left(\frac{\partial l\left[k\right]}{\partial \mathbf{P}_{m+1,x\left(y\right)}\left[k\right]}\odot \left(\left(j\mathbf{Q}_{m,x\left(y\right)}\left[k\right]\odot \mathbf{g}_{x\left(y\right)}\left[k\right]\right)*\overleftarrow {\left| \mathbf{Q}_{m,x\left(y\right)}\left[k\right]\right| ^{2}}\right)\right)^{*} \\
\tag{22}
\nabla _{{\mathbf{n}_{m,xy\left(yx\right)}}\left[k\right]} & =\left(\frac{\partial l\left[k\right]}{\partial \mathbf{n}_{m,xy\left(yx\right)}\left[k\right]}\right)^{*}\\ & =\left(\frac{\partial l\left[k\right]}{\partial \mathbf{P}_{m+1,x\left(y\right)}\left[k\right]}\odot \left(\left(j\mathbf{Q}_{m,x(y)}\left[k\right]\odot \mathbf{g}_{x(y)}\left[k\right]\right)*\overleftarrow{\left| \mathbf{Q}_{m,y(x)}[k]\right| ^{2}}\right)\right)^{*} \\
\tag{23}
\nabla _{{\mathbf{d}_{m,x\left(y\right)}}\left[k\right]} & =\left(\frac{\partial l\left[k\right]}{\partial \mathbf{d}_{m,x\left(y\right)}\left[k\right]}\right)^{*}\\ & =\left(\frac{\partial l[k]}{\partial \mathbf{Q}_{m,x(y)}[k]}*\overleftarrow{\mathbf{P}_{m,x(y)}[k]}\right)^{*}
\end{align*}
and are repeated to obtain the updates of each GDBP step.

 The overall parameter update process derived above creates a multi-modal learning environment in which the MIMO taps are updated stochastically symbol by symbol while the D-, N- and R-filter learn their parameters batch by batch. It should also be noted that other FOE, MIMO and carrier phase tracker can be used as long as their corresponding update equations are properly incorporated into the NN state in the forward pass.
 
### C. JAX-based framework
JAX {cite}`jax2018github` is developed by Google and provide universal support of complex-valued numbers, efficient acceleration, parallelization, vectorization (easy scalability to more dimensions) of codes and distributed gradient computations that collectively enables hardware optimization and accelerate execution time. It also allows automatic differentiation of loss functions consisting of arbitrary compositions of accelerated, vectorized, parallelized functions. Most notably, there are more and more platforms built around JAX since 2019 such as Reinforcement Learning, Graph NN {cite}`deepmind` and discipline-specific tools for molecular dynamics, differentiable convex optimization, differentiable cosmology, transportation optimization etc {cite}`n2cholas`. Increasingly more ML research and ML applications are based on JAX, indicating its emerging importance and popularity.} For our study, we developed a JAX-based ML-DSP framework for physical layer optical communications called COMMPLAX {cite}`commplax2021github` which allows abstraction of the adaptive DSP as stateful layer interface, resulting flexible composition of DSP and regular NN layers.

## Experimental Results
```{margin}
<br/><br/><br/><br/><br/><br/>
<br/><br/><br/><br/><br/><br/>
<br/><br/><br/><br/><br/><br/>
**About PRNG**

As pointed out in {cite}`eriksson2017applying`, we could easily tell that a weak PRNG has pattern learning issue if certain performance panelty is present by shifting to stronger PRBS, yet it is hard to assert the absence of such pattern learning even in "the strongest PRNG". 

Though QRNG is the best option we took, there are some PRNGs that might be safe for this application:
- [mt19937ar](http://www.math.sci.hiroshima-u.ac.jp/m-mat/MT/MT2002/emt19937ar.html) as the [default PRNG of MATLAB](https://www.mathworks.com/help/matlab/ref/rng.html) is polular and has prevailed for many years, but it failed some recent [randomness tests](https://www.pcg-random.org/other-rngs.html) and there is a [paper](https://arxiv.org/pdf/1910.06437.pdf) explaining MT's history and its weakness.
- [PCG Family](https://www.pcg-random.org/), probably the best PRNG in terms of many aspects, they have a nice benchmark chart in their homepage. It is the [default PRNG of Numpy](https://numpy.org/doc/stable/reference/random/generator.html).
```
{numref}`experimental-setup` shows our WDM experimental setup. At the transmitter side, 2 36 GBaud 16-QAM sequences mapped from 2 independent random bit sources are generated for even and odd channels for decorrelation purpose. The channel spacing is 50 GHz. After raised-cosine pulse shaping with 0.1 roll-off factor, the signal is loaded into 112GSa/s arbitrary waveform generator (AWG, Keysight 8194A) whose outputs are amplified by SHF-807 to drive the modulator. Polarization division multiplexing (PDM) emulator is used to emulate polarization multiplexed signals through polarization beam splitter (PBS)-delay-polarization beam combiner (PBC) approach. An EDFA is used to increase the signal power before launching into the 1125-km transmission link consisting of 15 heterogeneous spans. An EDFA-VOA-optical filter triplet is installed for each span which resembles practical settings. The signal launched power is balanced automatically through our link controller to automate sweeping. At the receiver side, a wavelength selective switch (WSS) and local oscillator (LO) are synchronized to collect each channel in time division multiplexed (TDM) manner followed by coherent detection, 80 GSa/s real-time oscilloscope and offline signal processing described above. The above processes are programmed to sweep the launched power for each of the 7 channels to form the training dataset. The whole process is repeated with a completely different random sequence to obtain the testing data. To eliminate any pattern recognition issues for PRBS-based bit sequences {cite}`eriksson2017applying`, we used the live quantum random numbers generator (QRNG) {cite}`symul2011real` through the public interface provided by Australian National University. Two unique sequences of 170624 symbols are obtained and each sequence repeated itself multiple times to form two sequences of 1 and 1.5 million symbols for training and testing respectively.
```{figure} ./images/fig3.svg
---
height: 500px
name: experimental-setup
---
Experimental setup for 7-channel WDM transmission experiment. AWG: arbitrary waveform generator. ADC: analog to digital converter; WSS: Wavelength selection switch; PDM Emu.: polarization division multiplex emulator; LO: local oscillator; ADC: analog to digital converter; DSO: digital sampling oscilloscope.
```
We focus on 5-span per step GDBP (i.e. 3 total steps for our 15 span link) as it is shown to have a good performance-complexity tradeoff for FDBP {cite}`gao2012assessment`. As a result of brute-force grid search similar to {cite}`de2015optimization`, the optimal nonlinear phase de-rotation factor is 0.15 and 1.1 for all 3 steps of DBP and FDBP respectively. A 3-dB bandwidth of 1.44GHz is found to be 

optimal for the Gaussian-shaped 41-tap LPF of the FDBP. The length of D-filters is 261 taps and the 61-tap R-filters are initialized as delta functions. The 31-tap MIMO filters $\mathbf{h}_{xx}$, $\mathbf{h}_{yy}$ are initialized as delta functions and $\mathbf{h}_{xy}$, $\mathbf{h}_{yx}$ are initialized as all-zero filters.

```{margin}
**Complex optimizers**

Unlike plain SGD optimizer, for Complex parameters, Adam optimizers from popular ML libraries do not work as expected out-of-the-box.  See issues of [Tensorflow](https://github.com/tensorflow/tensorflow/issues/38541), [PyTorch](https://github.com/pytorch/pytorch/issues/59998)). We have a [notebook](https://github.com/remifan/complex_adam/blob/master/complex_adam_example.ipynb) to demonstrate this issue. Though there is no standard definition of Complex-valued Adam, there are a few well-agreed options to choose from in practice, check this note {doc}`./notebooks/complex_optim` for further information.
```
Complex-valued Adam optimizer are used for training and a batch size of 500 symbols for each iteration provides optimal results from our experimental data. Adam’s learning rate is set to be $10^{-4}$ in first 500 batches and $10^{-5}$ afterwards. The evolution of MSE is shown in {numref}`training-loss` (a) and convergence is achieved after around 500 iterations for combined NN and MIMO-DDLMS training and MIMO-DDLMS training only. The steady-state training performance is stable within the 2000-iteration time window. Also, with FDBP configurations as initialization points, {numref}`training-loss` (b) shows the averaged normalized MSE between the $i^{th}$ and last iteration ($2000^{th}$ iteration in our study) over the D-, N- and R-filter taps.
```{figure} ./images/fig4.svg
---
height: 700px
name: training-loss
---
(a) MSE evolution for combined NN and MIMO-DDLMS training and MIMO-DDLMS training only. (b) Evolution of averaged normalized MSE of the D-, N- and R-filters of the 1st step of GDBP (collectively denoted by 𝜽) from their respective converged values.
```
 Throughout these iterations, the estimated phase noise and maxima of the 4 MIMO filter taps of the center channel is shown in {numref}`time-varying-states`, showing the presence of time-varying impairments across iterations and within each batch of 500 symbols. This indicates the concurrent learning of GDBP and MIMO parameters and thus the integration of ML and adaptive DSP in the learning process. The MIMO filter taps converge relatively quickly followed by continuous tracking of polarization effects. The reduction of the maxima of $\left| \mathbf{h}_{xx}\right| ,| \mathbf{h}_{yy}| $ over time indicates the presence of time-varying polarization-mode dispersion (PMD) in the testing data as the filters slowly deviate from a delta function.
```{figure} ./images/fig5.svg
---
height: 650px
name: time-varying-states
---
(a)Estimated phase noise for testing data of channel 4. Inset: phase noise within 1 batch of data (500 symbols). (b) evolution of the maximal tap of the 4 MIMO filters for testing data of channel 4.
```
Using the testing data, the $Q$-factor of the received signal distribution (derived from the bit error ratio(BER) through $Q=20\log _{10} \left[\sqrt{2}erfc^{-1}\left(2BER\right)\right]$ where $erfc^{-1}(\cdot)$ is the inverse complementary error function) vs launched power for channels 1 and 4 are shown in {numref}`Q-vs-lp` (a) and (b) respectively. As expected, FDBP outperforms standard DBP for the same number of steps. More importantly, EDBP and the most generalized GDBP outperform FDBP by an average of 0.08 dB and 0.13 dB respectively across the 7 channels and outperform DBP by 0.31 dB and 0.36 dB respectively. The optimal performance for each algorithm across the 7 channels are also shown in {numref}`Q-vs-lp`(c). For WDM systems in presence of time-varying impairments, the results represent the first experimental demonstration of EDBP and GDBP using practical training strategies enabled by our proposed stateful NN structure and the JAX-based framework. Furthermore, as the most general form with largest number of optimization variables for a given total number of steps, the GDBP is also the best single-channel DBP based-fiber nonlinearity compensation algorithm demonstrated in WDM experiments.
``````{margin}
```{admonition} Source notebook
{doc}`./notebooks/benchmark`.
```
``````

```{code-cell} ipython3
---
tags: [remove-input]
---
import plotly.express as px
import pandas as pd

df_src = pd.read_csv('../source_data/benchmark_regular_taps.csv')
labels = ['(a)', '(b)', '(c)']
figs = []

grp_ch = df_src.loc[df_src['ChInd'].isin([1, 4])].groupby('ChInd')
for n_ch, g_ch in grp_ch:
  figs.append(px.line(g_ch, x='LPdBm', y='Q', markers=True, color='Model', template='plotly_white',
                      title=labels.pop(0), labels={'Q': 'Q-factor (dB)', 'LPdBm': 'Launch power (dBm)'}))

df = df_src.groupby(['ChInd', 'Model'], as_index=False, sort=False)['Q'].max()
figs.append(px.scatter(df, x='ChInd', y='Q', color='Model', template='plotly_white', title=labels.pop(0),
                labels={'Q': 'Q-factor (dB)', 'ChInd': 'Channel index'}))
figs[0].update_layout(height=400, margin=dict(l=100,r=100,b=10,t=40,pad=0))
figs[1].update_layout(height=400, margin=dict(l=100,r=100,b=10,t=40,pad=0))
figs[2].update_layout(height=400, margin=dict(l=100,r=100,b=0,t=40,pad=0))
for f in figs:
  f.show()
```
```{figure} ./images/smallfig.png
:height: 0px
:name: Q-vs-lp

(a), (b) Q-factor vs. launched power for Channel 1 and 4 using CD compensation (CDC) only, DBP, FDBP, EDBP and GDBP for a 7 x 288 Gb/s PM-16QAM system transmitted over 1125 km. (c) Optimal Q-factor for various fiber nonlinearity compensation algorithms across the 7 WDM channels.
```

The learnt filter shapes of the GDBP for are shown in {numref}`learnt-params`. The optimized D-filters largely preserved the quadratic phase response for CD compensation while the amplitude responses deviate slightly from a spectrally-flat profile. For the N-filters, the optimal shape deviate from Gaussian and it accounts for the performance difference between FDBP and EDBP. It should be noted that the N-filter spectra is different for each step. We used a pair of 61-tap R-filters after frequency offset compensation and their optimized configurations do not seem to exhibit any interpretable structure and vary across different channels in a random fashion. This should be expected as the R-filter is meant to mitigate any residual or unmodelled linear/nonlinear impairments in the link. The R-filter is important as all the $Q$-factors in {numref}`Q-vs-lp` (except those of GDBP) will be reduced by around 0.06 dB without the R-filter. This also illustrates the versatility of GDBP as it implicitly incorporates the R-filter’s functionalities and it becomes redundant when GDBP is used. The subsequent 32-tap MIMO filters track channel dynamics such as SOP rotation and/or PMD as described above followed by a 1-tap equalizer for estimating and compensating laser phase noise.
``````{margin}
```{admonition} Source notebook
{doc}`./notebooks/learned_params`.
```
``````
```{code-cell} ipython3
---
tags: [remove-input]
---
import pandas as pd
import plt_util
  
pch1 = pd.read_csv('../source_data/params_ch1.csv')
pch4 = pd.read_csv('../source_data/params_ch4.csv')

plt_util.py_params(pch1, 'Ch1', dict(l=100,r=100,b=40,t=40,pad=0))
plt_util.py_params(pch4, 'Ch4', dict(l=100,r=100,b=0,t=40,pad=0))
```
```{figure} ./images/smallfig.png
---
height: 0px
name: learnt-params
---
Learnt D-, N- and R-filter taps configuration for Channel 1 and 4 of a 7 x 288 Gb/s PM-16QAM system transmitted over 1125 km.
```
In addition to pushing the limits of long-haul transmission with GDBP, we also apply our combined training method in low computational complexity scenarios. We studied the case ~15% and ~70% reduction in D-filter and N-filter lengths respectively and the transmission performance of channel 4 and learnt filter configurations are shown in {numref}`Q-vs-lp-fewer-taps`. Note that the R-filters are not shortened here as we want to ensure minimal residual distortions and optimize CDC, DBP, FDBP and EDBP performance for comparison with GDBP. It can be seen that in contrast to the previous study where the learnt N-filters plays a bigger role in performance improvements than the learnt D-filters, the learnt D-filters in this case account for the majority of the gain and significantly outperforms EDBP and other algorithms by around 1 dB. This serves as an experimental verification of previous simulation studies {cite}`hager2020physics` suggesting that optimized D-filters can relax the requirement on its length.
```{code-cell} ipython3
---
tags: [remove-input]
---
import pandas as pd
import plt_util
  
df_bm = pd.read_csv('../source_data/benchmark_few_taps.csv')
df_p = pd.read_csv('../source_data/params_ch4_few_taps.csv')
figs = []

figs.append(px.line(df_bm, x='LPdBm', y='Q', markers=True, color='Model', template='plotly_white',
                    title='(a)', labels={'Q': 'Q-factor (dB)', 'LPdBm': 'Launch power (dBm)'}))
figs.append(plt_util.py_params_few_taps(df_p, '(b)', margin=dict(l=100,r=100,b=0,t=30,pad=0)))

for f in figs:
  f.show()
```
```{figure} ./images/smallfig.png
---
height: 0px
name: Q-vs-lp-fewer-taps
---
(a) Q-factor vs. launched power using CD compensation (CDC) only, DBP, FDBP, EDBP and GDBP with reduced tap lengths for D-filter(221 taps) and N-filter(11 taps) for channel 4 of a 7 x 288 Gb/s PM-16QAM system transmitted over 1125 km; (b) Learnt D-and N-filter taps configuration.
```
We further studied the effects of D-filter and N-filter lengths on $Q$-factor and as shown in {numref}`Q-vs-taps`, GDBP is notably robust to D-filter shortening while EDBP and FDBP performance drastically degrades. Meanwhile, N-filter shortening does not affect performance as much.
``````{margin}
```{admonition} Source notebook
{doc}`./notebooks/benchmark_few_taps`.
```
``````
```{code-cell} ipython3
---
tags: [remove-input]
---
import pandas as pd
import plt_util
  
df = pd.read_csv('../source_data/Q_vs_taps.csv')

plt_util.py_qvstaps(df)
```
```{figure} ./images/smallfig.png
---
height: 0px
name: Q-vs-taps
---
Q-factor vs. D-filter and N-filter length for channel 4 with 0 dBm launched power using FDBP, EDBP and GDBP for a 7 x 288 Gb/s PM- 16QAM system transmitted over 1125 km.
```

## Conclusion
In this paper, we showed how one can combine neural network and adaptive DSP training by transforming the adaptive DSP as an additional stateful NN layer appended to the main NN so that the overall signal processing block amends itself to standard ML training techniques such as backpropagation with efficient and practical learning methodologies. Under this framework, conventional filter tap updates correspond to NN state updates and we derived the full forward and backward pass co-gradients and update equations for all the parameters of the NN and adaptive filter taps. Implemented through the new flexible JAX framework, we conducted the first experimental demonstration of EDBP and GDBP over a 7 x 288 Gb/s PM-16QAM system transmitted over 1125 km. With the largest number of optimization variables for a given total number of steps, GDBP is the best single-channel DBP based-fiber nonlinearity compensation algorithm demonstrated in WDM experiments. In complexity-constrained scenarios, transmission performance of GDBP is not severely compromised while those of other fiber nonlinearity compensation algorithms degrade significantly. Extensions to the combined learning of other transmission impairments such as transceiver impairments, characterization of the framework’s ability in tracking polarization effects and mode-coupling dynamics in multi-mode systems will be areas of future research. Finally, the developed JAX-based framework COMMPLAX improves coding flexibility, efficiency and versatility for ML/DSP in optical communications. With the increasing popularity of JAX in various research areas, it is the hope that COMMPLAX can foster more interdisciplinary research between signal processing in optical communications and the wider ML/Computational Science and Engineering community.

## References
```{bibliography}
:filter: docname in docnames
```