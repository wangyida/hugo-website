To formalize the down-sampling operation, we denote the input as the set
of feature vectors
$\mathcal{F}_\text{in}= \{\mathbf{f}_i\}_{i=1}^{|\mathcal{F}_\text{in}|}$
where $\mathbf{f}_i$ is a feature vector and $|\cdot|$ is the number of
elements in the set. Note that, in the first layer of the encoder,
$\mathcal{F}_\text{in}$ is then set to the coordinates of the input
point cloud. We introduce a novel down-sampling operation inspired from
the Iterative Closest Point (ICP)
algorithm [@besl1992method; @chen1992object]. Taking an arbitrary anchor
$\mathbf{f}$ from $\mathcal{F}_\text{in}$, we start by defining a vector
$\delta \in \mathbb{R}^{D_\text{in}}$. From the trainable variable
$\delta$, we find the feature closest to $\mathbf{f}+\delta$ and compute
the distance. This is formally formulated as a function where $\delta$
represents a displacement vector from $\mathbf{f}$. Multiple
displacement vectors are used to describe the local geometry, each with
a weight $\sigma \in \mathbb{R}$. We then assign the set as
$\{(\delta_i, \sigma_i)\}_{i=1}^s$ and aggregate them with the weighted
function where the constants $\alpha$ and $\beta$ are added for
numerical stability. Here, the hyperbolic tangent in $g(\mathbf{f})$
produces values closer to 1 when the distance $d(\cdot)$ is small and
closer to 0 when the distance is large. In practice, we can speed-up
[\[eq:closest_distance\]](#eq:closest_distance){reference-type="eqref"
reference="eq:closest_distance"} with the $k$-nearest neighbor search
for each anchor. A simple example of this operation is illustrated in .
This illustrates the operation in the first layer where we process the
point cloud so that we can geometrically plot a feature in
$\mathcal{F}_\text{in}$ with respect to
$\{(\delta_i, \sigma_i)\}_{i=1}^s$.

Furthermore, to enforce the influence of the anchor in this operation,
we also introduce the function $$\begin{aligned}
h ( \mathbf{f}) = 
\rho \cdot \mathbf{f}\end{aligned}$$ that projects $\mathbf{f}$ on
$\rho \in \mathbb{R}^{D_\text{in}}$, which is a trainable parameter.
Note that both functions $g(\cdot)$ and $h(\cdot)$ produce a scalar
value.

Thus, if we aim at building a set of output feature vectors, each with a
dimension of $D_\text{out}$, we construct the set as where different
sets of trainable parameters $\{(\delta_i,\sigma_i)\}_{i=1}^{s}$ are
assigned to each element, while different $\rho$ for each output vector.
Moreover, the variables $s$ in
[\[eq:g_func\]](#eq:g_func){reference-type="eqref"
reference="eq:g_func"} and $D_\text{out}$ in
[\[eq:down\]](#eq:down){reference-type="eqref" reference="eq:down"} are
the hyper-parameters. We label this operation as the *Feature
Extraction*.
