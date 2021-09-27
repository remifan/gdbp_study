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

# what is web-based interactive research paper presentation
```{only} html
[![DOI](https://badgen.net/badge/DOI/10.1109%2FJLT.2021.3111437/blue)](https://ieeexplore.ieee.org/document/9534655)
[![Jupyter Book Badge](images/jbook_badge.svg)](https://jupyterbook.org)
```

This is a extended web version of the [journal article](https://ieeexplore.ieee.org/abstract/document/9534655/?casa_token=k0nfUkc0KfsAAAAA:t6co-nT1B_q8jei3sQNvOprr33wIwCsCgkeh0GJMq9oSvv7s0NdV0HuYBVswxNnZVFnJOFAlTeU) titled *Combined neural network and adaptive DSP training for long-haul optical communications* published in *Journal of Lightwave Technology* with DOI:10.1109/JLT.2021.3111437

In addition to the basic formating with classic PDF article, you can expect more features from this material:
- with the source codes written as Jupyter notebooks, you can reproduce the results via cloud runtime (needs Colab access)
- additional Authors' remarks are present along the main content 
- in-page reader comments/responses
- interactive figures
- better cross-references
- equations with LaTex source codes
- continious improvements/revision of the content

We have seen [opinions about the traditional media of research article](https://www.theatlantic.com/science/archive/2018/04/the-scientific-paper-is-obsolete/556676/), if research paper is seen as "communication channel" to transmit scientific information, we are trying our best to maximize its "capacity" in this material.

## Source codes
``````{margin}
```{note}
Authors' remarks stay at this sidebar along with the table-of-content. We choose washed font-color to minimize distracition.
```
``````

At source notebook page, you can execute the source codes directly in [Colab](https://research.google.com/colaboratory/) by hovering on <i class="fas fa-rocket"></i> at the top bar and choose `Colab`. You may also download the notebook by hovering on <i class="fas fa-download"></i> and click `.ipynb` so that you can run them locally.



## Figures
Most figures in this article are reproduced into interactive graphs.
Below is an example.

You can:
- drag to zoom in and double click to resume
- move the cursor around to see the number indicators
- single click on a legend item to hide that trace, double click to isolate other traces
- see more options in the image toolbar


```{code-cell} ipython3
---
tags: [remove-input]
---
import plotly.graph_objects as go

# Create random data with numpy
import numpy as np
np.random.seed(1)

N = 100
random_x = np.linspace(0, 1, N)
random_y0 = np.random.randn(N) + 5
random_y1 = np.random.randn(N)
random_y2 = np.random.randn(N) - 5

fig = go.Figure()

# Add traces
fig.add_trace(go.Scatter(x=random_x, y=random_y0,
                    mode='markers',
                    name='markers'))
fig.add_trace(go.Scatter(x=random_x, y=random_y1,
                    mode='lines+markers',
                    name='lines+markers'))
fig.add_trace(go.Scatter(x=random_x, y=random_y2,
                    mode='lines',
                    name='lines'))
fig.update_layout(height=400, margin=dict(l=10,r=10,b=10,t=10,pad=0))
fig.show()
```

## Equations

Equations rendering is done by [MathJax](https://www.mathjax.org/), which allows checking $\LaTeX$ commands in the right click menu of each equation. 

## Feedbacks
You are welcome to comment via:
- selecting and annotating the contents (click the highlighted to reveal the existed comments). All the public annotations are visible by default, you may hide them by clicking <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 16 16" class=""><g fill-rule="evenodd"><rect fill="none" stroke="none" x="0" y="0" width="16" height="16"></rect><path fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 13c3.866 0 7-2.239 7-5s-3.134-5-7-5-7 2.239-7 5 3.134 5 7 5zm0-4a1 1 0 1 0 0-2 1 1 0 0 0 0 2z"></path></g></svg> button at the top right of each page.
You need a account of [Hyposhsis](https://web.hypothes.is/) to leave comments.
- raising issue or commenting in the discussion panel in the Github repository (<i class="fab fa-github"></i> link at the top).

## Cite
    @article{fan2021combined,
      title={Combined neural network and adaptive DSP training for long-haul optical communications},
      author={Fan, Qirui and Lu, Chao and Lau, Alan Pak Tao},
      journal={Journal of Lightwave Technology},
      year={2021},
      publisher={IEEE}
    }
