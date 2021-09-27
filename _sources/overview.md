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

# What is web-based interactive research paper presentation?
```{only} html
[![DOI](https://badgen.net/badge/DOI/10.1109%2FJLT.2021.3111437/blue)](https://ieeexplore.ieee.org/document/9534655)
[![Jupyter Book Badge](images/jbook_badge.svg)](https://jupyterbook.org)
```

This is a extended web version of the [journal article](https://ieeexplore.ieee.org/abstract/document/9534655/?casa_token=k0nfUkc0KfsAAAAA:t6co-nT1B_q8jei3sQNvOprr33wIwCsCgkeh0GJMq9oSvv7s0NdV0HuYBVswxNnZVFnJOFAlTeU) titled *Combined neural network and adaptive DSP training for long-haul optical communications* to be published in *Journal of Lightwave Technology* with DOI:10.1109/JLT.2021.3111437.

Here we have open-sourced the data we obtained from our experimental setup and you can run all the (well-documented) codes to reproduce the results presented in the paper. You can also select and display different results and view it from different angles. 

## What's the problem with regular research papers presented in static pdfs? Why bother making a web-based interactive version?
Desipte the time-tested robustness of PDF-based article, we have seen [opinions about this traditional media](https://www.theatlantic.com/science/archive/2018/04/the-scientific-paper-is-obsolete/556676/), and we understand most of the complains:
- much information is lost in the final results, which might lead to misinterpretion, and much time is wasted just to understand its content properly unless one is very familar with that topic.
- to respect "[Code as a Research Object](https://www.researchobject.org/initiative/code-as-a-research-object/)" initiative, it'd great to see a direct computation-results reproduction within the article itself, obviously static papers just get in the way.
- lack of continous revision (versioning) and citation including revison information.

Some other disciplines, like math and physics have long tried to extended their papers to use other media to present the computation process, the so-called [notebook interface](https://en.wikipedia.org/wiki/Notebook_interface) is one of such alternatives which was invented and soon became the standard interface of Mathematica more than 30 years ago!

Not everyone has a copy of Mathematica but web-browser is universal. Web-based presentation was considered as a bad option for research content due to its blur image-based equations, informal typesetting, lack of cross-referencing and other basic formating functions. Over the past years, there are many open-sourced communities trying to make a difference from specific aspacts:
- equations: [MathJax](https://www.mathjax.org/), [KaTex](https://katex.org/)
- figure: [Plotly](https://plotly.com/python/), [Altair](https://altair-viz.github.io/), [Bokeh](https://bokeh.org/)
- formating: [Sphinx](https://www.sphinx-doc.org/en/master/), [MyST](https://myst-parser.readthedocs.io/en/latest/index.html)

You might never hear these names but they have been partially used in many different areas. In the recent [Jupyter Book](https://jupyterbook.org) projoct, they are all well interagted as part of Jupyter Book ecosystem which allow powerful presentation based on all of these enabling tools.

While it is obvious that Optics and Photonics involve a lot more experimental work that are not meant to be 'reproducible' by someone else looking at the screen on the other side of the planet, open-sourcing the obtained data, enabling readers to understand their processing methodologies and try it themselves to improve reprodibility and readibility are becoming a required component when presenting one's work in all research disciplines. Therefore, it is important for us in Optics to also adopt this increasingly common trend in Science and Engineering research for the benefit of our own work and beyond. If research paper is seen as "communication channel" to transmit scientific information, we are trying our best to maximize its "capacity" in this material.

In addition to the basic formating with classic PDF article, you can expect more features from this material:
- with the source codes written as Jupyter notebooks, you can reproduce the results via cloud runtime (needs Colab access)
- additional Authors' remarks are present along the main content 
- in-page reader comments/responses
- interactive figures
- better cross-references
- equations with LaTex source codes
- continious improvements/revision of the content

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
