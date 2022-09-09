# PortAttribute

Provide basci performance attribution for portfolio

1. NAV time series based attribution

    ```python
    from portatt import ret_metric
    ret_metric.cal_ret_summary(ret, period, rf=0., benchmark=None)
    ```

2. Holding Based attribution
    Factor based attribution, which including ex post return attribution and ex post/ex ante risk attribution with the following setup
    ``` latext
        \sigma^2 = w^T \Sigma w = (w^TB)\Sigma_f(B^T w) + w^TSw \\
        MCR = \dfrac{\partial{\sigma}}{\partial{w}} = \dfrac{B\Sigma_fB^Tw + Sw}{\sigma} \\
        CR = w_i * MCR_i\\
        \sum CR_i  = \sigma
    ```


3. Return Linking (In progress)
    Return linking to make snapshot based attribution addable
