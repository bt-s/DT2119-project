from metrics import compute_metric, compute_metric_per_instrument
import pickle
import numpy as np

import plotly
import plotly.plotly as py
import plotly.graph_objs as go

method = 1

if method == 1:
    thetas = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18]
elif method == 2:
    thetas = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45,0.50, 0.55, 0.60]
else:
    print("Wrong argument for method")

names = ["without0", "without4", "without6", "with0", "with4", "with6"]

################################ Ground Truth #################################
f = open("datasets/y_test.pkl", "rb")
y_test = pickle.load(f)
f.close()

a = np.array(list(y_test.values()))
y_test = np.zeros((len(a), 11))
for i in range(len(a)):
    y_test[i] = a[i][0]
###############################################################################



############################### Plotly figures ################################
fig_per_inst = plotly.tools.make_subplots(rows=2, cols=3, 
        subplot_titles=('No BN, No Max-p', 'No BN, Max-p = 4',
            'No BN, Max-p = 6', 'BN, No Max_p',
            'BN, Max_p = 4', 'BN, Max_p = 6'))
fig_micro = plotly.tools.make_subplots(rows=2, cols=3, 
        subplot_titles=('No BN, No Max-p', 'No BN, Max-p = 4',
            'No BN, Max-p = 6', 'BN, No Max_p',
            'BN, Max_p = 4', 'BN, Max_p = 6'))
fig_macro = plotly.tools.make_subplots(rows=2, cols=3, 
    subplot_titles=('No BN, No Max-p', 'No BN, Max-p = 4',
        'No BN, Max-p = 6', 'BN, No Max_p',
        'BN, Max_p = 4', 'BN, Max_p = 6'))
###############################################################################



legend = True   # We want to plot the legend only once for every subplot
for i, name in enumerate(names):
    f = open("predictions/" + name, "rb")
    predictions_every_theta = pickle.load(f)
    f.close()

    ######################### Micro and Macro results #########################
    micro, macro = [], []
    max_best_theta, argmax_best_theta = -1, -1
    for predictions_theta in predictions_every_theta:
        micro.append(compute_metric(y_test, predictions_theta))
        macro.append(compute_metric(y_test, predictions_theta, mtype='macro'))
        if macro[-1][2] > max_best_theta:       #Useful for per_instrument
            argmax_best_theta = len(macro) - 1
            max_best_theta = macro[-1][2]

    micro_plt = [
            go.Scatter(
                y = [micro[i][0] for i in range(len(micro))],
                x = thetas,
                line=dict(
                    color='blue'
                    ),
                showlegend = legend,
                name = "precision"
                ),
            go.Scatter(
                y = [micro[i][1] for i in range(len(micro))],
                x = thetas,
                line=dict(
                    color='orange'
                    ),
                showlegend = legend,
                name = "recall"
                ),
            go.Scatter(
                y = [micro[i][2] for i in range(len(micro))],
                x = thetas,
                line=dict(
                    color='green'
                    ),
                showlegend = legend,
                name = "f1"
                )
            ]

    fig_micro.append_trace(micro_plt[0], i//3+1, i%3+1)
    fig_micro.append_trace(micro_plt[1], i//3+1, i%3+1)
    fig_micro.append_trace(micro_plt[2], i//3+1, i%3+1)


    macro_plt = [
            go.Scatter(
                y = [macro[i][0] for i in range(len(micro))],
                x = thetas,
                line=dict(
                    color='blue'
                    ),
                showlegend = legend,
                name = "precision"
                ),
            go.Scatter(
                y = [macro[i][1] for i in range(len(micro))],
                x = thetas,
                line=dict(
                    color='orange'
                    ),
                showlegend = legend,
                name = "recall"
                ),
            go.Scatter(
                y = [macro[i][2] for i in range(len(micro))],
                x = thetas,
                line=dict(
                    color='green'
                    ),
                showlegend = legend,
                name = "f1"
                )
            ]

    fig_macro.append_trace(macro_plt[0], i//3+1, i%3+1)
    fig_macro.append_trace(macro_plt[1], i//3+1, i%3+1)
    fig_macro.append_trace(macro_plt[2], i//3+1, i%3+1)
    ###########################################################################



    ######################### Per instrument results ##########################
    inst_p, inst_r, inst_f1 = compute_metric_per_instrument(y_test, predictions_every_theta[argmax_best_theta])

    x = list(inst_p.keys()) 
    per_inst = [
            go.Histogram(
                histfunc = "sum",
                y = list(inst_p.values()),
                x = x,
                marker=dict(
                    color='blue'
                    ),
                showlegend = legend,
                name = "precision"
                ),
            go.Histogram(
                histfunc = "sum",
                y = list(inst_r.values()),
                x = x,
                marker=dict(
                    color='orange'
                    ),
                showlegend = legend,
                name = "recall"
                ),
            go.Histogram(
                histfunc = "sum",
                y = list(inst_f1.values()),
                x = x,
                marker=dict(
                    color='green'
                    ),
                showlegend = legend,
                name = "f1"
                )
            ] 

    fig_per_inst.append_trace(per_inst[0], i//3+1, i%3+1)
    fig_per_inst.append_trace(per_inst[1], i//3+1, i%3+1)
    fig_per_inst.append_trace(per_inst[2], i//3+1, i%3+1)
    ###########################################################################

    legend = False

py.iplot(fig_per_inst, filename = "per_inst.html")
py.iplot(fig_micro, filename = "micro.html")
py.iplot(fig_macro, filename = "macro.html")
