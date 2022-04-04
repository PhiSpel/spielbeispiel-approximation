import streamlit as st

#from scipy.interpolate import interp1d, CubicSpline

#from sympy import *
#from sympy.parsing.sympy_parser import parse_expr
#from sympy.abc import x

import numpy as np
#import pandas as pd

#import altair as alt

import matplotlib.pyplot as plt
import matplotlib.font_manager

import random

# make sure the humor sans font is found. This only needs to be done once
# on a system, but it is done here at start up for usage on share.streamlit.io.
matplotlib.font_manager.findfont('Humor Sans', rebuild_if_missing=True)

#############################################
# Define the function that updates the plot #
#############################################

@st.cache(suppress_st_warning=True)
def update_approx(xs,data,datatype,approxtype):
    if approxtype == 'constant':
        z=np.polyfit(xs,data,0)
    elif approxtype == 'linear':
        z=np.polyfit(xs,data,1)
    elif approxtype == 'quadratic':
        z=np.polyfit(xs,data,2)
    elif approxtype == 'cubic':
        z=np.polyfit(xs,data,3)
    approx=np.poly1d(z)
    return approx

# To Do: Why does caching update_plot hang?
# @st.cache(suppress_st_warning=True)
def update_plot(xs, data, approx, f_input, show_solution, ticks_on):
    
    """
    Creates a Matplotlib plot if the dictionary st.session_state.handles is empty, otherwise
    updates a Matplotlib plot by modifying the plot handles stored in st.session_state.handles.
    The figure is stored in st.session_state.fig.

    :param t0: Evaluation point of the function/Taylor polynomial
    :param ft0: Function evaluated at t0
    :param xs: numpy-array of x-coordinates
    :param ys: numpy-array of f(x)-coordinates
    :param ps: numpy-array of P(x)-coordinates, where P is the Taylor polynomial
    :param visible: A flag wether the Taylor polynomial is visible or not
    :param xmin: minimum x-range value
    :param xmax: maximum x-range value
    :param ymin: minimum y-range value
    :param ymax: maximum y-range value
    :return: none.
    """
    
    tmin = min(xs)
    tmax = max(xs)
    length = tmax-tmin
    dt = round(length/10,1)
    
    ymin = min(data)
    ymax = max(data)
    heigth = ymax-ymin
    dy = round(heigth/10,1)
    
    if f_input:
        f = lambda x: eval(f_input)
    else:
        f = lambda x: 0
    ys= [f(x) for x in xs]
    
    approx = approx(xs)
        
    handles = st.session_state.handles

    ax = st.session_state.mpl_fig.axes[0]

    # if the dictionary of plot handles is empty, the plot does not exist yet. We create it. Otherwise the plot exists,
    # and we can update the plot handles in fs, without having to redraw everything (better performance).
    if not handles:
        #######################
        # Initialize the plot #
        #######################

        # plot the data points
        handles["datapoints"] = ax.plot(xs, data,
                                        color='g',
                                        linewidth=0,
                                        marker='o',
                                        ms=1,
                                        label='data points')[0]#.format(degree))[0]

        # plot f and append the plot handle
        handles["f_input"] = ax.plot(xs, ys,
                                      color='b',
                                      label="your best guess")[0]

        # plot approximation and append the plot handle
        handles["approx"] = ax.plot(xs, approx,
                                      color='orange',
                                      label="my best guess")[0]

        handles["approx"].set_visible(show_solution)

        ###############################
        # Beautify the plot some more #
        ###############################

        plt.title('Approximation of a series of data points')
        plt.xlabel('t', horizontalalignment='right', x=1)
        plt.ylabel('y', horizontalalignment='right', x=0, y=1)

        # set the z order of the axes spines
        for k, spine in ax.spines.items():
            spine.set_zorder(0)

        # set the axes locations and style
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_position(('data', 0))
        ax.spines['right'].set_color('none')

    else:
        ###################
        # Update the plot #
        ###################

        # Update the data points plot
        handles["datapoints"].set_xdata(xs)
        handles["datapoints"].set_ydata(data)

        # update the input plot
        handles["f_input"].set_xdata(xs)
        handles["f_input"].set_ydata(ys)

        # update the input plot
        handles["approx"].set_xdata(xs)
        handles["approx"].set_ydata(approx)

        # update the visibility of the Taylor expansion
        handles["approx"].set_visible(show_solution)

    # set x and y ticks, labels and limits respectively
    if ticks_on:
        xticks = [x for x in np.arange(tmin,tmax,dt).round(1)]
    else:
        xticks=[]
    xticklabels = [str(x) for x in xticks]
    
    if tmin <= 0 <= tmax:
        xticks.append(0)
        xticklabels.append("0")
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

    if ticks_on:
        yticks = [x for x in np.arange(round(ymin),round(ymax),dy).round(1)]
    else:
        yticks=[]
    yticklabels = [str(x) for x in yticks]
    if ymin <= 0 <= ymax:
        yticks.append(0)
        yticklabels.append("0")
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)

    # set the x and y limits
    ax.set_xlim([tmin-0.5, tmax+0.5])
    ax.set_ylim([ymin-0.5, ymax+0.5])
    

    # show legend
    legend_handles = [handles["datapoints"], ]
    legend_handles.append(handles['f_input'])
    if show_solution:
        legend_handles.append(handles["approx"])
    ax.legend(handles=legend_handles,
              loc='upper center',
              bbox_to_anchor=(0.5, -0.15),
              ncol=2)

    # make all changes visible
    st.session_state.mpl_fig.canvas.draw()

# with caching, the rerun-button may not work
@st.cache(suppress_st_warning=True)
def create_rnd_data(datatype,n,distribution,length):
    xs = np.random.rand(n)*length
    xs.sort()
    if distribution[0] == 'normal':
        sigma = distribution[1]
        a = random.random()
        b = random.random()
        c = random.random()
        dev = np.random.normal(0,sigma,len(xs))
        if datatype == 'constant':
            data = a + dev
        elif datatype == 'linear':
            #data = ax + b
            data = a*xs + b + dev
        elif datatype == 'quadratic':
            #data = ax^2+bx+c
            data = a*np.power(xs,2) + b*xs + c + dev
        elif datatype == 'custom':
            #data = ax^3+bx^2+cx+d
            if distribution[2]:
                f = lambda x: eval(distribution[2])
            else:
                f = lambda x: 0
            ys= [f(x) for x in xs]
            data = ys + dev
    elif distribution[0] == 'equal':
        a = random.random()
        b = random.random()
        c = random.random()
        dev = np.random.rand(n)*length
        if datatype == 'constant':
            data = a + dev
        elif datatype == 'linear':
            #data = ax + b
            data = a*xs + b + dev
        elif datatype == 'quadratic':
            #data = ax^2+bx+c
            data = a*np.power(xs,2) + b*xs + c + dev
        elif datatype == 'custom':
            #data = ax^3+bx^2+cx+d
            if distribution[2]:
                f = lambda x: eval(distribution[2])
            else:
                f = lambda x: 0
            ys= [f(x) for x in xs]
            data = ys + dev
    return xs, data

def create_new_data():
    st.session_state.create_new_data = 1
    return

# function for the button - does not work, for some reason, though
# def dont_create_new_data():
#     st.session_state.create_new_data = 0
#     return

if __name__ == '__main__':

    st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

    # create sidebar widgets

    st.sidebar.title("Advanced settings")
    
    # Data options
    
    datatype = st.sidebar.selectbox(label="data type",
                                options=('constant','linear','quadratic','custom'),
                                index=2,
                                on_change=create_new_data())
    
    if datatype == 'custom':
        f_data_input = st.sidebar.text_input(label='input the data function',
                             value='0.5*x**2 + 1*x - 2',
                             on_change=create_new_data())
        
    distribution_type = st.sidebar.select_slider('select distribution type',
                                             ['normal','equal'],
                                             on_change=create_new_data())
    
    # deviation could be dependent on the total height of the function - but this would need to be re-looped. can't be bothered...
    # if 'data' in st.session_state:
    #     height = max(st.session_state.data)
    # else:
    #     height = 100
    sigma = st.sidebar.number_input('deviation (standard deviation sigma for normal distribution, range for equal distribution)',
                                min_value=float(0),
                                max_value=float(100),
                                # value=0.1*height,
                                value=float(10),
                                step=0.01,
                                on_change=create_new_data())

    n = st.sidebar.slider(
                'number of data points',
                min_value=0,
                max_value=1000,
                value=100,
                on_change=create_new_data())
        
    
    length = st.sidebar.slider('length of interval',
                           min_value = float(1),
                           max_value = float(50),
                           value = float(10),
                           on_change=create_new_data())
    
    # Visualization Options
    st.sidebar.markdown("Visualization Options")

    # Good for in-classroom use
    qr = st.sidebar.checkbox(label="Display QR Code", value=False)

    # for now, I will assume matplotlib always works and we dont need the Altair backend
    backend = 'Matplotlib' #st.sidebar.selectbox(label="Backend", options=('Matplotlib', 'Altair'), index=0)

    # Create main page widgets

    tcol1, tcol2 = st.columns(2)

    with tcol1:
        st.title('Approximated Data Points')
    with tcol2:
        if qr:
            st.markdown('## <img src="https://api.qrserver.com/v1/create-qr-code/?size=150x150&data='
                        'https://share.streamlit.io/PhiSpel/spielbeispiel-interpolation/main" width="200"/>',
                        unsafe_allow_html=True)

    # prepare matplotlib plot
    def clear_figure():
        del st.session_state['mpl_fig']
        del st.session_state['handles']
        
    xkcd = st.sidebar.checkbox("use xkcd-style",
                               value=True,
                               on_change=clear_figure)

    ticks_on = st.sidebar.checkbox("show xticks and yticks",
                                   value=True,
                                   on_change=clear_figure)
    
    f_input = st.text_input(label='input your guessed function',
                             value='0.2*x**2 + 1*x - 2')#,
                             #on_change=dont_create_new_data())
    
    col1,col2,col3 = st.columns([1,1,2])
    with col1:
        show_solution = st.checkbox("show 'my' result",
                                    value=False)
    if show_solution:
        with col2:
            approxtype = st.selectbox(label='approximation type',
                                          options=('constant','linear','quadratic','cubic'),
                                          index=1)
    else: approxtype = 'constant'
    
    with col3:
        st.markdown('''If you want to create new random data, change the advanced settings (top-left), or clear the cache (press 'c')''')
    
    # the button function does not work, has something to do with cashing, I think...
    # with col4:
    #     st.button(label='create new data',on_click=create_new_data())
    
    if datatype == 'custom':
        distribution = [distribution_type,sigma,f_data_input]
    else:
        distribution = [distribution_type,sigma]
    
    if 'create_new_data' not in st.session_state:
        st.session_state.create_new_data = 1
    if 'xs' not in st.session_state:
        st.session_state.xs = []
    if 'data' not in st.session_state:
        st.session_state.data = []
        
    if st.session_state.create_new_data:
        st.session_state.xs,st.session_state.data = create_rnd_data(datatype,n,distribution,length)
        st.session_state.create_new_data = 0
    
    # update the data
    approx = update_approx(st.session_state.xs,st.session_state.data,datatype,approxtype)
    
    if show_solution:
        solution_description = r'''my best guess: $f(x)\approx '''
        factors=np.round(approx,2)
        deg = len(factors)-1
        for i in range(0,deg+1):
            if (factors[i] > 0) & (i > 0):
                solution_description+='+'
            if deg-i > 1:
                solution_description+=str(factors[i]) + 'x^' + str(deg-i)
            elif deg-i == 1:
                solution_description+=str(factors[i]) + 'x'
            elif deg-i == 0:
                solution_description+=str(factors[i])
        solution_description+='''$'''
        st.markdown(solution_description)
    
    if xkcd:
        # set rc parameters to xkcd style
        plt.xkcd()
    else:
        # reset rc parameters to default
        plt.rcdefaults()

    # initialize the Matplotlib figure and initialize an empty dict of plot handles
    if 'mpl_fig' not in st.session_state:
        st.session_state.mpl_fig = plt.figure(figsize=(8, 3))
        st.session_state.mpl_fig.add_axes([0., 0., 1., 1.])

    if 'handles' not in st.session_state:
        st.session_state.handles = {}

    # update plot
    update_plot(st.session_state.xs, st.session_state.data, approx, f_input, show_solution, ticks_on)
    st.pyplot(st.session_state.mpl_fig)