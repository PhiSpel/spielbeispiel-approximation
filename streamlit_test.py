import streamlit as st

import numpy as np
import math

import matplotlib.pyplot as plt
import matplotlib.font_manager

import random

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

# make sure the humor sans font is found. This only needs to be done once
# on a system, but it is done here at start up for usage on share.streamlit.io.
#matplotlib.font_manager.findfont('Humor Sans', rebuild_if_missing=True)

#############################################
# Define the function that updates the plot #
#############################################

#@st.cache(suppress_st_warning=True)
def update_approx():
    st.session_state.approx+=2
    return

# To Do: Why does caching update_plot hang?
# @st.cache(suppress_st_warning=True)
def update_plot():
    xs, data = st.session_state.xs, st.session_state.data
    
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

    # show legend
    legend_handles = [handles["datapoints"], ]
    ax.legend(handles=legend_handles,
              loc='upper center',
              bbox_to_anchor=(0.5, -0.15),
              ncol=2)

    # make all changes visible
    st.session_state.mpl_fig.canvas.draw()

#@st.cache    
def create_new_points():
    xs = np.random.rand(100)*10
    xs.sort()
    dev = np.random.rand(100)*2
    st.session_state.data = 2*xs + dev
    st.session_state.xs = xs
    return

def request_new_data():
    st.session_state.create_new_data = 1
    return

def clear_figure():
    del st.session_state['mpl_fig']
    del st.session_state['handles']

if 'create_new_data' not in st.session_state:
    st.session_state.create_new_data = 1
    st.write('Hi! I had to recreate create_new_data...')
if 'approx' not in st.session_state:
    st.session_state.approx = 0
if 'xs' not in st.session_state:
    st.session_state.xs = []
if 'data' not in st.session_state:
    st.session_state.data = []
       
#if __name__ == '__main__':
        
# create sidebar widgets

st.sidebar.title("Advanced settings")

# Create main page widgets

st.title('Approximated Data Points')

st.write('new_data:',st.session_state.create_new_data,'new_points:',st.session_state.create_new_points)
       

        
col1,col2 = st.columns(2)
with col1:
    st.session_state.f_input = st.text_input(label='input your guessed function',
                         value='0.2*x**2 + 1*x - 2',
                         help='''type e.g. 'math.sin(x)' to generate a sine function''')#,
                         #on_change=dont_create_new_data())
                         
with col2:
    st.button(label='hi',
              on_click=request_new_data)
    st.write(st.session_state.approx)
    st.write(st.session_state.create_new_data)

col1,col2,col3 = st.columns([1,1,2])
with col1:
    st.session_state.show_solution = st.checkbox("show the result of np.polyfit",
                                value=False,
                                on_change=clear_figure)
if st.session_state.show_solution:
    with col2:
        st.session_state.approxtype = st.selectbox(label='approximation type',
                                      options=('constant','linear','quadratic','cubic'),
                                      index=2,
                                      on_change=request_new_data())
else: st.session_state.approxtype = 'constant'
st.write(st.session_state.create_new_data)

# the button function does not work, has something to do with cashing, I think...
# with col4:
#     st.button(label='create new data',on_click=create_new_data())

# if datatype == 'custom':
#     distribution = [distribution_type,sigma,f_data_input]
# else:
#     distribution = [distribution_type,sigma]
       
if (st.session_state.create_new_data==1):# | (st.session_state.create_new_factors==1):
    create_new_points()
    update_approx()
    st.session_state.create_new_data = 0
    st.write(st.session_state.create_new_data)


plt.rcdefaults()

# initialize the Matplotlib figure and initialize an empty dict of plot handles
if 'mpl_fig' not in st.session_state:
    st.session_state.mpl_fig = plt.figure(figsize=(8, 3))
    st.session_state.mpl_fig.add_axes([0., 0., 1., 1.])

if 'handles' not in st.session_state:
    st.session_state.handles = {}

# update plot
update_plot()
st.pyplot(st.session_state.mpl_fig)