import math
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title='OMCA ANFD Liquid Level', layout='wide')
st.title('OMCA ANFD | GA-Based Liquid-Level Simulation')
st.caption('GA drawing 5332 Rev.2 | PFD shell ID 350 mm | nominal shell capacity 52 L')

ID_MM = 350.0
R_MM = ID_MM / 2
AREA_MM2 = math.pi * R_MM**2
NOMINAL_CAPACITY_L = 52.0
CYL_EQUIV_HEIGHT_MM = NOMINAL_CAPACITY_L * 1_000_000 / AREA_MM2

with st.sidebar:
    st.header('Simulation inputs')
    qty = st.number_input('Liquid quantity (L)', 0.0, NOMINAL_CAPACITY_L, 10.0, 0.1)
    qty_slider = st.slider('Liquid quantity slider (L)', 0.0, NOMINAL_CAPACITY_L, float(qty), 0.1)
    if abs(qty_slider-qty) > 1e-9:
        qty = qty_slider
    density = st.number_input('Liquid density (kg/L)', 0.001, 25.0, 1.000, 0.001)
    transparency = st.slider('Liquid transparency (%)', 0, 95, 45)
    tilt_deg = st.slider('Vessel tilt angle (deg)', 0.0, 60.0, 0.0, 1.0)
    show_nozzles = st.checkbox('Show all GA nozzle tags', True)
    show_supports = st.checkbox('Show supports and trunnions', True)

level_mm = qty * 1_000_000 / AREA_MM2
fill_pct = 100 * qty / NOMINAL_CAPACITY_L if NOMINAL_CAPACITY_L else 0
weight_kg = qty * density
free_l = NOMINAL_CAPACITY_L - qty

c1,c2,c3,c4 = st.columns(4)
c1.metric('Quantity', f'{qty:.2f} L')
c2.metric('Equivalent level', f'{level_mm:.1f} mm')
c3.metric('Nominal fill', f'{fill_pct:.1f}%')
c4.metric('Liquid weight', f'{weight_kg:.2f} kg')

# Vessel-centered engineering visualization, dimensions in mm.
fig = go.Figure()
nu, nz = 72, 28
u_vals = np.linspace(0, 2*np.pi, nu)
z_vals = np.linspace(0, CYL_EQUIV_HEIGHT_MM, nz)
U,Z = np.meshgrid(u_vals,z_vals)
X = R_MM*np.cos(U); Y=R_MM*np.sin(U)
fig.add_trace(go.Surface(x=X,y=Y,z=Z, surfacecolor=np.ones_like(Z), colorscale=[[0,'rgb(165,174,180)'],[1,'rgb(225,230,233)']], opacity=.20, showscale=False, name='SS shell', hoverinfo='skip'))
# top and bottom shell rims
for z, name in [(0,'Filter plate'),(CYL_EQUIV_HEIGHT_MM,'Top flange')]:
    fig.add_trace(go.Scatter3d(x=R_MM*np.cos(u_vals),y=R_MM*np.sin(u_vals),z=np.full(nu,z),mode='lines',line=dict(color='#cbd4d9',width=7),name=name,hoverinfo='name'))
# Liquid as a smooth cylinder up to calculated level. For tilt, free plane is horizontal in world; simple clipped samples.
alpha=max(.06,1-transparency/100)
if qty > 0:
    theta=math.radians(tilt_deg)
    # Render a smooth level surface with a gravity-horizontal plane visualized in vessel coordinates.
    rr=np.linspace(0,R_MM*.985,30); aa=np.linspace(0,2*np.pi,90)
    RR,AA=np.meshgrid(rr,aa)
    XL=RR*np.cos(AA); YL=RR*np.sin(AA)
    ZL=np.clip(level_mm - YL*math.tan(theta),0,CYL_EQUIV_HEIGHT_MM)
    fig.add_trace(go.Surface(x=XL,y=YL,z=ZL,colorscale=[[0,'rgb(15,125,230)'],[1,'rgb(80,200,255)']],opacity=max(.20,alpha),showscale=False,name='Liquid free surface',hovertemplate='Liquid surface<br>z=%{z:.1f} mm<extra></extra>'))
    # volume wall
    ZW=np.linspace(0,max(level_mm,1),18); UW=np.linspace(0,2*np.pi,90); UW,ZW=np.meshgrid(UW,ZW)
    XW=R_MM*.98*np.cos(UW); YW=R_MM*.98*np.sin(UW)
    fig.add_trace(go.Surface(x=XW,y=YW,z=ZW,colorscale=[[0,'rgb(0,80,180)'],[1,'rgb(45,175,245)']],opacity=max(.12,alpha*.7),showscale=False,name='Liquid volume',hoverinfo='skip'))
# agitator shaft and anchor frame
fig.add_trace(go.Scatter3d(x=[0,0],y=[0,0],z=[25,CYL_EQUIV_HEIGHT_MM+180],mode='lines',line=dict(color='#d7dde0',width=10),name='Agitator shaft'))
for side in [-1,1]:
    fig.add_trace(go.Scatter3d(x=[0,side*(R_MM-3),side*(R_MM-3),side*35],y=[0,0,0,0],z=[CYL_EQUIV_HEIGHT_MM*.75,CYL_EQUIV_HEIGHT_MM*.72,8,8],mode='lines',line=dict(color='#bdc5c9',width=12),name='Anchor agitator' if side==1 else None,showlegend=side==1))
# GA supports/trunnions
if show_supports:
    for x,y in [(-550,-550),(-550,550),(550,-550),(550,550)]:
        fig.add_trace(go.Scatter3d(x=[x,x*.55],y=[y,y*.55],z=[-650,120],mode='lines',line=dict(color='#77838a',width=16),name='Supports',showlegend=False))
    fig.add_trace(go.Scatter3d(x=[-350,350],y=[0,0],z=[CYL_EQUIV_HEIGHT_MM*.52]*2,mode='lines+markers',line=dict(color='#89949a',width=14),marker=dict(size=7),name='Trunnions'))
# nozzles from schedule
csv_path=Path(__file__).with_name('OMCA_ANFD_GA_Nozzle_Schedule.csv')
noz=pd.read_csv(csv_path)
if show_nozzles:
    for _,r in noz.iterrows():
        az=math.radians(float(r.azimuth_deg)); elev=math.radians(float(r.elevation_deg)); z=float(r.z_mm)-1100
        x0=R_MM*math.cos(az); y0=R_MM*math.sin(az)
        L=float(r.length_mm)
        x1=x0+L*math.cos(elev)*math.cos(az); y1=y0+L*math.cos(elev)*math.sin(az); z1=z+L*math.sin(elev)
        fig.add_trace(go.Scatter3d(x=[x0,x1],y=[y0,y1],z=[z,z1],mode='lines+text',text=['',str(r.tag)],textposition='top center',line=dict(color='#6fc2e8',width=4),showlegend=False,hovertemplate=f"Tag {r.tag}<br>Azimuth {r.azimuth_deg} deg<br>Elevation {r.elevation_deg} deg<extra></extra>"))
fig.update_layout(height=720,margin=dict(l=0,r=0,t=35,b=0),scene=dict(aspectmode='data',xaxis_title='X (mm)',yaxis_title='Y (mm)',zaxis_title='Z (mm)',camera=dict(eye=dict(x=1.7,y=1.7,z=1.25))),legend=dict(orientation='h'))
st.plotly_chart(fig,use_container_width=True)

st.subheader('Quantity-to-level table')
presets=sorted(set([0,1,2,5,10,15,20,25,30,35,40,45,50,52,round(qty,2)]))
table=pd.DataFrame({'Quantity (L)':presets})
table['Equivalent level (mm)']=table['Quantity (L)']*1_000_000/AREA_MM2
table['Nominal fill (%)']=table['Quantity (L)']/NOMINAL_CAPACITY_L*100
table['Weight at entered density (kg)']=table['Quantity (L)']*density
table['Free volume (L)']=NOMINAL_CAPACITY_L-table['Quantity (L)']
st.dataframe(table.style.format({'Equivalent level (mm)':'{:.1f}','Nominal fill (%)':'{:.1f}','Weight at entered density (kg)':'{:.2f}','Free volume (L)':'{:.2f}'}),use_container_width=True,hide_index=True)

st.info('Calculation basis: 350 mm internal diameter and 52 L nominal shell capacity from the GA. Level is a cylindrical-equivalent height. Exact local level near the dished/filter head requires the internal head profile or a volume calibration table.')
with open(Path(__file__).with_name('OMCA_ANFD_GA_Model_All_Nozzles_Supports.glb'),'rb') as f:
    st.download_button('Download GA-based GLB model',f,file_name='OMCA_ANFD_GA_Model_All_Nozzles_Supports.glb')
