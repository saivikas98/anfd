import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title='OMCA ANFD Simulation',layout='wide')
st.title('OMCA ANFD | Liquid, Powder, Tilt and Agitator Simulation')
st.caption('GA basis: ID 350 mm, nominal shell capacity 52 L')
R=175.0; AREA=math.pi*R**2; H=52e6/AREA; P=H/2; G=9810.0
NR,NA=42,120
r=(np.arange(NR)+.5)*R/NR; a=(np.arange(NA)+.5)*2*np.pi/NA
RR,AA=np.meshgrid(r,a,indexing='ij'); YY=RR*np.sin(AA); W=RR*(R/NR)*(2*np.pi/NA)

def rot(x,y,z,t):
 c,s=math.cos(t),math.sin(t); x=np.asarray(x); y=np.asarray(y); z=np.asarray(z)
 return x,y*c-(z-P)*s,y*s+(z-P)*c+P

def powder_grid(vol_l,t,aor):
 slope=math.tan(math.copysign(min(abs(t),math.radians(aor)),t)); lo,hi=-H,2*H; tgt=vol_l*1e6
 for _ in range(48):
  m=(lo+hi)/2; h=np.clip(m-YY*slope,0,H)
  if np.sum(h*W)<tgt: lo=m
  else: hi=m
 return np.clip((lo+hi)/2-YY*slope,0,H)

def liquid_plane(y,C,t):
 c=max(abs(math.cos(t)),1e-4)
 return np.clip(P+(C-P-y*math.sin(t))/c,0,H)

def displacement(top,base,shaft_d,lift):
 shaft=np.pi*(shaft_d/2)**2*np.clip(top-np.maximum(base,lift),0,None)
 legs=2*14*16*np.clip(np.minimum(top,lift+.72*H)-np.maximum(base,lift+8),0,None)
 arms=2*(R-38)*14*16*((top>lift+8)&(base<lift+24))
 return shaft+legs+arms

def liquid_grid(vol_l,powder,t,shaft_d,lift):
 tgt=vol_l*1e6; lo,hi=P-H-2*R,P+H+2*R
 for _ in range(55):
  C=(lo+hi)/2; top=liquid_plane(YY,C,t)
  v=np.sum(np.clip(top-powder,0,None)*W)-float(np.mean(displacement(top,powder,shaft_d,lift)))
  if v<tgt: lo=C
  else: hi=C
 top=liquid_plane(YY,(lo+hi)/2,t)
 return top,float(np.mean(displacement(top,powder,shaft_d,lift)))/1e6

def surf(h,t,color,name,opacity,wave=0,phase=0):
 rr=np.linspace(0,R*.985,30); aa=np.linspace(0,2*np.pi,96); Q,A=np.meshgrid(rr,aa)
 x=Q*np.cos(A); y=Q*np.sin(A); ir=np.minimum(NR-1,(Q/R*NR).astype(int)); ia=np.mod((A/(2*np.pi)*NA).astype(int),NA)
 z=h[ir,ia]+wave*(Q/R)*np.cos(A-phase); z=np.clip(z,0,H); x,y,z=rot(x,y,z,t)
 return go.Surface(x=x,y=y,z=z,colorscale=[[0,color],[1,color]],opacity=opacity,showscale=False,name=name,hoverinfo='name')

def liquid_wall(top,base,t,opacity):
 aa=np.linspace(0,2*np.pi,120); f=np.linspace(0,1,28); A,F=np.meshgrid(aa,f)
 x=R*.982*np.cos(A); y=R*.982*np.sin(A); ia=np.mod((A/(2*np.pi)*NA).astype(int),NA); ir=np.full_like(ia,NR-1)
 z0=base[ir,ia]; z1=np.maximum(z0,top[ir,ia]); z=z0+F*(z1-z0); x,y,z=rot(x,y,z,t)
 return go.Surface(x=x,y=y,z=z,colorscale=[[0,'rgb(0,82,190)'],[1,'rgb(55,185,248)']],opacity=opacity,showscale=False,name='Full liquid volume',hoverinfo='skip')

def line(x,y,z,t,name,show=True,width=10):
 x,y,z=rot(x,y,z,t)
 return go.Scatter3d(x=x,y=y,z=z,mode='lines',line=dict(color='#c8d0d4',width=width),name=name,showlegend=show)

def model(q_liq,q_pow,t,sd,lift,aor,trans,rpm,time,show,dem,wave=0):
 ph=powder_grid(q_pow,t,aor); lh,disp=liquid_grid(q_liq,ph,t,sd,lift); out=[]
 # shell and dished top
 if show['shell']:
  aa=np.linspace(0,2*np.pi,96); zz=np.linspace(0,H,24); A,Z=np.meshgrid(aa,zz); x=R*np.cos(A); y=R*np.sin(A); x,y,z=rot(x,y,Z,t)
  out.append(go.Surface(x=x,y=y,z=z,colorscale=[[0,'rgb(145,155,162)'],[1,'rgb(232,236,238)']],opacity=.16,showscale=False,name='SS shell'))
 if show['powder']: out.append(surf(ph,t,'rgb(213,167,34)','Powder bed',.92))
 if show['liquid']:
  op=max(.22,1-trans/100); out.append(liquid_wall(lh,ph,t,op*.75)); out.append(surf(lh,t,'rgb(35,165,245)','Liquid surface',max(.35,op),abs(wave),0 if wave>=0 else math.pi))
 rotation=rpm*2*math.pi*time/60
 if show['agitator']:
  out.append(line([0,0],[0,0],[lift,H+150+lift],t,'Shaft'))
  c,s=math.cos(rotation),math.sin(rotation)
  for side in (-1,1):
   x=np.array([0,side*(R-3),side*(R-3),side*38]); y=np.zeros(4); z=np.array([lift+.72*H,lift+.72*H,lift+8,lift+8])
   out.append(line(x*c-y*s,x*s+y*c,z,t,'Anchor' if side==1 else '',side==1,12))
 if show['particles']:
  n=dem['count']; idx=np.arange(n); q=(idx+.5)/n; rad=R*.94*np.sqrt(q); ang=idx*2.399963+rotation*dem['drag']*(.25+.75*q)
  excess=max(0,abs(math.degrees(t))-aor); drift=-math.copysign(1,t if t else 1)*min(1,excess/18)*(1-dem['mu']*.5)*R*.28
  x=rad*np.cos(ang); y=np.clip(rad*np.sin(ang)+drift,-R+2,R-2); rr=np.hypot(x,y); sc=np.minimum(1,(R-3)/np.maximum(rr,1e-6)); x*=sc; y*=sc
  ir=np.minimum(NR-1,(np.hypot(x,y)/R*NR).astype(int)); ia=np.mod((np.arctan2(y,x)/(2*np.pi)*NA).astype(int),NA); bed=ph[ir,ia]
  z=np.clip(bed*(.07+.86*((idx%29)/29))-dem['e']*np.abs(np.sin(idx*.73+time))*2,1,np.maximum(1,bed-1)); x,y,z=rot(x,y,z,t)
  out.append(go.Scatter3d(x=x,y=y,z=z,mode='markers',marker=dict(size=2.2,color='#e2aa1d',opacity=.82),name='DEM-style particles'))
 return out,ph,lh,disp

with st.sidebar:
 st.header('Process inputs')
 liq=st.number_input('Liquid quantity (L)',0.0,52.0,10.0,.1); pm=st.number_input('Powder quantity (kg)',0.0,100.0,1.8,.1); bd=st.number_input('Bulk density (kg/L)',.01,5.0,.22,.01)
 trans=st.slider('Liquid transparency (%)',0,95,42); sd=st.slider('Shaft diameter (mm)',30.0,100.0,68.0); lift=st.slider('Agitator lift (mm)',3.0,68.0,3.0); rpm=st.slider('Agitator speed (rpm)',0.0,30.0,8.0,.5); target=st.slider('Tilt angle (deg)',0.0,60.0,30.0); aor=st.slider('Angle of repose (deg)',15.0,50.0,30.0)
 st.header('DEM-style controls'); mu=st.slider('Friction coefficient',.1,.9,.45,.05); rest=st.slider('Restitution',0.0,.9,.25,.05); drag=st.slider('Agitator drag',0.0,1.0,.65,.05); count=st.slider('Particles',150,3000,900,50)
 st.header('Display'); show={'liquid':st.checkbox('Full liquid volume',True),'powder':st.checkbox('Powder bed',True),'particles':st.checkbox('Powder particles',True),'shell':st.checkbox('ANFD shell',True),'agitator':st.checkbox('Agitator',True)}; supports=st.checkbox('External supports',False); axes=st.checkbox('Axes/grid',False)
 st.header('Dynamics'); duration=st.slider('Tilt duration (s)',1.0,15.0,5.0,.5); settle=st.slider('Settling duration (s)',0.0,10.0,4.0,.5); zeta=st.slider('Slosh damping',.01,.5,.1,.01)

pow_l=pm/bd; th=math.radians(target); dem={'mu':mu,'e':rest,'drag':drag,'count':count}
# smooth tilt and damped slosh
n=48; t1=np.linspace(0,duration,n); u=t1/duration; ang=th*(10*u**3-15*u**4+6*u**5); h0=max(1,liq*1e6/AREA); om=math.sqrt(G*(1.841/R)*math.tanh((1.841/R)*h0)); q=R*.07*np.sin(om*t1)*np.sin(np.pi*u)*np.exp(-zeta*om*t1)
dt=t1[1]-t1[0]; ts=np.arange(dt,settle+dt/2,dt); qset=q[-1]*np.exp(-zeta*om*ts)*np.cos(om*ts); times=np.r_[t1,duration+ts]; angles=np.r_[ang,np.full(len(ts),th)]; modes=np.r_[q,qset]
final,ph,lh,disp=model(liq,pow_l,th,sd,lift,aor,trans,rpm,duration,show,dem)
fig=go.Figure(final)
if supports:
 for x,y in [(-550,-550),(-550,550),(550,-550),(550,550)]: fig.add_trace(go.Scatter3d(x=[x,x*.55],y=[y,y*.55],z=[-650,120],mode='lines',line=dict(color='#78848a',width=16),showlegend=False))
SIM_COLUMNS=['Time_s','Tilt_deg','Liquid_low_mm','Liquid_high_mm','Powder_low_mm','Powder_high_mm','Agitator_displacement_L','Slosh_modal_mm']
sim_outputs=[]
for T,A,Q in zip(times,angles,modes):
 mv,p1,l1,d1=model(liq,pow_l,float(A),sd,lift,aor,trans,rpm,float(T),show,dem,float(Q)); sim_outputs.append((T,A,Q,mv,p1,l1,d1))
frames=[go.Frame(data=o[3],name=str(i),traces=list(range(len(o[3])))) for i,o in enumerate(sim_outputs)]
hist=pd.DataFrame([{'Time_s':o[0],'Tilt_deg':math.degrees(o[1]),'Liquid_low_mm':float(np.max(o[5])),'Liquid_high_mm':float(np.min(o[5])),'Powder_low_mm':float(np.max(o[4])),'Powder_high_mm':float(np.min(o[4])),'Agitator_displacement_L':o[6],'Slosh_modal_mm':o[2]} for o in sim_outputs],columns=SIM_COLUMNS)
fig.frames=frames
fig.update_layout(height=760,margin=dict(l=0,r=0,t=35,b=0),scene=dict(aspectmode='data',xaxis=dict(visible=axes),yaxis=dict(visible=axes),zaxis=dict(visible=axes),bgcolor='rgb(247,249,251)'),updatemenus=[dict(type='buttons',buttons=[dict(label='▶ Animate',method='animate',args=[None,dict(frame=dict(duration=90,redraw=True),fromcurrent=True)]),dict(label='❚❚ Pause',method='animate',args=[[None],dict(frame=dict(duration=0),mode='immediate')])])])
st.plotly_chart(fig,use_container_width=True)
st.subheader('Liquid and powder level histories')
lf=go.Figure(); lf.add_trace(go.Scatter(x=hist.Time_s,y=hist.Liquid_low_mm,name='Liquid low')); lf.add_trace(go.Scatter(x=hist.Time_s,y=hist.Liquid_high_mm,name='Liquid high')); lf.add_trace(go.Scatter(x=hist.Time_s,y=hist.Powder_low_mm,name='Powder low')); lf.add_trace(go.Scatter(x=hist.Time_s,y=hist.Powder_high_mm,name='Powder high')); lf.update_layout(xaxis_title='Time (s)',yaxis_title='Level (mm)',hovermode='x unified'); st.plotly_chart(lf,use_container_width=True)
st.download_button('Download history CSV',hist.to_csv(index=False).encode(),'OMCA_history.csv','text/csv')
st.caption(f'Final immersed shaft/anchor displacement: {disp:.2f} L | Slosh natural period: {2*math.pi/om:.2f} s')
st.warning('Reduced-order engineering visualization; quantitative design requires calibrated DEM and CFD/VOF.')
