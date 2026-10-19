import math
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title='OMCA ANFD Multiphase Simulation',layout='wide')
st.title('OMCA ANFD | Coupled Liquid, Powder and Agitator Simulation')
st.caption('GA 5332 Rev.2 | ID 350 mm | nominal shell capacity 52 L')
R=175.0; AREA=math.pi*R**2; H=52e6/AREA; PIVOT=H/2; G=9810.0
NR,NA=48,120
r=(np.arange(NR)+.5)/NR*R; a=(np.arange(NA)+.5)/NA*2*np.pi
RR,AA=np.meshgrid(r,a,indexing='ij'); XX=RR*np.cos(AA); YY=RR*np.sin(AA)
W=RR*(R/NR)*(2*np.pi/NA)

def rotate(x,y,z,t):
 c,s=math.cos(t),math.sin(t); y=np.asarray(y); z=np.asarray(z)
 return np.asarray(x),y*c-(z-PIVOT)*s,y*s+(z-PIVOT)*c+PIVOT

def liquid_plane(y,C,t):
 c=max(abs(math.cos(t)),1e-5)
 return np.clip(PIVOT+(C-PIVOT-y*math.sin(t))/c,0,H)

def powder_bed(volume_l,t,aor):
 slope=math.tan(math.copysign(min(abs(t),math.radians(aor)),t)); target=volume_l*1e6
 lo,hi=-H,2*H
 for _ in range(55):
  m=(lo+hi)/2; hp=np.clip(m-YY*slope,0,H)
  if np.sum(hp*W)<target: lo=m
  else: hi=m
 return np.clip((lo+hi)/2-YY*slope,0,H),math.degrees(math.atan(slope))

def solid_displacement(top,base,shaft_d,lift):
 shaft=np.pi*(shaft_d/2)**2*np.clip(top-np.maximum(base,lift),0,None)
 legA=14*16; lo=lift+8; hi=lift+H*.72
 legs=2*legA*np.clip(np.minimum(top,hi)-np.maximum(base,lo),0,None)
 low=2*(R-38)*14*16*((top>lift+8)&(base<lift+24))
 up=2*(R-3)*14*16*((top>lift+H*.70)&(base<lift+H*.74))
 return shaft+legs+low+up

def solve_liquid(volume_l,powder,theta,shaft_d,lift):
 target=volume_l*1e6; lo=PIVOT-H-2*R; hi=PIVOT+H+2*R
 for _ in range(60):
  C=(lo+hi)/2; top=liquid_plane(YY,C,theta)
  raw=np.sum(np.clip(top-powder,0,None)*W)
  disp=float(np.mean(solid_displacement(top,powder,shaft_d,lift)))
  if raw-disp<target: lo=C
  else: hi=C
 C=(lo+hi)/2; top=liquid_plane(YY,C,theta)
 disp=float(np.mean(solid_displacement(top,powder,shaft_d,lift)))/1e6
 return top,disp

def slosh_response(volume_l,target,duration,zeta,gain,n=52):
 t=np.linspace(0,duration,n); u=t/duration; ang=target*(10*u**3-15*u**4+6*u**5)
 acc=target*(60*u-180*u**2+120*u**3)/duration**2
 h=max(1,volume_l*1e6/AREA); k=1.841/R; om=math.sqrt(G*k*math.tanh(k*h))
 q=np.zeros(n); v=np.zeros(n); dt=t[1]-t[0]
 for i in range(n-1):
  f=lambda Q,V,A:-2*zeta*om*V-om**2*Q-gain*PIVOT*A
  am=.5*(acc[i]+acc[i+1]); k1q=v[i]; k1v=f(q[i],v[i],acc[i])
  k2q=v[i]+dt*k1v/2; k2v=f(q[i]+dt*k1q/2,v[i]+dt*k1v/2,am)
  k3q=v[i]+dt*k2v/2; k3v=f(q[i]+dt*k2q/2,v[i]+dt*k2v/2,am)
  k4q=v[i]+dt*k3v; k4v=f(q[i]+dt*k3q,v[i]+dt*k3v,acc[i+1])
  q[i+1]=q[i]+dt*(k1q+2*k2q+2*k3q+k4q)/6
  v[i+1]=v[i]+dt*(k1v+2*k2v+2*k3v+k4v)/6
 return t,ang,q,om

def surface(h,t,color,name,opacity,wave=0,phase=0):
 rr=np.linspace(0,R*.985,32); aa=np.linspace(0,2*np.pi,100); Q,A=np.meshgrid(rr,aa)
 x=Q*np.cos(A); y=Q*np.sin(A); ir=np.minimum(NR-1,(Q/R*NR).astype(int)); ia=np.mod((A/(2*np.pi)*NA).astype(int),NA)
 z=h[ir,ia]
 if wave: z=np.clip(z+wave*(Q/R)*np.cos(A-phase),0,H)
 x,y,z=rotate(x,y,z,t)
 return go.Surface(x=x,y=y,z=z,colorscale=[[0,color],[1,color]],opacity=opacity,showscale=False,name=name,hoverinfo='name')

def liquid_volume_wall(top,base,t,opacity):
 # Closed outer liquid boundary between powder bed and liquid free surface.
 aa=np.linspace(0,2*np.pi,140); frac=np.linspace(0,1,34); A,F=np.meshgrid(aa,frac)
 x=R*.982*np.cos(A); y=R*.982*np.sin(A)
 ia=np.mod((A/(2*np.pi)*NA).astype(int),NA); ir=np.full_like(ia,NR-1)
 z0=base[ir,ia]; z1=np.maximum(z0,top[ir,ia]); z=z0+F*(z1-z0)
 x,y,z=rotate(x,y,z,t)
 return go.Surface(x=x,y=y,z=z,colorscale=[[0,'rgb(0,82,190)'],[1,'rgb(58,184,246)']],opacity=max(.20,opacity),showscale=False,name='Full liquid volume',hoverinfo='skip')

def annular_ring(z,t,r_outer,r_inner,color,name):
 aa=np.linspace(0,2*np.pi,120); rr=np.array([r_inner,r_outer]); RR,AA=np.meshgrid(rr,aa)
 x=RR*np.cos(AA); y=RR*np.sin(AA); zz=np.full_like(x,z); x,y,zz=rotate(x,y,zz,t)
 return go.Surface(x=x,y=y,z=zz,colorscale=[[0,color],[1,color]],opacity=.75,showscale=False,name=name,hoverinfo='name')

def shell(t):
 parts=[]
 aa=np.linspace(0,2*np.pi,110); zz=np.linspace(0,H,28); A,Z=np.meshgrid(aa,zz)
 x=R*np.cos(A); y=R*np.sin(A); x,y,z=rotate(x,y,Z,t)
 parts.append(go.Surface(x=x,y=y,z=z,colorscale=[[0,'rgb(145,155,162)'],[1,'rgb(229,234,237)']],opacity=.16,showscale=False,name='PFD shell ID350',hoverinfo='skip'))
 # Shallow dished top closure matching the GA vessel profile.
 rr=np.linspace(0,R,28); aa2=np.linspace(0,2*np.pi,110); RR,A2=np.meshgrid(rr,aa2)
 xh=RR*np.cos(A2); yh=RR*np.sin(A2); zh=H+105*(1-(RR/R)**2)
 xh,yh,zh=rotate(xh,yh,zh,t)
 parts.append(go.Surface(x=xh,y=yh,z=zh,colorscale=[[0,'rgb(175,183,188)'],[1,'rgb(236,239,241)']],opacity=.20,showscale=False,name='Dished top head',hoverinfo='skip'))
 parts.append(annular_ring(H,t,R*1.08,R*.91,'rgb(165,174,180)','Main top flange'))
 parts.append(annular_ring(0,t,R*1.09,0,'rgb(135,145,151)','20 micron filter plate'))
 # Lower clamp/filter-head rim.
 parts.append(annular_ring(-18,t,R*1.12,R*.84,'rgb(115,126,133)','Filter-head clamp'))
 return parts

def ln(x,y,z,t,name,show=True,w=10):
 x,y,z=rotate(x,y,z,t)
 return go.Scatter3d(x=x,y=y,z=z,mode='lines',line=dict(color='#cbd3d7',width=w),name=name,showlegend=show,hoverinfo='name')

def traces(liq,pow_l,t,sd,lift,aor,trans,rpm,time,wave=0,dem_mu=.45,dem_e=.25,dem_drag=.65,particle_mm=1.0,particle_count=900):
 ph,_=powder_bed(pow_l,t,aor); lh,disp=solve_liquid(liq,ph,t,sd,lift)
 alpha=max(.22,1-trans/100)
 out=[]
 if show_shell: out.extend(shell(t))
 if show_powder_surface: out.append(surface(ph,t,'rgb(210,163,36)','Powder bed',.92))
 if show_liquid_volume:
  out.append(liquid_volume_wall(lh,ph,t,alpha*.72))
  out.append(surface(lh,t,'rgb(35,165,245)','Liquid free surface',max(.32,alpha),abs(wave),0 if wave>=0 else math.pi))
 if show_agitator: out.append(ln([0,0],[0,0],[lift,H+160+lift],t,'Agitator shaft'))
 rot=rpm*2*math.pi/60*time; c,s=math.cos(rot),math.sin(rot)
 if show_agitator:
  for side in (-1,1):
   x=np.array([0,side*(R-3),side*(R-3),side*38]); y=np.zeros(4); z=np.array([lift+H*.72,lift+H*.72,lift+8,lift+8])
   out.append(ln(x*c-y*s,x*s+y*c,z,t,'Rotating anchor' if side==1 else '',side==1,12))
 # DEM-style representative particles: gravity, wall confinement, frictional slip,
 # agitator drag, centrifugal migration, collision jitter and repose-angle avalanching.
 n=int(np.clip(particle_count,150,3000)); idx=np.arange(n); q=(idx+.5)/n
 base_rad=R*.94*np.sqrt(q); base_ang=idx*2.399963
 omega=rpm*2*math.pi/60; slip=np.clip(dem_drag*(1-dem_mu*.45),.05,1.0)
 # Tangential entrainment is strongest near the anchor wall legs and larger radii.
 ang=base_ang+omega*time*slip*(.25+.75*q)
 # Centrifugal migration is bounded by the vessel wall. Units scaled for visualization.
 centrif=np.clip((omega**2)*base_rad/9810,0,.22)
 radial_mix=(4+10*slip)*np.sin(base_ang*2.7+omega*time*1.8)*(q*(1-q))
 rad=np.clip(base_rad*(1+.18*centrif)+radial_mix,particle_mm*.55,R-particle_mm*.55-3)
 # Gravity-low-side drift activates when vessel tilt exceeds the material repose angle.
 excess=max(0.0,abs(math.degrees(t))-aor); avalanche=min(1.0,excess/18.0)
 low_dir=-math.copysign(1.0,t if abs(t)>1e-9 else 1.0)
 ydrift=low_dir*avalanche*(1-dem_mu*.55)*R*.30*(1-np.exp(-time*1.4))
 xp=rad*np.cos(ang); yp=np.clip(rad*np.sin(ang)+ydrift,-R+particle_mm,R-particle_mm)
 # Enforce circular wall after downslope drift.
 rr=np.hypot(xp,yp); scale=np.minimum(1,(R-particle_mm-3)/np.maximum(rr,1e-9)); xp*=scale; yp*=scale
 ir=np.minimum(NR-1,(np.hypot(xp,yp)/R*NR).astype(int)); ia=np.mod((np.arctan2(yp,xp)/(2*np.pi)*NA).astype(int),NA)
 bed=ph[ir,ia]
 # Stratified settling plus deterministic collision/restitution bounce.
 layer=(idx%31)/31; collision=particle_mm*(.15+.65*dem_e)*np.abs(np.sin(idx*.731+time*(1.5+omega)))
 zp=np.maximum(particle_mm*.5,bed*(.06+.88*layer)-collision)
 # Hard powder-surface and filter-plate constraints.
 zp=np.clip(zp,particle_mm*.5,np.maximum(particle_mm*.5,bed-particle_mm*.35))
 xp,yp,zp=rotate(xp,yp,zp,t)
 marker_size=float(np.clip(1.3+particle_mm*.75,1.5,6.0))
 if show_particles:
  out.append(go.Scatter3d(x=xp,y=yp,z=zp,mode='markers',marker=dict(size=marker_size,color='#e2ab1e',opacity=.84),name='DEM-style powder particles',hovertemplate='Representative particle<extra></extra>'))
 return out,ph,lh,disp

with st.sidebar:
 st.header('Process inputs')
 liq=st.number_input('Liquid quantity (L)',0.0,52.0,10.0,.1)
 pm=st.number_input('Powder quantity (kg)',0.0,100.0,1.8,.1)
 bd=st.number_input('Powder bulk density (kg/L)',.01,5.0,.22,.01)
 rho=st.number_input('Liquid density (kg/L)',.001,25.0,1.0,.001)
 trans=st.slider('Liquid transparency (%)',0,95,42)
 sd=st.slider('Shaft diameter (mm)',30.0,100.0,68.0,1.0)
 lift=st.slider('Agitator lift (mm)',3.0,68.0,3.0,1.0)
 rpm=st.slider('Agitator speed (rpm)',0.0,30.0,8.0,.5)
 target=st.slider('Target tilt angle (deg)',0.0,60.0,30.0,1.0)
 aor=st.slider('Powder angle of repose (deg)',15.0,50.0,30.0,1.0)
 st.header('DEM-style powder controls')
 dem_mu=st.slider('Particle friction coefficient',0.10,0.90,0.45,0.05)
 dem_e=st.slider('Coefficient of restitution',0.00,0.90,0.25,0.05)
 dem_drag=st.slider('Agitator-particle drag coupling',0.00,1.00,0.65,0.05)
 particle_mm=st.slider('Representative particle diameter (mm)',0.2,5.0,1.0,0.1)
 particle_count=st.slider('Rendered representative particles',150,3000,900,50)
 st.header('Dynamics')
 duration=st.slider('Tilt duration (s)',1.0,15.0,5.0,.5)
 zeta=st.slider('Slosh damping ratio',.01,.50,.10,.01)
 gain=st.slider('Inertia coupling',0.0,2.0,1.0,.05)
 settle=st.slider('Settling duration (s)',0.0,15.0,6.0,.5)
 st.header('Display / equipment visibility')
 show_liquid_volume=st.checkbox('Show full liquid volume',True)
 show_powder_surface=st.checkbox('Show powder bed surface',True)
 show_particles=st.checkbox('Show DEM-style powder particles',True)
 show_shell=st.checkbox('Show ANFD shell and filter head',True)
 show_agitator=st.checkbox('Show shaft and anchor agitator',True)
 show_supports=st.checkbox('Show external support legs',False)
 show_trunnions=st.checkbox('Show trunnion axis',True)
 show_axes=st.checkbox('Show plot axes and grid',False)

pow_l=pm/bd; th=math.radians(target); final,ph,lh,disp=traces(liq,pow_l,th,sd,lift,aor,trans,rpm,duration,0,dem_mu,dem_e,dem_drag,particle_mm,particle_count)
metrics=st.columns(6)
vals=[f'{liq:.2f} L',f'{pow_l:.2f} L',f'{np.max(lh):.1f} mm',f'{np.min(lh):.1f} mm',f'{np.max(ph):.1f} mm',f'{disp:.2f} L']
for c,l,v in zip(metrics,['Liquid','Powder bulk volume','Liquid low side','Liquid high side','Powder low side','Immersed agitator displacement'],vals): c.metric(l,v)
fig=go.Figure(final)
if show_supports:
 for x,y in [(-550,-550),(-550,550),(550,-550),(550,550)]: fig.add_trace(go.Scatter3d(x=[x,x*.55],y=[y,y*.55],z=[-650,120],mode='lines',line=dict(color='#77838a',width=16),showlegend=False,name='External supports'))
if show_trunnions:
 fig.add_trace(go.Scatter3d(x=[-350,350],y=[0,0],z=[PIVOT,PIVOT],mode='lines+markers',line=dict(color='#89949a',width=14),marker=dict(size=7),name='Trunnion axis'))
tm,ang,q,om=slosh_response(liq,th,duration,zeta,gain); dt=tm[1]-tm[0]; ns=max(1,int(settle/dt)); ts=np.arange(1,ns+1)*dt
wd=om*math.sqrt(max(1e-9,1-zeta**2)); v0=(q[-1]-q[-2])/dt; qs=np.exp(-zeta*om*ts)*(q[-1]*np.cos(wd*ts)+(v0+zeta*om*q[-1])/wd*np.sin(wd*ts))
times=np.r_[tm,duration+ts]; angles=np.r_[ang,np.full(ns,th)]; modes=np.r_[q,qs]
frames=[]
records=[]
for i,(T,A,Q) in enumerate(zip(times,angles,modes)):
 mv,ph_i,lh_i,disp_i=traces(liq,pow_l,float(A),sd,lift,aor,trans,rpm,float(T),float(np.clip(Q,-R*.3,R*.3)),dem_mu,dem_e,dem_drag,particle_mm,particle_count)
 frames.append(go.Frame(data=mv,name=str(i),traces=list(range(len(mv)))))
 records.append({
  'Time_s':float(T),'Tilt_deg':math.degrees(float(A)),
  'Liquid_low_mm':float(np.max(lh_i)),'Liquid_high_mm':float(np.min(lh_i)),
  'Powder_low_mm':float(np.max(ph_i)),'Powder_high_mm':float(np.min(ph_i)),
  'Agitator_displacement_L':float(disp_i),'Slosh_modal_mm':float(Q)
 })
fig.frames=frames
fig.update_layout(height=780,margin=dict(l=0,r=0,t=42,b=0),scene=dict(aspectmode='data',xaxis=dict(title='X mm',visible=show_axes,showgrid=show_axes),yaxis=dict(title='Y mm',visible=show_axes,showgrid=show_axes),zaxis=dict(title='Z mm',visible=show_axes,showgrid=show_axes),camera=dict(eye=dict(x=1.55,y=1.65,z=1.25)),bgcolor='rgb(246,248,250)'),legend=dict(orientation='h'),updatemenus=[dict(type='buttons',x=.02,y=1.08,buttons=[dict(label='▶ Animate tilt + agitation',method='animate',args=[None,dict(frame=dict(duration=90,redraw=True),transition=dict(duration=25),fromcurrent=True)]),dict(label='❚❚ Pause',method='animate',args=[[None],dict(frame=dict(duration=0),mode='immediate')])])],sliders=[dict(active=len(frames)-1,currentvalue=dict(prefix='Time: '),steps=[dict(label=f'{times[i]:.1f}s',method='animate',args=[[str(i)],dict(mode='immediate',frame=dict(duration=0,redraw=True))]) for i in range(len(frames))])])
st.plotly_chart(fig,use_container_width=True)

st.subheader('Liquid and powder level histories')
HISTORY_COLUMNS=['Time_s','Tilt_deg','Liquid_low_mm','Liquid_high_mm','Powder_low_mm','Powder_high_mm','Agitator_displacement_L','Slosh_modal_mm']
hist=pd.DataFrame.from_records(records,columns=HISTORY_COLUMNS)
if hist.empty:
 st.error('No simulation records were generated. Check tilt and settling inputs.')
 st.stop()
level_fig=go.Figure()
level_fig.add_trace(go.Scatter(x=hist.Time_s,y=hist.Liquid_low_mm,name='Liquid low side',line=dict(color='#0b84f3',width=3)))
level_fig.add_trace(go.Scatter(x=hist.Time_s,y=hist.Liquid_high_mm,name='Liquid high side',line=dict(color='#59c6ff',width=3,dash='dash')))
level_fig.add_trace(go.Scatter(x=hist.Time_s,y=hist.Powder_low_mm,name='Powder low side',line=dict(color='#c58a00',width=3)))
level_fig.add_trace(go.Scatter(x=hist.Time_s,y=hist.Powder_high_mm,name='Powder high side',line=dict(color='#e7bd52',width=3,dash='dash')))
level_fig.update_layout(height=430,xaxis_title='Simulation time (s)',yaxis_title='Local level from filter plate (mm)',hovermode='x unified',legend=dict(orientation='h'))
st.plotly_chart(level_fig,use_container_width=True)

c1,c2=st.columns(2)
with c1:
 tilt_fig=go.Figure()
 tilt_fig.add_trace(go.Scatter(x=hist.Time_s,y=hist.Tilt_deg,name='Vessel tilt',line=dict(color='#6b5bd2',width=3)))
 tilt_fig.add_trace(go.Scatter(x=hist.Time_s,y=hist.Slosh_modal_mm,name='Slosh modal displacement',yaxis='y2',line=dict(color='#00a6a6',width=2)))
 tilt_fig.update_layout(height=340,xaxis_title='Time (s)',yaxis=dict(title='Tilt (deg)'),yaxis2=dict(title='Slosh (mm)',overlaying='y',side='right'),legend=dict(orientation='h'))
 st.plotly_chart(tilt_fig,use_container_width=True)
with c2:
 disp_fig=go.Figure(go.Scatter(x=hist.Time_s,y=hist.Agitator_displacement_L,fill='tozeroy',line=dict(color='#7b8790',width=3),name='Immersed agitator displacement'))
 disp_fig.update_layout(height=340,xaxis_title='Time (s)',yaxis_title='Displacement (L)')
 st.plotly_chart(disp_fig,use_container_width=True)

st.download_button('Download simulation history CSV',hist.to_csv(index=False).encode('utf-8'),'OMCA_ANFD_level_history.csv','text/csv')
st.subheader('Coupled calculation basis')
st.markdown('''
- Powder bulk volume is mass divided by bulk density. Powder slope is limited by angle of repose.
- Full liquid volume is rendered with a closed outer boundary between the powder bed and the gravity-horizontal free surface.
- Liquid level is solved after subtracting immersed shaft and anchor-frame displacement above the powder bed.
- Raising the agitator reduces immersed solid volume and changes liquid level.
- The anchor rotates at entered RPM. DEM-style representative particles respond to tangential drag, centrifugal migration, wall confinement, friction, restitution and repose-angle avalanching.
- Liquid inertia uses a damped first lateral slosh mode and continues after tilt stops.
''')
st.warning('The powder engine is DEM-style, not a validated discrete-element solver. It uses representative particles and reduced-order contact/drag rules. Quantitative design or scale-up requires calibrated DEM such as Altair EDEM, Ansys Rocky, or LIGGGHTS; exact liquid breakup requires CFD/VOF.')
with open(Path(__file__).with_name('OMCA_ANFD_GA_Model_All_Nozzles_Supports.glb'),'rb') as f: st.download_button('Download GA-based GLB model',f,file_name='OMCA_ANFD_GA_Model_All_Nozzles_Supports.glb')
