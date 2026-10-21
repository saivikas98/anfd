import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title='OMCA ANFD GA Digital Twin',layout='wide')
APP_VERSION='GA-V5-GLOBAL-COLOR-HOTFIX-2026-10-21'
polish='rgb(213,219,222)'  # Global compatibility alias used by agitator and drive geometry.
st.title('OMCA ANFD | GA-Shaped Multiphase Digital Twin')
st.success(f'Loaded app version: {APP_VERSION}')
st.caption('Vessel geometry rebuilt from the GA arrangement: flanged cylindrical body, dished heads, filter plate, trunnions, side gearbox/motor, nozzles and anchor agitator.')

# GA/process basis, millimetres
ID=350.0; R=ID/2; AREA=math.pi*R**2; CAP=52.0; H=CAP*1e6/AREA; P=H/2; G=9810.0
NR,NA=38,108
rv=(np.arange(NR)+.5)*R/NR; av=(np.arange(NA)+.5)*2*np.pi/NA
RR,AA=np.meshgrid(rv,av,indexing='ij'); YY=RR*np.sin(AA); W=RR*(R/NR)*(2*np.pi/NA)

def rx(x,y,z,t,pivot=P):
    c,s=math.cos(t),math.sin(t); x=np.asarray(x); y=np.asarray(y); z=np.asarray(z)
    return x,y*c-(z-pivot)*s,y*s+(z-pivot)*c+pivot

def surf_rev(profile,t,color,name,opacity=.45,n=96):
    a=np.linspace(0,2*np.pi,n); r=np.array([p[0] for p in profile]); z=np.array([p[1] for p in profile]); A,Rm=np.meshgrid(a,r); Z=np.repeat(z[:,None],n,axis=1)
    X=Rm*np.cos(A); Y=Rm*np.sin(A); X,Y,Z=rx(X,Y,Z,t)
    return go.Surface(x=X,y=Y,z=Z,colorscale=[[0,color],[1,color]],opacity=opacity,showscale=False,name=name,hoverinfo='name')

def ring(ro,ri,z,t,color,name,opacity=.8):
    a=np.linspace(0,2*np.pi,100); rr=np.array([ri,ro]); A,Q=np.meshgrid(a,rr); X=Q*np.cos(A); Y=Q*np.sin(A); Z=np.full_like(X,z); X,Y,Z=rx(X,Y,Z,t)
    return go.Surface(x=X,y=Y,z=Z,colorscale=[[0,color],[1,color]],opacity=opacity,showscale=False,name=name,hoverinfo='name')

def line(x,y,z,t,name,color='#c6ced2',width=10,show=True):
    x,y,z=rx(x,y,z,t)
    return go.Scatter3d(x=x,y=y,z=z,mode='lines',line=dict(color=color,width=width),name=name,showlegend=show,hoverinfo='name')

def powder_grid(vol,t,aor):
    slope=math.tan(math.copysign(min(abs(t),math.radians(aor)),t)); lo,hi=-H,2*H; target=vol*1e6
    for _ in range(50):
        m=(lo+hi)/2; h=np.clip(m-YY*slope,0,H)
        if np.sum(h*W)<target: lo=m
        else: hi=m
    return np.clip((lo+hi)/2-YY*slope,0,H)

def liquid_plane(y,C,t):
    c=max(abs(math.cos(t)),1e-4)
    return np.clip(P+(C-P-y*math.sin(t))/c,0,H)

def solid_disp(top,base,shaft,lift):
    shaft_v=np.pi*(shaft/2)**2*np.clip(top-np.maximum(base,lift),0,None)
    legs=2*14*16*np.clip(np.minimum(top,lift+.73*H)-np.maximum(base,lift+8),0,None)
    lower=2*(R-38)*14*16*((top>lift+8)&(base<lift+24))
    upper=2*(R-3)*14*16*((top>lift+.70*H)&(base<lift+.75*H))
    return shaft_v+legs+lower+upper

def liquid_grid(vol,powder,t,shaft,lift):
    target=vol*1e6; lo,hi=P-H-2*R,P+H+2*R
    for _ in range(58):
        C=(lo+hi)/2; top=liquid_plane(YY,C,t)
        liquid=np.sum(np.clip(top-powder,0,None)*W)-float(np.mean(solid_disp(top,powder,shaft,lift)))
        if liquid<target: lo=C
        else: hi=C
    top=liquid_plane(YY,(lo+hi)/2,t); disp=float(np.mean(solid_disp(top,powder,shaft,lift)))/1e6
    return top,disp

def field_surface(field,t,color,name,opacity,wave=0):
    rr=np.linspace(0,R*.984,30); aa=np.linspace(0,2*np.pi,100); Q,A=np.meshgrid(rr,aa); X=Q*np.cos(A); Y=Q*np.sin(A)
    ir=np.minimum(NR-1,(Q/R*NR).astype(int)); ia=np.mod((A/(2*np.pi)*NA).astype(int),NA); Z=field[ir,ia]+wave*(Q/R)*np.cos(A)
    X,Y,Z=rx(X,Y,np.clip(Z,0,H),t)
    return go.Surface(x=X,y=Y,z=Z,colorscale=[[0,color],[1,color]],opacity=opacity,showscale=False,name=name,hoverinfo='name')

def liquid_wall(top,base,t,opacity):
    a=np.linspace(0,2*np.pi,120); f=np.linspace(0,1,30); A,F=np.meshgrid(a,f); X=R*.98*np.cos(A); Y=R*.98*np.sin(A)
    ia=np.mod((A/(2*np.pi)*NA).astype(int),NA); ir=np.full_like(ia,NR-1); z0=base[ir,ia]; z1=np.maximum(z0,top[ir,ia]); Z=z0+F*(z1-z0)
    X,Y,Z=rx(X,Y,Z,t)
    return go.Surface(x=X,y=Y,z=Z,colorscale=[[0,'rgb(0,70,165)'],[1,'rgb(58,186,246)']],opacity=opacity,showscale=False,name='Full liquid volume',hoverinfo='skip')

def equipment_traces(t,show):
    out=[]; ss='rgb(166,174,179)'; polish='rgb(213,219,222)'; dark='rgb(70,78,83)'; blue='rgb(15,74,155)'
    # Actual ANFD silhouette: lower shallow filter head, straight shell, shallow dished top, flange rings.
    if show['shell']:
        profile=[(R*.72,-62),(R*.92,-38),(R,0),(R,H),(R*.96,H+38),(R*.75,H+76),(0,H+112)]
        out.append(surf_rev(profile,t,ss,'GA vessel shell',.28))
        out += [ring(R*1.12,R*.88,0,t,polish,'Bottom filter-head flange'),ring(R*1.10,R*.88,H,t,polish,'Top body flange'),ring(R*1.02,0,8,t,dark,'Filter plate',.85)]
    # Side trunnions and bearing shells rotate with vessel.
    if show['trunnions']:
        for side in (-1,1):
            out.append(line([side*R,side*(R+230)],[0,0],[P,P],t,'Trunnion' if side==1 else '',polish,16,side==1))
            out.append(surf_rev([(55,P-60),(55,P+60)],t,dark,'Bearing housing',.9,48))
    # GA nozzles distributed around shell/head; represented as flanged necks.
    if show['nozzles']:
        specs=[('A',220,H*.82,20),('B',310,H*.58,25),('B1',30,H*.72,22),('C',20,H*.88,18),('D',345,H*.78,18),('E',15,H*.94,20),('F',335,H+42,20),('G',285,H*.18,22),('P',210,H*.48,25),('P1',235,H*.15,20),('Q',55,H*.12,20),('R',80,H*.26,22),('S',65,H*.51,18),('T',45,H*.82,20),('U',120,H*.89,20),('V',130,H+54,22),('X1',42,H*.32,18),('X2',95,H*.38,18),('Y',195,H*.50,20),('Z',165,H*.87,18)]
        for tag,az,z0,rad in specs:
            a=math.radians(az); x0=R*math.cos(a); y0=R*math.sin(a); x1=(R+105)*math.cos(a); y1=(R+105)*math.sin(a)
            out.append(line([x0,x1],[y0,y1],[z0,z0],t,tag,'#8fd2ee',5,False))
    # Side gearbox and motor on drive axis (GA-style horizontal drive).
    if show['drive']:
        # simplified box/line envelope anchored to local -X side
        out.append(line([-R-20,-R-250],[0,0],[P,P],t,'Drive coupling',polish,18))
        out.append(go.Scatter3d(x=[-R-330],y=[0],z=[P],mode='markers',marker=dict(size=38,color=blue,symbol='square'),name='Motor/gearbox'))
    return out

def model(liq,powv,t,shaft,lift,aor,trans,rpm,time,show,dem,wave=0):
    polish='rgb(213,219,222)'
    polish='rgb(213,219,222)'
    ph=powder_grid(powv,t,aor); lh,disp=liquid_grid(liq,ph,t,shaft,lift); out=equipment_traces(t,show)
    if show['powder']: out.append(field_surface(ph,t,'rgb(211,165,36)','Powder bed',.94))
    if show['liquid']:
        op=max(.22,1-trans/100); out += [liquid_wall(lh,ph,t,op*.76),field_surface(lh,t,'rgb(46,172,245)','Liquid surface',max(.34,op),wave)]
    rotang=rpm*2*np.pi*time/60
    if show['agitator']:
        out.append(line([0,0],[0,0],[lift,H+140+lift],t,'Agitator shaft',polish,11))
        c,s=math.cos(rotang),math.sin(rotang)
        for side in (-1,1):
            x=np.array([0,side*(R-3),side*(R-3),side*38]); y=np.zeros(4); z=np.array([lift+.74*H,lift+.74*H,lift+8,lift+8]); out.append(line(x*c-y*s,x*s+y*c,z,t,'Anchor agitator' if side==1 else '',polish,13,side==1))
    if show['particles']:
        n=dem['count']; idx=np.arange(n); q=(idx+.5)/n; rad=R*.94*np.sqrt(q); aa=idx*2.399963+rotang*dem['drag']*(.2+.8*q)
        drift=-math.copysign(1,t if t else 1)*min(1,max(0,abs(math.degrees(t))-aor)/18)*(1-dem['mu']*.5)*R*.28
        x=rad*np.cos(aa); y=np.clip(rad*np.sin(aa)+drift,-R+3,R-3); rr=np.hypot(x,y); sc=np.minimum(1,(R-3)/np.maximum(rr,1e-6)); x*=sc; y*=sc
        ir=np.minimum(NR-1,(np.hypot(x,y)/R*NR).astype(int)); ia=np.mod((np.arctan2(y,x)/(2*np.pi)*NA).astype(int),NA); bed=ph[ir,ia]; z=np.clip(bed*(.08+.84*((idx%31)/31))-dem['e']*2*np.abs(np.sin(idx*.73+time)),1,np.maximum(1,bed-1)); x,y,z=rx(x,y,z,t)
        out.append(go.Scatter3d(x=x,y=y,z=z,mode='markers',marker=dict(size=2.2,color='#e5ad21',opacity=.82),name='DEM-style powder particles'))
    return out,ph,lh,disp

with st.sidebar:
    st.header('Process'); liq=st.number_input('Liquid quantity (L)',0.0,52.0,10.0,.1); pm=st.number_input('Powder quantity (kg)',0.0,100.0,1.8,.1); bd=st.number_input('Bulk density (kg/L)',.01,5.0,.22,.01)
    trans=st.slider('Liquid transparency (%)',0,95,42); shaft=st.slider('Shaft diameter (mm)',30.0,100.0,68.0); lift=st.slider('Agitator lift (mm)',3.0,68.0,3.0); rpm=st.slider('Agitator speed (rpm)',0.0,30.0,8.0,.5); target=st.slider('Tilt angle (deg)',0.0,60.0,30.0); aor=st.slider('Angle of repose (deg)',15.0,50.0,30.0)
    st.header('DEM-style powder'); mu=st.slider('Friction coefficient',.1,.9,.45,.05); rest=st.slider('Restitution',0.0,.9,.25,.05); drag=st.slider('Agitator drag',0.0,1.0,.65,.05); count=st.slider('Representative particles',150,3000,900,50)
    st.header('Equipment visibility'); show={'shell':st.checkbox('GA vessel shell/head/flanges',True),'trunnions':st.checkbox('Trunnions',True),'nozzles':st.checkbox('GA nozzle set',True),'drive':st.checkbox('Side gearbox/motor',True),'liquid':st.checkbox('Full liquid volume',True),'powder':st.checkbox('Powder bed',True),'particles':st.checkbox('Powder particles',True),'agitator':st.checkbox('Anchor agitator',True)}
    supports=st.checkbox('External support frame',False); axes=st.checkbox('Axes/grid',False)
    st.header('Dynamics'); duration=st.slider('Tilt duration (s)',1.0,15.0,5.0,.5); settle=st.slider('Settling duration (s)',0.0,10.0,4.0,.5); damping=st.slider('Slosh damping',.01,.5,.1,.01)

powv=pm/bd; th=math.radians(target); dem={'mu':mu,'e':rest,'drag':drag,'count':count}; n=45; t1=np.linspace(0,duration,n); u=t1/duration; ang=th*(10*u**3-15*u**4+6*u**5); om=math.sqrt(G*(1.841/R)*math.tanh((1.841/R)*max(1,liq*1e6/AREA))); q=R*.065*np.sin(om*t1)*np.sin(np.pi*u)*np.exp(-damping*om*t1); dt=t1[1]-t1[0]; ts=np.arange(dt,settle+dt/2,dt); times=np.r_[t1,duration+ts]; angles=np.r_[ang,np.full(len(ts),th)]; modes=np.r_[q,q[-1]*np.exp(-damping*om*ts)*np.cos(om*ts)]
final,ph,lh,disp=model(liq,powv,th,shaft,lift,aor,trans,rpm,duration,show,dem); fig=go.Figure(final)
if supports:
    for x,y in [(-550,-500),(-550,500),(550,-500),(550,500)]: fig.add_trace(go.Scatter3d(x=[x,x*.45],y=[y,y*.45],z=[-600,30],mode='lines',line=dict(color='#707c82',width=16),showlegend=False))
outputs=[]
for T,A,Q in zip(times,angles,modes):
    mv,p1,l1,d1=model(liq,powv,float(A),shaft,lift,aor,trans,rpm,float(T),show,dem,float(Q)); outputs.append((float(T),float(A),float(Q),mv,p1,l1,float(d1)))
fig.frames=[go.Frame(data=o[3],name=str(i),traces=list(range(len(o[3])))) for i,o in enumerate(outputs)]
hist=pd.DataFrame([{'Time_s':o[0],'Tilt_deg':math.degrees(o[1]),'Liquid_low_mm':float(np.max(o[5])),'Liquid_high_mm':float(np.min(o[5])),'Powder_low_mm':float(np.max(o[4])),'Powder_high_mm':float(np.min(o[4])),'Agitator_displacement_L':o[6]} for o in outputs])
fig.update_layout(height=780,scene=dict(aspectmode='data',xaxis=dict(visible=axes),yaxis=dict(visible=axes),zaxis=dict(visible=axes),bgcolor='rgb(247,249,251)'),margin=dict(l=0,r=0,t=34,b=0),updatemenus=[dict(type='buttons',buttons=[dict(label='▶ Animate',method='animate',args=[None,dict(frame=dict(duration=90,redraw=True),fromcurrent=True)]),dict(label='❚❚ Pause',method='animate',args=[[None],dict(frame=dict(duration=0),mode='immediate')])])])
st.plotly_chart(fig,use_container_width=True)
metrics=st.columns(5); metrics[0].metric('Liquid low side',f'{np.max(lh):.1f} mm'); metrics[1].metric('Liquid high side',f'{np.min(lh):.1f} mm'); metrics[2].metric('Powder low side',f'{np.max(ph):.1f} mm'); metrics[3].metric('Agitator displacement',f'{disp:.2f} L'); metrics[4].metric('Agitator OD',f'{ID-6:.0f} mm')
st.subheader('Liquid and powder levels'); chart=go.Figure(); chart.add_trace(go.Scatter(x=hist.Time_s,y=hist.Liquid_low_mm,name='Liquid low')); chart.add_trace(go.Scatter(x=hist.Time_s,y=hist.Liquid_high_mm,name='Liquid high')); chart.add_trace(go.Scatter(x=hist.Time_s,y=hist.Powder_low_mm,name='Powder low')); chart.add_trace(go.Scatter(x=hist.Time_s,y=hist.Powder_high_mm,name='Powder high')); chart.update_layout(xaxis_title='Time (s)',yaxis_title='Level from filter plate (mm)',hovermode='x unified'); st.plotly_chart(chart,use_container_width=True)
st.download_button('Download history CSV',hist.to_csv(index=False).encode(),'OMCA_GA_level_history.csv','text/csv')
st.info('Vessel silhouette now follows the GA arrangement: straight process shell, shallow bottom filter head, removable lower flange/clamp, shallow dished top, top body flange, trunnions, side drive and distributed nozzle set. Supports remain optional.')
