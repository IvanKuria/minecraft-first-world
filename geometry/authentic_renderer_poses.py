exec(open(__file__.replace('authentic_renderer_poses.py','upright_bundle.py')).read().split('sol=least_squares')[0])
f0=240/np.tan(np.deg2rad(35));records={}
for i,n in enumerate([11,12,13]):
 v0=x0[2+i*5:7+i*5];fit=least_squares(lambda v:(project(X,v,f0,f0)-obs[n]).ravel(),v0)
 v=fit.x;rot=R(v[0],v[1]);C=-rot.T@v[2:5];player=C+.3*rot[2];err=np.sqrt(np.mean(fit.fun**2)*2)
 records[n]={'world_translation_for_positive_coords':[128,40,128],'camera_center_local':C.tolist(),'player_position_local_for_gl_setback_0_3':player.tolist(),'yaw_degrees':float(np.rad2deg(v[0])),'pitch_degrees':float(np.rad2deg(v[1])),'fov_y_degrees':70,'width':852,'height':480,'all_landmark_rms_px':float(err),'training_uses':'cube and cliff; render overlay check is not independently held out'}
 print(n,records[n])
(P/'authentic-renderer-poses.json').write_text(json.dumps(records,indent=2))
