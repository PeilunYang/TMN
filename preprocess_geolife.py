import os
import numpy as np
import cPickle

beijing_lat_range = [39.6,40.7]
beijing_lon_range = [115.9,117.0]

path = 'Geolife/Data/'
dir_traj = os.listdir(path)

geolife_traj = []
for dirt in dir_traj:
    new_path = path + dirt + '/Trajectory/'
    files = os.listdir(new_path)
    for file in files:
    	f = open(new_path + file)
    	print(file + ' ' + dirt)
    	iter_f = iter(f)
    	tmp = []
    	for line in iter_f:
    		tmp.append(line)
    	#print(tmp)
    	del tmp[0]
    	del tmp[0]
    	del tmp[0]
    	del tmp[0]
    	del tmp[0]
    	#print(tmp[5])
    	del tmp[0]
    	#print(tmp)
    	geolife_traj.append(tmp)
    	f.close()
    	

Trajectory = []
count=0
for trajs in geolife_traj:
	inrange = True
	Traj = []
	for traj in trajs:
		tr = traj.split(',')
		lat = np.float64(tr[0])
		lon = np.float64(tr[1])
		if ( (lat < beijing_lat_range[0]) | (lat > beijing_lat_range[1]) | (lon < beijing_lon_range[0]) | (lon > beijing_lon_range[1]) ):
			inrange = False
		traj_tup = (lon, lat)
		Traj.append(traj_tup)

	if inrange != False:
		Trajectory.append(Traj)
		f=open('geolife_trajs', 'a')
		f.write(str(count) + '\t' + str(len(Traj)) + '\t')
		f.write(", ".join(str(x) for x in Traj))
		f.close()
		# if(count < 1800):
		# 	f=open('geolife_train', 'a')
		# 	f.write(str(count) + '\t' + str(len(Traj)) + '\t')
		# 	f.write(", ".join(str(x) for x in Traj))
		# 	f.close()
		# if((count >= 1800) and (count <= 9000):
		# 	f=open('geolife_test', 'a')
		# 	f.write(str(count) + '\t' + str(len(Traj)) + '\t')
		# 	f.write(", ".join(str(x) for x in Traj))
		# 	f.close()
		
print(len(Trajectory))

# cPickle.dump(Trajectory, open('geolife', 'w'))




