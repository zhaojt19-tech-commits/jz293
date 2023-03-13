#import urllib.request as urllibreq
import urllib as urllibreq
import cv2
import scipy.sparse as sparse
import numpy as np
import os
import sys

white = 245.
if len(sys.argv) <= 2:
	pid = 1
	total_pid = 1
else:
	pid = int(sys.argv[1])
	total_pid = int(sys.argv[2])
process_mode = "midreso"
#Set this mode to “highreso” (1um), “midreso” (10um) or “lowreso” (50um).
if process_mode == "highreso":
	resoX, resoY, resoZ = 1, 1, 1
elif process_mode == "midreso":
	resoX, resoY, resoZ = 10, 10, 10
elif process_mode == "lowreso":
	resoX, resoY, resoZ = 50, 50, 50
#Set this two value (larger than 1) to speed up the reading from raw images.
raw_resoX = 1
raw_resoY = 1
fout_dir = '/scratch/users/swang91/allen_brain/'
image_dir = '/scratch/users/swang91/allen_brain/image/'
def image_to_reference(sx, sy, section_number, section_thickness, align2d, align3d):
	# ---- transform sub_image sx,sy to volume vx,vy,vz
	vz = float(section_number) * float(section_thickness)
	vx = float(align2d[‘tvs_00’]) * sx + float(align2d[‘tvs_01’]) * sy + float(align2d[‘tvs_04’])
	vy = float(align2d[‘tvs_02’]) * sx + float(align2d[‘tvs_03’]) * sy + float(align2d[‘tvs_05’])
	# ---- transform volume vx,vy,vx to reference rx,ry,rz
	rx = float(align3d[‘tvr_00’]) * vx + float(align3d[‘tvr_01’]) * vy + float(align3d[‘tvr_02’]) * vz + float(align3d[‘tvr_09’])
	ry = float(align3d[‘tvr_03’]) * vx + float(align3d[‘tvr_04’]) * vy + float(align3d[‘tvr_05’]) * vz + float(align3d[‘tvr_10’])
	rz = float(align3d[‘tvr_06’]) * vx + float(align3d[‘tvr_07’]) * vy + float(align3d[‘tvr_08’]) * vz + float(align3d[‘tvr_11’])
	#rx, ry, rz = vx, vy, vz
	return rx, ry, rz

def floor(x):
	if x >= 0:
		return int(x)
	else:
		return int(x) - 1

base_url = "http://api.brain-map.org/api/v2/image_download/69750516?&initImage=expression&colormap=0"
data_dir = "data/"  
the_one  = ""
fin = open(data_dir+‘sectionInfoAll.txt’,‘r’)
genes_dict = {}
for gi, line in enumerate(fin):
	gene, name, days, section_thickness, dataset_id, section_number, image_id, align2d_str, align3d_str = line.strip().split(‘\t’)
	gtuple = (gene, name, days, section_thickness, dataset_id, section_number, image_id, align2d_str, align3d_str)
	if gene in genes_dict.keys():
		genes_dict[gene] = tuple(list(genes_dict[gene]) + [gtuple,])
	else:
		genes_dict[gene] = (gtuple,)
fin.close()
print("Total number of genes:", len(genes_dict.keys()))
genelist = genes_dict.keys()
for gi, gene in enumerate(genelist):
	if gi % total_pid != pid and total_pid>1:
		continue
	xyz_maxmin = []
	all_sec_xyz = []
	for si, gtuple in enumerate(genes_dict[gene]):
		allx, ally, allz, allexp = [], [], [], []
		gene, name, days, section_thickness, dataset_id, section_number, image_id, align2d_str, align3d_str = gtuple
		print(“Downloading gene: “+ str(gi) + “, ” + gene + “; section: “+ section_number + “, No. “, si)
		align2d = eval(align2d_str)
		align3d = eval(align3d_str)
		imagefile = image_dir + image_id + ‘.jpg’
		new_url = base_url.replace(‘69750516’, image_id)
		urllibreq.urlretrieve(new_url, imagefile)
		imagearray = cv2.imread(imagefile)
		width, height, numchan = imagearray.shape
		print(“Image downloaded, now processing.“)
		#Processing each section.
		for x in range(int(width/raw_resoX)):
			for y in range(int(height/raw_resoY)):
				if imagearray[int(raw_resoX * x)][int(raw_resoY * y)][0] < white:
					rx, ry, rz = image_to_reference(raw_resoX * x, raw_resoY * y, section_number, section_thickness, align2d, align3d)
					exp = (white - imagearray[raw_resoX * x][raw_resoY * y][0]) / white * 256.
					allx.append(floor(float(rx)/resoX))
					ally.append(floor(float(ry)/resoY))
					allz.append(floor(float(rz)/resoZ))
					allexp.append(float(exp))
		allx = np.array(allx)
		ally = np.array(ally)
		allz = np.array(allz)
		x_min, x_max = allx.min(), allx.max()
		y_min, y_max = ally.min(), ally.max()
		z_min, z_max = allz.min(), allz.max()
		xyz_maxmin.append([x_min, x_max, y_min, y_max, z_min, z_max])
		sec_xyz = {}
		num_exps = len(allexp)
		print(“raw num_exps for this section: “, num_exps)
		for i in range(num_exps):
			xyz = (allx[i], ally[i], allz[i])
			if not(xyz in sec_xyz.keys()):
				sec_xyz[xyz] = allexp[i]
			else:
				sec_xyz[xyz] += allexp[i]
		all_sec_xyz.append(sec_xyz)
	#Merge all the sections, record max and min x, y, z for all.
	xyz_maxmin = np.array(xyz_maxmin)
	print(“max and min has shape: “, xyz_maxmin.shape)
	x_min, x_max = xyz_maxmin[:,0].min(), xyz_maxmin[:,1].max()
	y_min, y_max = xyz_maxmin[:,2].min(), xyz_maxmin[:,3].max()
	z_min, z_max = xyz_maxmin[:,4].min(), xyz_maxmin[:,5].max()
	three_shape = np.array([x_max - x_min + 1, y_max- y_min + 1, z_max - z_min + 1])
	allxyz = {}
	for sec_xyz in all_sec_xyz:
		for xyz, exp in sec_xyz.items():
			new_xyz = (xyz[0] - x_min, xyz[1] - y_min, xyz[2] - z_min)
			if not(new_xyz in allxyz.keys()):
				allxyz[new_xyz] = exp
			else:
				allxyz[new_xyz] += exp
	allxy = []
	allz = []
	allexp = []
	for xyz, exp in allxyz.items():
		allxy.append(xyz[0] * three_shape[1] + xyz[1])
		allz.append(xyz[2])
		allexp.append(exp)
	#Saving
	num_exps = len(allexp)
	print(“discrete num_exps: “, num_exps)
	print(“discrete shape: “, three_shape)
	np.save(fout_dir+ process_mode + “/” + gene + “_shape”, three_shape)
	fout_z = fout_dir + process_mode  + “/” + gene + “_z.npz”
	coomat_z = sparse.coo_matrix((allexp, (allxy, allz)), shape=(three_shape[0] * three_shape[1], three_shape[2]))
	sparse.save_npz(fout_z, coomat_z)