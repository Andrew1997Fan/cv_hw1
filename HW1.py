import numpy as np
import cv2
import glob
import psutil
import scipy
import scipy.io as sio
import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
import open3d as o3d



from sklearn.preprocessing import normalize

lights_path = './test/bunny/LightSource.txt'
bunny_path = './test/bunny/*.npy'
mask_path = './test/bunny'
Lt = np.loadtxt(lights_path) 
name = None
delay = 0
depth = None
mask = None

M = []
for fname in sorted(glob.glob(bunny_path)):
    im = np.load(fname,allow_pickle=True).astype(np.float32) 
    im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    #mask = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if M == []:
        height, width = im.shape
        M = im.reshape((-1, 1))
    else:
        M = np.append(M, im.reshape((-1, 1)), axis=1)
M = np.asarray(M)



N = np.linalg.lstsq(Lt, M.T)[0].T
N = normalize(N, axis=1)


N = np.reshape(N, (height, width, 3))
N[:,:,0], N[:,:,2] = N[:,:,2], N[:,:,0].copy()
N = (N + 1.0) / 2.0

cv2.namedWindow('normal map')
cv2.imshow('normal map', N)
cv2.imwrite('./test/bunny/normal_map.bmp',N)
cv2.waitKey(0)
cv2.destroyAllWindows()

mask_gray = cv2.imread('./test/bunny/pic1.bmp')
mask = mask_gray.astype('uint8')
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
#cv2.imshow('graymask',mask)
print('mask_shape:',mask.shape)

im_h, im_w = mask.shape
N = np.reshape(N, (height, width, 3))

obj_h, obj_w = np.where(mask != 0)

no_pix = np.size(obj_h)

full2obj = np.zeros((im_h, im_w))
for idx in range(np.size(obj_h)):
    full2obj[obj_h[idx], obj_w[idx]] = idx

# Mz = v
M = scipy.sparse.lil_matrix((2*no_pix, no_pix))
v = np.zeros((2*no_pix, 1))


failed_rows = []
for idx in range(no_pix):
    h = obj_h[idx]
    w = obj_w[idx]
    n_x = N[h, w, 0]
    n_y = N[h, w, 1]
    n_z = N[h, w, 2]

    # z_(x+1, y) - z(x, y) = -nx / nz
    row_idx = idx * 2
    if mask[h, w+1]:
        idx_horiz = full2obj[h, w+1]
        M[row_idx, idx] = -1
        M[row_idx, idx_horiz] = 1
        v[row_idx] = -n_x / n_z
    elif mask[h, w-1]:
        idx_horiz = full2obj[h, w-1]
        M[row_idx, idx_horiz] = -1
        M[row_idx, idx] = 1
        v[row_idx] = -n_x / n_z
    else:
        failed_rows.append(row_idx)

    # z_(x, y+1) - z(x, y) = -ny / nz
    row_idx = idx * 2 + 1
    if mask[h+1, w]:
        idx_vert = full2obj[h+1, w]
        M[row_idx, idx] = 1
        M[row_idx, idx_vert] = -1
        v[row_idx] = -n_y / n_z
    elif mask[h-1, w]:
        idx_vert = full2obj[h-1, w]
        M[row_idx, idx_vert] = 1
        M[row_idx, idx] = -1
        v[row_idx] = -n_y / n_z
    else:
        failed_rows.append(row_idx)
    
M = M.todense()
M = np.delete(M, failed_rows, 0)
M = scipy.sparse.lil_matrix(M)
v = np.delete(v, failed_rows, 0)

print('M_shape :',M.shape)
print('M.T_shape :',(M.T).shape)
print('v_shape :',v.shape)


# MtM = np.matmul(M.T, M)
# Mtv = np.matmul(M.T, v)
z = scipy.sparse.linalg.spsolve(M.T @ M, M.T @ v)

std_z = np.std(z, ddof=1)
mean_z = np.mean(z)
z_zscore = (z - mean_z) / std_z

outlier_ind = np.abs(z_zscore) > 10
z_min = np.min(z[~outlier_ind])
z_max = np.max(z[~outlier_ind])

Z = mask.astype('float')
for idx in range(no_pix):
    h = obj_h[idx]
    w = obj_w[idx]
    Z[h, w] = (z[idx] - z_min) / (z_max - z_min)  # *255

depth = Z


# psutil.save_depthmap_as_npy(filename=filename, depth=depth)

if depth is None:
    raise ValueError("Surface depth `depth` is None")
if mask is not None:
    depth = depth * mask

depth = np.uint8(depth)

# cv2.namedWindow('Depth map')
# cv2.imshow('Depth map', Z)

if name is None:
    name = 'depth map'

cv2.imshow(name, depth)
cv2.imwrite('./test/bunny/depth_map.bmp',depth)
cv2.waitKey(delay)
cv2.destroyAllWindows()
cv2.waitKey(10)

image_row = width
image_col = height

# filepath = './test/bunny'


#def depth_visualization(D):
D_map = np.copy(np.reshape(depth, (image_row,image_col)))
cv2.imshow('cv2 show d_map',D_map)
cv2.waitKey(delay)
cv2.destroyAllWindows()
cv2.waitKey(1)
# D = np.uint8(D)
plt.figure()
plt.imshow(D_map)
#plt.show()
plt.colorbar(label='Distance to Camera')
plt.title('Depth map')
plt.xlabel('X Pixel')
plt.ylabel('Y Pixel')
plt.pause(5)
plt.close()

#save_ply(Z,filepath):
Z_map = np.reshape(depth, (image_row,image_col)).copy()
data = np.zeros((image_row*image_col,3),dtype=np.float32)
# let all point float on a base plane 
baseline_val = np.min(Z_map)
Z_map[np.where(Z_map == 0)] = baseline_val
for i in range(image_row):
    for j in range(image_col):
        idx = i * image_col + j
        data[idx][0] = j
        data[idx][1] = i
        data[idx][2] = Z_map[image_row - 1 - i][j]
# output to ply file
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(data)
o3d.io.write_point_cloud('./test/bunny/depth.ply', pcd,write_ascii=True)
print('successfully saved ply')

# def show_ply(filepath):
pcd = o3d.io.read_point_cloud('./test/bunny/depth.ply')
o3d.visualization.draw_geometries([pcd])
print('successfully visualize pcl')
