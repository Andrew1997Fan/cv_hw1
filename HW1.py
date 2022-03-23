import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
np.seterr(divide='ignore',invalid='ignore')

image_row = 0 
image_col = 0

# visualizing the mask (size : "image width" * "image height")
def mask_visualization(M):
    mask = np.copy(np.reshape(M, (image_row, image_col)))
    plt.figure()
    plt.imshow(mask, cmap='gray')
    plt.title('Mask')

# visualizing the unit normal vector in RGB color space
# N is the normal map which contains the "unit normal vector" of all pixels (size : "image width" * "image height" * 3)
def normal_visualization(N):
    # converting the array shape to (w*h) * 3 , every row is a normal vetor of one pixel
    N_map = np.copy(np.reshape(N, (image_row, image_col, 3)))
    # Rescale to [0,1] float number
    N_map = (N_map + 1.0) / 2.0
    plt.figure()
    plt.imshow(N_map)
    plt.title('Normal map')

# visualizing the depth on 2D image
# D is the depth map which contains "only the z value" of all pixels (size : "image width" * "image height")
def depth_visualization(D):
    D_map = np.copy(np.reshape(D, (image_row,image_col)))
    # D = np.uint8(D)
    plt.figure()
    plt.imshow(D_map)
    plt.colorbar(label='Distance to Camera')
    plt.title('Depth map')
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')

# convert depth map to point cloud and save it to ply file
# Z is the depth map which contains "only the z value" of all pixels (size : "image width" * "image height")
def save_ply(Z,filepath):
    Z_map = np.reshape(Z, (image_row,image_col)).copy()
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
    o3d.io.write_point_cloud(filepath, pcd,write_ascii=True)

# show the result of saved ply file
def show_ply(filepath):
    pcd = o3d.io.read_point_cloud(filepath)
    o3d.visualization.draw_geometries([pcd])

# read the .bmp file
def read_bmp(filepath):
    global image_row
    global image_col
    image = cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
    image_row , image_col = image.shape

    #a = type(image)
    #print(a)
   # print(image.shape)
   # print(image)
    return image


if __name__ == '__main__':
    # showing the windows of all visualization function
    
    #read file
    filepath1 = '/home/andrew/cv_hw1/test/bunny/pic1.bmp' 
    filepath2 = '/home/andrew/cv_hw1/test/bunny/pic2.bmp' 
    filepath3 = '/home/andrew/cv_hw1/test/bunny/pic3.bmp' 
    filepath4 = '/home/andrew/cv_hw1/test/bunny/pic4.bmp' 
    filepath5 = '/home/andrew/cv_hw1/test/bunny/pic5.bmp' 
    filepath6 = '/home/andrew/cv_hw1/test/bunny/pic6.bmp' 
    #read_bmp(filepath1)
    #read_bmp(filepath2)
    #print(np.reshape(read_bmp(filepath1),(1,14400)))
    #print(np.reshape(read_bmp(filepath2),(1,14400)))
    #print(np.reshape(read_bmp(filepath3),(1,14400)))
    #print(np.reshape(read_bmp(filepath4),(1,14400)))
    #print(np.reshape(read_bmp(filepath5),(1,14400)))
    #print(np.reshape(read_bmp(filepath6),(1,14400)))
    
    #normal map
    #light source unit vector 
    L = np.array([(238/np.sqrt(np.square(238)+np.square(235)+np.square(2360)),235/np.sqrt(np.square(238)+np.square(235)+np.square(2360)),2360/np.sqrt(np.square(238)+np.square(235)+np.square(2360))),
        (298/np.sqrt(np.square(298)+np.square(65)+np.square(2480)),65/np.sqrt(np.square(298)+np.square(65)+np.square(2480)),2480/np.sqrt(np.square(298)+np.square(65)+np.square(2480))),
        (-202/np.sqrt(np.square(-202)+np.square(225)+np.square(2240)),225/np.sqrt(np.square(-202)+np.square(225)+np.square(2240)),2240/np.sqrt(np.square(-202)+np.square(225)+np.square(2240))),
       (-252/np.sqrt(np.square(-252)+np.square(115)+np.square(2310)),115/np.sqrt(np.square(-252)+np.square(115)+np.square(2310)),2310/np.sqrt(np.square(-252)+np.square(115)+np.square(2310))),
       (18/np.sqrt(np.square(18)+np.square(45)+np.square(2270)),45/np.sqrt(np.square(18)+np.square(45)+np.square(2270)),2270/np.sqrt(np.square(18)+np.square(45)+np.square(2270))), 
           (-22/np.sqrt(np.square(-22)+np.square(295)+np.square(2230)),295/np.sqrt(np.square(-22)+np.square(295)+np.square(2230)),2230/np.sqrt(np.square(-22)+np.square(295)+np.square(2230)))])
    I = np.array([(np.reshape(read_bmp(filepath1),(14400))),(np.reshape(read_bmp(filepath2),(14400))),(np.reshape(read_bmp(filepath3),(14400))),(np.reshape(read_bmp(filepath4),(14400))),(np.reshape(read_bmp(filepath5),(14400))),(np.reshape(read_bmp(filepath6),(14400)))])
    KdN = np.dot((np.dot(np.linalg.pinv(np.dot(np.transpose(L),L)),np.transpose(L))),I)
    #print(np.dot(np.dot(inv,np.transpose(L)),I))
    print(KdN)
    print(L)
    N = np.divide(KdN,np.sqrt(np.square(KdN)))
    normal_visualization(np.transpose(N))
    plt.show()
