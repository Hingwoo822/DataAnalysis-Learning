# 27-K-Means（下）：如何使用K-Means对图像进行分割？

图像分割就是利用图像自身的信息，比如颜色、纹理、形状等特征进行划分，将图像分成不同的区域，划分出来， 划分出来的每个区域就相当于对图像中的像素进行了聚类。



### 将微信开屏封面进行分割

![image-20190906163303631](/Users/lirawx/Library/Application Support/typora-user-images/image-20190906163303631.png)



```python
import numpy as np
import PIL.Image as image
from sklearn import preprocessing
from sklearn.cluster import KMeans
from skimage import color
import matplotlib.image as mpimg

def load_data(filePath):
    # 读文件
    f = open(filePath, 'rb')
    data = []
    # 得到图像像素值
    img = image.open(f)
    # 得到图像尺寸
    width, height = img.size
    for x in range(width):
        for y in range(height):
            # 得到x，y的三个通道值
            R, G, B = img.getpixel((x, y))
            data.append([R, G, B])
    f.close()
    # 采用Min-Max规范化
    mm = preprocessing.MinMaxScaler()
    data = mm.fit_transform(data)
    return np.mat(data), width, height
  
  # 加载图像，得到规范化的结果 img，以及图像尺寸
img, width, height = load_data('/Users/lirawx/Documents/Notes/Learning/数据分析实战45/code/kmeans-master/weixin.jpg')

# 用K-Means对图像进行2聚类
kmeans = KMeans(n_clusters=2)
kmeans.fit(img)
label = kmeans.predict(img)
# 将图像聚类结果，转化成图像尺寸矩阵
label = label.reshape([width, height])
# 创建个新图像，用来保存图像聚类的结果，并设置不同的灰度值
pic_mark = image.new('L', (width, height))
for x in range(width):
    for y in range(height):
        # 根据类别设置灰度，类别0灰度为255，类别1灰度为127
        pic_mark.putpixel((x, y), int(256/(label[x][y]+1))-1)
pic_mark.save('weixin_mark.jpg', 'JPEG')

# 分割成16个部分
kmeans = KMeans(n_clusters=16)
kmeans.fit(img)
label = kmeans.predict(img)
label = label.reshape([width, height])
label_color = (color.label2rgb(label)*255).astype(np.uint8)
label_color = label_color.transpose(1, 0, 2) # 1,2维调换
images = image.fromarray(label_color)
images.save('weixin_mark_color.jpg')

# 创建个新图像，用来保存图像聚类压缩后的结果
img = image.new('RGB', (width, height))
for x in range(width):
    for y in range(height):
        R = kmeans.cluster_centers_[label[x, y], 0]
        G = kmeans.cluster_centers_[label[x, y], 1]
        B = kmeans.cluster_centers_[label[x, y], 2]
        img.putpixel((x, y), (int(R*256)-1, int(G*256)-1, int(B*256)-1))
img.save('weixin_new.jpg')

```

