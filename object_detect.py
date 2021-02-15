# https://qiita.com/sudominoru/items/bf3bd96c6921d9106742
# https://qiita.com/sudominoru/items/bf3bd96c6921d9106742
import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image, ImageFilter

im=Image.open('./PennFudanPed/PNGImages/FudanPed00001.png')
print(im.format, im.size, im.mode) #descrive format, size and mode
# WB convert, rotare , Gausian filter
new_im = im.convert('L').rotate(90).filter(ImageFilter.GaussianBlur())

# save image file ---> qualijty 1 worst to 95 better quality
# new_im.save('data/dst/lenna_square_pillow.jpg', quality=95)

# showing image default operation system image viewer 
# im.show()

mask = Image.open('./PennFudanPed/PedMasks/FudanPed00001_mask.png')
# 各マスクインスタンスは、ゼロからNまでの異なる色を持っています。
# ここで、Nはインスタンス（歩行者）の数です。視覚化を容易にするために、
# マスクにカラーパレットを追加しましょう。
mask.putpalette([
    0, 0, 0, # black background
    255, 0, 0, # index 1 is red
    255, 255, 0, # index 2 is yellow
    255, 153, 0, # index 3 is orange
])
# mask showing!!
# mask.show()

class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # すべての画像ファイルをロードし、並べ替えます
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # 画像とマスクを読み込みます
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        print(img_path)#print image path
        print(mask_path)#print mask path
        img = Image.open(img_path).convert("RGB")
        # 各色は異なるインスタンスに対応し、0が背景であるため、
        # マスクをRGBに変換していないことに注意してください
        mask = Image.open(mask_path)

        mask = np.array(mask)
        # インスタンスは異なる色としてエンコードされます
        obj_ids = np.unique(mask)
        # 最初のIDは背景なので、削除します
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        # 色分けされたマスクをバイナリマスクのセットに分割します
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        # 各マスクのバウンディングボックス座標を取得します
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        # クラスは1つだけです
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        # すべてのインスタンスが混雑していないと仮定します
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)



dataset = PennFudanDataset('PennFudanPed/')
#dataset[0]
imm=dataset[0]
print(imm)

import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots()

target = dataset[0][1]

# 1番目のインスタンスの masks
masks_0 = target['masks'][0,:,:]

# 1番目のインスタンスの boxes
boxes_0 = target['boxes'][0]

# mask を出力します
ax.imshow(masks_0)
# boxes を出力します
ax.add_patch(
     patches.Rectangle(
        (boxes_0[0], boxes_0[1]),boxes_0[2] - boxes_0[0], boxes_0[3] - boxes_0[1],
        edgecolor = 'blue',
        facecolor = 'red',
        fill=True,
        alpha=0.5
     ) )


plt.show()