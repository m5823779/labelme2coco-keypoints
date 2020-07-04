# labelme2coco_keypoint

This tool is to convert all labelme keypoints file to one single coco keypoints file


the following is coco keypoint format

{
    "info": info,
    
    "licenses": [license],
    
    "images": [image],
    
    "annotations": [annotation],
    
    "categories": [category] 
}



-image{
    "id": int,
    "width": int,
    "height": int,
    "file_name": str,
}


-annotation{
    "keypoints": [x1,y1,v1,...],
    "num_keypoints": int,
    "id": int,
    "image_id": int,
    "category_id": int,
    "segmentation": RLE or [polygon],
    "area": float,
    "bbox": [x,y,width,height],
    "iscrowd": 0 or 1,
}


-category{
    "id": int,
    "name": str,
    "supercategory": str,
    "keypoints": [str],
    "skeleton": [edge]
}



notice this version only support v = 0(unlabeled) & v = 2 (can visable and labeled) for each keypoints
and you should label unlabeled point to (0, 0) in labelme


![image](https://github.com/m5823779/labelme2coco_keypoint/blob/master/label_keypoint.gif)
