import cv2
import numpy as np
import os
import time
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F
def Normalize(data):
    out=[]
    for e in data:
        if e is not None:
            mn = np.min(e)
            mx = np.max(e - mn)
            if mx <= 0.001:
                out.append(np.zeros_like(e))
            else:
                nm = (e - mn)/(mx)
                out.append(nm)
    return out

def visualize(data):
    out=[]
    for score in data:
        score = score[:, :, np.newaxis]
        score = cv2.normalize(score, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        color_map = cv2.applyColorMap(score, cv2.COLORMAP_JET)
        h,w,c = color_map.shape
        ones = (np.ones(shape=(h,2,3))*255).astype(np.uint8)
        color_map = np.hstack([color_map,ones])
        out.append(color_map)
    return out

def get_response_show(score,shape):
    fig, ax = plt.subplots()
    score = cv2.resize(score ,(shape[1],shape[0]), interpolation=cv2.INTER_CUBIC)
    sns.heatmap(score, cmap='jet', ax=ax, square=True,)
    ax.set_title('a simple figure')
    fig.canvas.draw()
    #X = np.array(fig.canvas.renderer._renderer)[110:110+224, 60:60+224, :3][:, :, ::-1]
    X = np.array(fig.canvas.renderer._renderer)[58:58+370, 106:106+370,  :3][:, :, ::-1]
    X = cv2.resize(X, (shape[1],shape[0]), interpolation=cv2.INTER_CUBIC)
    return X



def show_dimp_reppoint_segm(
        temp_mask,
        temp_image,
        temp_bbox,

        search_mask,
        pred_mask,
        search_image,
        search_bbox,
        pred_bbox,
        corner_map_tl,
        corner_map_br,
        dimp_anno,
        dimp_maps,

        reppoint_init_bboxes,
        reppoint_refine_bboxes,
        reppoint_cls,
        reppoint_init_weight,
        reppoint_refine_weight,

        save_root):
    mean = np.array([0.485, 0.456, 0.406]).reshape(1,1,3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1,1,3)


    # assert shape[-1] == shape[0]
    #================search============

    search_image = search_image.permute(1,2,0).numpy().copy()
    shape = search_image.shape
    search_image = ((search_image*std + mean) * 255).astype(np.uint8)


    search_bbox = search_bbox.numpy()
    search_bbox = search_bbox.astype(np.int32)
    pred_bbox = pred_bbox.numpy() * shape[0]
    pred_bbox = pred_bbox.astype(np.int32)

    search_mask = search_mask.permute(1,2,0).numpy().copy()
    search_mask = (np.concatenate([search_mask]*3,-1)*255).astype(np.uint8)
    search_mask = cv2.rectangle(search_mask,(search_bbox[0],search_bbox[1]),(search_bbox[2],search_bbox[3]),(0,255,0),2)
    search_mask = cv2.resize(search_mask,shape[:2])

    pred_mask = pred_mask.permute(1,2,0).numpy().copy()
    pred_mask = (np.concatenate([pred_mask]*3,-1)*255).astype(np.uint8)
    pred_mask = cv2.rectangle(pred_mask,(search_bbox[0],search_bbox[1]),(search_bbox[2],search_bbox[3]),(0,255,0),2)
    pred_mask = cv2.resize(pred_mask,shape[:2])
    #================search============


    #==========================template
    temp_bbox = temp_bbox.numpy()
    temp_bbox = temp_bbox.astype(np.int32)

    temp_image = temp_image.permute(1,2,0).numpy().copy()
    temp_image = ((temp_image*std + mean) * 255).astype(np.uint8)
    temp_mask = temp_mask.permute(1,2,0).numpy().copy()
    temp_mask = (np.concatenate([temp_mask]*3,-1)*255).astype(np.uint8)
    temp_image = cv2.rectangle(temp_image,(temp_bbox[0],temp_bbox[1]),(temp_bbox[2],temp_bbox[3]),(0,255,0),2)
    temp_image = cv2.resize(temp_image,shape[:2])
    temp_mask = cv2.rectangle(temp_mask,(temp_bbox[0],temp_bbox[1]),(temp_bbox[2],temp_bbox[3]),(0,255,0),2)
    temp_mask = cv2.resize(temp_mask,shape[:2])
    #==========================template


    #=============================
    search_image1 = cv2.rectangle(search_image.copy(),(search_bbox[0],search_bbox[1]),(search_bbox[2],search_bbox[3]),(0,0,255),2)
    search_image1 = cv2.rectangle(search_image1,(pred_bbox[0],pred_bbox[1]),(pred_bbox[2],pred_bbox[3]),(0,255,0),2)
    search_image1 = cv2.putText(search_image1, 'corner',(20,50), cv2.FONT_HERSHEY_SIMPLEX,0.6, (0,0,255), 2, 2)


    corner_map_tl = get_response_show(corner_map_tl,shape)
    corner_map_br = get_response_show(corner_map_br,shape)
    corner_map_tl = cv2.rectangle(corner_map_tl,(search_bbox[0],search_bbox[1]),(search_bbox[2],search_bbox[3]),(0,0,255),2)
    corner_map_br = cv2.rectangle(corner_map_br,(search_bbox[0],search_bbox[1]),(search_bbox[2],search_bbox[3]),(0,0,255),2)

    dimp_anno = dimp_anno.numpy()
    dimp_anno = get_response_show(dimp_anno,shape)
    dimp_anno = cv2.rectangle(dimp_anno,(search_bbox[0],search_bbox[1]),(search_bbox[2],search_bbox[3]),(0,0,255),2)

    vis_search_img = np.hstack([temp_image,search_image1,corner_map_tl,corner_map_br,temp_mask])

    vis_search_img = np.hstack([temp_image,search_image1,temp_mask,pred_mask,search_mask])
    #=============================


    #===============response=====
    dimp_maps = [x.numpy() for x in dimp_maps]
    for i in range(len(dimp_maps)):
        max_score = np.max(dimp_maps[i])
        dimp_maps[i] = get_response_show(dimp_maps[i],shape)
        dimp_maps[i] = cv2.rectangle(dimp_maps[i],(search_bbox[0],search_bbox[1]),(search_bbox[2],search_bbox[3]),(0,0,255),2)
        dimp_maps[i] = cv2.putText(dimp_maps[i], '{}'.format(round(max_score,2)),(20,50), cv2.FONT_HERSHEY_SIMPLEX,0.6, (0,0,255), 2, 2)

    vis_response = np.hstack(dimp_maps[-5:])
    #===============response=====


    #===============dfcos========
    reppoint_cls = reppoint_cls.numpy()
    reppoint_init_bboxes = reppoint_init_bboxes.numpy()
    reppoint_refine_bboxes = reppoint_refine_bboxes.numpy()
    reppoint_init_weight = reppoint_init_weight.numpy()
    reppoint_refine_weight = reppoint_refine_weight.numpy()

    max_score = np.max(reppoint_cls)
    best_idx = np.argmax(reppoint_cls)

    num_qurey = len(reppoint_init_bboxes)
    h = w = int(np.sqrt(num_qurey))

    reppoint_init_weight = get_response_show(reppoint_init_weight.reshape(h,w),shape)
    reppoint_init_weight = cv2.rectangle(reppoint_init_weight,(search_bbox[0],search_bbox[1]),(search_bbox[2],search_bbox[3]),(0,0,255),2)
    reppoint_init_weight = cv2.putText(reppoint_init_weight, 'init_weight',(20,25), cv2.FONT_HERSHEY_SIMPLEX,0.6, (0,0,255), 2, 2)


    reppoint_refine_weight = get_response_show(reppoint_refine_weight.reshape(h,w),shape)
    reppoint_refine_weight = cv2.rectangle(reppoint_refine_weight,(search_bbox[0],search_bbox[1]),(search_bbox[2],search_bbox[3]),(0,0,255),2)
    reppoint_refine_weight = cv2.putText(reppoint_refine_weight, 'refine_weight',(20,25), cv2.FONT_HERSHEY_SIMPLEX,0.6, (0,0,255), 2, 2)


    reppoint_cls = get_response_show(reppoint_cls.reshape(h,w),shape)
    reppoint_cls = cv2.rectangle(reppoint_cls,(search_bbox[0],search_bbox[1]),(search_bbox[2],search_bbox[3]),(0,0,255),2)
    reppoint_cls = cv2.putText(reppoint_cls, 'reppoint_cls={}'.format(round(max_score,2)),(20,25), cv2.FONT_HERSHEY_SIMPLEX,0.6, (0,0,255), 2, 2)

    reppoint_init_bboxes = reppoint_init_bboxes[best_idx].astype(np.int32)
    reppoint_refine_bboxes = reppoint_refine_bboxes[best_idx].astype(np.int32)

    search_image_init = cv2.putText(search_image.copy(), 'init_bboxes',(20,25), cv2.FONT_HERSHEY_SIMPLEX,0.6, (0,255,0), 2, 2)
    search_image_init = cv2.rectangle(search_image_init,(search_bbox[0],search_bbox[1]),(search_bbox[2],search_bbox[3]),(0,0,255),2)
    search_image_init = cv2.rectangle(search_image_init,(reppoint_init_bboxes[0],reppoint_init_bboxes[1]),(reppoint_init_bboxes[2],reppoint_init_bboxes[3]),(0,255,0),2)

    search_image_refine = cv2.putText(search_image.copy(), 'refine_bboxes',(20,25), cv2.FONT_HERSHEY_SIMPLEX,0.6, (0,0,255), 2, 2)
    search_image_refine = cv2.rectangle(search_image_refine,(search_bbox[0],search_bbox[1]),(search_bbox[2],search_bbox[3]),(0,0,255),2)
    search_image_refine = cv2.rectangle(search_image_refine,(reppoint_refine_bboxes[0],reppoint_refine_bboxes[1]),(reppoint_refine_bboxes[2],reppoint_refine_bboxes[3]),(0,255,0),2)

    vis_reppoint = np.hstack([reppoint_cls,search_image_init,search_image_refine,reppoint_init_weight,reppoint_refine_weight])

    #===============dfcos========

    vis_show = np.vstack([vis_search_img,vis_response,vis_reppoint])

    os.makedirs(save_root,exist_ok = True)
    start_time = time.time()
    save_path = os.path.join(save_root,'{}.jpg'.format(start_time))
    cv2.imwrite(save_path,vis_show)







def show_dimp_reppoint_mask(
        temp_mask,
        temp_image,
        temp_bbox,
        search_image,
        search_bbox,
        pred_bbox,
        corner_map_tl,
        corner_map_br,
        dimp_anno,
        dimp_maps,

        reppoint_init_bboxes,
        reppoint_refine_bboxes,
        reppoint_cls,
        reppoint_init_weight,
        reppoint_refine_weight,

        save_root):
    mean = np.array([0.485, 0.456, 0.406]).reshape(1,1,3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1,1,3)


    # assert shape[-1] == shape[0]
    #================search============
    search_image = search_image.permute(1,2,0).numpy().copy()
    shape = search_image.shape
    search_image = ((search_image*std + mean) * 255).astype(np.uint8)
    search_bbox = search_bbox.numpy()
    search_bbox = search_bbox.astype(np.int32)
    pred_bbox = pred_bbox.numpy() * shape[0]
    pred_bbox = pred_bbox.astype(np.int32)
    #================search============


    #==========================template
    temp_bbox = temp_bbox.numpy()
    temp_bbox = temp_bbox.astype(np.int32)

    temp_image = temp_image.permute(1,2,0).numpy().copy()
    temp_image = ((temp_image*std + mean) * 255).astype(np.uint8)
    temp_mask = temp_mask.permute(1,2,0).numpy().copy()
    temp_mask = (np.concatenate([temp_mask]*3,-1)*255).astype(np.uint8)
    temp_image = cv2.rectangle(temp_image,(temp_bbox[0],temp_bbox[1]),(temp_bbox[2],temp_bbox[3]),(0,255,0),2)
    temp_image = cv2.resize(temp_image,shape[:2])
    temp_mask = cv2.rectangle(temp_mask,(temp_bbox[0],temp_bbox[1]),(temp_bbox[2],temp_bbox[3]),(0,255,0),2)
    temp_mask = cv2.resize(temp_mask,shape[:2])
    #==========================template


    #=============================
    search_image1 = cv2.rectangle(search_image.copy(),(search_bbox[0],search_bbox[1]),(search_bbox[2],search_bbox[3]),(0,0,255),2)
    search_image1 = cv2.rectangle(search_image1,(pred_bbox[0],pred_bbox[1]),(pred_bbox[2],pred_bbox[3]),(0,255,0),2)
    search_image1 = cv2.putText(search_image1, 'corner',(20,50), cv2.FONT_HERSHEY_SIMPLEX,0.6, (0,0,255), 2, 2)


    corner_map_tl = get_response_show(corner_map_tl,shape)
    corner_map_br = get_response_show(corner_map_br,shape)
    corner_map_tl = cv2.rectangle(corner_map_tl,(search_bbox[0],search_bbox[1]),(search_bbox[2],search_bbox[3]),(0,0,255),2)
    corner_map_br = cv2.rectangle(corner_map_br,(search_bbox[0],search_bbox[1]),(search_bbox[2],search_bbox[3]),(0,0,255),2)

    dimp_anno = dimp_anno.numpy()
    dimp_anno = get_response_show(dimp_anno,shape)
    dimp_anno = cv2.rectangle(dimp_anno,(search_bbox[0],search_bbox[1]),(search_bbox[2],search_bbox[3]),(0,0,255),2)

    vis_search_img = np.hstack([temp_image,search_image1,corner_map_tl,corner_map_br,temp_mask])
    #=============================


    #===============response=====
    dimp_maps = [x.numpy() for x in dimp_maps]
    for i in range(len(dimp_maps)):
        max_score = np.max(dimp_maps[i])
        dimp_maps[i] = get_response_show(dimp_maps[i],shape)
        dimp_maps[i] = cv2.rectangle(dimp_maps[i],(search_bbox[0],search_bbox[1]),(search_bbox[2],search_bbox[3]),(0,0,255),2)
        dimp_maps[i] = cv2.putText(dimp_maps[i], '{}'.format(round(max_score,2)),(20,50), cv2.FONT_HERSHEY_SIMPLEX,0.6, (0,0,255), 2, 2)

    vis_response = np.hstack(dimp_maps[-5:])
    #===============response=====


    #===============dfcos========
    reppoint_cls = reppoint_cls.numpy()
    reppoint_init_bboxes = reppoint_init_bboxes.numpy()
    reppoint_refine_bboxes = reppoint_refine_bboxes.numpy()
    reppoint_init_weight = reppoint_init_weight.numpy()
    reppoint_refine_weight = reppoint_refine_weight.numpy()

    max_score = np.max(reppoint_cls)
    best_idx = np.argmax(reppoint_cls)

    num_qurey = len(reppoint_init_bboxes)
    h = w = int(np.sqrt(num_qurey))

    reppoint_init_weight = get_response_show(reppoint_init_weight.reshape(h,w),shape)
    reppoint_init_weight = cv2.rectangle(reppoint_init_weight,(search_bbox[0],search_bbox[1]),(search_bbox[2],search_bbox[3]),(0,0,255),2)
    reppoint_init_weight = cv2.putText(reppoint_init_weight, 'init_weight',(20,25), cv2.FONT_HERSHEY_SIMPLEX,0.6, (0,0,255), 2, 2)


    reppoint_refine_weight = get_response_show(reppoint_refine_weight.reshape(h,w),shape)
    reppoint_refine_weight = cv2.rectangle(reppoint_refine_weight,(search_bbox[0],search_bbox[1]),(search_bbox[2],search_bbox[3]),(0,0,255),2)
    reppoint_refine_weight = cv2.putText(reppoint_refine_weight, 'refine_weight',(20,25), cv2.FONT_HERSHEY_SIMPLEX,0.6, (0,0,255), 2, 2)


    reppoint_cls = get_response_show(reppoint_cls.reshape(h,w),shape)
    reppoint_cls = cv2.rectangle(reppoint_cls,(search_bbox[0],search_bbox[1]),(search_bbox[2],search_bbox[3]),(0,0,255),2)
    reppoint_cls = cv2.putText(reppoint_cls, 'reppoint_cls={}'.format(round(max_score,2)),(20,25), cv2.FONT_HERSHEY_SIMPLEX,0.6, (0,0,255), 2, 2)

    reppoint_init_bboxes = reppoint_init_bboxes[best_idx].astype(np.int32)
    reppoint_refine_bboxes = reppoint_refine_bboxes[best_idx].astype(np.int32)

    search_image_init = cv2.putText(search_image.copy(), 'init_bboxes',(20,25), cv2.FONT_HERSHEY_SIMPLEX,0.6, (0,255,0), 2, 2)
    search_image_init = cv2.rectangle(search_image_init,(search_bbox[0],search_bbox[1]),(search_bbox[2],search_bbox[3]),(0,0,255),2)
    search_image_init = cv2.rectangle(search_image_init,(reppoint_init_bboxes[0],reppoint_init_bboxes[1]),(reppoint_init_bboxes[2],reppoint_init_bboxes[3]),(0,255,0),2)

    search_image_refine = cv2.putText(search_image.copy(), 'refine_bboxes',(20,25), cv2.FONT_HERSHEY_SIMPLEX,0.6, (0,0,255), 2, 2)
    search_image_refine = cv2.rectangle(search_image_refine,(search_bbox[0],search_bbox[1]),(search_bbox[2],search_bbox[3]),(0,0,255),2)
    search_image_refine = cv2.rectangle(search_image_refine,(reppoint_refine_bboxes[0],reppoint_refine_bboxes[1]),(reppoint_refine_bboxes[2],reppoint_refine_bboxes[3]),(0,255,0),2)

    vis_reppoint = np.hstack([reppoint_cls,search_image_init,search_image_refine,reppoint_init_weight,reppoint_refine_weight])

    #===============dfcos========

    vis_show = np.vstack([vis_search_img,vis_response,vis_reppoint])

    os.makedirs(save_root,exist_ok = True)
    start_time = time.time()
    save_path = os.path.join(save_root,'{}.jpg'.format(start_time))
    cv2.imwrite(save_path,vis_show)






def show_dimp_reppoint(
        temp_image,
        temp_bbox,
        search_image,
        search_bbox,
        pred_bbox,
        corner_map_tl,
        corner_map_br,
        dimp_anno,
        dimp_maps,

        reppoint_init_bboxes,
        reppoint_refine_bboxes,
        reppoint_cls,
        reppoint_init_weight,
        reppoint_refine_weight,

        save_root):
    mean = np.array([0.485, 0.456, 0.406]).reshape(1,1,3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1,1,3)


    # assert shape[-1] == shape[0]
    #================search============
    search_image = search_image.permute(1,2,0).numpy().copy()
    shape = search_image.shape
    search_image = ((search_image*std + mean) * 255).astype(np.uint8)
    search_bbox = search_bbox.numpy()
    search_bbox = search_bbox.astype(np.int32)
    pred_bbox = pred_bbox.numpy() * shape[0]
    pred_bbox = pred_bbox.astype(np.int32)
    #================search============

    #==========================template
    temp_image = temp_image.permute(1,2,0).numpy().copy()
    temp_image = ((temp_image*std + mean) * 255).astype(np.uint8)
    temp_bbox = temp_bbox.numpy()
    temp_bbox = temp_bbox.astype(np.int32)
    temp_image = cv2.rectangle(temp_image,(temp_bbox[0],temp_bbox[1]),(temp_bbox[2],temp_bbox[3]),(0,255,0),2)
    temp_image = cv2.resize(temp_image,shape[:2])
    #==========================template


    #=============================
    search_image1 = cv2.rectangle(search_image.copy(),(search_bbox[0],search_bbox[1]),(search_bbox[2],search_bbox[3]),(0,0,255),2)
    search_image1 = cv2.rectangle(search_image1,(pred_bbox[0],pred_bbox[1]),(pred_bbox[2],pred_bbox[3]),(0,255,0),2)
    search_image1 = cv2.putText(search_image1, 'corner',(20,50), cv2.FONT_HERSHEY_SIMPLEX,0.6, (0,0,255), 2, 2)


    corner_map_tl = get_response_show(corner_map_tl,shape)
    corner_map_br = get_response_show(corner_map_br,shape)
    corner_map_tl = cv2.rectangle(corner_map_tl,(search_bbox[0],search_bbox[1]),(search_bbox[2],search_bbox[3]),(0,0,255),2)
    corner_map_br = cv2.rectangle(corner_map_br,(search_bbox[0],search_bbox[1]),(search_bbox[2],search_bbox[3]),(0,0,255),2)

    dimp_anno = dimp_anno.numpy()
    dimp_anno = get_response_show(dimp_anno,shape)
    dimp_anno = cv2.rectangle(dimp_anno,(search_bbox[0],search_bbox[1]),(search_bbox[2],search_bbox[3]),(0,0,255),2)

    vis_search_img = np.hstack([temp_image,search_image1,corner_map_tl,corner_map_br,dimp_anno])
    #=============================


    #===============response=====
    dimp_maps = [x.numpy() for x in dimp_maps]
    for i in range(len(dimp_maps)):
        max_score = np.max(dimp_maps[i])
        dimp_maps[i] = get_response_show(dimp_maps[i],shape)
        dimp_maps[i] = cv2.rectangle(dimp_maps[i],(search_bbox[0],search_bbox[1]),(search_bbox[2],search_bbox[3]),(0,0,255),2)
        dimp_maps[i] = cv2.putText(dimp_maps[i], '{}'.format(round(max_score,2)),(20,50), cv2.FONT_HERSHEY_SIMPLEX,0.6, (0,0,255), 2, 2)

    vis_response = np.hstack(dimp_maps[-5:])
    #===============response=====


    #===============dfcos========
    reppoint_cls = reppoint_cls.numpy()
    reppoint_init_bboxes = reppoint_init_bboxes.numpy()
    reppoint_refine_bboxes = reppoint_refine_bboxes.numpy()
    reppoint_init_weight = reppoint_init_weight.numpy()
    reppoint_refine_weight = reppoint_refine_weight.numpy()

    max_score = np.max(reppoint_cls)
    best_idx = np.argmax(reppoint_cls)

    num_qurey = len(reppoint_init_bboxes)
    h = w = int(np.sqrt(num_qurey))

    reppoint_init_weight = get_response_show(reppoint_init_weight.reshape(h,w),shape)
    reppoint_init_weight = cv2.rectangle(reppoint_init_weight,(search_bbox[0],search_bbox[1]),(search_bbox[2],search_bbox[3]),(0,0,255),2)
    reppoint_init_weight = cv2.putText(reppoint_init_weight, 'init_weight',(20,25), cv2.FONT_HERSHEY_SIMPLEX,0.6, (0,0,255), 2, 2)


    reppoint_refine_weight = get_response_show(reppoint_refine_weight.reshape(h,w),shape)
    reppoint_refine_weight = cv2.rectangle(reppoint_refine_weight,(search_bbox[0],search_bbox[1]),(search_bbox[2],search_bbox[3]),(0,0,255),2)
    reppoint_refine_weight = cv2.putText(reppoint_refine_weight, 'refine_weight',(20,25), cv2.FONT_HERSHEY_SIMPLEX,0.6, (0,0,255), 2, 2)


    reppoint_cls = get_response_show(reppoint_cls.reshape(h,w),shape)
    reppoint_cls = cv2.rectangle(reppoint_cls,(search_bbox[0],search_bbox[1]),(search_bbox[2],search_bbox[3]),(0,0,255),2)
    reppoint_cls = cv2.putText(reppoint_cls, 'reppoint_cls={}'.format(round(max_score,2)),(20,25), cv2.FONT_HERSHEY_SIMPLEX,0.6, (0,0,255), 2, 2)

    reppoint_init_bboxes = reppoint_init_bboxes[best_idx].astype(np.int32)
    reppoint_refine_bboxes = reppoint_refine_bboxes[best_idx].astype(np.int32)

    search_image_init = cv2.putText(search_image.copy(), 'init_bboxes',(20,25), cv2.FONT_HERSHEY_SIMPLEX,0.6, (0,255,0), 2, 2)
    search_image_init = cv2.rectangle(search_image_init,(search_bbox[0],search_bbox[1]),(search_bbox[2],search_bbox[3]),(0,0,255),2)
    search_image_init = cv2.rectangle(search_image_init,(reppoint_init_bboxes[0],reppoint_init_bboxes[1]),(reppoint_init_bboxes[2],reppoint_init_bboxes[3]),(0,255,0),2)

    search_image_refine = cv2.putText(search_image.copy(), 'refine_bboxes',(20,25), cv2.FONT_HERSHEY_SIMPLEX,0.6, (0,0,255), 2, 2)
    search_image_refine = cv2.rectangle(search_image_refine,(search_bbox[0],search_bbox[1]),(search_bbox[2],search_bbox[3]),(0,0,255),2)
    search_image_refine = cv2.rectangle(search_image_refine,(reppoint_refine_bboxes[0],reppoint_refine_bboxes[1]),(reppoint_refine_bboxes[2],reppoint_refine_bboxes[3]),(0,255,0),2)

    vis_reppoint = np.hstack([reppoint_cls,search_image_init,search_image_refine,reppoint_init_weight,reppoint_refine_weight])

    #===============dfcos========

    vis_show = np.vstack([vis_search_img,vis_response,vis_reppoint])

    os.makedirs(save_root,exist_ok = True)
    start_time = time.time()
    save_path = os.path.join(save_root,'{}.jpg'.format(start_time))
    cv2.imwrite(save_path,vis_show)





def show_dimp_dfcos(
        temp_image,
        temp_bbox,
        search_image,
        search_bbox,
        pred_bbox,
        corner_map_tl,
        corner_map_br,
        dimp_anno,
        dimp_maps,

        dfcos_cls_fc,
        dfcos_cls_conv,
        dfcos_bbox_fc,
        dfcos_bbox_conv,
        label_cls,

        save_root):
    mean = np.array([0.485, 0.456, 0.406]).reshape(1,1,3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1,1,3)


    # assert shape[-1] == shape[0]
    #================search============
    search_image = search_image.permute(1,2,0).numpy().copy()
    shape = search_image.shape
    search_image = ((search_image*std + mean) * 255).astype(np.uint8)
    search_bbox = search_bbox.numpy()
    search_bbox = search_bbox.astype(np.int32)
    pred_bbox = pred_bbox.numpy() * shape[0]
    pred_bbox = pred_bbox.astype(np.int32)
    #================search============

    #==========================template
    temp_image = temp_image.permute(1,2,0).numpy().copy()
    temp_image = ((temp_image*std + mean) * 255).astype(np.uint8)
    temp_bbox = temp_bbox.numpy()
    temp_bbox = temp_bbox.astype(np.int32)
    temp_image = cv2.rectangle(temp_image,(temp_bbox[0],temp_bbox[1]),(temp_bbox[2],temp_bbox[3]),(0,255,0),2)
    temp_image = cv2.resize(temp_image,shape[:2])
    #==========================template


    #=============================
    search_image1 = cv2.rectangle(search_image.copy(),(search_bbox[0],search_bbox[1]),(search_bbox[2],search_bbox[3]),(0,0,255),2)
    search_image1 = cv2.rectangle(search_image1,(pred_bbox[0],pred_bbox[1]),(pred_bbox[2],pred_bbox[3]),(0,255,0),2)
    search_image1 = cv2.putText(search_image1, 'corner',(20,50), cv2.FONT_HERSHEY_SIMPLEX,0.6, (0,0,255), 2, 2)


    corner_map_tl = get_response_show(corner_map_tl,shape)
    corner_map_br = get_response_show(corner_map_br,shape)
    corner_map_tl = cv2.rectangle(corner_map_tl,(search_bbox[0],search_bbox[1]),(search_bbox[2],search_bbox[3]),(0,0,255),2)
    corner_map_br = cv2.rectangle(corner_map_br,(search_bbox[0],search_bbox[1]),(search_bbox[2],search_bbox[3]),(0,0,255),2)

    dimp_anno = dimp_anno.numpy()
    dimp_anno = get_response_show(dimp_anno,shape)
    dimp_anno = cv2.rectangle(dimp_anno,(search_bbox[0],search_bbox[1]),(search_bbox[2],search_bbox[3]),(0,0,255),2)

    vis_search_img = np.hstack([temp_image,search_image1,corner_map_tl,corner_map_br,dimp_anno])
    #=============================


    #===============response=====
    dimp_maps = [x.numpy() for x in dimp_maps]
    for i in range(len(dimp_maps)):
        max_score = np.max(dimp_maps[i])
        dimp_maps[i] = get_response_show(dimp_maps[i],shape)
        dimp_maps[i] = cv2.rectangle(dimp_maps[i],(search_bbox[0],search_bbox[1]),(search_bbox[2],search_bbox[3]),(0,0,255),2)
        dimp_maps[i] = cv2.putText(dimp_maps[i], '{}'.format(round(max_score,2)),(20,50), cv2.FONT_HERSHEY_SIMPLEX,0.6, (0,0,255), 2, 2)

    vis_response = np.hstack(dimp_maps[-5:])
    #===============response=====


    #===============dfcos========
    dfcos_cls_fc = dfcos_cls_fc.numpy()
    dfcos_bbox_fc = dfcos_bbox_fc.numpy()
    max_score_fc = np.max(dfcos_cls_fc)
    best_idx_fc = np.argmax(dfcos_cls_fc)

    dfcos_cls_conv = dfcos_cls_conv.numpy()
    dfcos_bbox_conv = dfcos_bbox_conv.numpy()
    max_score_conv = np.max(dfcos_cls_conv)
    best_idx_conv = np.argmax(dfcos_cls_conv)


    num_qurey = len(dfcos_cls_fc)
    h = w = int(np.sqrt(num_qurey))


    dfcos_cls_fc = get_response_show(dfcos_cls_fc.reshape(h,w),shape)
    dfcos_cls_conv = get_response_show(dfcos_cls_conv.reshape(h,w),shape)

    dfcos_cls_fc = cv2.rectangle(dfcos_cls_fc,(search_bbox[0],search_bbox[1]),(search_bbox[2],search_bbox[3]),(0,0,255),2)
    dfcos_cls_fc = cv2.putText(dfcos_cls_fc, 'dfcos_fc={}'.format(round(max_score_fc,2)),(20,25), cv2.FONT_HERSHEY_SIMPLEX,0.6, (0,0,255), 2, 2)

    dfcos_cls_conv = cv2.rectangle(dfcos_cls_conv,(search_bbox[0],search_bbox[1]),(search_bbox[2],search_bbox[3]),(0,0,255),2)
    dfcos_cls_conv = cv2.putText(dfcos_cls_conv, 'dfcos_conv={}'.format(round(max_score_conv,2)),(20,25), cv2.FONT_HERSHEY_SIMPLEX,0.6, (0,0,255), 2, 2)

    search_image = cv2.rectangle(search_image.copy(),(search_bbox[0],search_bbox[1]),(search_bbox[2],search_bbox[3]),(0,0,255),2)

    dfcos_bbox_conv_tmp = dfcos_bbox_conv[best_idx_fc].astype(np.int32)
    dfcos_bbox_fc_tmp = dfcos_bbox_fc[best_idx_fc].astype(np.int32)
    search_image2 = cv2.putText(search_image.copy(), 'dfcos_idx_fc',(20,25), cv2.FONT_HERSHEY_SIMPLEX,0.6, (0,0,255), 2, 2)
    search_image2 = cv2.putText(search_image2, 'dfcos_bbox_conv',(20,50), cv2.FONT_HERSHEY_SIMPLEX,0.6, (0,255,0), 2, 2)
    search_image2 = cv2.putText(search_image2, 'dfcos_bbox_fc',(20,75), cv2.FONT_HERSHEY_SIMPLEX,0.6, (255,0,0), 2, 2)
    search_image2 = cv2.rectangle(search_image2,(dfcos_bbox_conv_tmp[0],dfcos_bbox_conv_tmp[1]),(dfcos_bbox_conv_tmp[2],dfcos_bbox_conv_tmp[3]),(0,255,0),2)
    search_image2 = cv2.rectangle(search_image2,(dfcos_bbox_fc_tmp[0],dfcos_bbox_fc_tmp[1]),(dfcos_bbox_fc_tmp[2],dfcos_bbox_fc_tmp[3]),(255,0,0),2)


    dfcos_bbox_conv_tmp = dfcos_bbox_conv[best_idx_conv].astype(np.int32)
    dfcos_bbox_fc_tmp = dfcos_bbox_fc[best_idx_conv].astype(np.int32)
    search_image3 = cv2.putText(search_image.copy(), 'dfcos_idx_conv',(20,25), cv2.FONT_HERSHEY_SIMPLEX,0.6, (0,0,255), 2, 2)
    search_image3 = cv2.putText(search_image3, 'dfcos_bbox_conv',(20,50), cv2.FONT_HERSHEY_SIMPLEX,0.6, (0,255,0), 2, 2)
    search_image3 = cv2.putText(search_image3, 'dfcos_bbox_fc',(20,75), cv2.FONT_HERSHEY_SIMPLEX,0.6, (255,0,0), 2, 2)
    search_image3 = cv2.rectangle(search_image3,(dfcos_bbox_conv_tmp[0],dfcos_bbox_conv_tmp[1]),(dfcos_bbox_conv_tmp[2],dfcos_bbox_conv_tmp[3]),(0,255,0),2)
    search_image3 = cv2.rectangle(search_image3,(dfcos_bbox_fc_tmp[0],dfcos_bbox_fc_tmp[1]),(dfcos_bbox_fc_tmp[2],dfcos_bbox_fc_tmp[3]),(255,0,0),2)



    label_cls = (np.stack([label_cls]*3,-1)*255).astype(np.uint8).copy()
    # print(label_cls.shape)
    label_cls = cv2.resize(label_cls,(shape[1],shape[0]))
    label_cls = cv2.rectangle(label_cls,(search_bbox[0],search_bbox[1]),(search_bbox[2],search_bbox[3]),(0,0,255),2)


    vis_dfcos = np.hstack([search_image2,dfcos_cls_fc,search_image3,dfcos_cls_conv,label_cls])

    #===============dfcos========

    vis_show = np.vstack([vis_search_img,vis_response,vis_dfcos])

    os.makedirs(save_root,exist_ok = True)
    start_time = time.time()
    save_path = os.path.join(save_root,'{}.jpg'.format(start_time))
    cv2.imwrite(save_path,vis_show)




def show_dimp(
        temp_image,
        temp_bbox,
        search_image,
        search_bbox,
        pred_bbox,
        corner_map_tl,
        corner_map_br,
        dimp_anno,
        dimp_maps,
        save_root):
    mean = np.array([0.485, 0.456, 0.406]).reshape(1,1,3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1,1,3)


    # assert shape[-1] == shape[0]
    #================search============
    search_image = search_image.permute(1,2,0).numpy().copy()
    shape = search_image.shape
    search_image = ((search_image*std + mean) * 255).astype(np.uint8)
    search_bbox = search_bbox.numpy()
    search_bbox = search_bbox.astype(np.int32)
    pred_bbox = pred_bbox.numpy() * shape[0]
    pred_bbox = pred_bbox.astype(np.int32)
    #================search============

    #==========================template
    temp_image = temp_image.permute(1,2,0).numpy().copy()
    temp_image = ((temp_image*std + mean) * 255).astype(np.uint8)
    temp_bbox = temp_bbox.numpy()
    temp_bbox = temp_bbox.astype(np.int32)
    temp_image = cv2.rectangle(temp_image,(temp_bbox[0],temp_bbox[1]),(temp_bbox[2],temp_bbox[3]),(0,255,0),2)
    temp_image = cv2.resize(temp_image,shape[:2])
    #==========================template


    #=============================
    search_image = cv2.rectangle(search_image,(search_bbox[0],search_bbox[1]),(search_bbox[2],search_bbox[3]),(0,0,255),2)
    search_image = cv2.rectangle(search_image,(pred_bbox[0],pred_bbox[1]),(pred_bbox[2],pred_bbox[3]),(0,255,0),2)

    corner_map_tl = get_response_show(corner_map_tl,shape)
    corner_map_br = get_response_show(corner_map_br,shape)
    corner_map_tl = cv2.rectangle(corner_map_tl,(search_bbox[0],search_bbox[1]),(search_bbox[2],search_bbox[3]),(0,0,255),2)
    corner_map_br = cv2.rectangle(corner_map_br,(search_bbox[0],search_bbox[1]),(search_bbox[2],search_bbox[3]),(0,0,255),2)
    #=============================


    #===============response=====
    dimp_anno = dimp_anno.numpy()
    dimp_anno = get_response_show(dimp_anno,shape)
    dimp_anno = cv2.rectangle(dimp_anno,(search_bbox[0],search_bbox[1]),(search_bbox[2],search_bbox[3]),(0,0,255),2)


    dimp_maps = [x.numpy() for x in dimp_maps]
    for i in range(len(dimp_maps)):
        dimp_maps[i] = get_response_show(dimp_maps[i],shape)
        dimp_maps[i] = cv2.rectangle(dimp_maps[i],(search_bbox[0],search_bbox[1]),(search_bbox[2],search_bbox[3]),(0,0,255),2)
    #===============response=====

    vis_search_img = np.hstack([temp_image,search_image,corner_map_tl,corner_map_br,dimp_anno])
    vis_response = np.hstack(dimp_maps[-5:])

    vis_show = np.vstack([vis_search_img,vis_response])

    os.makedirs(save_root,exist_ok = True)
    start_time = time.time()
    save_path = os.path.join(save_root,'{}.jpg'.format(start_time))
    cv2.imwrite(save_path,vis_show)




def show_temporal(image_temp,bbox_temp,
                images_search,bboxes_search,
                respones,
                pred_respones,pred_scores,
                pred_boxes, pred_masks,
                save_root = '/home/tiger/tracking_code/TransT_M_ori/ltr'):
    # images_search [num_sequence, c, h, w]
    # bboxes_search [num_sequence, 4]
    # respones [num_sequence, h, w]

    mean = np.array([0.485, 0.456, 0.406]).reshape(1,1,3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1,1,3)

    #================template============
    image_temp = image_temp.permute(1,2,0).detach().cpu().numpy().copy()
    image_temp = ((image_temp*std + mean) * 255).astype(np.uint8)
    bbox_temp = bbox_temp.detach().cpu().numpy()
    bbox_temp[2] = bbox_temp[0] + bbox_temp[2]
    bbox_temp[3] = bbox_temp[1] + bbox_temp[3]
    bbox_temp = bbox_temp.astype(np.int32)
    #================template============

    #================search============
    images_search = images_search.permute(0,2,3,1).detach().cpu().numpy().copy()
    images_search = ((images_search*std.reshape(1,1,1,3) + mean.reshape(1,1,1,3)) * 255).astype(np.uint8)
    num_sequence,h,w,c = images_search.shape
    bboxes_search = bboxes_search.detach().cpu().numpy()
    bboxes_search[:,2] = bboxes_search[:,0] + bboxes_search[:,2]
    bboxes_search[:,3] = bboxes_search[:,1] + bboxes_search[:,3]
    bboxes_search = bboxes_search.astype(np.int32)
    #================search============

    #================response============
    respones = respones.detach().cpu().numpy()
    pred_respones = pred_respones.squeeze().detach().cpu().numpy()
    #================response============

    #================other_pred
    pred_masks = pred_masks.sigmoid().detach().cpu().numpy().squeeze()
    pred_masks = np.stack([(pred_masks*255).astype(np.uint8)]*3,-1)

    pred_scores = F.softmax(pred_scores, dim=1).data[:, 0].cpu().numpy()
    pred_boxes = pred_boxes.data.cpu().numpy()
    best_idx = np.argmax(pred_scores)
    pred_boxes = pred_boxes[best_idx]
    pred_boxes = pred_boxes * h
    pred_boxes[0] = pred_boxes[0] - pred_boxes[2]/2
    pred_boxes[1] = pred_boxes[1] - pred_boxes[3]/2
    pred_boxes[2] = pred_boxes[0] + pred_boxes[2]
    pred_boxes[3] = pred_boxes[1] + pred_boxes[3]
    pred_boxes = pred_boxes.astype(np.int32)

    feature_z = int(np.sqrt(len(pred_scores)))
    pred_scores = pred_scores.reshape(feature_z,feature_z)

    response_list = [pred_scores,pred_respones]
    name_list = ['scores','respones']
    #================other_pred


    # pred_bboxes_search = pred_bboxes_search.detach().cpu().numpy()[0]
    #==========================other_list=========
    image_temp = cv2.rectangle(image_temp,(bbox_temp[0],bbox_temp[1]),(bbox_temp[2],bbox_temp[3]),(0,0,255),2)
    image_temp = cv2.resize(image_temp,(w,h))
    image_temp = cv2.putText(image_temp, 'template',(20,50), cv2.FONT_HERSHEY_SIMPLEX,0.6, (0,0,255), 2, 2)

    search_tmp = cv2.rectangle(images_search[-1],(bboxes_search[-1][0],bboxes_search[-1][1]),(bboxes_search[-1][2],bboxes_search[-1][3]),(0,0,255),2)
    search_tmp = cv2.rectangle(search_tmp,(pred_boxes[0],pred_boxes[1]),(pred_boxes[2],pred_boxes[3]),(0,255,0),2)

    pred_masks = cv2.resize(pred_masks,(w,h))
    pred_masks = cv2.rectangle(pred_masks,(bboxes_search[-1][0],bboxes_search[-1][1]),(bboxes_search[-1][2],bboxes_search[-1][3]),(0,0,255),2)

    pred_list = [image_temp,search_tmp,pred_masks]
    for name,respone in zip(name_list,response_list):
        max_score = np.max(respone)
        respone_tmp = Normalize([respone])[0]
        respone_tmp = get_response_show(respone_tmp,shape = (h,w))
        respone_tmp = cv2.rectangle(respone_tmp,(bboxes_search[-1][0],bboxes_search[-1][1]),(bboxes_search[-1][2],bboxes_search[-1][3]),(0,0,255),2)
        respone_tmp = cv2.putText(respone_tmp, '{}={}'.format(name,round(max_score,2)),(20,50), cv2.FONT_HERSHEY_SIMPLEX,0.6, (0,0,255), 2, 2)
        pred_list.append(respone_tmp)
    pred_list = pred_list + [np.zeros_like(search_tmp)]*(num_sequence - len(pred_list))
    pred_list = np.hstack(pred_list)
    #==========================other_list=========

    #=========================label_list===========
    images_search_list = []
    for i in range(num_sequence):
        search_tmp = cv2.rectangle(images_search[i],(bboxes_search[i][0],bboxes_search[i][1]),(bboxes_search[i][2],bboxes_search[i][3]),(0,0,255),2)
        images_search_list.append(search_tmp)
    images_search_list = np.hstack(images_search_list)

    respones_list = []
    for i in range(num_sequence):
        max_score = np.max(respones[i])
        respone_tmp = Normalize([respones[i]])[0]
        respone_tmp = get_response_show(respone_tmp,shape = (h,w))
        respone_tmp = cv2.rectangle(respone_tmp,(bboxes_search[i][0],bboxes_search[i][1]),(bboxes_search[i][2],bboxes_search[i][3]),(0,0,255),2)
        respone_tmp = cv2.putText(respone_tmp, 's={}'.format(round(max_score,2)),(20,50), cv2.FONT_HERSHEY_SIMPLEX,0.6, (0,0,255), 2, 2)
        respones_list.append(respone_tmp)
    respones_list = np.hstack(respones_list)
    #=========================label_list===========


    show_img = np.vstack([pred_list,images_search_list,respones_list])
    os.makedirs(save_root,exist_ok = True)
    start_time = time.time()
    save_path = os.path.join(save_root,'{}.jpg'.format(start_time))
    cv2.imwrite(save_path,show_img)











def show_base(image_temp,bbox_temp,image_search,bbox_search,respones,save_root = '/home/tiger/tracking_code/TransT_M_ori/ltr'):
    mean = np.array([0.485, 0.456, 0.406]).reshape(1,1,3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1,1,3)
    image_temp = image_temp.permute(1,2,0).detach().cpu().numpy().copy()
    bbox_temp = bbox_temp.detach().cpu().numpy()
    bbox_temp[0] = max(bbox_temp[0],0)
    bbox_temp[1] = max(bbox_temp[1],0)
    bbox_temp[2] = bbox_temp[0] + bbox_temp[2]
    bbox_temp[3] = bbox_temp[1] + bbox_temp[3]
    bbox_temp = bbox_temp.astype(np.int32)
    image_search = image_search.permute(1,2,0).detach().cpu().numpy().copy()
    bbox_search = bbox_search.detach().cpu().numpy()
    bbox_search[0] = max(bbox_search[0],0)
    bbox_search[1] = max(bbox_search[1],0)
    bbox_search[2] = bbox_search[0] + bbox_search[2]
    bbox_search[3] = bbox_search[1] + bbox_search[3]
    bbox_search = bbox_search.astype(np.int32)


    # pred_bbox_search = pred_bbox_search.detach().cpu().numpy()[0]

    image_search = ((image_search*std + mean) * 255).astype(np.uint8)
    h,w,c = image_search.shape
    image_search = cv2.rectangle(image_search,(bbox_search[0],bbox_search[1]),(bbox_search[2],bbox_search[3]),(0,0,255),2)


    image_temp = ((image_temp*std + mean) * 255).astype(np.uint8)
    image_temp = cv2.rectangle(image_temp,(bbox_temp[0],bbox_temp[1]),(bbox_temp[2],bbox_temp[3]),(0,0,255),2)
    image_temp = cv2.resize(image_temp,(w,h))




    if respones is not None:
        respones = respones.detach().cpu().numpy()
        respones = visualize([respones])[0]
        respones = cv2.resize(respones,(w,h))
        respones = cv2.rectangle(respones,(bbox_search[0],bbox_search[1]),(bbox_search[2],bbox_search[3]),(0,0,255),2)
        show_img = np.hstack([image_temp,image_search,respones])
    else:
        show_img = np.hstack([image_temp,image_search])

    os.makedirs(save_root,exist_ok = True)
    start_time = time.time()
    save_path = os.path.join(save_root,'{}.jpg'.format(start_time))

    cv2.imwrite(save_path,show_img)











