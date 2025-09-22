import json
from pathlib import Path
import cv2
import yaml
import numpy as np
import copy

from utils.path_processor import path_unifier

def load_config(cfg_path="config.yaml"):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

def crop_labels(labels_raw, image_raw, points_clicked):

    points_clicked = np.array(points_clicked)
    min_x = min(points_clicked[:,0])
    max_x = max(points_clicked[:,0])
    min_y = min(points_clicked[:,1])
    max_y = max(points_clicked[:,1])

    labels_final = []
    for object in labels_raw:
        object_this = []
        for pixel in object:
            if pixel[0] >= min_x and pixel[0] <= max_x and pixel[1] >= min_y and pixel[1] <= max_y:
                object_this.append([pixel[0]-min_x,pixel[1]-min_y])
        
        if object_this:
            labels_final.append(object_this)
    
    image_cropped = image_raw[min_y:max_y,min_x:max_x]

    for object in labels_final:
        image_cropped_copy = copy.deepcopy(image_cropped)
        for pixel in object:
            x = int(pixel[0])
            y = int(pixel[1])
            cv2.circle(image_cropped_copy, (x,y), radius=1, color = (0, 0, 255), thickness=1)
    
        cv2.imshow("WINDOW",image_cropped_copy)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('y'):
            labels_target = object
            break
        else:
            labels_target = None
            continue

    cv2.destroyAllWindows()

    return labels_target, image_cropped



def main(cfg_path="config.yaml"):
    cfg = load_config(cfg_path)

    img_path_pattern    = cfg["image"]["path"]
    window_name         = cfg["image"]["window_name"]
    indices             = [int(n) for n in cfg.get("dataset_indices", [])]
    max_points          = int(cfg["image"]["max_points"])

    r = int(cfg["draw"]["point_radius"])
    color = tuple(int(c) for c in cfg["draw"]["color_bgr"])
    thickness = int(cfg["draw"]["thickness"])

    save_dir = cfg["save_dir"]

    label_path_pattern = cfg["labels"]["path"]
    for n in indices:
        img_path   = Path(img_path_pattern.format(num=n)).expanduser()
        label_path = Path(label_path_pattern.format(num=n)).expanduser()

        with open(label_path, "r") as f:
            label_raw = json.load(f)
            label_unified = path_unifier(label_raw)
            label_unified = [a + b[::-1] for a, b in label_unified]

        img = cv2.imread(str(img_path))
        if img is None:
            raise SystemExit(f"Could not read image: {img_path}")
        
        points_disp = []  
        disp = img
        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(points_disp) < max_points:
                points_disp.append([x, y])

                # cv2.circle(disp, (x, y), r, color, thickness)
                cv2.imshow(window_name, disp)
                print(f"Point {len(points_disp)} (display): ({x}, {y})")

        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, on_mouse)
        cv2.imshow(window_name, disp)

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or len(points_disp) >= max_points:
                break

        cv2.destroyAllWindows()



        label_target, img_crop = crop_labels(label_unified, img, points_disp)

        

        dict_lisa_format = {}
        dict_lisa_format["text"]        = ["PH1",
                                            "PH2", 
                                            "PH3",
                                              "PH4",
                                                "PH5",
                                                  "PH6"]
        dict_lisa_format["is_sentence"] = True
        
        shape_dict           = {}
        shape_dict["label"]      = "target"
        shape_dict["labels"]     = ["target"]
        shape_dict["shape_type"] = "polygon"
        shape_dict["image_name"] = str(f"rs{n}.jpg")
        shape_dict["points"]     = np.asarray(label_target, dtype=int).tolist()
        shape_dict["group_id"]   = None
        shape_dict["group_ids"]  = [None]
        shape_dict["flags"]      = {}

        dict_lisa_format["shapes"] = [shape_dict]

        cv2.imwrite(str(save_dir + f"rs{n}.jpg"), img_crop)
        with open(str(save_dir + f"rs{n}.json"), "w") as f:
            json.dump(dict_lisa_format, f, indent=2)



if __name__ == "__main__":
    main()
