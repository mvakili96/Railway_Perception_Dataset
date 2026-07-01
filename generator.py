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


def get_crop_bounds_from_points(points_clicked):
    points_clicked = np.array(points_clicked)
    min_x = int(min(points_clicked[:, 0]))
    max_x = int(max(points_clicked[:, 0]))
    min_y = int(min(points_clicked[:, 1]))
    max_y = int(max(points_clicked[:, 1]))
    return min_x, min_y, max_x, max_y


def get_fixed_crop_bounds(image_shape, crop_size=1024):
    image_h, image_w = image_shape[:2]
    if image_h < crop_size or image_w < crop_size:
        raise SystemExit(
            f"Fixed crop of {crop_size}x{crop_size} requires image size at least "
            f"{crop_size}x{crop_size}, but got {image_w}x{image_h}."
        )

    min_x = (image_w - crop_size) // 2
    max_x = min_x + crop_size
    max_y = image_h
    min_y = max_y - crop_size
    return min_x, min_y, max_x, max_y


def polygon_to_mask(points, shape_hw):
    mask = np.zeros(shape_hw, dtype=np.uint8)
    if points and len(points) >= 3:
        polygon = np.asarray(points, dtype=np.int32)
        cv2.fillPoly(mask, [polygon], 1)
    return mask


def build_full_track_bed_masks(labels_raw, image_shape):
    image_h, image_w = image_shape[:2]
    return [polygon_to_mask(label, (image_h, image_w)) for label in labels_raw]


def build_weight_map(full_masks, ego_full_idx, overlap_full_idx, crop_bounds):
    min_x, min_y, max_x, max_y = crop_bounds
    crop_h = max_y - min_y
    crop_w = max_x - min_x

    if not full_masks:
        return np.zeros((crop_h, crop_w), dtype=np.uint8), np.zeros((crop_h, crop_w), dtype=np.uint8)

    cropped_masks = [mask[min_y:max_y, min_x:max_x] for mask in full_masks]
    all_track_beds = np.clip(np.sum(cropped_masks, axis=0), 0, 1).astype(np.uint8)
    ego_mask = cropped_masks[ego_full_idx]
    overlap_mask = cropped_masks[overlap_full_idx]
    shared_with_selected = ((ego_mask == 1) & (overlap_mask == 1)).astype(np.uint8)
    weight_map = np.logical_xor(ego_mask == 1, overlap_mask == 1).astype(np.uint8)
    weight_map[all_track_beds == 0] = 0

    return weight_map, shared_with_selected


def visualize_weight_map(image_cropped, weight_map, shared_with_selected):
    vis = np.zeros_like(image_cropped)
    vis[weight_map == 1] = (255, 255, 255)
    vis[shared_with_selected == 1] = (0, 0, 255)

    overlay = cv2.addWeighted(image_cropped, 0.7, vis, 0.6, 0)
    cv2.imshow("WEIGHT_MAP", overlay)
    cv2.waitKey(0)
    cv2.destroyWindow("WEIGHT_MAP")


def ask_weight_map_mode(image_cropped, use_weight_key, zero_weight_key):
    image_cropped_copy = copy.deepcopy(image_cropped)
    instructions = [
        "Do you want weight map with this?",
        f"{use_weight_key}: yes, use two-route weighting",
        f"{zero_weight_key}: no, save all-zero weight map",
    ]

    for line_idx, text in enumerate(instructions):
        y = 24 + line_idx * 16
        cv2.putText(image_cropped_copy, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image_cropped_copy, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.imshow("WINDOW", image_cropped_copy)
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord(use_weight_key):
            return True
        if key == ord(zero_weight_key):
            return False


def render_selection_overlay(
    image_cropped,
    polygon_points,
    route_idx,
    total_routes,
    ego_display_idx,
    overlap_display_idx,
    ego_key,
    overlap_key,
    require_overlap_selection,
):
    image_cropped_copy = copy.deepcopy(image_cropped)
    for pixel in polygon_points:
        x = int(pixel[0])
        y = int(pixel[1])
        cv2.circle(image_cropped_copy, (x, y), radius=1, color=(0, 0, 255), thickness=1)

    status_ego = "unset" if ego_display_idx is None else str(ego_display_idx + 1)
    status_overlap = "unset" if overlap_display_idx is None else str(overlap_display_idx + 1)
    instructions = [
        f"Route {route_idx + 1}/{total_routes}",
        f"{ego_key}: ego route ({status_ego})",
    ]
    if require_overlap_selection:
        instructions.append(f"{overlap_key}: overlap route ({status_overlap})")
    instructions.append("other key: skip, q: cancel")

    for line_idx, text in enumerate(instructions):
        y = 24 + line_idx * 16
        cv2.putText(image_cropped_copy, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image_cropped_copy, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)

    return image_cropped_copy


def crop_labels(labels_raw, image_raw, crop_bounds):
    min_x, min_y, max_x, max_y = crop_bounds

    labels_final = []
    labels_indices = []
    for obj_idx, obj in enumerate(labels_raw):
        object_this = []
        for pixel in obj:
            if pixel[0] >= min_x and pixel[0] < max_x and pixel[1] >= min_y and pixel[1] < max_y:
                object_this.append([pixel[0] - min_x, pixel[1] - min_y])

        if object_this:
            labels_final.append(object_this)
            labels_indices.append(obj_idx)

    image_cropped = image_raw[min_y:max_y, min_x:max_x]
    return labels_final, labels_indices, image_cropped, (min_x, min_y, max_x, max_y)


def select_routes(image_cropped, labels_final, labels_indices, ego_key, overlap_key, require_overlap_selection):
    labels_target = None
    ego_full_idx = None
    overlap_full_idx = None
    ego_display_idx = None
    overlap_display_idx = None

    for idx, obj in enumerate(labels_final):
        image_cropped_copy = render_selection_overlay(
            image_cropped=image_cropped,
            polygon_points=obj,
            route_idx=idx,
            total_routes=len(labels_final),
            ego_display_idx=ego_display_idx,
            overlap_display_idx=overlap_display_idx,
            ego_key=ego_key,
            overlap_key=overlap_key,
            require_overlap_selection=require_overlap_selection,
        )
        cv2.imshow("WINDOW", image_cropped_copy)
        key = cv2.waitKey(0) & 0xFF
        if key == ord(ego_key):
            labels_target = obj
            ego_full_idx = labels_indices[idx]
            ego_display_idx = idx
            print(f"Selected route {idx + 1} as ego-path.")
        elif require_overlap_selection and key == ord(overlap_key):
            overlap_full_idx = labels_indices[idx]
            overlap_display_idx = idx
            print(f"Selected route {idx + 1} as overlap route.")
        elif key == ord('q'):
            break

        if ego_full_idx is not None and (not require_overlap_selection or overlap_full_idx is not None):
            if require_overlap_selection and ego_full_idx == overlap_full_idx:
                print("Ego-path and overlap route must be different. Keep selecting.")
                overlap_full_idx = None
                overlap_display_idx = None
                continue
            break

    cv2.destroyAllWindows()
    return labels_target, ego_full_idx, overlap_full_idx


def main(cfg_path="config.yaml"):
    cfg = load_config(cfg_path)

    img_path_pattern = cfg["image"]["path"]
    window_name = cfg["image"]["window_name"]
    indices = [int(n) for n in cfg.get("dataset_indices", [])]
    max_points = int(cfg["image"]["max_points"])
    use_fixed_crop = bool(cfg["image"].get("use_fixed_crop", False))

    r = int(cfg["draw"]["point_radius"])
    color = tuple(int(c) for c in cfg["draw"]["color_bgr"])
    thickness = int(cfg["draw"]["thickness"])

    save_dir = cfg["save_dir"]
    save_dir_path = Path(save_dir).expanduser()
    save_dir_path.mkdir(parents=True, exist_ok=True)

    weight_cfg = cfg.get("weight_map", {})
    save_weight_map = bool(weight_cfg.get("save", True))
    visualize_weight_map_flag = bool(weight_cfg.get("visualize", False))
    weight_suffix = str(weight_cfg.get("filename_suffix", "_weight.png"))
    ego_key = str(weight_cfg.get("ego_select_key", "y"))[:1]
    overlap_key = str(weight_cfg.get("overlap_select_key", "o"))[:1]
    use_weight_key = str(weight_cfg.get("use_weight_key", "d"))[:1]
    zero_weight_key = str(weight_cfg.get("zero_weight_key", "a"))[:1]
    if ego_key == overlap_key:
        raise SystemExit("weight_map.ego_select_key and weight_map.overlap_select_key must be different.")
    weight_map_dir = save_dir_path / "Weight_Maps"
    if save_weight_map:
        weight_map_dir.mkdir(parents=True, exist_ok=True)

    label_path_pattern = cfg["labels"]["path"]
    for n in indices:
        img_path = Path(img_path_pattern.format(num=n)).expanduser()
        label_path = Path(label_path_pattern.format(num=n)).expanduser()

        with open(label_path, "r") as f:
            label_raw = json.load(f)
            label_unified = path_unifier(label_raw)
            label_unified = [a + b[::-1] for a, b in label_unified]

        img = cv2.imread(str(img_path))
        if img is None:
            raise SystemExit(f"Could not read image: {img_path}")

        full_track_bed_masks = build_full_track_bed_masks(label_unified, img.shape)

        if use_fixed_crop:
            crop_bounds = get_fixed_crop_bounds(img.shape, crop_size=1024)
        else:
            points_disp = []
            disp = img

            def on_mouse(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN and len(points_disp) < max_points:
                    points_disp.append([x, y])
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

            if len(points_disp) != max_points:
                raise SystemExit(
                    f"Expected {max_points} clicked points for manual crop, but got {len(points_disp)}."
                )
            crop_bounds = get_crop_bounds_from_points(points_disp)

        labels_final, labels_indices, img_crop, crop_bounds = crop_labels(label_unified, img, crop_bounds)

        require_overlap_selection = ask_weight_map_mode(img_crop, use_weight_key, zero_weight_key)

        label_target, ego_full_idx, overlap_full_idx = select_routes(
            img_crop,
            labels_final,
            labels_indices,
            ego_key,
            overlap_key,
            require_overlap_selection,
        )

        if label_target is None or ego_full_idx is None:
            raise SystemExit(
                f"Selection incomplete. Press '{ego_key}' for the ego-path to save the sample."
            )

        if require_overlap_selection:
            if overlap_full_idx is None:
                raise SystemExit(
                    f"Selection incomplete. Press '{ego_key}' for the ego-path and '{overlap_key}' for the overlap route."
                )
            weight_map, shared_with_selected = build_weight_map(
                full_track_bed_masks,
                ego_full_idx,
                overlap_full_idx,
                crop_bounds,
            )
        else:
            crop_h = crop_bounds[3] - crop_bounds[1]
            crop_w = crop_bounds[2] - crop_bounds[0]
            weight_map = np.zeros((crop_h, crop_w), dtype=np.uint8)
            shared_with_selected = np.zeros_like(weight_map)

        dict_lisa_format = {}
        dict_lisa_format["text"] = [
            "If the right blade is open with a visible rail gap, the right route is the ego-route, and if the left blade is open with a visible rail gap, the left route is the ego-route. Which route corresponds to the ego-route through this switch?",
            "Based on the blade positions in this switch, which route corresponds to the route the train takes?",
            "Which path corresponds to the ego-path if the rails of the ego-path are continuously connected?",
            "By examining the switch geometry, which route corresponds to the route that remains continuous for the train?",
            "Which route corresponds to the active route if the rails of the ego-path are continuously connected and the open blade indicates the selected path?"
            ]
        dict_lisa_format["is_sentence"] = True

        shape_dict = {}
        shape_dict["label"] = "target"
        shape_dict["labels"] = ["target"]
        shape_dict["shape_type"] = "polygon"
        shape_dict["image_name"] = str(f"rs{n}.jpg")
        shape_dict["points"] = np.asarray(label_target, dtype=int).tolist()
        shape_dict["group_id"] = None
        shape_dict["group_ids"] = [None]
        shape_dict["flags"] = {}

        dict_lisa_format["shapes"] = [shape_dict]

        cv2.imwrite(str(save_dir_path / f"rs{n}.jpg"), img_crop)
        with open(str(save_dir_path / f"rs{n}.json"), "w") as f:
            json.dump(dict_lisa_format, f, indent=2)

        if save_weight_map:
            cv2.imwrite(str(weight_map_dir / f"rs{n}{weight_suffix}"), weight_map)

        if visualize_weight_map_flag:
            visualize_weight_map(img_crop, weight_map, shared_with_selected)


if __name__ == "__main__":
    main()
